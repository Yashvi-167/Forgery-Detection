"""
train_merged.py
===============
Merged training pipeline using ALL available data:

PHASE 1 — AI Forgery Classifier
  Dataset : D:\\ForensicSAME  (24 000 fake + 24 000 real images)
  Model   : AIForgeryClassifier (MobileNetV3-based)
  Output  : weight/ai_forgery_detector.pth
  Epochs  : 10  (more than the original 3-epoch quick run)

PHASE 2 — ForensicsSAM AI-LoRA Fine-tuning
  Dataset : D:\\tgif2_random  (SP forged with masks + authentic originals)
  Model   : ForensicsSAM with AI_LoRA adapters
  Output  : weight/ai_forgery_experts.pth
  Epochs  : 5  (limited on CPU; swap in real TGIF2 for full training)

Usage:
    python train_merged.py                  # both phases
    python train_merged.py --phase 1        # classifier only
    python train_merged.py --phase 2        # SAM LoRA only
"""

import os, sys, random, argparse, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU unless GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[train_merged] Device: {DEVICE}")

# ── Paths ──────────────────────────────────────────────────────────────────
FORENSICSAME_ROOT = r"D:\ForensicSAME"
TGIF2_ROOT        = r"D:\tgif2_random"
DATA_DIR          = "./data"
WEIGHT_DIR        = "./weight"

SAM_CHECKPOINT    = os.path.join(WEIGHT_DIR, "sam_vit_h_4b8939.pth")
ADV_DET_WEIGHT    = os.path.join(WEIGHT_DIR, "adversary_detector.pth")
FORGERY_EXP_WEIGHT= os.path.join(WEIGHT_DIR, "forgery_experts.pth")
AI_CLS_OUT        = os.path.join(WEIGHT_DIR, "ai_forgery_detector.pth")
AI_LORA_OUT       = os.path.join(WEIGHT_DIR, "ai_forgery_experts.pth")

os.makedirs(WEIGHT_DIR, exist_ok=True)

# ── Seed ───────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed()

# ===========================================================================
# PHASE 1 — AI Forgery Classifier
# ===========================================================================

class FlatImageDataset(Dataset):
    """Loads fake/real images from D:\\ForensicSAME flat structure."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, root: str, split: str, max_per_class: int = None,
                 img_size: int = 224):
        self.img_size = img_size
        fake_dir = os.path.join(root, split, "fake")
        real_dir = os.path.join(root, split, "real")
        self.samples = []
        for d, label in [(fake_dir, 1), (real_dir, 0)]:
            files = sorted([f for f in os.listdir(d)
                            if os.path.splitext(f)[1].lower() in self.IMG_EXTS])
            if max_per_class:
                files = files[:max_per_class]
            self.samples += [(os.path.join(d, f), label) for f in files]
        random.shuffle(self.samples)
        print(f"  [{split}] {len(self.samples)} images "
              f"({sum(1 for _,l in self.samples if l==1)} fake, "
              f"{sum(1 for _,l in self.samples if l==0)} real)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        t = torch.from_numpy(img).permute(2, 0, 1)
        return t, torch.tensor(label, dtype=torch.long)


def train_classifier(epochs: int = 10, batch_size: int = 32,
                     lr: float = 1e-3, max_per_class: int = None):
    print("\n" + "="*60)
    print("  PHASE 1: AI Forgery Classifier Training")
    print("="*60)
    from ai_forgery_detector import AIForgeryClassifier

    # Build datasets
    train_ds = FlatImageDataset(FORENSICSAME_ROOT, "train", max_per_class)
    val_ds   = FlatImageDataset(FORENSICSAME_ROOT, "val",   max_per_class)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = AIForgeryClassifier().to(DEVICE)
    # Resume from existing checkpoint if available
    if os.path.exists(AI_CLS_OUT):
        try:
            ckpt = torch.load(AI_CLS_OUT, map_location=DEVICE)
            model.load_state_dict(ckpt)
            print(f"  Resumed from {AI_CLS_OUT}")
        except Exception as e:
            print(f"  Could not resume ({e}), training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_rows = [["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]]
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)
            bar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss = t_loss / t_total
        train_acc  = t_correct / t_total * 100

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = criterion(logits, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total * 100

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}%")
        log_rows.append([epoch, f"{train_loss:.4f}", f"{train_acc:.1f}",
                          f"{val_loss:.4f}", f"{val_acc:.1f}"])

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), AI_CLS_OUT)
            print(f"    -> Best model saved  (val_acc={val_acc:.1f}%)")

    # Save training log
    log_path = os.path.join(WEIGHT_DIR, "classifier_training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"\n  Training log: {log_path}")
    print(f"  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {AI_CLS_OUT}")
    return best_val_acc


# ===========================================================================
# PHASE 2 — ForensicsSAM AI-LoRA Fine-tuning
# ===========================================================================

def train_sam_lora(epochs: int = 5, batch_size: int = 1,
                   lr: float = 5e-5, image_size: int = 512):
    print("\n" + "="*60)
    print("  PHASE 2: ForensicsSAM AI-LoRA Fine-tuning")
    print("="*60)

    # Check SAM weights
    for name, path in [("SAM backbone", SAM_CHECKPOINT),
                        ("Forgery experts", FORGERY_EXP_WEIGHT)]:
        if not os.path.exists(path):
            print(f"  [SKIP] Missing {name}: {path}")
            return
        print(f"  OK  {name}")

    # Check TGIF2 index files
    sp_train  = os.path.join(DATA_DIR, "tgif2_random_sp_train.txt")
    au_train  = os.path.join(DATA_DIR, "tgif2_random_au_train.txt")
    sp_val    = os.path.join(DATA_DIR, "tgif2_random_sp_val.txt")
    au_val    = os.path.join(DATA_DIR, "tgif2_random_au_val.txt")
    for f in [sp_train, au_train, sp_val, au_val]:
        if not os.path.exists(f):
            print(f"  [SKIP] Missing index: {f}")
            print("  Run: python integrate_tgif2.py --root D:\\tgif2_random")
            return
        print(f"  OK  {os.path.basename(f)}")

    # Load dataloader
    sys.path.insert(0, ".")
    from mini_dataloader.tgif_dataloader import TGIFDataloader

    train_dataset_list = [
        ("", sp_train, 1.0, 1, 1, 0, 1),   # forged + mask, ai_label=1
        ("", au_train, 1.0, 1, 0, 0, 0),   # authentic, ai_label=0
    ]
    val_dataset_list = [
        ("", sp_val, 1.0, 1, 1, 0, 1),
        ("", au_val, 1.0, 1, 0, 0, 0),
    ]

    train_ds = TGIFDataloader(train_dataset_list, input_size=image_size, mode="train")
    val_ds   = TGIFDataloader(val_dataset_list,   input_size=image_size, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"  Image size: {image_size}  |  Batch: {batch_size}  |  Epochs: {epochs}")

    # Load ForensicsSAM
    print("\n  Loading SAM backbone (this may take ~30s)...")
    from segment_anything import sam_model_registry
    from forensics_sam import ForensicsSAM

    sam, _ = sam_model_registry["vit_h"](
        image_size=image_size, checkpoint=SAM_CHECKPOINT)
    model = ForensicsSAM(sam, r=8).to(DEVICE)

    # Load pretrained forgery experts
    fe_ckpt = torch.load(FORGERY_EXP_WEIGHT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(fe_ckpt, strict=False)
    print(f"  Forgery experts loaded (missing={len(missing)}, extra={len(unexpected)})")

    # Freeze everything; unfreeze only AI LoRA params
    for name, param in model.named_parameters():
        param.requires_grad = False
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["ai_lora", "AI_LoRA", "ai_expert", "AI_expert"]):
            param.requires_grad = True
            trainable_params.append(param)

    if not trainable_params:
        # Fallback: fine-tune mask decoder only
        print("  [INFO] No AI LoRA params found — fine-tuning mask_decoder instead")
        for name, param in model.named_parameters():
            if "mask_decoder" in name:
                param.requires_grad = True
                trainable_params.append(param)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params "
          f"({n_trainable/n_total*100:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    log_rows = [["epoch", "train_loss", "val_loss"]]

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        t_loss, t_steps = 0.0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [SAM train]", leave=False)
        for batch in bar:
            imgs, masks, forged_label, adv_label, ai_label = batch
            imgs   = imgs.to(DEVICE)
            masks  = masks.to(DEVICE).float()
            adv_pred = torch.zeros(imgs.size(0), dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            mask_logit, cls_logit = model(imgs, adv_pred)

            # Resize predicted mask to match gt
            if mask_logit.shape[-2:] != masks.shape[-2:]:
                mask_logit = torch.nn.functional.interpolate(
                    mask_logit, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss_mask = bce(mask_logit, masks)
            loss_cls  = bce(cls_logit.view(-1),
                            forged_label.float().to(DEVICE))
            loss = loss_mask + 0.5 * loss_cls
            loss.backward()
            optimizer.step()

            t_loss  += loss.item()
            t_steps += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = t_loss / max(t_steps, 1)

        # ── Validate ──
        model.eval()
        v_loss, v_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, masks, forged_label, adv_label, ai_label = batch
                imgs   = imgs.to(DEVICE)
                masks  = masks.to(DEVICE).float()
                adv_pred = torch.zeros(imgs.size(0), dtype=torch.long, device=DEVICE)
                mask_logit, cls_logit = model(imgs, adv_pred)
                if mask_logit.shape[-2:] != masks.shape[-2:]:
                    mask_logit = torch.nn.functional.interpolate(
                        mask_logit, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss_mask = bce(mask_logit, masks)
                loss_cls  = bce(cls_logit.view(-1),
                                forged_label.float().to(DEVICE))
                v_loss  += (loss_mask + 0.5 * loss_cls).item()
                v_steps += 1

        val_loss = v_loss / max(v_steps, 1)
        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train Loss={train_loss:.4f}  |  Val Loss={val_loss:.4f}")
        log_rows.append([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only trainable weights
            state = {k: v for k, v in model.state_dict().items()
                     if any(p is model.state_dict().get(k) for p in trainable_params)
                     or any(key in k for key in ["ai_lora","AI_LoRA","mask_decoder"])}
            torch.save(state, AI_LORA_OUT)
            print(f"    -> Best model saved (val_loss={val_loss:.4f})")

    log_path = os.path.join(WEIGHT_DIR, "sam_lora_training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"\n  Training log: {log_path}")
    print(f"  Model saved to: {AI_LORA_OUT}")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, default=0,
                   help="1=classifier only, 2=SAM LoRA only, 0=both (default)")
    p.add_argument("--cls-epochs",   type=int,   default=10)
    p.add_argument("--cls-batch",    type=int,   default=32)
    p.add_argument("--cls-lr",       type=float, default=1e-3)
    p.add_argument("--cls-maximg",   type=int,   default=None,
                   help="Max images per class (None=all 24k)")
    p.add_argument("--lora-epochs",  type=int,   default=5)
    p.add_argument("--lora-batch",   type=int,   default=1)
    p.add_argument("--lora-lr",      type=float, default=5e-5)
    p.add_argument("--lora-imgsize", type=int,   default=512,
                   help="SAM image size for Phase 2 (256/512/1024)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    print("=" * 60)
    print("  Merged Dataset Training Pipeline")
    print(f"  ForensicSAME : {FORENSICSAME_ROOT}")
    print(f"  TGIF2 mock   : {TGIF2_ROOT}")
    print(f"  Device       : {DEVICE}")
    print("=" * 60)

    if args.phase in (0, 1):
        train_classifier(
            epochs=args.cls_epochs,
            batch_size=args.cls_batch,
            lr=args.cls_lr,
            max_per_class=args.cls_maximg,
        )

    if args.phase in (0, 2):
        train_sam_lora(
            epochs=args.lora_epochs,
            batch_size=args.lora_batch,
            lr=args.lora_lr,
            image_size=args.lora_imgsize,
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All done in {elapsed/60:.1f} minutes.")
    print(f"  Classifier  : {AI_CLS_OUT}")
    print(f"  SAM LoRA    : {AI_LORA_OUT}")
    print(f"{'='*60}")
