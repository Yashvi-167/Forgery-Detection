"""
train_ai_lora.py
================
Phase 2 Training: Fine-tune the AI-Forgery LoRA adapters (AI_LoRA_QKV and
AI_LoRA_FFN) inside ForensicsSAM so the model can accurately localise
AI-generated / diffusion-based forgery regions and produce pixel-level masks.

What is frozen vs trainable
-----------------------------
  FROZEN  : SAM backbone, shared forgery LoRAs, adversarial LoRAs,
            ForgeryDetector, prompt_encoder, mask_decoder
  TRAINED : AI_LoRA_QKV.{a_q, b_q, a_k, b_k, a_v, b_v}
            AI_LoRA_FFN.{a, b}
            (applied only to global-attention blocks)

Training data
-------------
  Positive (ai_label=1): TGIF / TGIF2 AI-forged images + ground-truth masks
  Negative (ai_label=0): Authentic images (empty masks)

Loss
----
  L = λ_mask * BCE(predicted_mask, gt_mask)
    + λ_cls  * BCE(cls_prediction,  forged_label)

Usage
-----
    python train_ai_lora.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────
from segment_anything import sam_model_registry
from forensics_sam import ForensicsSAM
from mini_dataloader import TGIFDataloader

# ===========================================================================
# ░░░  CONFIGURATION  ░░░
# ===========================================================================

TGIF_ROOT  = "./data/tgif"      # ← set to where you extracted TGIF/TGIF2
DATA_DIR   = "./data"
SAVE_DIR   = "./weight"
AI_WEIGHT_PATH = os.path.join(SAVE_DIR, "ai_forgery_experts.pth")

# SAM backbone
SAM_TYPE   = "vit_h"            # vit_b | vit_l | vit_h
CHECKPOINT = {
    "vit_b": "./weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/sam_vit_h_4b8939.pth",
}

# Training hyper-params
EPOCHS       = 20
BATCH_SIZE   = 2            # keep small — SAM uses 1024×1024 images
LR           = 5e-5
WEIGHT_DECAY = 1e-4
IMAGE_SIZE   = 1024         # SAM input resolution
NORMALIZE_TYPE = 2
NUM_WORKERS  = 2
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK    = 8

# Loss weights
LAMBDA_MASK = 1.0
LAMBDA_CLS  = 0.5

# ----- Dataset list -----
# Format: (root, txt_name, ratio, load_times, forged_label, adv_label, ai_label)
TRAIN_DATASET_LIST = [
    [TGIF_ROOT, "tgif.txt",    1.0, 1, 1, 0, 1],   # AI forged
    [TGIF_ROOT, "tgif_au.txt", 1.0, 1, 0, 0, 0],   # authentic
    # Optional: mix traditional forgeries to prevent catastrophic forgetting
    # ["./data", "casia1.txt", 0.2, 1, 1, 0, 0],
]

VAL_DATASET_LIST = [
    [TGIF_ROOT, "tgif_val.txt",    1.0, 1, 1, 0, 1],
    [TGIF_ROOT, "tgif_au_val.txt", 1.0, 1, 0, 0, 0],
]

# ===========================================================================
# ░░░  HELPERS  ░░░
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def mask_bce_loss(pred_logits, gt_masks):
    """
    pred_logits : (B, 1, H', W') — raw logits from SAM mask decoder
    gt_masks    : (B, 1, H', W') — binary {0, 1}
    """
    return F.binary_cross_entropy_with_logits(pred_logits, gt_masks)


def cls_bce_loss(cls_logits, forged_labels):
    """
    cls_logits    : (B, 1) — raw logit from ForgeryDetector
    forged_labels : (B,)   — binary {0, 1}
    """
    return F.binary_cross_entropy_with_logits(
        cls_logits.squeeze(1),
        forged_labels.float(),
    )


def build_weighted_sampler(dataset):
    ai_labels = dataset.data["ai_label"]
    n_pos = sum(ai_labels)
    n_neg = len(ai_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    weights = [weight_pos if lbl == 1 else weight_neg for lbl in ai_labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ===========================================================================
# ░░░  MAIN  ░░░
# ===========================================================================

def train():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[train_ai_lora] Device:   {DEVICE}")
    print(f"[train_ai_lora] SAM type: {SAM_TYPE}")
    print(f"[train_ai_lora] TGIF root: {TGIF_ROOT}")

    # ── Build SAM + ForensicsSAM ───────────────────────────────────────────
    sam, _image_embedding_size = sam_model_registry[SAM_TYPE](
        image_size=IMAGE_SIZE,
        checkpoint=CHECKPOINT[SAM_TYPE],
    )
    model = ForensicsSAM(
        sam,
        r=LORA_RANK,
        load_ai_weights=False,   # fresh AI LoRA — no weights yet
    ).to(DEVICE).train()

    # Freeze everything except AI LoRA params
    model.set_ai_lora_trainable(trainable=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"[train_ai_lora] Trainable: {trainable_params:,} / {total_params:,} params")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = TGIFDataloader(
        dataset_list=TRAIN_DATASET_LIST,
        input_size=IMAGE_SIZE,
        normalize_type=NORMALIZE_TYPE,
        augment_prob=0.5,
        enable_aug_types=[4, 5, 6],
        intensity={"rates": [0.8], "qfs": [75], "sds": [9], "ksizes": [9]},
        mode="train",
    )
    val_dataset = TGIFDataloader(
        dataset_list=VAL_DATASET_LIST,
        input_size=IMAGE_SIZE,
        normalize_type=NORMALIZE_TYPE,
        augment_prob=0.0,
        enable_aug_types=[],
        mode="val",
    )

    sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ── Optimiser & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # -- Train --
        model.train()
        loss_meter = AverageMeter()
        t = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

        for batch in t:
            images, gt_masks, forged_labels, _adv, ai_labels = batch
            images        = images.to(DEVICE)
            gt_masks      = gt_masks.to(DEVICE)
            forged_labels = forged_labels.to(DEVICE)
            ai_labels     = ai_labels.to(DEVICE)

            # Route: activate AI LoRA for AI-forged images in batch
            # For simplicity, if any sample in the batch is AI-forged, activate
            has_ai = bool(ai_labels.any().item())

            optimizer.zero_grad()
            mask_pred, cls_pred = model(
                images,
                activate_adv=False,
                activate_ai=has_ai,
            )

            loss_mask = mask_bce_loss(mask_pred, gt_masks)
            loss_cls  = cls_bce_loss(cls_pred, forged_labels)
            loss      = LAMBDA_MASK * loss_mask + LAMBDA_CLS * loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            loss_meter.update(loss.item())
            t.set_postfix(
                total=f"{loss.item():.4f}",
                mask=f"{loss_mask.item():.4f}",
                cls=f"{loss_cls.item():.4f}",
            )

        scheduler.step()
        print(f"  Train avg loss: {loss_meter.avg:.4f}")

        # -- Validation --
        model.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images, gt_masks, forged_labels, _adv, ai_labels = batch
                images        = images.to(DEVICE)
                gt_masks      = gt_masks.to(DEVICE)
                forged_labels = forged_labels.to(DEVICE)
                ai_labels     = ai_labels.to(DEVICE)

                has_ai = bool(ai_labels.any().item())
                mask_pred, cls_pred = model(
                    images,
                    activate_adv=False,
                    activate_ai=has_ai,
                )
                loss_mask = mask_bce_loss(mask_pred, gt_masks)
                loss_cls  = cls_bce_loss(cls_pred, forged_labels)
                loss      = LAMBDA_MASK * loss_mask + LAMBDA_CLS * loss_cls
                val_loss_meter.update(loss.item())

        avg_val_loss = val_loss_meter.avg
        print(f"  Val avg loss: {avg_val_loss:.4f}")

        # -- Save best --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_ai_parameters(AI_WEIGHT_PATH)
            print(f"  ✔ Best AI LoRA saved  (val_loss={best_val_loss:.4f})")

    print(f"\n[train_ai_lora] Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"[train_ai_lora] AI LoRA weights → {AI_WEIGHT_PATH}")


if __name__ == "__main__":
    train()
