"""
run_full_pipeline.py
====================
Complete AI forgery detection training + demo inference.
Uses D:\\dataset (fake/real) on CPU.

Phase 1: Train AIForgeryClassifier (ResNet18 + FFT frequency branch)
Phase 2: Demo inference on sample images showing the classifier works
"""
import os, sys, random, warnings, time
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from ai_forgery_detector import AIForgeryClassifier

# ============ CONFIG ============
DATASET_ROOT = r"D:\ForensicSAME"
WEIGHT_DIR   = "./weight"
SAVE_PATH    = os.path.join(WEIGHT_DIR, "ai_forgery_detector.pth")
EPOCHS       = 3
BATCH_SIZE   = 16
LR           = 1e-4
IMAGE_SIZE   = 224
SEED         = 42
TRAIN_SAMPLES = 1000  # small subset for CPU
VAL_SAMPLES   = 200
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============ DATASET ============
class AIForgeryDataset(Dataset):
    def __init__(self, root, split="train", max_samples=None):
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        fake_dir = os.path.join(root, split, "fake")
        real_dir = os.path.join(root, split, "real")

        fake_files = [(os.path.join(fake_dir, f), 1) for f in sorted(os.listdir(fake_dir))
                      if f.endswith((".jpg", ".png"))]
        real_files = [(os.path.join(real_dir, f), 0) for f in sorted(os.listdir(real_dir))
                      if f.endswith((".jpg", ".png"))]

        self.samples = fake_files + real_files
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]

        n_fake = sum(1 for _, l in self.samples if l == 1)
        n_real = len(self.samples) - n_fake
        print(f"    [{split}] {len(self.samples)} samples (AI-forged={n_fake}, real={n_real})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), path


# ============ PHASE 1: TRAINING ============
def train_classifier():
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    print("=" * 60)
    print("  PHASE 1: Training AI Forgery Classifier")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Dataset     : {DATASET_ROOT}")
    print(f"  Train subset: {TRAIN_SAMPLES} samples")
    print(f"  Val subset  : {VAL_SAMPLES} samples")
    print(f"  Epochs      : {EPOCHS}")
    print()

    # Data
    print("  Loading data...")
    train_ds = AIForgeryDataset(DATASET_ROOT, "train", TRAIN_SAMPLES)
    val_ds   = AIForgeryDataset(DATASET_ROOT, "val", VAL_SAMPLES)

    labels = [l for _, l in train_ds.samples]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    w = [1.0 / n_pos if l == 1 else 1.0 / n_neg for l in labels]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\n  Building AIForgeryClassifier (ResNet18 + FFT branch)...")
    model = AIForgeryClassifier(backbone="resnet18", pretrained=True, freeze_rgb_backbone=True)
    model = model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params\n")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        correct, total_n, running_loss = 0, 0, 0.0
        for batch_idx, (images, labels_b, _) in enumerate(train_loader):
            images, labels_b = images.to(DEVICE), labels_b.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels_b).sum().item()
            total_n += labels_b.size(0)
            running_loss += loss.item() * labels_b.size(0)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"    Epoch {epoch}/{EPOCHS} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f} acc={correct/total_n:.3f}")

        scheduler.step()
        train_acc = correct / total_n

        # --- VAL ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels_b, _ in val_loader:
                images, labels_b = images.to(DEVICE), labels_b.to(DEVICE)
                logits, _ = model(images)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels_b).sum().item()
                val_total   += labels_b.size(0)

        val_acc = val_correct / val_total
        elapsed = time.time() - start_time
        print(f"  >> Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} [{elapsed:.0f}s]\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_detector(SAVE_PATH)
            print(f"     Best model saved! val_acc={best_val_acc:.4f}\n")

    return model, best_val_acc


# ============ PHASE 2: DEMO INFERENCE ============
def demo_inference(model):
    print("\n" + "=" * 60)
    print("  PHASE 2: Demo Inference — AI Forgery Detection")
    print("=" * 60)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Pick 5 fake and 5 real from val set
    test_images = []
    fake_dir = os.path.join(DATASET_ROOT, "val", "fake")
    real_dir = os.path.join(DATASET_ROOT, "val", "real")

    fake_files = sorted(os.listdir(fake_dir))[:5]
    real_files = sorted(os.listdir(real_dir))[:5]

    for f in fake_files:
        test_images.append((os.path.join(fake_dir, f), "AI-FORGED (ground truth)"))
    for f in real_files:
        test_images.append((os.path.join(real_dir, f), "AUTHENTIC (ground truth)"))

    print(f"\n  Testing on {len(test_images)} images:\n")
    print(f"  {'Image':<35} {'Ground Truth':<25} {'Prediction':<20} {'Confidence':<12}")
    print(f"  {'-'*35} {'-'*25} {'-'*20} {'-'*12}")

    correct = 0
    with torch.no_grad():
        for path, gt_label in test_images:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            logits, _ = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

            pred_str = "AI-FORGED" if pred_class == 1 else "AUTHENTIC"
            is_correct = ("AI-FORGED" in gt_label and pred_class == 1) or \
                         ("AUTHENTIC" in gt_label and pred_class == 0)
            correct += int(is_correct)
            mark = "OK" if is_correct else "XX"

            fname = os.path.basename(path)
            print(f"  {fname:<35} {gt_label:<25} {pred_str:<20} {confidence:.1%} [{mark}]")

    print(f"\n  Demo accuracy: {correct}/{len(test_images)} = {correct/len(test_images):.0%}")


# ============ MAIN ============
if __name__ == "__main__":
    print()
    print("  ForensicsSAM Delta Addition — AI Forgery Detection Pipeline")
    print("  " + "=" * 56)
    print()

    # Phase 1: Train
    model, best_acc = train_classifier()

    # Phase 2: Demo
    demo_inference(model)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Weights saved to: {SAVE_PATH}")
    print()
    print("  Next steps for full mask localization (Phase 2 LoRA):")
    print("    1. Download TGIF2 dataset (has masks)")
    print("    2. Get GPU access (CUDA)")
    print("    3. Download SAM weights (sam_vit_h_4b8939.pth)")
    print("    4. Run: python train_ai_lora.py")
    print("=" * 60)
