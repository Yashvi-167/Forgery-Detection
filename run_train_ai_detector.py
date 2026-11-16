"""
run_train_ai_detector.py
========================
Actual executable training script for Phase 1 — AIForgeryClassifier.
Uses D:\\dataset (24k train + 6k val, fake/real).
Runs on CPU. Uses small subset for quick demo, full dataset for real training.
"""
import os, sys, random, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from ai_forgery_detector import AIForgeryClassifier

# ============ CONFIG ============
DATASET_ROOT = r"D:\dataset"
SAVE_PATH    = "./weight/ai_forgery_detector.pth"
EPOCHS       = 5
BATCH_SIZE   = 16
LR           = 1e-4
IMAGE_SIZE   = 224
SEED         = 42
MAX_SAMPLES  = 2000   # Use subset for CPU training (set None for full)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ============ SEED ============
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============ DATASET ============
class SimpleAIDataset(Dataset):
    """Simple dataset: fake=1, real=0."""
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
        print(f"  [{split}] {len(self.samples)} samples "
              f"(fake={sum(1 for _,l in self.samples if l==1)}, "
              f"real={sum(1 for _,l in self.samples if l==0)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ============ TRAINING ============
def train():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Max samples per split: {MAX_SAMPLES}")
    print()

    # Data
    print("Loading data...")
    train_ds = SimpleAIDataset(DATASET_ROOT, "train", MAX_SAMPLES)
    val_ds   = SimpleAIDataset(DATASET_ROOT, "val", MAX_SAMPLES // 4 if MAX_SAMPLES else None)

    # Weighted sampler
    labels = [l for _, l in train_ds.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w = [1.0/n_pos if l == 1 else 1.0/n_neg for l in labels]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\nBuilding AIForgeryClassifier (ResNet18 + FFT)...")
    model = AIForgeryClassifier(backbone="resnet18", pretrained=True, freeze_rgb_backbone=True)
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        t = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", ncols=90)
        for images, labels_batch in t:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total   += labels_batch.size(0)
            train_loss    += loss.item() * labels_batch.size(0)
            t.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct/train_total:.3f}")

        scheduler.step()
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # --- VAL ---
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", ncols=90):
                images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
                logits, _ = model(images)
                loss = criterion(logits, labels_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total   += labels_batch.size(0)
                val_loss    += loss.item() * labels_batch.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_detector(SAVE_PATH)
            print(f"  >>> Best model saved! val_acc={best_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Weights saved to : {SAVE_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
