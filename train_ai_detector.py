"""
train_ai_detector.py
====================
Phase 1 Training: Train the AIForgeryClassifier to distinguish
AI-generated / diffusion-based forgeries from traditional forgeries
and authentic images.

Training data
-------------
  Positive (ai_label=1): TGIF / TGIF2 AI-forged images
  Negative (ai_label=0): Authentic images from existing datasets
                         + optionally traditional forged images

The classifier only needs image-level binary labels (no masks needed),
so it converges quickly (5–10 epochs on TGIF).

Usage
-----
    python train_ai_detector.py

Configuration is set via constants at the top of this file.
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────
from mini_dataloader import TGIFDataloader
from ai_forgery_detector import AIForgeryClassifier

# ===========================================================================
# ░░░  CONFIGURATION  ░░░
# ===========================================================================

# ----- Paths -----
TGIF_ROOT = "./data/tgif"          # ← set to where you extracted TGIF/TGIF2
DATA_DIR  = "./data"
SAVE_DIR  = "./weight"
SAVE_PATH = os.path.join(SAVE_DIR, "ai_forgery_detector.pth")

# ----- Training -----
EPOCHS       = 10
BATCH_SIZE   = 16
LR           = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE   = 224          # classifier uses 224 (smaller than SAM's 1024)
NORMALIZE_TYPE = 2          # ImageNet normalisation
NUM_WORKERS  = 4
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Dataset list -----
# Format: (root, txt_name, ratio, load_times, forged_label, adv_label, ai_label)
TRAIN_DATASET_LIST = [
    # AI-forged images from TGIF (ai_label=1)
    [TGIF_ROOT, "tgif.txt",    1.0, 1, 1, 0, 1],
    # Authentic images from TGIF (ai_label=0)
    [TGIF_ROOT, "tgif_au.txt", 1.0, 1, 0, 0, 0],
    # Optionally mix in authentic images from traditional datasets:
    # ["./data", "casia1_au.txt", 1.0, 1, 0, 0, 0],
    # ["./data", "columbia_au.txt", 1.0, 1, 0, 0, 0],
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


def build_weighted_sampler(dataset):
    """
    Create a WeightedRandomSampler so that positive/negative classes are
    balanced during training regardless of dataset size ratio.
    """
    ai_labels = dataset.data["ai_label"]
    n_pos = sum(ai_labels)
    n_neg = len(ai_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # only one class – no rebalancing needed

    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    weights = [weight_pos if lbl == 1 else weight_neg for lbl in ai_labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# ===========================================================================
# ░░░  MAIN TRAINING LOOP  ░░░
# ===========================================================================

def train():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[train_ai_detector] Device: {DEVICE}")
    print(f"[train_ai_detector] TGIF root: {TGIF_ROOT}")

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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = AIForgeryClassifier(
        backbone="resnet18",
        pretrained=True,
        freeze_rgb_backbone=True,   # only train last 2 ResNet blocks + fusion
    ).to(DEVICE)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # -- Train --
        model.train()
        train_loss, train_acc = 0.0, 0.0
        t = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch in t:
            images, _masks, _forged, _adv, ai_labels = batch
            images    = images.to(DEVICE)
            ai_labels = ai_labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, ai_labels)
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(logits, ai_labels)
            train_loss += loss.item()
            train_acc  += acc
            t.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        scheduler.step()
        n = len(train_loader)
        print(f"  Train  — loss: {train_loss/n:.4f}  acc: {train_acc/n:.4f}")

        # -- Validation --
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images, _masks, _forged, _adv, ai_labels = batch
                images    = images.to(DEVICE)
                ai_labels = ai_labels.to(DEVICE)

                logits, _ = model(images)
                loss = criterion(logits, ai_labels)
                val_loss += loss.item()
                val_acc  += compute_accuracy(logits, ai_labels)

        n = len(val_loader)
        avg_val_acc = val_acc / n
        print(f"  Val    — loss: {val_loss/n:.4f}  acc: {avg_val_acc:.4f}")

        # -- Save best --
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            model.save_detector(SAVE_PATH)
            print(f"  ✔ Best model saved  (val_acc={best_val_acc:.4f})")

    print(f"\n[train_ai_detector] Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"[train_ai_detector] Weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
