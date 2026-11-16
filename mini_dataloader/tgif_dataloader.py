"""
TGIFDataloader
==============
Dataset loader for the TGIF (Text-Guided Inpainting Forgery) and TGIF2
datasets published by IDLab (Ghent University) at IEEE WIFS 2024.

TGIF structure on disk
----------------------
After downloading from Zenodo, the dataset has this layout:

    <tgif_root>/
        images/
            real/
                <id>.png   (authentic originals)
            fake/
                <id>.png   (AI-inpainted versions)
        masks/
            <id>.png       (binary forgery masks, white = forged region)

TGIF2 adds FLUX-based fakes and non-semantic random masks but follows
the same folder structure.

Usage
-----
1. Download the dataset and set TGIF_ROOT in your config.
2. Run  ``python scripts/prepare_tgif_index.py --root <TGIF_ROOT>``
   to generate  data/tgif.txt  and  data/tgif_au.txt.
3. Add entries to your dataset_list in the training script.

Index file format (consistent with existing ForensicsSAM txt files)
--------------------------------------------------------------------
Each line:  fake/<id>.png,masks/<id>.png
Authentic:  real/<id>.png   (no mask column → empty mask auto-created)
"""

import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Lambda
from tqdm import tqdm

from .post_process import (
    vertical_flip,
    horizontal_flip,
    rotate,
    gaussian_noise,
    gaussian_blur,
    jpeg_compression,
    resize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_lines(txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.read().splitlines() if l.strip()]


def _augment(image, mask, augment_prob=0.0, enable_aug_types=None,
             intensity=None, mode="train"):
    """Same augmentation logic as BasicDataloader.process()."""
    if mode == "train":
        num_aug_types = np.random.choice(4, 1)
        aug_types = np.random.choice([0, 1, 2], num_aug_types, replace=False)
        for aug_type in aug_types:
            if aug_type == 0:
                image, mask = horizontal_flip(image, mask, p=augment_prob)
            elif aug_type == 1:
                image, mask = vertical_flip(image, mask, p=augment_prob)
            elif aug_type == 2:
                image, mask = rotate(image, mask, p=augment_prob)

        if enable_aug_types is not None and len(enable_aug_types) > 0:
            aug_types = np.random.choice(enable_aug_types, 1, replace=False)
            for aug_type in aug_types:
                if aug_type == 3:
                    image, mask = resize(image, mask, rates=intensity.get("rates"), p=augment_prob)
                elif aug_type == 4:
                    image = jpeg_compression(image, qfs=intensity.get("qfs"), p=augment_prob)
                elif aug_type == 5:
                    image = gaussian_noise(image, sds=intensity.get("sds"), p=augment_prob)
                elif aug_type == 6:
                    image = gaussian_blur(image, ksizes=intensity.get("ksizes"), p=augment_prob)
    else:
        if enable_aug_types:
            for aug_type in enable_aug_types:
                if aug_type == 3:
                    image, mask = resize(image, mask, rates=intensity.get("rates"), p=augment_prob)
                elif aug_type == 4:
                    image = jpeg_compression(image, qfs=intensity.get("qfs"), p=augment_prob)
                elif aug_type == 5:
                    image = gaussian_noise(image, sds=intensity.get("sds"), p=augment_prob)
                elif aug_type == 6:
                    image = gaussian_blur(image, ksizes=intensity.get("ksizes"), p=augment_prob)
    return image, mask


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TGIFDataloader(Dataset):
    """
    Multi-source dataset loader that extends the existing BasicDataloader
    API with an extra ``ai_label`` field.

    Parameters
    ----------
    dataset_list : list of tuples
        Each tuple: (root_path, txt_name, ratio, load_times,
                     forged_label, adv_label, ai_label)
        ``ai_label`` = 1 for AI-generated forgeries (TGIF/TGIF2),
                       0 for traditional forgeries or authentic images.
    input_size : int
        SAM input size (default 1024).
    normalize_type : int
        0 = no normalisation, 1 = [-1,1], 2 = ImageNet (default).
    augment_prob : float
        Probability for each augmentation to be applied.
    enable_aug_types : list[int] | None
    intensity : dict | None
    mode : str
        'train' or 'val'.
    """

    def __init__(
        self,
        dataset_list,
        input_size: int = 1024,
        normalize_type: int = 2,
        augment_prob: float = 0.0,
        enable_aug_types=None,
        intensity=None,
        mode: str = "train",
    ):
        self.sam_input_size = input_size
        self.normalize_type = normalize_type
        self.augment_prob = augment_prob
        self.enable_aug_types = enable_aug_types if enable_aug_types is not None else []
        self.intensity = intensity or {}
        self.mode = mode

        self.data = {
            "image_path": [],
            "mask_path": [],
            "forged_label": [],
            "adv_label": [],
            "ai_label": [],      # ← NEW: 1 = AI-generated forgery
        }

        for entry in dataset_list:
            # Support both old 6-tuple (without ai_label) and new 7-tuple
            if len(entry) == 7:
                root, txt_name, ratio, load_times, forged_label, adv_label, ai_label = entry
            elif len(entry) == 6:
                root, txt_name, ratio, load_times, forged_label, adv_label = entry
                ai_label = 0  # traditional dataset, no AI forgery label
            else:
                raise ValueError(f"dataset_list entry must have 6 or 7 elements, got {len(entry)}")

            txt_path = os.path.join(root, txt_name)
            lines = _read_lines(txt_path)
            n = int(ratio * len(lines))
            lines = lines[:n]

            for _ in range(load_times):
                t = tqdm(lines, desc=f"Loading {txt_name}")
                for line in t:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        # format: ..., image_rel_path, mask_rel_path
                        img_rel = parts[-2]
                        msk_rel = parts[-1]
                        img_path = os.path.join(root, img_rel)
                        msk_path = os.path.join(root, msk_rel)
                    else:
                        # authentic-only line: just image path
                        img_path = os.path.join(root, parts[0])
                        msk_path = ""

                    self.data["image_path"].append(img_path)
                    self.data["mask_path"].append(msk_path)
                    self.data["forged_label"].append(forged_label)
                    self.data["adv_label"].append(adv_label)
                    self.data["ai_label"].append(ai_label)

        print(f"[TGIFDataloader] Total samples loaded: {len(self.data['image_path'])}")

        # ---- Normalisation transform ----
        def identity(x):
            return x

        if self.normalize_type == 0:
            norm = Lambda(identity)
        elif self.normalize_type == 1:
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:  # 2 — ImageNet
            norm = transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )

        self.image_transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            norm,
        ])
        self.mask_transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data["image_path"])

    def __getitem__(self, idx):
        # ---- Load image ----
        img = cv2.imread(self.data["image_path"][idx])
        if img is None:
            raise FileNotFoundError(
                f"[TGIFDataloader] Cannot read image: {self.data['image_path'][idx]}"
            )

        # ---- Load mask ----
        msk_path = self.data["mask_path"][idx]
        if msk_path == "":
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

        # ---- Augmentation ----
        img, mask = _augment(
            img, mask,
            augment_prob=self.augment_prob,
            enable_aug_types=self.enable_aug_types,
            intensity=self.intensity,
            mode=self.mode,
        )

        # ---- Resize ----
        if self.sam_input_size is not None:
            img = cv2.resize(img, (self.sam_input_size, self.sam_input_size), cv2.INTER_AREA)
            mask = cv2.resize(
                mask,
                (self.sam_input_size // 4, self.sam_input_size // 4),
                cv2.INTER_NEAREST,
            )

        # ---- Normalise pixel values ----
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
        mask = np.where(mask > 127, 255, 0).astype(np.float32) / 255.0

        # ---- Labels ----
        forged_label = torch.as_tensor(self.data["forged_label"][idx], dtype=torch.long)
        adv_label    = torch.as_tensor(self.data["adv_label"][idx],    dtype=torch.long)
        ai_label     = torch.as_tensor(self.data["ai_label"][idx],     dtype=torch.long)

        return (
            self.image_transform(img),
            self.mask_transform(mask),
            forged_label,
            adv_label,
            ai_label,
        )

    def get_name(self, idx):
        return self.data["image_path"][idx]
