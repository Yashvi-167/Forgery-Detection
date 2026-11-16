"""
prepare_tgif_index.py
=====================
Scans a downloaded TGIF or TGIF2 dataset directory and generates the
index text files expected by TGIFDataloader:

    data/tgif.txt      — AI-forged image paths + mask paths
    data/tgif_au.txt   — authentic (pre-inpainting) image paths

Expected dataset layout (Zenodo download)
-----------------------------------------

Option A — TGIF v1 (SD2, SDXL, Firefly):
    <root>/
        images/
            fake/   *.png  (AI-inpainted images)
            real/   *.png  (authentic originals)
        masks/
            *.png          (binary masks, same stem as fake images)

Option B — TGIF2 (adds FLUX fakes):
    <root>/
        images/
            fake/
                sd2/    *.png
                sdxl/   *.png
                firefly/   *.png
                flux/   *.png
            real/   *.png
        masks/
            *.png

The script auto-detects both layouts.

Usage
-----
    python scripts/prepare_tgif_index.py --root /path/to/tgif_dataset

Output (relative paths stored in txt files so they work with any root):
    data/tgif.txt
    data/tgif_au.txt
"""

import os
import argparse
import random
from pathlib import Path


def find_matching_mask(stem: str, masks_dir: Path) -> Path | None:
    """Try common mask filename patterns for a given image stem."""
    candidates = [
        masks_dir / f"{stem}.png",
        masks_dir / f"{stem}_mask.png",
        masks_dir / f"{stem}_gt.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_index(root: str, out_dir: str = "./data", val_split: float = 0.1,
                seed: int = 42):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = root / "images"
    masks_dir  = root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Could not find 'images/' under {root}. "
            "Check that the TGIF dataset is fully extracted."
        )

    # ---- Collect fake images ----
    fake_root = images_dir / "fake"
    real_root = images_dir / "real"

    fake_entries = []  # list of (img_rel, mask_rel)
    real_entries = []  # list of (img_rel,)

    if fake_root.exists():
        # Traverse one or two levels (flat or sub-model folders)
        img_paths = list(fake_root.glob("*.png")) + list(fake_root.glob("*.jpg"))
        # TGIF2: sub-model folders
        for sub in fake_root.iterdir():
            if sub.is_dir():
                img_paths += list(sub.glob("*.png")) + list(sub.glob("*.jpg"))

        for img_path in sorted(img_paths):
            stem = img_path.stem
            mask_path = find_matching_mask(stem, masks_dir)
            if mask_path is None:
                # Try masks sub-folder matching the sub-model
                sub_name = img_path.parent.name if img_path.parent != fake_root else ""
                if sub_name:
                    mask_path = find_matching_mask(stem, masks_dir / sub_name)
            if mask_path is None:
                print(f"  [WARN] No mask found for {img_path.name}, skipping.")
                continue
            img_rel  = str(img_path.relative_to(root)).replace("\\", "/")
            mask_rel = str(mask_path.relative_to(root)).replace("\\", "/")
            fake_entries.append((img_rel, mask_rel))
    else:
        print("[WARN] No 'fake/' sub-folder found. Check your dataset layout.")

    # ---- Collect real images ----
    if real_root.exists():
        real_imgs = list(real_root.glob("*.png")) + list(real_root.glob("*.jpg"))
        for img_path in sorted(real_imgs):
            img_rel = str(img_path.relative_to(root)).replace("\\", "/")
            real_entries.append(img_rel)
    else:
        print("[WARN] No 'real/' sub-folder found.")

    print(f"\n[TGIF Index] Found {len(fake_entries)} forged pairs, "
          f"{len(real_entries)} authentic images.")

    # ---- Split train / val ----
    random.seed(seed)

    # Forged
    random.shuffle(fake_entries)
    n_val_fake = max(1, int(len(fake_entries) * val_split))
    fake_val   = fake_entries[:n_val_fake]
    fake_train = fake_entries[n_val_fake:]

    # Authentic
    random.shuffle(real_entries)
    n_val_real = max(1, int(len(real_entries) * val_split))
    real_val   = real_entries[:n_val_real]
    real_train = real_entries[n_val_real:]

    def write_lines(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Wrote {len(lines):>7,} lines → {path}")

    # ---- Write forged index files ----
    write_lines(
        out_dir / "tgif.txt",
        [f"{img},{msk}" for img, msk in fake_train],
    )
    write_lines(
        out_dir / "tgif_val.txt",
        [f"{img},{msk}" for img, msk in fake_val],
    )

    # ---- Write authentic index files ----
    write_lines(out_dir / "tgif_au.txt",   real_train)
    write_lines(out_dir / "tgif_au_val.txt", real_val)

    print("\n[TGIF Index] Done.  Add these to your dataset_list:")
    print(f'  Forged train : ["{root}", "tgif.txt",     1, 1, 1, 0, 1]')
    print(f'  Authentic    : ["{root}", "tgif_au.txt",  1, 1, 0, 0, 0]')
    print(f'  Forged val   : ["{root}", "tgif_val.txt", 1, 1, 1, 0, 1]')


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TGIF dataset index files.")
    parser.add_argument("--root",      type=str, required=True,
                        help="Root directory of the downloaded TGIF/TGIF2 dataset.")
    parser.add_argument("--out_dir",   type=str, default="./data",
                        help="Output directory for .txt index files (default: ./data).")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation (default: 0.1).")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    build_index(
        root=args.root,
        out_dir=args.out_dir,
        val_split=args.val_split,
        seed=args.seed,
    )
