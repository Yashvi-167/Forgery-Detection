"""
create_mock_tgif2.py
====================
Creates a minimal TGIF2 Random-compatible directory structure at D:\\tgif2_random
using symlinks/copies of images from D:\\ForensicSAME, with synthetic binary masks.

This lets you test inspect_tgif2.py, integrate_tgif2.py, and verify_integration.py
end-to-end before the real dataset arrives.

It mimics the exact TGIF2 Random layout:
  original/{training,validation,testing}/   <- real images
  sd2-sp/{training,validation,testing}/     <- fake images (SP format filenames)
  masks-sd2/{training,validation,testing}/  <- binary masks (white circle on black)
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

SOURCE_FAKE = Path(r"D:\ForensicSAME\train\fake")
SOURCE_REAL = Path(r"D:\ForensicSAME\train\real")
MOCK_ROOT   = Path(r"D:\tgif2_random")

# How many images per split to include in mock
SPLIT_SIZES = {"training": 30, "validation": 10, "testing": 10}

SPLITS_FAKE_FILES = {}  # filled below
SPLITS_REAL_FILES = {}


def make_synthetic_mask(img_path: Path, out_path: Path):
    """Create a 50%-coverage white rectangle mask matching image dimensions."""
    if HAS_CV2:
        img = cv2.imread(str(img_path))
        if img is None:
            h, w = 512, 512
        else:
            h, w = img.shape[:2]
    else:
        h, w = 512, 512

    mask = np.zeros((h, w), dtype=np.uint8)
    # White rectangle covering roughly center 40% region
    y1, y2 = int(h * 0.3), int(h * 0.7)
    x1, x2 = int(w * 0.3), int(w * 0.7)
    mask[y1:y2, x1:x2] = 255

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_CV2:
        cv2.imwrite(str(out_path), mask)
    else:
        # fallback: write a tiny PNG manually using raw bytes
        import struct, zlib
        def png_chunk(name, data):
            c = struct.pack('>I', len(data)) + name + data
            return c + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff)
        raw = b'\x89PNG\r\n\x1a\n'
        ihdr = struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)
        raw += png_chunk(b'IHDR', ihdr)
        rows = b''
        for row in mask:
            rows += b'\x00' + bytes(row.tolist())
        raw += png_chunk(b'IDAT', zlib.compress(rows))
        raw += png_chunk(b'IEND', b'')
        out_path.write_bytes(raw)


def run():
    print("=" * 60)
    print("  Creating Mock TGIF2 Random Fixture")
    print(f"  Source fake : {SOURCE_FAKE}")
    print(f"  Source real : {SOURCE_REAL}")
    print(f"  Output root : {MOCK_ROOT}")
    print("=" * 60)

    if not SOURCE_FAKE.exists():
        print(f"\n[ERROR] Source fake dir not found: {SOURCE_FAKE}")
        return
    if not SOURCE_REAL.exists():
        print(f"\n[ERROR] Source real dir not found: {SOURCE_REAL}")
        return

    fake_files = sorted([f for f in SOURCE_FAKE.iterdir()
                         if f.suffix.lower() in {".jpg", ".png"}])
    real_files = sorted([f for f in SOURCE_REAL.iterdir()
                         if f.suffix.lower() in {".jpg", ".png"}])

    print(f"\n  Source: {len(fake_files)} fake, {len(real_files)} real images available")

    # Assign images to splits
    total_fake = sum(SPLIT_SIZES.values())
    total_real = sum(SPLIT_SIZES.values())
    random.seed(42)
    random.shuffle(fake_files)
    random.shuffle(real_files)

    split_fake = {}
    split_real = {}
    fi = ri = 0
    for split, n in SPLIT_SIZES.items():
        split_fake[split] = fake_files[fi:fi+n]
        split_real[split] = real_files[ri:ri+n]
        fi += n
        ri += n

    # Use a fake coco_id counter
    coco_counter = 100000

    created_images = 0
    created_masks  = 0

    for split, fake_imgs in split_fake.items():
        print(f"\n  Split: {split}")

        # --- SP fake images ---
        sp_dir   = MOCK_ROOT / "sd2-sp"   / split
        mask_dir = MOCK_ROOT / "masks-sd2" / split
        sp_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(fake_imgs):
            coco_id  = str(coco_counter + i)
            # TGIF2 SP naming: {coco_id}_mask_random.png_sd2_{var}.png
            sp_name   = f"{coco_id}_mask_random.png_sd2_0{src.suffix}"
            mask_name = f"{coco_id}_mask_random.png"

            # Copy image (preserve extension so cv2 can read it)
            dst_img = sp_dir / sp_name
            shutil.copy2(src, dst_img)
            created_images += 1

            # Create synthetic mask
            dst_mask = mask_dir / mask_name
            make_synthetic_mask(src, dst_mask)
            created_masks += 1

        print(f"    sd2-sp/{split}/   : {len(fake_imgs)} images")
        print(f"    masks-sd2/{split}/: {len(fake_imgs)} masks")
        coco_counter += len(fake_imgs)

        # --- Authentic originals ---
        orig_dir = MOCK_ROOT / "original" / split
        orig_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(split_real[split]):
            coco_id  = str(coco_counter + i)
            orig_name = f"{coco_id}_orig{src.suffix}"
            shutil.copy2(src, orig_dir / orig_name)
            created_images += 1

        print(f"    original/{split}/  : {len(split_real[split])} authentic images")
        coco_counter += len(split_real[split])

    print(f"\n  Total images created : {created_images}")
    print(f"  Total masks created  : {created_masks}")
    print(f"\n  Mock TGIF2 root ready at: {MOCK_ROOT}")
    print("\n  Now run:")
    print("    python inspect_tgif2.py --root D:\\tgif2_random")
    print("    python integrate_tgif2.py --root D:\\tgif2_random")
    print("    python verify_integration.py --root D:\\tgif2_random")


if __name__ == "__main__":
    run()
