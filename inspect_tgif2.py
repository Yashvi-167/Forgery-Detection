"""
inspect_tgif2.py
================
Inspects the TGIF2 Random dataset directory structure and reports:
  - Image counts per generator, split, and type (SP / FR / original)
  - Mask counts per generator, split
  - Naming convention examples
  - Image dimension samples
  - Image <-> mask correspondence check (100 random pairs)

Usage:
    python inspect_tgif2.py --root D:\\tgif2_random [--output data/TGIF2_Random/inspection_report.txt]

IMPORTANT:
    Set TGIF2_ROOT below (or pass --root) to your TGIF2 Random download path.
    Download from: https://cloud.ilabt.imec.be/index.php/s/GDGewtTFcHccaNj
"""

import os
import sys
import random
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

# -- try optional deps gracefully --------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# -- TGIF2 Random layout constants -------------------------------------------
SPLITS = ["training", "validation", "testing"]

# SP generator folders  -> their mask folder
SP_GENERATORS = {
    "sd2-sp":          "masks-sd2",
    "flux1schnell-sp": "masks-flux",
    "flux1dev-sp":     "masks-flux",
    "flux1filldev-sp": "masks-flux",
}

# FR generator folders (kept separate — localization annotations unreliable)
FR_GENERATORS = [
    "sd2-fr", "sdxl-fr",
    "flux1schnell-fr", "flux1dev-fr", "flux1filldev-fr",
]

AUTHENTIC_DIR = "original"


# -- Helpers ------------------------------------------------------------------

def count_images(directory: Path):
    """Return count of .png / .jpg / .jpeg files, recursively."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*")
               if f.suffix.lower() in {".png", ".jpg", ".jpeg"})


def sample_files(directory: Path, n: int = 5):
    """Return up to n file names from directory (non-recursive)."""
    if not directory.exists():
        return []
    files = [f.name for f in directory.iterdir()
             if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return files[:n]


def get_image_dim(path: Path):
    """Return (H, W) of an image, or None."""
    if not HAS_CV2:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img.shape[:2]


def extract_coco_id_from_sp(filename: str, gen_folder: str):
    """
    SP naming: {coco_id}_mask_random.png_{gen_model}_{var_id}.png
    Returns coco_id string or None.
    """
    # Strip the trailing .png extension
    name = filename
    # Strip any image extension (.png, .jpg, .jpeg)
    for ext in (".png", ".jpg", ".jpeg"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    # Find the gen_model tag
    for tag in ["_flux1filldev_", "_flux1dev_", "_flux1schnell_", "_sd2_", "_sdxl_"]:
        idx = name.rfind(tag)
        if idx != -1:
            coco_part = name[:idx]  # e.g. "123456_mask_random.png"
            # coco_id is the part before _mask_random.png
            if "_mask_random.png" in coco_part:
                coco_id = coco_part.split("_mask_random.png")[0]
                return coco_id
    return None


def build_expected_mask_name(coco_id: str):
    """Mask filename: {coco_id}_mask_random.png"""
    return f"{coco_id}_mask_random.png"


# -- Main inspection -----------------------------------------------------------

def inspect(root: Path, output_path: Path):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  TGIF2 Random Dataset Inspection Report")
    log(f"  Root: {root}")
    log("=" * 70)

    if not root.exists():
        log(f"\n[ERROR] Root directory does not exist: {root}")
        log("  Please download TGIF2 Random from:")
        log("  https://cloud.ilabt.imec.be/index.php/s/GDGewtTFcHccaNj")
        log("  and set --root to the extracted directory.\n")
        _write_report(output_path, lines)
        return

    # -- 1. Authentic originals -----------------------------------------------
    log("\n-- Authentic Originals ---------------------------------------------")
    total_auth = 0
    for split in SPLITS:
        d = root / AUTHENTIC_DIR / split
        n = count_images(d)
        total_auth += n
        samples = sample_files(d, 3)
        log(f"  {split:12s}: {n:6d} images   e.g. {samples}")
    log(f"  {'TOTAL':12s}: {total_auth:6d}")

    # -- 2. SP generators -----------------------------------------------------
    log("\n-- SP (Spliced-in-Place) Generators --------------------------------")
    sp_counts = defaultdict(lambda: defaultdict(int))
    sp_total = 0
    for gen, mask_dir_name in SP_GENERATORS.items():
        log(f"\n  Generator: {gen}")
        for split in SPLITS:
            d = root / gen / split
            n = count_images(d)
            sp_counts[gen][split] = n
            sp_total += n
            samples = sample_files(d, 2)
            log(f"    {split:12s}: {n:6d} images   e.g. {samples}")

        # Mask folder
        mask_base = root / mask_dir_name
        log(f"    Mask folder: {mask_dir_name}/")
        for split in SPLITS:
            md = mask_base / split
            nm = count_images(md)
            msamp = sample_files(md, 2)
            log(f"      masks/{split:12s}: {nm:6d}   e.g. {msamp}")

    log(f"\n  SP TOTAL (all generators + splits): {sp_total}")

    # -- 3. FR generators -----------------------------------------------------
    log("\n-- FR (Fully Regenerated) Generators -------------------------------")
    log("   (Kept separate — not used for localization training)")
    fr_total = 0
    for gen in FR_GENERATORS:
        d = root / gen
        if not d.exists():
            log(f"  {gen}: NOT FOUND")
            continue
        n = count_images(d)
        fr_total += n
        log(f"  {gen}: {n} images")
    log(f"  FR TOTAL: {fr_total}")

    # -- 4. Naming convention examples ----------------------------------------
    log("\n-- Naming Convention Examples ---------------------------------------")
    for gen in SP_GENERATORS:
        d = root / gen / "training"
        samples = sample_files(d, 3)
        if samples:
            log(f"  {gen}/training/")
            for s in samples:
                log(f"    {s}")
                cid = extract_coco_id_from_sp(s, gen)
                log(f"    └- coco_id extracted: {cid}")
                if cid:
                    expected_mask = build_expected_mask_name(cid)
                    log(f"    └- expected mask    : {expected_mask}")
            break

    # -- 5. Image dimensions --------------------------------------------------
    log("\n-- Image Dimension Samples -----------------------------------------")
    if HAS_CV2:
        dims_seen = set()
        for gen in list(SP_GENERATORS.keys())[:1]:
            d = root / gen / "training"
            if d.exists():
                for f in list(d.iterdir())[:20]:
                    if f.suffix.lower() in {".png", ".jpg"}:
                        dim = get_image_dim(f)
                        if dim:
                            dims_seen.add(dim)
        if dims_seen:
            log(f"  Unique (H, W) seen in {gen}/training: {sorted(dims_seen)}")
        else:
            log("  Could not read images (no images found in training split).")
    else:
        log("  [SKIP] cv2 not installed — cannot read image dimensions.")

    # -- 6. Image ↔ Mask correspondence check ---------------------------------
    log("\n-- Image ↔ Mask Correspondence Check (up to 100 pairs) -------------")
    checked = 0
    matched = 0
    missing = []

    for gen, mask_dir_name in SP_GENERATORS.items():
        for split in SPLITS:
            img_dir  = root / gen / split
            mask_dir = root / mask_dir_name / split
            if not img_dir.exists():
                continue
            files = [f for f in img_dir.iterdir()
                     if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            random.shuffle(files)
            for f in files[:25]:
                cid = extract_coco_id_from_sp(f.name, gen)
                if cid is None:
                    continue
                expected = mask_dir / build_expected_mask_name(cid)
                checked += 1
                if expected.exists():
                    matched += 1
                else:
                    missing.append(str(expected))
                if checked >= 100:
                    break
            if checked >= 100:
                break
        if checked >= 100:
            break

    log(f"  Checked : {checked}")
    log(f"  Matched : {matched}")
    log(f"  Missing : {len(missing)}")
    if missing:
        log("  First 5 missing masks:")
        for m in missing[:5]:
            log(f"    {m}")

    # -- 7. Summary -----------------------------------------------------------
    log("\n-- Summary ---------------------------------------------------------")
    log(f"  Authentic originals  : {total_auth:7d}")
    log(f"  SP fake (all gen)    : {sp_total:7d}  <- primary localization data")
    log(f"  FR fake (all gen)    : {fr_total:7d}  <- kept separate")
    log(f"  Mask correspondence  : {matched}/{checked} checked pairs matched")
    log()
    log("  Recommended index files to generate:")
    log("    data/tgif2_random_sp_train.txt   (SP train, with masks)")
    log("    data/tgif2_random_sp_val.txt")
    log("    data/tgif2_random_sp_test.txt")
    log("    data/tgif2_random_au_train.txt   (authentic, no mask column)")
    log("    data/tgif2_random_au_val.txt")
    log("    data/tgif2_random_au_test.txt")
    log()
    log("  Next step: python integrate_tgif2.py --root <TGIF2_RANDOM_ROOT>")
    log("=" * 70)

    _write_report(output_path, lines)
    print(f"\n[inspect_tgif2] Report written to: {output_path}")


def _write_report(output_path: Path, lines):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inspect TGIF2 Random dataset structure.")
    p.add_argument("--root",   default=r"D:\tgif2_random",
                   help="Path to extracted TGIF2 Random root directory")
    p.add_argument("--output", default=r"data/TGIF2_Random/inspection_report.txt",
                   help="Where to write the inspection report")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect(Path(args.root), Path(args.output))


