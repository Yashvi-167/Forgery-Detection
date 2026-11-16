"""
integrate_tgif2.py
==================
Integrates TGIF2 Random into the ForensicsSAM data pipeline.

What it does:
  1. Discovers all SP images and matches them to their masks
  2. Discovers authentic originals
  3. Detects and skips exact duplicates (SHA-256)
  4. Generates ForensicsSAM-compatible index files (.txt)
  5. Writes metadata.csv, dataset_statistics.csv, missing_masks.csv
  6. Does NOT modify original TGIF2 files
  7. Does NOT overwrite existing datasets

Usage:
    python integrate_tgif2.py --root D:\\tgif2_random [--forensics_root .]

After running, add these to your training dataset_list:
    (r"D:\\tgif2_random", "data/tgif2_random_sp_train.txt", 1.0, 1, 1, 0, 1)
    (r"D:\\tgif2_random", "data/tgif2_random_au_train.txt", 1.0, 4, 0, 0, 0)
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import random
from pathlib import Path
from collections import defaultdict

SPLITS_MAP = {          # TGIF2 folder name  ->  output split name
    "training":   "train",
    "validation": "val",
    "testing":    "test",
}

# SP generator -> its mask subfolder under tgif2_root
SP_GENERATORS = {
    "sd2-sp":          "masks-sd2",
    "flux1schnell-sp": "masks-flux",
    "flux1dev-sp":     "masks-flux",
    "flux1filldev-sp": "masks-flux",
}

AUTHENTIC_DIR = "original"

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


# -- Naming helpers ------------------------------------------------------------

def extract_coco_id(filename: str):
    """
    SP naming: {coco_id}_mask_random.png_{gen}_{var_id}.png
    Returns (coco_id, gen_tag, var_id) or (None, None, None).
    """
    name = filename
    # Strip any image extension (.png, .jpg, .jpeg)
    for ext in (".png", ".jpg", ".jpeg"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    # gen tags ordered longest-first to avoid partial match
    for tag in ["_flux1filldev_", "_flux1dev_", "_flux1schnell_", "_sdxl_", "_sd2_"]:
        idx = name.rfind(tag)
        if idx != -1:
            coco_part = name[:idx]  # e.g. "123456_mask_random.png"
            var_part  = name[idx + len(tag):]  # e.g. "0"
            if "_mask_random.png" in coco_part:
                coco_id = coco_part.split("_mask_random.png")[0]
                return coco_id, tag.strip("_"), var_part
    return None, None, None


def mask_filename(coco_id: str):
    return f"{coco_id}_mask_random.png"


def sha256(path: Path, chunk=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# -- Core integration ----------------------------------------------------------

def integrate(tgif2_root: Path, forensics_root: Path):
    data_dir = forensics_root / "data"
    out_dir  = data_dir / "TGIF2_Random"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TGIF2 Random -> ForensicsSAM Integration")
    print(f"  TGIF2 root     : {tgif2_root}")
    print(f"  ForensicsSAM   : {forensics_root}")
    print(f"  Output index   : {data_dir}")
    print(f"  Output metadata: {out_dir}")
    print("=" * 70)

    if not tgif2_root.exists():
        print(f"\n[ERROR] TGIF2 root not found: {tgif2_root}")
        print("  Download from: https://cloud.ilabt.imec.be/index.php/s/GDGewtTFcHccaNj")
        print("  Then re-run: python integrate_tgif2.py --root <download_path>")
        sys.exit(1)

    # -- Containers -----------------------------------------------------------
    # SP records: list of dicts with keys: image_path, mask_path, generator,
    #             tgif_split, out_split, coco_id, variation_id
    sp_records  = []
    # Authentic records: list of dicts: image_path, tgif_split, out_split, coco_id
    au_records  = []
    missing_masks  = []
    duplicate_hashes = set()
    errors = []

    # -- STEP 1: Discover SP images + match masks ------------------------------
    print("\n[1/5] Discovering SP images and matching masks...")

    for gen, mask_dir_name in SP_GENERATORS.items():
        gen_dir  = tgif2_root / gen
        mask_base = tgif2_root / mask_dir_name
        if not gen_dir.exists():
            print(f"  [WARN] Generator folder not found, skipping: {gen_dir}")
            continue

        for tgif_split, out_split in SPLITS_MAP.items():
            img_dir  = gen_dir / tgif_split
            mask_dir = mask_base / tgif_split
            if not img_dir.exists():
                continue

            files = [f for f in sorted(img_dir.iterdir())
                     if f.suffix.lower() in IMAGE_EXTS]

            for f in files:
                coco_id, gen_tag, var_id = extract_coco_id(f.name)
                if coco_id is None:
                    errors.append(f"Cannot parse coco_id from: {f}")
                    continue

                expected_mask = mask_dir / mask_filename(coco_id)

                if not expected_mask.exists():
                    missing_masks.append({
                        "image": str(f),
                        "expected_mask": str(expected_mask),
                        "generator": gen,
                        "split": out_split,
                    })
                    continue  # skip â€” no reliable mask

                sp_records.append({
                    "image_path":   f,
                    "mask_path":    expected_mask,
                    "generator":    gen,
                    "tgif_split":   tgif_split,
                    "out_split":    out_split,
                    "coco_id":      coco_id,
                    "variation_id": var_id,
                })

    print(f"  SP records found       : {len(sp_records)}")
    print(f"  Missing masks          : {len(missing_masks)}")
    print(f"  Parse errors           : {len(errors)}")

    # -- STEP 2: Discover authentic originals ----------------------------------
    print("\n[2/5] Discovering authentic originals...")

    auth_dir = tgif2_root / AUTHENTIC_DIR
    if not auth_dir.exists():
        print(f"  [WARN] Authentic directory not found: {auth_dir}")
    else:
        for tgif_split, out_split in SPLITS_MAP.items():
            d = auth_dir / tgif_split
            if not d.exists():
                continue
            files = sorted([f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS])
            for f in files:
                # Authentic naming: {coco_id}_orig.png or {coco_id}_orig_{size}.png
                coco_id = f.stem.split("_orig")[0] if "_orig" in f.stem else f.stem
                au_records.append({
                    "image_path": f,
                    "tgif_split": tgif_split,
                    "out_split":  out_split,
                    "coco_id":    coco_id,
                })

    print(f"  Authentic records found: {len(au_records)}")

    # -- STEP 3: Duplicate detection (SHA-256) ---------------------------------
    print("\n[3/5] Duplicate detection (SHA-256)...")
    seen_hashes = {}
    deduped_sp = []
    dup_count = 0
    for rec in sp_records:
        h = sha256(rec["image_path"])
        if h in seen_hashes:
            dup_count += 1
            print(f"  [DUP] {rec['image_path'].name} == {seen_hashes[h]}")
        else:
            seen_hashes[h] = rec["image_path"].name
            deduped_sp.append(rec)

    print(f"  Duplicates removed: {dup_count}")
    print(f"  SP records after dedup: {len(deduped_sp)}")
    sp_records = deduped_sp

    # -- STEP 4: Cross-split leakage check by coco_id -------------------------
    print("\n[4/5] Cross-split leakage check by coco_id...")
    coco_by_split = defaultdict(set)
    for rec in sp_records:
        coco_by_split[rec["out_split"]].add(rec["coco_id"])
    for rec in au_records:
        coco_by_split[rec["out_split"]].add(rec["coco_id"])

    splits_list = list(SPLITS_MAP.values())
    leakage_count = 0
    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            s1, s2 = splits_list[i], splits_list[j]
            overlap = coco_by_split[s1] & coco_by_split[s2]
            if overlap:
                print(f"  [WARN] {len(overlap)} coco_ids appear in both {s1} and {s2}")
                leakage_count += len(overlap)
    if leakage_count == 0:
        print("  No cross-split coco_id leakage detected.")

    # -- STEP 5: Write index files ---------------------------------------------
    print("\n[5/5] Writing index files...")

    # Paths in index files are RELATIVE to tgif2_root
    # so the dataloader is called with root=tgif2_root

    index_files = {}
    for out_split in ["train", "val", "test"]:
        # SP (forged with masks)
        # We write ABSOLUTE paths so BasicDataloader can be called with root=""
        sp_key = f"tgif2_random_sp_{out_split}.txt"
        sp_path = data_dir / sp_key
        sp_lines = []
        for rec in sp_records:
            if rec["out_split"] != out_split:
                continue
            img_abs  = rec["image_path"].as_posix()
            mask_abs = rec["mask_path"].as_posix()
            sp_lines.append(f"{img_abs},{mask_abs}")

        with open(sp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sp_lines))
        index_files[sp_key] = len(sp_lines)
        print(f"  Wrote {sp_key}: {len(sp_lines)} lines")

        # Authentic (no mask column)
        au_key = f"tgif2_random_au_{out_split}.txt"
        au_path = data_dir / au_key
        au_lines = []
        for rec in au_records:
            if rec["out_split"] != out_split:
                continue
            img_abs = rec["image_path"].as_posix()
            au_lines.append(img_abs)

        with open(au_path, "w", encoding="utf-8") as f:
            f.write("\n".join(au_lines))
        index_files[au_key] = len(au_lines)
        print(f"  Wrote {au_key}: {len(au_lines)} lines")

    # -- Write metadata.csv ---------------------------------------------------
    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "mask_path", "generator", "tgif_split",
            "out_split", "coco_id", "variation_id", "type"])
        writer.writeheader()
        for rec in sp_records:
            writer.writerow({
                "image_path":   str(rec["image_path"]),
                "mask_path":    str(rec["mask_path"]),
                "generator":    rec["generator"],
                "tgif_split":   rec["tgif_split"],
                "out_split":    rec["out_split"],
                "coco_id":      rec["coco_id"],
                "variation_id": rec["variation_id"],
                "type":         "SP_FAKE",
            })
        for rec in au_records:
            writer.writerow({
                "image_path":   str(rec["image_path"]),
                "mask_path":    "",
                "generator":    "authentic",
                "tgif_split":   rec["tgif_split"],
                "out_split":    rec["out_split"],
                "coco_id":      rec["coco_id"],
                "variation_id": "",
                "type":         "AUTHENTIC",
            })
    print(f"\n  Wrote metadata: {meta_path}")

    # -- Write dataset_statistics.csv -----------------------------------------
    stats = []
    for out_split in ["train", "val", "test"]:
        sp_n  = sum(1 for r in sp_records if r["out_split"] == out_split)
        au_n  = sum(1 for r in au_records if r["out_split"] == out_split)
        total = sp_n + au_n
        ratio = f"{sp_n}:{au_n}" if au_n > 0 else f"{sp_n}:0"
        stats.append({"split": out_split, "sp_fake": sp_n,
                      "authentic": au_n, "total": total,
                      "fake_to_real_ratio": ratio})
        per_gen = defaultdict(int)
        for r in sp_records:
            if r["out_split"] == out_split:
                per_gen[r["generator"]] += 1
        for g, n in per_gen.items():
            stats.append({"split": f"{out_split}/{g}", "sp_fake": n,
                          "authentic": 0, "total": n, "fake_to_real_ratio": ""})

    stats_path = out_dir / "dataset_statistics.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "sp_fake", "authentic", "total", "fake_to_real_ratio"])
        writer.writeheader()
        writer.writerows(stats)
    print(f"  Wrote statistics: {stats_path}")

    # -- Write missing_masks.csv -----------------------------------------------
    mm_path = out_dir / "missing_masks.csv"
    with open(mm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "expected_mask", "generator", "split"])
        writer.writeheader()
        writer.writerows(missing_masks)
    print(f"  Wrote missing masks: {mm_path} ({len(missing_masks)} entries)")

    # -- Final summary ---------------------------------------------------------
    total_sp = len(sp_records)
    total_au = len(au_records)
    print("\n" + "=" * 70)
    print("  INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"  SP (forged) images integrated : {total_sp}")
    print(f"  Authentic images integrated   : {total_au}")
    print(f"  Missing masks (excluded)      : {len(missing_masks)}")
    print(f"  Duplicates removed            : {dup_count}")
    print(f"  Cross-split leakage           : {leakage_count} coco_ids")
    print()
    print("  Index files created:")
    for k, n in index_files.items():
        print(f"    data/{k}: {n} lines")
    print()
    print("  To use in ForensicsSAM training, add to your dataset_list:")
    print(f'    ("", "data/tgif2_random_sp_train.txt", 1.0, 1, 1, 0, 1)')
    print(f'    ("", "data/tgif2_random_au_train.txt", 1.0, 4, 0, 0, 0)')
    print()
    print("  Next step: python verify_integration.py --root D:\\tgif2_random")
    print("=" * 70)


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Integrate TGIF2 Random into ForensicsSAM.")
    p.add_argument("--root",          default=r"D:\tgif2_random",
                   help="Path to extracted TGIF2 Random root directory")
    p.add_argument("--forensics_root", default=".",
                   help="Path to ForensicsSAM-main root (default: current directory)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    integrate(Path(args.root), Path(args.forensics_root))


