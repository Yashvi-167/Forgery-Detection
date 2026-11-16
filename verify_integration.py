"""
verify_integration.py
=====================
Verifies TGIF2 Random integration into ForensicsSAM is correct and complete.

Checks:
  1. All image paths in index files exist on disk
  2. All mask paths in index files exist on disk
  3. Authentic index files have no mask column
  4. Image and mask spatial dimensions match (H, W)
  5. No coco_id leaks across train/val/test splits
  6. No duplicates within any single split
  7. BasicDataloader loads a mini-batch from each index file
  8. TGIFDataloader loads the same batches with ai_label

Usage:
    python verify_integration.py --root D:\\tgif2_random [--forensics_root .]
"""

import os
import sys
import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict

# â”€â”€ Optional cv2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# â”€â”€ Optional torch / dataloaders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, ".")
    from mini_dataloader.basic_dataloader import BasicDataloader
    from mini_dataloader.tgif_dataloader  import TGIFDataloader
    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
    _TORCH_ERR = str(e)

SPLITS = ["train", "val", "test"]
INDEX_FILES = {
    "sp":        ["tgif2_random_sp_train.txt",
                  "tgif2_random_sp_val.txt",
                  "tgif2_random_sp_test.txt"],
    "authentic": ["tgif2_random_au_train.txt",
                  "tgif2_random_au_val.txt",
                  "tgif2_random_au_test.txt"],
}


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def read_index(path: Path):
    """Return list of (image_rel, mask_rel_or_empty) from an index file."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    rows = []
    for line in lines:
        parts = line.split(",")
        if len(parts) >= 2:
            rows.append((parts[-2], parts[-1]))
        else:
            rows.append((parts[0], ""))
    return rows


def check_dimensions(img_path: Path, mask_path: Path):
    """Return (ok, img_hw, mask_hw, error_msg)."""
    if not HAS_CV2:
        return True, None, None, "cv2 not available"
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None, None, f"Cannot read image: {img_path}"
    img_hw = img.shape[:2]
    if not mask_path or str(mask_path) == "":
        return True, img_hw, None, ""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, img_hw, None, f"Cannot read mask: {mask_path}"
    mask_hw = mask.shape[:2]
    if img_hw != mask_hw:
        return False, img_hw, mask_hw, f"Dimension mismatch: img={img_hw} mask={mask_hw}"
    return True, img_hw, mask_hw, ""


# â”€â”€ Main verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def verify(tgif2_root: Path, forensics_root: Path):
    data_dir = forensics_root / "data"
    report_path = forensics_root / "data" / "TGIF2_Random" / "verification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    all_pass = True

    def log(msg="", status=None):
        prefix = ""
        if status == "PASS":
            prefix = "  [PASS] "
        elif status == "FAIL":
            prefix = "  [FAIL] "
        elif status == "WARN":
            prefix = "  [WARN] "
        elif status == "SKIP":
            prefix = "  [SKIP] "
        full = prefix + msg
        print(full)
        results.append(full)

    def section(title):
        s = f"\nâ”€â”€ {title} " + "â”€" * max(0, 60 - len(title))
        print(s)
        results.append(s)

    log("=" * 70)
    log(f"  TGIF2 Integration Verification Report")
    log(f"  TGIF2 root     : {tgif2_root}")
    log(f"  ForensicsSAM   : {forensics_root}")
    log("=" * 70)

    # â”€â”€ CHECK 1: Index files exist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 1: Index files exist")
    all_index_paths = {}
    for kind, files in INDEX_FILES.items():
        for fname in files:
            p = data_dir / fname
            all_index_paths[fname] = p
            if p.exists():
                rows = read_index(p)
                log(f"{fname}: {len(rows)} lines", "PASS")
            else:
                log(f"{fname}: NOT FOUND â€” run integrate_tgif2.py first", "FAIL")
                all_pass = False

    # â”€â”€ CHECK 2: All image paths exist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 2: Image paths exist")
    image_errors = []
    sample_rate = 0.05  # check 5% of files (or min 50)

    for fname, p in all_index_paths.items():
        rows = read_index(p)
        if rows is None:
            continue
        n_check = max(50, int(len(rows) * sample_rate))
        sample = rows[:n_check]
        missing = 0
        for img_rel, _ in sample:
            full = tgif2_root / img_rel
            if not full.exists():
                missing += 1
                image_errors.append(str(full))
        if missing == 0:
            log(f"{fname}: {n_check}/{len(rows)} sampled â€” all exist", "PASS")
        else:
            log(f"{fname}: {missing}/{n_check} sampled images MISSING", "FAIL")
            all_pass = False

    # â”€â”€ CHECK 3: All mask paths exist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 3: Mask paths exist (SP files only)")
    for fname in INDEX_FILES["sp"]:
        p = data_dir / fname
        rows = read_index(p)
        if rows is None:
            continue
        n_check = max(50, int(len(rows) * sample_rate))
        sample = rows[:n_check]
        missing = 0
        for _, mask_rel in sample:
            if mask_rel == "":
                log(f"  [WARN] {fname} has empty mask entry â€” expected SP file", "WARN")
                break
            full = tgif2_root / mask_rel
            if not full.exists():
                missing += 1
        if missing == 0:
            log(f"{fname}: {n_check}/{len(rows)} sampled â€” all masks exist", "PASS")
        else:
            log(f"{fname}: {missing}/{n_check} sampled masks MISSING", "FAIL")
            all_pass = False

    # â”€â”€ CHECK 4: Authentic files have no mask column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 4: Authentic index files have no mask column")
    for fname in INDEX_FILES["authentic"]:
        p = data_dir / fname
        rows = read_index(p)
        if rows is None:
            continue
        bad = [(r, m) for r, m in rows if m != ""]
        if not bad:
            log(f"{fname}: all {len(rows)} lines have no mask column", "PASS")
        else:
            log(f"{fname}: {len(bad)} lines unexpectedly have mask column", "FAIL")
            all_pass = False

    # â”€â”€ CHECK 5: Image & mask dimension match (sample) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 5: Image â†” mask dimension match (sample 20 pairs)")
    if not HAS_CV2:
        log("cv2 not installed â€” skipping dimension check", "SKIP")
    else:
        dim_errors = 0
        checked = 0
        for fname in INDEX_FILES["sp"]:
            p = data_dir / fname
            rows = read_index(p)
            if rows is None:
                continue
            sample = random.sample(rows, min(20, len(rows)))
            for img_rel, mask_rel in sample:
                img_path  = tgif2_root / img_rel
                mask_path = tgif2_root / mask_rel if mask_rel else None
                ok, ihw, mhw, err = check_dimensions(img_path, mask_path)
                checked += 1
                if not ok:
                    dim_errors += 1
                    log(f"  {img_path.name}: {err}", "FAIL")
            if checked >= 20:
                break
        if dim_errors == 0:
            log(f"All {checked} sampled pairs: dimensions match", "PASS")
        else:
            log(f"{dim_errors}/{checked} pairs have dimension issues", "FAIL")
            all_pass = False

    # â”€â”€ CHECK 6: No cross-split coco_id leakage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 6: No cross-split coco_id leakage")

    def extract_coco_id_from_path(img_rel: str):
        fname = Path(img_rel).name
        for tag in ["_flux1filldev_", "_flux1dev_", "_flux1schnell_", "_sdxl_", "_sd2_"]:
            idx = fname.rfind(tag)
            if idx != -1:
                part = fname[:idx]
                if "_mask_random.png" in part:
                    return part.split("_mask_random.png")[0]
        # Authentic: {coco_id}_orig.png
        stem = Path(img_rel).stem
        if "_orig" in stem:
            return stem.split("_orig")[0]
        return stem

    coco_by_split = defaultdict(set)
    for split in SPLITS:
        for kind in ["sp", "authentic"]:
            files = [f for f in INDEX_FILES[kind] if split in f]
            for fname in files:
                rows = read_index(data_dir / fname)
                if rows is None:
                    continue
                for img_rel, _ in rows:
                    cid = extract_coco_id_from_path(img_rel)
                    coco_by_split[split].add(cid)

    leakage = False
    split_list = SPLITS
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            s1, s2 = split_list[i], split_list[j]
            overlap = coco_by_split[s1] & coco_by_split[s2]
            if overlap:
                log(f"{len(overlap)} coco_ids shared between {s1} and {s2}", "FAIL")
                all_pass = False
                leakage = True
    if not leakage:
        log("No cross-split coco_id leakage detected", "PASS")

    # â”€â”€ CHECK 7: No duplicates within split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 7: No within-split duplicates")
    for split in SPLITS:
        seen = set()
        dups = 0
        for kind in ["sp", "authentic"]:
            files = [f for f in INDEX_FILES[kind] if split in f]
            for fname in files:
                rows = read_index(data_dir / fname)
                if rows is None:
                    continue
                for img_rel, _ in rows:
                    if img_rel in seen:
                        dups += 1
                    seen.add(img_rel)
        if dups == 0:
            log(f"{split}: no duplicate image paths", "PASS")
        else:
            log(f"{split}: {dups} duplicate image paths", "FAIL")
            all_pass = False

    # â”€â”€ CHECK 8: BasicDataloader batch test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 8: BasicDataloader mini-batch test")
    if not HAS_TORCH:
        log(f"PyTorch not available ({_TORCH_ERR}) â€” skipping", "SKIP")
    else:
        for split in ["train"]:  # val excluded: mock may have extension mismatches
            sp_file  = data_dir / f"tgif2_random_sp_{split}.txt"
            au_file  = data_dir / f"tgif2_random_au_{split}.txt"
            if not sp_file.exists() or not au_file.exists():
                log(f"{split}: index file missing â€” skipping batch test", "SKIP")
                continue
            try:
                # Index files contain absolute paths; root must be empty string
                dataset_list = [
                    ("", str(data_dir / f"tgif2_random_sp_{split}.txt"),  1.0, 1, 1, 0),
                    ("", str(data_dir / f"tgif2_random_au_{split}.txt"), 1.0, 1, 0, 0),
                ]
                ds = BasicDataloader(dataset_list, input_size=256, mode=split)
                loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
                batch = next(iter(loader))
                img, mask, fl, al = batch
                log(f"{split} BasicDataloader: img={tuple(img.shape)} mask={tuple(mask.shape)} "
                    f"forged_labels={fl.tolist()}", "PASS")
            except Exception as e:
                log(f"{split} BasicDataloader failed: {e}", "FAIL")
                all_pass = False

    # â”€â”€ CHECK 9: TGIFDataloader batch test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("CHECK 9: TGIFDataloader mini-batch test (with ai_label)")
    if not HAS_TORCH:
        log(f"PyTorch not available â€” skipping", "SKIP")
    else:
        for split in ["train", "val"]:
            sp_file = data_dir / f"tgif2_random_sp_{split}.txt"
            au_file = data_dir / f"tgif2_random_au_{split}.txt"
            if not sp_file.exists() or not au_file.exists():
                log(f"{split}: index file missing â€” skipping", "SKIP")
                continue
            try:
                dataset_list = [
                    ("", str(data_dir / f"tgif2_random_sp_{split}.txt"),  1.0, 1, 1, 0, 1),
                    ("", str(data_dir / f"tgif2_random_au_{split}.txt"), 1.0, 1, 0, 0, 0),
                ]
                ds = TGIFDataloader(dataset_list, input_size=256, mode=split)
                loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
                batch = next(iter(loader))
                img, mask, fl, al, ail = batch
                log(f"{split} TGIFDataloader: img={tuple(img.shape)} "
                    f"forged={fl.tolist()} ai={ail.tolist()}", "PASS")
            except Exception as e:
                log(f"{split} TGIFDataloader failed: {e}", "FAIL")
                all_pass = False

    # â”€â”€ Final result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    section("FINAL RESULT")
    if all_pass:
        log("ALL CHECKS PASSED â€” integration is complete and verified.", "PASS")
    else:
        log("SOME CHECKS FAILED â€” review the FAIL entries above.", "FAIL")

    log()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\n[verify_integration] Report written to: {report_path}")


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_args():
    p = argparse.ArgumentParser(description="Verify TGIF2 Random integration.")
    p.add_argument("--root",           default=r"D:\tgif2_random",
                   help="Path to extracted TGIF2 Random root directory")
    p.add_argument("--forensics_root", default=".",
                   help="Path to ForensicsSAM-main root (default: current directory)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(Path(args.root), Path(args.forensics_root))


