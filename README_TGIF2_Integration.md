# TGIF2 Random → ForensicsSAM Integration Guide

Complete step-by-step guide to download, inspect, integrate, and verify the
**TGIF2 Random** dataset into the ForensicsSAM project.

---

## 1. Download TGIF2 Random

Download from the official IDLab link:

> **[https://cloud.ilabt.imec.be/index.php/s/GDGewtTFcHccaNj](https://cloud.ilabt.imec.be/index.php/s/GDGewtTFcHccaNj)**

Extract to `D:\tgif2_random`. The expected structure after extraction:

```
D:\tgif2_random\
  original\
    training\     ← authentic COCO images (~2 440)
    validation\   (~341)
    testing\      (~343)
  masks-sd2\
    training\     ← binary random masks for SD2
    validation\
    testing\
  masks-flux\
    training\     ← binary random masks for FLUX models
    validation\
    testing\
  sd2-sp\
    training\     ← SD2 inpainted images (~7 320)
    validation\
    testing\
  flux1schnell-sp\
    training\
    validation\
    testing\
  flux1dev-sp\
    training\
    validation\
    testing\
  flux1filldev-sp\
    training\
    validation\
    testing\
  sd2-fr\         ← KEPT SEPARATE (not used for localization)
  flux1schnell-fr\
  flux1dev-fr\
  flux1filldev-fr\
```

> **Note on disk space:** TGIF2 Random contains ~200k+ images. Expect ~50–100 GB.

---

## 2. Run Inspection

```powershell
cd "D:\sem5\AI Project\ForensicsSAM-main"
& C:\effort_env\Scripts\python.exe inspect_tgif2.py --root D:\tgif2_random
```

This generates `data/TGIF2_Random/inspection_report.txt` and prints:
- Image counts per generator, split, and type
- Naming convention examples
- Mask correspondence check for 100 random pairs

---

## 3. Run Integration

```powershell
& C:\effort_env\Scripts\python.exe integrate_tgif2.py --root D:\tgif2_random
```

This creates the following index files in `data/`:

| File | Contents | Lines (approx.) |
|------|----------|-----------------|
| `tgif2_random_sp_train.txt` | SP fake images + masks, train split | ~21 960 |
| `tgif2_random_sp_val.txt`   | SP fake images + masks, val split   | ~3 069  |
| `tgif2_random_sp_test.txt`  | SP fake images + masks, test split  | ~3 087  |
| `tgif2_random_au_train.txt` | Authentic originals, train split    | ~2 440  |
| `tgif2_random_au_val.txt`   | Authentic originals, val split      | ~341    |
| `tgif2_random_au_test.txt`  | Authentic originals, test split     | ~343    |

And metadata in `data/TGIF2_Random/`:

| File | Contents |
|------|----------|
| `metadata.csv`           | Per-image: path, mask, generator, split, coco_id, variation |
| `dataset_statistics.csv` | Counts per split/generator, fake:real ratio |
| `missing_masks.csv`      | SP images whose mask could not be matched |
| `inspection_report.txt`  | Full inspection output |

---

## 4. Run Verification

```powershell
& C:\effort_env\Scripts\python.exe verify_integration.py --root D:\tgif2_random
```

Checks performed:
- ✅ All index file entries resolve to existing files
- ✅ All mask paths exist and are readable
- ✅ Authentic index files have no mask column
- ✅ Image ↔ mask spatial dimensions match
- ✅ No coco_id leaks across train/val/test splits
- ✅ No duplicate image paths within any split
- ✅ `BasicDataloader` loads a batch of 4 from each split
- ✅ `TGIFDataloader` loads batches correctly with `ai_label`

Report saved to `data/TGIF2_Random/verification_report.txt`.

---

## 5. Use in Training

### Index file format (same as all other ForensicsSAM datasets)

```
# SP (forged) lines — image_rel,mask_rel:
sd2-sp/training/123456_mask_random.png_sd2_0.png,masks-sd2/training/123456_mask_random.png

# Authentic lines — image_rel only (zero mask auto-created):
original/training/123456_orig.png
```

### Add to dataset_list in your training script

```python
TGIF2_ROOT = r"D:\tgif2_random"

dataset_list_train = [
    # ── Existing traditional datasets ──────────────────────────────────────
    (r"D:\datasets\casia", "data/casia1.txt",      1.0, 1, 1, 0),
    (r"D:\datasets\casia", "data/casia1_au.txt",   1.0, 1, 0, 0),
    # ── TGIF2 Random (AI forgery) ──────────────────────────────────────────
    # SP fake images (forged_label=1, adv_label=0, ai_label=1)
    (TGIF2_ROOT, "data/tgif2_random_sp_train.txt", 1.0, 1, 1, 0, 1),
    # Authentic originals upsampled 4× to balance heavy fake ratio
    (TGIF2_ROOT, "data/tgif2_random_au_train.txt", 1.0, 4, 0, 0, 0),
]
```

> **Balancing note:** SP images outnumber authentics ~9:1.
> Setting `load_times=4` for authentic images is a simple upsampling strategy
> that does not duplicate files on disk — the dataloader just iterates them 4×.
> Adjust as needed. This is recorded in `dataset_statistics.csv`.

### Which dataloader to use

| Scenario | Dataloader | Returns |
|----------|-----------|---------|
| Traditional forgery training (no AI label needed) | `BasicDataloader` | `(image, mask, forged_label, adv_label)` |
| TGIF2 / AI forgery training (with AI label) | `TGIFDataloader` | `(image, mask, forged_label, adv_label, ai_label)` |
| Mixed training | `TGIFDataloader` | Same — 6-tuple entries auto-get `ai_label=0` |

---

## 6. What was NOT changed

- ✅ ForensicsSAM model architecture — **untouched**
- ✅ `BasicDataloader` — **untouched**
- ✅ Existing dataset index files — **untouched**
- ✅ Original TGIF2 files — **untouched** (index files reference them in-place)

---

## 7. FR images (Fully Regenerated)

FR images (`sd2-fr`, `flux1schnell-fr`, etc.) are **not integrated** into the
localization pipeline because:

- The whole image is regenerated — the "forged region" is semantically ambiguous
- Mask annotations are less reliable for the entire-image-regeneration case

FR images can still be used for **image-level classification** (fake vs real)
by adding them to the `AIForgeryClassifier` training with no mask requirement.
A separate index file for this purpose can be generated by modifying
`integrate_tgif2.py` to also scan FR generator folders.

---

## 8. Citation

```bibtex
@article{mareen2026tgif2,
  author    = {Mareen, Hannes and Karageorgiou, Dimitrios and Giakoumoglou, Paschalis
               and Lambert, Peter and Papadopoulos, Symeon and Van Wallendael, Glenn},
  title     = {{TGIF2}: Extended Text-Guided Inpainting Forgery Dataset \& Benchmark},
  journal   = {Journal on Information Security},
  year      = {2026},
  publisher = {Springer}
}
```
