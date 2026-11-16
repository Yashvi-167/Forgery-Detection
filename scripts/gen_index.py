"""Generate index files from D:\\dataset for AI forgery training."""
import os

DATASET_ROOT = r"D:\dataset"
OUT_DIR = r"D:\sem5\AI Project\ForensicsSAM-main\data"
os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "val"]:
    # Fake (AI-forged) images
    fake_dir = os.path.join(DATASET_ROOT, split, "fake")
    fake_files = sorted(os.listdir(fake_dir))
    fake_lines = [os.path.join(split, "fake", f) for f in fake_files if f.endswith((".jpg", ".png"))]

    # Real (authentic) images
    real_dir = os.path.join(DATASET_ROOT, split, "real")
    real_files = sorted(os.listdir(real_dir))
    real_lines = [os.path.join(split, "real", f) for f in real_files if f.endswith((".jpg", ".png"))]

    suffix = "" if split == "train" else "_val"

    # Write fake index
    fake_path = os.path.join(OUT_DIR, f"tgif{suffix}.txt")
    with open(fake_path, "w") as f:
        f.write("\n".join(fake_lines))
    print(f"Wrote {len(fake_lines)} lines -> {fake_path}")

    # Write real index
    real_path = os.path.join(OUT_DIR, f"tgif_au{suffix}.txt")
    with open(real_path, "w") as f:
        f.write("\n".join(real_lines))
    print(f"Wrote {len(real_lines)} lines -> {real_path}")

print("\nDone! Index files created.")
