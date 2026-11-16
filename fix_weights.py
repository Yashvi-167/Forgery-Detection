"""Move extracted weights from weight/weight/ up to weight/ and run inference."""
import shutil, os

# Fix nested extraction
src = "weight/weight"
dst = "weight"
if os.path.isdir(src):
    for f in os.listdir(src):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
        print(f"  Moved: {f}")
    os.rmdir(src)
    print("Weight files in correct location.")
else:
    print("Already in correct location.")

print("\nFinal weight/ contents:")
for f in sorted(os.listdir(dst)):
    path = os.path.join(dst, f)
    if os.path.isfile(path):
        mb = os.path.getsize(path) / 1e6
        print(f"  {f:<40} {mb:.0f} MB")
