import os
import zipfile
from PIL import Image
from tqdm import tqdm
import shutil

# === CONFIG ===
zip_path = "datasets/archive.zip"     # input zip
work_dir = "datasets"              # output root dir
hr_dir = os.path.join(work_dir, "ffhq_hr")
lr_dir = os.path.join(work_dir, "ffhq_lr")
lr_size = 64                       # target LR resolution (change if needed)
repack = True                      # set True if you want zipped datasets again

# === STEP 1: Unpack zip into HR dir ===
print("Unpacking HR images...")
os.makedirs(hr_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(hr_dir)

# Some zips may have nested folders (e.g., ffhq/00001.png)
# Move all images into hr_dir root
for root, _, files in os.walk(hr_dir):
    for fname in files:
        if fname.lower().endswith((".png", ".jpg")):
            src = os.path.join(root, fname)
            dst = os.path.join(hr_dir, fname)
            if src != dst:
                shutil.move(src, dst)

# === STEP 2: Create LR dataset ===
print("Creating LR images...")
os.makedirs(lr_dir, exist_ok=True)

for fname in tqdm(os.listdir(hr_dir)):
    if fname.lower().endswith((".png", ".jpg")):
        hr_path = os.path.join(hr_dir, fname)
        lr_path = os.path.join(lr_dir, fname)

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
        lr_img.save(lr_path)

# === STEP 3: Repack into zips (optional) ===
if repack:
    print("Repacking into zips...")
    for folder in [hr_dir, lr_dir]:
        out_zip = folder + ".zip"
        with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(folder):
                zf.write(os.path.join(folder, fname), fname)

print("✅ Done!")
print(f"HR images: {hr_dir}")
print(f"LR images: {lr_dir}")
if repack:
    print(f"HR zip: {hr_dir}.zip")
    print(f"LR zip: {lr_dir}.zip")
