import os
import shutil

dest_dir = r"d:\CMP23-89\Model\perfect_dataset"

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

for split in ["train", "valid"]:
    for label in ["fake", "real"]:
        os.makedirs(os.path.join(dest_dir, split, label), exist_ok=True)

# 1. 2000 StyleGAN images
print("Copying 2000 StyleGAN images...")
sg_src = r"d:\CMP23-89\Model\Final Dataset"
for cls, target in [("Fake", "fake"), ("Real", "real")]:
    imgs = sorted(os.listdir(os.path.join(sg_src, cls)))[:1000]
    for i, img in enumerate(imgs):
        split = "train" if i < 800 else "valid"
        shutil.copy2(os.path.join(sg_src, cls, img), os.path.join(dest_dir, split, target, f"sg_{img}"))

# 2. 2000 CIFAKE images
print("Copying 2000 CIFAKE images...")
cf_src = r"d:\CMP23-89\Model\archive (1)\train"
for cls, target in [("FAKE", "fake"), ("REAL", "real")]:
    imgs = sorted(os.listdir(os.path.join(cf_src, cls)))[:1000]
    for i, img in enumerate(imgs):
        split = "train" if i < 800 else "valid"
        shutil.copy2(os.path.join(cf_src, cls, img), os.path.join(dest_dir, split, target, f"cf_{img}"))

# 3. 2000 Original Deepfake images
print("Copying 2000 Original Dataset images...")
old_src = r"d:\CMP23-89\Model\archive\Dataset"
if not os.path.exists(old_src):
    old_src = r"d:\CMP23-89\Model\Dataset"
for cls, target in [("Test Fake", "fake"), ("Test Real", "real")]:
    imgs = sorted(os.listdir(os.path.join(old_src, cls)))[:1000]
    for i, img in enumerate(imgs):
        split = "train" if i < 800 else "valid"
        shutil.copy2(os.path.join(old_src, cls, img), os.path.join(dest_dir, split, target, f"old_{img}"))

print("Perfectly Balanced 1:1:1 Dataset created successfully! 6000 total images.")
