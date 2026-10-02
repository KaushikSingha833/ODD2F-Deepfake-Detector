import os
import shutil

dest_dir = r"d:\CMP23-89\Model\ultimate_dataset"

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

for split in ["train", "valid"]:
    for label in ["fake", "real"]:
        os.makedirs(os.path.join(dest_dir, split, label), exist_ok=True)

# 1. Add 4000 StyleGAN images (2000 Real, 2000 Fake)
print("Copying 4000 StyleGAN images...")
sg_src = r"d:\CMP23-89\Model\Final Dataset"
for cls, target in [("Fake", "fake"), ("Real", "real")]:
    imgs = sorted(os.listdir(os.path.join(sg_src, cls)))[:2000]
    for i, img in enumerate(imgs):
        split = "train" if i < 1600 else "valid"
        shutil.copy2(os.path.join(sg_src, cls, img), os.path.join(dest_dir, split, target, f"sg_{img}"))

# 2. Add 1000 CIFAKE images (Midjourney) to prevent forgetting
print("Copying 1000 CIFAKE images to prevent forgetting...")
cf_src = r"d:\CMP23-89\Model\CIFAKE"
for cls, target in [("FAKE", "fake"), ("REAL", "real")]:
    # CIFAKE has train/test split already, we'll just grab from train
    imgs = sorted(os.listdir(os.path.join(cf_src, "train", cls)))[:500]
    for i, img in enumerate(imgs):
        split = "train" if i < 400 else "valid"
        shutil.copy2(os.path.join(cf_src, "train", cls, img), os.path.join(dest_dir, split, target, f"cf_{img}"))

# 3. Add 1000 Original Deepfake images to prevent forgetting
print("Copying 1000 Original Dataset images to prevent forgetting...")
old_src = r"d:\CMP23-89\Model\Dataset"
for cls, target in [("Test Fake", "fake"), ("Test Real", "real")]:
    imgs = sorted(os.listdir(os.path.join(old_src, cls)))[:500]
    for i, img in enumerate(imgs):
        split = "train" if i < 400 else "valid"
        shutil.copy2(os.path.join(old_src, cls, img), os.path.join(dest_dir, split, target, f"old_{img}"))

print("Ultimate Dataset created successfully! 6000 total images mixed safely!")
