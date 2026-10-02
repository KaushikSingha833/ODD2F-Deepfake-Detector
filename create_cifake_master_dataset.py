import os
import shutil

dest_dir = r"d:\CMP23-89\Model\cifake_master_dataset"

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

for split in ["train", "valid"]:
    for label in ["fake", "real"]:
        os.makedirs(os.path.join(dest_dir, split, label), exist_ok=True)

# 1. Add 4000 CIFAKE images (Midjourney/Latent Diffusion)
print("Copying 4000 CIFAKE images as the primary training target...")
cf_src = r"d:\CMP23-89\Model\archive (1)\train"
for cls, target in [("FAKE", "fake"), ("REAL", "real")]:
    imgs = sorted(os.listdir(os.path.join(cf_src, cls)))[:2000]
    for i, img in enumerate(imgs):
        split = "train" if i < 1600 else "valid"
        shutil.copy2(os.path.join(cf_src, cls, img), os.path.join(dest_dir, split, target, f"cf_{img}"))

# 2. Add 1000 StyleGAN images to prevent forgetting
print("Copying 1000 StyleGAN images to prevent forgetting...")
sg_src = r"d:\CMP23-89\Model\Final Dataset"
for cls, target in [("Fake", "fake"), ("Real", "real")]:
    imgs = sorted(os.listdir(os.path.join(sg_src, cls)))[:500]
    for i, img in enumerate(imgs):
        split = "train" if i < 400 else "valid"
        shutil.copy2(os.path.join(sg_src, cls, img), os.path.join(dest_dir, split, target, f"sg_{img}"))

# 3. Add 1000 Original Deepfake images to prevent forgetting
print("Copying 1000 Original Dataset images to prevent forgetting...")
old_src = r"d:\CMP23-89\Model\archive\Dataset"
if not os.path.exists(old_src):
    old_src = r"d:\CMP23-89\Model\Dataset" # Fallback if path differs
for cls, target in [("Test Fake", "fake"), ("Test Real", "real")]:
    imgs = sorted(os.listdir(os.path.join(old_src, cls)))[:500]
    for i, img in enumerate(imgs):
        split = "train" if i < 400 else "valid"
        shutil.copy2(os.path.join(old_src, cls, img), os.path.join(dest_dir, split, target, f"old_{img}"))

print("CIFAKE Master Dataset created successfully! 6000 total images mixed safely!")
