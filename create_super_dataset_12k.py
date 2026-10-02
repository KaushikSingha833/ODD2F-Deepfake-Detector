import os
import shutil

dest_dir = r"d:\CMP23-89\Model\super_dataset_12k"

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

for split in ["train", "valid"]:
    for label in ["fake", "real"]:
        os.makedirs(os.path.join(dest_dir, split, label), exist_ok=True)

def copy_images(src_dir, fake_folder, real_folder, prefix, count=2000):
    print(f"Copying {count*2} images from {prefix}...")
    for cls, target in [(fake_folder, "fake"), (real_folder, "real")]:
        full_src = os.path.join(src_dir, cls)
        if not os.path.exists(full_src):
            print(f"WARNING: Source not found {full_src}")
            continue
        imgs = sorted(os.listdir(full_src))[:count]
        for i, img in enumerate(imgs):
            split = "train" if i < int(count * 0.8) else "valid"
            shutil.copy2(os.path.join(full_src, img), os.path.join(dest_dir, split, target, f"{prefix}_{img}"))

# 1. 4,000 StyleGAN images (2000 fake, 2000 real)
copy_images(r"d:\CMP23-89\Model\Final Dataset", "Fake", "Real", "sg", count=2000)

# 2. 4,000 CIFAKE images (2000 fake, 2000 real)
copy_images(r"d:\CMP23-89\Model\archive (1)\train", "FAKE", "REAL", "cf", count=2000)

# 3. 4,000 Original Deepfake images (2000 fake, 2000 real)
old_src = r"d:\CMP23-89\Model\archive\Dataset"
if not os.path.exists(old_src):
    old_src = r"d:\CMP23-89\Model\Dataset"
copy_images(old_src, "Test Fake", "Test Real", "old", count=2000)

print("Super Balanced 12K Dataset created successfully! 12,000 total images.")
