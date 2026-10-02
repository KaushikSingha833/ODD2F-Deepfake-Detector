import os
import shutil
import random

dest_dir = r"d:\CMP23-89\Model\stylegan_dataset"

# Clean up old stylegan_dataset if it exists
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# 2000 Fake + 2000 Real = 4000 total images.
# This is double what we used before, but small enough that it won't crash 
# your laptop and will still finish in about 1.5 - 2 hours!
NUM_EACH = 2000

src_folder = r"d:\CMP23-89\Model\Final Dataset"

for class_idx, class_name in enumerate(["Fake", "Real"]):
    target_class = class_name.lower()
    os.makedirs(os.path.join(dest_dir, "train", target_class), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", target_class), exist_ok=True)
    
    class_folder = os.path.join(src_folder, class_name)
    imgs = sorted(os.listdir(class_folder))[:NUM_EACH]
    
    # 80/20 Train/Valid Split
    train_imgs = imgs[:int(NUM_EACH * 0.8)]
    valid_imgs = imgs[int(NUM_EACH * 0.8):]
    
    print(f"Copying {len(imgs)} images for {class_name}...")
    for img in train_imgs:
        shutil.copy2(os.path.join(class_folder, img), os.path.join(dest_dir, "train", target_class, f"sg_{img}"))
    for img in valid_imgs:
        shutil.copy2(os.path.join(class_folder, img), os.path.join(dest_dir, "valid", target_class, f"sg_{img}"))

print(f"\nStyleGAN Dataset created successfully with {NUM_EACH * 2} images!")
print("This will give it massive exposure to StyleGAN without melting your CPU.")
