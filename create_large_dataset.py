import os
import shutil
import random

source_dir = r"d:\CMP23-89\Model\archive\real_vs_fake\real-vs-fake\train"
dest_dir = r"d:\CMP23-89\Model\large_dataset"

# Clean up old large_dataset
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# 2,500 total images (1,250 Fake, 1,250 Real)
IMAGES_PER_CLASS = 1250 

for class_name in ["fake", "real"]:
    # Create directories
    os.makedirs(os.path.join(dest_dir, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", class_name), exist_ok=True)

    src_class_dir = os.path.join(source_dir, class_name)
    all_images = os.listdir(src_class_dir)
    # Sort alphabetically for determinism
    all_images.sort()
    
    # Grab the first 5000 images
    selected_images = all_images[:IMAGES_PER_CLASS]
    
    # 80% train (4000), 20% valid (1000)
    train_images = selected_images[:int(IMAGES_PER_CLASS * 0.8)]
    valid_images = selected_images[int(IMAGES_PER_CLASS * 0.8):]
    
    print(f"Copying {IMAGES_PER_CLASS} {class_name} images...")
    for img in train_images:
        shutil.copy2(os.path.join(src_class_dir, img), os.path.join(dest_dir, "train", class_name, img))
    for img in valid_images:
        shutil.copy2(os.path.join(src_class_dir, img), os.path.join(dest_dir, "valid", class_name, img))

print("Massive dataset chunk created successfully!")
