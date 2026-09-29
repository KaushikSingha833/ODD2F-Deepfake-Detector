import os
import shutil

source_dir = r"d:\CMP23-89\Model\archive\real_vs_fake\real-vs-fake\train"
dest_dir = r"d:\CMP23-89\Model\mini_dataset"

# Clean up old mini_dataset
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

for class_name in ["fake", "real"]:
    # Create directories
    os.makedirs(os.path.join(dest_dir, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", class_name), exist_ok=True)

    src_class_dir = os.path.join(source_dir, class_name)
    all_images = os.listdir(src_class_dir)
    # Sort alphabetically so we can deterministically get the next chunk
    all_images.sort()
    
    # Take the NEXT 250 images (indices 500 to 750 for this chunk)
    # Wait, the user wants the NEXT 500 images total, which is 250 fake + 250 real.
    # So we take 250 per class. Indices 500 to 750.
    selected_images = all_images[500:750]
    
    train_images = selected_images[:200]
    valid_images = selected_images[200:]
    
    print(f"Copying NEXT 250 {class_name} images...")
    for img in train_images:
        shutil.copy2(os.path.join(src_class_dir, img), os.path.join(dest_dir, "train", class_name, img))
    for img in valid_images:
        shutil.copy2(os.path.join(src_class_dir, img), os.path.join(dest_dir, "valid", class_name, img))

print("Next chunk dataset created successfully!")
