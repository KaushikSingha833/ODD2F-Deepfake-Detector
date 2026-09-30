import os
import shutil

dest_dir = r"d:\CMP23-89\Model\mixed_dataset"

# Clean up old mixed_dataset if it exists
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# We will take 625 images from the old dataset, and 625 from CIFAKE 
# (1250 total per class, 2500 total images = fast 2-hour training!)
NUM_OLD = 625
NUM_CIFAKE = 625

old_source = r"d:\CMP23-89\Model\archive\real_vs_fake\real-vs-fake\train"
cifake_source = r"d:\CMP23-89\Model\archive (1)\train"

for class_name in ["fake", "real"]:
    # Create output directories
    os.makedirs(os.path.join(dest_dir, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", class_name), exist_ok=True)
    
    # 1. Grab old images
    old_cls_dir = os.path.join(old_source, class_name)
    old_imgs = sorted(os.listdir(old_cls_dir))[:NUM_OLD]
    
    # 2. Grab CIFAKE images (CIFAKE uses uppercase folder names)
    cifake_cls_dir = os.path.join(cifake_source, class_name.upper())
    cifake_imgs = sorted(os.listdir(cifake_cls_dir))[:NUM_CIFAKE]
    
    # Process Old Images (80% train, 20% valid)
    train_old = old_imgs[:int(NUM_OLD * 0.8)]
    valid_old = old_imgs[int(NUM_OLD * 0.8):]
    print(f"Copying {len(old_imgs)} old {class_name} images...")
    for img in train_old:
        shutil.copy2(os.path.join(old_cls_dir, img), os.path.join(dest_dir, "train", class_name, f"old_{img}"))
    for img in valid_old:
        shutil.copy2(os.path.join(old_cls_dir, img), os.path.join(dest_dir, "valid", class_name, f"old_{img}"))
        
    # Process CIFAKE Images (80% train, 20% valid)
    train_cifake = cifake_imgs[:int(NUM_CIFAKE * 0.8)]
    valid_cifake = cifake_imgs[int(NUM_CIFAKE * 0.8):]
    print(f"Copying {len(cifake_imgs)} CIFAKE {class_name} images...")
    for img in train_cifake:
        shutil.copy2(os.path.join(cifake_cls_dir, img), os.path.join(dest_dir, "train", class_name, f"cifake_{img}"))
    for img in valid_cifake:
        shutil.copy2(os.path.join(cifake_cls_dir, img), os.path.join(dest_dir, "valid", class_name, f"cifake_{img}"))

print("\nMixed dataset created successfully! 50% GANs, 50% Latent Diffusion.")
print(f"Total Images: {(NUM_OLD + NUM_CIFAKE) * 2}")
