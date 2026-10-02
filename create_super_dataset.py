import os
import shutil

dest_dir = r"d:\CMP23-89\Model\super_dataset"

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# 450 from each = 1350 per class (2700 images total for fast 2.5 hour training)
NUM_EACH = 450

datasets = [
    {"path": r"d:\CMP23-89\Model\archive\real_vs_fake\real-vs-fake\train", "fake": "fake", "real": "real", "prefix": "old"},
    {"path": r"d:\CMP23-89\Model\archive (1)\train", "fake": "FAKE", "real": "REAL", "prefix": "cifake"},
    {"path": r"d:\CMP23-89\Model\Final Dataset", "fake": "Fake", "real": "Real", "prefix": "stylegan"}
]

for class_idx, class_name in enumerate(["fake", "real"]):
    os.makedirs(os.path.join(dest_dir, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "valid", class_name), exist_ok=True)
    
    for ds in datasets:
        src_folder = os.path.join(ds["path"], ds[class_name])
        imgs = sorted(os.listdir(src_folder))[:NUM_EACH]
        
        # 80/20 Train/Valid Split
        train_imgs = imgs[:int(NUM_EACH * 0.8)]
        valid_imgs = imgs[int(NUM_EACH * 0.8):]
        
        for img in train_imgs:
            shutil.copy2(os.path.join(src_folder, img), os.path.join(dest_dir, "train", class_name, f"{ds['prefix']}_{img}"))
        for img in valid_imgs:
            shutil.copy2(os.path.join(src_folder, img), os.path.join(dest_dir, "valid", class_name, f"{ds['prefix']}_{img}"))

print("Super Dataset Created Successfully!")
print("It contains a perfect mix of:")
print("- Older Deepfakes (DeepFaceLab)")
print("- StyleGAN2 (thispersondoesnotexist.com)")
print("- Latent Diffusion (Midjourney/DALL-E)")
