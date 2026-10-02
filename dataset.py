import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ela_utils import compute_ela

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform_rgb=None, transform_ela=None):
        """
        Assumes data_dir has subdirectories for classes, e.g., 'Real' and 'Fake'
        or '0' and '1'.
        """
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        
        # Sort classes to ensure consistent label mapping
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        self.transform_rgb = transform_rgb
        self.transform_ela = transform_ela

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            rgb_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Handle corrupt images by returning a blank image or raising
            print(f"Error loading {img_path}: {e}")
            rgb_img = Image.new('RGB', (224, 224))
            
        import random
        from io import BytesIO
        
        # Apply Random JPEG Compression to the original image during training
        # This mimics the "in-the-wild" browser compression!
        if self.transform_rgb:
            quality = random.randint(60, 100)
            buffer = BytesIO()
            rgb_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            rgb_img = Image.open(buffer).convert('RGB')
            ela_img = compute_ela(rgb_img, quality=95)
        else:
            ela_img = compute_ela(rgb_img, quality=95)
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = transforms.ToTensor()(rgb_img)
            
        if self.transform_ela:
            ela_tensor = self.transform_ela(ela_img)
        else:
            ela_tensor = transforms.ToTensor()(ela_img)
            
        return rgb_tensor, ela_tensor, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_dir, val_dir, batch_size=32):
    # Data Augmentation based on the paper
    # Horizontal flip, rotation, random crop, color jitter, Gaussian blur
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ELA transformation might need different augmentation
    # (E.g., color jitter might corrupt the ELA meaning, but flip/crop is fine)
    # The paper mentions CutMix, which is usually applied at batch level during training.
    train_transform_ela = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DeepfakeDataset(train_dir, transform_rgb=train_transform, transform_ela=train_transform_ela)
    val_dataset = DeepfakeDataset(val_dir, transform_rgb=val_transform, transform_ela=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
