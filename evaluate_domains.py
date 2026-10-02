import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from model import ODD2F
from ela_utils import compute_ela

class CustomValDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.domains = []
        
        for label_idx, class_name in enumerate(["fake", "real"]):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir): continue
            for img_name in os.listdir(class_dir):
                if img_name.startswith("sg_"): domain = "StyleGAN"
                elif img_name.startswith("cf_"): domain = "CIFAKE"
                elif img_name.startswith("old_"): domain = "Original"
                else: domain = "Unknown"
                
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(label_idx)
                self.domains.append(domain)
                
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            rgb_img = Image.open(img_path).convert('RGB')
            ela_img = compute_ela(img_path).convert('RGB')
            if self.transform:
                rgb_img = self.transform(rgb_img)
                ela_img = self.transform(ela_img)
            return rgb_img, ela_img, self.labels[idx], self.domains[idx]
        except:
            return None, None, -1, "Error"

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.Tensor(), torch.Tensor(), torch.Tensor(), []
    rgb, ela, labels, domains = zip(*batch)
    return torch.stack(rgb), torch.stack(ela), torch.tensor(labels), domains

def evaluate_domains():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ODD2F().to(device)
    model.load_state_dict(torch.load(r"d:\CMP23-89\Model\checkpoints\best_model.pth", map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = CustomValDataset(r"d:\CMP23-89\Model\perfect_dataset\valid", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    domain_correct = {"StyleGAN": 0, "CIFAKE": 0, "Original": 0}
    domain_total = {"StyleGAN": 0, "CIFAKE": 0, "Original": 0}
    
    print("Evaluating domains... This will take a moment.")
    with torch.no_grad():
        for rgb, ela, labels, domains in val_loader:
            if len(labels) == 0: continue
            rgb, ela, labels = rgb.to(device), ela.to(device), labels.to(device)
            outputs = model(rgb, ela)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(labels)):
                d = domains[i]
                if d in domain_total:
                    domain_total[d] += 1
                    if predicted[i] == labels[i]:
                        domain_correct[d] += 1
                        
    print("\n--- EXACT DOMAIN ACCURACY REPORT ---")
    for d in ["StyleGAN", "CIFAKE", "Original"]:
        acc = 100.0 * domain_correct[d] / max(1, domain_total[d])
        print(f"{d} Accuracy: {domain_correct[d]}/{domain_total[d]} ({acc:.2f}%)")

if __name__ == "__main__":
    evaluate_domains()
