import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ODD2F
from ela_utils import compute_ela

def test_usernot_folder():
    folder_path = r"d:\CMP23-89\Model\userNot"
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
        
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("No images found in the userNot folder!")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ODD2F().to(device)
    model.load_state_dict(torch.load(r"d:\CMP23-89\Model\checkpoints\best_model.pth", map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    correct_fakes = 0
    total = len(images)
    
    print(f"Found {total} images in userNot folder. Running detection...\n")
    
    with torch.no_grad():
        for i, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            try:
                rgb_img = Image.open(img_path).convert('RGB')
                ela_img = compute_ela(img_path).convert('RGB')
                
                rgb_tensor = transform(rgb_img).unsqueeze(0).to(device)
                ela_tensor = transform(ela_img).unsqueeze(0).to(device)
                
                outputs = model(rgb_tensor, ela_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                # Class 0 is Fake, Class 1 is Real
                if predicted.item() == 0:
                    correct_fakes += 1
                    status = "✅ Correctly detected as FAKE"
                else:
                    status = "❌ Missed! Predicted as REAL"
                    
                print(f"[{i+1}/{total}] {img_name}: {status}")
            except Exception as e:
                print(f"[{i+1}/{total}] {img_name}: Failed to process ({e})")
                
    print(f"\nFinal Score: {correct_fakes} / {total} StyleGAN images correctly detected as FAKE ({(correct_fakes/total)*100:.1f}%)")

if __name__ == "__main__":
    test_usernot_folder()
