import os
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from model import ODD2F
from ela_utils import compute_ela
import time

def test_tpdne():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ODD2F().to(device)
    model.load_state_dict(torch.load(r"d:\CMP23-89\Model\checkpoints\best_model.pth", map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    num_tests = 10
    correct_fakes = 0
    
    print(f"Testing {num_tests} live StyleGAN images from thispersondoesnotexist.com...")
    
    for i in range(num_tests):
        try:
            # TPDNE returns a random JPEG image on the root URL or /image
            response = requests.get("https://thispersondoesnotexist.com/image", headers=headers, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Save it temporarily to compute ELA properly (since ELA resaves)
            temp_path = f"temp_tpdne_{i}.jpg"
            img.save(temp_path, format="JPEG", quality=100)
            
            # Prepare inputs
            ela_img = compute_ela(temp_path).convert('RGB')
            rgb_tensor = transform(img).unsqueeze(0).to(device)
            ela_tensor = transform(ela_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(rgb_tensor, ela_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
            # Class 0 is Fake, Class 1 is Real
            if predicted.item() == 0:
                correct_fakes += 1
                result = "Correctly detected as FAKE"
            else:
                result = "Missed! Predicted as REAL"
                
            print(f"Image {i+1}: {result}")
            
            os.remove(temp_path)
            time.sleep(1) # Be polite to the server
            
        except Exception as e:
            print(f"Image {i+1} Failed to download/process: {e}")
            
    print(f"\nFinal Score: {correct_fakes} / {num_tests} StyleGAN images correctly detected as FAKE ({(correct_fakes/num_tests)*100:.1f}%)")

if __name__ == "__main__":
    test_tpdne()
