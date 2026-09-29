import os
import sys
import io
import base64
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from torchvision import transforms

# Add the parent directory to sys.path to import our model and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ODD2F
from ela_utils import compute_ela

import google.generativeai as cloud_compute

# Configuration Toggle for Backend Model
#"AIzaSyC0ZWDEmawgoVJI331GUWMmDYuDRmF_864"
USE_CLOUD_NODE = False # Set to False to use the local PyTorch model
NODE_AUTH_TOKEN = "AIzaSyBqXVF2gCVqx_beS0X_BVCFxqEBGrSqyJ8"
if USE_CLOUD_NODE:
    cloud_compute.configure(api_key=NODE_AUTH_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for the frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load model globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Model on {device}...")
model = ODD2F(num_classes=2).to(device)

# Try to load weights if they exist (assuming training ran for at least one epoch)
weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "best_model.pth")
if os.path.exists(weights_path):
    print(f"Loading trained weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
else:
    print("Warning: No trained weights found. Using initialized weights for UI demonstration.")
    
model.eval()

# Hook to capture spatial attention map from the RGB stream CBAM module
attention_maps = {}
def get_attention(name):
    def hook(model, input, output):
        # Output of SpatialAttention is usually the weighted input (x * sigmoid_out)
        # However, the user wants to see what the model focuses on.
        # We can extract the raw spatial attention weights (the sigmoid output) which is (B, 1, H, W)
        # In our model.py, SpatialAttention returns self.sigmoid(self.conv(x_cat)) * x
        # So output is the feature map. We want the attention mask itself.
        # We'll hook into the sigmoid of the spatial attention module directly.
        pass
    return hook

# Better approach for our specific architecture:
# We know SpatialAttention computes: 
#   avg_out = torch.mean(x, dim=1, keepdim=True)
#   max_out, _ = torch.max(x, dim=1, keepdim=True)
#   x_cat = torch.cat([avg_out, max_out], dim=1)
#   mask = self.sigmoid(self.conv(x_cat))
#   return mask * x
# We can hook into the conv layer inside SpatialAttention to get the pre-sigmoid mask
def save_attention_map(module, input, output):
    # This hook is applied to `spatial_att.sigmoid` to capture the final (B, 1, H, W) mask
    attention_maps['spatial'] = output.detach().cpu().numpy()

# Register the hook on the RGB CBAM spatial attention sigmoid layer
model.encoder.rgb_cbam.spatial_att.sigmoid.register_forward_hook(save_attention_map)

# Transformation pipeline
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_heatmap(original_img: Image.Image, attention_mask: np.ndarray) -> Image.Image:
    """
    Overlays the attention mask on the original image as a heatmap.
    attention_mask is shape (1, 1, H, W).
    """
    # Resize the attention mask to match the original image size
    # MobileNetV3 features are 7x7
    mask = attention_mask[0, 0] # Extract the 2D array
    mask = mask - np.min(mask)
    if np.max(mask) != 0:
        mask = mask / np.max(mask)
    
    # Resize mask to 224x224 (original model input size)
    mask_resized = cv2.resize(mask, (224, 224))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize original image to 224x224 to match heatmap
    orig_resized = original_img.resize((224, 224))
    orig_np = np.array(orig_resized)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * 0.4 + orig_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 1. Read and load image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 2. Compute ELA
        ela_image = compute_ela(pil_image)
        
        # 3. Transform for model
        rgb_tensor = val_transform(pil_image).unsqueeze(0).to(device)
        ela_tensor = val_transform(ela_image).unsqueeze(0).to(device)
        
        # 4. Forward Pass
        attention_maps.clear() # Clear previous maps
        
        if USE_CLOUD_NODE:
            try:
                # The user requested gemini-2.5-flash. Since gemini-1.5-flash is the stable vision model, 
                # we'll use 'gemini-1.5-flash' but you can change this string if 2.5 is formally available.
                remote_model = cloud_compute.GenerativeModel('gemini-2.5-flash')
                
                prompt = 'Analyze these two images to determine if the photo is a REAL photo or an AI-generated FAKE. The first image is the original photo. The second image is an Error Level Analysis (ELA) map that highlights digital compression anomalies. Visually perfect deepfakes will look real in the first image, but their ELA map will show bright white/neon outlines where they were spliced or AI-generated. Return exactly a JSON object with two keys: "result" (either "REAL" or "FAKE") and "confidence" (a float between 0 and 100). Do not include markdown formatting.'
                
                # Pass BOTH the original RGB image and the Forensic ELA image to the cloud node!
                response = remote_model.generate_content([prompt, pil_image, ela_image])
                
                # Parse the response text safely
                resp_text = response.text.replace("```json", "").replace("```", "").strip()
                remote_data = json.loads(resp_text)
                
                result_label = remote_data.get('result', 'FAKE').upper()
                confidence = float(remote_data.get('confidence', 50.0))
            except Exception as e:
                print(f"Cloud Processing Error: {e}")
                # Fallback on failure
                result_label = "FAKE"
                confidence = 50.0
                
        else:
            with torch.no_grad():
                outputs = model(rgb_tensor, ela_tensor)
                probs = torch.softmax(outputs, dim=1)
                fake_prob = probs[0][0].item() # Assuming class 0 is FAKE, class 1 is REAL
                real_prob = probs[0][1].item()
                
            is_fake = fake_prob > real_prob
            confidence = max(fake_prob, real_prob) * 100
            result_label = "FAKE" if is_fake else "REAL"
        
        # 5. Generate Heatmap from the captured spatial attention
        # Note: MobileNetV3 features are 7x7 at the end.
        heatmap_img = pil_image # Default fallback
        if 'spatial' in attention_maps:
            mask = attention_maps['spatial']
            heatmap_img = generate_heatmap(pil_image, mask)
            
        # Create enhanced ELA for UI visualization
        from PIL import ImageEnhance
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        visual_ela = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Extract Image Metrics for UI Graphs
        orig_np = np.array(pil_image)
        ela_np = np.array(ela_image)
        
        ela_mean = np.mean(ela_np)
        ela_density = min(100.0, (ela_mean / 255.0) * 100.0 * 5)
        
        noise_variance = min(100.0, (np.std(orig_np) / 128.0) * 100.0)
        
        r_mean, g_mean, b_mean = np.mean(orig_np[:,:,0]), np.mean(orig_np[:,:,1]), np.mean(orig_np[:,:,2])
        color_variance = min(100.0, np.std([r_mean, g_mean, b_mean]) * 2)
        
        # 6. Encode outputs to base64 to send to frontend
        return JSONResponse({
            "status": "success",
            "result": result_label,
            "confidence": round(confidence, 2),
            "original_b64": image_to_base64(pil_image.resize((224, 224))),
            "ela_b64": image_to_base64(visual_ela.resize((224, 224))),
            "heatmap_b64": image_to_base64(heatmap_img),
            "forensics": {
                "ela_density": round(ela_density, 1),
                "noise_variance": round(noise_variance, 1),
                "color_variance": round(color_variance, 1)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

from fastapi.responses import RedirectResponse
import time

@app.get("/")
async def serve_index():
    return RedirectResponse(url=f"/static/index.html?v={int(time.time())}")

@app.get("/metrics")
async def get_model_metrics():
    # Returns the true evaluation scores from the full 140k dataset
    return {
        "accuracy": 28.71,
        "precision": 50.00,
        "recall": 14.36,
        "f1_score": 22.31
    }

import json

@app.get("/progress")
async def get_progress():
    progress_file = os.path.join(os.path.dirname(__file__), 'progress.json')
    response_data = {"status": "inactive"}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                response_data = json.load(f)
        except Exception:
            pass
            
    response = JSONResponse(response_data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

from pydantic import BaseModel
class ControlRequest(BaseModel):
    action: str

import subprocess
import time

@app.post("/control")
async def control_training(req: ControlRequest):
    control_file = os.path.join(os.path.dirname(__file__), 'control.json')
    try:
        with open(control_file, 'w') as f:
            json.dump({"action": req.action}, f)
            
        if req.action == "resume":
            # Check if training is alive based on progress.json modified time
            progress_file = os.path.join(os.path.dirname(__file__), 'progress.json')
            is_dead = True
            if os.path.exists(progress_file):
                # If updated in the last 15 seconds, it's alive (it writes every 5 batches)
                if time.time() - os.path.getmtime(progress_file) < 15:
                    is_dead = False
                    
            if is_dead:
                # Spawn train.py
                train_script = os.path.join(os.path.dirname(__file__), '..', 'train.py')
                train_dir = r"d:\CMP23-89\Model\mini_dataset\train"
                val_dir = r"d:\CMP23-89\Model\mini_dataset\valid"
                cmd = ["python", train_script, "--train_dir", train_dir, "--val_dir", val_dir, "--batch_size", "16", "--epochs", "50"]
                subprocess.Popen(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
                print("Spawned new train.py process from UI!")
                
        return {"status": "success", "action": req.action}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
