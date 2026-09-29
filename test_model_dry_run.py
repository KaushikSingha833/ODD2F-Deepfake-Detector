import torch
from model import ODD2F
from ela_utils import compute_ela
from PIL import Image
import numpy as np
from torchvision import transforms

def test_model_architecture():
    print("Initializing ODD2F Model...")
    # Initialize the model (setting pretrained=False for the test to avoid downloading weights just yet)
    model = ODD2F(num_classes=2)
    model.eval()
    
    print("Generating dummy tensors...")
    # Create dummy tensors representing a batch of 2 RGB images (224x224)
    # MobileNetV3 expects 3-channel RGB images
    dummy_rgb = torch.randn(2, 3, 224, 224)
    
    # ELA stream expects the same shape
    dummy_ela = torch.randn(2, 3, 224, 224)
    
    print(f"Passing tensors through the model...")
    print(f"Input RGB shape: {dummy_rgb.shape}")
    print(f"Input ELA shape: {dummy_ela.shape}")
    
    with torch.no_grad():
        output = model(dummy_rgb, dummy_ela)
        
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
    print("\nSuccess! The model architecture compiled and processed the forward pass without shape mismatch errors.")

if __name__ == "__main__":
    test_model_architecture()
