import io
import os
from PIL import Image, ImageChops
import numpy as np

def compute_ela(image_path_or_pil, quality=95):
    """
    Computes the Error Level Analysis (ELA) of an image.
    
    Args:
        image_path_or_pil: Path to the image file or a PIL Image object.
        quality: The JPEG quality level to use for resaving (default is 95 as per paper).
        
    Returns:
        A PIL Image representing the absolute difference (ELA map).
    """
    if isinstance(image_path_or_pil, str):
        original = Image.open(image_path_or_pil).convert('RGB')
    else:
        original = image_path_or_pil.convert('RGB')
        
    # Resave the image in memory with the specified quality
    buffer = io.BytesIO()
    original.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Load the compressed image
    compressed = Image.open(buffer)
    
    # Compute the pixel-wise absolute difference
    ela_image = ImageChops.difference(original, compressed)
    
    # Optional: Enhance the difference to make it more visible (though CNNs can learn from raw diff)
    # The paper says: "Compute the pixel-wise absolute difference between the original and resaved image."
    # We will just return the raw difference image.
    
    # Finding the max difference and enhancing it for better representation (common in ELA)
    # extrama = ela_image.getextrema()
    # max_diff = max([ex[1] for ex in extrama])
    # if max_diff == 0:
    #     max_diff = 1
    # scale = 255.0 / max_diff
    # ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

if __name__ == "__main__":
    # Small test
    # Create a dummy image
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    ela = compute_ela(dummy)
    print("ELA image size:", ela.size)
