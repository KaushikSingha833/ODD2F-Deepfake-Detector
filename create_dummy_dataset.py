import os
import numpy as np
from PIL import Image

def create_dataset(base_dir, num_train=50, num_test=10):
    for split in ['train', 'test']:
        for label in ['REAL', 'FAKE']:
            os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)
            
            num_images = num_train if split == 'train' else num_test
            for i in range(num_images):
                # Generate random noise image
                # To make it slightly distinguishable for the model:
                # REAL gets slightly more green, FAKE gets slightly more red
                img_array = np.random.randint(0, 200, (224, 224, 3), dtype=np.uint8)
                if label == 'REAL':
                    img_array[:, :, 1] += 50 # Add green
                else:
                    img_array[:, :, 0] += 50 # Add red
                    
                img_array = np.clip(img_array, 0, 255)
                img = Image.fromarray(img_array)
                img.save(os.path.join(base_dir, split, label, f"img_{i}.jpg"))

if __name__ == "__main__":
    create_dataset("dummy_dataset")
    print("Dummy dataset created successfully at 'dummy_dataset'!")
