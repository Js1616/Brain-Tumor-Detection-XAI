import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np 
import pandas as pd 
# Dataset paths
source_dataset = r"C:\Users\js731\Downloads\Brain-Tumor-Detection-XAI\data\contrast_dataset"
resized_dataset = r"C:\Users\js731\Downloads\Brain-Tumor-Detection-XAI\data\brain_tumor_dataset_resized"

# Dataset structure
splits = ["Training", "Validation", "Testing"]
classes = ["notumor", "glioma", "pituitary", "meningioma"]

# Target size
target_size = (224, 224)

# Create directory structure for resized images
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(resized_dataset, split, cls), exist_ok=True)

# Counter for statistics
total_images = 0
resized_count = 0
error_count = 0

print("Starting image resizing to 224×224...")

# Process all images
for split in splits:
    for cls in classes:
        source_folder = os.path.join(source_dataset, split, cls)
        dest_folder = os.path.join(resized_dataset, split, cls)
        
        # Get list of image files
        image_files = [f for f in os.listdir(source_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nProcessing: {split}/{cls} ({len(image_files)} images)")
        
        for img_file in tqdm(image_files, desc=f"  {split}/{cls}"):
            source_path = os.path.join(source_folder, img_file)
            dest_path = os.path.join(dest_folder, img_file)
            
            try:
                # Open image
                with Image.open(source_path) as img:
                    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image using LANCZOS resampling (high quality)
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Save resized image
                    img_resized.save(dest_path, quality=95)
                    
                    resized_count += 1
                    
            except Exception as e:
                print(f"  Error processing {img_file}: {str(e)}")
                error_count += 1
            
            total_images += 1

# Print summary

print("RESIZING COMPLETE!")

print(f"Total images processed: {total_images}")
print(f"Successfully resized: {resized_count}")
print(f"Errors: {error_count}")
print(f"Output location: {resized_dataset}")


# Verify a few images
print("\nVerifying resized images...")
for split in splits[:1]:  # Check first split only
    for cls in classes[:1]:  # Check first class only
        folder = os.path.join(resized_dataset, split, cls)
        if os.path.exists(folder):
            sample_imgs = os.listdir(folder)[:3]
            if sample_imgs:
                img_path = os.path.join(folder, sample_imgs[0])
                with Image.open(img_path) as img:
                    print(f"Sample: {split}/{cls}/{sample_imgs[0]} -> Size: {img.size}")
                    


