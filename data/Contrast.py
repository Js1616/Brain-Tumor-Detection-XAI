import os
import cv2
from tqdm import tqdm

#  PATHS 
input_dataset = r"C:\Users\js731\Downloads\Brain-Tumor-Detection-XAI\data\brain_tumor_dataset_final"
output_dataset = r"C:\Users\js731\Downloads\Brain-Tumor-Detection-XAI\data\contrast_dataset"

classes = ["notumor", "glioma", "pituitary", "meningioma"]
splits = ["Training", "Validation", "Testing"]

#  CLAHE FUNCTION 
def apply_clahe(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading {image_path}")
        return None

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge and convert back
    lab = cv2.merge((l_clahe, a, b))
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced_img


#  MAIN PROCESS 
def process_dataset():
    print("\n APPLYING CLAHE TO DATASET ")

    total_before = 0
    total_after = 0

    for split in splits:
        print(f"\nProcessing {split}...")

        for cls in classes:
            src_folder = os.path.join(input_dataset, split, cls)
            dst_folder = os.path.join(output_dataset, split, cls)

            os.makedirs(dst_folder, exist_ok=True)

            images = [f for f in os.listdir(src_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"{cls}: {len(images)} images")

            total_before += len(images)

            for img in tqdm(images, desc=f"{split}-{cls}"):
                src_path = os.path.join(src_folder, img)
                dst_path = os.path.join(dst_folder, img)

                enhanced = apply_clahe(src_path)

                if enhanced is not None:
                    cv2.imwrite(dst_path, enhanced)
                    total_after += 1

    
    print(" SUMMARY")

    print(f"Total Images Before: {total_before}")
    print(f"Total Images After : {total_after}")

    if total_before == total_after:
        print(" All images processed successfully")
    else:
        print(" Some images were skipped")


#  RUN 
if __name__ == "__main__":
    process_dataset()