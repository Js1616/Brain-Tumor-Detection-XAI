import os
import shutil
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# PATHS
source_dataset = r"C:\Users\js731\Downloads\Brain-Tumor-Detection-XAI\data\brain_tumor_dataset_resized"
balanced_dataset = r"C:\Users\js731\Downloads\Brain-Tumor-Detection-XAI\data\balanced_dataset"

classes = ["notumor", "glioma", "pituitary", "meningioma"]
splits = ["Training", "Validation", "Testing"]

# AUGMENTATION 
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
])

# FUNCTIONS 
def get_images(folder):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    return [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]


def copy_image(src, dst):
    shutil.copy2(src, dst)


def augment_image(src_path, dst_path):
    image = Image.open(src_path).convert("RGB")
    aug = transform(image)
    aug.save(dst_path)


# MAIN
for split in splits:
    print(f"\n PROCESSING {split} ")

    # Step 1: Count all classes
    class_counts = {}
    for cls in classes:
        folder = os.path.join(source_dataset, split, cls)
        class_counts[cls] = len(get_images(folder))
        print(f"{cls}: {class_counts[cls]}")

    # Step 2: Find max
    max_count = max(class_counts.values())
    print(f"Target (max count): {max_count}")

    # Step 3: Create folders
    for cls in classes:
        os.makedirs(os.path.join(balanced_dataset, split, cls), exist_ok=True)

    # Step 4: Process each class
    for cls in classes:
        src_folder = os.path.join(source_dataset, split, cls)
        dst_folder = os.path.join(balanced_dataset, split, cls)

        images = get_images(src_folder)
        current_count = len(images)

        print(f"\nProcessing {cls} ({current_count})")

        #  Copy all original images
        for img in tqdm(images, desc=f"{cls} copying"):
            copy_image(
                os.path.join(src_folder, img),
                os.path.join(dst_folder, img)
            )

        #  Augment if needed
        if current_count < max_count:
            needed = max_count - current_count
            print(f"Augmenting {needed} images...")

            for i in tqdm(range(needed), desc=f"{cls} augmenting"):
                img_name = images[i % len(images)]
                src_path = os.path.join(src_folder, img_name)

                new_name = f"{cls}_aug_{i:04d}.jpg"
                dst_path = os.path.join(dst_folder, new_name)

                augment_image(src_path, dst_path)

        else:
            print(f"{cls} already at max, no augmentation")

before_counts = {}

for split in splits:
    before_counts[split] = {}
    for cls in classes:
        folder = os.path.join(source_dataset, split, cls)
        before_counts[split][cls] = len(get_images(folder))


after_counts = {}

for split in splits:
    after_counts[split] = {}
    for cls in classes:
        folder = os.path.join(balanced_dataset, split, cls)
        after_counts[split][cls] = len(get_images(folder))

# VERIFY 
print("\n FINAL DISTRIBUTION ")

for split in splits:
    print(f"\n{split}:")
    for cls in classes:
        folder = os.path.join(balanced_dataset, split, cls)
        count = len(get_images(folder))
        print(f"{cls}: {count}")
        
        
print("\n" + "="*60)
print("DATASET SUMMARY (BEFORE vs AFTER)")
print("="*60)

for split in splits:
    print(f"\n🔹 {split}:")

    before_total = sum(before_counts[split].values())
    after_total = sum(after_counts[split].values())

    print("\nClass-wise:")
    for cls in classes:
        b = before_counts[split][cls]
        a = after_counts[split][cls]
        print(f"{cls:12s} | Before: {b:5d} → After: {a:5d}")

    print(f"\nTotal Images:")
    print(f"Before: {before_total}")
    print(f"After : {after_total}")

    #  Balance Check
    unique_counts = set(after_counts[split].values())
    if len(unique_counts) == 1:
        print(" Dataset is BALANCED")
    else:
        print(" Dataset is NOT balanced")        