import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# CONFIGURATION 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "MODEL_PATH": os.path.join(BASE_DIR, "models", "best_model.pth"),
    "TEST_DIR":   os.path.join(BASE_DIR, "data", "balanced_dataset", "Testing"),
    "BATCH_SIZE": 32,
    "IMG_SIZE":   (224, 224),
    "NUM_CLASSES": 4,
    "CLASS_NAMES": ["glioma", "meningioma", "pituitary", "notumor"]
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_to_rgb(x):
    return x.convert("RGB")

# DATA PREP 
def get_transforms(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Lambda(convert_to_rgb),  # ✅ FIXED
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_test_dataloader(data_dir, transform, batch_size=32):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print(f"Loaded test set: {len(dataset)} samples")
    print(f"Detected classes: {dataset.classes}")
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    ), dataset.classes

# MODEL LOADING
def load_model(weights_path, num_classes=4, device="cpu"):
    model = timm.create_model(
        'tf_efficientnetv2_s',
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.4,
        drop_path_rate=0.1
    )
    
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

# INFERENCE 
def run_inference(model, dataloader, device):
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    return np.concatenate(all_labels), np.concatenate(all_preds)

#  METRICS & PLOTTING 
def compute_metrics(true_labels, pred_labels, class_names):
    acc = accuracy_score(true_labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
    return acc, prec, rec, f1, cm, report

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'shrink': 0.8}
    )
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Brain Tumor Classification Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
def plot_class_metrics(true_labels, pred_labels, class_names, save_path="class_metrics.png"):
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )

    x = np.arange(len(class_names))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, precision, width=0.2, label='Precision')
    plt.bar(x, recall, width=0.2, label='Recall')
    plt.bar(x + 0.2, f1, width=0.2, label='F1-score')

    plt.xticks(x, class_names, rotation=30)
    plt.ylim(0, 1.05)
    plt.title("Class-wise Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Class metrics graph saved to: {save_path}")

def plot_prediction_distribution(true_labels, pred_labels, class_names, save_path="prediction_distribution.png"):
    true_counts = np.bincount(true_labels)
    pred_counts = np.bincount(pred_labels)

    x = np.arange(len(class_names))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, true_counts, width=0.4, label='True')
    plt.bar(x + 0.2, pred_counts, width=0.4, label='Predicted')

    plt.xticks(x, class_names, rotation=30)
    plt.title("True vs Predicted Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Prediction distribution saved to: {save_path}")
    
def plot_accuracy_pie(acc, save_path="accuracy_pie.png"):
    labels = ['Correct', 'Incorrect']
    values = [acc, 1 - acc]

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.2f%%')
    plt.title("Model Accuracy Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Accuracy pie chart saved to: {save_path}")
    

# MAIN EXECUTION 
def main():
    
    print("BRAIN TUMOR MODEL EVALUATION")
    
    transform = get_transforms(CONFIG["IMG_SIZE"])
    test_loader, detected_classes = create_test_dataloader(CONFIG["TEST_DIR"], transform, CONFIG["BATCH_SIZE"])
    
    if detected_classes != CONFIG["CLASS_NAMES"]:
        print(f"Warning: Test folder classes {detected_classes} differ from expected {CONFIG['CLASS_NAMES']}.")
        CONFIG["CLASS_NAMES"] = detected_classes
        
    model = load_model(CONFIG["MODEL_PATH"], CONFIG["NUM_CLASSES"], DEVICE)
    print(f"Inference device: {DEVICE}")
    
    print("\nRunning batch-wise inference...")
    true_labels, pred_labels = run_inference(model, test_loader, DEVICE)
    
    acc, prec, rec, f1, cm, report = compute_metrics(true_labels, pred_labels, CONFIG["CLASS_NAMES"])
    
    print("EVALUATION RESULTS")
    
    print(f"Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall:    {rec:.4f}")
    print(f"Weighted F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(cm, CONFIG["CLASS_NAMES"])
    
    plot_class_metrics(true_labels, pred_labels, CONFIG["CLASS_NAMES"])
    plot_prediction_distribution(true_labels, pred_labels, CONFIG["CLASS_NAMES"])
    plot_accuracy_pie(acc)

if __name__ == "__main__":
    main()