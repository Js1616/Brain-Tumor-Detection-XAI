import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  

try:
    from torch.amp import autocast, GradScaler
    AMP_NEW = True
except:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW = False

import timm
import numpy as np

# CONFIGURATION 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "balanced_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
VAL_DIR   = os.path.join(DATA_DIR, "Validation")
TEST_DIR  = os.path.join(DATA_DIR, "Testing")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 50
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-4
DROP_RATE = 0.4
CLIP_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 10
USE_CLASS_WEIGHTS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def convert_to_rgb(x):
    return x.convert("RGB")

# GRAPH FUNCTION 
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    # Loss Curve
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    # Accuracy Curve
    plt.figure()
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()

    print("Training graphs saved!")

# DATA LOADING 
def get_data_loaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=data_transforms['val'])

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    if USE_CLASS_WEIGHTS:
        class_counts = np.zeros(NUM_CLASSES)
        for _, label in train_dataset:
            class_counts[label] += 1
        weights = 1.0 / class_counts
        weights = torch.FloatTensor(weights / weights.sum() * NUM_CLASSES).to(DEVICE)
    else:
        weights = None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, weights

# MODEL 
def build_model():
    model = timm.create_model(
        'tf_efficientnetv2_s',
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=DROP_RATE,
        drop_path_rate=0.1
    )
    return model.to(DEVICE)

# TRAIN 
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        context = autocast(device_type='cuda') if AMP_NEW and device.type == 'cuda' else autocast()

        with context:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    return running_loss / total, correct.double() / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            context = autocast(device_type='cuda') if AMP_NEW and device.type == 'cuda' else autocast()

            with context:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

    return running_loss / total, correct.double() / total

# MAIN 
def main():
    train_loader, val_loader, class_weights = get_data_loaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nStarting training on {DEVICE}...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc     = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved best model\n")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered")
                break

    if best_state_dict:
        model.load_state_dict(best_state_dict)

    _, final_val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")

    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    main()