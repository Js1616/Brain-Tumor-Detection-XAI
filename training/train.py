import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

print("CUDA Available:", torch.cuda.is_available())

#  MAIN FUNCTION (IMPORTANT for Windows) 
def main():

    # PATH 
    data_dir = r"C:\Users\js731\OneDrive\Desktop\Brain-Tumor-Detection-XAI\src\data\balanced_dataset"

    # TRANSFORM 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    #  DATASET 
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/Validation", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Testing", transform=transform)

    #  DATALOADER 
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # DEVICE 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL (ResNet18 )
    model = models.resnet18(pretrained=True)

    # Replace last layer
    model.fc = nn.Linear(model.fc.in_features, 4)

    model = model.to(device)

    #  LOSS & OPTIMIZER 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #  EVALUATE 
    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    # TRAINING 
    epochs = 10  # increased

    print(" Training started...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            if batch_idx == 0:
                print(f"Epoch {epoch+1}: First batch loaded")

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print(" Training completed!")

# RUN 
if __name__ == "__main__":
    main()