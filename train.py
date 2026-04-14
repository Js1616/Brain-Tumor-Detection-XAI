import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

print("CUDA Available:", torch.cuda.is_available())

def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "backend", "data", "balanced_dataset")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/Validation", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Testing", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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

    epochs = 10
    print("Training started...")

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

    print("Training completed!")

if __name__ == "__main__":
    main()