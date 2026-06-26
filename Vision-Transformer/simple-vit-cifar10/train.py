import torch
import torch.nn as nn
import torch.optim as optim

from simple_vit import SimpleViT
from utils import get_cifar10_loaders


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, _ = get_cifar10_loaders(batch_size=32)

    model = SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 5

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Train Acc: {train_acc:.2f}%"
        )

    torch.save(model.state_dict(), "simple_vit_cifar10.pth")
    print("Model saved as simple_vit_cifar10.pth")


if __name__ == "__main__":
    train()