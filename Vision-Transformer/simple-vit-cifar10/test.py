import torch

from simple_vit import SimpleViT
from utils import get_cifar10_loaders


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, test_loader = get_cifar10_loaders(batch_size=32)

    model = SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    ).to(device)

    model.load_state_dict(
        torch.load("simple_vit_cifar10.pth", map_location=device)
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    test()