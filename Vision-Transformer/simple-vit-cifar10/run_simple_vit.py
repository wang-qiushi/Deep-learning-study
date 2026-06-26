import torch
from simple_vit import SimpleViT

model = SimpleViT(
    image_size=224,
    patch_size=16,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024
)

img = torch.randn(1, 3, 224, 224)
out = model(img)

print("input shape:", img.shape)
print("output shape:", out.shape)