import torchvision
from torch import nn

def build_model(num_classes: int):
    model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")

    # Freeze backbone (WAJIB untuk dataset kecil)
    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier[1] = nn.Linear(1280, num_classes)
    return model
