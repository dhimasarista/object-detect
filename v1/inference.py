import torch
import cv2
import torchvision
from torchvision import transforms
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("models/metal_cls_v2.pt", map_location=DEVICE)
CLASSES = ckpt["classes"]

model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, len(CLASSES))
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def classify(img):
    print("=== classify() start ===")
    
    # Transform image
    x = transform(img).unsqueeze(0).to(DEVICE)
    print(f"Input shape (after transform & unsqueeze): {x.shape}")
    print(f"Input dtype: {x.dtype}, device: {x.device}")

    with torch.no_grad():
        logits = model(x)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits sample: {logits[0][:5]}")  # tampil sebagian

        probs = torch.softmax(logits, dim=1)[0]
        print(f"Probabilities shape: {probs.shape}")
        print(f"Probabilities sample: {probs[:5]}")  # tampil sebagian
        print(f"Sum of probabilities (should be 1.0): {probs.sum().item():.4f}")

    idx = probs.argmax()  # Tensor scalar
    print(f"Predicted class index: {idx.item()}")
    print(f"Predicted class name: {CLASSES[idx.item()]}")
    print(f"Predicted probability: {probs[idx].item():.4f}")
    
    print("=== classify() end ===\n")
    return CLASSES[idx.item()], probs[idx].item()
