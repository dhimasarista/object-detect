import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

os.makedirs("models", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

DATA_DIR = "/content/data/processed"
BATCH = 16
EPOCHS = 5

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
])

dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0
)


model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1280, len(dataset.classes))
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Optional: speed up for demo
# model.features.requires_grad_(False)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

    for x, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} finished | avg loss = {avg_loss:.4f}")

torch.save({
    "model": model.state_dict(),
    "classes": dataset.classes
}, "models/metal_cls.pt")

print("Training done & model saved")
