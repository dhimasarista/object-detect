import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os

from utils import train_transform_pil, eval_transform_pil
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    train_ds = ImageFolder("dataset/train", transform=train_transform_pil())
    val_ds   = ImageFolder("dataset/val", transform=eval_transform_pil())

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model(num_classes=len(train_ds.classes))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=LR
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1} | avg loss = {avg_loss:.4f}")

    # Save model
    torch.save({
        "model": model.state_dict(),
        "classes": train_ds.classes
    }, os.path.join(MODEL_DIR, "metal_classifier.pt"))

    print("Training selesai. Model disimpan.")

if __name__ == "__main__":
    main()
