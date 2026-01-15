from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import train_transform_pil, eval_transform_pil

train_ds = ImageFolder("dataset/train", transform=train_transform_pil())
val_ds   = ImageFolder("dataset/val", transform=eval_transform_pil())

print("Classes:", train_ds.classes)

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

x, y = next(iter(train_dl))
print("Batch shape:", x.shape)
print("Labels:", y)
