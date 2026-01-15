import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report

from utils import eval_transform_pil
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ckpt = torch.load("models/metal_classifier.pt", map_location=DEVICE)
    classes = ckpt["classes"]

    test_ds = ImageFolder("dataset/test", transform=eval_transform_pil())
    test_dl = DataLoader(test_ds, batch_size=16)

    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    main()
