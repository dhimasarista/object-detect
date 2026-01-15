import torch
import cv2
from model import build_model
from utils import eval_transform_cv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "models/metal_classifier.pt"

# =========================
# LOAD CNN MODEL
# =========================
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
CLASSES = ckpt["classes"]

cnn = build_model(num_classes=len(CLASSES))
cnn.load_state_dict(ckpt["model"])
cnn.to(DEVICE)
cnn.eval()

transform = eval_transform_cv()

def classify_cnn(roi_bgr):
    """
    Input  : ROI BGR (OpenCV)
    Output : label (str), confidence (float)
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    x = transform(roi_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(cnn(x), dim=1)[0]

    idx = probs.argmax().item()
    return CLASSES[idx], probs[idx].item()
