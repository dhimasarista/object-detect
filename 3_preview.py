import cv2
import random
from pathlib import Path

BASE = Path("dataset/train")
CLASSES = ["metal", "plastik", "muka"]

for cls in CLASSES:
    imgs = list((BASE / cls).glob("*"))
    img_path = random.choice(imgs)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Gagal load {img_path}")
        continue

    cv2.imshow(cls, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
