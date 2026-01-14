import cv2
import os
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

MIN_AREA = 800        # filter noise kecil
IMG_SIZE = 224        # resize output
DEBUG = False         # True = tampilkan bbox
AUGMENT = True        # True = random flip / brightness

# =========================
# UTILS
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def preprocess(img):
    """Grayscale + CLAHE + blur + Otsu threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def resize_and_pad(img, size=224):
    """Resize image to fit in square, keep aspect ratio, pad with black"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img, (nw, nh))

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    return img_padded

def augment(img):
    """Random flip & brightness"""
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    factor = 0.8 + np.random.rand() * 0.4  # brightness 0.8-1.2
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img

def extract_objects(img, thresh):
    """Extract objects based on contours and filter rules"""
    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        rect_area = w*h
        solidity = float(area) / rect_area
        aspect_ratio = w / float(h)

        if solidity < 0.4:
            continue
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue

        obj = img[y:y+h, x:x+w]
        if obj.size == 0:
            continue

        obj = resize_and_pad(obj, IMG_SIZE)
        if AUGMENT:
            obj = augment(obj)

        objects.append((obj, (x, y, w, h)))

    return objects

# =========================
# MAIN PIPELINE
# =========================
def process_folder(src_dir: Path, dst_dir: Path):
    ensure_dir(dst_dir)

    for img_path in src_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        thresh = preprocess(img)
        objects = extract_objects(img, thresh)

        print(f"[INFO] {img_path.name}: extracted {len(objects)} objects")

        for i, (obj, bbox) in enumerate(objects):
            out_name = f"{img_path.stem}_{i}.jpg"
            out_path = dst_dir / out_name
            cv2.imwrite(str(out_path), obj)

            if DEBUG:
                x, y, w, h = bbox
                dbg = img.copy()
                cv2.rectangle(dbg, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("debug", dbg)
                cv2.waitKey(0)

def walk_dataset(src_root: Path, dst_root: Path):
    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        if not files:
            continue

        rel = root_path.relative_to(src_root)
        out_path = dst_root / rel

        print(f"[INFO] Processing folder: {rel}")
        process_folder(root_path, out_path)

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    walk_dataset(RAW_DIR, OUT_DIR)
    print("✅ Dataset preparation finished")
