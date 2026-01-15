import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
CAMERA_ID = 0
CONF_THRESH = 0.35
IMG_SIZE = 640

# COCO class IDs
# 0 = person (muka kandidat)
# 39 = bottle
# 41 = cup
# 44 = spoon
VALID_CLASSES = {
    0: "person",
    39: "bottle",
    41: "cup",
    44: "spoon"
}

# =========================
# LOAD YOLO
# =========================
model = YOLO("yolov8n.pt")

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Camera tidak bisa dibuka")

print("[INFO] YOLO webcam started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLO inference
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VALID_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = VALID_CLASSES[cls_id]
            conf = box.conf[0].item()

            # Filter ukuran box (hindari noise)
            w, h = x2 - x1, y2 - y1
            if w * h < 2000:
                continue

            color = (0, 255, 0) if cls_id != 0 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf*100:.1f}%",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("YOLO Webcam (Detector Only)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
