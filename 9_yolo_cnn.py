import cv2
from ultralytics import YOLO
from cnn_inference import classify_cnn

# =========================
# CONFIG
# =========================
CAMERA_ID = 0
YOLO_CONF = 0.35
CNN_CONF = 0.80
IMG_SIZE = 640

# COCO classes
VALID_CLASSES = {
    0: "person",   # muka kandidat
    39: "bottle",
    41: "cup",
    44: "spoon"
}

# =========================
# LOAD YOLO
# =========================
yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Camera tidak bisa dibuka")

print("[INFO] YOLO + CNN webcam started. ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = yolo(
        frame,
        imgsz=IMG_SIZE,
        conf=YOLO_CONF,
        verbose=False
    )

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VALID_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Filter ukuran box
            w, h = x2 - x1, y2 - y1
            if w * h < 2000:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # =========================
            # CNN CLASSIFICATION
            # =========================
            label, conf = classify_cnn(roi)
            if conf < CNN_CONF:
                continue

            color = (0, 255, 0) if label != "muka" else (0, 0, 255)

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

    cv2.imshow("YOLO + CNN Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
