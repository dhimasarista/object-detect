import cv2
from inference import classify  

def preprocess(frame):
    """Convert frame to grayscale, blur, and threshold"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    return gray, thresh


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue  # skip frame kalau gagal baca

    gray, thresh = preprocess(frame)

    # Pastikan versi OpenCV kompatibel
    contours_info = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500 or area > 30000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # ---- FILTER BENTUK (anti wajah / non-metal) ----
        aspect = w / float(h)
        if aspect < 0.3 or aspect > 3.0:
            continue

        # ---- FILTER BRIGHTNESS ----
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        mean_intensity = roi_gray.mean()
        if mean_intensity > 190:
            continue

        # ---- INFERENCE ----
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        try:
            label, conf = classify(roi)
        except Exception as e:
            print(f"Skipping ROI due to inference error: {e}")
            continue

        # ---- FILTER CONFIDENCE ----
        if conf < 0.85:
            continue

        # ---- DRAW RESULTS ----
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf*100:.1f}%",
            (x, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Metal Detection (CV + ML)", frame)
    cv2.imshow("Threshold", thresh)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
