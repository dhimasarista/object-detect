import cv2
from inference import classify

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,th = cv2.threshold(
        blur,0,255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    return th

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    thresh = preprocess(frame)
    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue

        x,y,w,h = cv2.boundingRect(c)
        obj = frame[y:y+h, x:x+w]

        if obj.size == 0:
            continue

        label, conf = classify(obj)

        text = f"{label} {conf*100:.1f}%"
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,text,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,0),2)

    cv2.imshow("Metal Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
