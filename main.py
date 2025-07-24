import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re

# ========= 1. Hàm tiền xử lý biển số =========
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpen = cv2.filter2D(enhanced, -1, sharpen_kernel)
    if sharpen.shape[1] < 200:
        sharpen = cv2.resize(sharpen, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return sharpen

# ========= 2. Hàm hậu xử lý văn bản OCR =========
def clean_ocr_text(text):
    text = re.sub(r'[^A-Z0-9\-]', '', text.upper())  # Giữ lại A-Z, 0-9, dấu gạch
    return text

# ========= 3. Khởi tạo model YOLO + EasyOCR =========
yolo_model = YOLO("../model_detect_License_Plate/runs/detect/train/weights/license_plate_detector_yolov8.pt")  # Đường dẫn model YOLO
ocr_reader = easyocr.Reader(['en'], gpu=True)

# ========= 4. Đọc video =========
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)

output_path = "output_video.avi"
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]

        if label == 'license_plate' and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_region = frame[y1:y2, x1:x2]
            processed = preprocess_plate(plate_region)

            ocr_result = ocr_reader.readtext(processed, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')

            text = ""
            if ocr_result:
                _, raw_text, prob = ocr_result[0]
                text = clean_ocr_text(raw_text)
                print(f"[Frame {frame_idx}] Biển số: {text} (Conf: {prob:.2f})")

            # Vẽ box và biển số lên frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if text:
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"==>>> Video kết quả đã lưu vào: {output_path}")