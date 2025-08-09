# Project_License_Plate_Recognition_Yolov8_EasyOCR

## Introduction

This project uses the YOLOv8 model to detect license plates in videos and EasyOCR to recognize characters on the plates. The results are exported to a new video with the recognized license plate displayed directly on each frame.

### Demo

▶️ [Watch the demo video](https://drive.google.com/file/d/1ioxro5FmDVlINGWTFQ2yDBOaa36MJudf/view?usp=drive_link)

## System Requirements

- Python 3.x
- Libraries: OpenCV, NumPy, EasyOCR, ultralytics (YOLO)
- GPU (recommended for faster OCR processing)

Install required libraries:
```sh
pip install opencv-python numpy easyocr ultralytics
```

## Project Structure

- `main.py`: Main source code for license plate detection and recognition from video.
- `car_video.mp4`: Input video containing vehicles for license plate recognition.
- `output_video.avi`: Output video with recognized license plates.
- `license_plate_detector_yolov8.pt`: Pre-trained YOLOv8 weights for license plate detection.

## Usage Guide

1. Place the video to be processed in the same folder as `main.py` and rename it to `car_video.mp4` (or update the `video_path` variable in the source code).
2. Make sure the YOLOv8 weights file is available at `license_plate_detector_yolov8.pt`.
3. Run the program:
    ```sh
    python main.py
    ```
4. The result video will be saved as `output_video.avi`.

## Workflow

- Read each frame from the video.
- Use YOLOv8 to detect license plate regions.
- Preprocess the license plate region (grayscale, enhance, sharpen, resize if needed).
- Recognize characters using EasyOCR, keeping only valid characters (A-Z, 0-9, hyphen).
- Draw bounding boxes and display the license plate text on the frame.
- Save the processed video.

## Notes

- The YOLOv8 model must be pre-trained with a suitable license plate dataset.
- EasyOCR recognizes English letters and numbers; you can customize the language if needed.
- Make sure your GPU has the appropriate drivers and CUDA installed if using EasyOCR with GPU.

## Contact

For questions, please contact via email: nguyenphuongv07@gmail.com.
