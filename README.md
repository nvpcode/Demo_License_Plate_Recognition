# Project_License_Plate_Recognition

## Introduction

This project uses the YOLO model to detect license plates in videos and EasyOCR to recognize characters on the plates. The results are exported to a new video with the recognized license plate displayed directly on each frame.

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
- `../model_detect_License_Plate/runs/detect/train/weights/best.pt`: Pre-trained YOLO weights for license plate detection.

## Usage Guide

1. Place the video to be processed in the same folder as `main.py` and rename it to `car_video.mp4` (or update the `video_path` variable in the source code).
2. Make sure the YOLO weights file is available at `../model_detect_License_Plate/runs/detect/train/weights/best.pt`.
3. Run the program:
    ```sh
    python main.py
    ```
4. The result video will be saved as `output_video.avi`.

## Workflow

- Read each frame from the video.
- Use YOLO to detect license plate regions.
- Preprocess the license plate region (grayscale, enhance, sharpen, resize if needed).
- Recognize characters using EasyOCR, keeping only valid characters (A-Z, 0-9, hyphen).
- Draw bounding boxes and display the license plate text on the frame.
- Save the processed video.

## Notes

- The YOLO model must be pre-trained with a suitable license plate dataset.
- EasyOCR recognizes English letters and numbers; you can customize the language if needed.
- Make sure your GPU has the appropriate drivers and CUDA installed if using EasyOCR with GPU.

### Contact

For questions, please contact via email: nguyenphuongv07@gmail.com.
