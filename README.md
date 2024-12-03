# Object Detection using YOLO (v3)

This project demonstrates real-time object detection using YOLOv3 with OpenCV. The model detects objects in a live webcam feed.

## Usage
Before running, ensure that you have the following files:
- **yolov3.weights**
- **yolov3.cfg**
- **coco.names**

## How to Run

1. Clone the repository:
```bash
  git clone https://github.com/siddharth0607/yolo_v3_object_detection.git
```

2. Install the dependencies:
```bash
  pip install -r requirements.txt
```

3. Run the code:
```bash
  python main.py
```

## Notes
- Make sure the paths to the YOLO weights, configuration, and class labels in **main.py** are correct.
- Adjust the confidence threshold in the script as needed.
