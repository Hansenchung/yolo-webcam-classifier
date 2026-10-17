# Real-Time YOLO Webcam Classifier & Detector

A lightweight, real-time computer vision project built with **Python**, **OpenCV**, and **Ultralytics YOLOv8**. It captures live video feeds from your webcam, processes each frame horizontally flipped, and runs high-speed object or gesture detection out-of-the-box.

---

## Features

* **Real-Time Inference:** Leverages Ultralytics YOLOv8 for fast and efficient frame processing.
* **Webcam Integration:** Automatically hooks into your default webcam device (`cv2.VideoCapture(0)`).
* **Automatic Bounding Boxes:** Utilizes Ultralytics' built-in `.plot()` method to seamlessly render bounding boxes, confidence scores, and class labels.
* **Customizable Weights:** Easily switch between pre-trained baseline models (e.g., `yolov8n.pt`) and custom-trained weights (such as hand gesture or finger-counting models from Roboflow Universe).

---

## Project Structure

```text
├── webcam_classifier.py    # Main script to run webcam capture and YOLO inference
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
