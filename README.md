# License Plate Recognition with Vision Transformers (ViT)

This project implements an end-to-end license plate detection and recognition system for traffic monitoring, developed for the **Traffic Modeling, Simulation and Control** course.

## 🚀 Overview
The system leverages state-of-the-art **Vision Transformer (ViT)** architectures to identify vehicles and log their traversal through a video stream.

- **Detection:** Custom-trained **RF-DETR** (Detection Transformer).
- **Recognition:** **TrOCR** (Transformer-based OCR) from Hugging Face.
- **Tracking:** Temporal smoothing with IOU-based tracking and majority-vote OCR stabilization.

## ✨ Key Features
- **High Accuracy:** Custom-trained detection model achieved **98.7% mAP50**.
- **Stable Logging:** Records vehicle IDs, license plate numbers, and traversal time ranges (First Seen -> Last Seen) to `traversed_vehicles.csv`.
- **Flicker Reduction:** Majority voting logic ensures stable character recognition across consecutive frames.
- **GPU Accelerated:** Optimized for CUDA-enabled devices.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/erik-learning/license-plate-recognition.git
cd license-plate-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

To run the recognition pipeline on a video file:
```bash
python main.py --input path/to/video.mp4 --output result.mp4
```

The system will:
1. Process the video using the custom RF-DETR model.
2. Annotate bounding boxes and recognized plates in real-time.
3. Save the stabilized traffic log to `traversed_vehicles_v3.csv`.

## 📁 Repository Structure
- `main.py`: The primary inference pipeline.
- `weights/best.pt`: Custom-trained RF-DETR model weights.
- `train_rfdetr.py`: Script used for fine-tuning the detector.
- `prepare_data.py`: Dataset conversion and preparation logic.
- `requirements.txt`: Necessary Python packages.

## 🎓 Course Details
- **Course:** Traffic Modeling, Simulation and Control
- **Topic:** #31 License Plate Recognition with Vision Transformers
