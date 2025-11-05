Real-Time Object Detection using YOLOv4-Tiny (CPU)

This project performs real-time object detection from a webcam feed using the YOLOv4-Tiny model and OpenCV DNN — optimized to run smoothly on CPU (no GPU required).

🚀 Features

🎥 Real-time detection using your laptop webcam

💻 Runs entirely on CPU — no GPU needed

🧾 Detects 80 COCO classes (person, car, cell phone, etc.)

🕒 Live FPS counter and timestamp overlay

🔢 Object count display per frame

💾 (Optional) Detection logging to file

⚙️ Easy to extend for Flask, Streamlit, or GUI applications

🧩 Project Structure
Object_Detection/
│
├── demo_yolov4_tiny_cpu.py      # Main Python script (runs detection)
├── yolov4-tiny.cfg              # YOLOv4-Tiny network configuration
├── yolov4-tiny.weights          # Pretrained model weights (~23 MB)
├── coco.names                   # 80 COCO class labels
├── detections_log.txt           # (Optional) Log of detections
└── README.md                    # Project documentation

🛠️ Requirements

Install dependencies before running the project:

pip install opencv-python numpy
