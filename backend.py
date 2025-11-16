from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np

app = FastAPI()

# ----------------------------
# Load YOLO Model Once
# ----------------------------
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"

with open(NAMES, "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]

net = cv2.dnn.readNet(WEIGHTS, CFG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # Convert bytes -> numpy
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detections": [], "error": "decode_failed"}

    h, w = frame.shape[:2]

    # YOLO blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detections = []

    for out in outs:
        for det in out:
            scores = det[5:]
            cls = np.argmax(scores)
            conf = float(scores[cls])
            if conf < 0.3:
                continue
            cx, cy, bw, bh = det[:4] * np.array([w, h, w, h])
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": conf,
                "class_id": int(cls)
            })

    return {"detections": detections}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
