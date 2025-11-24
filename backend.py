# /mnt/data/backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import os
from typing import List

# Paths - make sure these files are in the same folder as backend.py or update paths
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"

# Check files exist early
for p in (CFG, WEIGHTS, NAMES):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

# Load classes
with open(NAMES, "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]

# Load network once
net = cv2.dnn.readNet(WEIGHTS, CFG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

app = FastAPI(title="YOLOv4-Tiny Backend")

def run_yolo_on_image(img: np.ndarray, conf_threshold=0.3, nms_thresh=0.4) -> List[dict]:
    """
    img: BGR image (numpy)
    returns: list of detections: { "bbox": [x1,y1,x2,y2], "score": float, "class_id": int }
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence >= conf_threshold:
                cx, cy, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype(float)
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)
                boxes.append([x1, y1, int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

    # convert to xyxy for response
    if len(boxes) == 0:
        return []

    try:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_thresh)
    except Exception:
        idxs = []

    dets = []
    # idxs may be a numpy array or list
    if hasattr(idxs, "flatten"):
        idxs = idxs.flatten().tolist()
    for i in idxs:
        if i < 0 or i >= len(boxes):
            continue
        x, y, bw, bh = boxes[i]
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(w - 1, x + bw))
        y2 = int(min(h - 1, y + bh))
        dets.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(confidences[i]),
            "class_id": int(class_ids[i])
        })
    return dets

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), conf: float = 0.3):
    """
    Accepts a multipart/form-data file field named 'file' (image/jpeg).
    Optional query param 'conf' sets confidence threshold (0..1).
    Returns: { "detections": [ {bbox, score, class_id}, ... ] }
    """
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")

    body = await file.read()
    nparr = np.frombuffer(body, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    dets = run_yolo_on_image(img, conf_threshold=conf, nms_thresh=0.4)
    return JSONResponse({"detections": dets})

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
