import cv2
import numpy as np
import time
from collections import Counter

# Load YOLOv4-Tiny Model
cfg_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"
names_path = "coco.names"

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet(weights_path, cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📷 Starting camera... Press 'q' to quit")

prev_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera read failed, skipping frame...")
        continue

    height, width, _ = frame.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detection lists
    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            if len(scores) == 0:
                continue
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x, center_y, w, h = (detection[0:4] *
                                            np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression safely
    try:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    except Exception as e:
        print("⚠️ NMSBoxes failed:", e)
        indexes = []

    detected_labels = []

    #Safe detection loop
    if indexes is not None and len(indexes) > 0:
        # Flatten index array safely
        if isinstance(indexes, np.ndarray):
            indexes = indexes.flatten()
        for i in indexes:
            if i < 0 or i >= len(boxes):  
                continue
            x, y, w, h = boxes[i]
            if class_ids[i] < len(classes):
                label = str(classes[class_ids[i]])
            else:
                label = "unknown"
            conf = confidences[i]
            color = (0, 255, 0)

            detected_labels.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                        font, 0.6, color, 2)

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                font, 0.7, (255, 255, 0), 2)

    # Object Count
    counts = Counter(detected_labels)
    y_offset = 50
    for label, count in counts.items():
        cv2.putText(frame, f"{label}: {count}",
                    (10, y_offset), font, 0.6, (0, 255, 255), 2)
        y_offset += 25

    # Show the frame
    cv2.imshow("YOLOv4-Tiny (CPU)", frame)

    # Safe exit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        print("🛑 Exiting...")
        break
    
cap.release()
cv2.destroyAllWindows()