import cv2

# Load COCO labels properly
with open("coco.names", "r") as f:
    LABELS = [l.strip() for l in f.readlines()]

def draw_detections(frame, detections):
    img = frame.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cls = det["class_id"]

        label = f"{LABELS[cls]} {score*100:.1f}%" if cls < len(LABELS) else f"ID:{cls}"

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return img
