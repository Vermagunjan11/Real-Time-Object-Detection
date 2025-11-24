import cv2

# Load actual class names
with open("coco.names", "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]

def draw_detections(frame, detections):
    img = frame.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cls_id = det["class_id"]

        # Get class name safely
        if 0 <= cls_id < len(CLASSES):
            class_name = CLASSES[cls_id]
        else:
            class_name = "unknown"

        label = f"{class_name} {score*100:.1f}%"

        # Clamp bounding box
        x1 = max(0, min(w-1, int(x1)))
        y1 = max(0, min(h-1, int(y1)))
        x2 = max(0, min(w-1, int(x2)))
        y2 = max(0, min(h-1, int(y2)))

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    return img
