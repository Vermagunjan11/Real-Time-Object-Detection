import cv2
from typing import List, Tuple

# Simple label map, user can change or load from a file
DEFAULT_LABELS = {0: "object"}

def draw_detections(frame, detections: List[dict], labels: dict = None):
    """
    frame: numpy BGR image (HxWx3)
    detections: list of { "bbox":[x1,y1,x2,y2], "score":float, "class_id":int }
    Returns annotated frame (copy).
    """
    if labels is None:
        labels = DEFAULT_LABELS
    out = frame.copy()
    h, w = out.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # ensure ints and clipped
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y2 = max(0, min(h - 1, int(round(y2))))
        score = det.get("score", 0.0)
        cls = int(det.get("class_id", 0))
        color = (36, 255, 12)  # green in BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        label = f"{labels.get(cls, str(cls))} {score*100:.1f}%"
        # put filled rectangle as background for text
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - 18 if y1 - 18 > 0 else y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 3 if y1 - 3 > 0 else y1 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
