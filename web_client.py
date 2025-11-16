import requests
import io

# Default backend URL (change if needed)
BACKEND_URL = "http://localhost:8000"

UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload_image"


def send_jpeg_bytes(jpeg_bytes: bytes, timeout: float = 5.0):
    """
    Send jpeg bytes to backend and return parsed detections.
    Returns dict: {"detections": [...], "status": "ok" } or {"error": "..."}
    """
    files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
    try:
        r = requests.post(UPLOAD_ENDPOINT, files=files, timeout=timeout)
    except Exception as e:
        return {"error": f"request failed: {e}"}

    if r.status_code != 200:
        return {"error": f"backend status {r.status_code}: {r.text}"}
    try:
        data = r.json()
    except Exception as e:
        return {"error": f"invalid json from backend: {e}"}
    # normalize
    return parse_response(data)


def parse_response(data: dict):
    """
    Expecting {"detections": [ { "bbox": [x1,y1,x2,y2], "score": float, "class_id": int }, ... ] }
    If backend returns another shape, modify here.
    """
    if "detections" not in data:
        return {"error": "no 'detections' in response", "raw": data}
    dets = []
    for d in data["detections"]:
        # try several possible keys/personalities
        if isinstance(d, dict):
            bbox = d.get("bbox") or d.get("box") or d.get("box_xyxy")
            score = d.get("score") or d.get("confidence") or d.get("conf")
            cls = d.get("class_id") or d.get("class") or d.get("label")
            # canonicalize
            if bbox is None:
                continue
            try:
                x1, y1, x2, y2 = [float(x) for x in bbox]
            except Exception:
                continue
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score) if score is not None else 1.0,
                "class_id": int(cls) if cls is not None else 0
            })
    return {"detections": dets}
