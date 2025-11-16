import requests

BACKEND_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload_image"

def send_jpeg_bytes(jpeg_bytes: bytes, timeout: float = 5.0):
    files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
    try:
        r = requests.post(UPLOAD_ENDPOINT, files=files, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}

    if r.status_code != 200:
        return {"error": f"backend error {r.status_code}"}

    try:
        data = r.json()
    except:
        return {"error": "invalid JSON from backend"}

    return {"detections": data.get("detections", [])}
