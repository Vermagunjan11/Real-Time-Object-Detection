import requests

# dynamic backend URL (initially empty)
BACKEND_URL = None

def set_backend_url(url: str):
    global BACKEND_URL
    BACKEND_URL = url.rstrip("/")  # remove trailing slash

def send_jpeg_bytes(jpeg_bytes: bytes, timeout: float = 5.0):
    """
    Send JPEG to backend URL set from streamlit.
    """
    if BACKEND_URL is None:
        return {"error": "Backend URL not set"}

    upload_url = f"{BACKEND_URL}/upload_image"

    files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
    try:
        r = requests.post(upload_url, files=files, timeout=timeout)
    except Exception as e:
        return {"error": f"Request failed: {e}"}

    if r.status_code != 200:
        return {"error": f"Backend status {r.status_code}: {r.text}"}

    try:
        return r.json()
    except:
        return {"error": "Invalid JSON from backend"}
