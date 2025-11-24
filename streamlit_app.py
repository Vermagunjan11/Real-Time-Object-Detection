import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import io

from web_client import send_jpeg_bytes, set_backend_url
from visual import draw_detections

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration,
    WebRtcMode,
)

# RTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Real-Time Object Detection", layout="centered")

st.title("🔍 Real-Time Object Detection — Streamlit Frontend")

st.markdown(
    """
### Choose mode:
- **Snapshot:** capture a single image  
- **Live:** webcam real-time detection  
"""
)

mode = st.radio("Mode", ["Snapshot", "Live"])

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    # Backend URL (IMPORTANT)
    backend_url = st.text_input("Backend URL", "http://localhost:8000")
    st.caption("Make sure backend is running on this URL.")

    # Send the URL to web_client.py
    set_backend_url(backend_url)

    conf_threshold = st.slider("Confidence Threshold (%)", 0, 100, 30)
    st.write("Frames below this confidence will be filtered out.")


# ----------------------------------------------------------------
# SNAPSHOT MODE
# ----------------------------------------------------------------
if mode == "Snapshot":
    st.subheader("📸 Snapshot Mode")
    img_file = st.camera_input("Take a picture")

    if img_file is not None:
        st.image(img_file, caption="Captured image", use_container_width=True)

        bytes_data = img_file.getvalue()

        with st.spinner("Sending to backend..."):
            try:
                result = send_jpeg_bytes(bytes_data) or {}
            except Exception as e:
                st.error(f"Backend error: {e}")
                result = {}

        detections = result.get("detections", [])

        # Convert bytes to image
        try:
            pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            frame = np.array(pil_image)[:, :, ::-1]  # RGB → BGR
        except Exception as e:
            st.error(f"Failed to read captured image: {e}")
            frame = None

        if frame is not None:
            # Apply confidence filter
            detections = [d for d in detections if d.get("score", 0) * 100 >= conf_threshold]

            # Draw detections
            try:
                annotated = draw_detections(frame, detections)
                annotated = annotated[:, :, ::-1]  # BGR → RGB
                st.image(annotated, caption="Detections", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to draw detections: {e}")

            st.json({"detections": detections})


# ----------------------------------------------------------------
# LIVE MODE (webcam)
# ----------------------------------------------------------------
else:
    st.subheader("🎥 Live Webcam Detection")
    st.write("Click **Start** to begin streaming.")

    class LiveProcessor(VideoTransformerBase):
        def __init__(self):
            self.last = time.time()
            self.fps = 0.0

        def recv(self, frame):
            # frame: av.VideoFrame -> convert to ndarray BGR
            img = frame.to_ndarray(format="bgr24")

            # Encode image to JPEG bytes to send to backend
            ret, jpg = cv2.imencode(".jpg", img)
            if not ret:
                # If encoding failed, just return original image
                return img

            try:
                res = send_jpeg_bytes(jpg.tobytes()) or {}
            except Exception:
                # On backend failure, continue without detections
                res = {}

            # Extract detections safely
            dets = res.get("detections", [])
            if not isinstance(dets, list):
                dets = []

            # Apply confidence filter
            dets = [d for d in dets if d.get("score", 0) * 100 >= conf_threshold]

            # Draw boxes (visual.draw_detections should expect BGR ndarray)
            try:
                img = draw_detections(img, dets)
            except Exception:
                # If draw fails, keep original image
                pass

            # FPS calculation (smoothed)
            now = time.time()
            dt = now - self.last
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * instant_fps
            self.last = now

            cv2.putText(
                img,
                f"FPS: {self.fps:.1f}",
                (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Return ndarray BGR
            return img

    # Start WebRTC - use enum for mode (not a string) and correct argument name
    ctx = webrtc_streamer(
        key="object-detect",
        mode=WebRtcMode.SENDRECV,  # <--- use enum, not "sendrecv" string
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=LiveProcessor,  # correct factory name expected by streamlit-webrtc
        async_processing=True,
    )
