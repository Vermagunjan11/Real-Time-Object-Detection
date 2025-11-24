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
            result = send_jpeg_bytes(bytes_data)

        detections = result.get("detections", [])

        # Convert bytes to image
        pil_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        frame = np.array(pil_image)[:, :, ::-1]  # RGB → BGR

        # Apply confidence filter
        detections = [d for d in detections if d["score"] * 100 >= conf_threshold]

        # Draw detections
        annotated = draw_detections(frame, detections)
        annotated = annotated[:, :, ::-1]  # BGR → RGB

        st.image(annotated, caption="Detections", use_container_width=True)

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
            self.fps = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Encode image to JPEG
            ret, jpg = cv2.imencode(".jpg", img)
            if not ret:
                return img

            # Send to backend
            res = send_jpeg_bytes(jpg.tobytes())

            # Extract detections
            dets = res.get("detections", [])
            dets = [d for d in dets if d["score"] * 100 >= conf_threshold]

            # Draw boxes
            img = draw_detections(img, dets)

            # FPS
            now = time.time()
            dt = now - self.last
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1 / dt)
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

            return img

    # Start WebRTC
    ctx = webrtc_streamer(
    key="object-detect",
    mode="sendrecv",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=BackendPoster,     # updated
    async_processing=True,                     # updated
    )

