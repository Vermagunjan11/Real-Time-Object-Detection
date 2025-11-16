import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from web_client import send_jpeg_bytes
from visual import draw_detections
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Optional: if connecting remotely, provide STUN/TURN here (not required for local)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(page_title="Realtime Object Detection (Local)", layout="centered")

st.title("Lightweight Real-Time Object Detection — Frontend")

st.markdown(
    """
    **Mode:** Choose Snapshot (single images) or Live (continuous frames).
    The app sends frames to your backend `/upload_image` endpoint.
    """
)

mode = st.radio("Mode", ["Snapshot", "Live"])

# small sidebar controls
with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend base URL", value="http://localhost:8000")
    st.write("Note: change backend URL in `web_client.py` if you need persistent change.")
    conf_display = st.slider("Display confidence threshold (%)", 0, 100, 30)

# snapshot mode
if mode == "Snapshot":
    st.subheader("Snapshot capture")
    img_file = st.camera_input("Take a photo")
    if img_file is not None:
        # show preview
        st.image(img_file, caption="Captured image", use_column_width=True)
        # convert to bytes
        bytes_data = img_file.getvalue()
        with st.spinner("Sending to backend..."):
            result = send_jpeg_bytes(bytes_data, timeout=10.0)
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            dets = result.get("detections", [])
            # convert to np array and draw
            pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            frame = np.array(pil)[:, :, ::-1].copy()  # RGB->BGR
            # filter by conf
            dets = [d for d in dets if d.get("score", 0) * 100 >= conf_display]
            annotated = draw_detections(frame, dets)
            annotated = annotated[:, :, ::-1]  # BGR->RGB for display
            st.image(annotated, caption="Detections", use_column_width=True)
            st.json({"detections": dets})

# live mode
else:
    st.subheader("Live feed (webcam)")
    st.write("Press **Start** to enable webcam and begin streaming frames to the backend.")

    class BackendPoster(VideoTransformerBase):
        def __init__(self):
            # no heavy state here
            self._last_time = time.time()
            self._fps = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # encode to JPEG
            ret, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                return frame
            jpg_bytes = jpg.tobytes()
            # send to backend
            res = send_jpeg_bytes(jpg_bytes, timeout=3.0)
            if "detections" in res:
                dets = res["detections"]
                # optional threshold
                dets = [d for d in dets if d.get("score", 0) * 100 >= conf_display]
                img = draw_detections(img, dets)
            else:
                # draw a small warning corner if backend fails
                cv2.putText(img, "Backend error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # update fps
            now = time.time()
            dt = now - self._last_time if self._last_time else 1.0
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_time = now
            cv2.putText(img, f"FPS: {self._fps:.1f}", (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return img

    ctx = webrtc_streamer(
        key="object-detect",
        mode="sendrecv",
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=BackendPoster,
        async_transform=True,
    )

    if ctx.state.playing:
        st.write("Streaming…")
    else:
        st.write("Click Start to begin streaming the webcam feed.")
