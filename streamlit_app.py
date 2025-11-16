import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import io

from web_client import send_jpeg_bytes
from visual import draw_detections

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Real-Time Object Detection")

st.title("Real-Time Object Detection — Streamlit Frontend")

mode = st.radio("Choose Mode:", ["Snapshot", "Live"])

conf_threshold = st.slider("Min Confidence (%)", 0, 100, 30)

# -------------- SNAPSHOT MODE ----------------
if mode == "Snapshot":
    img_file = st.camera_input("Capture Image")

    if img_file is not None:
        bytes_data = img_file.getvalue()

        st.image(bytes_data, caption="Captured")

        result = send_jpeg_bytes(bytes_data)
        dets = result.get("detections", [])

        pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        frame = np.array(pil)[:, :, ::-1]

        dets = [d for d in dets if d["score"] * 100 >= conf_threshold]

        annotated = draw_detections(frame, dets)
        annotated = annotated[:, :, ::-1]

        st.image(annotated, caption="Detections")
        st.json({"detections": dets})


# -------------- LIVE MODE ----------------
else:
    class Processor(VideoTransformerBase):
        def __init__(self):
            self.last = time.time()
            self.fps = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            ret, jpg = cv2.imencode(".jpg", img)
            data = send_jpeg_bytes(jpg.tobytes())

            dets = data.get("detections", [])
            dets = [d for d in dets if d["score"] * 100 >= conf_threshold]

            img = draw_detections(img, dets)

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1/(now-self.last))
            self.last = now

            cv2.putText(img, f"FPS: {self.fps:.1f}", (10, img.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            return img

    webrtc_streamer(
        key="example",
        mode="sendrecv",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=Processor,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
