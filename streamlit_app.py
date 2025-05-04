import streamlit as st
from PIL import Image
from ultralytics import YOLO
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Real-time Detection with YOLO and Webcam")

# Load YOLOv8 model
model = YOLO("/workspaces/blank-app/best.pt")  # Replace with your trained model path

# Define the video processor
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_img = results[0].plot()
        return annotated_img

# Webcam stream
webrtc_streamer(key="yolo", video_processor_factory=YOLOTransformer)

st.write("Upload an image below (optional)")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Detection
    results = model(img)
    for r in results:
        st.image(r.plot(), caption="Detected Image", use_column_width=True)
