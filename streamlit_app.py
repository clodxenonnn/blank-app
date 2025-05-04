import streamlit as st
from PIL import Image
import numpy as np
import torch
import tempfile

st.title("🐶 YOLOv9 Dog Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    # Load YOLOv9 model (make sure you have trained weights)
    model = torch.hub.load("WongKinYiu/yolov9", "custom", "yolov9.pt", trust_repo=True)
    results = model(image_path)

    # Show predictions
    results.show()  # opens window locally
    st.image(results.render()[0])  # display image with boxes in Streamlit

    # Text result
    for pred in results.pandas().xyxy[0].iterrows():
        label = pred[1]['name']
        conf = pred[1]['confidence']
        st.markdown(f"- Detected: **{label}**, Confidence: **{conf:.2f}**")
