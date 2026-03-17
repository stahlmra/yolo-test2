import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# YOLO Modell laden
model = YOLO("yolov8n.pt")

st.title("🧠 Bildanalyse mit YOLO")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Originalbild", use_column_width=True)

    # YOLO Objekterkennung
    results = model(img_array)
    boxes = results[0].boxes
    names = model.names

    detected_objects = []

    # Bild für Bounding Boxes vorbereiten
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        detected_objects.append(f"{label} ({conf:.2f})")

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 15), f"{label} {conf:.2f}", fill="green", font=font)

    st.subheader("🔍 Erkannte Objekte")
    if detected_objects:
        for obj in detected_objects:
            st.write(f"- {obj}")
    else:
        st.write("Keine Objekte erkannt")

    st.image(img_with_boxes, caption="Mit Bounding Boxes", use_column_width=True)
