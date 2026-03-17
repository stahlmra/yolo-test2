import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os
import base64
from openai import OpenAI

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# YOLO Modell laden
model = YOLO("yolov8n.pt")

st.title("🧠 Bildanalyse mit YOLO + KI")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Originalbild", use_column_width=True)

    # YOLO Erkennung
    results = model(img_array)
    boxes = results[0].boxes
    names = model.names

    detected_objects = []
    img_with_boxes = img_array.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]

        detected_objects.append(f"{label} ({conf:.2f})")

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_with_boxes,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    st.subheader("🔍 Erkannte Objekte")
    if detected_objects:
        for obj in detected_objects:
            st.write(f"- {obj}")
    else:
        st.write("Keine Objekte erkannt")

    st.image(img_with_boxes, caption="Mit Bounding Boxes", use_column_width=True)

    # Bild für API vorbereiten
    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # OpenAI Beschreibung
    st.subheader("📝 KI-Beschreibung")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Beschreibe dieses Bild auf Deutsch."},
                        {
                            "type": "input_image",
                            "image_base64": image_base64,
                        },
                    ],
                }
            ],
        )

        description = response.output[0].content[0].text
        st.write(description)

    except Exception as e:
        st.error(f"API Fehler: {e}")
