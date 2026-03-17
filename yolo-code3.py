import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# YOLOv8 Modell laden (Standard: yolov8n.pt)
model = YOLO("yolov8n.pt")

st.title("Bildinhalt-Analysator mit YOLOv8")
st.write("Lade ein Bild hoch, und das Modell beschreibt, was darauf zu sehen ist.")

# Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Bild", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Bild öffnen
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    # Temporäre Datei speichern, da YOLO Pfad oder Array akzeptiert
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    # Objekterkennung durchführen
    results = model(tmp_path)[0]
    
    # Erkannten Objekte extrahieren
    detected_objects = results.names  # Klassen-Name Mapping
    labels = [detected_objects[int(box.cls)] for box in results.boxes]  # erkannten Labels
    
    if labels:
        st.subheader("Erkannte Objekte:")
        st.write(", ".join(labels))
    else:
        st.write("Keine Objekte erkannt.")
