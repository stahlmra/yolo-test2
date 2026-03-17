import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# App-Titel
st.title("🖼️ YOLOv8 Bildanalyse")

st.write("Lade ein Bild hoch, und die App erkennt automatisch Objekte, Personen, Tiere oder Szenen.")

# Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Bild öffnen
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    st.write("Analyse läuft...")

    # YOLOv8 Modell laden (ultralytics YOLOv8n v8.0 ist klein und schnell)
    model = YOLO("yolov8n.pt")  # Stelle sicher, dass yolov8n.pt heruntergeladen wird

    # Vorhersage auf das Bild
    results = model(image)

    # Ergebnisse anzeigen
    st.subheader("Erkannte Objekte:")
    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- {cls} ({conf:.2f})")

    # Optional: Bild mit Bounding Boxes
    annotated_img = results[0].plot()  # Numpy array mit Annotationen
    st.image(annotated_img, caption="Bild mit erkannten Objekten", use_column_width=True)
