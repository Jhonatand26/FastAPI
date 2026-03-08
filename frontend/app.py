import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# configuracion de la pagina de streamlit
st.set_page_config(
    page_title="Detector de Objetos Aletoso con YOLO26s + FastAPI",
    layout="centered",
    page_icon=":camera:",
)
# titulo y descripcion de la aplicacion
st.title("Detector de Objetos Aletoso con YOLO26s + FastAPI + Streamlit")
st.write(
    "Sube una imagen para detectar vehículos, peatones y ciclistas utilizando el modelo YOLO26s a través de una API FastAPI."
)

API_URL = "http://127.0.0.1:8000/api/v1/detect"
uploaded_file = st.file_uploader(
    "Selecciona una imagen (JPEG o PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # mostrar las columnas para comparar la imagen original con la imagen procesada
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Imagen Original")
        st.image(image, width="stretch")
    if st.button("Detectar Objetos"):
        try:
            uploaded_file.seek(
                0
            )  # Asegurarse de que el puntero del archivo esté al inicio
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            data = response.json()
            inference_time = data.get("inference_time", 0)
            detections = data.get("detections", [])
            st.success(f"Detección completada en {inference_time:.2f} ms")
            draw = ImageDraw.Draw(image)
            for det in detections:
                box = det["box"]
                class_name = det["class_name"]
                confidence = det["confidence"]
                x_min, y_min, x_max, y_max = (
                    box["x_min"],
                    box["y_min"],
                    box["x_max"],
                    box["y_max"],
                )
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
                font = ImageFont.load_default()
                text = f"{class_name} ({confidence:.2f})"
                bbox = draw.textbbox((x_min, y_min), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.rectangle(
                    [(x_min, y_min - text_h), (x_min + text_w, y_min)],
                    fill="red",
                )
                draw.text((x_min, y_min - text_h), text, fill="white", font=font)
            with col2:
                st.subheader("Imagen con Detecciones")
                st.image(image, width="stretch")
        except requests.exceptions.RequestException as e:
            st.error(f"Error al conectar con la API: {e}")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
