import io
import time
from PIL import Image
from ultralytics import YOLO

from .schemas import DetectionsResponse, Detection, BoundingBox


class YoloObjectDetection:
    # Inicializa el modelo de YOLO para la descarga del modelo.
    def __init__(self, model_name: str = "yolo26s.pt"):
        print(f"Cargando modelo Yolo: {model_name}")
        self.model = YOLO(model_name)

    def predict(self, image_bytes: bytes) -> DetectionsResponse:
        # cargar la imagen desde los bytes y convertirla a RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Realizar la inferencia y medir el tiempo de inferencia (inicialmente en milisegundos)
        start_time = time.time()
        # Realizar la inferencia utilizando el modelo de YOLO y obtener los resultados
        results = self.model(image)

        end_time = time.time()

        inference_time = (
            round(end_time - start_time, 2) * 1000
        )  # Convert to milliseconds
        detections = []
        for result in results:
            boxes = result.boxes
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=conf,
                        box=BoundingBox(
                            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
                        ),
                    )
                )

        return DetectionsResponse(inference_time=inference_time, detections=detections)
