# Backedn usando FastAPI para detección de objetos con YOLOv8
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import DetectionsResponse
from app.yolo_service import YoloObjectDetection


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar el modelo de YOLO al iniciar la aplicación
    ml_models["yolo"] = YoloObjectDetection(model_name="yolo26s.pt")
    print("Modelo YOLO cargado exitosamente.")
    yield
    # Ejecucion de shutdown
    ml_models.clear()
    print("Modelos limpiados al finalizar la app y recursos liberados.")


# Instanciamos la aplicacion de FastAPI con el contexto de vida util definido
app = FastAPI(
    title="API de Deteccion de OBjetos con YOLO ALETOSAAA",
    description="Una API Ppara detectar vehiculos, peatones y ciclistas en imagenes",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("api/v1/detect", response_model=DetectionsResponse)
async def detect_objects(file: UploadFile) -> DetectionsResponse:
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, detail="Archivo no soportado. Solo se permiten JPEG y PNG."
        )

    try:
        image_bytes = await file.read()
        yolo_service: YoloObjectDetection = ml_models.get("yolo")
        if not yolo_service:
            raise HTTPException(
                status_code=500, detail="Modelo de detección no disponible."
            )

        detections_response = yolo_service.predict(image_bytes)
        return detections_response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la imagen: {str(e)}"
        )
