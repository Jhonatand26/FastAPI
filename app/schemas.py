from pydantic import BaseModel, Field  # P
from typing import List


class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Coordenada x minima del cuadro delimitador")
    y_min: float = Field(..., description="Coordenada y minima del cuadro delimitador")
    x_max: float = Field(..., description="Coordenada x maxima del cuadro delimitador")
    y_max: float = Field(..., description="Coordenada y maxima del cuadro delimitador")


class Detection(BaseModel):
    class_name: str = Field(
        ..., description="Nombre de la clase detectada (vehículo, peatón, etc.)"
    )
    confidence: float = Field(..., description="Confianza de la detección")
    box: BoundingBox = Field(..., description="Cuadro delimitador de la detección")


class DetectionsResponse(BaseModel):
    inference_time: float = Field(..., description="Tiempo de inferencia en segundos")
    detections: List[Detection] = Field(..., description="Lista de detecciones")
