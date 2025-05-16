from typing import Dict, List, Optional, Union
from pydantic import BaseModel


class ImageUpload(BaseModel):
    image: str  # Base64 encoded image string


class HealthResponse(BaseModel):
    status: str
    version: str


class ClassificationResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float


class PredictionItem(BaseModel):
    class_name: str
    probability: float


class DetailedClassificationResponse(BaseModel):
    predictions: List[PredictionItem]
    top_prediction: str
    processing_time: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
