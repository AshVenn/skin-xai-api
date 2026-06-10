from pydantic import BaseModel
from typing import List


class TopKPrediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class: str
    confidence: float
    top_k: List[TopKPrediction]
    heatmap_base64: str
    heatmap_content_type: str = "image/png"
    warning: str