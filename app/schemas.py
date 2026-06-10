from typing import List

from pydantic import BaseModel


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


class AssistantTopKItem(BaseModel):
    class_name: str
    confidence: float


class AssistantRequest(BaseModel):
    predicted_class: str
    confidence: float
    top_k: List[AssistantTopKItem]
    user_question: str


class AssistantResponse(BaseModel):
    answer: str
    warning: str
