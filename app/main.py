import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.assistant_service import DERMALENS_SAFETY_WARNING, assistant_service
from app.model_service import model_service
from app.schemas import AssistantRequest, AssistantResponse, PredictionResponse


app = FastAPI(
    title="DermaLens API",
    description="FastAPI backend for DermaLens visual skin insights and assistant support.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SAFETY_WARNING = DERMALENS_SAFETY_WARNING
logger = logging.getLogger(__name__)


@app.get("/")
def root():
    return {
        "message": "DermaLens API is running.",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "num_classes": model_service.num_classes,
        "image_size": model_service.image_size,
    }


@app.get("/classes")
def get_classes():
    return {
        "idx_to_class": model_service.idx_to_class,
        "class_to_idx": model_service.class_to_idx,
    }


@app.get("/assistant/health")
def assistant_health():
    return assistant_service.get_health_status()


@app.post("/assistant", response_model=AssistantResponse)
def ask_assistant(request: AssistantRequest):
    if not request.predicted_class.strip():
        raise HTTPException(status_code=400, detail="predicted_class must not be empty.")

    if not request.user_question.strip():
        raise HTTPException(status_code=400, detail="user_question must not be empty.")

    if len(request.user_question) > 500:
        raise HTTPException(status_code=400, detail="user_question must be 500 characters or fewer.")

    if not assistant_service.is_available():
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured on the server.",
        )

    try:
        return assistant_service.ask_assistant(
            predicted_class=request.predicted_class.strip(),
            confidence=request.confidence,
            top_k=request.top_k,
            user_question=request.user_question.strip(),
        )
    except Exception as error:
        logger.exception("Assistant service failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="Assistant service is temporarily unavailable.",
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPEG, PNG, or WEBP image.",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty image file.",
        )

    try:
        result = model_service.predict_with_gradcam(image_bytes)

        return PredictionResponse(
            predicted_class_id=result["predicted_class_id"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            top_k=result["top_k"],
            heatmap_base64=result["heatmap_base64"],
            warning=SAFETY_WARNING,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
