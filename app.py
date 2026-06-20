import logging
import os
from contextlib import asynccontextmanager

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import PromptInjectionFeatureEngineer
from src.model import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/xgboost_model.json")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    probability: float
    is_injection: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


booster: xgb.Booster | None = None
feature_engineer: PromptInjectionFeatureEngineer | None = None


@asynccontextmanager
async def lifespan(app):
    global booster, feature_engineer
    try:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        feature_engineer = PromptInjectionFeatureEngineer()
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        booster = None
        feature_engineer = None
    yield


app = FastAPI(
    title="Prompt Injection Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if booster is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predict(request.text, booster, feature_engineer)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = booster is not None and feature_engineer is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )
