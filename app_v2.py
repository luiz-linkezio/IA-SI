import logging
import os
from contextlib import asynccontextmanager

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src_v2.embeddings import EmbeddingGenerator
from src_v2.features import PromptInjectionFeatureEngineerV2
from src_v2.model import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH_V2", "models/xgboost_model_v2.json")


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
feature_engineer: PromptInjectionFeatureEngineerV2 | None = None
embedding_generator: EmbeddingGenerator | None = None


@asynccontextmanager
async def lifespan(app):
    global booster, feature_engineer, embedding_generator
    try:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        feature_engineer = PromptInjectionFeatureEngineerV2()
        embedding_generator = EmbeddingGenerator()
        logger.info(f"V2 model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load V2 model: {e}")
        booster = None
        feature_engineer = None
        embedding_generator = None
    yield


app = FastAPI(
    title="Prompt Injection Detection API V2",
    version="2.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if booster is None or feature_engineer is None or embedding_generator is None:
        raise HTTPException(status_code=503, detail="V2 model not loaded")
    try:
        result = predict(request.text, booster, feature_engineer, embedding_generator)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"V2 prediction failed: {e}")
        raise HTTPException(status_code=500, detail="V2 prediction failed")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = booster is not None and feature_engineer is not None and embedding_generator is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )