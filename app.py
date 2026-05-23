from contextlib import asynccontextmanager

import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import PromptInjectionFeatureEngineer
from src.model import predict


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int
    probability: float
    is_injection: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app):
    global booster, feature_engineer
    booster = xgb.Booster()
    booster.load_model("models/xgboost_model.json")
    feature_engineer = PromptInjectionFeatureEngineer()
    yield


app = FastAPI(
    title="Prompt Injection Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    result = predict(request.text, booster, feature_engineer)
    return PredictResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=booster is not None)