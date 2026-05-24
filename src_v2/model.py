import pandas as pd
import xgboost as xgb

from src_v2.embeddings import EmbeddingGenerator, EMBEDDING_DIM
from src_v2.features import FEATURE_NAMES_V2, PromptInjectionFeatureEngineerV2


def load_model(path: str = "models/xgboost_model_v2.json") -> xgb.Booster:
    try:
        booster = xgb.Booster()
        booster.load_model(path)
    except (FileNotFoundError, ValueError) as e:
        raise FileNotFoundError(f"Failed to load model from {path}: {e}")
    return booster


def predict(
    text: str,
    booster: xgb.Booster,
    feature_engineer: PromptInjectionFeatureEngineerV2 | None = None,
    embedding_generator: EmbeddingGenerator | None = None,
) -> dict:
    if feature_engineer is None:
        feature_engineer = PromptInjectionFeatureEngineerV2()
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()

    handcrafted = feature_engineer.extract_all(text)
    embedding = embedding_generator.encode(text)

    features = {**handcrafted}
    for i, val in enumerate(embedding):
        features[f"embedding_{i}"] = val

    all_feature_names = FEATURE_NAMES_V2 + [f"embedding_{i}" for i in range(EMBEDDING_DIM)]
    df = pd.DataFrame([features], columns=all_feature_names)
    df = df[all_feature_names]

    dm = xgb.DMatrix(df)
    best_iter = getattr(booster, "best_iteration", 0)
    iteration_range = (0, best_iter + 1)
    proba = float(booster.predict(dm, iteration_range=iteration_range)[0])
    label = int(proba > 0.5)
    return {
        "label": label,
        "probability": round(proba, 4),
        "is_injection": label == 1,
    }