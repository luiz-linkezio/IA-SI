import xgboost as xgb

from src.features import PromptInjectionFeatureEngineer


def load_model(path: str = "models/xgboost_model.json") -> xgb.Booster:
    try:
        booster = xgb.Booster()
        booster.load_model(path)
    except (FileNotFoundError, ValueError) as e:
        raise FileNotFoundError(f"Failed to load model from {path}: {e}")
    return booster


def predict(
    text: str,
    booster: xgb.Booster,
    feature_engineer: PromptInjectionFeatureEngineer | None = None,
) -> dict:
    if feature_engineer is None:
        feature_engineer = PromptInjectionFeatureEngineer()

    df = feature_engineer.extract_as_dataframe(text)
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

