import xgboost as xgb

from src.features import FEATURE_NAMES, PromptInjectionFeatureEngineer


def load_model(path="models/xgboost_model.json"):
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def predict(text, booster, feature_engineer=None):
    if feature_engineer is None:
        feature_engineer = PromptInjectionFeatureEngineer()

    df = feature_engineer.extract_as_dataframe(text)
    dm = xgb.DMatrix(df)
    iteration_range = (0, booster.best_iteration + 1)
    proba = float(booster.predict(dm, iteration_range=iteration_range)[0])
    label = int(proba >= 0.5)
    return {
        "label": label,
        "probability": round(proba, 4),
        "is_injection": bool(label == 1),
    }