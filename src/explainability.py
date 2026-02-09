import shap
import pandas as pd

def generate_shap(forecast: pd.DataFrame) -> shap.Explanation:
    components = forecast[[
        "trend", "weekly", "yearly", "holidays",
        "marketing_spend", "promotion"
    ]].dropna()

    explainer = shap.Explainer(lambda x: x.sum(axis=1), components)
    return explainer(components)
