import numpy as np
import pandas as pd

def generate_dataset() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
    n = len(dates)

    trend = 0.03 * np.arange(n) + 10 * np.sin(np.arange(n) / 400)
    yearly = 15 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    weekly = 5 * np.sin(2 * np.pi * dates.dayofweek / 7)

    marketing_spend = np.random.gamma(2, 20, n)
    promotion = np.random.binomial(1, 0.1, n)
    noise = np.random.normal(0, 4, n)

    sales = (
        200 + trend + yearly + weekly
        + 0.6 * marketing_spend
        + 20 * promotion
        + noise
    )

    return pd.DataFrame({
        "ds": dates,
        "y": sales,
        "marketing_spend": marketing_spend,
        "promotion": promotion
    })
