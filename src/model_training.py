from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd

def get_holidays() -> pd.DataFrame:
    return pd.DataFrame({
        "holiday": "festival",
        "ds": pd.to_datetime([
            "2019-10-27", "2020-11-14", "2021-11-04",
            "2022-10-24", "2023-11-12", "2024-11-01"
        ]),
        "lower_window": -2,
        "upper_window": 2
    })

def tune_prophet(df: pd.DataFrame, holidays: pd.DataFrame) -> dict:
    param_grid = {
        "changepoint_prior_scale": [0.05, 0.1, 0.15],
        "seasonality_prior_scale": [5, 10, 15],
        "seasonality_mode": ["additive", "multiplicative"]
    }

    best_rmse = float("inf")
    best_params = {}

    for cps in param_grid["changepoint_prior_scale"]:
        for sps in param_grid["seasonality_prior_scale"]:
            for mode in param_grid["seasonality_mode"]:
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode=mode,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    holidays=holidays
                )
                model.add_regressor("marketing_spend")
                model.add_regressor("promotion")
                model.fit(df)

                cv = cross_validation(
                    model,
                    initial="1095 days",
                    period="180 days",
                    horizon="365 days"
                )
                rmse = performance_metrics(cv)["rmse"].mean()

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        "changepoint_prior_scale": cps,
                        "seasonality_prior_scale": sps,
                        "seasonality_mode": mode
                    }

    return best_params
