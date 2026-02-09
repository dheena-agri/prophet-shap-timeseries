from prophet import Prophet
from data_generation import generate_dataset
from model_training import tune_prophet, get_holidays
from explainability import generate_shap

def main() -> None:
    df = generate_dataset()
    holidays = get_holidays()
    best_params = tune_prophet(df, holidays)

    model = Prophet(
        **best_params,
        yearly_seasonality=True,
        weekly_seasonality=True,
        holidays=holidays
    )
    model.add_regressor("marketing_spend")
    model.add_regressor("promotion")
    model.fit(df)

    future = model.make_future_dataframe(periods=180)
    future["marketing_spend"] = df["marketing_spend"].mean()
    future["promotion"] = 0

    forecast = model.predict(future)
    shap_values = generate_shap(forecast)
    shap.summary_plot(shap_values)

if __name__ == "__main__":
    main()
