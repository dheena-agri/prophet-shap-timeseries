# Advanced Time Series Forecasting with Prophet and SHAP

## Author
**Dheenadhayalan**

---

## Project Overview
This project implements an advanced time series forecasting pipeline using **Facebook Prophet** with a strong emphasis on **model explainability** through **SHAP (SHapley Additive Explanations)**.

The objective is to build a forecasting model that can handle complex real-world patterns such as non-linear trends, multiple seasonalities, holidays, and external regressors, while also providing transparent explanations for its predictions.

---

## Key Objectives
- Generate or acquire a multi-year time series dataset
- Model trends, seasonality, and holidays using Prophet
- Tune hyperparameters using time series cross-validation
- Explain model predictions using SHAP
- Deliver production-ready, well-documented Python code

---

## Dataset Description
- **Type:** Synthetic sales time series
- **Time span:** 2018 – 2024 (7 years)
- **Frequency:** Daily
- **Target variable:** Sales
- **Patterns included:**
  - Non-linear trend with changepoints
  - Weekly seasonality
  - Yearly seasonality
  - Holiday/festival effects
  - External regressors:
    - Marketing spend
    - Promotion indicator

Synthetic data was used to ensure reproducibility and controlled experimentation while closely mimicking real-world business data.

---

## Methodology

### 1. Data Generation
A realistic dataset was programmatically generated with embedded seasonal patterns, noise, and event-driven effects.

### 2. Forecasting Model
Facebook Prophet was used due to its:
- Additive and interpretable structure
- Built-in support for holidays
- Automatic changepoint detection
- Robust handling of missing data

### 3. Hyperparameter Tuning
Model parameters were optimized using **walk-forward (rolling-origin) cross-validation**, ensuring temporal integrity.

**Tuned parameters:**
- `changepoint_prior_scale`
- `seasonality_prior_scale`
- `seasonality_mode` (additive vs multiplicative)

**Evaluation metric:** RMSE

### 4. Explainability with SHAP
SHAP was integrated to explain the contribution of:
- Trend component
- Fourier-based seasonal components
- Holiday effects
- External regressors

Both global and local explanations were generated.

---

## Results Summary
- The trend component was the strongest long-term driver
- Yearly seasonality and holidays explained recurring demand peaks
- Marketing spend showed a positive but diminishing effect
- Promotions caused short-term sales spikes
- Weekly seasonality contributed consistent minor variations

---

## Project Structure
