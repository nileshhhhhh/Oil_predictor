import numpy as np
import pandas as pd

def create_horizon_dataset(
    df, horizon, lags=(1, 2, 3, 6, 12), rolling_windows=(3, 6, 12), for_inference=False
):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    data = df.copy()

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").set_index("date")
    else:
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

    data = data[["price"]].copy()
    data["current_price"] = data["price"]

    # Historical prices
    for lag in lags:
        data[f"price_lag_{lag}"] = data["price"].shift(lag)

    # Historical movement features
    for period in (1, 3, 6, 12):
        data[f"price_change_{period}"] = data["price"].diff(period)

        data[f"price_return_{period}"] = data["price"].pct_change(
            periods=period, fill_method=None
        )

    # Rolling features available at the forecast origin
    for window in rolling_windows:
        rolling_price = data["price"].rolling(window)

        data[f"rolling_mean_{window}"] = rolling_price.mean()

        data[f"rolling_std_{window}"] = rolling_price.std()

        data[f"rolling_min_{window}"] = rolling_price.min()

        data[f"rolling_max_{window}"] = rolling_price.max()

        data[f"price_vs_mean_{window}"] = data["price"] - data[f"rolling_mean_{window}"]

    # Volatility
    monthly_return = data["price"].pct_change(fill_method=None)

    for window in (3, 6, 12):
        data[f"volatility_{window}"] = monthly_return.rolling(window).std()

    # Trend
    data["time_index"] = np.arange(len(data))

    # Direct future target
    data["target_price"] = data["price"].shift(-horizon)

    data["target_date"] = data.index + pd.DateOffset(months=horizon)

    # Calendar features for the predicted month
    target_month = data["target_date"].dt.month

    data["target_month_sin"] = np.sin(2 * np.pi * (target_month - 1) / 12)

    data["target_month_cos"] = np.cos(2 * np.pi * (target_month - 1) / 12)

    data = data.replace([np.inf, -np.inf], np.nan)

    if for_inference:
        # Keep rows even if target_price is NaN (unknown future value) -
        # this preserves the most recent forecast origins, which are the
        # ones we actually want predictions for. Only drop rows with
        # missing feature values (e.g. warm-up rows for lag/rolling windows).
        feature_columns = [col for col in data.columns if col != "target_price"]
        data = data.dropna(subset=feature_columns).drop(columns="price")
    else:
        data = data.dropna().drop(columns="price")

    data.index.name = "forecast_origin"

    return data
