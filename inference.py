import pandas as pd
from pathlib import Path

from preprocessing import data_cleaning
from feature_engineering import create_horizon_dataset
from model_io import load_model


def build_inference_features(
    csv_path, horizon, lags=(1, 2, 3, 6, 12), rolling_windows=(3, 6, 12)
):
    """
    Turn a raw price CSV into the same feature matrix the model was
    trained on, keeping the most recent rows even though their
    target_price is unknown (that's exactly what we want to predict).
    """
    df = data_cleaning(csv_path)

    horizon_df = create_horizon_dataset(
        df=df,
        horizon=horizon,
        lags=lags,
        rolling_windows=rolling_windows,
        for_inference=True,
    )

    drop_columns = [
        column for column in ("target_date", "target_price") if column in horizon_df.columns
    ]
    X = horizon_df.drop(columns=drop_columns)

    return X, horizon_df


def predict(model_path, csv_path, horizon=1, lags=(1, 2, 3, 6, 12), rolling_windows=(3, 6, 12)):
    """
    Load the pickled model and score it on a given dataset.

    Parameters
    ----------
    model_path : path to the .pkl file produced by model_io.save_model
    csv_path   : path to a raw price CSV, same schema as the training data
    horizon    : must match the horizon the model was trained on

    Returns
    -------
    pd.DataFrame indexed by forecast_origin, with columns:
        target_date      - the date being forecast
        predicted_price  - the model's prediction
    """
    model = load_model(model_path)

    X, horizon_df = build_inference_features(
        csv_path=csv_path, horizon=horizon, lags=lags, rolling_windows=rolling_windows
    )

    predictions = pd.Series(model.predict(X), index=X.index, name="predicted_price")

    result = pd.DataFrame({"predicted_price": predictions})

    if "target_date" in horizon_df.columns:
        result.insert(0, "target_date", horizon_df["target_date"])

    return result


def save_predictions(predictions, horizon, reports_dir="./reports", filename=None):
    """
    Save inference predictions to CSV, flagging the last `horizon` rows
    as the genuine forward-looking forecast - i.e. the forecast_origins
    whose target_price is unknown/beyond the available data, which is
    exactly what "predicting `horizon` steps ahead" means in practice.

    For horizon=1, only the very last row is flagged.
    For horizon=3, the last 3 rows are flagged.
    """
    predictions = predictions.copy()
    predictions["is_forecast"] = False
    predictions.iloc[-horizon:, predictions.columns.get_loc("is_forecast")] = True

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"test_predictions_h{horizon}.csv"
    save_path = reports_dir / filename

    predictions.to_csv(save_path)
    print(f"Saved predictions to {save_path} ({horizon} row(s) flagged as is_forecast=True)")

    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with a saved model.")
    parser.add_argument("--model-path", default=r"../data/best_model.pkl")
    parser.add_argument("--csv-path", default=r"../data/crude-oil-price.csv")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--reports-dir", default="./reports")
    args = parser.parse_args()

    predictions = predict(args.model_path, args.csv_path, args.horizon)
    print(predictions)
    save_predictions(predictions, args.horizon, reports_dir=args.reports_dir)
