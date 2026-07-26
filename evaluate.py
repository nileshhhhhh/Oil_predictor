from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def evaluate_model(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def save_metrics(test_metrics, model_comparison, horizon, reports_dir="./reports"):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.insert(0, "Model", model_comparison.loc[0, "Model"])
    metrics_df.insert(1, "Horizon", horizon)

    save_path = reports_dir / f"test_metrics_h{horizon}.csv"
    metrics_df.to_csv(save_path, index=False)
    print(f"Saved test metrics to {save_path}")
    return save_path