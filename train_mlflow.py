import hashlib
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from pipeline import main


PROJECT_DIR = Path(__file__).resolve().parent

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = "crude-oil-forecasting"


def train_horizon(horizon):
    csv_path = PROJECT_DIR / "data" / "crude-oil-price.csv"
    model_path = PROJECT_DIR / "artifacts" / f"best_model_h{horizon}.pkl"
    figures_dir = PROJECT_DIR / "figures"
    reports_dir = PROJECT_DIR / "reports"

    validation_months = 24
    test_months = 24

    with mlflow.start_run(run_name=f"horizon-{horizon}"):
        mlflow.set_tags({
            "task": "oil-price-forecasting",
            "horizon_months": str(horizon),
            "evaluation": "held-out-test",
        })

        mlflow.log_params({
            "horizon": horizon,
            "validation_months": validation_months,
            "test_months": test_months,
            "xgb_search_iterations": 40,
            "dataset": csv_path.name,
            "dataset_sha256": hashlib.sha256(
                csv_path.read_bytes()
            ).hexdigest(),
        })

        model, test_metrics, predictions = main(
            csv_path=str(csv_path),
            horizon=horizon,
            validation_months=validation_months,
            test_months=test_months,
            model_path=str(model_path),
            figures_path=str(figures_dir),
            reports_path=str(reports_dir),
            show=False,
        )

        # Accuracy of the selected model on the held-out test set.
        mlflow.log_metrics({
            f"test_{name.lower()}": float(value)
            for name, value in test_metrics.items()
        })

        # Capture both candidates' validation scores and parameters.
        comparison_path = reports_dir / "model_comparison.csv"
        comparison = pd.read_csv(comparison_path)

        mlflow.log_param(
            "selected_model",
            comparison.iloc[0]["Model"],
        )

        for _, row in comparison.iterrows():
            candidate = row["Model"].lower()

            mlflow.log_metrics({
                f"{candidate}_cv_rmse": float(row["CV_RMSE"]),
                f"{candidate}_val_rmse": float(row["RMSE"]),
                f"{candidate}_val_mae": float(row["MAE"]),
                f"{candidate}_val_r2": float(row["R2"]),
            })

            mlflow.log_param(
                f"{candidate}_best_parameters",
                row["Best_Parameters"],
            )

        # Save an MLflow model, including its Python environment.
        # Both your Ridge pipeline and XGBRegressor expose sklearn APIs.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        # Preserve this run's files before another horizon overwrites them.
        artifacts = [
            comparison_path,
            reports_dir / f"test_metrics_h{horizon}.csv",
            reports_dir / f"test_predictions_h{horizon}.csv",
            figures_dir / f"test_predictions_h{horizon}.png",
            model_path,
        ]

        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))

        print(
            f"Horizon {horizon} logged. "
            f"Run ID: {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    # Your pipeline currently uses relative paths internally.
    os.chdir(PROJECT_DIR)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    for horizon in (1, 2, 3):
        train_horizon(horizon)