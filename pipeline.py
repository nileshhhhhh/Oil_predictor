from pathlib import Path

from preprocessing import data_cleaning
from feature_engineering import create_horizon_dataset
from split import split_horizon_dataset, prepare_train_val_test, time_series_split
from tune import tune_ridge_regressor, tune_xgb_regressor, model_comparison, best_model
from evaluate import save_metrics
from plots import show_plot
from model_io import save_model
from inference import predict as run_inference, save_predictions

csv_path = r'./data/crude-oil-price.csv'
validation_months = 24
test_months = 24
model_comparison_path = r'./reports/model_comparison.csv'
figures_path = r'./figures'
reports_path = r'./reports'


def main(csv_path, horizon, validation_months, test_months, model_path, figures_path, reports_path=reports_path, show=True):
    df = data_cleaning(csv_path)
    df = create_horizon_dataset(df=df, horizon=horizon)
    train_df, validation_df, test_df = split_horizon_dataset(
        df, validation_months=validation_months, test_months=test_months
    )
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test(
        train_df=train_df, validation_df=validation_df, test_df=test_df
    )
    tscv = time_series_split(horizon=horizon)
    ridge_results = tune_ridge_regressor(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, cv=tscv
    )
    xgb_results = tune_xgb_regressor(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, cv=tscv, n_iter=40
    )
    comparison = model_comparison(ridge_results, xgb_results)
    Path(model_comparison_path).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(model_comparison_path)

    final_model, test_metrics, test_predictions = best_model(
        comparison, ridge_results, xgb_results, X_train, y_train, X_val, y_val, X_test, y_test
    )
    save_metrics(test_metrics, comparison, horizon)

    show_plot(y_test, test_predictions, comparison, horizon, figures_path, show=show)

    saved_model_path = save_model(final_model, model_path)
    print(f"Saved best model ({comparison.loc[0, 'Model']}) to {saved_model_path}")

    predictions = run_inference(model_path=saved_model_path, csv_path=csv_path, horizon=horizon)
    save_predictions(predictions, horizon, reports_dir=reports_path)

    return final_model, test_metrics, predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the crude-oil price model and immediately run inference with it."
    )
    parser.add_argument(
        "--mode", choices=["both", "predict"], default="both",
        help="'both' (default) trains then runs inference. 'predict' skips training and reuses the saved model for --horizon."
    )
    parser.add_argument("--csv-path", default=csv_path)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--validation-months", type=int, default=validation_months)
    parser.add_argument("--test-months", type=int, default=test_months)
    parser.add_argument("--model-path", default=None, help="Defaults to ./artifacts/best_model_h<horizon>.pkl")
    parser.add_argument("--figures-dir", default=figures_path, help="Directory to save the test-predictions plot into.")
    parser.add_argument("--reports-dir", default=reports_path, help="Directory to save metrics/predictions CSVs into.")
    parser.add_argument("--no-show", action="store_true", help="Skip displaying the plot interactively (it is still saved).")
    args = parser.parse_args()

    model_path = args.model_path or f"./artifacts/best_model_h{args.horizon}.pkl"

    if args.mode == "both":
        main(
            csv_path=args.csv_path,
            horizon=args.horizon,
            validation_months=args.validation_months,
            test_months=args.test_months,
            model_path=model_path,
            figures_path=args.figures_dir,
            reports_path=args.reports_dir,
            show=not args.no_show,
        )
    else:
        predictions = run_inference(
            model_path=model_path, csv_path=args.csv_path, horizon=args.horizon
        )
        save_predictions(predictions, args.horizon, reports_dir=args.reports_dir)
