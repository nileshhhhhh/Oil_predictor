# Crude Oil Price Forecasting Pipeline

A small ML pipeline that trains a model to forecast crude oil prices `h` months
ahead, compares Ridge vs. HistGradientBoostingRegressor, saves the best model, and immediately runs
inference with it.

## Folder structure

```
Oil_predictor/
├── data/
│   └── crude-oil-price.csv     # raw input data (date, price, percentChange, change)
├── artifacts/
│   └── best_model_h{horizon}.pkl   # pickled best model, one file per trained horizon
├── figures/
│   └── test_predictions_h{horizon}.png  # actual vs. predicted plot on the test set
├── reports/
│   ├── model_comparison.csv    # Ridge vs. HistGradientBoostingRegressor: CV score + validation metrics + best params
│   ├── test_metrics_h{horizon}.csv     # test-set RMSE/MAE/R2 for the chosen model
│   └── test_predictions_h{horizon}.csv # every prediction, with the true forecast row(s) flagged
│
├── preprocessing.py         # data_cleaning(): load CSV, drop bad rows, parse dates
├── feature_engineering.py   # create_horizon_dataset(): lags, rolling stats, target shifting
├── split.py                 # time-based train/validation/test split + TimeSeriesSplit for CV
├── tune.py                  # hyperparameter search for Ridge and HistGradientBoostingRegressor, model_comparison(), best_model()
├── evaluate.py               # evaluate_model() metrics, save_metrics() to reports/
├── plots.py                  # show_plot(): draws + saves the actual-vs-predicted chart
├── model_io.py               # save_model() / load_model() — pickle persistence
├── inference.py               # build features for new data, predict(), save_predictions()
├── pipeline.py               # train/predict one horizon through the CLI
├── train_mlflow.py           # train horizons 1, 2, 3 and log runs to MLflow
└── requirements.txt          # pinned Python dependencies
```

**Flow at a glance:** `preprocessing` → `feature_engineering` → `split` → `tune`
(trains + compares Ridge and HistGradientBoostingRegressor) → `evaluate`/`plots` (score + visualize on
the test set) → `model_io` (pickle the winner) → `inference` (forecast with it) →
results land in `reports/` and `figures/`. Run `pipeline.py` for one horizon,
or `train_mlflow.py` to train all three horizons with MLflow tracking.

## Environment setup

Requires **Python 3.12**. Install the pinned dependencies from `requirements.txt`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The models use scikit-learn; XGBoost and its separate macOS `libomp` installation
are no longer required. Existing XGBoost pickle files must be retrained before
using prediction mode without XGBoost.

## Track horizons 1, 2, and 3 with MLflow

Local tracking requires no account. The workflow uses two terminals: one for the
MLflow server and one for training. Activate `.venv` in both terminals.

### 1. Start the tracking server

From the repository root, run:

```bash
source .venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Leave this terminal running and open <http://127.0.0.1:5000> in your browser.
Run information is stored in `mlflow.db`; uploaded models and files are stored
in `mlartifacts/`. Start the server from the same directory on subsequent runs
to reuse that history. To stop the server, press `Ctrl+C` in its terminal.

### 2. Train and log all horizons

In a second terminal, from the repository root:

```bash
source .venv/bin/activate
python train_mlflow.py
```

The script trains horizons **1, 2, and 3 sequentially**, using 24 months each for
validation and testing, and saves plots without opening a window. Each horizon
compares Ridge with `HistGradientBoostingRegressor` and logs a separate run in
the **`crude-oil-forecasting`** experiment:

```text
crude-oil-forecasting
├── horizon-1
├── horizon-2
└── horizon-3
```

Each invocation creates new runs, even though the run names repeat. Local output
files for the same horizon are overwritten; the uploaded artifacts remain
attached to their individual MLflow runs.

The tracking URI defaults to `http://127.0.0.1:5000`. If port 5000 is occupied,
start the server with `--port 5001`, then run training with:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python train_mlflow.py
```

`train_mlflow.py` currently has no command-line flags. Change the horizon tuple
in its main block to train a subset, or the values in `train_horizon()` to change
the validation/test durations. Running `pipeline.py` directly does not log to MLflow.

### 3. Inspect the experiment

In the UI, open **Experiments → crude-oil-forecasting**, select runs, and use
**Compare**. Each run records:

| Item | Logged values |
|---|---|
| Parameters | Horizon, validation/test durations, dataset filename and SHA-256, selected model, and each candidate's best parameters |
| Test metrics | `test_rmse`, `test_mae`, `test_r2` for the selected model |
| Candidate metrics | CV RMSE and validation RMSE/MAE/R², prefixed with `ridge_` or `histgradientboostingregressor_` |
| Artifacts | Model comparison CSV, test metrics CSV, predictions CSV, test plot, and pickle file |
| MLflow model | Selected estimator logged under `model`, including environment metadata |

The script currently logs the parameter `xgb_search_iterations=40`; this is a
legacy label for the 40 randomized-search candidates now used for
`HistGradientBoostingRegressor`. XGBoost is not trained.

Lower RMSE/MAE and higher R² indicate better accuracy. Compare repeated runs of
the same horizon first; different horizons forecast different target months.
The model is selected by validation RMSE, with test metrics reported afterward.
Individual tuning trials are not logged as separate runs, and the script does
not automatically register models in the Model Registry.

### Tracking scope and persistence

This workflow tracks training and historical evaluation. To monitor future
forecast accuracy, retain the original forecasts and their model/run IDs, join
them to actual prices when those prices become available, and log new evaluation
metrics. The script does not schedule evaluations or send alerts.

Keep `.venv/`, `mlflow.db` (including SQLite sidecar files), `mlartifacts/`, and
`mlruns/` out of version control. To back up experiment history, stop the server
and copy both the database and artifact directory together.

## How to test the model

Run all commands from the repository root (`Oil_predictor/`) with `.venv` activated.

### 1. Train + forecast (the default flow)

```bash
python pipeline.py --horizon 1
```

This does everything in one go:
1. Builds lag/rolling-window features and does a chronological train/validation/test split.
2. Tunes Ridge (grid search) and HistGradientBoostingRegressor (randomized search) with time-series cross-validation.
3. Picks the model with the lowest validation RMSE, refits it on train+validation, and scores it on the held-out test set.
4. Saves the plot, the metrics, the pickled model, and the predictions.

**What `--horizon` means:** the number of months ahead the model predicts. `--horizon 1` trains a model that predicts next month's price from this month's data; `--horizon 3` trains a model that predicts 3 months ahead. Each horizon gets its own model (`artifacts/best_model_h{horizon}.pkl`) and its own metrics, predictions, and plot files. `reports/model_comparison.csv` is shared and overwritten by each training run; MLflow preserves a copy for each tracked run.

Other useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--horizon` | `1` | months ahead to forecast |
| `--validation-months` | `24` | months held out for validation |
| `--test-months` | `24` | months held out for final testing |
| `--model-path` | `./artifacts/best_model_h{horizon}.pkl` | where the pickled model is saved/loaded from |
| `--figures-dir` | `./figures` | where the prediction plot is saved |
| `--reports-dir` | `./reports` | where the CSV reports are saved |
| `--no-show` | off | save the plot but don't pop up a window (use this on headless machines) |
| `--mode` | `both` | `both` = train + inference; `predict` = skip training, just run inference with the already-saved model for this horizon |

Example — train and forecast 3 months ahead, no plot popup:

```bash
python pipeline.py --horizon 3 --no-show
```

Example — reuse an already-trained horizon-1 model to forecast on a different CSV, without retraining:

```bash
python pipeline.py --mode predict --horizon 1 --csv-path ./data/new_prices.csv
```

### 2. Interpreting the results

**`reports/model_comparison.csv`** — one row per candidate model (Ridge, HistGradientBoostingRegressor),
ranked by validation RMSE. Shows `CV_RMSE` (cross-validation score during tuning),
`RMSE`/`MAE`/`R2` (on the validation set), and the winning `Best_Parameters`.
Row 0 is always the model that got picked and pickled.

**`reports/test_metrics_h{horizon}.csv`** — the headline numbers: how the
*winning* model performed on the untouched test set.

| Metric | What it tells you |
|---|---|
| `RMSE` | Typical prediction error, in price units (same scale as oil price, e.g. $9.94 means predictions are typically off by ~$9.94). Penalizes large misses more heavily. |
| `MAE` | Average absolute error in price units — easier to read than RMSE, less sensitive to outliers. |
| `R2` | Fraction of variance explained, 1.0 = perfect. Negative means the model does worse than just predicting the average price — worth investigating if you see this. |

**`reports/test_predictions_h{horizon}.csv`** — every prediction the model made,
one row per `forecast_origin` (the month the forecast was made *from*):

| Column | Meaning |
|---|---|
| `forecast_origin` | the month whose data was used to make the prediction |
| `target_date` | the month being predicted (`forecast_origin` + `horizon` months) |
| `predicted_price` | the model's forecast for `target_date` |
| `is_forecast` | `True` only for the last `horizon` row(s) — see below |

**Why `is_forecast` matters:** most rows in this file are predictions for months
we already have actual prices for, which is how we measure accuracy. But the
last `horizon` row(s) are different — their `target_date` is *beyond* the data
in `crude-oil-price.csv`, so there's no actual price to compare against yet.
Those are the genuine, currently-unknown forecast(s) the model is producing.
For `--horizon 1`, that's the single last row; for `--horizon 3`, it's the last
3 rows.

**`figures/test_predictions_h{horizon}.png`** — actual vs. predicted price on
the test set, so you can visually sanity-check how well the winning model
tracks reality (systematic lag, whether it captures turning points, etc.)
before trusting its forecast.
