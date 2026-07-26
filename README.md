# Crude Oil Price Forecasting Pipeline

A small ML pipeline that trains a model to forecast crude oil prices `h` months
ahead, compares Ridge vs. XGBoost, saves the best model, and immediately runs
inference with it.

## Folder structure

```
app/
├── data/
│   └── crude-oil-price.csv     # raw input data (date, price, percentChange, change)
├── artifacts/
│   └── best_model_h{horizon}.pkl   # pickled best model, one file per trained horizon
├── figures/
│   └── test_predictions_h{horizon}.png  # actual vs. predicted plot on the test set
├── reports/
│   ├── model_comparison.csv    # Ridge vs. XGBoost: CV score + validation metrics + best params
│   ├── test_metrics_h{horizon}.csv     # test-set RMSE/MAE/R2 for the chosen model
│   └── test_predictions_h{horizon}.csv # every prediction, with the true forecast row(s) flagged
│
├── preprocessing.py         # data_cleaning(): load CSV, drop bad rows, parse dates
├── feature_engineering.py   # create_horizon_dataset(): lags, rolling stats, target shifting
├── split.py                 # time-based train/validation/test split + TimeSeriesSplit for CV
├── tune.py                  # hyperparameter search for Ridge and XGBoost, model_comparison(), best_model()
├── evaluate.py               # evaluate_model() metrics, save_metrics() to reports/
├── plots.py                  # show_plot(): draws + saves the actual-vs-predicted chart
├── model_io.py               # save_model() / load_model() — pickle persistence
├── inference.py               # build features for new data, predict(), save_predictions()
└── pipeline.py               # entry point — wires everything together, has the CLI
```

**Flow at a glance:** `preprocessing` → `feature_engineering` → `split` → `tune`
(trains + compares Ridge and XGBoost) → `evaluate`/`plots` (score + visualize on
the test set) → `model_io` (pickle the winner) → `inference` (forecast with it) →
results land in `reports/` and `figures/`. `pipeline.py` is the only file you run
directly; everything else is a module it imports.

## Environment setup

Requires **Python 3.10+**. There's no `requirements.txt` in this repo yet — for
now, install the dependencies manually:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

If you set up a `requirements.txt`, freeze exactly these:

```
pandas
numpy
scikit-learn
xgboost
matplotlib
```

Recommended: use a virtual environment before installing.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost matplotlib
```

## How to test the model

All commands are run from inside the `app/` folder.

### 1. Train + forecast (the default flow)

```bash
python pipeline.py --horizon 1
```

This does everything in one go:
1. Builds lag/rolling-window features and does a chronological train/validation/test split.
2. Tunes Ridge (grid search) and XGBoost (randomized search) with time-series cross-validation.
3. Picks the model with the lowest validation RMSE, refits it on train+validation, and scores it on the held-out test set.
4. Saves the plot, the metrics, the pickled model, and the predictions.

**What `--horizon` means:** the number of months ahead the model predicts. `--horizon 1` trains a model that predicts next month's price from this month's data; `--horizon 3` trains a model that predicts 3 months ahead. Each horizon gets its own model (`artifacts/best_model_h{horizon}.pkl`) and its own report files — training horizon 3 does not overwrite horizon 1's results.

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

**`reports/model_comparison.csv`** — one row per candidate model (Ridge, XGBoost),
ranked by test RMSE. Shows `CV_RMSE` (cross-validation score during tuning),
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
