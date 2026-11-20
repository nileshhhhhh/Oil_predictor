import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import evaluate_model

def tune_ridge_regressor(X_train, y_train, X_val, y_val, cv, alpha_values=None):
    """
    Tune Ridge regression using time-series cross-validation
    and evaluate the best model on the validation set.

    Returns
    -------
    dict containing:
        model
        search
        best_params
        cv_rmse
        val_predictions
        val_metrics
    """

    if alpha_values is None:
        alpha_values = np.logspace(-4, 4, 50)

    # Scaling and Ridge model
    ridge_pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

    parameter_grid = {"model__alpha": alpha_values}

    # Hyperparameter tuning
    ridge_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    ridge_search.fit(X_train, y_train)

    # Best fitted model
    best_model = ridge_search.best_estimator_

    # Validation predictions
    val_predictions = best_model.predict(X_val)

    # Preserve validation dates
    val_predictions = pd.Series(
        val_predictions, index=y_val.index, name="ridge_prediction"
    )

    # Validation metrics
    val_metrics = evaluate_model(y_val, val_predictions)

    cv_rmse = -ridge_search.best_score_

    print("Best Ridge parameters:")
    print(ridge_search.best_params_)

    print(f"\nCross-validation RMSE: {cv_rmse:.4f}")

    print("\nValidation metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    return {
        "model": best_model,
        "search": ridge_search,
        "best_params": ridge_search.best_params_,
        "cv_rmse": cv_rmse,
        "val_predictions": val_predictions,
        "val_metrics": val_metrics,
    }


def tune_hist_gradient_boosting_regressor(
    X_train,
    y_train,
    X_val,
    y_val,
    cv,
    param_distributions=None,
    n_iter=40,
    random_state=42,
):
    """
    Tune HistGradientBoostingRegressor using time-series cross-validation
    and evaluate the best model on the validation set.

    Returns
    -------
    dict containing:
        model
        search
        best_params
        cv_rmse
        val_predictions
        val_metrics
    """

    if param_distributions is None:
        param_distributions = {
            "max_iter": [100, 200, 300, 500, 700],
            "max_leaf_nodes": [7, 15, 31],
            "max_depth": [None, 2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_samples_leaf": [5, 10, 20, 30],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
        }

    hist_model = HistGradientBoostingRegressor(
        loss="squared_error",
        # Keep validation chronological through the existing CV splits.
        early_stopping=False,
        random_state=random_state,
    )

    # Hyperparameter tuning
    hist_search = RandomizedSearchCV(
        estimator=hist_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    hist_search.fit(X_train, y_train)

    # Best fitted model
    best_model = hist_search.best_estimator_

    # Validation predictions
    val_predictions = best_model.predict(X_val)

    # Preserve validation dates
    val_predictions = pd.Series(
        val_predictions, index=y_val.index, name="hist_prediction"
    )

    # Validation metrics
    val_metrics = evaluate_model(y_val, val_predictions)

    cv_rmse = -hist_search.best_score_

    print("Best HistGradientBoosting parameters:")
    print(hist_search.best_params_)

    print(f"\nCross-validation RMSE: {cv_rmse:.4f}")

    print("\nValidation metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    return {
        "model": best_model,
        "search": hist_search,
        "best_params": hist_search.best_params_,
        "cv_rmse": cv_rmse,
        "val_predictions": val_predictions,
        "val_metrics": val_metrics,
    }


def model_comparison(ridge_results, hist_results):
    model_comparison = pd.DataFrame([
        {
            "Model": "Ridge",
            "CV_RMSE": ridge_results["cv_rmse"],
            **ridge_results["val_metrics"],
            "Best_Parameters": str(
                ridge_results["best_params"]
            )
        },
        {
            "Model": "HistGradientBoostingRegressor",
            "CV_RMSE": hist_results["cv_rmse"],
            **hist_results["val_metrics"],
            "Best_Parameters": str(
                hist_results["best_params"]
            )
        }
    ])

    model_comparison = (
        model_comparison
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return model_comparison

def best_model(model_comparison, ridge_results, hist_results, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Refit the overall best model (picked from model_comparison) on
    train+validation data, and score it on the held-out test set.

    Returns
    -------
    final_model : the refit estimator (this is what gets pickled)
    test_metrics : dict of RMSE/MAE/R2 on the test set
    test_predictions : pd.Series of test-set predictions
    """
    best_model_name = model_comparison.loc[0, "Model"]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    if best_model_name == "Ridge":
        final_model = clone(ridge_results["model"])

    elif best_model_name == "HistGradientBoostingRegressor":
        final_model = clone(hist_results["model"])

    else:
        raise ValueError("Unknown model selected.")

    final_model.fit(X_train_val, y_train_val)

    test_predictions = pd.Series(
        final_model.predict(X_test),
        index=y_test.index
    )

    test_metrics = evaluate_model(
        y_test,
        test_predictions
    )

    return final_model, test_metrics, test_predictions
