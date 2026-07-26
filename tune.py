import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
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


def tune_xgb_regressor(
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
    Tune XGBRegressor using time-series cross-validation
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
            "n_estimators": [100, 200, 300, 500, 700],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_weight": [1, 3, 5, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        # RandomizedSearchCV handles outer parallelization
        n_jobs=1,
    )

    # Hyperparameter tuning
    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    xgb_search.fit(X_train, y_train)

    # Best fitted model
    best_model = xgb_search.best_estimator_

    # Validation predictions
    val_predictions = best_model.predict(X_val)

    # Preserve validation dates
    val_predictions = pd.Series(
        val_predictions, index=y_val.index, name="xgb_prediction"
    )

    # Validation metrics
    val_metrics = evaluate_model(y_val, val_predictions)

    cv_rmse = -xgb_search.best_score_

    print("Best XGBoost parameters:")
    print(xgb_search.best_params_)

    print(f"\nCross-validation RMSE: {cv_rmse:.4f}")

    print("\nValidation metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")

    return {
        "model": best_model,
        "search": xgb_search,
        "best_params": xgb_search.best_params_,
        "cv_rmse": cv_rmse,
        "val_predictions": val_predictions,
        "val_metrics": val_metrics,
    }


def model_comparison(ridge_results, xgb_results):
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
            "Model": "XGBRegressor",
            "CV_RMSE": xgb_results["cv_rmse"],
            **xgb_results["val_metrics"],
            "Best_Parameters": str(
                xgb_results["best_params"]
            )
        }
    ])

    model_comparison = (
        model_comparison
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return model_comparison

def best_model(model_comparison, ridge_results, xgb_results, X_train, y_train, X_val, y_val, X_test, y_test):
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

    elif best_model_name == "XGBRegressor":
        final_model = clone(xgb_results["model"])

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