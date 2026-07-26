import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_horizon_dataset(horizon_df, validation_months=24, test_months=24):
    data = horizon_df.sort_values("target_date").copy()

    target_dates = pd.DatetimeIndex(data["target_date"].unique()).sort_values()

    required_months = validation_months + test_months

    if len(target_dates) <= required_months:
        raise ValueError("There are not enough observations for the requested split.")

    test_start = target_dates[-test_months]
    validation_start = target_dates[-required_months]

    train_df = data[data["target_date"] < validation_start].copy()

    validation_df = data[
        (data["target_date"] >= validation_start) & (data["target_date"] < test_start)
    ].copy()

    test_df = data[data["target_date"] >= test_start].copy()

    return train_df, validation_df, test_df


def prepare_train_val_test(
    train_df,
    validation_df,
    test_df,
    target_column="target_price",
    columns_to_remove=("target_date", "target_price"),
):
    # Remove only columns that actually exist
    def split_X_y(dataset):
        drop_columns = [
            column for column in columns_to_remove if column in dataset.columns
        ]

        X = dataset.drop(columns=drop_columns).copy()
        y = dataset[target_column].copy()

        return X, y

    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(validation_df)
    X_test, y_test = split_X_y(test_df)

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def time_series_split(horizon):
    tscv = TimeSeriesSplit(n_splits=5, test_size=12, gap=max(horizon - 1, 0))
    return tscv
