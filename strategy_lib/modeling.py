import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ml_dataset(df: pd.DataFrame, feature_cols: list[str], hold_days: int, threshold: float):
    working = df.copy()
    working["fwd_return"] = working["close"].pct_change(hold_days).shift(-hold_days)
    working["label"] = (working["fwd_return"] > threshold).astype(int)

    available_cols = [column for column in feature_cols if column in working.columns]
    data = working[available_cols + ["label", "fwd_return"]].dropna()

    x_data = data[available_cols]
    y_data = data["label"]
    dates = data.index
    fwd_return = data["fwd_return"]
    return x_data, y_data, dates, fwd_return, available_cols


def build_all_ml_datasets(full_data: dict, stocks: list[str], hold_periods: dict[str, int], threshold: float, feature_cols: list[str]):
    datasets = {}
    for ticker in stocks:
        datasets[ticker] = {}
        for period_name, hold_days in hold_periods.items():
            x_data, y_data, dates, fwd_return, available_cols = build_ml_dataset(
                full_data[ticker], feature_cols, hold_days, threshold
            )
            datasets[ticker][period_name] = {
                "X": x_data,
                "y": y_data,
                "dates": dates,
                "fwd_return": fwd_return,
                "feature_cols": available_cols,
            }
    return datasets


def train_evaluate_lr(ml_datasets: dict, ticker: str, period: str, n_splits: int = 5) -> dict:
    dataset = ml_datasets[ticker][period]
    x_values = dataset["X"].values
    y_values = dataset["y"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x_values), start=1):
        x_train, x_test = x_values[train_idx], x_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_prob = pipeline.predict_proba(x_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        fold_results.append({"fold": fold_index, "auc": auc, "acc": (y_pred == y_test).mean()})
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    pipeline.fit(x_values, y_values)

    return {
        "model": pipeline,
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "y_prob": np.array(y_prob_all),
        "fold_df": pd.DataFrame(fold_results),
        "feature_cols": dataset["feature_cols"],
    }


def train_all_models(ml_datasets: dict, stocks: list[str], hold_periods: dict[str, int], n_splits: int = 5):
    trained_models = {}
    for ticker in stocks:
        trained_models[ticker] = {}
        for period in hold_periods:
            trained_models[ticker][period] = train_evaluate_lr(
                ml_datasets, ticker, period, n_splits=n_splits
            )
    return trained_models


def generate_signal_series(ml_datasets: dict, trained_models: dict, full_data: dict, stocks: list[str], target_hold: str):
    signal_series = {}
    for ticker in stocks:
        dataset = ml_datasets[ticker][target_hold]
        model = trained_models[ticker][target_hold]["model"]
        x_values = dataset["X"].values
        dates = dataset["dates"]

        prob_buy = model.predict_proba(x_values)[:, 1]
        signal = (prob_buy > 0.5).astype(int)

        signal_series[ticker] = pd.DataFrame(
            {
                "signal": signal,
                "prob_buy": prob_buy,
                "close": full_data[ticker].loc[dates, "close"],
            },
            index=dates,
        )
    return signal_series
