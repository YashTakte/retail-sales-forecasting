"""
Rolling-window backtesting.

A single train/test split can be lucky or unlucky depending on which
months land in the test set. Backtesting slides the split backwards
several times and averages the accuracy, which gives a number you can
actually trust (and defend in an interview).

For each window we train on everything before the test slice and predict
the next `horizon` periods — never peeking at the future.

Run:  python src/backtest.py
"""

import numpy as np
import pandas as pd

import config
import metrics
from model_prophet import ProphetForecaster
from model_lightgbm import LightGBMForecaster


def backtest_prophet(df, group, target, horizon, n_windows):
    """Average accuracy of per-series Prophet models over rolling windows."""
    window_scores = []
    for k in range(n_windows):
        y_true_all, y_pred_all = [], []
        series = df.groupby(group) if group else [(None, df)]
        for _, g in series:
            g = g.sort_values("period")
            end = len(g) - k
            train, test = g.iloc[:end - horizon], g.iloc[end - horizon:end]
            if (train[target] > 0).sum() < 12:
                continue
            m = ProphetForecaster().fit(train["period"], train[target])
            y_true_all.extend(test[target].values)
            y_pred_all.extend(m.predict(test["period"]).values)
        if y_true_all:
            window_scores.append(metrics.accuracy(y_true_all, y_pred_all))
    return float(np.mean(window_scores)), window_scores


def backtest_lightgbm(df, group, target, horizon, n_windows):
    """Average accuracy of the global LightGBM model over rolling windows."""
    model = LightGBMForecaster(group=group, target=target)
    feat = model.build_features(df)
    periods = np.sort(feat["period"].unique())

    window_scores = []
    for k in range(n_windows):
        if len(periods) - k - horizon < 18:
            break
        test_periods = periods[len(periods) - k - horizon: len(periods) - k]
        cutoff = test_periods[0]

        train = feat[feat["period"] < cutoff].dropna(subset=model.features)
        test = feat[feat["period"].isin(test_periods)].copy()
        if train.empty or test.empty:
            continue

        model.model.fit(train[model.features], train[target])
        test["y_pred"] = model.predict(test).values
        window_scores.append(metrics.accuracy(test[target], test["y_pred"]))
    return float(np.mean(window_scores)), window_scores


def run(horizon: int = 6, n_windows: int = 6):
    total = pd.read_csv(config.PROCESSED_DIR / "monthly_total.csv", parse_dates=["period"])
    cat = pd.read_csv(config.PROCESSED_DIR / "monthly_category.csv", parse_dates=["period"])
    sub = pd.read_csv(config.PROCESSED_DIR / "monthly_subcategory.csv", parse_dates=["period"])

    print(f"Rolling backtest  |  horizon={horizon} months  |  windows={n_windows}\n")

    rows = [
        ("total revenue   (Prophet) ", *backtest_prophet(total, None, "sales", horizon, n_windows)),
        ("category revenue (LightGBM)", *backtest_lightgbm(cat, "category", "sales", horizon, n_windows)),
        ("subcat units    (LightGBM)", *backtest_lightgbm(sub, "subcategory", "quantity", horizon, n_windows)),
    ]
    for label, mean, scores in rows:
        pretty = ", ".join(f"{s:.0f}" for s in scores)
        print(f"  {label}:  {mean:5.1f}%   (windows: {pretty})")
    return rows


if __name__ == "__main__":
    run()
