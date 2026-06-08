"""
Forecast accuracy metrics.

The headline number throughout this project is WMAPE-based accuracy:

    accuracy = 100 - WMAPE      (higher is better, capped at 100)

WMAPE (weighted mean absolute percentage error) divides the total
absolute error by the total actual sales. Compared to plain MAPE it
doesn't blow up when an individual period is near zero, and it naturally
weights busy periods more heavily, which is what a planner actually
cares about. It's the standard metric in demand forecasting.
"""

import numpy as np


def _arr(y_true, y_pred):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def wmape(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true, y_pred)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return 100.0 * np.abs(y_true - y_pred).sum() / denom


def accuracy(y_true, y_pred) -> float:
    score = 100.0 - wmape(y_true, y_pred)
    return max(0.0, score)  # don't report negative "accuracy"


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true, y_pred)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true, y_pred)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def summary(y_true, y_pred) -> dict:
    """Everything at once, rounded for printing/JSON."""
    return {
        "accuracy": round(accuracy(y_true, y_pred), 2),
        "wmape": round(wmape(y_true, y_pred), 2),
        "mae": round(mae(y_true, y_pred), 2),
        "rmse": round(rmse(y_true, y_pred), 2),
    }
