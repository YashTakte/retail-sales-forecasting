"""
Inference helpers used by both the API and the Gradio app.

Training saves two LightGBM models (total revenue, and per-subcategory
units). To forecast the future we have to generate predictions one step
at a time: each new month's lag features depend on the months before it,
including the ones we just predicted. This file hides that recursion
behind two simple functions.
"""

import pickle

import pandas as pd

import config


def _load_model(filename: str):
    with open(config.MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)


def _history(table: str, group_value: str | None, group_col: str | None, target: str):
    """Pull the historical series we extend into the future."""
    df = pd.read_csv(config.PROCESSED_DIR / f"{table}.csv", parse_dates=["period"])
    if group_col and group_value:
        df = df[df[group_col] == group_value]
    cols = ["period", target] + ([group_col] if group_col else [])
    return df[cols].sort_values("period").reset_index(drop=True)


def _forecast_recursive(model, history, target, group_col, group_value, months,
                        discount=None, round_int=False):
    """
    Walk forward `months` steps, feeding each prediction back in so the
    next step's lag/rolling features are available.

    `discount`: if the model uses discount as a feature, this is the value
    to assume for the future months. None means "carry the series' recent
    average" (a neutral baseline); a number sets a what-if scenario.
    `round_int`: round the forecast to a whole number (units can't be
    fractional).
    """
    series = history.copy()
    results = []
    last = series["period"].max()

    # Default future discount = recent average, so the baseline forecast
    # reflects how this series usually trades.
    if discount is None and "discount" in series.columns:
        discount = float(series["discount"].tail(12).mean())

    for _ in range(months):
        next_period = (last + pd.offsets.MonthBegin(1))
        row = {"period": next_period, target: 0.0}
        if group_col:
            row[group_col] = group_value
        if "discount" in series.columns:
            row["discount"] = discount
        series = pd.concat([series, pd.DataFrame([row])], ignore_index=True)

        feat = model.build_features(series)
        pred = float(model.predict(feat.iloc[[-1]]).iloc[0])

        series.loc[series.index[-1], target] = pred
        value = max(0, round(pred)) if round_int else round(pred, 2)
        results.append({"period": next_period.date().isoformat(), "forecast": value})
        last = next_period

    return results


def forecast_total_revenue(months: int = 6):
    # Prophet builds its own future timeline, so no recursion needed here.
    model = _load_model("prophet_total_sales.pkl")
    history = _history("monthly_total", None, None, "sales")
    last = history["period"].max()
    future = pd.date_range(last + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    bounds = model.predict_with_bounds(pd.Series(future))
    return [
        {
            "period": d.date().isoformat(),
            "forecast": round(float(r.yhat), 2),
            "lower": round(float(r.yhat_lower), 2),
            "upper": round(float(r.yhat_upper), 2),
        }
        for d, (_, r) in zip(future, bounds.iterrows())
    ]


def forecast_subcategory_units(subcategory: str, months: int = 6, discount: float | None = None):
    """
    Forecast monthly unit demand for one subcategory.

    `discount`: optional what-if. Leave as None for the baseline (recent
    average discount); pass e.g. 0.30 to ask "what if we run this at 30%
    off?". The model learned the discount-demand relationship from history.
    """
    model = _load_model("lightgbm_subcategory_units.pkl")
    full = pd.read_csv(config.PROCESSED_DIR / "monthly_subcategory.csv", parse_dates=["period"])
    if subcategory not in set(full["subcategory"]):
        raise ValueError(f"unknown subcategory: {subcategory}")

    cols = ["period", "subcategory", "quantity"]
    if "discount" in full.columns:
        cols.append("discount")
    history = (
        full[full["subcategory"] == subcategory][cols]
        .sort_values("period").reset_index(drop=True)
    )
    # Units are whole numbers — you can't sell 60.3 phones.
    return _forecast_recursive(model, history, "quantity", "subcategory",
                               subcategory, months, discount=discount, round_int=True)


def forecast_category_sales(category: str, months: int = 6):
    """Forecast monthly revenue for one of the 3 product categories."""
    model = _load_model("lightgbm_category_sales.pkl")
    full = pd.read_csv(config.PROCESSED_DIR / "monthly_category.csv", parse_dates=["period"])
    if category not in set(full["category"]):
        raise ValueError(f"unknown category: {category}")
    history = (
        full[full["category"] == category][["period", "category", "sales"]]
        .sort_values("period").reset_index(drop=True)
    )
    return _forecast_recursive(model, history, "sales", "category", category, months)


def available_categories() -> list[str]:
    full = pd.read_csv(config.PROCESSED_DIR / "monthly_category.csv")
    return sorted(full["category"].unique().tolist())


def available_subcategories() -> list[str]:
    full = pd.read_csv(config.PROCESSED_DIR / "monthly_subcategory.csv")
    return sorted(full["subcategory"].unique().tolist())
