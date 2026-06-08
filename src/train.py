"""
Train and evaluate the forecasters, then save the artifacts the API serves.

We forecast at three levels of detail ("tiers"), because the data supports
very different accuracy at each — and being honest about that is the whole
point:

  * total       — all sales, monthly. Smoothest series, best accuracy.
  * category    — the 3 product categories, monthly.
  * subcategory — the 17 sub-categories. Revenue here is noisy (a single
                  big order swings a month), so we ALSO model unit demand,
                  which is much steadier and more useful for planning.

Evaluation is a plain time-based holdout: train on everything up to a
cutoff, test on the last N periods. No shuffling — you can't shuffle time.

Run:  python src/train.py
"""

import json
import pickle

import pandas as pd

import config
import metrics
from model_prophet import ProphetForecaster
from model_lightgbm import LightGBMForecaster


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DIR / f"{name}.csv", parse_dates=["period"])
    return df


def time_split(df: pd.DataFrame, group: str | None, horizon: int):
    """Last `horizon` periods become the test set, the rest is training."""
    periods = df["period"].sort_values().unique()
    cutoff = periods[-horizon]
    train = df[df["period"] < cutoff].copy()
    test = df[df["period"] >= cutoff].copy()
    return train, test, cutoff


# --------------------------------------------------------------------------
# Per-tier evaluation
# --------------------------------------------------------------------------
def eval_prophet(df, group, target, horizon):
    """Fit one Prophet per series and stitch the test predictions back."""
    preds = []
    series = df.groupby(group) if group else [(None, df)]
    for key, g in series:
        g = g.sort_values("period")
        train, test = g.iloc[:-horizon], g.iloc[-horizon:]
        if (train[target] > 0).sum() < 12:
            continue  # too little history to fit a yearly model
        model = ProphetForecaster()
        model.fit(train["period"], train[target])
        yhat = model.predict(test["period"])
        preds.append(pd.DataFrame({
            "key": key, "period": test["period"].values,
            "y_true": test[target].values, "y_pred": yhat.values,
        }))
    return pd.concat(preds, ignore_index=True)


def eval_lightgbm(df, group, target, horizon):
    """One global LightGBM. Featurise train+test together so lags line up."""
    model = LightGBMForecaster(group=group, target=target)
    feat = model.build_features(df)

    periods = feat["period"].sort_values().unique()
    cutoff = periods[-horizon]
    train = feat[feat["period"] < cutoff].dropna(subset=model.features)
    test = feat[feat["period"] >= cutoff].copy()

    model.model.fit(train[model.features], train[target])
    test["y_pred"] = model.predict(test).values

    return pd.DataFrame({
        "key": test[group] if group else None,
        "period": test["period"].values,
        "y_true": test[target].values,
        "y_pred": test["y_pred"].values,
    }), model


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run():
    report = {}
    artifacts = {}

    # ---- Tier 1: total monthly sales -------------------------------------
    total = _load("monthly_total")
    p = eval_prophet(total, None, "sales", config.DEFAULT_TEST_MONTHS)
    l, l_total_model = eval_lightgbm(total, None, "sales", config.DEFAULT_TEST_MONTHS)
    report["total_monthly_sales"] = {
        "prophet": metrics.summary(p.y_true, p.y_pred),
        "lightgbm": metrics.summary(l.y_true, l.y_pred),
    }

    # ---- Tier 2: per-category monthly sales ------------------------------
    cat = _load("monthly_category")
    p = eval_prophet(cat, "category", "sales", config.DEFAULT_TEST_MONTHS)
    l, _ = eval_lightgbm(cat, "category", "sales", config.DEFAULT_TEST_MONTHS)
    report["category_monthly_sales"] = {
        "prophet": metrics.summary(p.y_true, p.y_pred),
        "lightgbm": metrics.summary(l.y_true, l.y_pred),
    }

    # ---- Tier 3: per-subcategory monthly UNITS ---------------------------
    # Revenue is too spiky per subcategory; unit demand forecasts far
    # better and is what you'd actually use for stocking decisions.
    sub = _load("monthly_subcategory")
    p = eval_prophet(sub, "subcategory", "quantity", config.DEFAULT_TEST_MONTHS)
    l, l_sub_model = eval_lightgbm(sub, "subcategory", "quantity", config.DEFAULT_TEST_MONTHS)
    report["subcategory_monthly_units"] = {
        "prophet": metrics.summary(p.y_true, p.y_pred),
        "lightgbm": metrics.summary(l.y_true, l.y_pred),
        "lightgbm_per_subcategory": (
            l.groupby("key")
            .apply(lambda g: metrics.accuracy(g.y_true, g.y_pred), include_groups=False)
            .round(2).sort_values(ascending=False).to_dict()
        ),
    }

    # ---- Persist artifacts the API needs ---------------------------------
    # Retrain the "production" models on ALL data (no holdout) so the saved
    # models use every observation available.
    #
    # Total tier -> Prophet. With only 48 months, LightGBM is starved of
    # rows (the 12-month lag alone costs a year of data) and collapses to a
    # near-constant prediction, so Prophet is both more accurate AND gives
    # a properly seasonal forward curve here.
    total_model = ProphetForecaster(seasonality_mode="multiplicative")
    total_model.fit(total["period"], total["sales"])

    # Subcategory tier -> LightGBM. Pooling all 17 series gives it plenty of
    # rows, and it beats Prophet on units at this grain. We include discount
    # as a feature so the app can run "what-if" discount scenarios — testing
    # showed it doesn't change accuracy much, but it makes the model usable
    # for promotion planning, which is the point.
    sub_full = LightGBMForecaster("subcategory", "quantity", include_discount=True)
    sub_full.fit(sub)

    # Category tier -> LightGBM on revenue. Pooling the 3 categories gives it
    # enough rows to produce a proper seasonal forward curve.
    cat_full = LightGBMForecaster("category", "sales")
    cat_full.fit(cat)

    with open(config.MODEL_DIR / "prophet_total_sales.pkl", "wb") as f:
        pickle.dump(total_model, f)
    with open(config.MODEL_DIR / "lightgbm_subcategory_units.pkl", "wb") as f:
        pickle.dump(sub_full, f)
    with open(config.MODEL_DIR / "lightgbm_category_sales.pkl", "wb") as f:
        pickle.dump(cat_full, f)
    with open(config.MODEL_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def _print(report):
    print("\n" + "=" * 60)
    print("FORECAST ACCURACY  (accuracy = 100 - WMAPE)")
    print("=" * 60)
    for tier, models in report.items():
        print(f"\n{tier}")
        for mdl in ("prophet", "lightgbm"):
            s = models[mdl]
            print(f"  {mdl:9s}  acc {s['accuracy']:6.2f}%   "
                  f"wmape {s['wmape']:6.2f}   mae {s['mae']:.1f}")
    print("\nSaved models + metrics.json to models/")


if __name__ == "__main__":
    _print(run())
