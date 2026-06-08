"""
Feature engineering for the LightGBM model.

Prophet reads a raw date+value series and works out trend and seasonality
on its own. LightGBM can't do that, it only sees the columns we give it,
so here we turn a date into something a tree model can split on:

  * calendar parts (month, quarter, week-of-year)
  * cyclical encodings, so December and January sit next to each other
  * lag features (what were sales N periods ago)
  * rolling averages (the recent trend), carefully shifted so a row never
    gets to peek at its own value.

These work for both the weekly and monthly tables; the lag list is just
chosen to make sense at whichever grain you pass in.
"""

import numpy as np
import pandas as pd

# Lags/windows in *periods*. At a monthly grain, 12 == one year.
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]


def add_calendar(df: pd.DataFrame, date_col: str = "period") -> pd.DataFrame:
    d = df[date_col].dt
    df = df.copy()
    df["year"] = d.year
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["weekofyear"] = d.isocalendar().week.astype(int)
    # Cyclical month so the model knows month 12 and month 1 are neighbours.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lags(df: pd.DataFrame, group: str | None, target: str = "sales") -> pd.DataFrame:
    """
    Add lag and rolling-mean features.

    `group` is the column that separates one series from another (e.g.
    'subcategory'). Pass None when the table is a single series (total).
    """
    df = df.sort_values(([group] if group else []) + ["period"]).copy()
    grouped = df.groupby(group)[target] if group else df[target]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    for w in ROLL_WINDOWS:
        # shift(1) before rolling so the current row is never part of its
        # own average — that would leak the answer into the features.
        base = grouped.shift(1)
        roll = base.rolling(w).mean()
        df[f"rollmean_{w}"] = roll.reset_index(level=0, drop=True) if group else roll
    return df


def build_features(df: pd.DataFrame, group: str | None, target: str = "sales") -> pd.DataFrame:
    df = add_calendar(df)
    df = add_lags(df, group, target)
    if group:
        df[f"{group}_code"] = df[group].astype("category").cat.codes
    return df


def feature_columns(group: str | None, include_discount: bool = False) -> list[str]:
    cols = [
        "year", "month", "quarter", "weekofyear",
        "month_sin", "month_cos",
        *[f"lag_{l}" for l in LAGS],
        *[f"rollmean_{w}" for w in ROLL_WINDOWS],
    ]
    if group:
        cols.append(f"{group}_code")
    if include_discount:
        cols.append("discount")
    return cols
