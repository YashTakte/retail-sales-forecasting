"""
Data pipeline: raw CSVs -> clean, aggregated tables ready for modelling.

The raw data is the classic Superstore export split across four files.
This module does the unglamorous-but-essential work:

  1. read the four files (minding their separators and encodings)
  2. parse dates and join order lines to their product category
  3. roll the line-level orders up to weekly and monthly series at three
     levels of detail: total, per-category, and per-subcategory.

Run it directly to (re)build everything under data/processed/.
"""

import pandas as pd

import config


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def _read(name: str, encoding: str) -> pd.DataFrame:
    """Thin wrapper so every read uses the right separator/encoding."""
    return pd.read_csv(config.RAW_DIR / name, sep=";", encoding=encoding)


def load_raw() -> dict[str, pd.DataFrame]:
    return {key: _read(fname, enc) for key, (fname, enc) in config.RAW_FILES.items()}


# --------------------------------------------------------------------------
# 2. Join order lines to product info
# --------------------------------------------------------------------------
def build_order_lines() -> pd.DataFrame:
    """One tidy row per order line, with its category and sub-category."""
    raw = load_raw()
    orders, products = raw["orders"].copy(), raw["products"]

    # Dates come in as DD/MM/YYYY. Fail loudly if anything doesn't parse,
    # rather than silently dropping revenue.
    orders["Order Date"] = pd.to_datetime(
        orders["Order Date"], format="%d/%m/%Y", errors="coerce"
    )
    bad = orders["Order Date"].isna().sum()
    if bad:
        raise ValueError(f"{bad} orders had dates that wouldn't parse")

    # Products.csv lists some IDs more than once (same product, different
    # name spellings) but the category never disagrees, so we can safely
    # keep one row per ID before joining.
    lookup = products[["Product ID", "Category", "Sub-Category"]].drop_duplicates("Product ID")

    df = orders.merge(lookup, on="Product ID", how="left", validate="many_to_one")
    missing = df["Sub-Category"].isna().sum()
    if missing:
        raise ValueError(f"{missing} order lines didn't match any product")

    # Rename to tidy snake_case and keep only what the models need.
    return (
        df.rename(columns={
            "Order Date": "date",
            "Category": "category",
            "Sub-Category": "subcategory",
            "Sales": "sales",
            "Quantity": "quantity",
            "Discount": "discount",
        })[["date", "category", "subcategory", "sales", "quantity", "discount", "Order ID"]]
    )


# --------------------------------------------------------------------------
# 3. Aggregate to time series
# --------------------------------------------------------------------------
def _fill_calendar(df: pd.DataFrame, group_cols: list[str], freq: str) -> pd.DataFrame:
    """
    Make every group span the full date range with no gaps.

    Retail series have weeks/months with zero sales for a given
    subcategory. Models need a continuous index, so we build the full
    grid and fill the holes with zeros.
    """
    periods = pd.date_range(df["period"].min(), df["period"].max(), freq=freq)
    keys = df[group_cols].drop_duplicates()

    grid = keys.merge(pd.Series(periods, name="period"), how="cross")
    out = grid.merge(df, on=group_cols + ["period"], how="left")
    for col in ("sales", "quantity", "orders", "discount"):
        if col in out:
            out[col] = out[col].fillna(0)
    return out.sort_values(group_cols + ["period"]).reset_index(drop=True)


def _aggregate(lines: pd.DataFrame, level: str, freq: str) -> pd.DataFrame:
    """
    Roll order lines up to a time series.

    level: 'total' | 'category' | 'subcategory'
    freq:  a pandas offset alias, e.g. 'W-MON' or 'MS' (month start)
    """
    df = lines.copy()
    df["period"] = df["date"].dt.to_period(
        "W-SUN" if freq.startswith("W") else "M"
    ).dt.start_time

    if level == "total":
        group_cols = []
    elif level == "category":
        group_cols = ["category"]
    elif level == "subcategory":
        group_cols = ["category", "subcategory"]
    else:
        raise ValueError(f"unknown level: {level}")

    agg = (
        df.groupby(group_cols + ["period"], as_index=False)
        .agg(sales=("sales", "sum"),
             quantity=("quantity", "sum"),
             orders=("Order ID", "nunique"),
             discount=("discount", "mean"))
    )

    # Total level has no group key, so the calendar fill is trivial.
    if not group_cols:
        full = pd.date_range(agg["period"].min(), agg["period"].max(), freq=freq)
        agg = (pd.DataFrame({"period": full})
               .merge(agg, on="period", how="left")
               .fillna({"sales": 0, "quantity": 0, "orders": 0, "discount": 0})
               .sort_values("period").reset_index(drop=True))
        return agg

    return _fill_calendar(agg, group_cols, freq)


def build_all() -> dict[str, pd.DataFrame]:
    """Build every table we serve or model, and cache to data/processed/."""
    lines = build_order_lines()

    tables = {
        "weekly_subcategory": _aggregate(lines, "subcategory", config.WEEK_ANCHOR),
        "monthly_subcategory": _aggregate(lines, "subcategory", "MS"),
        "monthly_category": _aggregate(lines, "category", "MS"),
        "monthly_total": _aggregate(lines, "total", "MS"),
    }
    for name, tbl in tables.items():
        tbl.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    lines = build_order_lines()
    print(f"Order lines : {len(lines):,}")
    print(f"Date range  : {lines.date.min().date()} -> {lines.date.max().date()}")
    print(f"Total sales : ${lines.sales.sum():,.2f}")
    print(f"Subcats     : {lines.subcategory.nunique()}\n")

    tables = build_all()
    for name, tbl in tables.items():
        print(f"  {name:22s} {tbl.shape[0]:>5} rows  ->  data/processed/{name}.csv")
