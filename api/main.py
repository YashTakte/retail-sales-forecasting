"""
FastAPI service exposing the trained forecasters over HTTP.

Two prediction endpoints plus a metrics endpoint:

  GET /health                         -> liveness check
  GET /metrics                        -> backtested accuracy per tier
  GET /forecast/total?months=6        -> total monthly revenue forecast
  GET /forecast/subcategory/{name}    -> monthly unit-demand forecast
  GET /subcategories                  -> list of valid subcategory names

Start locally:
    uvicorn api.main:app --reload --app-dir .
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

# Make src/ importable whether we run from the repo root or inside Docker.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config                       # noqa: E402
import forecast                     # noqa: E402
from model_prophet import ProphetForecaster  # noqa: E402,F401  (needed for unpickling)

app = FastAPI(
    title="Retail Sales Forecasting API",
    description="Prophet + LightGBM demand forecasts over the Superstore dataset.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/subcategories")
def subcategories():
    return {"subcategories": forecast.available_subcategories()}


@app.get("/categories")
def categories():
    return {"categories": forecast.available_categories()}


@app.get("/metrics")
def get_metrics():
    """Return the accuracy report written by train.py, if present."""
    path = config.MODEL_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found — run train.py first")
    return json.loads(path.read_text())


@app.get("/forecast/total")
def forecast_total(months: int = Query(6, ge=1, le=24)):
    return {
        "tier": "total_revenue",
        "unit": "USD",
        "months": months,
        "forecast": forecast.forecast_total_revenue(months),
    }


@app.get("/forecast/category/{name}")
def forecast_category(name: str, months: int = Query(6, ge=1, le=24)):
    try:
        result = forecast.forecast_category_sales(name, months)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {
        "tier": "category_revenue",
        "category": name,
        "unit": "USD",
        "months": months,
        "forecast": result,
    }


@app.get("/forecast/subcategory/{name}")
def forecast_subcategory(
    name: str,
    months: int = Query(6, ge=1, le=24),
    discount: float = Query(None, ge=0, le=1,
                            description="What-if average discount, 0-1 (e.g. 0.3 = 30% off). Omit for baseline."),
):
    try:
        result = forecast.forecast_subcategory_units(name, months, discount=discount)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {
        "tier": "subcategory_units",
        "subcategory": name,
        "unit": "units",
        "months": months,
        "discount_scenario": discount,
        "forecast": result,
    }
