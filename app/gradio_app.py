"""
Gradio interface for the forecasting models.

Two tabs:
  * Total revenue   — whole-store monthly revenue, forecast with Prophet,
                      shown with a confidence band (likely range).
  * Subcategory     — monthly unit demand for any of the 17 subcategories,
                      forecast with LightGBM, with a discount "what-if"
                      slider so you can see how a promotion would change
                      predicted demand.

Both charts are interactive (hover, zoom) and render on page load, so a
first-time visitor sees a result immediately. The UI calls the same
forecast helpers the API uses, so the two can never drift apart.

Run:  python app/gradio_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import gradio as gr

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config                       # noqa: E402
import forecast                     # noqa: E402
import reconcile                    # noqa: E402
from model_prophet import ProphetForecaster  # noqa: E402,F401  (unpickling)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _history(table, col=None, value=None):
    df = pd.read_csv(config.PROCESSED_DIR / f"{table}.csv", parse_dates=["period"])
    if col and value:
        df = df[df[col] == value]
    return df.sort_values("period")


def _accuracy(tier_key, model_key):
    """Pull a backtested-ish accuracy number from the saved metrics file."""
    try:
        report = json.loads((config.MODEL_DIR / "metrics.json").read_text())
        return report[tier_key][model_key]["accuracy"]
    except Exception:
        return None


def _base_chart(title, ylabel, value_prefix=""):
    fig = go.Figure()
    fig.update_layout(
        title=title, yaxis_title=ylabel, xaxis_title="Month",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60, r=20, b=40, l=70),
    )
    fig.update_xaxes(tickformat="%b %Y")
    fig._prefix = value_prefix  # stash for hovertemplate reuse
    return fig


# --------------------------------------------------------------------------
# Tab 1 — total revenue (Prophet, with confidence band)
# --------------------------------------------------------------------------
def predict_total(months):
    months = int(months)
    fc = forecast.forecast_total_revenue(months)
    hist = _history("monthly_total").tail(18)

    fc_dates = pd.to_datetime([r["period"] for r in fc])
    yhat = [r["forecast"] for r in fc]
    lower = [r["lower"] for r in fc]
    upper = [r["upper"] for r in fc]

    fig = _base_chart("Total Monthly Revenue Forecast (model: Prophet)",
                      "Revenue (USD)", "$")
    ht = "%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"

    # Confidence band: upper then lower with fill between.
    fig.add_trace(go.Scatter(x=fc_dates, y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_dates, y=lower, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(221,107,32,0.18)",
                             name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["period"], y=hist["sales"], name="History",
                             mode="lines", line=dict(color="#2b6cb0", width=2.5),
                             hovertemplate=ht))
    fig.add_trace(go.Scatter(x=fc_dates, y=yhat, name="Forecast",
                             mode="lines+markers",
                             line=dict(color="#dd6b20", width=2.5, dash="dash"),
                             marker=dict(size=7), hovertemplate=ht))

    table = pd.DataFrame(fc).rename(columns={
        "period": "Month", "forecast": "Forecast ($)",
        "lower": "Low ($)", "upper": "High ($)"})
    return fig, table


# --------------------------------------------------------------------------
# Tab 2 — category revenue (LightGBM)
# --------------------------------------------------------------------------
def predict_category(category, months):
    months = int(months)
    fc = forecast.forecast_category_sales(category, months)
    hist = _history("monthly_category", "category", category).tail(18)

    fc_dates = pd.to_datetime([r["period"] for r in fc])
    yhat = [r["forecast"] for r in fc]

    fig = _base_chart(
        f"{category} — Monthly Revenue Forecast (model: LightGBM)",
        "Revenue (USD)", "$")
    ht = "%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"
    fig.add_trace(go.Scatter(x=hist["period"], y=hist["sales"], name="History",
                             mode="lines", line=dict(color="#2b6cb0", width=2.5),
                             hovertemplate=ht))
    fig.add_trace(go.Scatter(x=fc_dates, y=yhat, name="Forecast",
                             mode="lines+markers",
                             line=dict(color="#dd6b20", width=2.5, dash="dash"),
                             marker=dict(size=7), hovertemplate=ht))

    table = pd.DataFrame(fc).rename(columns={
        "period": "Month", "forecast": "Forecast ($)"})
    return fig, table


# --------------------------------------------------------------------------
# Tab 3 — subcategory units (LightGBM, with discount what-if)
# --------------------------------------------------------------------------
def predict_subcat(subcategory, months, discount_pct):
    months = int(months)
    discount = float(discount_pct) / 100.0
    fc = forecast.forecast_subcategory_units(subcategory, months, discount=discount)
    hist = _history("monthly_subcategory", "subcategory", subcategory).tail(18)

    fc_dates = pd.to_datetime([r["period"] for r in fc])
    yhat = [r["forecast"] for r in fc]

    fig = _base_chart(
        f"{subcategory} — Monthly Unit Demand Forecast (model: LightGBM)", "Units")
    ht = "%{x|%b %Y}<br>%{y:,.0f} units<extra></extra>"
    fig.add_trace(go.Scatter(x=hist["period"], y=hist["quantity"], name="History",
                             mode="lines", line=dict(color="#2b6cb0", width=2.5),
                             hovertemplate=ht))
    fig.add_trace(go.Scatter(x=fc_dates, y=yhat,
                             name=f"Forecast @ {int(discount_pct)}% off",
                             mode="lines+markers",
                             line=dict(color="#dd6b20", width=2.5, dash="dash"),
                             marker=dict(size=7), hovertemplate=ht))

    table = pd.DataFrame(fc).rename(columns={
        "period": "Month", "forecast": "Forecast (units)"})
    return fig, table


# --------------------------------------------------------------------------
# Tab 4 — hierarchical reconciliation (coherent across all levels)
# --------------------------------------------------------------------------
def predict_reconciled(months):
    months = int(months)
    df = reconcile.reconciled_future(months, method="mint")
    df["forecast"] = df["forecast"].clip(lower=0)  # tiny negatives -> 0

    # Chart: total line + a stacked bar of category contributions per month.
    totals = df[df["level"] == "total"].sort_values("period")
    cats = df[df["level"] == "cat"]
    months_axis = pd.to_datetime(totals["period"])

    fig = _base_chart("Coherent Forecast — Levels That Sum Correctly (MinT)",
                      "Revenue (USD)", "$")
    for cat_name in sorted(cats["name"].unique()):
        d = cats[cats["name"] == cat_name].sort_values("period")
        fig.add_trace(go.Bar(x=pd.to_datetime(d["period"]), y=d["forecast"],
                             name=cat_name))
    fig.add_trace(go.Scatter(x=months_axis, y=totals["forecast"], name="Total",
                             mode="lines+markers",
                             line=dict(color="#1a202c", width=2.5),
                             hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>"))
    fig.update_layout(barmode="stack")

    # Table: wide view, one row per month, columns = total + 3 categories.
    pivot = (df[df["level"].isin(["total", "cat"])]
             .pivot_table(index="period", columns="name", values="forecast",
                          aggfunc="sum").reset_index())
    pivot = pivot.rename(columns={"period": "Month", "All": "Total ($)"})
    return fig, pivot



def build_ui():
    acc_total = _accuracy("total_monthly_sales", "prophet")
    acc_cat = _accuracy("category_monthly_sales", "lightgbm")
    acc_sub = _accuracy("subcategory_monthly_units", "lightgbm")

    with gr.Blocks(title="Retail Sales Forecasting") as demo:
        gr.Markdown(
            "# Retail Sales Forecasting\n"
            "Forecasts built on four years of Superstore orders "
            "(17 subcategories, ~$2.3M in sales). The store's sales are "
            "predicted at three levels: total revenue with **Prophet**, and "
            "category revenue and subcategory unit demand with **LightGBM**. "
            "Charts are interactive — hover to read values, drag to zoom."
        )

        # ---- Tab 1 -------------------------------------------------------
        with gr.Tab("Total revenue"):
            gr.Markdown(
                f"**Model:** Prophet  •  **Backtested accuracy:** "
                f"{acc_total:.0f}%  •  **Forecasts:** total store revenue per month"
                if acc_total else "**Model:** Prophet"
            )
            months1 = gr.Slider(1, 12, value=6, step=1, label="Months to forecast")
            btn1 = gr.Button("Update forecast", variant="primary")
            plot1 = gr.Plot()
            gr.Markdown(
                "*The blue line is actual past revenue; the orange dashed line "
                "is the prediction. The shaded band is the likely range — the "
                "model's confidence interval, not a guarantee.*"
            )
            tbl1 = gr.Dataframe()
            btn1.click(predict_total, inputs=months1, outputs=[plot1, tbl1])

        # ---- Tab 2 -------------------------------------------------------
        with gr.Tab("Category revenue"):
            gr.Markdown(
                f"**Model:** LightGBM  •  **Backtested accuracy:** "
                f"{acc_cat:.0f}%  •  **Forecasts:** revenue per month, by category"
                if acc_cat else "**Model:** LightGBM"
            )
            with gr.Row():
                cat = gr.Dropdown(forecast.available_categories(),
                                  value="Technology", label="Category")
                months_c = gr.Slider(1, 12, value=6, step=1, label="Months to forecast")
            btn_c = gr.Button("Update forecast", variant="primary")
            plot_c = gr.Plot()
            gr.Markdown(
                "*Revenue forecast for one of the three product categories — "
                "the middle view between the whole-store total and the "
                "individual subcategories.*"
            )
            tbl_c = gr.Dataframe()
            btn_c.click(predict_category, inputs=[cat, months_c], outputs=[plot_c, tbl_c])

        # ---- Tab 3 -------------------------------------------------------
        with gr.Tab("Subcategory units"):
            gr.Markdown(
                f"**Model:** LightGBM  •  **Backtested accuracy:** "
                f"{acc_sub:.0f}%  •  **Forecasts:** units sold per month, by subcategory"
                if acc_sub else "**Model:** LightGBM"
            )
            with gr.Row():
                sub = gr.Dropdown(forecast.available_subcategories(),
                                  value="Phones", label="Subcategory")
                months2 = gr.Slider(1, 12, value=6, step=1, label="Months to forecast")
            disc = gr.Slider(0, 50, value=0, step=5,
                             label="What-if: average discount (%)")
            btn2 = gr.Button("Update forecast", variant="primary")
            plot2 = gr.Plot()
            gr.Markdown(
                "*Blue is past units sold; orange dashed is predicted demand. "
                "Drag the discount slider to see how a promotion would change "
                "demand — the model learned this relationship from the data.*"
            )
            tbl2 = gr.Dataframe()
            btn2.click(predict_subcat, inputs=[sub, months2, disc], outputs=[plot2, tbl2])

        # ---- Tab 4 -------------------------------------------------------
        with gr.Tab("Coherent forecast"):
            gr.Markdown(
                "**Method:** MinT reconciliation  •  **What it does:** makes "
                "the three levels agree — subcategories sum to categories, "
                "categories sum to the total"
            )
            months_r = gr.Slider(1, 12, value=6, step=1, label="Months to forecast")
            btn_r = gr.Button("Update forecast", variant="primary")
            plot_r = gr.Plot()
            gr.Markdown(
                "*Forecasting each level on its own leaves them inconsistent — "
                "the parts don't add up to the whole. Reconciliation adjusts "
                "all the forecasts together so they're guaranteed to sum "
                "correctly. The stacked bars are the categories; the black "
                "line is the total they add up to.*"
            )
            tbl_r = gr.Dataframe()
            btn_r.click(predict_reconciled, inputs=months_r, outputs=[plot_r, tbl_r])

        # ---- Render all charts on page load so nothing is blank ----------
        demo.load(predict_total, inputs=months1, outputs=[plot1, tbl1])
        demo.load(predict_category, inputs=[cat, months_c], outputs=[plot_c, tbl_c])
        demo.load(predict_subcat, inputs=[sub, months2, disc], outputs=[plot2, tbl2])
        demo.load(predict_reconciled, inputs=months_r, outputs=[plot_r, tbl_r])

    return demo


if __name__ == "__main__":
    # Bind to 0.0.0.0 so this also works inside Docker, but tell the user
    # the address they can actually click in a browser.
    print("\nOpen the app at:  http://127.0.0.1:7860\n")
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
