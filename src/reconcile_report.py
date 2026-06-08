"""
Build the sales-dollar hierarchy, forecast every node independently with
Prophet, then compare independent forecasts against the three reconciliation
methods on a held-out test window.

This is the "does reconciliation help?" experiment. It reports, for each
method: accuracy across all nodes, at the total level, at the subcategory
level, and the coherence error (how badly the levels fail to sum — which
reconciliation drives to zero).

Run:  python src/reconcile_report.py
"""

import warnings
import logging

import numpy as np
import pandas as pd

import config
import metrics
from reconcile import Hierarchy, METHODS, coherence_error
from model_prophet import ProphetForecaster

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _load_hierarchy():
    sub = pd.read_csv(config.PROCESSED_DIR / "monthly_subcategory.csv",
                      parse_dates=["period"])
    bottom = (sub.pivot_table(index="period", columns="subcategory",
                              values="sales", aggfunc="sum")
              .fillna(0).sort_index())
    leaf_to_parent = (sub.drop_duplicates("subcategory")
                      .set_index("subcategory")["category"].to_dict())
    hierarchy = Hierarchy(list(bottom.columns), leaf_to_parent)
    return bottom, hierarchy


def _base_forecasts(node_actuals, dates, train_idx, test_idx):
    """Forecast every node independently with Prophet."""
    H = len(test_idx)
    base = np.zeros((H, node_actuals.shape[1]))
    for j in range(node_actuals.shape[1]):
        s = pd.Series(node_actuals[:, j], index=dates)
        m = ProphetForecaster(seasonality_mode="multiplicative")
        m.fit(pd.Series(s.index[train_idx]), pd.Series(s.values[train_idx]))
        base[:, j] = m.predict(pd.Series(s.index[test_idx])).values
    return base


def run(horizon: int = 6):
    bottom, hierarchy = _load_hierarchy()
    node_actuals = hierarchy.aggregate(bottom.values)   # T x n_nodes
    dates = bottom.index
    n = len(bottom)
    train_idx = np.arange(0, n - horizon)
    test_idx = np.arange(n - horizon, n)
    actual_test = node_actuals[test_idx]
    n_leaves = len(hierarchy.bottom)

    base = _base_forecasts(node_actuals, dates, train_idx, test_idx)

    def report(name, pred):
        acc_all = metrics.accuracy(actual_test.flatten(), pred.flatten())
        acc_tot = metrics.accuracy(actual_test[:, 0], pred[:, 0])
        acc_sub = metrics.accuracy(actual_test[:, -n_leaves:].flatten(),
                                   pred[:, -n_leaves:].flatten())
        incoh = coherence_error(hierarchy, pred)
        print(f"  {name:12s}  all {acc_all:5.1f}%   total {acc_tot:5.1f}%   "
              f"subcat {acc_sub:5.1f}%   incoherence ${incoh:,.0f}")

    print(f"Hierarchical reconciliation  |  horizon = {horizon} months")
    print(f"Nodes: 1 total + {len(hierarchy.parents)} categories + "
          f"{n_leaves} subcategories = {len(hierarchy.node_labels)}\n")
    print("  method        accuracy (all / total / subcat)        coherence")
    report("independent", base)
    for name, fn in METHODS.items():
        report(name, fn(hierarchy, base))

    print("\nReconciliation forces the levels to sum correctly (incoherence "
          "-> $0). On this noisy data the accuracy is about the same, and "
          "MinT improves the total level slightly while guaranteeing coherence.")


if __name__ == "__main__":
    run()
