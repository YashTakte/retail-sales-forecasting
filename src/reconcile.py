r"""
Hierarchical forecast reconciliation (on the sales-dollar hierarchy).

The store's sales form a hierarchy:

        TOTAL
       /  |  \
   Furn  Off  Tech          (3 categories)
   / \   /|\   /|\
  ...17 subcategories...     (bottom level)

If you forecast each level on its own, the numbers don't add up — the 17
subcategory forecasts won't sum to the category forecasts, which won't sum
to the total. That's "incoherent", and it's a real problem: a planner can't
trust figures that contradict each other.

Reconciliation fixes this. It adjusts all the forecasts together so they're
guaranteed to sum correctly, using the structure of the hierarchy. We
implement three standard approaches:

  * bottom_up — forecast the 17 leaves, sum upward. Simple, always coherent.
  * ols       — project the independent forecasts onto the coherent space
                (ordinary least squares).
  * mint      — Minimum Trace reconciliation (Wickramasuriya et al., 2019),
                the modern default. Coherent AND tends to improve the top
                levels by letting the stable aggregate inform the noisy parts.

On this dataset reconciliation's main win is coherence rather than a big
accuracy jump (the data is noisy), and MinT delivers that at no accuracy
cost while slightly improving the total-level forecast.
"""

import numpy as np
import pandas as pd


class Hierarchy:
    """Holds the structure that links bottom-level series to all nodes."""

    def __init__(self, bottom_labels: list[str], leaf_to_parent: dict[str, str]):
        self.bottom = list(bottom_labels)
        self.leaf_to_parent = leaf_to_parent
        self.parents = sorted(set(leaf_to_parent.values()))
        self.S, self.node_labels = self._build_summing_matrix()

    def _build_summing_matrix(self):
        """
        S maps the bottom level to every node in the hierarchy.
        Row order: TOTAL, then one row per category, then the 17 leaves.
        """
        rows, labels = [], []

        rows.append(np.ones(len(self.bottom)))
        labels.append("TOTAL")

        for p in self.parents:
            rows.append(np.array([1.0 if self.leaf_to_parent[b] == p else 0.0
                                  for b in self.bottom]))
            labels.append(f"CAT::{p}")

        for i, b in enumerate(self.bottom):
            r = np.zeros(len(self.bottom)); r[i] = 1.0
            rows.append(r)
            labels.append(f"SUB::{b}")

        return np.vstack(rows), labels

    def aggregate(self, bottom_values: np.ndarray) -> np.ndarray:
        """Turn bottom-level values (T x n_leaves) into all nodes (T x n_nodes)."""
        return bottom_values @ self.S.T


# --------------------------------------------------------------------------
# Reconciliation methods. Each takes base forecasts for ALL nodes
# (shape: horizon x n_nodes) and returns coherent forecasts of the same shape.
# --------------------------------------------------------------------------
def _reconcile_with_G(S: np.ndarray, G: np.ndarray, base: np.ndarray) -> np.ndarray:
    # base: H x n_nodes -> reconciled: H x n_nodes
    return (S @ (G @ base.T)).T


def bottom_up(hierarchy: Hierarchy, base: np.ndarray) -> np.ndarray:
    # Use only the leaf forecasts and sum upward.
    n_leaves = len(hierarchy.bottom)
    bottom_base = base[:, -n_leaves:]
    return bottom_base @ hierarchy.S.T


def ols(hierarchy: Hierarchy, base: np.ndarray) -> np.ndarray:
    S = hierarchy.S
    G = np.linalg.inv(S.T @ S) @ S.T
    return _reconcile_with_G(S, G, base)


def mint(hierarchy: Hierarchy, base: np.ndarray) -> np.ndarray:
    """MinT with a structural weight matrix (robust, no variance estimation)."""
    S = hierarchy.S
    w = 1.0 / S.sum(axis=1)        # structural scaling per node
    W = np.diag(w)
    G = np.linalg.inv(S.T @ W @ S) @ S.T @ W
    return _reconcile_with_G(S, G, base)


METHODS = {"bottom_up": bottom_up, "ols": ols, "mint": mint}


def coherence_error(hierarchy: Hierarchy, forecasts: np.ndarray) -> float:
    """Mean absolute gap between the total and the sum of the leaves."""
    n_leaves = len(hierarchy.bottom)
    total = forecasts[:, 0]
    leaf_sum = forecasts[:, -n_leaves:].sum(axis=1)
    return float(np.abs(total - leaf_sum).mean())


def reconciled_future(months: int = 6, method: str = "mint") -> pd.DataFrame:
    """
    Produce a coherent forward forecast across all hierarchy nodes.

    Forecasts every node independently with Prophet on the full history,
    then reconciles with the chosen method so the levels sum correctly.
    Returns a tidy DataFrame: period, level, name, forecast ($).
    """
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    import config
    from model_prophet import ProphetForecaster

    sub = pd.read_csv(config.PROCESSED_DIR / "monthly_subcategory.csv",
                      parse_dates=["period"])
    bottom = (sub.pivot_table(index="period", columns="subcategory",
                              values="sales", aggfunc="sum")
              .fillna(0).sort_index())
    leaf_to_parent = (sub.drop_duplicates("subcategory")
                      .set_index("subcategory")["category"].to_dict())
    hierarchy = Hierarchy(list(bottom.columns), leaf_to_parent)

    node_actuals = hierarchy.aggregate(bottom.values)
    future = pd.date_range(bottom.index.max() + pd.offsets.MonthBegin(1),
                           periods=months, freq="MS")

    base = np.zeros((months, node_actuals.shape[1]))
    for j in range(node_actuals.shape[1]):
        m = ProphetForecaster(seasonality_mode="multiplicative")
        m.fit(pd.Series(bottom.index), pd.Series(node_actuals[:, j]))
        base[:, j] = m.predict(pd.Series(future)).values

    recon = METHODS[method](hierarchy, base)

    rows = []
    for i, d in enumerate(future):
        for j, label in enumerate(hierarchy.node_labels):
            level, _, name = label.partition("::")
            rows.append({
                "period": d.date().isoformat(),
                "level": {"TOTAL": "total"}.get(label, level.lower()),
                "name": name or "All",
                "forecast": round(float(recon[i, j]), 2),
            })
    return pd.DataFrame(rows)
