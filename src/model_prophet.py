"""
Prophet forecaster.

Prophet is Facebook's additive time-series model: it decomposes a series
into trend + seasonality and is genuinely good at "this thing has a yearly
rhythm and a slow upward drift" — which describes retail sales well.

We wrap it in a small class so the rest of the project can fit/predict
without caring about Prophet's particular column-naming quirks (it insists
on `ds` for dates and `y` for the value).
"""

import logging
import warnings

import pandas as pd

# Prophet is chatty on import and during fitting; quiet it down.
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet


class ProphetForecaster:
    """One Prophet model for a single series."""

    def __init__(self, seasonality_mode: str = "multiplicative",
                 changepoint_prior_scale: float = 0.05):
        # multiplicative seasonality fits retail better: the December bump
        # is a *percentage* lift on a growing baseline, not a fixed amount.
        self.kwargs = dict(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        self.model: Prophet | None = None

    def fit(self, dates: pd.Series, values: pd.Series) -> "ProphetForecaster":
        train = pd.DataFrame({"ds": dates.values, "y": values.values})
        self.model = Prophet(**self.kwargs)
        self.model.fit(train)
        return self

    def predict(self, dates: pd.Series) -> pd.Series:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        future = pd.DataFrame({"ds": pd.Series(dates).values})
        # Sales can't be negative, so clip the lower tail at zero.
        yhat = self.model.predict(future)["yhat"].clip(lower=0)
        return yhat.reset_index(drop=True)

    def predict_with_bounds(self, dates: pd.Series) -> pd.DataFrame:
        """Like predict(), but also returns Prophet's uncertainty interval.

        Returns columns: yhat, yhat_lower, yhat_upper. These power the
        shaded confidence band on the charts — "we expect X, likely between
        lower and upper".
        """
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        future = pd.DataFrame({"ds": pd.Series(dates).values})
        out = self.model.predict(future)[["yhat", "yhat_lower", "yhat_upper"]]
        return out.clip(lower=0).reset_index(drop=True)
