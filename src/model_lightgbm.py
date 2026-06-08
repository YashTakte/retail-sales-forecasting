"""
LightGBM forecaster.

Where Prophet models each series on its own, we train a single LightGBM
model across *all* series at once. The engineered features (lags, rolling
means, calendar parts, and a series-id code) let one gradient-boosted model
learn both the shared seasonality and each series' own level. This
"global model" approach is usually stronger than many tiny per-series
models when individual series are short or noisy.

We use an L1 (mean-absolute-error) objective because it lines up with the
WMAPE metric we report and is more robust to the occasional huge order.
"""

import lightgbm as lgb
import pandas as pd

from features import build_features, feature_columns


class LightGBMForecaster:
    def __init__(self, group: str | None, target: str = "sales",
                 include_discount: bool = False):
        # `group` names the column that distinguishes series (e.g.
        # 'subcategory'); None means we're modelling one combined series.
        # `include_discount` adds average discount as a feature, which lets
        # the app run "what-if" scenarios (see the Gradio discount slider).
        self.group = group
        self.target = target
        self.include_discount = include_discount
        self.features = feature_columns(group, include_discount)
        self.model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.group, self.target)

    def fit(self, train_df: pd.DataFrame) -> "LightGBMForecaster":
        feat = self._frame(train_df).dropna(subset=self.features)
        self.model.fit(feat[self.features], feat[self.target])
        return self

    def predict(self, feat_df: pd.DataFrame) -> pd.Series:
        # Caller passes a frame that already went through build_features
        # (needed because lag features depend on history, so the train and
        # test rows have to be featurised together — see train.py).
        preds = self.model.predict(feat_df[self.features])
        return pd.Series(preds, index=feat_df.index).clip(lower=0)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._frame(df)
