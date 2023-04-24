from typing import Optional, Union

import pandas as pd
from fold.models.base import TimeSeriesModel
from fold.transformations.difference import Difference

from .ar import AR


class ARIMA(TimeSeriesModel):

    ar_model = None
    diff_models = None
    ma_model = None

    def __init__(self, p: int, d: int, q: int) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.name = f"ARIMRA-{str(p)}-{str(d)}-{str(q)}"
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            model_type=TimeSeriesModel.Properties.ModelType.regressor,
            memory_size=p,
            _internal_supports_minibatch_backtesting=True,
        )
        assert d >= 0, "d must be above 0"
        assert d <= 1, "we currently don't support d > 1 just yet"
        assert q <= p, "currently we don't support q > p"
        if d > 0:
            self.diff_model = Difference(d)
        if p > 0:
            self.ar_model = AR(p)
        if q > 0:
            self.ma_model = AR(q)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.d > 0:
            self.diff_model.fit(X, y, sample_weights)
            X = self.diff_model.transform(X, in_sample=True).iloc[self.d :]
            y = self.diff_model.transform(y, in_sample=True).iloc[self.d :]

        if self.p > 0:
            self.ar_model.fit(X, y, sample_weights)
        residuals = (
            y
            if self.ar_model is None
            else y - self.ar_model.predict_in_sample(X, y.shift(1))
        )

        if self.q > 0:
            self.ma_model.fit(X, residuals, sample_weights)

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if self.d > 0:
            self.diff_model.update(X, y, sample_weights)
            X = self.diff_model.transform(X, in_sample=False)
            y = self.diff_model.transform(y, in_sample=False)

        if self.p > 0:
            self.ar_model.update(X, y, sample_weights)
        residuals = (
            y
            if self.ar_model is None
            else y - self.ar_model.predict_in_sample(X, y.shift(1))
        )

        if self.q > 0:
            self.ma_model.update(X, residuals, sample_weights)

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        if self.d > 0:
            X = self.diff_model.transform(X, in_sample=True)
            past_y = self.diff_model.transform(past_y, in_sample=True)

        def undifference_if_needed(result: pd.Series) -> pd.Series:
            if self.d > 0:
                return self.diff_model.inverse_transform(result)
            else:
                return result

        ar_result = (
            past_y
            if self.ar_model is None
            else past_y - self.ar_model.predict(X, past_y)
        )
        if self.ma_model is not None:
            ma_result = self.ma_model.predict(X, past_y)
            return undifference_if_needed(ar_result + ma_result)
        else:
            return undifference_if_needed(ar_result)

    predict_in_sample = predict
