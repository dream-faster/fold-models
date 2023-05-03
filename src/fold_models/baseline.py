from typing import Union

import pandas as pd
from fold.base import Tunable, fit_noop
from fold.models.base import TimeSeriesModel
from fold.models.baseline import Naive  # noqa


class NaiveSeasonal(TimeSeriesModel, Tunable):
    """
    A model that predicts the last value seen in the same season.
    """

    name = "NaiveSeasonal"

    def __init__(self, seasonal_length: int) -> None:
        assert seasonal_length > 1, "seasonal_length must be greater than 1"
        self.seasonal_length = seasonal_length
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            memory_size=seasonal_length,
            _internal_supports_minibatch_backtesting=True,
        )

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            past_y.iloc[-self.seasonal_length].squeeze(),
            index=X.index[-1:None],
        )

    def predict_in_sample(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return past_y.shift(self.seasonal_length - 1)

    fit = fit_noop
    update = fit

    def get_params(self) -> dict:
        return {"seasonal_length": self.seasonal_length}


class MovingAverage(TimeSeriesModel, Tunable):
    """
    A model that predicts the mean of the last values seen.
    """

    name = "MovingAverage"

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            memory_size=window_size,
            _internal_supports_minibatch_backtesting=True,
        )

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            past_y.rolling(self.window_size).mean()[-1], index=X.index[-1:None]
        )

    def predict_in_sample(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return past_y.rolling(self.window_size).mean()

    fit = fit_noop
    update = fit

    def get_params(self) -> dict:
        return {"window_size": self.window_size}


class ExponentiallyWeightedMovingAverage(TimeSeriesModel, Tunable):
    """
    A model that predicts the exponentially weighed mean of the last values seen.
    """

    name = "ExponentiallyWeightedMovingAverage"

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            memory_size=self.window_size * 4,
            _internal_supports_minibatch_backtesting=True,
        )

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        # it's an online transformation, so len(X) will be always 1,
        return pd.Series(
            past_y.ewm(alpha=1 / self.window_size, adjust=True).mean()[-1],
            index=X.index[-1:None],
        )

    def predict_in_sample(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return past_y.ewm(alpha=1 / self.window_size, adjust=True).mean()

    fit = fit_noop
    update = fit

    def get_params(self) -> dict:
        return {"window_size": self.window_size}
