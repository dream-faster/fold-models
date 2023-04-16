from typing import Optional, Union

import numpy as np
import pandas as pd
from fold.models.base import TimeSeriesModel
from sklearn.linear_model import LinearRegression, SGDRegressor


class AR(TimeSeriesModel):
    def __init__(self, p: int) -> None:
        self.p = p
        self.name = f"AR-{str(p)}"
        self.properties = TimeSeriesModel.Properties(
            requires_X=False,
            mode=TimeSeriesModel.Properties.Mode.online,
            model_type=TimeSeriesModel.Properties.ModelType.regressor,
            memory_size=p,
            _internal_supports_minibatch_backtesting=False,
        )
        # self.models = [LinearRegression() for _ in range(p)]
        self.models = [SGDRegressor() for _ in range(p)]

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        # Using Least Squares as it's faster than SGD for the initial fit
        # for index, model in enumerate(self.models, start=1):
        #     model.fit(
        #         y.shift(index).to_frame()[index:],
        #         y[index:],
        #         sample_weight=sample_weights[-index:]
        #         if sample_weights is not None
        #         else None,
        #     )
        # self.parameters = [
        #     {
        #         "coef_": model.coef_[0],
        #         "intercept_": model.intercept_,
        #     }
        #     for model in self.models
        # ]
        for index, model in enumerate(self.models, start=1):
            model.fit(
                y.shift(index).to_frame()[index:],
                y[index:],
                sample_weight=sample_weights[-index:]
                if sample_weights is not None
                else None,
            )

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        if isinstance(self.models[0], LinearRegression):
            self.models = [SGDRegressor(warm_start=True) for _ in range(self.p)]
            for index, (model, parameters) in enumerate(
                zip(self.models, self.parameters), start=1
            ):
                model.fit(
                    y.shift(index).to_frame()[-index:],
                    y[-index:],
                    coef_init=parameters["coef_"],
                    intercept_init=parameters["intercept_"],
                    sample_weight=sample_weights[-index:]
                    if sample_weights is not None
                    else None,
                )
        else:
            for index, model in enumerate(self.models, start=1):
                model.partial_fit(
                    y.shift(index).to_frame()[-index:],
                    y[-index:],
                    sample_weight=sample_weights[-index:]
                    if sample_weights is not None
                    else None,
                )

    def predict_in_sample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return predict(self.models, y.shift(1), indices=X.index)

    def predict(
        self, X: pd.DataFrame, past_y: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        return predict(self.models, past_y, indices=X.index)


def predict(models, memory_y: pd.Series, indices) -> pd.Series:
    if len(memory_y) == 1:
        return pd.Series(models[0].predict(memory_y.to_frame()), index=indices.iloc[-1])

    preds = [
        np.concatenate(
            [
                np.zeros((index,)),
                lr.predict(memory_y.shift(index - 1).to_frame()[index:]),
            ]
        )
        for index, lr in enumerate(models, start=1)
    ]
    return pd.Series(np.vstack(preds).sum(axis=0), index=indices)
