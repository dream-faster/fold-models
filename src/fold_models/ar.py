from typing import Optional, Union

import numpy as np
import pandas as pd
from fold.models.base import Model
from pytest import param
from sklearn.linear_model import LinearRegression, SGDRegressor


class AR(Model):
    def __init__(self, p: int) -> None:
        self.p = p
        self.name = f"AR-{str(p)}"
        self.properties = Model.Properties(
            requires_X=False,
            mode=Model.Properties.Mode.online,
            model_type=Model.Properties.ModelType.regressor,
            memory_size=p,
            _internal_supports_minibatch_backtesting=False,
        )
        self.models = [LinearRegression() for _ in range(p)]

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        # Using Least Squares as it's faster than SGD for the initial fit
        for index, model in enumerate(self.models, start=1):
            model.fit(
                y.shift(index).to_frame()[index:],
                y[index:],
                sample_weight=sample_weights[-index:]
                if sample_weights is not None
                else None,
            )
        self.parameters = [
            {
                "coef_": model.coef_[0],
                "intercept_": model.intercept_,
            }
            for model in self.models
        ]

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
                model.fit(
                    y.shift(index).to_frame()[-index:],
                    y[-index:],
                    sample_weight=sample_weights[-index:]
                    if sample_weights is not None
                    else None,
                )

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return predict(
            self.models, self._state.memory_y, in_sample=False, index=X.index
        )

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return predict(self.models, self._state.memory_y, in_sample=True, index=X.index)


def predict(models, memory_y: pd.Series, in_sample: bool, index) -> pd.Series:
    preds = [
        np.concatenate(
            [
                np.empty((index,)),
                lr.predict(
                    memory_y.shift(
                        index if in_sample is True else index - 1
                    ).to_frame()[index:]
                ),
            ]
        )
        for index, lr in enumerate(models, start=1)
    ]
    return pd.Series(np.vstack(preds).sum(axis=0), index=index)
