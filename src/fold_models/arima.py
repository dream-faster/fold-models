from typing import Union

from fold.base import Composite, Transformation
from fold.composites.residual import ModelResiduals
from fold.composites.target import TransformTarget
from fold.transformations.difference import Difference

from .ar import AR


def ARMA(
    p: int, q: int, ma_model_online: bool = True
) -> Union[Composite, Transformation]:
    assert p >= 0, "p must be above 0"
    assert q >= 0, "q must be above 0"

    if q > 0:
        ma_model = AR(q)
        ma_model.properties._internal_supports_minibatch_backtesting = (
            not ma_model_online
        )
        model = ModelResiduals(
            primary=AR(p),
            meta=ma_model,
        )
    else:
        model = AR(p)

    return model


def ARIMA(p: int, d: int, q: int, ma_model_online: bool = True) -> Composite:
    assert p >= 0, "p must be above 0"
    assert q >= 0, "q must be above 0"
    assert (
        d >= 1
    ), "d must be above 1, if you don't need differencing, use the `ARMA` class"

    if q > 0:
        ma_model = AR(q)
        ma_model.properties._internal_supports_minibatch_backtesting = (
            not ma_model_online
        )
        model = ModelResiduals(
            primary=AR(p),
            meta=ma_model,
        )
    else:
        model = AR(p)

    return TransformTarget(
        wrapped_pipeline=model, y_pipeline=[Difference(1) for _ in range(0, d)]
    )
