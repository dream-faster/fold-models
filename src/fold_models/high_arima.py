from typing import Union

from fold.base import Composite, Transformation
from fold.composites.residual import ModelResiduals
from fold.composites.target import TransformTarget
from fold.transformations.difference import Difference

from .ar import AR


def ARIMA(p: int, d: int, q: int) -> Union[Composite, Transformation]:
    assert p >= 0, "p must be above 0"
    assert q >= 0, "q must be above 0"
    assert d >= 0, "d must be above 0"
    assert d <= 1, "we currently don't support d > 1 just yet"

    if q > 0:
        model = ModelResiduals(
            primary=AR(p),
            meta=AR(q),
        )
    else:
        model = AR(p)

    if d == 0:
        return model
    else:
        return TransformTarget(wrapped_pipeline=model, y_pipeline=Difference(1))
