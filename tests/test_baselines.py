import numpy as np
import pytest
from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Test
from fold.utils.tests import generate_sine_wave_data, tuneability_test

from fold_models.baseline import (
    ExponentiallyWeightedMovingAverage,
    MovingAverage,
    NaiveSeasonal,
)


def check_if_not_nan(x):
    assert not x.isna().squeeze().any()


test_assert = Test(fit_func=check_if_not_nan, transform_func=lambda X: X)


@pytest.mark.parametrize("online", [True, False])
def test_baseline_naive_seasonal(online: bool) -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    naive_seasonal = NaiveSeasonal(seasonal_length=12)
    naive_seasonal.properties._internal_supports_minibatch_backtesting = not online
    pred, _ = train_backtest([naive_seasonal, test_assert], X, y, splitter)
    assert np.isclose(
        pred.squeeze(), y[pred.index], atol=0.02
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets
    tuneability_test(naive_seasonal, dict(seasonal_length=5))


@pytest.mark.parametrize("online", [True, False])
def test_baseline_mean(online: bool) -> None:
    X, y = generate_sine_wave_data(cycles=10, length=400)
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    ma = MovingAverage(window_size=12)
    ma.properties._internal_supports_minibatch_backtesting = not online
    pred, _ = train_backtest([ma, test_assert], X, y, splitter)
    assert np.isclose(
        y.shift(1).rolling(12).mean()[pred.index], pred.squeeze(), atol=0.01
    ).all()
    assert (
        len(pred) == 400 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets

    tuneability_test(
        ma,
        dict(window_size=5),
        tolerance=0.0001,
    )


@pytest.mark.parametrize("online", [True, False])
def test_baseline_ewmean(online: bool) -> None:
    X, y = generate_sine_wave_data(cycles=10, length=400)
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    ma = ExponentiallyWeightedMovingAverage(window_size=12)
    ma.properties._internal_supports_minibatch_backtesting = not online
    pred, _ = train_backtest([ma, test_assert], X, y, splitter)
    assert np.isclose(
        y.shift(1).ewm(alpha=1 / 12, adjust=True).mean()[pred.index],
        pred.squeeze(),
        atol=0.01,
    ).all()
    assert (
        len(pred) == 400 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets

    tuneability_test(
        ma,
        dict(window_size=5),
        tolerance=0.05,
    )
