import numpy as np
from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.transformations.dev import Test
from fold.utils.tests import generate_sine_wave_data

from fold_models.baseline import MovingAverage, Naive, NaiveSeasonal


def test_baseline_naive() -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    def check_if_not_nan(x):
        assert not x.isna().squeeze().any()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        Naive(),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (
        pred.squeeze() == y.shift(1)[pred.index]
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_naive_seasonal() -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    def check_if_not_nan(x):
        assert not x.isna().squeeze().any()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        NaiveSeasonal(seasonal_length=12),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert np.isclose(
        pred.squeeze(), y[pred.index], atol=0.02
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_mean() -> None:
    X, y = generate_sine_wave_data(cycles=10, length=400)

    def check_if_not_nan(x):
        assert not x.isna().squeeze().any()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        MovingAverage(window_size=12),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert np.isclose(
        y.shift(1).rolling(12).mean()[pred.index], pred.squeeze(), atol=0.01
    ).all()
    assert (
        len(pred) == 400 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets
