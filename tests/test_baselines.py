import numpy as np
from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Test
from fold.utils.tests import generate_sine_wave_data

from fold_models.baseline import MovingAverage, Naive, NaiveSeasonal


def check_if_not_nan(x):
    assert not x.isna().squeeze().any()


test_assert = Test(fit_func=check_if_not_nan, transform_func=lambda X: X)


def test_baseline_naive() -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        Naive(),
        test_assert,
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (
        pred.squeeze() == y.shift(1)[pred.index]
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_naive_online() -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    naive = Naive()
    naive.properties._internal_supports_minibatch_backtesting = False
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    pred, _ = train_backtest(naive, X, y, splitter)
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

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        NaiveSeasonal(seasonal_length=12),
        test_assert,
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert np.isclose(
        pred.squeeze(), y[pred.index], atol=0.02
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_naive_seasonal_online() -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    naive_seasonal = NaiveSeasonal(seasonal_length=12)
    naive_seasonal.properties._internal_supports_minibatch_backtesting = False
    pred, _ = train_backtest(naive_seasonal, X, y, splitter)
    assert np.isclose(
        pred.squeeze(), y[pred.index], atol=0.02
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_mean() -> None:
    X, y = generate_sine_wave_data(cycles=10, length=400)
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        MovingAverage(window_size=12),
        test_assert,
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert np.isclose(
        y.shift(1).rolling(12).mean()[pred.index], pred.squeeze(), atol=0.01
    ).all()
    assert (
        len(pred) == 400 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_mean_online() -> None:
    X, y = generate_sine_wave_data(cycles=10, length=400)
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    ma = MovingAverage(window_size=12)
    ma.properties._internal_supports_minibatch_backtesting = False
    pred, _ = train_backtest(ma, X, y, splitter)
    assert np.isclose(
        y.shift(1).rolling(12).mean()[pred.index], pred.squeeze(), atol=0.01
    ).all()
    assert (
        len(pred) == 400 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets
