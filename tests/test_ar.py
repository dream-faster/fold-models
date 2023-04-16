import numpy as np
from fold.loop import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data
from fold_wrapper import WrapStatsForecast, WrapStatsModels
from statsforecast.models import ARIMA as StatsForecastARIMA
from statsmodels.tsa.arima.model import ARIMA as StatsModelARIMA

from fold_models.ar import AR


def test_ar_equivalent() -> None:
    _, y = generate_sine_wave_data(length=70, freq="s")

    model = AR(1)
    splitter = ExpandingWindowSplitter(initial_train_window=40, step=1)
    pred_own_ar, _ = train_backtest(model, None, y, splitter)

    # model = WrapStatsForecast.from_model(StatsForecastARIMA((1, 0, 0)))
    model = WrapStatsModels(
        StatsModelARIMA, init_args={"order": (1, 0, 0)}, online_mode=True
    )

    pred_statsforecast_ar, _ = train_backtest(model, None, y, splitter)
    assert np.isclose(
        pred_statsforecast_ar.squeeze(), pred_own_ar.squeeze(), atol=0.001
    ).all()


def test_ar_speed() -> None:
    _, y = generate_monotonous_data(length=7000, freq="s")

    model = AR(2)
    splitter = ExpandingWindowSplitter(initial_train_window=0.1, step=0.1)
    train_backtest(model, None, y, splitter)


def test_statsforecast_arima_speed() -> None:
    _, y = generate_monotonous_data(length=7000, freq="s")

    model = WrapStatsForecast.from_model(StatsForecastARIMA((2, 0, 0)))
    splitter = ExpandingWindowSplitter(initial_train_window=0.1, step=0.1)
    train_backtest(model, None, y, splitter)
