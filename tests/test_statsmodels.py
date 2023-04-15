from fold.splitters import ExpandingWindowSplitter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from utils import (
    run_pipeline_and_check_if_results_close_exogenous,
    run_pipeline_and_check_if_results_close_univariate,
)

from fold_models.statsmodels import WrapStatsModels


def test_statsmodels_univariate_arima() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsModels(model_class=ARIMA, init_args={"order": (1, 1, 0)}),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


def test_statsforecast_univariate_arima_online() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsModels(
            model_class=ARIMA,
            init_args={"order": (1, 1, 0)},
            online_mode=True,
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )


def test_statsforecast_multivariate_arima_online() -> None:
    run_pipeline_and_check_if_results_close_exogenous(
        model=WrapStatsModels(
            model_class=ARIMA,
            init_args={"order": (1, 1, 0)},
            use_exogenous=True,
            online_mode=True,
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )


def test_statsmodels_multivariate_arima() -> None:
    run_pipeline_and_check_if_results_close_exogenous(
        model=WrapStatsModels(
            model_class=ARIMA, init_args={"order": (1, 1, 0)}, use_exogenous=True
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )


def test_statsmodels_univariate_exponential_smoothing() -> None:
    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsModels(
            model_class=ExponentialSmoothing,
            init_args={},
        ),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=1),
    )
