from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

from dataset import DataSet
from forecast_retirement.base import ForecastModel, ModelConfig
from time_series_retirement import TimeSeriesForecast


def subset_period(
    x: np.ndarray,
    periodicity: int,
    offset: int,
) -> np.ndarray:
    last_period = len(x) - periodicity
    last_matching_period = last_period + offset % periodicity
    period_idx = range(int(last_matching_period), -1, -periodicity)
    x_subset: np.ndarray = x[period_idx]
    return x_subset


class ArimaConfig(ModelConfig):
    # p = AR lag; q = MA lag; d = difference
    # lags: <x: int> will use saturated lag polynomials from 1 to x
    # lags: <[x, y, ...]: list> will set only lag polynomials specified
    p: Union[int, List[int]]
    d: int = 0
    q: Union[int, List[int]]
    fit_params: Dict[str, Any] = {"method": "statespace"}



class ArimaModel(ForecastModel):
    def __init__(
        self,
        dataset: DataSet,
        config: ArimaConfig,
    ) -> None:
        super().__init__(dataset, config)
        self.p = config.p
        self.d = config.d
        self.q = config.q
        self.fit_params = config.fit_params
        self.model: Optional[ARIMAResultsWrapper] = None
        self.in_sample_resids: np.ndarray

    def reset(self) -> None:
        self.model = None

    def fit(
        self,
        data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
    ) -> Any:
        """"""
        data = self.validate_data(data)
        df = self.dataset.fit_transform(data)
        # replace np.nan or +/- inf with 0
        df.replace(
            [np.inf, -np.inf, np.nan], 0, inplace=True
        )
        model = ARIMA(
            df,
            order=(self.p, self.d, self.q)
        )
        self.model = model.fit(**self.fit_params)
        self.in_sample_resids = self.model.resid

    def forecast(
        self,
        data: pd.DataFrame,
        n_periods: int,
        n_paths: int,
        include_history: bool = False,
        exogenous: Optional[TimeSeriesForecast] = None,
    ) -> TimeSeriesForecast:

        if include_history:
            # todo: implement this
            # To forecast from any period T, we need to train
            # a new model w/ observations_t where t <= T.
            raise ValueError("Backtesting over full times series not yet implemented")

        if not self.model:
            raise ValueError("Must fit model before calling forecast")

        data = self.validate_data(data)
        df = self.dataset.transform(data)

        eval_model = self.model.apply(df)

        sim_df = eval_model.simulate(
            nsimulations=n_periods, repetitions=n_paths, anchor="end"
        )

        # ARIMAResults.simulate() return shape is (n_periods, 1, n_paths)
        # TimeSeriesForecast.data requires shape (dates, periods, paths, variables)
        sim = 1 * sim_df.values[None, :, :, None]

        dates = df.index[-1:] + self.freq.value.increment

        tsf = TimeSeriesForecast.load_from_array(
            data=sim,
            variables=self.variables,
            dates=dates,
            freq=self.freq,
        )
        tsf.data = self.dataset.inv_transform_xr(tsf.data)
        return tsf


