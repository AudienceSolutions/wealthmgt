"""
ForecastModel abstract class.
"""
from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from dataset import DataSet
from time_series_retirement import TimeSeriesForecast


def build_sequences(data: pd.DataFrame, n_sequence: int) -> np.ndarray:
    n_samples, n_features = data.shape
    x = np.zeros((n_samples, n_sequence, n_features), dtype=np.float32)
    for i in range(n_sequence):
        j = n_sequence - i - 1
        df = data.shift(i)
        x[:, j, :] = df.values.astype(np.float32)
    return x


def build_training_sequences(
    data: pd.DataFrame,
    n_sequence: int,
    full_sequence: bool = True,
    only_valid: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    x = build_sequences(data, n_sequence)
    if full_sequence:
        y = build_sequences(data.shift(-1), n_sequence)
    else:
        y = build_sequences(data.shift(-1), n_sequence=1)
    if only_valid:
        valid = np.isfinite(x) & np.isfinite(y)
        valid = np.all(valid, axis=(1, 2))
        x, y = x[valid], y[valid]
    return x, y


class ModelConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class ForecastModel(ABC):
    """
    Abstract ForecastModel class
    """

    def __init__(self, dataset: DataSet, config: ModelConfig = ModelConfig()) -> None:
        """
        Initialize a ForecastModel
        """
        self.dataset = dataset
        self.config = config
        self.variables = self.dataset.variables
        self.freq = self.dataset.freq
        self.n_variables = len(self.variables)

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(data.index, pd.DatetimeIndex)
        assert set(data.columns) == set(self.dataset.variables)
        data = data.sort_index().copy()
        return data

    def reset(self) -> None:
        """
        clear state
        """
        pass

    def persist(self, filename: str) -> str:
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    @staticmethod
    def load(filename: str) -> ForecastModel:
        with open(filename, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, ForecastModel)
        return model

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
    ) -> Any:
        """
        fit the model
        """
        pass

    @abstractmethod
    def forecast(
        self,
        data: pd.DataFrame,
        n_periods: int,
        n_paths: int,
        include_history: bool = False,
        exogenous: Optional[TimeSeriesForecast] = None,
    ) -> TimeSeriesForecast:
        """
        produce distribution of forecasted series
        """
        pass

    def backtest(
        self,
        data: pd.DataFrame,
        n_periods: int,
        n_paths: int,
        n_window: Optional[int] = None,
        exogenous: Optional[TimeSeriesForecast] = None,
    ) -> TimeSeriesForecast:
        """
        run the backtest
        """
        data = self.validate_data(data)
        dates = pd.to_datetime(list(data.index))
        if n_window is not None:
            # need at least `n_window` dates
            dates = dates[n_window:]

        # todo: we should be able to parallelize this
        forecasts: List[TimeSeriesForecast] = []
        for date in dates:
            df = data[data.index <= date]

            # todo: should we fit_transform here or inside self.fit?
            # df = self.dataset.fit_transform(df)

            if n_window is not None:
                df = df.tail(n_window)

            self.reset()
            # todo: do we want to save the output of model.fit?
            # todo: do we want to use any `test_data`?
            self.fit(data=df, test_data=None)

            # we assume that model.fit/forecast will handle `exogenous` correctly,
            # `exogenous` is likely coming from a different BacktestModel.backtest
            forecast = self.forecast(
                data=df,
                n_periods=n_periods,
                n_paths=n_paths,
                include_history=False,
                exogenous=exogenous,
            )
            forecasts.append(forecast)

        tsf = TimeSeriesForecast.load_from_tsfs(forecasts, combine_dim="date")
        return tsf