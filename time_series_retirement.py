"""
An `xarray` driven time series class

TODO: add 'label' dimension to TimeSeriesForecast (e.g. a geography or sector)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import MonthEnd

DEFAULT_QUANTILES = np.array(
    [
        1e-05,
        5e-05,
        0.0001,
        0.00025,
        0.0005,
        0.001,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.975,
        0.99,
        0.995,
    ]
)

SHOCK_PERIOD_ATTR_SUFFIX = "_shock_periods"


@dataclass
class FrequencyData:
    code: str
    days_per_period: int
    periods_per_year: int
    periods_per_quarter: Optional[int]
    periods_per_month: Optional[int]
    increment: pd.DateOffset


class Frequency(Enum):
    """
    should be consistent with the codes described here:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    """

    DAILY = FrequencyData("D", 1, 365, 90, 30, pd.DateOffset(days=1))
    WEEKLY = FrequencyData("W", 7, 52, 13, 4, pd.DateOffset(days=7))
    BIWEEKLY = FrequencyData("2W", 14, 26, 6, 2, pd.DateOffset(days=14))
    MONTHLY = FrequencyData("M", 30, 12, 3, 1, MonthEnd())
    QUARTERLY = FrequencyData("3M", 90, 4, 1, None, MonthEnd(3))
    SEMIANNUALLY = FrequencyData("6M",182,2, None, None, MonthEnd(6))
    ANNUALLY = FrequencyData("12M", 365, 1, None, None, MonthEnd(12))


def to_datetime(dates: Any) -> pd.DatetimeIndex:
    return pd.to_datetime(list(dates))


def infer_freq(dates: pd.DatetimeIndex) -> Frequency:
    dates = dates.sort_values()
    days_diff = (dates[1:] - dates[:-1]).days
    median_diff = np.median(days_diff)
    if median_diff == 1:
        freq = Frequency.DAILY
    elif median_diff == 7:
        freq = Frequency.WEEKLY
    elif (28 <= median_diff) and (median_diff <= 33):
        freq = Frequency.MONTHLY
    elif (85 <= median_diff) and (median_diff <= 95):
        freq = Frequency.QUARTERLY
    elif (175 <= median_diff) and (median_diff <= 190):
        freq = Frequency.SEMIANNUALLY
    elif (360 <= median_diff) and (median_diff <= 370):
        freq = Frequency.ANNUALLY
    else:
        raise ValueError("cannot determine frequency")
    return freq


class InvalidPerilOrder(Exception):
    pass


class TimeSeriesForecast:
    """
    An `xarray` driven time series forecast class.
    Uses xarray.Dataset to represent multidimensional time series

    each variable:
        - must have a `date` dimension (a pandas.DatetimeIndex)
        - may have a `period` dimension (a pandas.Int64Index)
        - may have a `path` dimension (not indexed)

    The `date` dimension are the "as of" dates for any path forecast
    The `period` dimension are path forecasts at integer multiples of `freq` from `date`
    Note that the 0th `period` for any `date` is an observed value, not a prediction

    The `freq` describes both
        - the frequency of the as_of `date`
        - the frequency of the `period`
    and is common to each underlying DataArray.
    Use .align() before initializing a TimeSeriesForecast to change a Dataset's freq

    Examples of suitable DataArrays:
        1. an index's mean forecast of the next 3 months, from today's date
        - 1 asof date, 1 path, 3 periods
        - frequency must be supplied

        2. an index's distributional forecast of the next 3 months, from today's date
        - 1 asof date, N paths, 3 periods
        - frequency must be supplied

        3. an index's last 12 months of observed values
        - 12 asof dates, 1 path, 1 period
        - frequency can be inferred from the asof dates

        4. a collection of historical mean forecasts,
        - each for the next 3 months from each of 12 prior asof dates
        - 12 asof dates, 1 path, 3 periods
        - frequency can be inferred from the asof dates

        5. a collection of historical distributional forecasts,
        - each for the next 3 months from each of 12 prior asof dates
        - 12 asof dates, N paths, 3 periods
        - frequency can be inferred from the asof dates
    """

    def __init__(
        self,
        data: xr.Dataset,
        freq: Frequency,
    ) -> None:
        data.attrs["frequency_name"] = freq.name
        conformed = self._conform_dataset(data, freq)
        self._validate_dataset(conformed, freq)
        self.data = conformed

    @property
    def freq(self) -> Frequency:
        # constructing from the freq name, saved on the attributes of the xr.Dataset,
        # so that nothing is lost when using save() and load_from_netcdf()
        return Frequency.__members__[self.data.frequency_name]

    def _conform_dataset(self, data: xr.Dataset, freq: Frequency) -> xr.Dataset:
        if "period" not in data.dims:
            data = data.expand_dims("period")
        if "path" not in data.dims:
            data = data.expand_dims("path")
        # ensure period dimension is labeled
        if "period" not in data.indexes:
            n_periods = len(data.period)
            periods = np.arange(n_periods)
            data = data.assign_coords({"period": periods})
        # ensure data array is ordered and sorted
        data = data.transpose("date", "period", "path")
        data = data.sortby(["date", "period"])
        return data

    def _validate_dataset(self, data: xr.Dataset, freq: Frequency) -> None:
        assert "date" in data.dims
        assert "period" in data.dims
        assert "path" in data.dims
        assert isinstance(data.indexes["date"], pd.DatetimeIndex)
        # assert isinstance(data.indexes["period"], pd.Int64Index)
        assert len(data.dims) == 3
        assert data.attrs["frequency_name"] == freq.name
        if len(data.date) != 1:
            assert infer_freq(pd.DatetimeIndex(data.date.values)) == freq

    def to_dict(self) -> Dict[str, Any]:
        ds = self.data.copy(deep=True)
        ds.coords["date"] = ds.coords["date"].dt.strftime("%Y-%m-%d")
        res: Dict[str, Any] = ds.to_dict(data=True)
        return res

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> TimeSeriesForecast:
        ds = xr.Dataset.from_dict(data)
        ds.coords["date"] = pd.to_datetime(ds.coords["date"].values)
        ds = ds.astype(float)
        freq = Frequency.__members__[ds.attrs["frequency_name"]]
        return TimeSeriesForecast(ds, freq)

    def save(self, path: str) -> None:
        self.data.attrs["frequency_name"] = self.freq.name
        self.data.to_netcdf(path, mode="w", engine="netcdf4")

    def old_align(
        self,
        new_freq: Frequency,
        new_periods: List[int],
    ) -> TimeSeriesForecast:
        """
        i.e. go from a quarterly time series of quarterly forecasts
            to a monthly time series of monthly forecasts.
        """
        # todo: this isn't 100% correct
        # todo: deprecate this ... see pricing.policy.compute_payout()
        rds = self.data.resample({"date": new_freq.value.code})
        tol = f"{new_freq.value.days_per_period}D"
        new_data = rds.ffill(tolerance=tol)
        new_data = new_data.reindex({"period": new_periods})
        return TimeSeriesForecast(new_data, new_freq)

    @staticmethod
    def align(
        dataset: xr.Dataset,
        existing_freq: Frequency,
        new_freq: Frequency,
        date_fill_type: str = "ffill",
        period_fill_type: str = "ffill",
    ) -> xr.Dataset:
        """
        resample both the as_of dates and the periods of an xr.Dataset
        e.g.
            monthly -> quarterly
            monthly (missing some months) -> monthly (all months)
            quarterly -> monthly
        """

        def resample(
            data: xr.Dataset, target_dim: str, fill_type: str, new_freq: Frequency
        ) -> xr.Dataset:
            rds = data.resample({target_dim: new_freq.value.code})
            adj: xr.Dataset
            if fill_type == "ffill":
                tol = f"{new_freq.value.days_per_period}D"
                adj = rds.ffill(tolerance=tol)
            elif fill_type == "interp":
                adj = rds.interpolate(kind="linear")
            else:
                raise ValueError("fill type not supported")
            return adj

        assert "date" in dataset.dims, "supplied dataset must have a date dimension"
        assert "period" in dataset.dims, "supplied dataset must have a period dimension"

        # for each as_of date, convert the numeric periods
        # to dates, then resample according to the new freq
        resampled_periods = []
        for date in dataset.date.values:
            period_idx = pd.date_range(
                start=date,
                periods=len(dataset.period),
                freq=existing_freq.value.code,
            )
            new_coords = dataset.sel(date=date).assign_coords({"period": period_idx})
            new_periods = resample(new_coords, "period", period_fill_type, new_freq)
            n_periods = range(len(new_periods.period))
            new_coords = new_periods.assign_coords({"period": n_periods})
            resampled_periods.append(new_coords)

        # re-combine the resampled (split) Datasets,
        # one for each as_of date, back into a single Dataset.
        # Then resample the as_of dates according to the new freq
        new_periods_combined = xr.concat(resampled_periods, dim="date")
        new_dates = resample(new_periods_combined, "date", date_fill_type, new_freq)
        # assign the new frequency to an attribute
        new_dates.attrs["frequency_name"] = new_freq.name
        return new_dates

    @staticmethod
    def load_from_netcdf(path: str) -> TimeSeriesForecast:
        """
        Load from a saved netcdf, e.g. TimeSeriesForecast.save()
        """
        with xr.open_dataset(path, engine="netcdf4", chunks="auto") as data:
            data.load()
        assert "frequency_name" in data.attrs
        freq = Frequency.__members__[data.attrs["frequency_name"]]
        return TimeSeriesForecast(data, freq)

    @staticmethod
    def load_from_dataframe(
        data: pd.DataFrame, freq: Optional[Frequency] = None
    ) -> TimeSeriesForecast:
        """
        `data` is a DataFrame with a DatetimeIndex. The columns are variables and the
        index is the asof dates. There is only 1 period. IOW, the data are historical
        observations of each variable (column).
        """
        data = data.astype(float)
        data.index = pd.to_datetime(list(data.index))
        data.index.name = "date"
        data_arrays = {}
        for col in data.columns:
            array = xr.DataArray(
                data[col],
                coords={"date": data.index},
                dims=["date"],
            )
            if freq is None:
                freq = infer_freq(data.index)
            data_arrays[col] = array
        return TimeSeriesForecast(
            xr.Dataset(data_arrays), freq or infer_freq(data.index)
        )

    @staticmethod
    def load_from_array(
        data: np.ndarray,
        variables: List[str],
        dates: pd.DatetimeIndex,
        freq: Frequency,
        periods: Optional[List[int]] = None,
    ) -> TimeSeriesForecast:
        """
        `data` is an array with shape: (dates, [periods], [paths], variables)
        """
        assert data.ndim >= 2
        assert data.ndim <= 4
        if data.ndim == 2:
            # assume missing periods and paths dimensions
            data = data[:, None, None, :]
        elif data.ndim == 3:
            # assume missing paths dimension
            data = data[:, :, None, :]
        assert data.ndim == 4
        if periods is None:
            # assume periods start at 0
            periods = list(range(data.shape[1]))
        assert data.shape[0] == len(dates)
        assert data.shape[1] == len(periods)
        assert data.shape[3] == len(variables)

        da = xr.DataArray(
            data,
            coords={"date": dates, "period": periods, "variable": variables},
            dims=["date", "period", "path", "variable"],
        )
        ds = da.to_dataset("variable")
        return TimeSeriesForecast(ds, freq)

    @staticmethod
    def load_from_tsfs(
        tsfs: List[TimeSeriesForecast], combine_dim: str
    ) -> TimeSeriesForecast:
        assert len(set([tsf.freq for tsf in tsfs])) == 1  # all have same freq
        freq = tsfs[0].freq
        combined: xr.Dataset = xr.concat([tsf.data for tsf in tsfs], dim=combine_dim)
        return TimeSeriesForecast(combined, freq)
