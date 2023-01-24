from __future__ import annotations

from typing import Dict, Hashable, List

import pandas as pd
import xarray as xr
from pydantic import BaseModel

from time_series import Frequency
from transforms import LevelTransform, Transform


class IndexSpec(BaseModel):
    name: str
    transform: Transform
    exogenous: bool = False

    class Config:
        arbitrary_types_allowed = True


class DataSet:
    def __init__(self, indices: List[IndexSpec], freq: Frequency) -> None:
        self.indices = indices
        self.variables = [str(ix.name) for ix in self.indices]
        self.endog = [str(ix.name) for ix in self.indices if not ix.exogenous]
        self.exog = [str(ix.name) for ix in self.indices if ix.exogenous]
        self.freq = freq

    @staticmethod
    def from_names(
        variables: List[str], freq: Frequency, exogenous: List[str] = []
    ) -> DataSet:
        indices: List[IndexSpec] = []
        for name in variables:
            index = IndexSpec(
                name=name,
                transform=LevelTransform(),
                exogenous=(name in exogenous),
            )
            indices.append(index)
        return DataSet(indices, freq)

    def fit(self, data: pd.DataFrame) -> None:
        for index in self.indices:
            assert index.name in data.columns
            df = data[index.name].astype(float)
            index.transform.fit(df)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        res: Dict[str, pd.Series] = {}
        for index in self.indices:
            assert index.name in data.columns
            df = data[index.name].astype(float)
            res[index.name] = index.transform.transform(df)
        return pd.DataFrame(res)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        res: Dict[str, pd.Series] = {}
        for index in self.indices:
            assert index.name in data.columns
            df = data[index.name].astype(float)
            res[index.name] = index.transform.fit_transform(df)
        return pd.DataFrame(res)

    def inv_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        res: Dict[str, pd.Series] = {}
        for index in self.indices:
            if index.name in data.columns:
                df = data[index.name].astype(float)
                res[index.name] = index.transform.inv_transform(df)
        return pd.DataFrame(res)

    def transform_xr(self, data: xr.Dataset) -> xr.Dataset:
        res: Dict[Hashable, xr.DataArray] = {}
        for i, index in enumerate(self.indices):
            if index.name in data:
                val = index.transform.transform_xr(data[index.name])
                res[index.name] = val
        return xr.Dataset(res, attrs=data.attrs)

    def inv_transform_xr(self, data: xr.Dataset) -> xr.Dataset:
        res: Dict[Hashable, xr.DataArray] = {}
        for i, index in enumerate(self.indices):
            if index.name in data:
                val = index.transform.inv_transform_xr(data[index.name])
                res[index.name] = val
        return xr.Dataset(res, attrs=data.attrs)

