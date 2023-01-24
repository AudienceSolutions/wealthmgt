"""
methods to transform time series
todo: multiple transforms per series in DataSet
todo: IndexSpex and DataSet support for index labels
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import xarray as xr

TRANSFORM_CLASSES = [
    "ValueTransform",
    "LevelTransform",
    "LogTransform",
    "LogitTransform",
    "StandardizeTransform",
    "MaxAbsTransform",
    "DifferenceTransform",
    "PercentDifferenceTransform",
    "CumulativeTransform",
    "TrailingAvgTransform",
    "LagTransform",
    "ProductTransform",
    "CompositeTransform",
]


class Transform(ABC):
    """
    Transform a time-series
    can be either a pd.Series or pd.DataFrame, with a DatetimeIndex
    """

    name: str = "transform"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        # method to serialize object
        res = dict(**self.kwargs)
        # special cases
        if "transforms" in res:
            res["transforms"] = [t.to_dict() for t in res["transforms"]]
        res["class"] = self.__class__.__name__
        res["name"] = self.name
        return res

    @staticmethod
    def from_dict(transform_dict: Dict[str, Any]) -> Transform:
        # copy dict
        transform_dict = dict(**transform_dict)
        class_name = transform_dict.pop("class")
        assert class_name in TRANSFORM_CLASSES
        transform_class: Type[Transform] = eval(class_name)
        name = transform_dict.pop("name")
        # special cases:
        if "transforms" in transform_dict:
            transform_dict["transforms"] = [
                Transform.from_dict(t) for t in transform_dict["transforms"]
            ]
        transform: Transform = transform_class(**transform_dict)
        transform.name = name
        return transform

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def fit(self, data: pd.Series) -> None:
        pass

    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def inv_transform(self, data: pd.Series) -> pd.Series:
        pass

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError("transform_xr not implemented")

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError("inv_transform_next not implemented")


class ValueTransform(Transform):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__(value=value)
        self.value = value

    def fit(self, data: pd.Series) -> None:
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        return pd.Series(self.value, index=data.index)

    def inv_transform(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError("inv_transform not implemented")

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = xr.full_like(data, self.value)
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError("inv_transform_xr not implemented")


class LevelTransform(Transform):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__(scale=scale)
        self.scale = scale

    def fit(self, data: pd.Series) -> None:
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        return self.scale * data

    def inv_transform(self, data: pd.Series) -> pd.Series:
        return data / self.scale

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = self.scale * data
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data / self.scale
        return res


class LogTransform(Transform):
    def __init__(self, eps: float = 0.0):
        super().__init__(eps=eps)
        self.eps = eps

    def fit(self, data: pd.Series) -> None:
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        return np.log(self.eps + data)

    def inv_transform(self, data: pd.Series) -> pd.Series:
        return np.exp(data) - self.eps

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res = xr.DataArray(np.log(self.eps + data))
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res = xr.DataArray(np.exp(data) - self.eps)
        return res


class LogitTransform(Transform):
    def __init__(self, eps: float = 0.0):
        super().__init__(eps=eps)
        self.eps = eps

    def fit(self, data: pd.Series) -> None:
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        res = self.eps + data * (1 - 2 * self.eps)
        res = np.log(res / (1 - res))
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        res = 1 / (1 + np.exp(-data))
        res = (res - self.eps) / (1 - 2 * self.eps)
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res = xr.DataArray(self.eps + data * (1 - 2 * self.eps))
        res = xr.DataArray(np.log(res / (1 - res)))
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res = xr.DataArray(1 / (1 + np.exp(-data)))
        res = (res - self.eps) / (1 - 2 * self.eps)
        return res


class StandardizeTransform(Transform):
    avg: Optional[float] = None
    std: Optional[float] = None

    def fit(self, data: pd.Series) -> None:
        self.avg = data.mean()
        self.std = data.std()

    def transform(self, data: pd.Series) -> pd.Series:
        if (self.avg is None) or (self.std is None):
            raise ValueError("must call fit before evaluating transform")
        res = (data - self.avg) / self.std
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        if (self.avg is None) or (self.std is None):
            raise ValueError("must call fit before evaluating transform")
        res = self.std * data + self.avg
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        if (self.avg is None) or (self.std is None):
            raise ValueError("must call fit before evaluating transform")
        res: xr.DataArray = (data - self.avg) / self.std
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        if (self.avg is None) or (self.std is None):
            raise ValueError("must call fit before evaluating transform")
        res: xr.DataArray = self.std * data + self.avg
        return res


class MaxAbsTransform(Transform):
    scale: Optional[float] = None

    def fit(self, data: pd.Series) -> None:
        self.scale = np.abs(data).max()

    def transform(self, data: pd.Series) -> pd.Series:
        if self.scale is None:
            raise ValueError("must call fit before evaluating transform")
        res = data / self.scale
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        if self.scale is None:
            raise ValueError("must call fit before evaluating transform")
        res = self.scale * data
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        if self.scale is None:
            raise ValueError("must call fit before evaluating transform")
        res: xr.DataArray = data / self.scale
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        if self.scale is None:
            raise ValueError("must call fit before evaluating transform")
        res: xr.DataArray = self.scale * data
        return res


class DifferenceTransform(Transform):
    fit_data: Optional[pd.Series] = None

    def __init__(self, lag: int = 1):
        super().__init__(lag=lag)
        self.lag = lag

    def fit(self, data: pd.Series) -> None:
        self.fit_data = data.copy()

    def transform(self, data: pd.Series) -> pd.Series:
        res = data - data.shift(self.lag)
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        if self.fit_data is None:
            raise ValueError("must call fit before evaluating transform")
        res = data.copy()
        for i in range(self.lag):
            df = res.iloc[i :: self.lag].cumsum()
            bias = (self.fit_data - df).mean()
            res.iloc[i :: self.lag] = df + bias
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data - data.shift({"period": self.lag})
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError(
            "inverses of stateful transforms not implemented for xarray"
        )


class PercentDifferenceTransform(Transform):
    fit_data: Optional[pd.Series] = None

    def __init__(self, lag: int = 1, add_one: bool = False):
        super().__init__(lag=lag, add_one=add_one)
        self.lag = lag
        self.add_one = add_one

    def fit(self, data: pd.Series) -> None:
        self.fit_data = data.copy()

    def transform(self, data: pd.Series) -> pd.Series:
        lagged = data.shift(self.lag)
        res = (data - lagged) / np.abs(lagged)
        if self.add_one:
            res = res + 1
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        if self.fit_data is None:
            raise ValueError("must call fit before evaluating transform")
        res = data.copy()
        if self.add_one:
            res = res - 1
        for i in range(self.lag):
            df = (1 + res.iloc[i :: self.lag]).cumprod()
            bias = (self.fit_data / df).mean()
            res.iloc[i :: self.lag] = df * bias
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        lagged = data.shift({"period": self.lag})
        res: xr.DataArray = (data - lagged) / np.abs(lagged)
        if self.add_one:
            res = res + 1
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError(
            "inverses of stateful transforms not implemented for xarray"
        )


class CumulativeTransform(Transform):
    fit_data: Optional[pd.Series] = None

    def __init__(self, trailing: int = 1) -> None:
        super().__init__(trailing=trailing)
        self.trailing = trailing

    def fit(self, data: pd.Series) -> None:
        self.fit_data = data.copy()

    def transform(self, data: pd.Series) -> pd.Series:
        res = data.rolling(self.trailing).sum()
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        if self.fit_data is None:
            raise ValueError("must call fit before evaluating transform")
        res = data - data.shift()
        for i in range(self.trailing):
            df = res.iloc[i :: self.trailing].cumsum()
            bias = (self.fit_data - df).mean()
            res.iloc[i :: self.trailing] = df + bias
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data.rolling({"period": self.trailing}).sum()
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError(
            "inverses of stateful transforms not implemented for xarray"
        )


class TrailingAvgTransform(Transform):
    fit_data: Optional[pd.Series] = None

    def __init__(self, trailing: int = 1) -> None:
        super().__init__(trailing=trailing)
        self.trailing = trailing

    def fit(self, data: pd.Series) -> None:
        self.fit_data = data.copy()

    def transform(self, data: pd.Series) -> pd.Series:
        res = data.rolling(self.trailing).mean()
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        if self.fit_data is None:
            raise ValueError("must call fit before evaluating transform")
        res = self.trailing * (data - data.shift())
        for i in range(self.trailing):
            df = res.iloc[i :: self.trailing].cumsum()
            bias = (self.fit_data - df).mean()
            res.iloc[i :: self.trailing] = df + bias
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data.rolling({"period": self.trailing}).mean()
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError(
            "inverses of stateful transforms not implemented for xarray"
        )


class LagTransform(Transform):
    def __init__(self, lag: int = 1):
        super().__init__(lag=lag)
        self.lag = lag

    def fit(self, data: pd.Series) -> None:
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        res = data.shift(self.lag)
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        res = data.shift(-self.lag)
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data.shift({"period": self.lag})
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data.shift({"period": -self.lag})
        return res


class ProductTransform(Transform):
    fit_data: Optional[pd.Series] = None

    def __init__(self, transforms: List[Transform]) -> None:
        super().__init__(transforms=transforms)
        self.transforms = transforms

    def fit(self, data: pd.Series) -> None:
        self.fit_data = data.copy()
        for transform in self.transforms:
            transform.fit(data)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        res: Optional[pd.Series] = None
        for transform in self.transforms:
            if res is None:
                res = transform.fit_transform(data)
            else:
                res = res * transform.fit_transform(data)
        return res

    def transform(self, data: pd.Series) -> pd.Series:
        res: Optional[pd.Series] = None
        for transform in self.transforms:
            if res is None:
                res = transform.transform(data)
            else:
                res = res * transform.transform(data)
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError("cannot compute inverse of MultiplicationTransform")

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = xr.DataArray(1)
        for transform in self.transforms:
            res = res * transform.transform_xr(data)
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError("cannot compute inverse of MultiplicationTransform")


class CompositeTransform(Transform):
    def __init__(self, transforms: List[Transform]) -> None:
        super().__init__(transforms=transforms)

        self.transforms = transforms

    def fit(self, data: pd.Series) -> None:
        res = data
        for transform in self.transforms:
            res = transform.fit_transform(res)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        res = data
        for transform in self.transforms:
            res = transform.fit_transform(res)
        return res

    def transform(self, data: pd.Series) -> pd.Series:
        res = data
        for transform in self.transforms:
            res = transform.transform(res)
        return res

    def inv_transform(self, data: pd.Series) -> pd.Series:
        res = data
        for transform in self.transforms[::-1]:
            res = transform.inv_transform(res)
        return res

    def transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data
        for transform in self.transforms:
            res = transform.transform_xr(res)
        return res

    def inv_transform_xr(self, data: xr.DataArray) -> xr.DataArray:
        res: xr.DataArray = data
        for transform in self.transforms[::-1]:
            res = transform.inv_transform_xr(res)
        return res


def get_named_transforms() -> Dict[str, Transform]:
    named_transforms = {
        "LEVEL": LevelTransform(scale=1),
        "MOM_CHANGE": CompositeTransform(
            [PercentDifferenceTransform(lag=1), LevelTransform(scale=100)]
        ),
        "QOQ_CHANGE": CompositeTransform(
            [PercentDifferenceTransform(lag=3), LevelTransform(scale=100)]
        ),
        "YOY_CHANGE": CompositeTransform(
            [PercentDifferenceTransform(lag=12), LevelTransform(scale=100)]
        ),
        "CUMUL_T12M_MOM_CHANGE": CompositeTransform(
            [
                CumulativeTransform(trailing=12),
                PercentDifferenceTransform(lag=1),
                LevelTransform(scale=100),
            ]
        ),
        "CUMUL_T12M_QOQ_CHANGE": CompositeTransform(
            [
                CumulativeTransform(trailing=12),
                PercentDifferenceTransform(lag=3),
                LevelTransform(scale=100),
            ]
        ),
        "CUMUL_T12M_YOY_CHANGE": CompositeTransform(
            [
                CumulativeTransform(trailing=12),
                PercentDifferenceTransform(lag=12),
                LevelTransform(scale=100),
            ]
        ),
        "T12M_AVG_MOM_CHANGE": CompositeTransform(
            [
                PercentDifferenceTransform(lag=1),
                TrailingAvgTransform(trailing=12),
                LevelTransform(scale=100),
            ]
        ),
        "T12M_AVG_QOQ_CHANGE": CompositeTransform(
            [
                PercentDifferenceTransform(lag=3),
                TrailingAvgTransform(trailing=12),
                LevelTransform(scale=100),
            ]
        ),
        "T12M_AVG_YOY_CHANGE": CompositeTransform(
            [
                PercentDifferenceTransform(lag=12),
                TrailingAvgTransform(trailing=12),
                LevelTransform(scale=100),
            ]
        ),
        "T12M_AVG": TrailingAvgTransform(trailing=12),
        "T24M_AVG": TrailingAvgTransform(trailing=24),
        "T12M_AVG_YOY_CHANGE_FORECAST": CompositeTransform(
            [
                LagTransform(lag=1),
                ProductTransform(
                    [
                        CompositeTransform(
                            [
                                PercentDifferenceTransform(lag=12, add_one=True),
                                TrailingAvgTransform(trailing=12),
                            ]
                        ),
                        LagTransform(lag=12),
                    ]
                ),
            ]
        ),
        "T24M_AVG_YOY_CHANGE_FORECAST": CompositeTransform(
            [
                LagTransform(lag=1),
                ProductTransform(
                    [
                        CompositeTransform(
                            [
                                PercentDifferenceTransform(lag=12, add_one=True),
                                TrailingAvgTransform(trailing=24),
                            ]
                        ),
                        LagTransform(lag=12),
                    ]
                ),
            ]
        ),
    }

    for key, val in named_transforms.items():
        val.name = key

    return named_transforms
