from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from forecast.distributions import (  # UniformPiecewise,
    Beta,
    Binomial,
    Gamma,
    Normal,
    Poisson,
    StudentT,
    UniformInt,
    Univariate,
)

DISTRIBUTIONS = {
    "Beta": Beta,
    "Binomial": Binomial,
    "Gamma": Gamma,
    "Normal": Normal,
    "Poisson": Poisson,
    "StudentT": StudentT,
    "UniformInt": UniformInt,
    # "UniformPiecewise": UniformPiecewise([((0, 5), 0.5), ((5, 10), 0.25), 0.5]),
}


@dataclass
class Shock:
    freq: float  # probability of occurrence over the total policy period
    period: Univariate  # 0th-indexed period in which shock occurs
    severity: Univariate  # units = % deviation from trend
    duration: Univariate  # units = periods

    def sample(
        self,
        n_paths: int,
        conditional_peril: Optional[PerilForecast] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Notes: conditional_peril should generally be passed through Peril.forecast()
        """

        if conditional_peril:
            n_periods = len(conditional_peril.result)
            occurs_conditional_on = conditional_peril.shock["occurs"]
            period_conditional_on = conditional_peril.shock["period"]
            occurs = np.where(
                occurs_conditional_on,
                np.random.choice(
                    (True, False), size=n_paths, p=(self.freq, 1 - self.freq)
                ),
                occurs_conditional_on,
            )
            period = np.clip(
                self.period.sample(n_paths).astype(int) + period_conditional_on,
                period_conditional_on,
                n_periods - 1,
            )
        else:
            occurs = np.random.choice(
                (True, False), size=n_paths, p=(self.freq, 1 - self.freq)
            )
            period = self.period.sample(n_paths).astype(int)
        sev = self.severity.sample(n_paths)
        dur = self.duration.sample(n_paths).astype(int)
        return {"occurs": occurs, "period": period, "severity": sev, "duration": dur}


@dataclass
class Recovery:
    """
    Note: A relapse is NOT processed sequentially after a recovery begins;
    instead, if a relapse is drawn it simply extends the duration
    and increases the severity of the recovery. Thus the `period` of the relapse
    is ignored.
    """

    duration: Univariate  # number of periods to return to trend
    level: Univariate  # percent of trend to return to, at end of duration
    # relapse: Shock  # a _specific_ shock that can occur during recovery

    def sample(self, n_paths: int) -> Dict[str, np.ndarray]:
        duration = self.duration.sample(n_paths).astype(int)
        level = self.level.sample(n_paths)
        # relapse = self.relapse.sample(n_paths)
        # relapse_bool = relapse["occurs"] == 1
        # extend duration and level by any relapses
        # duration[relapse_bool] += relapse["duration"][relapse_bool]
        # level[relapse_bool] *= 1 - relapse["severity"][relapse_bool]
        return {"duration": duration, "level": level}


class InvalidShockPeriods(Exception):
    pass


class InvalidShockDuration(Exception):
    pass


class InvalidPerilOrder(Exception):
    pass


@dataclass
class PerilForecast:
    peril_name: str
    result: np.ndarray  # shape=(n_periods, n_paths); values = % of trend for the index
    shock: Dict[str, np.ndarray]  # result of Shock.sample()
    recovery: Dict[str, np.ndarray]  # result of Recovery.sample(

    @property
    def shock_periods(self) -> np.ndarray:
        periods = self.shock["period"].copy()
        periods[self.shock["occurs"] == 0] = -1  # no shock => encode period = -1
        return periods


class Peril:
    def __init__(
        self,
        shock: Shock,
        recovery: Recovery,
        name: str,
        conditional_on: str = None,
    ) -> None:
        self.shock = shock
        self.recovery = recovery
        self.name = name
        self.conditional_on: Optional[str] = conditional_on

    def _validate_periods(self, periods: np.ndarray, n_periods: int) -> None:
        if np.any(periods > (n_periods - 1)):
            raise InvalidShockPeriods(
                f"Shock period as large as {periods.max()} exceeds n_periods; "
                f"cap Shock.period distribution at n_periods."
            )

    def _validate_durations(self, durations: np.ndarray) -> None:
        if np.any(durations <= 0):
            raise InvalidShockDuration(
                f"Shock durations must be at least 1 period; "
                f"{durations.min()} is too small; floor Shock.duration at 1"
            )

    def forecast(
        self,
        n_periods: int,
        n_paths: int,
        conditional_peril: Optional[PerilForecast] = None,
        allow_permanent_shocks: bool = True,
    ) -> PerilForecast:
        """
        Produces a `result` matrix of shape (n_periods, n_paths) with element values
        for un-shocked periods == 1 and % severity/recovery otherwise
        Properties of a result:
        - All shocks are downward; all recoveries upward (# todo: allow opposite)
        - Shocks immediately induce their `severity`, and remain there for `duration`
        - if `allow_permanent_shocks` == False, then recoveries reach `level` at end of
            their `duration` but then go back to the orig values, i.e. "1", thereafter.
            If == True, then recoveries reach `level` at end of `duration` and remain
            there through n_periods.
        - The path from shock.severity to recovery.level is spread evenly
            over the recovery duration
        A note on iteration method:
            Shocks / recoveries are applied over all simulations with matrix calcs,
            but sequentially by iterating over the durations of shocks and recoveries.
            Two shocks might be:
            - sim_1: @ period 2, applied for _3_ periods
            - sim_2: @ period 2, applied for _4_ periods
            So you'd want to edit sim_1 periods 2, 3, 4 and sim_2 periods 2, 3, 4, 5.
            Because of these different lengths we can't apply shocks with pure slicing.
        A note on conditionality:
            When the optional attribute 'conditional_on' is not None, the peril's
            occurrence and periods of occurrence are explicitly conditional on
            another named peril.
        """

        # 1. sample shock and recovery distributions with option for conditionality

        if conditional_peril:
            shock = self.shock.sample(n_paths, conditional_peril)
        else:
            shock = self.shock.sample(n_paths)
        (active,) = np.where(shock["occurs"])  # indices of paths with shocks
        recov = self.recovery.sample(n_paths)
        periods = shock["period"].copy()
        shock_len = shock["duration"].copy()
        self._validate_periods(periods[active], n_periods)
        self._validate_durations(shock_len[active])

        # 2. apply shocks

        result = np.ones((n_periods, n_paths))

        while len(active) > 0:
            result[periods[active], active] *= 1 - shock["severity"][active]
            shock_len[active] -= 1
            periods[active] += 1
            (active_check,) = np.where((shock_len > 0) & (periods < n_periods))
            active = np.intersect1d(active, active_check)

        # 3. apply recoveries (already net of relapses)

        recov_len = recov["duration"].copy()
        target = np.tile(recov["level"], (n_periods, 1))  # (n_periods, n_paths)

        # reset active index, w/ exceptions
        (active,) = np.where(shock["occurs"])
        (active_check,) = np.where((recov_len > 0) & (periods < n_periods))
        active = np.intersect1d(active, active_check)

        while len(active) > 0:
            prior = result[periods[active] - 1, active]
            gap = target[periods[active], active] - prior
            recov_this_period = gap / recov_len[active]
            result[periods[active], active] = prior + recov_this_period
            recov_len[active] -= 1
            periods[active] += 1
            (active_check,) = np.where((recov_len > 0) & (periods < n_periods))
            active = np.intersect1d(active, active_check)

        # 4. roll shocks to the final period

        if allow_permanent_shocks:

            (active,) = np.where(shock["occurs"])  # reset
            active = np.intersect1d(active, np.where(periods < n_periods))

            while len(active) > 0:
                result[periods[active], active] = result[periods[active] - 1, active]
                periods += 1
                active = np.intersect1d(active, np.where(periods < n_periods))

        return PerilForecast(
            peril_name=self.name, result=result, shock=shock, recovery=recov
        )


def load_perils_from_csv(path: str) -> List[Peril]:
    df_peril = pd.read_csv(path)
    df_peril["component_name"].replace({"peril": "freq"}, inplace=True)

    dists = []
    grps = df_peril.groupby(["idx_ds", "prefix", "component_name"])
    for key in grps.groups.keys():
        df = grps.get_group(key)

        if "diststring" in df.param.values:
            dist = DISTRIBUTIONS[df[df.param == "diststring"]["value"].iloc[0]]

            df = df[df.param != "diststring"]
            params = {row.param: pd.to_numeric(row.value) for row in df.itertuples()}

            dist = dist(**params)

        else:
            dist = df["value"].iloc[0]

        row = df.drop(["param", "value"], axis=1).to_dict(orient="records")[0]

        dist = copy.deepcopy(dist)
        row["dist"] = dist

        dists.append(row)
    df_dists = pd.DataFrame(dists)

    perils = []
    grps = df_dists.groupby(["idx_ds"])
    for key in grps.groups.keys():
        df = grps.get_group(key)

        # Build shock
        df_sub = df[df.prefix == "shock"]
        params = {row.component_name: row.dist for row in df_sub.itertuples()}
        shock = Shock(**params)

        # Build relapse
        df_sub = df[df.prefix == "relapse"]
        params = {row.component_name: row.dist for row in df_sub.itertuples()}
        relapse = Shock(**params)

        # Build recovery
        df_sub = df[df.prefix == "recovery"]
        params = {row.component_name: row.dist for row in df_sub.itertuples()}
        params["relapse"] = relapse
        recovery = Recovery(**params)

        # Build peril
        peril = Peril(shock, recovery, df_sub["peril_name"].iloc[0])

        perils.append(peril)

    return perils
