import numpy as np
from forecast_retirement.distributions import Normal, StudentT
from time_series_retirement import Frequency, FrequencyData, TimeSeriesForecast
from transforms import LevelTransform, LogTransform
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
from forecast_retirement.structural import ArimaConfig, ArimaModel
from dataset import DataSet, IndexSpec
from datetime import datetime
import matplotlib.pyplot as plt

class ModelConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class simulator_tool_config(ModelConfig):

    country_currency: str
    current_age: int
    retirement_age: int
    life_expectancy: int
    working_lifespan: int
    retirement_lifespan: int
    forecast_horizon: int
    paths: int
    variance_multiplier: float
    income_tax_rate: float
    post_retirement_tax_rate: float
    expected_long_term_inflation: float
    expected_real_income_growth: float
    usa_contributions_pct_income: str
    financial_series:pd.DataFrame
    post_retirement_spending_needs: pd.Series
    retirement_spending_factor: float
    irs_contribution_limits: Dict
    periods_per_yr: int
    pre_retirement_wgt: pd.DataFrame
    post_retirement_wgt: pd.DataFrame
    fin_mkt_series_list: List

class simulator_tool:

    r"""
    Class that creates object with all properties needed to calculate savings &
    contributions across tax-advantaged and non-advantaged savings accounts.

    Parameters
    ----------
    """

    def __init__(
        self,
        config: simulator_tool_config,
    ):

        self.country_currency = config.country_currency
        self.current_age = config.current_age
        self.retirement_age = config.retirement_age
        self.life_expectancy = config.life_expectancy
        self.working_lifespan = self.retirement_age - self.current_age
        self.retirement_lifespan = self.life_expectancy - self.retirement_age
        self.forecast_horizon = config.forecast_horizon
        self.paths = config.paths
        self.variance_multiplier = config.variance_multiplier
        self.income_tax_rate = config.income_tax_rate
        self.post_retirement_tax_rate = config.post_retirement_tax_rate
        self.expected_long_term_inflation = config.expected_long_term_inflation
        self.real_income_growth = config.expected_real_income_growth
        self.income_growth_rate = (
            self.expected_long_term_inflation + self.real_income_growth
        )
        self.usa_contributions_pct_income = config.usa_contributions_pct_income
        self.financial_series = config.financial_series
        self.post_retirement_spending_needs = config.post_retirement_spending_needs
        self.retirement_spending_factor = config.retirement_spending_factor
        self.irs_contribution_limits = pd.DataFrame(
            {k: v for k, v in config.irs_contribution_limits.items()}
        ).T
        self.periods_per_yr = config.periods_per_yr
        self.pre_retirement_wgt = config.pre_retirement_wgt
        self.post_retirement_wg = config.post_retirement_wgt
        self.fin_mkt_series_list = config.fin_mkt_series_list
        self._datetime_index()

    def simulate(self) -> TimeSeriesForecast:
        # 1. Run Simulation

        """Note_run univariate T across each of the following variables to generate
        N simulated paths for each variable:
        i)"sp5 returns";
        ii)"tbill returns";
        iii)"tbond returns";
        iv)"corp_bond returns";
        v)"real estate returns";
        vi)"private equity returns";
        vii)"inflation";
        """

        tsf_list = []

        # list the six features to sim
        self.fin_mkt_series_list.append("inflation")
        sim_fields = self.fin_mkt_series_list

        fin_mkt_series_sim_list = []
        for ix in range(0, len(self.fin_mkt_series_list)):
            if "inflation" not in self.fin_mkt_series_list[ix]:
                mu = self.financial_series[self.fin_mkt_series_list[ix]].mean() / \
                self.periods_per_yr
                sd = self.financial_series[self.fin_mkt_series_list[ix]].std() / (
                        self.periods_per_yr ** 0.5)
                normal_dist = Normal(mu, sd)
                normal_dist = normal_dist.sample(
                    (1 + self.life_expectancy - self.current_age) * self.periods_per_yr
                )
                fin_mkt_series_sim_list.append([normal_dist])
            else:
                # annual inflation
                loc = self.expected_long_term_inflation / self.periods_per_yr
                scale = 0.01
                dof = 5
                t_dist = StudentT(loc, scale, dof)
                t_dist = t_dist.sample(
                    (1 + self.life_expectancy - self.current_age) * self.periods_per_yr
                )
                fin_mkt_series_sim_list.append([t_dist])

        # create object with our variables to be simulated
        sim_arr = np.concatenate(
            (
                fin_mkt_series_sim_list
            ),
            axis=0,
        )
        sim_df = pd.DataFrame(sim_arr.T, columns=sim_fields, index=self.datetime_index)

        # model the trend of each of the four series via ARIMA
        # 2a. fit a model

        for sim_feature in range(len(sim_fields)):
            sim_name = sim_fields[sim_feature]
            sim_series = sim_df[[sim_name]]
            spec = IndexSpec(name=sim_name, transform=LevelTransform())
            dataset = DataSet([spec], freq=self.frequency)

            arima_config = ArimaConfig(p=1, d=0, q=1)
            arima_model = ArimaModel(dataset, arima_config)
            arima_model.fit(sim_series)

            """Note: Predict from seed data. Use 20% latest obs from training as seed 
             to forecast."""

            seed = sim_series
            forecast = arima_model.forecast(
                seed, n_periods=self.forecast_horizon, n_paths=self.paths
            )

            # 5c. normalize the forecast to the desired mean for income and spend
            # dates, periods, paths
            if sim_name == "T.Bill":
                forecast.data["3-Month T.Bill_returns"] = np.clip(forecast.data[
                                                           "3-Month T.Bill_returns"],0,None)
            tsf = TimeSeriesForecast(forecast.data, freq=self.frequency)
            tsf_list.append(tsf)

        """Note: combine the time series forecasts objects into one"""
        for tsf in range(1, len(tsf_list)):
            tsf_list[0].data[sim_fields[tsf]] = tsf_list[tsf].data[sim_fields[tsf]]
        tsf = tsf_list[0]

        # 5e. assign tsf to be named sim
        sim = tsf
        self.sim = sim

        return sim

    def _datetime_index(self) -> pd.DatetimeIndex:
        now = datetime.utcnow()
        start_date = pd.to_datetime(f"{now.strftime('%Y%m%d')}") + pd.offsets.MonthEnd(
            0
        )
        end_date = start_date + pd.offsets.MonthEnd(
            1+ (self.life_expectancy - self.current_age) * self.periods_per_yr *
            12/self.periods_per_yr
        )

        if self.periods_per_yr == 26:
            freq = Frequency.BIWEEKLY.value.code
            self.frequency = Frequency.BIWEEKLY
        elif self.periods_per_yr == 12:
            freq = Frequency.MONTHLY.value.code
            self.frequency = Frequency.MONTHLY
        elif self.periods_per_yr == 4:
            freq = Frequency.QUARTERLY.value.code
            self.frequency = Frequency.QUARTERLY
        elif self.periods_per_yr == 2:
            freq = Frequency.SEMIANNUALLY.value.code
            self.frequency = Frequency.SEMIANNUALLY
        elif self.periods_per_yr == 1:
            freq = Frequency.ANNUALLY.value.code
            self.frequency = Frequency.ANNUALLY
        self.datetime_index = pd.date_range(start=start_date, end=end_date, freq=freq)
