import json
from typing import Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel
from usa_retirement_plan import usa_plan, usa_plan_config


def class_usa_config(self):
    usa_config = usa_plan_config(
        country_currency=self.country_currency,
        irs_contribution_limits=self.irs_contribution_limits,
        non_tax_advantaged_savings_balance=self.non_tax_advantaged_savings_balance,
        current_age=self.current_age,
        life_expectancy=self.life_expectancy,
        expected_long_term_inflation=self.expected_long_term_inflation,
        income_growth_rate=self.income_growth_rate,
        annual_pretax_income=self.annual_pretax_income,
        annual_aftertax_income=self.annual_aftertax_income,
        pre_retirement_tax_rate=self.income_tax_rate,
        post_retirement_tax_rate=self.post_retirement_tax_rate,
        usa_poverty_line_spending=self.usa_poverty_line_spending,
        periods_per_yr=self.periods_per_yr,
    )
    usa_plan_detail = usa_plan(usa_config)

    return usa_plan_detail


class ModelConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class investor_selector_config(ModelConfig):

    country_currency: str
    current_age: int
    retirement_age: float
    life_expectancy: int
    income_tax_rate: float
    post_retirement_tax_rate: float
    expected_long_term_inflation: float
    expected_real_income_growth: float
    annual_pretax_income: float
    usa_contributions_pct_income: str
    pre_retirement_investment_expectations: pd.DataFrame
    post_retirement_investment_expectations: pd.DataFrame
    retirement_spending_factor: float
    irs_contribution_limits: Dict
    usa_poverty_line_spending: float
    periods_per_yr: int


class investor_selector:

    r"""
    Class that creates object with all properties needed to calculate savings &
    contributions across tax-advantaged and non-advantaged savings accounts.

    Parameters
    ----------
    """

    def __init__(
        self,
        config: investor_selector_config,
    ):

        self.country_currency = config.country_currency
        self.current_age = config.current_age
        self.retirement_age = config.retirement_age
        self.life_expectancy = config.life_expectancy
        self.income_tax_rate = config.income_tax_rate
        self.post_retirement_tax_rate = config.post_retirement_tax_rate
        self.expected_long_term_inflation = config.expected_long_term_inflation
        self.real_income_growth = config.expected_real_income_growth
        self.income_growth_rate = (
            self.expected_long_term_inflation + self.real_income_growth
        )
        self.annual_pretax_income = config.annual_pretax_income
        self.annual_aftertax_income = self.annual_pretax_income * (
            1 - self.income_tax_rate
        )
        self.usa_contributions_pct_income = config.usa_contributions_pct_income
        self.pre_retirement_investment_expectations = (
            config.pre_retirement_investment_expectations
        )
        self.post_retirement_investment_expectations = (
            config.post_retirement_investment_expectations
        )
        self.retirement_spending_factor = config.retirement_spending_factor
        self.irs_contribution_limits = pd.DataFrame(
            {k: v for k, v in config.irs_contribution_limits.items()}
        ).T
        self.usa_poverty_line_spending = config.usa_poverty_line_spending
        self.periods_per_yr = config.periods_per_yr

    def calc_pre_retirement_savings(
        self,
        simulation: bool,
    ) -> pd.DataFrame:

        if "usa" in self.country_currency:
            usa_plan_detail = class_usa_config(self)

        """Note: structure object for implying portfolio ann. expected ret"""
        pre_retirement_investment_expectations_df = (
            self.pre_retirement_investment_expectations
        )

        if not simulation:
            # imply ann rate of ret on pre_retirement allocation mix
            pre_retirement_investment_expectations_df[
                "wt"
            ] = pre_retirement_investment_expectations_df["wt"].astype(float)
            pre_retirement_investment_expectations_df[
                "ret"
            ] = pre_retirement_investment_expectations_df["ret"].astype(float)
            self.pre_retirement_investment_rate_of_ret = (
                pre_retirement_investment_expectations_df["wt"].dot(
                    pre_retirement_investment_expectations_df["ret"]
                )
            )

        if "usa" in self.country_currency:
            if isinstance(self.usa_contributions_pct_income, str):
                self.usa_contributions_pct_income = (
                    self.usa_contributions_pct_income.replace("'", '"')
                )
                """Note: read-in dictionary of expectations via json"""
                self.usa_contributions_pct_income = json.loads(
                    self.usa_contributions_pct_income
                )
                usa_contributions_pct_income_df = pd.DataFrame(
                    {k: [v] for k, v in self.usa_contributions_pct_income.items()}
                )
                usa_contributions_pct_income_df = usa_contributions_pct_income_df.apply(
                    pd.to_numeric, errors="coerce"
                )
                self.usa_contributions_pct_income_contrib_deficiency = (
                    usa_contributions_pct_income_df.copy()
                )

            elif isinstance(self.usa_contributions_pct_income, dict):
                usa_contributions_pct_income_df = pd.DataFrame(
                    {k: [v] for k, v in self.usa_contributions_pct_income.items()}
                )
                usa_contributions_pct_income_df = usa_contributions_pct_income_df.apply(
                    pd.to_numeric, errors="coerce"
                )
                self.usa_contributions_pct_income_contrib_deficiency = (
                    usa_contributions_pct_income_df.copy()
                )

            if simulation:
                (
                    pre_retirement_savings_list,
                    sim_pre_retirement_investment_ret_period,
                ) = usa_plan_detail.simul_build_usa_pre_retirement_df(
                    usa_contributions_pct_income_df,
                    self.sim_pre_retirement_inflation,
                    self.sim_pre_retirement_income,
                    self.retirement_age,
                    self.pre_retirement_investment_expectations,
                    self.sim_pre_retire_invest_ret_dict_list,
                    self.periods_per_yr,
                )
                self.sim_pre_retirement_investment_ret_period = (
                    sim_pre_retirement_investment_ret_period
                )
            else:
                (
                    pre_retirement_savings_list,
                    income_growth_rate_factor,
                ) = usa_plan_detail.build_usa_pre_retirement_df(
                    usa_contributions_pct_income_df,
                    self.retirement_age,
                    self.pre_retirement_investment_rate_of_ret,
                    self.periods_per_yr,
                )

        # if not contrib_deficiency and not retire_age_adjust:
        self.pre_retirement_savings_df = pd.DataFrame(pre_retirement_savings_list)

        if simulation:
            self.aftertax_income_at_retirement = self.pre_retirement_savings_df[
                "periodic_after_tax_income"
            ][-1:]
            self.pretax_income_at_retirement = self.pre_retirement_savings_df[
                "periodic_pre_tax_income"
            ][-1:]
        else:
            self.aftertax_income_at_retirement = (
                self.annual_aftertax_income
                / self.periods_per_yr
                * income_growth_rate_factor
            )
            self.pretax_income_at_retirement = (
                self.annual_pretax_income
                / self.periods_per_yr
                * income_growth_rate_factor
            )

    def calc_post_retirement_savings(
        self,
        retirement_age: int,
        simulation: bool,
    ) -> pd.DataFrame:

        if "usa" in self.country_currency:
            usa_plan_detail = class_usa_config(self)
            poverty_line_spending = self.usa_poverty_line_spending
        pre_inflation_poverty_line_spending = int(
            poverty_line_spending
            / (
                (1 + self.expected_long_term_inflation / self.periods_per_yr)
                ** ((self.retirement_age - self.current_age) * self.periods_per_yr)
            )
        )

        assert (
            self.annual_pretax_income / self.periods_per_yr
            >= 2
            * pre_inflation_poverty_line_spending
            / (
                (1 + self.expected_long_term_inflation / self.periods_per_yr)
                ** ((self.retirement_age - self.current_age) * self.periods_per_yr)
            )
        ), (
            "Error: annual "
            "pretax income is too low."
            f"Set annual pre-tax income to at least "
            f"{2 * {self.periods_per_yr} * pre_inflation_poverty_line_spending}."
        )

        """Note: structure object for implying portfolio ann. expected ret"""
        post_retirement_investment_expectations_df = (
            self.post_retirement_investment_expectations
        )

        if not simulation:
            # imply ann rate of ret on pre_retirement allocation mix
            post_retirement_investment_expectations_df[
                "wt"
            ] = post_retirement_investment_expectations_df["wt"].astype(float)
            post_retirement_investment_expectations_df[
                "ret"
            ] = post_retirement_investment_expectations_df["ret"].astype(float)
            self.post_retirement_investment_rate_of_ret = (
                post_retirement_investment_expectations_df["wt"].dot(
                    post_retirement_investment_expectations_df["ret"]
                )
            )

        # calc social security benefits
        if "usa" in self.country_currency:
            social_security_benefit = usa_plan.social_security_calc(
                self, self.periods_per_yr
            )

        """Note: to calculate 401k, ira, and conventional savings distributions we need " \
        "the investor's after-tax income at retirement. Premise is in retirement
        the investor will want to spend at a rate <=1 relative to spend in year prior
        to retirement.
        """
        # calc spending needs
        # if not contrib_deficiency and not distrib_deficiency and not retire_age_adjust:
        expected_retirement_spend_period_0 = max(
            poverty_line_spending,
            (
                np.median(
                    np.array(
                        [
                            self.aftertax_income_at_retirement,
                            self.pretax_income_at_retirement,
                        ]
                    )
                )
                - self.pre_retirement_savings_df["contributions"][-1:].iloc[0]
            )
            * self.retirement_spending_factor,
        )
        if "usa" in self.country_currency:
            if not simulation:
                self.sim_pre_retirement_investment_ret_period = None
            begin_retirement_balances = usa_plan.begin_balance_usa_retirement(
                self, simulation=simulation
            )
        self.retirement_spending_factor_distrib_deficiency = (
            self.retirement_spending_factor
        )
        if "usa" in self.country_currency:
            if simulation:
                (
                    post_retirement_distributions_list,
                    sim_post_retirement_investment_ret_period,
                ) = usa_plan_detail.simul_build_usa_post_retirement_df(
                    begin_retirement_balances,
                    self.sim_post_retirement_spending_needs,
                    retirement_age,
                    social_security_benefit,
                    self.post_retirement_investment_expectations,
                    self.sim_post_retire_invest_ret_dict_list,
                    self.periods_per_yr,
                )
                self.sim_post_retirement_investment_ret_period = (
                    sim_post_retirement_investment_ret_period
                )
            else:
                post_retirement_distributions_list = (
                    usa_plan_detail.build_usa_post_retirement_df(
                        begin_retirement_balances,
                        expected_retirement_spend_period_0,
                        retirement_age,
                        social_security_benefit,
                        self.post_retirement_investment_rate_of_ret,
                        self.periods_per_yr,
                    )
                )

        expected_retirement_spend_period_0 = max(
            poverty_line_spending,
            (
                np.median(
                    np.array(
                        [
                            self.aftertax_income_at_retirement,
                            self.pretax_income_at_retirement,
                        ]
                    )
                )
                - self.pre_retirement_savings_df["contributions"][-1:].iloc[0]
            )
            * self.retirement_spending_factor,
        )

        if "usa" in self.country_currency:
            post_retirement_distributions_list = (
                usa_plan_detail.build_usa_post_retirement_df(
                    begin_retirement_balances,
                    expected_retirement_spend_period_0,
                    retirement_age,
                    social_security_benefit,
                    self.post_retirement_investment_rate_of_ret,
                    self.periods_per_yr,
                )
            )

        self.post_retirement_savings_df = pd.DataFrame(
            post_retirement_distributions_list
        )
