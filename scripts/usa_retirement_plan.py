from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ModelConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class usa_plan_config(ModelConfig):

    country_currency: str
    irs_contribution_limits: pd.DataFrame
    non_tax_advantaged_savings_balance: float
    current_age: int
    life_expectancy: int
    expected_long_term_inflation: float
    income_growth_rate: float
    annual_pretax_income: float
    annual_aftertax_income: float
    pre_retirement_tax_rate: float
    post_retirement_tax_rate: float


class usa_plan:

    r"""
    Class that creates object with all properties needed to calculate savings &
    contributions across tax-advantaged and non-advantaged savings accounts.

    Parameters
    ----------
    """

    def __init__(
        self,
        config: usa_plan_config,
    ):
        self.country_currency = config.country_currency
        self.irs_contribution_limits = config.irs_contribution_limits
        self.non_tax_advantaged_savings_balance = (
            config.non_tax_advantaged_savings_balance
        )
        self.current_age = config.current_age
        self.life_expectancy = config.life_expectancy
        self.expected_long_term_inflation = config.expected_long_term_inflation
        self.income_growth_rate = config.income_growth_rate
        self.annual_pretax_income = config.annual_pretax_income
        self.annual_aftertax_income = config.annual_aftertax_income
        self.pre_retirement_tax_rate = config.pre_retirement_tax_rate
        self.post_retirement_tax_rate = config.post_retirement_tax_rate

    def usa_distributions_calculator(
        self,
        periodic_expected_retirement_spend: float,
        social_security_benefit: float,
        non_tax_advantaged_savings_balance: float,
    ) -> Dict:

        # estimate heirarchical logic for distributing from different tax advantaged and not tax advantaged over time

        if periodic_expected_retirement_spend - social_security_benefit < 0:
            distributions = {
                "social_security_benefit": social_security_benefit,
                "non_tax_advantaged_distrib": 0,
            }
        elif (
            periodic_expected_retirement_spend
            - social_security_benefit
            - non_tax_advantaged_savings_balance
        ) < 0:
            distributions = {
                "social_security_benefit": social_security_benefit,
                "non_tax_advantaged_distrib": max(
                    0,
                    (periodic_expected_retirement_spend - social_security_benefit),
                )
                / (1 - self.post_retirement_tax_rate),
            }

        else:
            distributions = 0

        return distributions

    def usa_capped_contributions(
        self, capped_contributions: pd.DataFrame
    ) -> pd.DataFrame:
        # reset irs_contributions for subsequent use
        irs_contribution_limits = self.irs_contribution_limits.copy()
        irs_contribution_limits.reset_index(inplace=True)
        irs_contribution_limits.rename(columns={"index": "plan_type"}, inplace=True)

        individual_ira_list = []
        individual_401k_list = []
        employer_401k_list = []
        conventional_list = []
        for ix in range(0, len(capped_contributions)):
            if (
                "individual" in capped_contributions["plan_type"][ix]
                and "ira" in capped_contributions["plan_type"][ix]
                and "total" not in capped_contributions["plan_type"][ix]
            ):
                individual_ira_list.append(capped_contributions["contributions"][ix])
            elif (
                "individual" in capped_contributions["plan_type"][ix]
                and "401k" in capped_contributions["plan_type"][ix]
                and "total" not in capped_contributions["plan_type"][ix]
                and "employer" not in capped_contributions["plan_type"][ix]
            ):
                individual_401k_list.append(capped_contributions["contributions"][ix])
            elif (
                "employer" in capped_contributions["plan_type"][ix]
                and "401k" in capped_contributions["plan_type"][ix]
                and "total" not in capped_contributions["plan_type"][ix]
            ):
                employer_401k_list.append(capped_contributions["contributions"][ix])
            elif "non_tax_advantaged_savings" in capped_contributions["plan_type"][ix]:
                conventional_list.append(capped_contributions["contributions"][ix])
        # slice irs_contribution_limits df to ensure contributions correctly estimated
        irs_contribution_limits_401k_individual = irs_contribution_limits.loc[
            (irs_contribution_limits["plan_type"].str.contains("401k"))
            & (
                irs_contribution_limits["plan_type"].str.contains("total")
                & (~irs_contribution_limits["plan_type"].str.contains("employer"))
            )
        ]

        irs_contribution_limits_401k_employer = irs_contribution_limits.loc[
            (irs_contribution_limits["plan_type"].str.contains("401k"))
            & (
                irs_contribution_limits["plan_type"].str.contains("employer")
                & (~irs_contribution_limits["plan_type"].str.contains("total"))
            )
        ]

        irs_contribution_limits_ira = irs_contribution_limits.loc[
            (irs_contribution_limits["plan_type"].str.contains("ira"))
            & (irs_contribution_limits["plan_type"].str.contains("total"))
        ]

        """Note: update the capped_contributions object using:
        i)both irs_contribution_limit_401k and irs_contribution_limit_ira;
        2)both individual_401k_list and individual_ira_list

        3a)Read individual ira_list and sum ira contributions from list
        3b)Read individual 401k_list and sum 401k contributions from list

        4)Rescale ira against irs_contribution_limit_ira;
        Rescale individual 401k against irs_contributions_limit_401k_individual;
        Rescale individual + employer 401k against
        irs_contributions_limit_401k_individual plus employer;
        """

        contribution_pre_rescale_ira = sum(individual_ira_list)
        contribution_pre_rescale_401k_individual = sum(individual_401k_list)
        contribution_pre_rescale_401k_employer = sum(employer_401k_list)

        individual_ira_list = list(
            (
                np.array(irs_contribution_limits_ira[self.age_category])
                / (contribution_pre_rescale_ira + 1e-10)
            )
            * individual_ira_list
        )
        contribution_post_rescale_ira = sum(individual_ira_list)
        individual_ira_list.append(contribution_post_rescale_ira)

        individual_401k_list = list(
            (
                min(
                    np.array(irs_contribution_limits_401k_individual[self.age_category])
                    / (contribution_pre_rescale_401k_individual + 1e-10),
                    1,
                )
            )
            * individual_401k_list
        )
        contribution_post_rescale_401k_individual = sum(individual_401k_list)
        individual_401k_list.append(contribution_post_rescale_401k_individual)

        arr_1 = np.array(irs_contribution_limits_401k_employer[self.age_category]) / (
            contribution_pre_rescale_401k_employer
        )
        arr_2 = np.array([1, 1])
        arr = np.array((arr_1, arr_2))
        employer_401k_list = list((arr.min(axis=0)) * employer_401k_list)
        contribution_post_rescale_401k_employer = sum(employer_401k_list)
        if np.isnan(employer_401k_list).all():
            employer_401k_list = ([0] * (len(employer_401k_list) + 1))[
                : len(employer_401k_list) + 1
            ]
        else:
            employer_401k_list.append(contribution_post_rescale_401k_employer)

        capped_contributions_arr = np.concatenate(
            (
                [np.array(individual_ira_list)],
                [np.array(individual_401k_list)],
                [np.array(employer_401k_list)],
                [np.array(conventional_list)],
            ),
            axis=1,
        ).T

        capped_contributions_arr = np.concatenate(
            (
                capped_contributions["plan_type"].values[:, None],
                capped_contributions_arr,
            ),
            axis=1,
        )

        capped_contributions_df = pd.DataFrame(
            capped_contributions_arr, columns=["plan_type", "contributions"]
        )

        return capped_contributions_df

    def simul_build_usa_pre_retirement_df(
        self,
        savings_pct_income_df,
        sim_pre_retirement_inflation,
        sim_pre_retirement_income,
        retirement_age: int,
        pre_retirement_investment_expectations: pd.DataFrame,
        sim_pre_retire_invest_ret_dict_list: List,
        periods_per_yr: int,
    ) -> List[Dict]:
        pre_retirement_wgt = pre_retirement_investment_expectations[["wt"]]
        # convert dict list into dataframe
        sim_pre_retire_invest_ret_df = pd.concat(
            [
                pd.DataFrame.from_dict(x, orient="index")
                for x in sim_pre_retire_invest_ret_dict_list
            ]
        )
        pre_retirement_savings_list = []
        # logic here to handle limits on retirement contributions
        irs_contribution_limits = self.irs_contribution_limits.copy()
        irs_contribution_limits.reset_index(inplace=True)
        irs_contribution_limits.rename(columns={"index": "plan_type"}, inplace=True)

        non_tax_advantaged_savings_balance = self.non_tax_advantaged_savings_balance
        """Note: when retirement age is float due to incrementing age by sub-year periods, the range indexing in
        for loop immediately below needs to be defined consistent with number of distinct sub-year periods between 
        retirement age and current age. For example, if retirement age is 66.5, and we have 12 periods per year, 
        and current age is 29, then range length of loop immediately below should be (37yr * 12prd/yr) + 6 = 450"""
        num_periods_init = (int(retirement_age) - self.current_age) * periods_per_yr
        sub_yr_approx = int(
            round((retirement_age - int(retirement_age)) / (1 / periods_per_yr))
        )
        num_periods = num_periods_init + sub_yr_approx
        for period in range(0, num_periods):
            sim_pre_retirement_investment_ret_period = (
                sim_pre_retire_invest_ret_df[period].values @ pre_retirement_wgt
            ).iloc[0]
            age = self.current_age + period / periods_per_yr
            inv_period = period

            # account_balances
            begin_balance = +non_tax_advantaged_savings_balance

            aftertax_savings_pct_income = savings_pct_income_df[
                savings_pct_income_df.columns[
                    savings_pct_income_df.columns.str.contains("non_tax_advantaged")
                ]
            ]
            aftertax_savings = (
                sim_pre_retirement_income[period] * aftertax_savings_pct_income
            ).T
            contributions = pd.concat([aftertax_savings], axis=0)
            contributions = pd.DataFrame(
                contributions.values,
                columns=["contributions"],
                index=contributions.index,
            )
            contributions.reset_index(inplace=True)
            contributions.rename(columns={"index": "plan_type"}, inplace=True)

            irs_contribution_limits.loc[:, "age < 50":"age >= 50"] *= (
                1 + sim_pre_retirement_inflation[period]
            )
            contributions_v_limit = pd.merge(
                irs_contribution_limits, contributions, how="left", on=["plan_type"]
            )

            """Note: determine the age_category within contributions to use."""
            if age < 50:
                age_category = "age < 50"
            else:
                age_category = "age >= 50"

            self.age_category = age_category

            capped_contributions = contributions_v_limit[
                [age_category, "contributions"]
            ].apply(lambda x: x[0] if x[1] > x[0] else x[1], axis=1)

            capped_contributions = pd.DataFrame(
                capped_contributions.values,
                index=contributions_v_limit["plan_type"],
                columns=["contributions"],
            )
            capped_contributions.reset_index(inplace=True)

            capped_contributions = self.usa_capped_contributions(capped_contributions)

            contrib_final_estimate_non_tax_advantaged_savings = (
                capped_contributions.loc[
                    (capped_contributions["plan_type"] == "non_tax_advantaged_savings")
                ]
            )

            non_tax_advantaged_savings_balance = (
                non_tax_advantaged_savings_balance
                + (
                    contrib_final_estimate_non_tax_advantaged_savings[
                        "contributions"
                    ].iloc[0]
                )
            ) * (1 + sim_pre_retirement_investment_ret_period)

            final_contrib_estimate = +contrib_final_estimate_non_tax_advantaged_savings[
                "contributions"
            ].iloc[0]
            ending_balance = +non_tax_advantaged_savings_balance
            investment_balance_change = ending_balance - begin_balance

            pre_retirement_savings_list.append(
                {
                    "age": age,
                    "inv_period": inv_period,
                    "periodic_pre_tax_income": sim_pre_retirement_income[period]
                    * (1 + self.pre_retirement_tax_rate),
                    "periodic_after_tax_income": sim_pre_retirement_income[period],
                    "begin_total_balance": begin_balance,
                    "non_tax_advantaged_savings_balance": non_tax_advantaged_savings_balance,
                    "non_tax_advantaged_savings_contribution": contrib_final_estimate_non_tax_advantaged_savings[
                        "contributions"
                    ].iloc[
                        0
                    ],
                    "contributions": final_contrib_estimate,
                    "total_balance_change": investment_balance_change,
                    "ending_total_balance": ending_balance,
                }
            )

        return pre_retirement_savings_list, sim_pre_retirement_investment_ret_period

    def build_usa_pre_retirement_df(
        self,
        savings_pct_income_df: pd.DataFrame,
        retirement_age: int,
        pre_retirement_investment_rate_of_ret: float,
        periods_per_yr: int,
    ) -> List[Dict]:
        pre_retirement_investment_rate_of_ret = (
            pre_retirement_investment_rate_of_ret / periods_per_yr
        )
        pre_retirement_savings_list = []
        # logic here to handle limits on retirement contributions
        irs_contribution_limits = self.irs_contribution_limits.copy()
        irs_contribution_limits.reset_index(inplace=True)
        irs_contribution_limits.rename(columns={"index": "plan_type"}, inplace=True)

        non_tax_advantaged_savings_balance = self.non_tax_advantaged_savings_balance
        """Note: when retirement age is float due to incrementing age by sub-year periods, the range indexing in
        for loop immediately below needs to be defined consistent with number of distinct sub-year periods between 
        retirement age and current age. For example, if retirement age is 66.5, and we have 12 periods per year, 
        and current age is 29, then range length of loop immediately below should be (37yr * 12prd/yr) + 6 = 450"""
        num_periods_init = (int(retirement_age) - self.current_age) * periods_per_yr
        sub_yr_approx = int(
            round((retirement_age - int(retirement_age)) / (1 / periods_per_yr))
        )
        num_periods = num_periods_init + sub_yr_approx
        for period in range(0, num_periods):
            age = self.current_age + period / periods_per_yr
            inv_period = period
            # periodically compounded factors or inflation, income, investment_ret
            inflation_factor = 1 + self.expected_long_term_inflation / periods_per_yr
            # let us assume there is annual increase in income [not same as periodic, if > 1 period per year]
            if period % periods_per_yr == 0:
                income_growth_rate_factor = (1 + self.income_growth_rate) ** (
                    inv_period / periods_per_yr
                )

            # account_balances
            begin_balance = +non_tax_advantaged_savings_balance

            """Note: synax adj to enable read-in via json"""
            pretax_savings_pct_income = savings_pct_income_df[
                savings_pct_income_df.columns[
                    savings_pct_income_df.columns.str.contains("traditional")
                ]
            ]
            pretax_savings = (
                self.annual_pretax_income
                / periods_per_yr
                * income_growth_rate_factor
                * pretax_savings_pct_income
            ).T
            aftertax_savings_pct_income = savings_pct_income_df[
                savings_pct_income_df.columns[
                    ~savings_pct_income_df.columns.str.contains("traditional")
                ]
            ]
            aftertax_savings = (
                self.annual_aftertax_income
                / periods_per_yr
                * income_growth_rate_factor
                * aftertax_savings_pct_income
            ).T
            contributions = pd.concat([pretax_savings, aftertax_savings], axis=0)
            contributions = pd.DataFrame(
                contributions.values,
                columns=["contributions"],
                index=contributions.index,
            )
            contributions.reset_index(inplace=True)
            contributions.rename(columns={"index": "plan_type"}, inplace=True)

            irs_contribution_limits.loc[:, "age < 50":"age >= 50"] *= inflation_factor
            contributions_v_limit = pd.merge(
                irs_contribution_limits, contributions, how="left", on=["plan_type"]
            )

            """Note: determine the age_category within contributions to use."""
            if age < 50:
                age_category = "age < 50"
            else:
                age_category = "age >= 50"

            self.age_category = age_category

            capped_contributions = contributions_v_limit[
                [age_category, "contributions"]
            ].apply(lambda x: x[0] if x[1] > x[0] else x[1], axis=1)

            capped_contributions = pd.DataFrame(
                capped_contributions.values,
                index=contributions_v_limit["plan_type"],
                columns=["contributions"],
            )
            capped_contributions.reset_index(inplace=True)

            contrib_final_estimate_non_tax_advantaged_savings = (
                capped_contributions.loc[
                    (capped_contributions["plan_type"] == "non_tax_advantaged_savings")
                ]
            )
            non_tax_advantaged_savings_balance = (
                non_tax_advantaged_savings_balance
                + (
                    contrib_final_estimate_non_tax_advantaged_savings[
                        "contributions"
                    ].iloc[0]
                )
            ) * (1 + pre_retirement_investment_rate_of_ret)

            final_contrib_estimate = +contrib_final_estimate_non_tax_advantaged_savings[
                "contributions"
            ].iloc[0]
            ending_balance = +non_tax_advantaged_savings_balance
            investment_balance_change = ending_balance - begin_balance
            pre_retirement_savings_list.append(
                {
                    "age": age,
                    "inv_period": inv_period,
                    "periodic_pre_tax_income": self.annual_pretax_income
                    / periods_per_yr
                    * income_growth_rate_factor,
                    "periodic_after_tax_income": self.annual_aftertax_income
                    / periods_per_yr
                    * income_growth_rate_factor,
                    "begin_total_balance": begin_balance,
                    "non_tax_advantaged_savings_balance": non_tax_advantaged_savings_balance,
                    "non_tax_advantaged_savings_contribution": contrib_final_estimate_non_tax_advantaged_savings[
                        "contributions"
                    ].iloc[
                        0
                    ],
                    "contributions": final_contrib_estimate,
                    "total_balance_change": investment_balance_change,
                    "ending_total_balance": ending_balance,
                }
            )
        return pre_retirement_savings_list, income_growth_rate_factor

    def begin_balance_usa_retirement(
        self,
        simulation: bool,
        sim_pre_retirement_investment_ret_period=None,
    ) -> Dict:
        non_tax_advantaged_savings_balance = self.pre_retirement_savings_df[
            "non_tax_advantaged_savings_balance"
        ][-1:]
        non_tax_advantaged_savings_contribution = self.pre_retirement_savings_df[
            "non_tax_advantaged_savings_contribution"
        ][-1:]
        if simulation:
            post_retirement_investment_rate_of_ret = (
                self.sim_pre_retirement_investment_ret_period
            )
        else:
            post_retirement_investment_rate_of_ret = (
                self.post_retirement_investment_rate_of_ret / self.periods_per_yr
            )
        begin_non_tax_advantaged_savings_balance = (
            non_tax_advantaged_savings_balance + non_tax_advantaged_savings_contribution
        ) * (1 + post_retirement_investment_rate_of_ret)
        begin_balance = +begin_non_tax_advantaged_savings_balance
        begin_balances = {
            "begin_retirement_balance": begin_balance,
            "begin_non_tax_advantaged_savings_balance": begin_non_tax_advantaged_savings_balance,
        }

        return begin_balances

    def social_security_calc(self, periods_per_yr) -> float:
        social_security_benefit = max(
            18000 / periods_per_yr,
            (self.pre_retirement_savings_df["periodic_pre_tax_income"].mean() * 0.03)
            / periods_per_yr,
        )
        return social_security_benefit

    def simul_build_usa_post_retirement_df(
        self,
        begin_retirement_balances: Dict,
        sim_post_retirement_spending_needs: float,
        retirement_age: int,
        social_security_benefit: float,
        post_retirement_investment_expectations: pd.DataFrame,
        sim_post_retire_invest_ret_dict_list: List,
        periods_per_yr: int,
    ) -> List[Dict]:
        post_retirement_wgt = post_retirement_investment_expectations[["wt"]]
        # convert dict list into dataframe
        sim_post_retire_invest_ret_df = pd.concat(
            [
                pd.DataFrame.from_dict(x, orient="index")
                for x in sim_post_retire_invest_ret_dict_list
            ]
        )
        non_tax_advantaged_savings_balance = begin_retirement_balances[
            "begin_non_tax_advantaged_savings_balance"
        ].iloc[0]
        if isinstance(non_tax_advantaged_savings_balance, pd.Series):
            non_tax_advantaged_savings_balance = (
                non_tax_advantaged_savings_balance.iloc[0]
            )

        post_retirement_distributions_list = []
        """Note: when retirement age is float due to incrementing age by sub-year periods, the range indexing in
        for loop immediately below needs to be defined consistent with number of distinct sub-year periods between 
        retirement age and life expectancy. For example, if retirement age is 66.5, and we have 12 periods per year, 
        and life expectancy is 85, then range length of loop immediately below should be (18yr * 12prd/yr) + 6 = 222"""
        num_periods_init = (
            1 + (self.life_expectancy - int(retirement_age)) * periods_per_yr
        )
        sub_yr_approx = int(
            round((retirement_age - int(retirement_age)) / (1 / periods_per_yr))
        )
        num_periods = num_periods_init - sub_yr_approx
        for period in range(0, num_periods):
            sim_post_retirement_investment_ret_period = (
                sim_post_retire_invest_ret_df[period].values @ post_retirement_wgt
            ).iloc[0]
            age = retirement_age + period / periods_per_yr
            inv_period = period
            # account_balances
            begin_balance = +non_tax_advantaged_savings_balance

            # funding the annual expected spend
            """Note: tax-exempt retirement distributions should be the first source of
            funding. Take tax-advantaged taxable retirement distributions if and/or
            when tax exempt retirement distribution amounts can no longer cover annual
            expected spend. Take non-tax advantaged retirement distributions if and/or
            when tax advantaged retirement distributions can no longer cover annual
            expected spend."""

            # social security
            # traditional ira
            # traditional 401k
            # roth ira
            # roth 410k
            # non-tax advantaged savings

            distributions = self.usa_distributions_calculator(
                sim_post_retirement_spending_needs[period],
                social_security_benefit,
                non_tax_advantaged_savings_balance,
            )
            if distributions == 0:
                break

            # period N distributions
            distrib_non_tax_advantaged_savings = distributions[
                "non_tax_advantaged_distrib"
            ]
            total_distributions = +distrib_non_tax_advantaged_savings
            non_tax_advantaged_savings_balance = max(
                0,
                (
                    non_tax_advantaged_savings_balance
                    - distrib_non_tax_advantaged_savings
                )
                * (1 + sim_post_retirement_investment_ret_period / periods_per_yr),
            )
            ending_balance = +non_tax_advantaged_savings_balance
            investment_balance_change = ending_balance - begin_balance

            post_retirement_distributions_list.append(
                {
                    "age": age,
                    "ret_period": inv_period,
                    "begin_total_balance": begin_balance,
                    "non_tax_advantaged_savings_balance": non_tax_advantaged_savings_balance,
                    "non_tax_advantaged_savings_distribution": distrib_non_tax_advantaged_savings,
                    "distributions": total_distributions,
                    "total_balance_change": investment_balance_change,
                    "ending_total_balance": ending_balance,
                }
            )
        return post_retirement_distributions_list, post_retirement_distributions_list

    def build_usa_post_retirement_df(
        self,
        begin_retirement_balances: Dict,
        periodic_retirement_spend_period_0: float,
        retirement_age: int,
        social_security_benefit: float,
        post_retirement_investment_rate_of_ret: float,
        periods_per_yr: int,
    ) -> List[Dict]:

        non_tax_advantaged_savings_balance = begin_retirement_balances[
            "begin_non_tax_advantaged_savings_balance"
        ]
        if isinstance(non_tax_advantaged_savings_balance, pd.Series):
            non_tax_advantaged_savings_balance = (
                non_tax_advantaged_savings_balance.iloc[0]
            )

        post_retirement_distributions_list = []
        """Note: when retirement age is float due to incrementing age by sub-year periods, the range indexing in
        for loop immediately below needs to be defined consistent with number of distinct sub-year periods between 
        retirement age and life expectancy. For example, if retirement age is 66.5, and we have 12 periods per year, 
        and life expectancy is 85, then range length of loop immediately below should be (18yr * 12prd/yr) + 6 = 222"""
        num_periods_init = (
            1 + (self.life_expectancy - int(retirement_age)) * periods_per_yr
        )
        sub_yr_approx = int(
            round((retirement_age - int(retirement_age)) / (1 / periods_per_yr))
        )
        num_periods = num_periods_init - sub_yr_approx
        for period in range(0, num_periods):
            age = retirement_age + period / periods_per_yr
            inv_period = period
            # periodically compounded factors or inflation, income, investment_ret
            inflation_factor = (
                1 + self.expected_long_term_inflation / periods_per_yr
            ) ** (inv_period)

            # periodic expected spend
            periodic_expected_retirement_spend = (
                periodic_retirement_spend_period_0 * inflation_factor
            )
            # account_balances
            begin_balance = +non_tax_advantaged_savings_balance

            # funding the annual expected spend
            """Note: tax-exempt retirement distributions should be the first source of
            funding. Take tax-advantaged taxable retirement distributions if and/or
            when tax exempt retirement distribution amounts can no longer cover annual
            expected spend. Take non-tax advantaged retirement distributions if and/or
            when tax advantaged retirement distributions can no longer cover annual
            expected spend."""

            # social security
            # traditional ira
            # traditional 401k
            # roth ira
            # roth 410k
            # non-tax advantaged savings

            distributions = self.usa_distributions_calculator(
                periodic_expected_retirement_spend,
                social_security_benefit,
                non_tax_advantaged_savings_balance,
            )
            if distributions == 0:
                break

            # period N distributions
            distrib_non_tax_advantaged_savings = (
                distributions["non_tax_advantaged_distrib"] / periods_per_yr
            )
            total_distributions = +distrib_non_tax_advantaged_savings

            non_tax_advantaged_savings_balance = max(
                0,
                (
                    non_tax_advantaged_savings_balance
                    - distrib_non_tax_advantaged_savings
                )
                * (1 + post_retirement_investment_rate_of_ret / 2),
            )
            ending_balance = +non_tax_advantaged_savings_balance
            investment_balance_change = ending_balance - begin_balance

            post_retirement_distributions_list.append(
                {
                    "age": age,
                    "ret_period": inv_period,
                    "begin_total_balance": begin_balance,
                    "non_tax_advantaged_savings_balance": non_tax_advantaged_savings_balance,
                    "non_tax_advantaged_savings_distribution": distrib_non_tax_advantaged_savings,
                    "distributions": total_distributions,
                    "total_balance_change": investment_balance_change,
                    "ending_total_balance": ending_balance,
                }
            )
        return post_retirement_distributions_list
