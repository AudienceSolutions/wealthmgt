import json
import logging
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from retirement_plan import investor_selector, investor_selector_config

from forecast_retirement_calc.simulator import simulator_tool, simulator_tool_config

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

now = datetime.utcnow()
timestamp = now.strftime("%Y%m%d%H%M%S")

# define the simulation parameter a False
# we locally redefine to True at bottom of code when applying sim code
simulation = False

if __name__ == "__main__":

    parser = ArgumentParser(description="")

    parser.add_argument(
        "--country_currency",
        type=str,
        default="usa_usd",
        help="currently only usa_usd needed but may change in future?",
    )

    parser.add_argument(
        "--current_age",
        type=int,
        default=35,
        help="how old are you now?",
    )

    parser.add_argument(
        "--retirement_age",
        type=int,
        default=68,
        help="at what age would you prefer to retire?",
    )

    parser.add_argument(
        "--life_expectancy",
        type=int,
        default=85,
        help="expected age at death",
    )

    parser.add_argument(
        "--periods_per_yr",
        type=int,
        default=1,
        help="periodicity_of_analysis",
    )

    parser.add_argument(
        "--income_tax_rate",
        type=float,
        default="2.5E-1",
        help="effective personal income tax rate pre-retirment",
    )

    parser.add_argument(
        "--post_retirement_tax_rate",
        type=float,
        default="2E-1",
        help="effective personal income tax rate in retirement",
    )

    parser.add_argument(
        "--expected_real_income_growth",
        type=float,
        default="1E-2",
        help="expected annual rate of income growth",
    )

    parser.add_argument(
        "--expected_long_term_inflation",
        type=float,
        default="2E-2",
        help="expected annual rate of inflation in long run",
    )

    parser.add_argument(
        "--fin_mkt_series",
        type=str,
        default="{'fin_mkt_series': ['S&P 500 (includes dividends)','3-month T.Bill',\
                'US T. Bond','Baa Corporate Bond', 'Real Estate','Private Equity']}",
        help="fin mkt series of interest",
    )

    parser.add_argument(
        "--pre_retirement_investment_expectations",
        type=str,
        help="pre_retirement_investment_expectations",
        default="{'S&P 500': {'wt': '0.40'},\
                '3 mo T-bill': {'wt': '0.05'},\
                'US T-bond': {'wt': '0.15'},\
                'Baa Corporate Bond': {'wt': '0.15'},\
                'Real Estate': {'wt': '0.15'},\
                'Private Equity': {'wt': '0.1'}}",
    )

    parser.add_argument(
        "--post_retirement_investment_expectations",
        type=str,
        help="post_retirement_investment_expectations",
        default="{'S&P 500': {'wt': '0.2'},\
                '3 mo T-bill': {'wt': '0.15'},\
                'US T-bond': {'wt': '0.3'},\
                'Baa Corporate Bond': {'wt': '0.05'},\
                'Real Estate': {'wt': '0.15'},\
                'Private Equity': {'wt': '0.15'}}",
    )

    parser.add_argument(
        "--retirement_spending_factor",
        type=float,
        help="annual expenditure in retirement as fraction of expenditure final year "
        "pre-retirement",
        default=0.85,
    )

    parser.add_argument(
        "--contribution_limits",
        type=str,
        help=".json with dict of irs_contribution_limits_to_tax_advantaged",
        default="contribution_limits.json",
    )

    parser.add_argument(
        "--usa_contributions_pct_income",
        type=str,
        help="% savings to varied accounts",
        default="{'non_tax_advantaged_savings': 0.1}",
    )

    parser.add_argument(
        "--usa_poverty_line_spending",
        type=str,
        help="minimum annual spend for those at poverty line -- if user annual income is "
        "at or below income during pre-retirement set annual spend in retirement "
        "at poverty level",
        default="{'usa_usd': 20e3}",
    )

    parser.add_argument(
        "--income_retirement_balance",
        type=str,
        help="country specific income and retirement balance"
        "[necessary given scale differences between sovereign FX]",
        default="{'usa_usd': {'annual_pre_tax_income': 40e3,\
                    'non_tax_advantaged_savings': 1e5}\
                }",
    )

    parser.add_argument(
        "--paths",
        type=float,
        help="number of simulation paths to run",
        default=1e1,
    )

    args = parser.parse_args()

    # open irs_contribution_limits
    with open(args.contribution_limits, "r") as f:
        contribution_limits = json.load(f)
    irs_contribution_limits = contribution_limits["usa"]

    # open the income_retirement_balance command line argument
    income_retirement_balance = args.income_retirement_balance
    income_retirement_balance = income_retirement_balance.replace("'", '"')
    """Note: read-in list of constraints via json"""
    income_retirement_balance = json.loads(income_retirement_balance)
    annual_pre_tax_income = income_retirement_balance[args.country_currency][
        "annual_pre_tax_income"
    ]

    # forecast horizon is used with simulator
    forecast_horizon = (
        1 + (args.life_expectancy - args.current_age) * args.periods_per_yr
    )
    # periods_per_yr
    periods_per_yr = args.periods_per_yr

    # open the usa_contributions_pct_income command line argument
    usa_contributions_pct_income = args.usa_contributions_pct_income
    usa_contributions_pct_income = usa_contributions_pct_income.replace("'", '"')
    """Note: read-in usa_contributions_pct_income via json"""
    usa_contributions_pct_income = json.loads(usa_contributions_pct_income)

    # open the usa_poverty_line_spending command line argument
    usa_poverty_line_spending = args.usa_poverty_line_spending
    usa_poverty_line_spending = usa_poverty_line_spending.replace("'", '"')
    """Note: read-in usa_poverty_line_spending via json"""
    usa_poverty_line_spending = json.loads(usa_poverty_line_spending)

    # open the fin_mkt_series command line argument
    fin_mkt_series = args.fin_mkt_series
    fin_mkt_series = fin_mkt_series.replace("'", '"')
    """Note: read-in fin_mkt_series via json"""
    fin_mkt_series = json.loads(fin_mkt_series)

    # open the pre_retirement_investment_expectations command line argument
    pre_retirement_investment_expectations = args.pre_retirement_investment_expectations
    pre_retirement_investment_expectations = (
        pre_retirement_investment_expectations.replace("'", '"')
    )

    """Note: read-in pre_retirement_investment_expectations via json"""
    pre_retirement_investment_expectations = json.loads(
        pre_retirement_investment_expectations
    )

    # 1. get the returns series
    financial_series_df = pd.read_csv("data/fin_mkt_ann_ret.csv")

    # 1.a define each input series
    financial_series_dict_list = (
        []
    )  # create list dict to contain series names/hist means
    fin_mkt_series_list = list(fin_mkt_series.values())[0]
    for ix in range(0, len(fin_mkt_series_list)):
        financial_series_dict_list.append(
            {
                f"{fin_mkt_series_list[ix]}_hist_ret": financial_series_df[
                    fin_mkt_series_list[ix]
                ].mean()
            }
        )

    financial_series_list = []  # create list of hist mean returns per series
    for ix in range(0, len(financial_series_dict_list)):
        for key, val in financial_series_dict_list[ix].items():
            financial_series_list.append(financial_series_dict_list[ix][key])

    pre_retirement_investment_expectations = pd.DataFrame(
        pre_retirement_investment_expectations
    ).T
    pre_retirement_investment_expectations_arr = np.concatenate(
        (
            pre_retirement_investment_expectations["wt"].values[None, :],
            np.hstack(np.array([financial_series_list]))[None, :],
        ),
        axis=0,
    ).T

    pre_retirement_investment_expectations = pd.DataFrame(
        pre_retirement_investment_expectations_arr,
        columns=["wt", "ret"],
        index=pre_retirement_investment_expectations.index,
    )

    # open the post_retirement_investment_expectations command line argument
    post_retirement_investment_expectations = (
        args.post_retirement_investment_expectations
    )
    post_retirement_investment_expectations = (
        post_retirement_investment_expectations.replace("'", '"')
    )
    """Note: read-in post_retirement_investment_expectations via json"""
    post_retirement_investment_expectations = json.loads(
        post_retirement_investment_expectations
    )

    post_retirement_investment_expectations = pd.DataFrame(
        post_retirement_investment_expectations
    ).T
    post_retirement_investment_expectations_arr = np.concatenate(
        (
            post_retirement_investment_expectations["wt"].values[None, :],
            np.hstack(np.array([financial_series_list]))[None, :],
        ),
        axis=0,
    ).T

    post_retirement_investment_expectations = pd.DataFrame(
        post_retirement_investment_expectations_arr,
        columns=["wt", "ret"],
        index=post_retirement_investment_expectations.index,
    )

    usa_poverty_line_spending = (
        usa_poverty_line_spending["usa_usd"]
        / periods_per_yr
        * (1 + args.expected_long_term_inflation / periods_per_yr)
        ** ((args.retirement_age - args.current_age) * periods_per_yr)
    )

    """Note: investment_rate_of_ret can be revised as matrix of asset class
    expected returns. Will be matrix of 5-10 asset class portfolio weights
    taken in product with asset class expected returns to arrive at investor
    periodic rate of return."""
    investor_config = investor_selector_config(
        country_currency=args.country_currency,
        current_age=args.current_age,
        retirement_age=args.retirement_age,
        life_expectancy=args.life_expectancy,
        income_tax_rate=args.income_tax_rate,
        post_retirement_tax_rate=args.post_retirement_tax_rate,
        expected_long_term_inflation=args.expected_long_term_inflation,
        expected_real_income_growth=args.expected_real_income_growth,
        annual_pretax_income=annual_pre_tax_income,
        usa_contributions_pct_income=args.usa_contributions_pct_income,
        pre_retirement_investment_expectations=pre_retirement_investment_expectations,
        post_retirement_investment_expectations=post_retirement_investment_expectations,
        retirement_spending_factor=args.retirement_spending_factor,
        irs_contribution_limits=irs_contribution_limits,
        usa_poverty_line_spending=usa_poverty_line_spending,
        periods_per_yr=periods_per_yr,
    )

    investor_plan_detail = investor_selector(investor_config)
    if "usa" in investor_plan_detail.country_currency:
        investor_plan_detail.non_tax_advantaged_savings_balance = (
            income_retirement_balance[f"{args.country_currency}"][
                "non_tax_advantaged_savings"
            ]
        )
        investor_plan_detail.usa_contributions_pct_income = usa_contributions_pct_income

    # calc pre-retirement savings
    investor_plan_detail.calc_pre_retirement_savings(simulation=simulation)

    # calc post-retirement income & spending
    investor_plan_detail.calc_post_retirement_savings(
        retirement_age=investor_plan_detail.retirement_age,
        simulation=simulation,
    )

    simulator_config = simulator_tool_config(
        country_currency=args.country_currency,
        current_age=args.current_age,
        retirement_age=args.retirement_age,
        life_expectancy=args.life_expectancy,
        working_lifespan=(args.retirement_age - args.current_age),
        retirement_lifespan=(args.life_expectancy - args.retirement_age),
        forecast_horizon=forecast_horizon,
        paths=args.paths,
        variance_multiplier=0.5,
        income_tax_rate=args.income_tax_rate,
        post_retirement_tax_rate=args.post_retirement_tax_rate,
        expected_long_term_inflation=args.expected_long_term_inflation,
        expected_real_income_growth=args.expected_real_income_growth,
        usa_contributions_pct_income=args.usa_contributions_pct_income,
        financial_series=financial_series_df,
        post_retirement_spending_needs=investor_plan_detail.post_retirement_savings_df[
            "distributions"
        ],
        retirement_spending_factor=args.retirement_spending_factor,
        irs_contribution_limits=irs_contribution_limits,
        periods_per_yr=args.periods_per_yr,
        pre_retirement_wgt=investor_plan_detail.pre_retirement_investment_expectations[
            ["wt"]
        ],
        post_retirement_wgt=investor_plan_detail.post_retirement_investment_expectations[
            ["wt"]
        ],
        fin_mkt_series_list=fin_mkt_series_list,
    )

    simulator_detail = simulator_tool(simulator_config)
    simulator_detail.simulate()
    LOGGER.info("feed simulations into estimation of prob distrib")

    # define that the simulation boolean is True
    simulation = True
    #
    wealth_df_list = []  # list will store time path of user wealth in each sim path
    # income
    investor_plan_detail.sim_pre_retirement_income = (
        investor_plan_detail.pre_retirement_savings_df["periodic_after_tax_income"]
    )
    # spending
    investor_plan_detail.sim_post_retirement_spending_needs = (
        investor_plan_detail.post_retirement_savings_df["distributions"]
    )
    print("beginning to run sims")
    for path in range(0, int(args.paths)):
        LOGGER.info(f"running simulation path number {path}")

        # step 1: go into pre-calc-retirement-savings
        # calc pre-retirement savings
        sim_pre_retire_invest_ret_dict_list = []
        for ix in range(0, len(fin_mkt_series_list)):
            sim_pre_retire_invest_ret_dict_list.append(
                {
                    fin_mkt_series_list[ix]: simulator_detail.sim.data[
                        fin_mkt_series_list[ix]
                    ].values[0, :, :][
                        : simulator_detail.working_lifespan * periods_per_yr, path
                    ]
                }
            )

        investor_plan_detail.sim_pre_retire_invest_ret_dict_list = (
            sim_pre_retire_invest_ret_dict_list
        )

        # portfolio returns
        # inflation
        investor_plan_detail.sim_pre_retirement_inflation = simulator_detail.sim.data[
            "inflation"
        ].values[0, :, :][: simulator_detail.working_lifespan * periods_per_yr, path]

        print("running simulation path number {path}")
        investor_plan_detail.calc_pre_retirement_savings(simulation=simulation)

        # step 2: go into post-calc-retirement-savings
        # calc post-retirement savings
        sim_post_retire_invest_ret_dict_list = []
        for ix in range(0, len(fin_mkt_series_list)):
            sim_post_retire_invest_ret_dict_list.append(
                {
                    fin_mkt_series_list[ix]: simulator_detail.sim.data[
                        fin_mkt_series_list[ix]
                    ].values[0, :, :][
                        simulator_detail.working_lifespan
                        * periods_per_yr : simulator_detail.life_expectancy
                        * periods_per_yr,
                        path,
                    ]
                }
            )
        # inflation
        investor_plan_detail.sim_post_retirement_inflation = simulator_detail.sim.data[
            "inflation"
        ].values[0, :, :][
            simulator_detail.working_lifespan * periods_per_yr - 1 :, path
        ]
        investor_plan_detail.sim_post_retire_invest_ret_dict_list = (
            sim_post_retire_invest_ret_dict_list
        )
        investor_plan_detail.calc_post_retirement_savings(
            retirement_age=investor_plan_detail.retirement_age,
            simulation=simulation,
        )

        wealth_df_list.append(
            pd.concat(
                [
                    investor_plan_detail.pre_retirement_savings_df.loc[
                        :,
                        investor_plan_detail.pre_retirement_savings_df.columns
                        != "inv_period",
                    ],
                    investor_plan_detail.post_retirement_savings_df.loc[
                        :,
                        investor_plan_detail.post_retirement_savings_df.columns
                        != "ret_period",
                    ],
                ]
            )
        )
        wealth_df_list[path].set_index(["age"], inplace=True)
        wealth_df_list[path] = wealth_df_list[path][["ending_total_balance"]]

    # let's pad df with certain number of rows [value 0]
    m = (args.life_expectancy - args.current_age) + 1
    wealth_df = [
        x.reindex(range(args.current_age, args.life_expectancy + 1), fill_value=0)
        for x in wealth_df_list
    ]

    wealth_df = pd.concat(wealth_df, axis=1)
    # save the 2-D [num paths; num periods ahead] wealth sim object as .csv
    wealth_df.to_csv(f"output/wealth/user_wealth_{timestamp}.csv")

    # distribution of results
    wealth_df_percentiles = wealth_df.quantile(
        [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=1
    )
    wealth_df_percentiles.T.plot()
    plt.suptitle("User Wealth Distribution Simulation through Time", fontsize=8)

    plt.savefig(f"output/wealth/user_wealth_{timestamp}.png")

    for ix in range(0, len(fin_mkt_series_list)):
        fin_mkt_rets = pd.DataFrame(
            simulator_detail.sim.data[fin_mkt_series_list[ix]].values[0, :, :],
            index=wealth_df.index,
        )
        fin_mkt_rets.to_csv(
            f"output/fin_mkt_rets/{fin_mkt_series_list[ix]}_" f"{timestamp}.csv"
        )

        # distribution of results
        fin_mkt_rets_percentiles = fin_mkt_rets.quantile(
            [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=1
        )
        fin_mkt_rets_percentiles.T.plot()
        plt.suptitle(
            f"{fin_mkt_series_list[ix]} Fin Mkt Ret Distribution Simulation through "
            f"Time",
            fontsize=8,
        )
        plt.savefig(f"output/fin_mkt_rets/{fin_mkt_series_list[ix]}_{timestamp}.png")
