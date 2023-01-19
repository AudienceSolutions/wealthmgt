# wealthmgt
# Retirement Planner Tool

## Simulation Framework

### Steps to running the process 
Copy the first command into your preferred text editor [e.g., notepad++, word, etc..]  
Note 1: the \` character at end of each line is essential once you copy-paste the command into pycharm terminal and hit enter  
Note 2: make sure the working directory in the pycharm terminal is the parent directory of /scripts/  
Note 3: command line argument begins with: python scripts/...


python "scripts/retirment_planner.py" \`      
--country_currency "usa_usd" \`  
--current_age 35 \`  
--retirement_age 68 \`  
--life_expectancy 85 \`  
--periods_per_yr 1 \`  
--income_tax_rate 2.5E-1 \`  
--post_retirement_tax_rate 2E-1 \`  
--expected_real_income_growth 1E-2 \`  
--expected_long_term_inflation 2E-2 \`   
--fin_mkt_series "{'fin_mkt_series': ['S&P 500 (includes dividends)','3-month T.Bill',  \ 
                'US T. Bond','Baa Corporate Bond', 'Real Estate','Private Equity']}"  \`  
--pre_retirement_investment_expectations "{'S&P 500': {'wt': '0.40'}, \`  
                '3 mo T-bill': {'wt': '0.05'}, \`  
                'US T-bond': {'wt': '0.15'}, \`  
                'Baa Corporate Bond': {'wt': '0.15'}, \`  
                'Real Estate': {'wt': '0.15'}, \`  
                'Private Equity': {'wt': '0.1'}}"  \`  
--post_retirement_investment_expectations "{'S&P 500': {'wt': '0.2'}, \`  
                '3 mo T-bill': {'wt': '0.15'}, \`  
                'US T-bond': {'wt': '0.3'}, \`  
                'Baa Corporate Bond': {'wt': '0.05'}, \`  
                'Real Estate': {'wt': '0.15'}, \`  
                'Private Equity': {'wt': '0.15'}}" \`  
--retirement_spending_factor 0.85 \`  
--contribution_limits "contribution_limits.json" \`  
--usa_contributions_pct_income "{'non_tax_advantaged_savings': 0.1}" \`  
--usa_poverty_line_spending "{'usa_usd': 20e3}" \`  
--income_retirement_balance "{'usa_usd': {'annual_pre_tax_income': 40e3, \`    
                    'non_tax_advantaged_savings': 1e5}}" \`    
--paths 1e1
