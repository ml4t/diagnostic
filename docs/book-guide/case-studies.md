# Case Studies

All nine case studies use `ml4t.diagnostic`, but they do not all hit the same surface
area. This page shows where the library appears in the end-to-end workflow.

## Shared Patterns

Across case studies, the recurring entry points are:

- `05_evaluation.py` for HAC-adjusted IC and FDR-controlled feature screening
- `*_backtest.py` for portfolio evaluation and reporting
- `*_portfolio_management.py` for allocator comparison via `PortfolioAnalysis`
- `*_costs.py` for cost-aware performance diagnostics
- `*_risk_management.py` for risk decomposition, barrier analysis, or exposure work
- `*_strategy_analysis.py` for synthesis and cross-model comparison

## Case-Study Map

| Case study | Evaluation | Backtest and portfolio | Costs and risk |
|---|---|---|---|
| CME Futures | `case_studies/cme_futures/05_evaluation.py` | `case_studies/cme_futures/16_backtest.py`, `case_studies/cme_futures/17_portfolio_management.py` | `case_studies/cme_futures/18_costs.py`, `case_studies/cme_futures/19_risk_management.py`, `case_studies/cme_futures/20_strategy_analysis.py` |
| Crypto Perps Funding | `case_studies/crypto_perps_funding/05_evaluation.py` | `case_studies/crypto_perps_funding/14_backtest.py`, `case_studies/crypto_perps_funding/15_portfolio_management.py` | `case_studies/crypto_perps_funding/16_costs.py`, `case_studies/crypto_perps_funding/17_risk_management.py`, `case_studies/crypto_perps_funding/18_strategy_analysis.py` |
| ETFs | `case_studies/etfs/05_evaluation.py` | `case_studies/etfs/15_backtest.py`, `case_studies/etfs/16_portfolio_management.py` | `case_studies/etfs/17_costs.py`, `case_studies/etfs/18_risk_management.py`, `case_studies/etfs/19_strategy_analysis.py` |
| FX Pairs | `case_studies/fx_pairs/05_evaluation.py` | Backtest and portfolio stages are introduced later in the sequence for this study | Evaluation is the main current `ml4t.diagnostic` touchpoint |
| Nasdaq100 Microstructure | `case_studies/nasdaq100_microstructure/05_evaluation.py` | `case_studies/nasdaq100_microstructure/14_backtest.py`, `case_studies/nasdaq100_microstructure/15_portfolio_management.py` | `case_studies/nasdaq100_microstructure/16_costs.py`, `case_studies/nasdaq100_microstructure/17_risk_management.py`, `case_studies/nasdaq100_microstructure/18_strategy_analysis.py` |
| S&P 500 Equity Option Analytics | `case_studies/sp500_equity_option_analytics/05_evaluation.py` | `case_studies/sp500_equity_option_analytics/14_backtest.py`, `case_studies/sp500_equity_option_analytics/15_portfolio_management.py` | `case_studies/sp500_equity_option_analytics/16_costs.py`, `case_studies/sp500_equity_option_analytics/17_risk_management.py`, `case_studies/sp500_equity_option_analytics/18_strategy_analysis.py` |
| S&P 500 Options | `case_studies/sp500_options/06_evaluation.py` | `case_studies/sp500_options/16_backtest.py`, `case_studies/sp500_options/17_portfolio_management.py` | `case_studies/sp500_options/18_costs.py`, `case_studies/sp500_options/19_risk_management.py`, `case_studies/sp500_options/20_strategy_analysis.py` |
| US Equities Panel | `case_studies/us_equities_panel/05_evaluation.py` | `case_studies/us_equities_panel/17_backtest.py`, `case_studies/us_equities_panel/18_portfolio_management.py` | `case_studies/us_equities_panel/19_costs.py`, `case_studies/us_equities_panel/20_risk_management.py`, `case_studies/us_equities_panel/21_strategy_analysis.py` |
| US Firm Characteristics | `case_studies/us_firm_characteristics/05_evaluation.py` | `case_studies/us_firm_characteristics/16_backtest.py`, `case_studies/us_firm_characteristics/17_portfolio_management.py` | `case_studies/us_firm_characteristics/18_costs.py`, `case_studies/us_firm_characteristics/19_risk_management.py`, `case_studies/us_firm_characteristics/20_strategy_analysis.py` |

## Shared Book Utilities

Several book utilities wrap the production API so readers can reuse a stable interface
across chapters and case studies:

- `code/case_studies/utils/backtest_loaders.py`
- `code/case_studies/utils/backtest_stats.py`
- `code/case_studies/utils/backtest_tearsheets.py`
- `code/case_studies/utils/factor_attribution.py`

These helpers are useful when reading the book, but for reusable application code the
library APIs in the main docs are the intended entry points.
