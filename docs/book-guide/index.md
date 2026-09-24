# Book Guide

The public *Machine Learning for Trading, Third Edition* companion shows
Diagnostic in longer research workflows. The links below point to notebooks
at book commit [`2d6e8f95eeccaee66906245606471f570b5807e5`](https://github.com/stefan-jansen/machine-learning-for-trading/tree/2d6e8f95eeccaee66906245606471f570b5807e5).
All selected notebooks directly call `ml4t.diagnostic`; some also teach the
underlying method manually. The complete book workflows may require data and
optional packages beyond the [synthetic quickstart](../getting-started/quickstart.md).

Start with a Diagnostic guide for supported inputs and result checks, then
open the book notebook for its surrounding research workflow. A call in a
notebook does not make the entire book workflow a supported library API.

## Validation and signal research

| Book notebook | Classification and use | Start in Diagnostic |
|---|---|---|
| [Ch06: CV foundations](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/06_strategy_definition/02_cv_foundations.ipynb) | Direct CV splitter calls, plus manual fold construction | [Cross-validation](../user-guide/cross-validation.md) |
| [Ch07: multiple testing](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/07_defining_the_learning_task/07_multiple_testing.ipynb) | Direct statistical-test calls, plus method exposition | [Statistical tests](../user-guide/statistical-tests.md) |
| [Ch07: causal sanity checks](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/07_defining_the_learning_task/08_causal_sanity_checks.ipynb) | Direct Diagnostic checks in a broader causal review | [Statistical tests](../user-guide/statistical-tests.md) |
| [Ch08: feature selection](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/08_financial_features/05_feature_selection.ipynb) | Direct feature-statistic and importance calls | [Feature selection](../user-guide/feature-selection.md) |
| [Ch08: robustness and sensitivity](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/08_financial_features/06_robustness_sensitivity.ipynb) | Direct Diagnostic signal checks in a wider workflow | [Signal analysis](../getting-started/quickstart.md) |
| [Ch08: event studies](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/08_financial_features/07_event_studies.ipynb) | Direct EventStudyAnalysis use | [Evaluation workflows](../api/index.md#evaluation-workflows) |
| [Ch09: visual diagnostics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/09_model_based_features/01_visual_diagnostics.ipynb) | Direct feature-diagnostic calls | [Feature diagnostics](../user-guide/feature-diagnostics.md) |
| [Ch09: ARIMA features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/09_model_based_features/07_arima_features.ipynb) | Direct stationarity and autocorrelation checks in an ARIMA lesson | [Feature diagnostics](../user-guide/feature-diagnostics.md) |
| [Ch09: GARCH volatility](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/09_model_based_features/08_garch_volatility.ipynb) | Direct volatility diagnostic in a GARCH lesson | [Feature diagnostics](../user-guide/feature-diagnostics.md) |
| [Ch09: Wasserstein regimes](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/09_model_based_features/12_wasserstein_regimes.ipynb) | Direct drift-distance call in a regime workflow | [Feature diagnostics](../user-guide/feature-diagnostics.md) |

## Backtests, portfolios, and trades

| Book notebook | Classification and use | Start in Diagnostic |
|---|---|---|
| [Ch16: performance reporting](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/09_performance_reporting.ipynb) | Direct backtest-bridge and reporting calls | [Backtest tearsheets](../user-guide/backtest-tearsheets.md) |
| [Ch16: Sharpe inference](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/11_sharpe_ratio_inference.ipynb) | Direct Sharpe-inference calls, plus manual explanation | [Statistical tests](../user-guide/statistical-tests.md) |
| [Ch16: DSR validation](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/12_dsr_validation.ipynb) | Direct deflated-Sharpe calls and validation examples | [Statistical tests](../user-guide/statistical-tests.md) |
| [Ch16: RAS protocol](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/13_ras_protocol.ipynb) | Direct RAS adjustment calls | [Statistical tests](../user-guide/statistical-tests.md) |
| [Ch17: portfolio metrics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/01_portfolio_metrics.ipynb) | Direct PortfolioAnalysis use | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch17: mean-variance optimization](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/02_mean_variance_optimization.ipynb) | Direct portfolio diagnostics in an allocator lesson | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch17: robust optimization](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/03_robust_optimization.ipynb) | Direct portfolio diagnostics in an allocator lesson | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch17: Kelly criterion](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/04_kelly_criterion.ipynb) | Direct portfolio diagnostics in a sizing lesson | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch17: hierarchical risk parity](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/06_hierarchical_risk_parity.ipynb) | Direct portfolio diagnostics in an allocator lesson | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch17: library comparison](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/08_library_comparison.ipynb) | Direct Diagnostic comparison calls | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch17: allocator comparison](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/09_allocator_comparison.ipynb) | Direct Diagnostic metrics in a broader allocator comparison | [Evaluation workflow](../user-guide/workflows.md) |
| [Ch19: VaR and CVaR](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/01_var_cvar.ipynb) | Direct distribution diagnostic in a risk lesson | [Feature diagnostics](../user-guide/feature-diagnostics.md) |
| [Ch19: exit strategies](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/02_exit_strategies.ipynb) | Direct BarrierAnalysis use | [Evaluation workflows](../api/index.md#evaluation-workflows) |
| [Ch19: factor exposure](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/04_factor_exposure.ipynb) | Direct factor-analysis calls | [Evaluation workflows](../api/index.md#evaluation-workflows) |
| [Ch19: trade-SHAP diagnostics](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/05_trade_shap_diagnostics.ipynb) | Direct TradeAnalysis and TradeShapAnalyzer use | [Trade analysis](../user-guide/trade-analysis.md) |
| [Ch19: drift detection](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/07_drift_detection.ipynb) | Direct drift-diagnostic calls | [Feature diagnostics](../user-guide/feature-diagnostics.md) |

The [case-study evaluation notebooks](case-studies.md) show the same
statistical checks across nine datasets.
