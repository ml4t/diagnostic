# Book Guide

Use this section to move between the production `ml4t.diagnostic` API and the
pedagogical workflows in *Machine Learning for Trading, Third Edition*.

The book code lives under:

- `third_edition/code/<chapter>/...` for chapter notebooks/scripts
- `third_edition/code/case_studies/<case_study>/...` for end-to-end workflows

## How To Use This Guide

- Start in the book when you want concepts, derivations, and step-by-step builds.
- Start in the library docs when you want the stable API, configuration patterns, and
  reusable workflows.
- Use the chapter map below to jump between the two.

## Chapter Map

| Book chapter | What it teaches | Relevant `ml4t.diagnostic` APIs | Book paths |
|---|---|---|---|
| Ch06 Strategy Definition | Leak-free time-series validation, purging, embargo, fold visualization | `WalkForwardCV`, `CombinatorialCV`, `plot_cv_folds()` | `code/06_strategy_definition/01_cv_foundations.py` |
| Ch07 Defining the Learning Task | Multiple testing, FDR, HAC-adjusted IC, causal sanity checks | `benjamini_hochberg_fdr()`, `compute_ic_hac_stats()`, `compute_min_trl()`, `compute_pbo()` | `code/07_defining_learning_task/07_multiple_testing.py`, `code/07_defining_learning_task/08_causal_sanity_checks.py` |
| Ch08 Feature Engineering | Feature triage, robustness checks, event studies | `analyze_signal()`, `analyze_ml_importance()`, `FeatureSelector`, `EventStudyAnalysis` | `code/08_feature_engineering/05_feature_selection.py`, `code/08_feature_engineering/06_robustness_sensitivity.py`, `code/08_feature_engineering/07_event_studies.py` |
| Ch09 Model-Based Features | Stationarity, distribution diagnostics, autocorrelation, drift | `FeatureDiagnostics`, `analyze_stationarity()`, `analyze_distribution()`, `analyze_autocorrelation()`, drift tools | `code/09_model_based_features/01_visual_diagnostics.py`, `code/09_model_based_features/07_arima_features.py`, `code/09_model_based_features/08_garch_volatility.py`, `code/09_model_based_features/12_wasserstein_regimes.py` |
| Ch16 Strategy Simulation | Performance reporting, tearsheets, Sharpe inference, DSR, RAS | `PortfolioAnalysis`, `BacktestProfile`, `generate_backtest_tearsheet()`, `generate_tearsheet_from_result()`, `deflated_sharpe_ratio()`, `compute_pbo()` | `code/16_strategy_simulation/09_performance_reporting.py`, `code/16_strategy_simulation/11_sharpe_ratio_inference.py`, `code/16_strategy_simulation/12_dsr_validation.py`, `code/16_strategy_simulation/13_ras_protocol.py` |
| Ch17 Portfolio Construction | Portfolio metrics, allocator comparison, robust optimization | `PortfolioAnalysis`, `compute_allocator_metrics()` wrappers used in the book | `code/17_portfolio_construction/01_portfolio_metrics.py`, `code/17_portfolio_construction/02_mean_variance_optimization.py`, `code/17_portfolio_construction/03_robust_optimization.py`, `code/17_portfolio_construction/04_kelly_criterion.py`, `code/17_portfolio_construction/06_hierarchical_risk_parity.py`, `code/17_portfolio_construction/08_library_comparison.py`, `code/17_portfolio_construction/09_allocation_horse_race.py` |
| Ch19 Risk Management | VaR/CVaR, barrier analysis, factor exposure, trade-SHAP, drift | `analyze_distribution()`, `BarrierAnalysis`, `compute_factor_model()`, `compute_return_attribution()`, `compute_risk_attribution()`, `TradeAnalysis`, `TradeShapAnalyzer` | `code/19_risk_management/01_var_cvar.py`, `code/19_risk_management/02_exit_strategies.py`, `code/19_risk_management/04_factor_exposure.py`, `code/19_risk_management/05_trade_shap_diagnostics.py`, `code/19_risk_management/07_drift_detection.py` |

## From Book Notebook To Production Workflow

The book often builds the method manually first and then introduces the library call:

- Ch06 shows fold arithmetic and purge logic before using `WalkForwardCV`.
- Ch07 computes statistical adjustments step by step before switching to
  `compute_ic_hac_stats()` and `benjamini_hochberg_fdr()`.
- Ch08 explains feature triage manually, then points to `analyze_signal()` and
  `FeatureSelector` for reusable pipelines.
- Ch16 and the case studies wrap portfolio and tearsheet reporting behind shared
  helpers that call `PortfolioAnalysis` and the backtest visualization layer.

For the end-to-end view, continue to [Case Studies](case-studies.md).
