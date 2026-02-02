# API Reference

Complete API documentation for ML4T Diagnostic.

## Core Module

::: ml4t.diagnostic
    options:
      show_root_heading: true
      members:
        - Evaluator
        - EvaluationResult
        - ValidatedCrossValidation
        - analyze_signal
        - SignalResult
        - get_agent_docs

## Configuration

::: ml4t.diagnostic.config
    options:
      show_root_heading: true
      members:
        - DiagnosticConfig
        - StatisticalConfig
        - PortfolioConfig
        - TradeConfig
        - SignalConfig
        - RuntimeConfig

## Cross-Validation

::: ml4t.diagnostic.splitters
    options:
      show_root_heading: true
      members:
        - CombinatorialCV
        - WalkForwardCV
        - CalendarSplitter
        - GroupIsolationSplitter

## Statistical Tests

::: ml4t.diagnostic.evaluation.stats
    options:
      show_root_heading: true
      members:
        - deflated_sharpe_ratio
        - deflated_sharpe_ratio_from_statistics
        - compute_min_trl
        - min_trl_fwer
        - compute_ras
        - fdr_correction
        - compute_hac_stats

## Feature Diagnostics

::: ml4t.diagnostic.evaluation
    options:
      show_root_heading: true
      members:
        - FeatureDiagnostics
        - analyze_stationarity
        - analyze_autocorrelation
        - analyze_distribution

## Metrics

::: ml4t.diagnostic.evaluation.metrics
    options:
      show_root_heading: true
      members:
        - compute_ic_series
        - compute_mdi_importance
        - compute_pfi_importance
        - compute_shap_importance
        - compute_h_statistic

## Trade Analysis

::: ml4t.diagnostic.evaluation
    options:
      show_root_heading: true
      members:
        - TradeAnalysis
        - TradeShapAnalyzer
        - analyze_excursions

## Data Integration

::: ml4t.diagnostic.integration
    options:
      show_root_heading: true
      members:
        - DataQualityReport
        - DataQualityMetrics
        - DataAnomaly
