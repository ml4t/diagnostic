# Changelog

All notable changes to ml4t-diagnostic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **ValidatedCrossValidation**: Orchestrates CPCV with DSR computation for robust strategy validation
  - `ValidatedCrossValidation` class combines CPCV and DSR in one workflow
  - `validated_cross_val_score()` convenience function for quick evaluation
  - `ValidationResult` with summary(), to_dict(), and human-readable interpretation
  - Reduces 20+ line manual workflow to 5 lines
- **Result interface standardization**: Added `interpret()` method to BaseResult and key result classes
  - `BaseResult.interpret()` - returns list of human-readable insights
  - `PSRResult.interpret()` - PSR-specific interpretation
  - `DSRResult.interpret()` - DSR interpretation with recommendations
- **ml4t-data integration contract**: Data quality validation framework
  - `DataQualityReport` - complete quality report with metrics, anomalies, recommendations
  - `DataQualityMetrics` - quantitative quality metrics (completeness, accuracy, consistency)
  - `DataAnomaly` - individual anomaly record (price spikes, gaps, OHLC violations)
  - `DataValidationRequest` - request from diagnostic to data for validation
- HTML report generator tests for full coverage
- Comprehensive validation module tests (dataframe, returns, timeseries)
- docs/quickstart.md - Minimal examples (5-15 lines each)
- docs/configuration.md - Configuration guide with presets and settings
- **Test Coverage Improvements** (Jan 2026):
  - 50 Information Coefficient tests (`test_information_coefficient.py`)
  - 14 Real Data Integration tests with Wiki Prices (`test_equity_real_data.py`)
  - Additional validation tests for `validate_frequency`, `validate_index`
  - IC module coverage: 9% → 98%
  - Validation modules: 70-100% coverage

### Fixed
- DSR/PSR/MinTRL test fixtures now match López de Prado et al. (2025) paper values
- All xfail tests in DSR validation now pass
- MinTRL formula uses SR₀ (not observed SR) in variance adjustment

### Changed
- Validation modules: 99%+ test coverage (dataframe 100%, returns 100%, timeseries 98%)
- Reporting modules: 96-100% test coverage (base 100%, JSON 100%, HTML 100%, Markdown 96%)

## [1.1.0] - 2025-11-15

### Added
- **Trade SHAP Analysis**: `TradeShapAnalyzer` for SHAP-based trade error pattern analysis
- **Price Excursion Analysis**: `analyze_excursions()` for TP/SL parameter optimization
- **Barrier Analysis**: `BarrierAnalysis` module for triple barrier outcome evaluation
- **Portfolio Analysis**: `PortfolioAnalysis` as pyfolio replacement with modern API
- **IC Time Series**: Alphalens-replacement IC computation with decay analysis
- **Dashboard Export**: Export and caching functions for evaluation dashboards
- **Percentile Computation**: CV prediction percentile module

### Changed
- **D06 Config Consolidation**: Reduced 61+ config classes to 10 primary configs
  - `DiagnosticConfig` (feature analysis)
  - `StatisticalConfig` (multiple testing)
  - `PortfolioConfig` (portfolio analysis)
  - `TradeConfig` (trade analysis + SHAP)
  - `SignalConfig` (signal analysis)
  - `EventConfig`, `BarrierConfig`, `ReportConfig`, `RuntimeConfig`
- **Single-level nesting**: `config.stationarity.enabled` pattern
- **Preset methods preserved**: `for_quick_analysis()`, `for_research()`, `for_production()`
- All old config class names available as deprecated aliases for backward compatibility

### Fixed
- DSR variance calculation now correctly follows López de Prado et al. (2025)
- Pandas index bug in `plot_quantile_returns`
- TradeMetrics.to_dict() now includes computed properties
- 181 MyPy type errors resolved

### Removed
- Invalid bootstrap tests that caused false failures
- Unused MultiIndex support in DataFrameAdapter (-60% code)

## [1.0.0] - 2025-09-01

### Added
- **Four-Tier Validation Framework**:
  1. Feature Analysis (pre-modeling)
  2. Model Diagnostics (during/after modeling)
  3. Backtest Analysis (post-modeling)
  4. Portfolio Analysis (production)

- **Cross-Validation Methods**:
  - Combinatorial Purged Cross-Validation (CPCV)
  - Purged Walk-Forward Cross-Validation
  - Calendar-aware splitters
  - Group isolation splitters

- **Statistical Tests**:
  - Deflated Sharpe Ratio (DSR) for multiple testing
  - Rademacher Anti-Serum (RAS) for overfitting detection
  - Benjamini-Hochberg FDR corrections
  - HAC-adjusted Information Coefficient

- **Feature Importance** (7 methods):
  - MDI (Mean Decrease Impurity)
  - PFI (Permutation Feature Importance)
  - MDA (Mean Decrease Accuracy)
  - SHAP values
  - Conditional IC
  - H-statistic
  - Consensus ensemble

- **Feature Diagnostics**:
  - Stationarity tests (ADF, KPSS, Phillips-Perron)
  - Autocorrelation analysis (ACF/PACF)
  - Volatility analysis (GARCH, ARCH)
  - Distribution analysis (tails, moments, normality)
  - Drift detection (PSI)

- **Visualization**:
  - Interactive Plotly charts
  - Heatmaps for IC correlation and feature importance
  - Time-series plots for returns and equity curves
  - Distribution plots for returns and residuals

- **Reporting**:
  - HTML reports (self-contained with embedded Plotly)
  - JSON export (machine-readable)
  - Markdown reports (version-control friendly)

- **Pydantic v2 Configuration**: Full validation and serialization support

- **40+ Statistical Metrics**: Comprehensive coverage of evaluation metrics

### References
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Bailey & López de Prado (2014). "The Deflated Sharpe Ratio"
- López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "How to use the Sharpe Ratio"
- Paleologo, G. (2024). "Elements of Quantitative Investing" (RAS implementation)

---

## Migration Notes

### From v1.0.0 to v1.1.0

**Config class renames** (old names still work as aliases):
```python
# Old (deprecated but functional)
from ml4t.diagnostic.config import FeatureEvaluatorConfig

# New (preferred)
from ml4t.diagnostic.config import DiagnosticConfig
```

**TradeShapConfig merged into TradeConfig**:
```python
# Old
from ml4t.diagnostic.config import TradeShapConfig

# New
from ml4t.diagnostic.config import TradeConfig
config = TradeConfig(
    shap=TradeShapSettings(...)
)
```

[Unreleased]: https://github.com/ml4t/ml4t-diagnostic/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ml4t/ml4t-diagnostic/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ml4t/ml4t-diagnostic/releases/tag/v1.0.0
