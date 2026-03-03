# Changelog

All notable changes to ml4t-diagnostic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0b1] - 2026-03-03

### Added
- **Tail risk visualization**: VaR/CVaR histogram + metrics table for `risk_manager` tearsheet
- **Fold-aware SHAP bridge**: `compute_fold_shap()` for walk-forward CV model SHAP analysis
- **SHAP pattern plots**: error pattern bars + worst trades stacked bars
- **Tearsheet integration**: `shap_result` threaded through generate chain;
  `BacktestTearsheet.generate()` preserves `enable_section()`/`disable_section()` customizations
- LightGBM and XGBoost added to dev dependencies (55 previously-skipped tests now run)

### Changed
- **BREAKING**: `ml4t.diagnostic.evaluation` namespace trimmed from ~130 to 54 exports.
  Low-level functions (binary metrics, stationarity tests, distribution analysis,
  portfolio metric functions, etc.) are no longer re-exported. Import from submodules directly:
  - `from ml4t.diagnostic.evaluation.portfolio_analysis import sharpe_ratio`
  - `from ml4t.diagnostic.evaluation.stationarity import adf_test`
  - `from ml4t.diagnostic.evaluation.binary_metrics import precision, recall`
- `TradeShapAnalyzer` and result types promoted to `evaluation.__all__`
- Removed unused extras (`advanced`, `deep`, `all-ml`) and numpy/scipy upper version caps
- Added `strict=False` to `zip()` calls, simplified redundant Polars/pandas type branches

### Removed
- `test_engineer_integration.py` — obsolete after FeatureSelector migration (splitter behavior
  covered by 321 dedicated splitter tests)

## [0.1.0a11] - 2026-03-03

### Added
- **FeatureSelector migration** from ml4t-engineer:
  - `FeatureSelector` class with IC, importance, correlation, and drift filtering pipeline
  - `FeatureICResults`, `FeatureImportanceResults`, `FeatureOutcomeResult` types
  - Correlation filtering uses pure Polars (no pandas dependency)
  - 43 tests (30 unit + 13 integration)
- **SignalResult → Visualization bridge**:
  - `SignalResult.to_ic_result()`, `.to_quantile_result()`, `.to_tear_sheet()`
  - Direct bridge from signal analysis to interactive plots
- Feature selection user guide (`docs/user-guide/feature-selection.md`)

### Fixed
- Theme fixes: Pattern A elimination, THEME_DARK legend/axis configuration
- `show()` method on plot functions

### Changed
- Methods docs updated for all evaluation modules
- HTML report palette improvements

## [0.1.0a10] - 2026-02-27

### Added
- **Calendar-First WalkForwardCV** (major):
  - All fold arithmetic in session space (trading days), not sample space
  - NYSE default calendar for WalkForwardCV
  - Non-trading row filtering: weekends/holidays excluded from fold indices
  - `"4W"` = 20 trading sessions, not 28 calendar days
  - New `TradingCalendar` methods: `get_sessions_and_mask()`, `time_spec_to_sessions()`
  - `filter_non_trading: bool = True` in SplitterConfig
  - Graceful fallback for numpy arrays (no timestamps → sample-based splitting)
  - Panel data support with duplicate timestamp handling
  - 73 new calendar tests, 321 total splitter tests
- `entity_col` parameter for `compute_ic_series()` (panel data IC computation)

### Deprecated
- `align_to_sessions` parameter — calendar-first splitting subsumes session alignment

## [0.1.0a5] - 2026-02-11

### Added
- **CV Fold Visualization**: `plot_cv_folds()` for interactive timeline visualization
  - Train/validation/test periods with purge gaps
  - Works with WalkForwardCV and CombinatorialCV splitters
  - Supports pandas, Polars, numpy data with automatic timestamp detection
  - Interactive hover, theme support (default, dark, print, presentation)
  - 36 tests

## [0.1.0a4] - 2026-02-04

### Fixed
- `compute_forward_returns()` now uses PRICE date universe instead of FACTOR date universe
  - Bug caused ~6% IC error when factor dates were subset of price dates
  - Forward returns correctly compute "N trading days forward" using price calendar

## [0.1.0a3] - 2026-01-19

### Added
- **ValidatedCrossValidation**: Combines CPCV + DSR in one workflow
  - `validated_cross_val_score()` convenience function
  - `ValidationResult` with summary(), to_dict(), interpretation
- **Result interface standardization**: `interpret()` method on BaseResult and key result classes
- **ml4t-data integration contract**: DataQualityReport, DataQualityMetrics, DataAnomaly
- Canonical integration surface at `ml4t.diagnostic.api`
- Machine-readable API contract (`tests/contracts/public_api_contract.json`)
- 50 IC tests, 14 real data integration tests
- IC module coverage: 9% → 98%

### Fixed
- DSR/PSR/MinTRL test fixtures match López de Prado et al. (2025) paper values
- MinTRL formula uses SR₀ (not observed SR) in variance adjustment

## [0.1.0a2] - 2025-11-15

### Added
- **Trade SHAP Analysis**: `TradeShapAnalyzer` for SHAP-based trade error pattern analysis
- **Price Excursion Analysis**: `analyze_excursions()` for TP/SL parameter optimization
- **Barrier Analysis**: `BarrierAnalysis` module for triple barrier outcome evaluation
- **Portfolio Analysis**: `PortfolioAnalysis` as pyfolio replacement with modern API
- **IC Time Series**: Alphalens-replacement IC computation with decay analysis
- Dashboard export and caching functions

### Changed
- Config consolidation: 61+ config classes → 10 primary configs
  (`DiagnosticConfig`, `StatisticalConfig`, `PortfolioConfig`, `TradeConfig`,
  `SignalConfig`, `EventConfig`, `BarrierConfig`, `ReportConfig`, `RuntimeConfig`)
- Single-level nesting: `config.stationarity.enabled` pattern
- Preset methods: `for_quick_analysis()`, `for_research()`, `for_production()`

### Fixed
- DSR variance calculation follows López de Prado et al. (2025)
- Pandas index bug in `plot_quantile_returns`
- TradeMetrics.to_dict() includes computed properties

### Removed
- Invalid bootstrap tests that caused false failures
- Unused MultiIndex support in DataFrameAdapter

## [0.1.0a1] - 2025-09-01

### Added
- **Four-Tier Validation Framework**:
  1. Feature Analysis (pre-modeling)
  2. Model Diagnostics (during/after modeling)
  3. Backtest Analysis (post-modeling)
  4. Portfolio Analysis (production)
- **Cross-Validation**: CPCV, Purged Walk-Forward CV, calendar-aware splitters
- **Statistical Tests**: DSR, RAS, Benjamini-Hochberg FDR, HAC-adjusted IC
- **Feature Importance** (7 methods): MDI, PFI, MDA, SHAP, Conditional IC, H-statistic, Consensus
- **Feature Diagnostics**: Stationarity (ADF, KPSS, PP), ACF/PACF, GARCH, distribution, drift (PSI)
- **Visualization**: Interactive Plotly charts, heatmaps, time-series, distribution plots
- **Reporting**: HTML (self-contained), JSON, Markdown
- **Pydantic v2 Configuration**: Full validation and serialization

### References
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Bailey & López de Prado (2014). "The Deflated Sharpe Ratio"
- López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). "How to use the Sharpe Ratio"
- Paleologo, G. (2024). "Elements of Quantitative Investing" (RAS implementation)

---

## Migration Notes

### From 0.1.0a1 to 0.1.0a2

**Config class renames** (old names removed):
```python
# Old
from ml4t.diagnostic.config import FeatureEvaluatorConfig

# New
from ml4t.diagnostic.config import DiagnosticConfig
```

### From 0.1.0a9 to 0.1.0a10

**Calendar-first CV** replaces `align_to_sessions`:
```python
# Old (deprecated)
cv = WalkForwardCV(n_splits=5, align_to_sessions=True, session_col="date")

# New (preferred)
cv = WalkForwardCV(n_splits=5, calendar="NYSE")
```

[Unreleased]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a11...HEAD
[0.1.0a11]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a10...v0.1.0a11
[0.1.0a10]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a5...v0.1.0a10
[0.1.0a5]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/ml4t/ml4t-diagnostic/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/ml4t/ml4t-diagnostic/releases/tag/v0.1.0a1
