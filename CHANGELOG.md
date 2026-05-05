# Changelog

All notable changes to ml4t-diagnostic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b18] - 2026-05-05

### Fixed
- **Python 3.14 pandas floor**: lowered the Python 3.14 `pandas`
  requirement from `>=3.0.0` to `>=2.3.3`, which already publishes `cp314`
  wheels and avoids unnecessary dependency conflicts with downstream libraries.
- **Transitive dependency floor**: updated the `ml4t-engineer` dependency to
  `>=0.1.0b7`, the first release that also lowers its Python 3.14 `pandas`
  floor to `>=2.3.3`.

## [0.1.0b17] - 2026-05-05

### Fixed
- **Python 3.14 scikit-learn floor**: lowered the Python 3.14 `scikit-learn`
  requirement from `>=1.8.0` to `>=1.7.2`, which already publishes `cp314`
  wheels and avoids unnecessary dependency conflicts with downstream libraries.
- **Transitive dependency floor**: updated the `ml4t-engineer` dependency to
  `>=0.1.0b6`, the first release that also lowers its Python 3.14
  `scikit-learn` floor to `>=1.7.2`.

## [0.1.0b16] - 2026-05-05

### Fixed
- **Python 3.14 install support**: widened `requires-python` from `<3.14` to
  `<3.15` so `ml4t-diagnostic` no longer blocks Python 3.14 environments.
- **Python 3.14 dependency resolution**: pinned Python 3.14 onto `ml4t-engineer`
  and the compiled scientific stack (`pandas`, `pyarrow`, `scipy`,
  `scikit-learn`, `statsmodels`, `arch`, `numba`) release lines that publish
  `cp314` wheels, avoiding source-build failures during installation.
- **Python 3.14 CI coverage**: added Python 3.14 to the main test matrix to
  keep packaging support aligned with exercised CI coverage.

## [0.1.0b15] - 2026-04-30

### Added
- **Daily-metric uncertainty helpers** for cross-sectional ranking signals
  (`ml4t.diagnostic.metrics.uncertainty`):
  - `cross_sectional_auc_series(predictions, labels, ...)` — per-date AUC
    across the cross-section using a vectorized rank-based Mann-Whitney U
    formula in pure Polars (no `sklearn` per-date loop).
  - `compute_ic_uncertainty(daily_ic, horizon, ...)` — bundles naive,
    Newey-West HAC, and stationary block-bootstrap intervals for the mean
    of a daily-IC series. The HAC lag defaults to
    `max(horizon - 1, NW_auto)` because forward returns of horizon `H`
    induce up to `H-1` lags of serial dependence in the IC.
  - `compute_auc_uncertainty(daily_auc, horizon, ...)` — analogous wrapper
    for daily AUC, with the null centred on `null_value=0.5`.
- All three helpers exported from `ml4t.diagnostic.metrics`.

### Notes
- These helpers move the primary uncertainty estimate for
  cross-sectional ranking signals from "fold-level std (N≈8–16 folds)" to
  the daily-pooled IC/AUC time series (N=hundreds), which is the natural
  observational unit for daily ranking skill.

## [0.1.0b13] - 2026-04-22

### Added
- Canonical `ml4t.diagnostic.metrics` namespace for reusable metrics, including
  explicit IC helpers: `cross_sectional_ic_series()`, `cross_sectional_ic()`,
  and `pooled_ic()`.

### Changed
- Documentation examples now use `ml4t.diagnostic.metrics` for metrics imports.
- `ml4t.diagnostic.evaluation.metrics` remains available as a compatibility
  import surface.

## [0.1.0b8] - 2026-04-01

### Added
- **Feed-contract-aware run artifact normalization**: `profile_from_run_artifacts()`
  now normalizes prediction and signal surfaces using the feed's configured
  timestamp and entity columns before downstream profile and tearsheet logic runs
- **Regression coverage for custom feed columns**: integration tests now cover
  artifact directories where surfaces use non-default column names such as
  `ts_event` and `ticker`

### Changed
- **Artifact contract dependency**: artifact resolver and feed-related integration
  tests now use `ml4t-specs`, and the package declares `ml4t-specs` as a core dependency
- **Docs landing page rewrite**: the homepage now leads with the practical
  validation questions practitioners need answered instead of feature positioning copy
- **Agent guide standardization**: packaged navigation docs now use the canonical
  `AGENTS.md` filename throughout the repository, with updated discovery in
  `get_agent_docs()`

## [0.1.0b7] - 2026-03-28

### Added
- **Methodology tab**: 7th tearsheet tab documenting computation methods, data sources,
  and active fallbacks per section. Live ⚠/✓ badges show whether each data surface
  comes from a real artifact or reconstruction.
- **Data provenance footnotes**: Inline warnings on every section that uses reconstructed
  data (e.g., "⚠ Data source: reconstructed from weights (no real execution data)")
- **Real artifact preference**: `profile_from_run_artifacts()` now loads `fills.parquet`,
  `equity.parquet`, and `portfolio_state.parquet` when present on disk

### Fixed
- **Silent fallback elimination**: All 12 silent `except Exception: return empty` patterns
  now emit `warnings.warn()` with the actual error. Covers: prediction enrichment,
  config parsing, surface coercion, SQLite registry lookup, metadata extraction.
- **Turnover metric**: AVG TURNOVER was 0.00 due to dilution by non-trading days
- Weights-as-signals semantic mismatch now tracked in `data_sources`

### Changed
- `BacktestProfile` gains `data_sources: dict[str, str]` field tracking provenance
  of each data surface ("artifact" vs "reconstructed from ...")

## [0.1.0b6] - 2026-03-27

### Added
- **Rebalance timeline chart** on Trading tab: bar chart of filled notional per rebalance event
  with implementation cost overlay and symbols-touched annotations
- **Execution quality chart** on Trading tab: implementation shortfall distribution histogram
  with median/mean annotations per trade
- **`load_fama_french_5factor()`** convenience function for one-line FF5 data loading
- **`[factors]` optional extra**: `pip install ml4t-diagnostic[factors]` pulls in ml4t-data
  for automatic Fama-French/AQR data sourcing
- FF5 daily test fixture (`tests/fixtures/ff5_daily.parquet`) — Factors tab now renders
  in all matrix test scenarios

### Fixed
- **Turnover metric**: AVG TURNOVER was 0.00 due to mean being diluted by non-trading days.
  Now computed from rebalance-day rows only (e.g., 0.13 instead of 0.00)

## [0.1.0b4] - 2026-03-27

### Added
- **Backtest tearsheet overhaul** — 6-tab layout (Overview, Performance, Trading, Validation, ML, Factors):
  - Overview: KPI strip, equity+sidebar (66/34), credibility/cost pair, top/bottom contributors, monthly heatmap with annual column
  - Performance: equity curve, drawdowns table, underwater, rolling 2x2, annual+distribution pair, stock attribution
  - Trading: activity strip, exposure timeline, cost bridge+sensitivity, attribution, MFE/MAE+duration, worst trades, trade waterfall (collapsible)
  - Validation: validity card, CI+MinTRL pair, Sharpe bootstrap, drawdown anatomy (collapsible)
  - ML (conditional): IC time series (full width), decile returns + prediction-trade alignment pair, signal utilization KPI
  - Factors (conditional): exposures+legend sidebar, regression stats table, attribution waterfall, risk donut
- **ML prediction-trade bridge**: signal utilization rate in KPI strip, trade outcome by entry prediction decile chart
- **Export system**: `export_workspaces()` with 7 named bundles (executive, full_report, etc.)
- **Factor regression table**: Beta, SE, t-stat, p-value, R², Adj R² per factor

### Fixed
- Heatmap color scale: percentile-based clamping for extreme returns (crypto ±240%)
- ML column detection: `actual` added as outcome column candidate
- Sidebar drawdown >100%: `_is_fraction` heuristic removed, always format as fraction
- Cost bridge: Gross PnL bar now red when strategy loses money
- Attribution: filtered to traded symbols only (was showing entire portfolio)
- VaR annotation positioning: placed right of vertical lines, no overlap with axes
- MFE/MAE axis scaling, factor waterfall additivity, exit efficiency units
- Polars join datetime precision alignment (μs vs ms)

### Changed
- **BREAKING**: Prediction calibration "win rate" chart removed from ML tab (misleading for predictions)
- ML tab restructured: IC time series full width, "Quintile Returns" → "Decile Returns", paired with prediction scatter
- Trade waterfall enabled by default (collapsible for >100 trades)

### Removed
- 324 lines dead code: stale renderers (`_render_activity_overview`, `_render_key_insights`,
  `_render_key_metrics_table`, `_render_ml_summary`), orphaned `plot_prediction_calibration`,
  `_create_ml_summary_html`, broken `_is_fraction` heuristic

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
