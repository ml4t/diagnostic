# Installation

## Requirements

- CPython 3.12, 3.13, or 3.14
- Polars 0.20+

## Basic Installation

```bash
pip install ml4t-diagnostic
```

## Optional Dependencies

ML4T Diagnostic has optional dependency groups for different use cases:

### Visualization

For Plotly charts, tearsheets, and PDF export:

```bash
pip install ml4t-diagnostic[viz]
```

Includes: `plotly`, `matplotlib`, `seaborn`, `kaleido`, `pypdf`

### Machine Learning Backends

For LightGBM and XGBoost model analysis:

On macOS, install the OpenMP runtime required by LightGBM first:

```bash
brew install libomp
```

Then install the optional dependencies on every supported platform:

```bash
pip install ml4t-diagnostic[ml]
```

Includes: `lightgbm`, `xgboost`

### Backtest Bridge

For `ml4t-backtest` integration and result-to-tearsheet bridges:

```bash
pip install ml4t-diagnostic[backtest]
```

Includes: `ml4t-backtest`

### Dashboard

For the optional Streamlit trade diagnostics dashboard:

```bash
pip install ml4t-diagnostic[dashboard]
```

Includes: `streamlit`

### Full Installation

Install all optional dependencies:

```bash
pip install ml4t-diagnostic[all]
```

## Development Installation

For contributing to ML4T Diagnostic:

```bash
git clone https://github.com/ml4t/diagnostic.git
cd ml4t-diagnostic
pip install -e ".[all,dev]"
```

## Using The Book Code Locally

If you are running the third-edition notebooks or case studies against a local checkout,
install the library in editable mode so the book code sees your current branch:

```bash
uv pip install -e /path/to/ml4t-diagnostic
```

See the [Book Guide](../book-guide/index.md) for the chapter and case-study map.
For the new reporting bridge, see the [Backtest Tearsheets](../user-guide/backtest-tearsheets.md) guide.

## Verify Installation

```python
import ml4t.diagnostic as diag
print(diag.__version__)
```

## Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| polars | ≥0.20.0 | Primary data processing |
| pandas | ≥2.0.0 | Compatibility layer |
| pyarrow | ≥14.0.0 | Pandas/Polars interoperability |
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.17.0 | Scientific computing |
| scikit-learn | ≥1.3.0 | ML utilities |
| joblib | ≥1.3.0 | Parallel computation |
| statsmodels | ≥0.14.0 | Statistical tests |
| tqdm | ≥4.66.0 | Progress reporting |
| pydantic | ≥2.13.4, <3 | Configuration validation |
| pyyaml | ≥6.0 | YAML configuration |
| pandas-market-calendars | ≥4.0.0 | Trading calendars |
| jinja2 | ≥3.1.0 | Report templates |
| arch | ≥7.2.0 | GARCH models |

### Optional

| Package | Group | Purpose |
|---------|-------|---------|
| lightgbm | ml | Gradient boosting |
| xgboost | ml | Gradient boosting |
| shap | ml | SHAP explanations (not installed on Intel macOS with Python 3.14) |
| numba | perf | JIT acceleration (not installed on Intel macOS with Python 3.14) |
| plotly | viz | Interactive charts |
| matplotlib | viz | Static charts |

### Migrating from beta releases

The stable 0.1.0 API removes beta features that were not validated for the
supported release platforms:

- the `gpu` and `tracking` extras
- `WandbLogger` and `log_experiment`
- `LoggingConfig.use_wandb`, `wandb_project`, and `wandb_entity`
- the `use_gpu` argument from `compute_shap_importance` and `TradeShapAnalyzer`
- the unvalidated `corrado` event-study test option; use `t_test` or `boehmer`

Install the `ml` extra for the supported SHAP implementation. Existing logging
configuration files containing removed fields now fail validation instead of
silently ignoring them.
