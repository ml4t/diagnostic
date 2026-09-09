# ml4t-diagnostic

[![Python 3.12-3.14](https://img.shields.io/badge/python-3.12--3.14-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-diagnostic)](https://pypi.org/project/ml4t-diagnostic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Signal diagnostics, statistical validation, and backtest evaluation for quantitative trading workflows.

Use `ml4t-diagnostic` to evaluate cross-sectional signals, construct purged
time-series validation folds, correct strategy statistics for selection bias,
analyze feature and trade behavior, and produce backtest reports.

## ML4T Library Ecosystem

`ml4t-diagnostic` is one of seven libraries supporting the workflow described
in [Machine Learning for Trading](https://www.ml4trading.io/).

![ML4T library ecosystem](docs/images/ml4t_ecosystem_workflow_color.png)

It accepts engineered features, predictions, and backtest results from the
other ML4T libraries, but the primary signal-analysis workflow below has no
external service or special hardware requirement.

## Installation and Support

The supported Python versions are 3.12, 3.13, and 3.14 on Linux, macOS, and
Windows.

```bash
uv add ml4t-diagnostic
```

Python 3.15 is not currently supported because required dependency wheels are
still being qualified. Progress is tracked in
[issue #45](https://github.com/ml4t/diagnostic/issues/45).

## Quick Start

This example creates a synthetic cross-sectional factor whose score affects
the next price change, then measures its information coefficient and quantile
spread.

```python
import numpy as np
import polars as pl

from ml4t.diagnostic import analyze_signal

rng = np.random.default_rng(42)
dates = pl.date_range(pl.date(2025, 1, 1), pl.date(2025, 2, 28), eager=True)[:40]
assets = [f"asset_{index:02d}" for index in range(20)]

factor_rows = []
price_rows = []
prices = np.full(len(assets), 100.0)
for date in dates:
    scores = rng.normal(size=len(assets))
    factor_rows.extend(
        {"date": date, "asset": asset, "factor": score}
        for asset, score in zip(assets, scores, strict=True)
    )
    price_rows.extend(
        {"date": date, "asset": asset, "price": price}
        for asset, price in zip(assets, prices, strict=True)
    )
    prices *= 1 + 0.002 * scores + rng.normal(scale=0.005, size=len(assets))

result = analyze_signal(
    factor=pl.DataFrame(factor_rows),
    prices=pl.DataFrame(price_rows),
    periods=(1, 5),
)

assert result.ic["1D"] > 0.1
print(f"IC (1D): {result.ic['1D']:.4f}")
print(f"IC t-stat (1D): {result.ic_t_stat['1D']:.2f}")
print(f"Q5-Q1 spread (1D): {result.spread['1D']:.2%}")
```

`analyze_signal` returns information coefficients, significance statistics,
quantile returns, spreads, turnover, and related diagnostics for each requested
forward period. See the executable
[quickstart tutorial](docs/getting-started/quickstart.md) for the input schema
and a multiple-testing example.

## Main Capabilities

| Area | Public workflows |
|------|------------------|
| Signal analysis | `analyze_signal`, HAC-adjusted IC, quantile profiles, turnover |
| Cross-validation | `WalkForwardCV`, `CombinatorialCV`, `ValidatedCrossValidation` |
| Selection bias | Deflated Sharpe Ratio, PBO, RAS, FDR control, White's Reality Check |
| Feature analysis | `FeatureDiagnostics`, importance, interactions, drift, causality audit |
| Backtest analysis | `BacktestProfile`, portfolio metrics, factor attribution, trade diagnostics |
| Reporting | Plotly charts, dashboards, HTML tearsheets, static export |

## Optional Features

Install only the integrations needed by your workflow:

```bash
uv add 'ml4t-diagnostic[viz]'       # Plotly charts and static export
uv add 'ml4t-diagnostic[ml]'        # LightGBM, XGBoost, and supported SHAP builds
uv add 'ml4t-diagnostic[perf]'      # Optional Numba acceleration
uv add 'ml4t-diagnostic[backtest]'  # ml4t-backtest result bridge
uv add 'ml4t-diagnostic[data]'      # ml4t-data integration
uv add 'ml4t-diagnostic[factors]'   # Factor-data sourcing through ml4t-data
uv add 'ml4t-diagnostic[dashboard]' # Streamlit dashboard
uv add 'ml4t-diagnostic[all]'       # All supported optional features
```

LightGBM requires an OpenMP runtime on macOS. SHAP and Numba are excluded on
Intel macOS with Python 3.14 because compatible wheels are unavailable. Static
Plotly image and PDF export through current Kaleido releases may require a
local Chrome or Chromium installation. Core signal analysis does not require
these optional runtimes.

## Documentation

- [Documentation](https://www.ml4trading.io/docs/diagnostic/)
- [Installation and optional dependencies](docs/getting-started/installation.md)
- [Cross-validation](docs/user-guide/cross-validation.md)
- [Statistical tests](docs/user-guide/statistical-tests.md)
- [Feature diagnostics](docs/user-guide/feature-diagnostics.md)
- [Feature selection](docs/user-guide/feature-selection.md)
- [Backtest tearsheets](docs/user-guide/backtest-tearsheets.md)
- [Trade analysis](docs/user-guide/trade-analysis.md)
- [API reference](docs/api/index.md)
- [Book guide](docs/book-guide/index.md)
- [Issue tracker](https://github.com/ml4t/diagnostic/issues)
- [Release notes](https://github.com/ml4t/diagnostic/releases)

## Related Libraries

- [ml4t-data](https://github.com/ml4t/data) provides market and factor data.
- [ml4t-engineer](https://github.com/ml4t/engineer) creates model features.
- [ml4t-models](https://github.com/ml4t/models) trains and evaluates models.
- [ml4t-backtest](https://github.com/ml4t/backtest) produces backtest results.
- [ml4t-live](https://github.com/ml4t/live) runs qualified strategies live.
- [ml4t-specs](https://github.com/ml4t/specs) defines shared artifact contracts.

## Development

```bash
git clone https://github.com/ml4t/diagnostic.git
cd diagnostic
uv sync --all-extras --dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check
uv run pytest tests/ -q -n auto --timeout 120
uv run mkdocs build --strict
pre-commit run --all-files
```

Pull requests should identify an owning issue and state any compatibility or
release impact.

## License

[MIT License](LICENSE)
