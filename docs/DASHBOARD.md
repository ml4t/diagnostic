# Trade SHAP Dashboard

The Streamlit dashboard presents trade-level SHAP results, statistical checks,
worst trades, feature effects, and recurring error patterns.

## Install

```bash
pip install "ml4t-diagnostic[dashboard]"
```

## Run the checked example

Clone the repository, then run the public example:

```bash
streamlit run examples/trade_shap_dashboard_demo.py
```

The release test suite starts this file with Streamlit's application test
runner and fails if the app raises an exception.

## Load saved results

The sidebar accepts JSON only. Pickle input is not supported because loading a
pickle can execute code. Treat uploaded JSON as untrusted data and validate its
contents before sharing or archiving reports.

The normalized result can contain:

- trade identifiers, timestamps, symbols, PnL, and percentage returns
- entry and exit prices, duration, direction, and quantity
- feature values and aligned SHAP values
- aggregate statistical validation results
- clustered error patterns and their supporting trades

The dashboard disables sections whose required data is absent.

## Export

The application can export normalized trades and patterns as CSV and the full
dashboard as HTML. CSV exports may begin with characters that spreadsheet
applications interpret as formulas; sanitize them before opening untrusted
exports in a spreadsheet.

## Supported import surfaces

The application implementation lives under
`ml4t.diagnostic.evaluation.trade_dashboard`. The compatibility module
`ml4t.diagnostic.evaluation.trade_shap_dashboard` remains available for code
written against beta releases. New applications should start from the checked
example script so data normalization and Streamlit state follow the supported
path.
