# visualization/ - Plotly And Report Rendering

Visualization primitives and report assembly for signals, features, portfolio analytics, factor attribution, and backtest tearsheets.

## Main Areas

| Path | Purpose |
|------|---------|
| [backtest/](backtest/AGENTS.md) | HTML tearsheets, templates, statistical-validity and ML sections |
| [factor/](factor/AGENTS.md) | Factor exposure, attribution, and diagnostics plots |
| `cv_plots.py` | Fold and timeline visualization |
| `feature_plots.py`, `interaction_plots.py` | Feature importance and interaction charts |
| `signal/` and `portfolio/` | Signal and portfolio plotting packages |
| `portfolio.create_round_trip_dashboard` | Prepared FIFO statistics, lifetimes, Beta(1,1), distributions and asset/sector attribution through the existing `PortfolioTearSheet` renderer |
| `compat.pyfolio.tears.create_round_trip_tear_sheet` | Familiar gross daily round-trip orchestration and completed-table plots reuse native kernels/renderer; explicit legacy definitions, estimated-denominator provenance and separately flagged terminal closes |
| Portfolio `intraday=` preparation | Explicit complete intraday snapshots keep their own clock in the same renderer; no return/EOD resampling or continuous-time-peak inference |
| Portfolio `stress=`, `live=` and `wealth_cones=` preparation | Owned native samples and training-only distributions rendered by the same report; reject different-account inputs, retain exact JSON clocks and separately label quantiles, std bounds and actual live observations |
| Explicit `estimated_intraday` results | Daily transaction-notional projections have separately labeled estimated panels; daily labels and unknown/inferred capital are not observed account facts |
| `report_generation.py` | Combined HTML and PDF export helpers |

## Notes

- `visualization/__init__.py` re-exports the public plotting surface.
- The newer reporting bridge lands in `visualization/backtest/` plus `integration/`.
