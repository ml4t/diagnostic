# reporting/ - Renderer Abstraction

Format-specific report generators for structured diagnostic results.

## Main Exports

- `ReportFactory`
- `ReportFormat`
- `ReportGenerator`
- `HTMLReportGenerator`
- `JSONReportGenerator`
- `MarkdownReportGenerator`

## Notes

- This package renders structured result objects into transport formats.
- The richer backtest tearsheet stack lives under `integration/` and `visualization/backtest/`.
