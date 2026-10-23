"""Use Data's acquisition adapters with Diagnostic, or replay the captured inputs offline.

The default reads bundled provider-derived CSVs and makes no network requests.
--fetch opts into Data's providers; no acquisition or cache implementation lives here.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
from ml4t.diagnostic.compat import empyrical as ep
from ml4t.diagnostic.compat import pyfolio as pf
from ml4t.diagnostic.evaluation import PortfolioAnalysis

DATA = Path(__file__).parent / "data"


def analyze_acquired_inputs(
    returns: pd.Series,
    factors: pd.DataFrame,
    treasury_yields: pd.DataFrame,
    *,
    factor_units: str = "decimal_returns",
    treasury_units: str = "annual_percent_yield",
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Consume the declared Data helper units without converting an annual quote to daily RF."""
    for frame, declared, required in (
        (factors, factor_units, "decimal_returns"),
        (treasury_yields, treasury_units, "annual_percent_yield"),
    ):
        if declared != required or frame.attrs.get("units", required) != required:
            raise ValueError(f"Expected declared {required} inputs")
    if not {"Mkt-RF", "RF"}.issubset(factors):
        raise ValueError("Fama-French inputs require separate Mkt-RF and RF columns")
    if not isinstance(treasury_yields.index, pd.DatetimeIndex) or treasury_yields.shape[1] != 1:
        raise ValueError("Treasury quotes require one dated annual-percent yield column")
    benchmark = (factors["Mkt-RF"] + factors["RF"]).rename("Market total return")
    analysis = PortfolioAnalysis.from_frames(
        returns, benchmark=benchmark, calendar=None, periods_per_year=periods_per_year
    )
    periodic_rf = factors["RF"].reindex(returns.index)
    dated_rf_sharpe = ep.sharpe_ratio(
        returns, risk_free=periodic_rf, annualization=periods_per_year
    )
    report = pf.create_full_tear_sheet(
        returns,
        benchmark_rets=benchmark,
        periods_per_year=periods_per_year,
        header_rows={
            "RF-adjusted Sharpe": f"{dated_rf_sharpe:.6g} (dated periodic RF)",
            "FRED quotes": "Annual percent yield; not substituted for periodic RF",
        },
        return_fig=True,
    )
    report.availability["acquisition_inputs"] = {
        "status": "available",
        "factor_units": factor_units,
        "treasury_units": treasury_units,
        "benchmark_definition": "Mkt-RF + RF; dated decimal total market returns",
        "report_statistics_risk_free": "Original familiar report defaults; zero RF",
        "rf_adjusted_sharpe": float(dated_rf_sharpe) if np.isfinite(dated_rf_sharpe) else None,
        "return_observations": len(returns),
        "factor_observations": len(factors),
        "treasury_quote_observations": len(treasury_yields),
        "availability": "Unknown historical publication instants; cache mtime is not known-at",
        "reason": (
            "Supplied adjusted-close returns. No transactions or marked account holdings are "
            "inferred. FRED annual quotes are retained separately from daily decimal RF; "
            "no automatic yield-to-return conversion."
        ),
    }
    return {"analysis": analysis, "report": report, "dated_rf_sharpe": dated_rf_sharpe}


def load_captured_inputs(
    directory: Path = DATA,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Read ordinary provider-derived CSV inputs without Data or a legacy package."""
    returns = pd.read_csv(directory / "acquired_returns.csv", index_col=0)
    factors = pd.read_csv(directory / "acquired_factors.csv", index_col=0)
    yields = pd.read_csv(directory / "acquired_treasury_yields.csv", index_col=0)
    for frame in (returns, factors):
        frame.index = pd.to_datetime(frame.index, utc=True)
    yields.index = pd.to_datetime(yields.index)
    return returns["SPY"], factors, yields


def fetch_acquired_inputs(
    symbol: str, start: str, end: str, cache_directory: Path
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Opt into existing Data providers and request-scoped legacy CSV caches.

    Yahoo's end is exclusive; Fama-French/FRED bounds are inclusive. FRED is
    fetched explicitly, not cached by get_returns_cached or used as a return.
    CSV cache freshness follows Data's file-mtime contract, not publication time.
    """
    if not symbol.strip() or date.fromisoformat(start) >= date.fromisoformat(end):
        raise ValueError("Supply a symbol and increasing YYYY-MM-DD date bounds")
    from ml4t.data.compat.empyrical import (
        get_fama_french,
        get_returns_cached,
        get_symbol_returns_from_yahoo,
        get_treasury_yield,
    )

    scope = f"{start}-{end}"
    returns = get_returns_cached(
        cache_directory / f"prices-{quote(symbol, safe='')}-{scope}.csv",
        get_symbol_returns_from_yahoo,
        end,
        symbol=symbol,
        start=start,
        end=end,
    )
    factors = get_returns_cached(
        cache_directory / f"factors-{scope}.csv", get_fama_french, end, start=start, end=end
    )
    yields = get_treasury_yield(start, end)
    return returns[symbol], factors, yields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-01-06")
    parser.add_argument("--cache-directory", type=Path, default=Path(".cache/acquisition"))
    parser.add_argument("--output", type=Path, default=Path("acquisition-report.html"))
    parser.add_argument("--periods-per-year", type=int, default=252)
    args = parser.parse_args()
    if args.periods_per_year <= 0:
        parser.error("--periods-per-year must be positive")
    inputs = (
        fetch_acquired_inputs(args.symbol, args.start, args.end, args.cache_directory)
        if args.fetch
        else load_captured_inputs()
    )
    result = analyze_acquired_inputs(*inputs, periods_per_year=args.periods_per_year)
    result["report"].save_html(args.output, include_plotlyjs=True)
    print(json.dumps(result["report"].availability["acquisition_inputs"], indent=2))
    print("Wrote acquired-input report; daily decimal RF and annual percent yield stay distinct")
    source = (
        "explicit live Data providers"
        if args.fetch
        else "captured provider-derived CSVs; no network"
    )
    print(f"Source: {source}")


if __name__ == "__main__":
    main()
