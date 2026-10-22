"""Run the familiar complete returns report without a Reloaded runtime."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat import pyfolio as pf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("returns-report.html"))
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    dates = pd.bdate_range("2024-01-02", periods=281, tz="UTC")
    benchmark = pd.Series(
        np.sin(np.arange(280) / 8) * 0.008 + 0.0001, index=dates[1:], name="Market"
    )
    price_returns = 0.7 * benchmark.to_numpy() + np.cos(np.arange(280) / 5) * 0.004 + 0.0002
    prices = pd.Series(np.r_[100.0, 100 * np.cumprod(1 + price_returns)], index=dates)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    before = returns.copy(deep=True)
    options = {
        "benchmark_rets": benchmark,
        "live_start_date": returns.index[260],
        "bootstrap": args.bootstrap,
        "return_fig": True,
    }
    report = pf.create_returns_tear_sheet(
        returns, n_samples=32, num_samples=64, random_seed=7, **options
    )
    assert len(report.figures) == 15 + int(args.bootstrap)
    expected = prices.iloc[-1] / prices.iloc[0]
    assert np.isclose(report.figures["Cumulative wealth"].data[2].y[-1], expected)
    assert report.figures["Cumulative wealth"].layout.meta["training_observations"] == 260
    assert report.figures["Rolling beta"].layout.meta["window_observations"] == [127, 253]
    if args.compare:
        import pyfolio
        from matplotlib import pyplot

        assert version("pyfolio-reloaded") == "0.10.0"
        original = pyfolio.create_returns_tear_sheet(
            returns, cone_std=None, set_context=False, **options
        )
        try:
            original.set_dpi(30)
            original.canvas.draw()
            np.testing.assert_allclose(
                original.axes[0].lines[2].get_ydata(),
                report.figures["Cumulative wealth"].data[2].y,
                rtol=1e-12,
            )
        finally:
            pyplot.close("all")
        print("Actual Pyfolio reference returns sheet executed; live wealth matches")
    report.save_html(args.output, include_plotlyjs=True)
    pd.testing.assert_series_equal(returns, before)
    print({name: version(name) for name in ("ml4t-diagnostic", "pandas", "numpy", "plotly")})
    print(
        f"Complete returns manifest: {len(report.figures)} panels; 260 training / 20 live observations"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
