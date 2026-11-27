"""Outside bar labels on the prediction-decile trade chart must fit inside its plot area.

Each bar carries a two-line label ("N=18", "Win 72%") placed outside the bar. With plotly's
autorange the first line above the tallest bar was clipped by the plot edge, and the last line
below the most negative bar ran into the x tick labels.
"""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.visualization.backtest.ml_plots import plot_prediction_trade_alignment

# Two lines of 9 px text plus the offset between bar and label.
LABEL_PX = 2 * 9 + 6


def _profile(pnl: np.ndarray) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    trades = pl.DataFrame({"entry_prediction_value": rng.normal(size=pnl.size), "pnl": pnl}).sort(
        "entry_prediction_value"
    )
    return SimpleNamespace(prediction_enriched_trades_df=trades)


def _room_px(fig) -> tuple[float, float]:
    """Pixels between the extreme bars and the plot edges."""
    lo, hi = fig.layout.yaxis.range
    plot_px = fig.layout.height - fig.layout.margin.t - fig.layout.margin.b
    means = fig.data[0].y
    scale = plot_px / (hi - lo)
    return (hi - max(means)) * scale, (min(means) - lo) * scale


@pytest.mark.parametrize(
    "pnl",
    [
        # deciles with mixed signs, the tallest bar positive (the case seen in a tearsheet)
        np.repeat([-400, 900, -400, 2100, -1350, 2800, 100, -500, -380, 820], 18).astype(float),
        # one large loss decile
        np.repeat([-9000, 50, 60, 70, 80, 90, 100, 110, 120, 130], 18).astype(float),
    ],
)
def test_two_line_labels_fit_above_and_below_bars(pnl):
    fig = plot_prediction_trade_alignment(_profile(pnl))
    assert fig is not None
    assert fig.layout.yaxis.range is not None, "autorange leaves too little room for the labels"
    above, below = _room_px(fig)
    assert above >= LABEL_PX
    assert below >= LABEL_PX


def test_all_positive_bars_keep_zero_baseline():
    pnl = np.repeat(np.linspace(100, 1000, 10), 18)
    fig = plot_prediction_trade_alignment(_profile(pnl))
    lo, _ = fig.layout.yaxis.range
    assert lo == 0
    assert _room_px(fig)[0] >= LABEL_PX
