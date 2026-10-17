"""Real constant/unavailable IC workflows must retain observations without failed fits."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic import analyze_signal
from ml4t.diagnostic.visualization.signal import plot_ic_histogram, plot_ic_qq, plot_ic_ts


def _price_produced_ic(*, unavailable=False, dates_count=25, intraday=False, nanosecond=False):
    dates = (
        pd.date_range("2025-01-02", periods=dates_count, freq="h", tz="UTC")
        if intraday
        else pd.bdate_range("2025-01-02", periods=dates_count, tz="UTC")
    )
    if nanosecond:
        dates = pd.date_range("2025-01-02", periods=dates_count, freq="100ns", tz="UTC")
    assets = [f"asset-{number:02}" for number in range(12)]
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    factors = pd.DataFrame(
        {"factor": np.tile(np.arange(1, 13), dates_count)}, index=index
    ).reset_index()
    growth = 1 + np.arange(1, 13) / 64
    prices = (
        np.arange(1, 13)[None, :] * 2.0 ** np.arange(dates_count)[:, None]
        if unavailable
        else 64 * growth[None, :] ** np.arange(dates_count)[:, None]
    )
    marks = pd.DataFrame({"price": prices.reshape(-1)}, index=index).reset_index()
    return analyze_signal(
        factors,
        marks,
        periods=(1,),
        quantiles=3,
        filter_zscore=None,
        compute_turnover_flag=False,
    ).to_ic_result()


def test_constant_price_produced_ic_retains_every_histogram_sample_without_singular_kde():
    ic = _price_produced_ic()
    assert ic.ic_by_date == {"1D": [1.0] * 24}
    figure = plot_ic_histogram(ic)
    assert list(figure.data[0].x) == [1.0] * 24
    assert not any(trace.name == "KDE" for trace in figure.data)
    assert any("KDE unavailable" in str(note.text) for note in figure.layout.annotations)


@pytest.mark.parametrize("unavailable,dates_count", [(True, 25), (False, 2), (False, 25)])
def test_unavailable_or_degenerate_price_produced_ic_does_not_invent_fitted_quantiles(
    unavailable, dates_count
):
    ic = _price_produced_ic(unavailable=unavailable, dates_count=dates_count)
    figure = plot_ic_qq(ic)
    assert any("Q-Q fit unavailable" in str(note.text) for note in figure.layout.annotations)
    assert not any(trace.name == "Normal Reference" for trace in figure.data)
    observed = [] if unavailable else [1.0] * (dates_count - 1)
    assert [value for trace in figure.data for value in trace.y] == observed


def test_intraday_price_produced_ic_rolling_window_is_not_labelled_as_elapsed_days():
    ic = _price_produced_ic(intraday=True)
    figure = plot_ic_ts(ic, rolling_window=22, show_significance=False)
    assert figure.data[0].name == "Observed IC"
    rolling = next(trace for trace in figure.data if "Rolling" in trace.name)
    assert rolling.name == "22-Observation Rolling Mean"
    assert np.isnan(rolling.y[:21]).all()
    assert list(rolling.y[21:]) == [1.0] * 3


def test_price_produced_ic_time_series_retains_nanosecond_dates_already_owned_by_its_model():
    ic = _price_produced_ic(nanosecond=True)
    expected = pd.DatetimeIndex(ic.dates).as_unit("ns")
    assert len(expected.unique()) == 24
    figure = plot_ic_ts(ic, show_significance=False)
    actual = pd.DatetimeIndex(figure.data[0].x).as_unit("ns")
    assert list(actual.asi8) == list(expected.asi8)
