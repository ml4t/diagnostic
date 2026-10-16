"""A real short-position producer must not become zero risk at nonpositive NAV."""

from datetime import UTC, datetime
from html.parser import HTMLParser

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result
from ml4t.diagnostic.integration.backtest_analytics import compute_occupancy_metrics


class _ExposureCaption(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "p" and dict(attrs).get("data-exposure-normalization") == "partial":
            self.active = True

    def handle_endtag(self, tag):
        if tag == "p":
            self.active = False

    def handle_data(self, text):
        if self.active:
            self.text.append(text)


class _OnceShort(Strategy):
    def __init__(self, trade=True):
        self.submitted = not trade

    def on_data(self, timestamp, data, context, broker):
        if "A" in data and not self.submitted:
            broker.submit_order("A", -10)
            self.submitted = True


def _actual_result(final_price=200.0, trade=True):
    marks = [100.0, 100.0, final_price]
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6]],
            "asset": ["A"] * 3,
            **dict.fromkeys(["open", "high", "low", "close"], marks),
            "volume": [10000.0] * 3,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        _OnceShort(trade),
        BacktestConfig(initial_cash=1000, allow_short_selling=True),
    ).run()
    if trade:
        assert len(result.fills) == 1
        assert result.fills[0].quantity == 10 and result.fills[0].price == 100
        assert result.fills[0].timestamp == prices["timestamp"][1]
    else:
        assert not result.fills
    return result


@pytest.mark.parametrize("final_price", [200.0, 250.0])
def test_real_nonpositive_account_equity_retains_dollars_and_unknown_normalized_exposure(
    final_price,
):
    result = _actual_result(final_price)
    raw = result.to_portfolio_state_dataframe().clone()
    profile = analyze_backtest_result(result)
    timeline = profile.occupancy["timeline"]
    last = timeline.row(-1, named=True)
    assert last["equity"] == 2000 - 10 * final_price
    assert last["gross_exposure"] == 10 * final_price
    assert last["net_exposure"] == -10 * final_price and last["open_positions"] == 1
    for name in ["invested_fraction", "gross_exposure_fraction", "net_exposure_fraction"]:
        assert last[name] is None
    assert last["normalization_status"] == "nonpositive_equity"
    assert timeline.select(raw.columns).equals(raw)
    assert result.to_portfolio_state_dataframe().equals(raw)


@pytest.mark.parametrize("final_price", [200.0, 250.0])
def test_partial_nav_normalization_discloses_coverage_and_never_inserts_zero_in_average(
    final_price,
):
    profile = analyze_backtest_result(_actual_result(final_price))
    occupancy = profile.occupancy
    metrics = occupancy["metrics"]
    assert metrics["avg_gross_exposure"] == pytest.approx(0.5)
    assert metrics["avg_invested_fraction"] == pytest.approx(0.5)
    assert metrics["avg_net_exposure"] == pytest.approx(-0.5)
    assert metrics["max_gross_exposure"] == 1 and metrics["max_net_exposure"] == 1
    assert metrics["time_in_market"] == pytest.approx(2 / 3)
    assert occupancy["coverage"] == {
        "n_observations": 3,
        "n_normalized_observations": 2,
        "normalization_coverage": 2 / 3,
        "status": "partial",
    }
    availability = profile.availability.families["occupancy"]
    assert availability.status.value == "partial" and availability.coverage == pytest.approx(2 / 3)


@pytest.mark.parametrize("final_price", [200.0, 250.0])
def test_real_invalid_nav_end_window_has_no_fabricated_average_or_zero_position_count(final_price):
    result = _actual_result(final_price)
    selected = result.to_portfolio_state_dataframe().filter(pl.col("equity") <= 0)
    assert selected.height == 1
    occupancy = compute_occupancy_metrics(selected)
    for name in [
        "avg_invested_fraction",
        "avg_gross_exposure",
        "max_gross_exposure",
        "avg_net_exposure",
        "max_net_exposure",
    ]:
        assert occupancy["metrics"][name] is None
    assert occupancy["metrics"]["time_in_market"] == 1
    assert occupancy["metrics"]["avg_open_positions"] == 1
    assert occupancy["coverage"]["status"] == "unavailable"
    assert occupancy["coverage"]["n_normalized_observations"] == 0


def test_real_outside_observation_window_reports_no_samples_instead_of_cash_zero():
    raw = _actual_result(150).to_portfolio_state_dataframe()
    selected = raw.filter(pl.col("timestamp") > pl.col("timestamp").max())
    assert selected.height == 0
    occupancy = compute_occupancy_metrics(selected)
    assert all(value is None for value in occupancy["metrics"].values())
    assert occupancy["coverage"] == {
        "n_observations": 0,
        "n_normalized_observations": 0,
        "normalization_coverage": None,
        "status": "unavailable",
    }
    assert occupancy["timeline"].select(selected.columns).equals(selected)


@pytest.mark.parametrize("trade", [False, True])
def test_actual_all_cash_zero_and_positive_nav_leverage_remain_defined(trade):
    profile = analyze_backtest_result(_actual_result(150.0, trade))
    last = profile.occupancy["timeline"].row(-1, named=True)
    assert last["equity"] == (500 if trade else 1000)
    assert last["gross_exposure_fraction"] == (3 if trade else 0)
    assert last["net_exposure_fraction"] == (-3 if trade else 0)


@pytest.mark.parametrize("final_price", [200.0, 250.0])
def test_real_profile_exposure_plot_preserves_unsupported_ratio_as_a_gap(final_price):
    from ml4t.diagnostic.visualization.backtest import plot_occupancy_overview

    profile = analyze_backtest_result(_actual_result(final_price))
    figure = plot_occupancy_overview(profile)
    assert list(figure.data[0].y) == [0.0, 1.0, None]
    assert list(figure.data[1].y) == [0.0, -1.0, None]
    assert list(figure.data[2].y) == [0, 1, 1]
    assert figure.layout.meta["normalization_coverage"] == pytest.approx(2 / 3)
    for trace in figure.data:
        assert list(trace.customdata[-1]) == [
            2000 - 10 * final_price,
            10 * final_price,
            -10 * final_price,
            "nonpositive_equity",
        ]


def test_actual_zero_nav_full_report_discloses_the_conditional_normalization_sample():
    from ml4t.diagnostic.integration import generate_tearsheet_from_result

    html = generate_tearsheet_from_result(_actual_result(200), include_statistical=False)
    caption = _ExposureCaption()
    caption.feed(html)
    text = " ".join(caption.text)
    assert "2 of 3 snapshots" in text
    assert "defined ratios only" in text


def test_actual_negative_nav_performance_report_retains_explicit_simple_return_limit():
    from ml4t.diagnostic.integration import generate_tearsheet_from_result

    with pytest.raises(ValueError, match="simple returns below -1"):
        generate_tearsheet_from_result(_actual_result(250), include_statistical=False)
