"""Quantile spread uncertainty must use the same dated/grouped bucket samples."""

import itertools
import math
import statistics

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def _inputs(representation="polars", nanoseconds=False):
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    if nanoseconds:
        dates = pd.DatetimeIndex([dates[0] + pd.Timedelta(i * 100, unit="ns") for i in range(6)])
    assets = list("ABCDEF")
    outcomes = [-0.25, 0.125, 0.25, 0.5, 0.375, 0.875]
    prices = {}
    for i, (asset, outcome) in enumerate(zip(assets, outcomes, strict=True)):
        middle = 1 + (i + 1) / 8
        after = middle * (1 + outcome)
        second = after * (1 + (i + 1) / 16)
        close = second * (1 + outcomes[-1 - i])
        prices[asset] = [1.0, middle, after, second, close, close]
    prices = pd.DataFrame(prices, index=dates)
    factor = prices.pct_change(fill_method=None).loc[dates[[1, 3]]].stack(future_stack=True)
    factor.index = factor.index.set_names(["date", "asset"])
    factor = factor.rename("factor").drop((dates[3], "F"))
    marks = prices.stack(future_stack=True).rename("close").rename_axis(["date", "asset"])
    forward = compute_forward_returns(marks.reset_index(), periods=[1, 2], group_col="asset")
    forward = forward.set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]].rename(
        columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"}
    )
    groups = pd.Series(
        [
            "L" if asset in ("ABC" if day == dates[1] else "ADE") else "H"
            for day, asset in factor.index
        ],
        index=factor.index,
        name="group",
    )
    rows = []
    for day, selected in [(dates[1], assets), (dates[3], assets[:-1])]:
        buckets = [1, 1, 2, 2, 3, 3] if day == dates[1] else [1, 1, 2, 3, 3]
        for asset, bucket in zip(selected, buckets, strict=True):
            i = assets.index(asset)
            one = outcomes[i] if day == dates[1] else outcomes[-1 - i]
            two = (1 + one) * (1 + (i + 1) / 16) - 1 if day == dates[1] else one
            rows.append((day, asset, groups.loc[(day, asset)], bucket, one, two))
    if representation == "polars":
        factor, forward, groups = (
            pl.from_pandas(frame.reset_index()) for frame in [factor, forward, groups]
        )
    panel = prepare_signal_panel(factor, forward, groups=groups, quantiles=3, max_loss=0)
    assert sorted(
        (pd.Timestamp(row["date"]).value, row["asset"], row["quantile"], row["group"])
        for row in panel.data.to_pandas().to_dict(orient="records")
    ) == sorted((day.value, asset, bucket, group) for day, asset, group, bucket, _, _ in rows)
    return panel, rows


def _expected(rows, by_date, by_group, demeaned, group_adjust, upper=3, lower=1):
    daily = {}
    for day, _, group, bucket, one, two in rows:
        for index, value in enumerate([one, two], start=4):
            peers = [r[index] for r in rows if r[0] == day and (not group_adjust or r[2] == group)]
            adjusted = value - math.fsum(peers) / len(peers) if demeaned or group_adjust else value
            key = (day.value, group if by_group else None, bucket, f"{index - 3}D")
            daily.setdefault(key, []).append(adjusted)
    samples = daily
    if not by_date:
        samples = {}
        for (_, group, bucket, horizon), values in daily.items():
            samples.setdefault((None, group, bucket, horizon), []).append(
                math.fsum(values) / len(values)
            )
        if by_group:
            for group, bucket, horizon in itertools.product(["L", "H"], [1, 2, 3], ["1D", "2D"]):
                samples.setdefault((None, group, bucket, horizon), [])
    pairs = {
        (day, group, horizon) for day, group, bucket, horizon in samples if bucket in [upper, lower]
    }
    expected = {}
    for day, group, horizon in pairs:
        legs = [samples.get((day, group, bucket, horizon)) for bucket in [upper, lower]]
        means = [math.fsum(values) / len(values) if values else None for values in legs]
        errors = [
            statistics.stdev(values) / math.sqrt(len(values))
            if values and len(values) > 1
            else None
            for values in legs
        ]
        expected[(day, group, horizon)] = {
            "spread": means[0] - means[1] if all(mean is not None for mean in means) else None,
            "standard_error": math.hypot(*errors)
            if all(error is not None for error in errors)
            else None,
            "upper_mean": means[0],
            "lower_mean": means[1],
            "upper_standard_error": errors[0],
            "lower_standard_error": errors[1],
            "upper_n_obs": len(legs[0]) if legs[0] is not None else None,
            "lower_n_obs": len(legs[1]) if legs[1] is not None else None,
        }
    return expected


def _spread(panel, representation, route, **options):
    if route == "panel":
        return panel.quantile_spread(**options)
    from ml4t.diagnostic.signal import compute_quantile_spread

    statistics_options = {
        key: value
        for key, value in options.items()
        if key in {"by_date", "by_group", "demeaned", "group_adjust"}
    }
    statistics_frame = panel.quantile_statistics(**statistics_options)
    before = statistics_frame.clone()
    value = statistics_frame.to_pandas() if representation == "pandas" else statistics_frame
    original = value.copy(deep=True) if representation == "pandas" else value.clone()
    result = compute_quantile_spread(
        value, **{key: value for key, value in options.items() if key not in statistics_options}
    )
    if representation == "pandas":
        pd.testing.assert_frame_equal(value, original)
    else:
        assert value.equals(original)
    assert statistics_frame.equals(before)
    return result


@pytest.mark.parametrize(
    "representation,route,by_date,by_group,demeaned,group_adjust",
    list(
        itertools.product(
            ["pandas", "polars"],
            ["panel", "statistics"],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        )
    ),
)
def test_spread_preserves_quote_derived_bucket_samples_and_independent_adjustments(
    representation, route, by_date, by_group, demeaned, group_adjust
):
    panel, rows = _inputs(representation, nanoseconds=True)
    original = panel.data
    result = _spread(
        panel,
        representation,
        route,
        upper_quantile=3,
        lower_quantile=1,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
    )
    expected = _expected(rows, by_date, by_group, demeaned, group_adjust)
    assert result.height == len(expected)
    for row in result.to_pandas().to_dict(orient="records"):
        key = (
            pd.Timestamp(row["date"]).value if by_date else None,
            row.get("group"),
            row["return_column"],
        )
        for name, value in expected[key].items():
            actual = row[name]
            assert (
                pd.isna(actual)
                if value is None
                else actual == pytest.approx(value, rel=1e-12, abs=1e-14)
            )
        assert row["upper_quantile"] == 3 and row["lower_quantile"] == 1
        assert row["sample_unit"] == ("asset" if by_date else "date")
        assert row["return_adjustment"] == (
            "date_group_mean" if group_adjust else "date_mean" if demeaned else "none"
        )
        assert row["exposure_convention"] == "long_upper_short_lower_unit_each"
    assert panel.data.equals(original)


@pytest.mark.parametrize(
    "representation,route", list(itertools.product(["pandas", "polars"], ["panel", "statistics"]))
)
def test_optional_errors_and_same_bucket_keep_reference_independent_error_convention(
    representation, route
):
    panel, _ = _inputs(representation)
    result = _spread(
        panel,
        representation,
        route,
        upper_quantile=2,
        lower_quantile=2,
        by_date=False,
        demeaned=False,
    )
    for row in result.to_dicts():
        assert row["spread"] == 0
        assert row["standard_error"] == pytest.approx(math.sqrt(2) * row["upper_standard_error"])
    without = _spread(
        panel,
        representation,
        route,
        upper_quantile=2,
        lower_quantile=2,
        by_date=False,
        demeaned=False,
        include_standard_error=False,
    )
    assert without["standard_error"].null_count() == without.height
    assert without["spread"].to_list() == result["spread"].to_list()


@pytest.mark.parametrize("scale", [1e-308, 1e308])
@pytest.mark.parametrize(
    "representation,route", list(itertools.product(["pandas", "polars"], ["panel", "statistics"]))
)
def test_propagated_error_remains_representable_for_actual_pnl_nav_outcomes(
    scale, representation, route
):
    panel, _ = _inputs(representation)
    factor = (
        panel.data.filter(pl.col("date") == panel.data["date"].min())
        .head(4)
        .select("date", "asset", "factor")
    )
    # A supplied PnL/NAV workflow can emit tiny or huge returns absent quote rounding.
    returns = factor.select("date", "asset").with_columns(
        pl.Series("1D", [-1 / scale, 1 / scale, -1 / scale, 1 / scale])
    )
    panel = prepare_signal_panel(factor, returns, quantiles=2, max_loss=0)
    result = _spread(
        panel,
        representation,
        route,
        upper_quantile=2,
        lower_quantile=1,
        by_date=True,
        demeaned=False,
    )
    row = result.row(0, named=True)
    assert row["spread"] == 0
    assert row["standard_error"] == pytest.approx(math.sqrt(2) / scale, rel=1e-14, abs=0)
    assert row["upper_n_obs"] == row["lower_n_obs"] == 2


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, 7])
@pytest.mark.parametrize("route", ["panel", "statistics"])
def test_invalid_or_absent_bucket_rejected_before_changing_owned_panel(bad, route):
    panel, _ = _inputs()
    original = panel.data
    with pytest.raises(ValueError, match="quantile|bucket"):
        _spread(panel, "polars", route, upper_quantile=bad, lower_quantile=1)
    assert panel.data.equals(original)


def test_duplicate_statistics_batches_reject_instead_of_multiplying_spread_rows():
    from ml4t.diagnostic.signal import compute_quantile_spread

    panel, _ = _inputs()
    frame = panel.quantile_statistics(by_date=True)
    batch = pl.concat([frame, frame])
    with pytest.raises(ValueError, match="duplicate"):
        compute_quantile_spread(batch, upper_quantile=3, lower_quantile=1)
    assert batch.height == 2 * frame.height


@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_noncontiguous_bucket_labels_and_requested_horizon_order_survive_spread_join(
    representation,
):
    from ml4t.diagnostic.signal import compute_quantile_spread, compute_quantile_statistics

    panel, _ = _inputs(representation, nanoseconds=True)
    source = panel.data.with_columns((pl.col("quantile") * 2).alias("quantile"))
    statistics_frame = compute_quantile_statistics(
        source, return_columns=["2D", "1D"], by_date=True, demeaned=False
    )
    actual = compute_quantile_spread(
        statistics_frame.to_pandas() if representation == "pandas" else statistics_frame,
        upper_quantile=6,
        lower_quantile=2,
    )
    expected = panel.quantile_spread(
        upper_quantile=3, lower_quantile=1, by_date=True, demeaned=False
    )
    assert actual["return_column"].unique(maintain_order=True).to_list() == ["2D", "1D"]
    assert (
        actual.drop("upper_quantile", "lower_quantile")
        .sort("return_column", "date")
        .equals(expected.drop("upper_quantile", "lower_quantile").sort("return_column", "date"))
    )


def test_empty_prepared_sample_returns_typed_empty_spread_without_inventing_legs():
    panel, _ = _inputs()
    data = panel.data
    warmup = data.select("date", "asset", "factor").with_columns(
        pl.col("factor").shift(20).over("asset")
    )
    empty = prepare_signal_panel(
        warmup,
        data.select("date", "asset", "1D", "2D"),
        quantiles=3,
        max_loss=1,
    )
    result = empty.quantile_spread(upper_quantile=3, lower_quantile=1, by_date=True)
    assert result.is_empty()
    assert result.schema["date"] == data.schema["date"]
    assert result.schema["spread"] == result.schema["standard_error"] == pl.Float64


def test_mean_only_statistics_support_omitted_errors_without_zero_uncertainty():
    from ml4t.diagnostic.signal import compute_quantile_spread

    panel, _ = _inputs()
    frame = panel.quantile_statistics().drop("standard_error")
    result = compute_quantile_spread(
        frame, upper_quantile=3, lower_quantile=1, include_standard_error=False
    )
    assert result["spread"].null_count() == 0
    assert result["standard_error"].null_count() == result.height
    with pytest.raises(ValueError, match="standard_error"):
        compute_quantile_spread(frame, upper_quantile=3, lower_quantile=1)
