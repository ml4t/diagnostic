"""Defined IC means must survive even when variance or inference is unavailable."""

import json
import math

import pandas as pd
import polars as pl
import pytest

import ml4t.diagnostic as diagnostic
from ml4t.diagnostic.metrics import (
    compute_forward_returns,
    cross_sectional_ic,
    cross_sectional_ic_series,
)
from ml4t.diagnostic.metrics.ic_inference import compute_ic_summary_stats
from ml4t.diagnostic.signal import prepare_data


def _inputs(case):
    dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
    x = [0.125, 0.25, 0.5, 1.0] if case != "undefined" else [0.25] * 4
    y = (
        x
        if case in ["positive", "constant_two"]
        else [-v for v in x]
        if case == "negative"
        else [0.25, -0.125, 0.5, 0.125]
    )
    if case == "mixed_two":
        y = x
    z = [-v for v in x] if case == "mixed_two" else y
    wide = pd.DataFrame(
        {
            asset: [
                128.0,
                128.0 * (1 + a),
                128.0 * (1 + a) * (1 + b),
                128.0 * (1 + a) * (1 + b) * (1 + a),
                128.0 * (1 + a) * (1 + b) * (1 + a) * (1 + c),
            ]
            for asset, a, b, c in zip(["A", "B", "C", "D"], x, y, z, strict=True)
        },
        index=dates,
    )
    selected = [dates[1], dates[3]] if case in ["constant_two", "mixed_two"] else [dates[1]]
    features = (
        wide.pct_change(fill_method=None)
        .loc[selected]
        .stack(future_stack=True)
        .rename("factor")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    prices = (
        wide.stack(future_stack=True)
        .rename("price")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    return features, prices


@pytest.mark.parametrize(
    "case", ["positive", "negative", "zero", "undefined", "constant_two", "mixed_two"]
)
@pytest.mark.parametrize("route", ["raw", "prepared", "direct", "helper"])
@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_causal_ic_points_do_not_become_missing_means_or_fabricated_zero_ratios(
    case, route, representation, tmp_path
):
    factor, prices = _inputs(case)
    if representation == "polars":
        factor, prices = pl.from_pandas(factor), pl.from_pandas(prices)
    expected_series = {
        "positive": [1.0],
        "negative": [-1.0],
        "zero": [0.0],
        "undefined": [],
        "constant_two": [1.0, 1.0],
        "mixed_two": [1.0, -1.0],
    }[case]
    n = len(expected_series)
    mean = math.fsum(expected_series) / n if n else math.nan
    std = (
        math.sqrt(math.fsum((v - mean) ** 2 for v in expected_series) / (n - 1))
        if n > 1
        else math.nan
    )
    ratio = mean / std if std > 0 else math.nan
    positive_fraction = sum(v > 0 for v in expected_series) / n if n else math.nan
    signal = None
    if route in ["raw", "prepared"]:
        if route == "raw":
            signal = diagnostic.analyze_signal(
                factor,
                prices,
                periods=(1,),
                quantiles=2,
                min_assets=2,
                filter_zscore=None,
                compute_turnover_flag=False,
            )
        else:
            panel = prepare_data(factor, prices, periods=(1,), quantiles=2, filter_zscore=None)
            signal = diagnostic.analyze_prepared_signal(
                panel, periods=(1,), quantiles=2, min_assets=2
            )
        assert signal.ic_series["1D"] == pytest.approx(expected_series, abs=1e-14)
        observed = {
            "mean": signal.ic["1D"],
            "std": signal.ic_std["1D"],
            "pct": signal.ic_positive_pct["1D"] / 100,
            "ratio": signal.ic_ir["1D"],
            "t": signal.ic_t_stat["1D"],
            "p": signal.ic_p_value["1D"],
        }
    else:
        predictions = (
            factor.rename({"factor": "prediction"})
            if representation == "polars"
            else factor.rename(columns={"factor": "prediction"})
        )
        forward = compute_forward_returns(prices, price_col="price", periods=1, group_col="asset")
        forward = (
            forward.rename({"fwd_ret_1": "forward_return"})
            if representation == "polars"
            else forward.rename(columns={"fwd_ret_1": "forward_return"})
        )
        series = cross_sectional_ic_series(predictions, forward, entity_col="asset", min_obs=2)
        if route == "direct":
            result = cross_sectional_ic(predictions, forward, entity_col="asset", min_obs=2)
            assert result["n_periods"] == n
            observed = {
                "mean": result["ic_mean"],
                "std": result["ic_std"],
                "pct": result["pct_positive"],
                "ratio": result["ic_ir"],
                "t": result["ic_t"],
                "p": result["p_value"],
            }
        else:
            result = compute_ic_summary_stats(series)
            assert result["n_periods"] == n
            observed = {
                "mean": result["mean_ic"],
                "std": result["std_ic"],
                "pct": result["pct_positive"],
                "t": result["t_stat"],
                "p": result["p_value"],
            }
    for key, expected in {
        "mean": mean,
        "std": std,
        "pct": positive_fraction,
        "ratio": ratio,
    }.items():
        if key not in observed:
            continue
        if math.isnan(expected):
            assert math.isnan(observed[key]), (key, observed)
        else:
            assert observed[key] == pytest.approx(expected, abs=1e-14), (key, observed)
    if case == "mixed_two":
        assert observed["t"] == pytest.approx(0.0) and observed["p"] == pytest.approx(1.0)
    else:
        assert math.isnan(observed["t"]) and math.isnan(observed["p"])
    if signal is not None:
        path = tmp_path / "summary.json"
        signal.to_json(str(path))
        encoded = json.loads(
            path.read_text(), parse_constant=lambda value: pytest.fail(f"nonstandard JSON {value}")
        )
        assert encoded["ic_ir"]["1D"] == (None if math.isnan(ratio) else ratio)
        restored = diagnostic.SignalResult.from_json(str(path))
        if math.isnan(ratio):
            assert math.isnan(restored.ic_ir["1D"])
        else:
            assert restored.ic_ir["1D"] == pytest.approx(ratio)


@pytest.mark.parametrize("route", ["raw", "prepared"])
@pytest.mark.parametrize("representation", ["pandas", "polars"])
def test_real_single_date_ic_null_variance_restores_numeric_nan_for_result_consumers(
    route, representation, tmp_path
):
    factor, prices = _inputs("positive")
    if representation == "polars":
        factor, prices = pl.from_pandas(factor), pl.from_pandas(prices)
    if route == "raw":
        result = diagnostic.analyze_signal(
            factor,
            prices,
            periods=(1,),
            quantiles=2,
            min_assets=2,
            filter_zscore=None,
            compute_turnover_flag=False,
        )
    else:
        result = diagnostic.analyze_prepared_signal(
            prepare_data(factor, prices, periods=(1,), quantiles=2, filter_zscore=None),
            periods=(1,),
            quantiles=2,
            min_assets=2,
        )
    assert math.isnan(result.ic_std["1D"])
    path = tmp_path / "single-observation.json"
    result.to_json(str(path))
    restored = diagnostic.SignalResult.from_json(str(path))
    for field in ["ic", "ic_std", "ic_t_stat", "ic_p_value", "ic_ir", "ic_positive_pct"]:
        expected = getattr(result, field)["1D"]
        observed = getattr(restored, field)["1D"]
        if math.isnan(expected):
            assert math.isnan(observed), field
        else:
            assert observed == pytest.approx(expected), field
    restored.summary()
