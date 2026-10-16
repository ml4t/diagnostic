"""Paired and cohort uncertainty must not compact different observation dates together."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.uncertainty import (
    compute_backtest_uncertainty,
    compute_paired_uncertainty,
    compute_reality_check,
    compute_selection_adjustment,
)


def test_multicolumn_return_export_cannot_be_flattened_into_one_portfolio_history():
    a, b, _ = _nav_returns()
    exported = pd.concat([a, b], axis=1).to_numpy()
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_backtest_uncertainty(exported, n_boot=25, block_length=2)


def _nav_returns():
    dates = pd.bdate_range("2025-01-02", periods=13, tz="UTC")
    rng = np.random.default_rng(415)
    nav_a = pd.Series(
        100.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.001, 0.012, 12)]), index=dates
    )
    nav_b = pd.Series(
        100.0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.0005, 0.009, 12)]), index=dates
    )
    a = nav_a.drop(dates[4]).reindex(dates).pct_change(fill_method=None).iloc[1:]
    b = nav_b.drop(dates[8]).reindex(dates).pct_change(fill_method=None).iloc[1:]
    common = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return a, b, common


def _run(route, a, b):
    if route == "paired":
        return compute_paired_uncertainty(a, b, n_boot=25, block_length=2, seed=17)
    if route == "selection":
        return compute_selection_adjustment({"a": a, "b": b})
    return compute_reality_check({"a": a}, b, n_bootstrap=25, block_size=2, seed=17)


@pytest.mark.parametrize("representation", ["pandas", "polars", "array", "list"])
@pytest.mark.parametrize("periods_per_year", [12, 252])
def test_nav_gaps_use_the_same_finite_observation_pairs_not_equal_compacted_lengths(
    representation, periods_per_year
):
    a, b, common = _nav_returns()
    original_a, original_b = a.copy(deep=True), b.copy(deep=True)
    converters = {
        "pandas": lambda x: x,
        "polars": lambda x: pl.Series(x.to_numpy()),
        "array": lambda x: x.to_numpy(),
        "list": lambda x: x.tolist(),
    }
    result = compute_paired_uncertainty(
        converters[representation](a),
        converters[representation](b),
        periods_per_year=periods_per_year,
        n_boot=25,
        block_length=2,
        seed=17,
    )
    difference = common.a - common.b
    assert len(common) == result.n_observations == 8
    assert result.information_ratio == pytest.approx(
        difference.mean() / difference.std(ddof=1) * np.sqrt(periods_per_year)
    )
    pd.testing.assert_series_equal(a, original_a)
    pd.testing.assert_series_equal(b, original_b)


@pytest.mark.parametrize("route", ["paired", "selection", "reality"])
def test_cohort_gaps_match_the_explicitly_prealigned_public_workflow(route):
    a, b, common = _nav_returns()
    result = _run(route, a, b)
    expected = _run(route, common.a, common.b)
    for name, value in asdict(expected).items():
        observed = getattr(result, name)
        if isinstance(value, float):
            assert observed == pytest.approx(value, nan_ok=True), name
        else:
            assert observed == value, name
    if route == "selection":
        sharpes = {
            key: common[key].mean() / common[key].std(ddof=1) * np.sqrt(252) for key in ["a", "b"]
        }
        assert result.leader == max(sharpes, key=sharpes.get)
        assert result.leader_sharpe == pytest.approx(sharpes[result.leader])


@pytest.mark.parametrize("route", ["paired", "selection", "reality"])
def test_reordered_matching_labels_align_before_joint_missingness_and_bootstrapping(route):
    a, b, common = _nav_returns()
    reordered = b.sample(frac=1, random_state=24)
    original = reordered.copy(deep=True)
    result = asdict(_run(route, a, reordered))
    expected = asdict(_run(route, common.a, common.b))
    for name, value in expected.items():
        if isinstance(value, float):
            assert result[name] == pytest.approx(value, nan_ok=True), name
        else:
            assert result[name] == value, name
    pd.testing.assert_series_equal(reordered, original)


@pytest.mark.parametrize(
    "problem",
    [
        "different_window",
        "duplicate_dates",
        "missing_date",
        "mixed_positional",
        "unequal_original_lengths",
        "two_dimensional",
    ],
)
@pytest.mark.parametrize("route", ["paired", "selection", "reality"])
def test_invalid_cohort_identity_is_rejected_before_pairing_or_variant_selection(problem, route):
    a, b, _ = _nav_returns()
    if problem == "different_window":
        b.index = b.index + pd.offsets.BDay(1)
    elif problem == "duplicate_dates":
        b.index = pd.DatetimeIndex([b.index[0], b.index[0], *b.index[2:]])
    elif problem == "missing_date":
        b.index = pd.DatetimeIndex([pd.NaT, *b.index[1:]])
    elif problem == "mixed_positional":
        b = b.to_numpy()
    elif problem == "unequal_original_lengths":
        a, b = a.to_numpy(), np.r_[b.to_numpy(), np.nan]
    else:
        a, b = a.to_numpy().reshape(6, 2), b.to_numpy().reshape(6, 2)
    with pytest.raises(
        ValueError, match="aligned|dates|dated|labels|length|unique|nonmissing|one-dimensional"
    ):
        _run(route, a, b)


@pytest.mark.parametrize("route", ["paired", "selection", "reality"])
def test_complete_cohort_keeps_observed_zero_cash_sessions(route):
    a, b, common = _nav_returns()
    a, b = common.a.copy(), common.b.copy()
    a.iloc[:3] = b.iloc[:3] = 0.0
    result = asdict(_run(route, a, b))
    positional = asdict(_run(route, a.to_numpy(), b.to_numpy()))
    for name, value in positional.items():
        if isinstance(value, float):
            assert result[name] == pytest.approx(value, nan_ok=True), name
        else:
            assert result[name] == value, name
    if route == "paired":
        assert result["n_observations"] == len(a)
