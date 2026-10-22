"""Familiar regression must retain compound alpha, supplied samples and caller ownership."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import alphalens as al


def _panel():
    dates = pd.bdate_range("2025-02-03", periods=8, tz="UTC")
    universe = np.array([-0.03, 0.01, 0.04, -0.02, 0.005, 0.02, -0.01, 0.015])
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    factor = pd.Series(np.tile([1.0, 0.0], len(dates)), index=index)
    outcomes = np.column_stack([0.001 + 0.6 * universe, -0.001 + 1.4 * universe])
    data = al.utils.get_clean_factor(
        factor,
        pd.DataFrame({"1D": outcomes.ravel(), "5D": 2 * outcomes.ravel()}, index=index),
        groupby={"A": "red", "B": "blue"},
        quantiles=2,
        max_loss=0,
    )
    return data, dates, universe


def test_cleaned_factor_to_familiar_regression_compounds_each_horizons_intercept():
    data, _, _ = _panel()
    before = data.copy(deep=True)
    result = al.performance.factor_alpha_beta(data, demeaned=False)
    assert result.index.tolist() == ["Ann. alpha", "beta"]
    assert result.columns.tolist() == ["1D", "5D"]
    np.testing.assert_allclose(result.loc["beta"], 0.6, atol=1e-12)
    np.testing.assert_allclose(
        result.loc["Ann. alpha"],
        [np.expm1(np.log1p(0.001) * 252), np.expm1(np.log1p(0.002) * 50.4)],
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("representation", ["series", "frame", "no-factor"])
def test_supplied_regression_ignores_weights_preserves_unmatched_dates_and_series_name(
    representation,
):
    data, dates, universe = _panel()
    supplied = pd.Series(-0.002 - 0.3 * universe, index=dates, name="caller label").iloc[::-1]
    supplied.iloc[2] = np.nan
    supplied.loc[dates[-1] + pd.Timedelta("10D")] = 9.0
    if representation != "series":
        supplied = supplied.rename("5D").to_frame()
    if representation == "no-factor":
        data = data.drop(columns="factor")
    before, source_before = supplied.copy(deep=True), data.copy(deep=True)
    result = al.performance.factor_alpha_beta(data, returns=supplied, group_adjust=True)
    column = "1D" if representation == "series" else "5D"
    # 5D universe means are twice 1D means. Unmatched dates cannot enter this fit.
    assert result.loc["beta", column] == pytest.approx(-0.3 if column == "1D" else -0.15)
    scale = 252 if column == "1D" else 50.4
    assert result.loc["Ann. alpha", column] == pytest.approx(np.expm1(np.log1p(-0.002) * scale))
    if isinstance(supplied, pd.Series):
        pd.testing.assert_series_equal(supplied, before)
    else:
        pd.testing.assert_frame_equal(supplied, before)
    pd.testing.assert_frame_equal(data, source_before)


@pytest.mark.parametrize("demeaned", [False, True])
@pytest.mark.parametrize("group_adjust", [False, True])
@pytest.mark.parametrize("equal_weight", [False, True])
def test_actual_reference_regression_keeps_weight_options_and_same_universe_sample(
    demeaned, group_adjust, equal_weight
):
    reference = pytest.importorskip("alphalens.performance")
    profile = version("alphalens-reloaded")
    if profile not in {"0.5.0", "0.4.6"}:
        pytest.skip("Frozen and corrected reference profiles are qualified separately")
    dates = pd.bdate_range("2025-02-03", periods=8, tz="UTC")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]], names=["date", "asset"])
    factors = pd.Series(np.tile([1.0, 2.0, 4.0, 3.0], len(dates)), index=index)
    x = np.array([-0.03, 0.01, 0.04, -0.02, 0.005, 0.02, -0.01, 0.015])
    outcomes = x[:, None] * [0.5, 1.0, 1.5, 2.0] + [0.001, -0.002, 0.003, -0.001]
    data = al.utils.get_clean_factor(
        factors,
        pd.DataFrame({"1D": outcomes.ravel(), "5D": 2 * outcomes.ravel()}, index=index),
        groupby={"A": "red", "B": "red", "C": "blue", "D": "blue"},
        quantiles=2,
        max_loss=0,
    )
    options = {
        "demeaned": demeaned,
        "group_adjust": group_adjust,
        "equal_weight": equal_weight,
    }
    expected = reference.factor_alpha_beta(data, **options)
    actual = al.performance.factor_alpha_beta(data, reference_version=profile, **options)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=1e-12)


def test_explicit_regression_scales_override_legacy_durations_without_changing_native_defaults():
    data, _, _ = _panel()
    result = al.performance.factor_alpha_beta(
        data, demeaned=False, annualization_factors={"1D": 365, "5D": 73}
    )
    assert result.loc["Ann. alpha", "1D"] == pytest.approx(np.expm1(np.log1p(0.001) * 365))
    assert result.loc["Ann. alpha", "5D"] == pytest.approx(np.expm1(np.log1p(0.002) * 73))
