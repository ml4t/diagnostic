"""Tests for calibration diagnostics of probability forecasts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.binary_metrics import wilson_score_interval
from ml4t.diagnostic.metrics import (
    PREDICTION_MARKET_BIN_EDGES,
    brier_decomposition,
    brier_score,
    calibration_slope,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
    reliability_table,
)

DECILES = tuple(np.linspace(0.0, 1.0, 11))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _market_prices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Cent-quoted prices spread across [0.01, 0.99], denser near the extremes."""
    raw = rng.beta(0.6, 0.6, size=n)
    return np.clip(np.round(raw, 2), 0.01, 0.99)


def _favorite_longshot(p: np.ndarray, b: float) -> np.ndarray:
    """True win probability when prices are compressed toward 0.5 by factor b."""
    return _expit(b * _logit(p))


# ============================================================================
# Scores and Brier decomposition
# ============================================================================


def test_brier_and_log_loss_match_hand_computed_values() -> None:
    """Guards against a wrong loss formula or a sign error in the log terms."""
    f = [0.2, 0.2, 0.8, 0.8]
    y = [0, 1, 1, 1]
    assert brier_score(f, y) == pytest.approx((0.04 + 0.64 + 0.04 + 0.04) / 4)
    expected_ll = -(math.log(0.8) + math.log(0.2) + 2 * math.log(0.8)) / 4
    assert log_loss(f, y) == pytest.approx(expected_ll)


def test_log_loss_clips_certain_wrong_forecasts_to_a_finite_value() -> None:
    """A market priced at 0 that resolves YES must not make the loss infinite."""
    value = log_loss([0.0, 1.0], [1, 1])
    assert math.isfinite(value)
    assert value == pytest.approx(-math.log(1e-15) / 2, rel=1e-6)
    expected = -(math.log(0.01) + math.log(0.99)) / 2
    assert log_loss([0.0, 1.0], [1, 1], eps=0.01) == pytest.approx(expected)


def test_brier_decomposition_matches_hand_computed_components() -> None:
    """Guards against swapping reliability and resolution or a wrong base rate."""
    d = brier_decomposition([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1])
    assert d.reliability == pytest.approx((2 * 0.3**2 + 2 * 0.2**2) / 4)
    assert d.resolution == pytest.approx((2 * 0.25**2 + 2 * 0.25**2) / 4)
    assert d.uncertainty == pytest.approx(0.75 * 0.25)
    assert d.brier == pytest.approx(0.19)
    assert d.n_bins == 2
    assert d.within_bin_variance == 0.0
    assert d.within_bin_covariance == 0.0


@pytest.mark.parametrize("bins", [None, DECILES, PREDICTION_MARKET_BIN_EDGES])
def test_brier_decomposition_identity_is_exact(bins: tuple[float, ...] | None) -> None:
    """The components must add back to the Brier score for any binning.

    With continuous forecasts inside wide bins the classic three-term identity
    does not hold; dropping either within-bin term breaks this test.
    """
    rng = np.random.default_rng(7)
    f = rng.uniform(size=5_000)
    y = (rng.uniform(size=5_000) < _favorite_longshot(f, 1.4)).astype(int)
    w = rng.uniform(0.5, 2.0, size=5_000)
    for weights in (None, w):
        d = brier_decomposition(f, y, bins=bins, weights=weights)
        recomposed = (
            d.reliability
            - d.resolution
            + d.uncertainty
            + d.within_bin_variance
            - d.within_bin_covariance
        )
        assert recomposed == pytest.approx(d.brier, abs=1e-12)
        assert d.brier == pytest.approx(brier_score(f, y, weights=weights))
    if bins is not None:
        assert d.within_bin_variance > 0.0


def test_brier_reliability_separates_calibrated_from_distorted_prices() -> None:
    """Reliability must be near zero for calibrated prices and not for distorted ones."""
    rng = np.random.default_rng(11)
    p = _market_prices(200_000, rng)
    u = rng.uniform(size=p.size)
    calibrated = brier_decomposition(p, (u < p).astype(int))
    distorted = brier_decomposition(p, (u < _favorite_longshot(p, 1.5)).astype(int))
    assert calibrated.reliability < 2e-4
    assert distorted.reliability > 10 * calibrated.reliability


# ============================================================================
# Reliability table
# ============================================================================


def test_reliability_table_recovers_planted_favorite_longshot_bias() -> None:
    """Each bin's interval must cover the planted true frequency.

    Longshot bins must show observed below price and favorite bins above it,
    with the price itself outside the interval in the tails.
    """
    rng = np.random.default_rng(3)
    p = _market_prices(300_000, rng)
    q = _favorite_longshot(p, 1.5)
    y = (rng.uniform(size=p.size) < q).astype(int)

    table = reliability_table(p, y)
    edges = np.asarray(PREDICTION_MARKET_BIN_EDGES)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    true_freq = np.bincount(idx, weights=q, minlength=len(edges) - 1) / np.bincount(
        idx, minlength=len(edges) - 1
    ).clip(1)

    rows = table.filter(pl.col("n") > 0)
    assert rows["ci_method"].unique().to_list() == ["wilson"]
    covered = 0
    for row in rows.iter_rows(named=True):
        j = int(np.searchsorted(edges, row["bin_lower"]))
        covered += row["ci_lower"] <= true_freq[j] <= row["ci_upper"]
    assert covered >= len(rows) - 1

    low = rows.filter(pl.col("bin_upper") <= 0.1)
    high = rows.filter(pl.col("bin_lower") >= 0.9)
    assert (low["difference"] < 0).all()
    assert (high["difference"] > 0).all()
    assert (low["ci_upper"] < low["mean_forecast"]).all()
    assert (high["ci_lower"] > high["mean_forecast"]).all()


def test_reliability_table_bins_are_half_open_and_include_one() -> None:
    """A price on an edge belongs to the bin above it; a price of 1 to the last bin."""
    table = reliability_table([0.0, 0.01, 0.019, 0.02, 0.99, 1.0], [0, 0, 0, 1, 1, 1])
    counts = dict(zip(table["bin_lower"].to_list(), table["n"].to_list(), strict=True))
    assert counts[0.0] == 1
    assert counts[0.01] == 2
    assert counts[0.02] == 1
    assert counts[0.99] == 2
    assert table["n"].sum() == 6


def test_computed_bin_edges_assign_edge_prices_to_the_upper_bin() -> None:
    """np.linspace gives 0.30000000000000004; a price of 0.3 must still go to [0.3, 0.4)."""
    assert DECILES[3] != 0.3
    table = reliability_table([0.3, 0.7], [0, 1], bins=DECILES)
    occupied = table.filter(pl.col("n") > 0)["bin_lower"].to_list()
    assert occupied == pytest.approx([0.3, 0.7])


def test_reliability_table_empty_bins_are_null() -> None:
    """Empty bins stay in the table with null statistics, not NaN or zeros."""
    table = reliability_table([0.3, 0.35], [0, 1], bins=DECILES)
    assert table.height == 10
    empty = table.filter(pl.col("n") == 0)
    assert empty.height == 9
    assert empty["observed_frequency"].null_count() == 9
    assert empty["ci_lower"].null_count() == 9


def test_reliability_wilson_interval_matches_library_wilson() -> None:
    """The vectorized interval must agree with ``wilson_score_interval``."""
    rng = np.random.default_rng(5)
    p = rng.uniform(size=2_000)
    y = (rng.uniform(size=2_000) < p).astype(int)
    table = reliability_table(p, y, bins=DECILES, confidence=0.9)
    for row in table.iter_rows(named=True):
        k = round(row["observed_frequency"] * row["n"])
        lo, hi = wilson_score_interval(k, row["n"], confidence=0.9)
        assert row["ci_lower"] == pytest.approx(lo)
        assert row["ci_upper"] == pytest.approx(hi)


def test_cluster_bootstrap_widens_when_outcomes_are_shared_within_clusters() -> None:
    """Perfectly correlated clusters of size m carry ~1/m of the information.

    Resampling observations instead of clusters, or ignoring the cluster
    ids, gives intervals as narrow as the independent case and fails here.
    """
    rng = np.random.default_rng(21)
    n_clusters, size = 600, 9
    cluster_p = rng.uniform(0.05, 0.95, size=n_clusters)
    p = np.repeat(cluster_p, size)
    clusters = np.repeat(np.arange(n_clusters), size)
    shared = np.repeat((rng.uniform(size=n_clusters) < cluster_p).astype(int), size)
    independent = (rng.uniform(size=p.size) < p).astype(int)

    def widths(y: np.ndarray) -> np.ndarray:
        t = reliability_table(p, y, bins=DECILES, clusters=clusters, n_bootstrap=500)
        assert t["ci_method"].unique().to_list() == ["cluster_bootstrap"]
        return (t["ci_upper"] - t["ci_lower"]).to_numpy()

    ratio = np.median(widths(shared) / widths(independent))
    assert ratio > 0.7 * math.sqrt(size)


def test_cluster_bootstrap_with_singleton_clusters_matches_wilson_width() -> None:
    """With one observation per cluster the bootstrap must agree with Wilson."""
    rng = np.random.default_rng(8)
    p = rng.uniform(0.1, 0.9, size=20_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    wilson = reliability_table(p, y, bins=DECILES)
    boot = reliability_table(p, y, bins=DECILES, clusters=np.arange(p.size), n_bootstrap=400)
    ratio = (boot["ci_upper"] - boot["ci_lower"]) / (wilson["ci_upper"] - wilson["ci_lower"])
    assert ratio.drop_nulls().is_between(0.8, 1.2).all()
    assert boot["n_clusters"].to_list() == boot["n"].to_list()


def test_cluster_bootstrap_counts_clusters_per_bin_and_is_seeded() -> None:
    """``n_clusters`` counts distinct clusters per bin; the seed fixes the interval."""
    p = np.array([0.15, 0.15, 0.15, 0.55, 0.55, 0.85])
    y = np.array([0, 1, 0, 1, 1, 1])
    clusters = pl.Series(["a", "a", "b", "c", "c", "a"])
    t1 = reliability_table(p, y, bins=DECILES, clusters=clusters, n_bootstrap=200, seed=4)
    t2 = reliability_table(p, y, bins=DECILES, clusters=clusters, n_bootstrap=200, seed=4)
    by_bin = dict(zip(t1["bin_lower"].round(2).to_list(), t1["n_clusters"].to_list(), strict=True))
    assert by_bin[0.1] == 2
    assert by_bin[0.5] == 1
    assert by_bin[0.8] == 1
    assert t1.equals(t2)

    rng = np.random.default_rng(0)
    big_p = rng.uniform(size=3_000)
    big_y = (rng.uniform(size=3_000) < big_p).astype(int)
    big_c = rng.integers(0, 300, size=3_000)
    a = reliability_table(big_p, big_y, bins=DECILES, clusters=big_c, n_bootstrap=200, seed=1)
    b = reliability_table(big_p, big_y, bins=DECILES, clusters=big_c, n_bootstrap=200, seed=2)
    assert not a["ci_lower"].equals(b["ci_lower"])


def test_weights_equal_duplicated_observations() -> None:
    """An observation with weight 2 must count exactly like two copies of it."""
    rng = np.random.default_rng(9)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    w = rng.integers(1, 4, size=500)
    p_dup, y_dup = np.repeat(p, w), np.repeat(y, w)

    assert brier_score(p, y, weights=w) == pytest.approx(brier_score(p_dup, y_dup))
    assert log_loss(p, y, weights=w) == pytest.approx(log_loss(p_dup, y_dup))
    weighted = reliability_table(p, y, bins=DECILES, weights=w)
    duplicated = reliability_table(p_dup, y_dup, bins=DECILES)
    for column in ("mean_forecast", "observed_frequency", "difference"):
        assert np.allclose(weighted[column].to_numpy(), duplicated[column].to_numpy())
    assert weighted["ci_method"][0] == "wilson_effective_n"
    for spec in ("linear", "logit"):
        a = calibration_slope(p, y, spec=spec, weights=w)
        b = calibration_slope(p_dup, y_dup, spec=spec)
        assert a.slope == pytest.approx(b.slope, rel=1e-6)
        assert a.intercept == pytest.approx(b.intercept, abs=1e-6)


# ============================================================================
# Calibration error summaries
# ============================================================================


def test_ece_and_mce_match_hand_computed_values() -> None:
    """Guards against an unweighted average over bins or including empty bins."""
    f = [0.15, 0.15, 0.15, 0.15, 0.85]
    y = [0, 0, 0, 1, 0]
    assert expected_calibration_error(f, y, bins=DECILES) == pytest.approx(0.8 * 0.1 + 0.2 * 0.85)
    assert max_calibration_error(f, y, bins=DECILES) == pytest.approx(0.85)


def test_ece_is_small_when_calibrated_and_large_when_distorted() -> None:
    """ECE must detect a planted distortion that calibrated data does not show."""
    rng = np.random.default_rng(13)
    p = _market_prices(200_000, rng)
    u = rng.uniform(size=p.size)
    assert expected_calibration_error(p, (u < p).astype(int)) < 0.005
    assert expected_calibration_error(p, (u < _favorite_longshot(p, 1.5)).astype(int)) > 0.02


# ============================================================================
# Calibration slope
# ============================================================================


@pytest.mark.parametrize("spec", ["linear", "logit"])
def test_calibration_slope_is_one_for_calibrated_prices(spec: str) -> None:
    """Calibrated prices must give a slope whose interval covers 1."""
    rng = np.random.default_rng(17)
    p = _market_prices(100_000, rng)
    y = (rng.uniform(size=p.size) < p).astype(int)
    result = calibration_slope(p, y, spec=spec)
    assert result.slope_ci[0] <= 1.0 <= result.slope_ci[1]
    assert result.intercept_ci[0] <= 0.0 <= result.intercept_ci[1]
    assert result.slope_p_value > 0.05
    assert result.slope == pytest.approx(1.0, abs=0.03)


def test_logit_calibration_slope_recovers_planted_slope() -> None:
    """The logit slope must recover b in logit(q) = b * logit(p) and reject b = 1."""
    rng = np.random.default_rng(19)
    p = _market_prices(100_000, rng)
    y = (rng.uniform(size=p.size) < _favorite_longshot(p, 1.3)).astype(int)
    result = calibration_slope(p, y, spec="logit")
    assert result.slope_ci[0] <= 1.3 <= result.slope_ci[1]
    assert result.slope_ci[0] > 1.0
    assert result.slope_p_value < 1e-6


def test_linear_calibration_slope_signals_favorite_longshot_bias() -> None:
    """Overpriced longshots and underpriced favorites must give b > 1, a < 0."""
    rng = np.random.default_rng(23)
    p = _market_prices(100_000, rng)
    y = (rng.uniform(size=p.size) < _favorite_longshot(p, 1.5)).astype(int)
    result = calibration_slope(p, y, spec="linear")
    assert result.slope_ci[0] > 1.0
    assert result.intercept_ci[1] < 0.0


def test_calibration_slope_matches_statsmodels_robust_covariance() -> None:
    """The hand-written sandwich must equal statsmodels' HC1 and CR1 estimates."""
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(29)
    p = rng.uniform(0.02, 0.98, size=4_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    groups = rng.integers(0, 200, size=p.size)
    exog = sm.add_constant(p)

    hc1 = sm.OLS(y, exog).fit(cov_type="HC1")
    ours = calibration_slope(p, y)
    assert [ours.intercept_se, ours.slope_se] == pytest.approx(list(hc1.bse), rel=1e-8)

    cr1 = sm.OLS(y, exog).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ours = calibration_slope(p, y, clusters=groups)
    assert [ours.intercept_se, ours.slope_se] == pytest.approx(list(cr1.bse), rel=1e-8)
    assert ours.n_clusters == 200

    logit_exog = sm.add_constant(np.log(p / (1 - p)))
    glm = sm.GLM(y, logit_exog, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    ours = calibration_slope(p, y, spec="logit", clusters=groups)
    assert [ours.intercept, ours.slope] == pytest.approx(list(glm.params), rel=1e-6)
    assert [ours.intercept_se, ours.slope_se] == pytest.approx(list(glm.bse), rel=1e-6)


def test_cluster_robust_slope_se_widens_with_shared_outcomes() -> None:
    """Ignoring within-event correlation understates the slope standard error."""
    rng = np.random.default_rng(31)
    n_clusters, size = 2_000, 9
    cluster_p = rng.uniform(0.05, 0.95, size=n_clusters)
    p = np.repeat(cluster_p, size)
    clusters = np.repeat(np.arange(n_clusters), size)
    y = np.repeat((rng.uniform(size=n_clusters) < cluster_p).astype(int), size)
    clustered = calibration_slope(p, y, clusters=clusters)
    naive = calibration_slope(p, y)
    assert clustered.se_type == "cluster"
    assert naive.se_type == "HC1"
    assert clustered.slope_se / naive.slope_se > 0.8 * math.sqrt(size)


# ============================================================================
# Inputs
# ============================================================================


def test_accepts_polars_and_pandas_inputs() -> None:
    """Series inputs must give the same answer as NumPy arrays."""
    f = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    expected = brier_score(f, y)
    assert brier_score(pl.Series(f), pl.Series(y)) == pytest.approx(expected)
    assert brier_score(pd.Series(f), pd.Series(y.astype(bool))) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("forecasts", "outcomes", "message"),
    [
        ([0.1, np.nan], [0, 1], "NaN"),
        ([0.1, 1.2], [0, 1], r"\[0, 1\]"),
        ([0.1, 0.2], [0, 2], "0 or 1"),
        ([0.1, 0.2], [0], "same length"),
        ([], [], "empty"),
    ],
)
def test_rejects_invalid_inputs(forecasts: list[float], outcomes: list[int], message: str) -> None:
    """Invalid probabilities or outcomes must fail loudly instead of being dropped."""
    with pytest.raises(ValueError, match=message):
        brier_score(forecasts, outcomes)


@pytest.mark.parametrize("bins", [(0.0, 0.5, 0.4, 1.0), (0.1, 1.0), (0.0, 0.9), (0.5,)])
def test_rejects_invalid_bin_edges(bins: tuple[float, ...]) -> None:
    """Bins must cover [0, 1] with strictly increasing edges."""
    with pytest.raises(ValueError, match="bins"):
        reliability_table([0.2, 0.7], [0, 1], bins=bins)


def test_calibration_slope_rejects_constant_forecasts() -> None:
    """A constant forecast does not identify a slope."""
    with pytest.raises(ValueError, match="constant"):
        calibration_slope([0.5, 0.5, 0.5], [0, 1, 1])
