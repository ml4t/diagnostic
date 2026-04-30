"""Tests for cross-sectional AUC and IC/AUC uncertainty helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from ml4t.diagnostic.metrics import (
    compute_auc_uncertainty,
    compute_ic_uncertainty,
    cross_sectional_auc_series,
    cross_sectional_ic_series,
)

# ---------------------------------------------------------------------------
# cross_sectional_auc_series
# ---------------------------------------------------------------------------


def test_auc_series_matches_sklearn_per_date():
    rng = np.random.default_rng(0)
    n_dates = 25
    n_per_date = 40
    dates: list[int] = []
    symbols: list[str] = []
    preds: list[float] = []
    labels_list: list[int] = []
    for d in range(n_dates):
        scores = rng.normal(size=n_per_date)
        labels_arr = (scores + rng.normal(scale=2.0, size=n_per_date) > 0).astype(int)
        for i in range(n_per_date):
            dates.append(d)
            symbols.append(f"S{i:03d}")
            preds.append(float(scores[i]))
            labels_list.append(int(labels_arr[i]))
    df = pl.DataFrame({"date": dates, "symbol": symbols, "prediction": preds, "label": labels_list})

    out = cross_sectional_auc_series(df, df, entity_col="symbol", min_obs=5)
    assert out.columns[:4] == ["date", "auc", "n_obs", "n_pos"]

    aucs_polars = out.drop_nulls("auc").sort("date")["auc"].to_numpy()
    aucs_sklearn = []
    for d in df.partition_by("date", maintain_order=True):
        s = d["prediction"].to_numpy()
        y = d["label"].to_numpy()
        if y.min() == y.max():
            continue
        aucs_sklearn.append(roc_auc_score(y, s))
    aucs_sklearn = np.array(aucs_sklearn)

    assert aucs_polars.shape == aucs_sklearn.shape
    np.testing.assert_allclose(aucs_polars, aucs_sklearn, atol=1e-9)


def test_auc_series_handles_degenerate_dates():
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4 + ["2024-01-03"] * 12,
            "symbol": [f"S{i}" for i in list(range(4)) * 2 + list(range(12))],
            "prediction": [0.1, 0.2, 0.3, 0.4]  # all-positive labels day
            + [0.1, 0.2, 0.3, 0.4]  # all-negative labels day
            + list(np.linspace(0, 1, 12)),
            "label": [1, 1, 1, 1] + [0, 0, 0, 0] + [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        }
    )
    out = cross_sectional_auc_series(df, df, entity_col="symbol", min_obs=4)
    aucs = dict(zip(out["date"].to_list(), out["auc"].to_list()))
    assert aucs["2024-01-01"] is None
    assert aucs["2024-01-02"] is None
    assert aucs["2024-01-03"] is not None and 0.0 <= aucs["2024-01-03"] <= 1.0


# ---------------------------------------------------------------------------
# compute_ic_uncertainty
# ---------------------------------------------------------------------------


def test_ic_uncertainty_keys_and_shapes():
    rng = np.random.default_rng(1)
    daily = pl.DataFrame({"date": np.arange(400), "ic": rng.normal(0.02, 0.06, size=400)})
    out = compute_ic_uncertainty(daily, horizon=5, n_boot=200, seed=7)

    expected = {
        "mean_ic",
        "std_ic",
        "n_days",
        "pct_positive",
        "se_naive",
        "ci_naive_lower",
        "ci_naive_upper",
        "se_hac",
        "ci_hac_lower",
        "ci_hac_upper",
        "t_hac",
        "p_hac",
        "hac_lag",
        "ci_boot_lower",
        "ci_boot_upper",
        "boot_block_size",
        "n_boot",
    }
    assert expected.issubset(out.keys())
    assert out["n_days"] == 400
    assert out["hac_lag"] >= 4  # horizon - 1 = 4
    assert out["n_boot"] == 200
    assert out["ci_naive_lower"] < out["mean_ic"] < out["ci_naive_upper"]
    assert out["ci_hac_lower"] < out["mean_ic"] < out["ci_hac_upper"]
    assert out["ci_boot_lower"] < out["mean_ic"] < out["ci_boot_upper"]


def test_ic_uncertainty_short_series_returns_nan():
    out = compute_ic_uncertainty(pl.DataFrame({"ic": [0.1, 0.2]}), horizon=1, n_boot=50)
    assert out["n_days"] == 2
    assert np.isnan(out["mean_ic"])
    assert np.isnan(out["se_hac"])


def test_ic_uncertainty_hac_widens_with_positive_autocorr():
    """Positively autocorrelated IC series → HAC SE > naive SE."""
    rng = np.random.default_rng(42)
    n = 600
    eps = rng.normal(scale=0.05, size=n)
    rho = 0.6
    ic = np.zeros(n)
    for t in range(1, n):
        ic[t] = rho * ic[t - 1] + eps[t]
    ic = ic + 0.02
    out = compute_ic_uncertainty(pl.DataFrame({"ic": ic}), horizon=5, n_boot=300, seed=11)
    assert out["se_hac"] > out["se_naive"]
    assert out["ci_hac_upper"] - out["ci_hac_lower"] > out["ci_naive_upper"] - out["ci_naive_lower"]


def test_ic_uncertainty_accepts_series_array_dataframe():
    rng = np.random.default_rng(3)
    arr = rng.normal(0.01, 0.05, size=300)
    a = compute_ic_uncertainty(arr, horizon=1, n_boot=100, seed=0)
    b = compute_ic_uncertainty(pl.Series(arr), horizon=1, n_boot=100, seed=0)
    c = compute_ic_uncertainty(pl.DataFrame({"ic": arr}), horizon=1, n_boot=100, seed=0)
    assert a["mean_ic"] == pytest.approx(b["mean_ic"])
    assert a["mean_ic"] == pytest.approx(c["mean_ic"])
    assert a["se_hac"] == pytest.approx(b["se_hac"])
    assert a["se_hac"] == pytest.approx(c["se_hac"])


# ---------------------------------------------------------------------------
# compute_auc_uncertainty
# ---------------------------------------------------------------------------


def test_auc_uncertainty_centers_on_null_value():
    rng = np.random.default_rng(5)
    daily = pl.DataFrame({"auc": 0.5 + rng.normal(0.04, 0.05, size=300)})
    out = compute_auc_uncertainty(daily, horizon=5, n_boot=200, seed=2)
    assert "mean_auc" in out
    assert out["mean_auc"] == pytest.approx(np.mean(daily["auc"].to_numpy()), rel=1e-6)
    # H0 here is AUC = 0.5; mean is 0.54-ish so t-stat should be positive
    assert out["t_hac"] > 0


def test_auc_uncertainty_chance_level_ci_contains_null():
    """Chance-level AUC: HAC CI should straddle 0.5 most of the time."""
    contains_null = 0
    n_trials = 25
    for s in range(n_trials):
        rng = np.random.default_rng(100 + s)
        daily = pl.DataFrame({"auc": 0.5 + rng.normal(0, 0.03, size=300)})
        out = compute_auc_uncertainty(daily, horizon=1, n_boot=100, seed=s)
        if out["ci_hac_lower"] <= 0.5 <= out["ci_hac_upper"]:
            contains_null += 1
    # 95% CI should cover the null in the vast majority of trials
    assert contains_null >= 20


# ---------------------------------------------------------------------------
# end-to-end: ic_series → uncertainty
# ---------------------------------------------------------------------------


def test_ic_series_to_uncertainty_pipeline():
    rng = np.random.default_rng(13)
    n_dates = 200
    n_per = 50
    dates: list[int] = []
    symbols: list[str] = []
    preds: list[float] = []
    rets: list[float] = []
    for d in range(n_dates):
        eps = rng.normal(scale=1.0, size=n_per)
        ret = rng.normal(scale=1.0, size=n_per)
        pred = 0.15 * ret + eps
        for i in range(n_per):
            dates.append(d)
            symbols.append(f"S{i}")
            preds.append(float(pred[i]))
            rets.append(float(ret[i]))
    df = pl.DataFrame(
        {"date": dates, "symbol": symbols, "prediction": preds, "forward_return": rets}
    )
    series = cross_sectional_ic_series(df, df, entity_col="symbol", min_obs=5)
    unc = compute_ic_uncertainty(series, horizon=1, n_boot=300, seed=21)

    assert unc["mean_ic"] > 0  # weak positive signal
    assert unc["ci_hac_lower"] > -0.5
    assert unc["ci_hac_upper"] < 0.5
    assert unc["n_days"] >= 100
