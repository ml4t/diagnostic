"""Tests for the look-ahead / feature causality auditor.

All extractors here are synthetic in-memory Polars functions - no external data.
The panels are built with a fixed seed and the leaky-vs-causal contrast is kept
crisp so the tests genuinely prove the gate discriminates.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic import (
    CausalityError,
    CausalityReport,
    assert_causal,
    audit_lookahead,
)
from ml4t.diagnostic.evaluation.causality import LeakEvent

SEED = 20260720


def _panel(n_per_symbol: int = 40, symbols: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    """Build a fixed-seed (symbol, timestamp, x) panel."""
    rng = np.random.default_rng(SEED)
    rows: dict[str, list] = {"symbol": [], "timestamp": [], "x": []}
    for s in symbols:
        # Drift + noise so the whole-series mean differs sharply from any prefix mean.
        walk = np.cumsum(rng.standard_normal(n_per_symbol)) + np.arange(n_per_symbol) * 0.3
        for i in range(n_per_symbol):
            rows["symbol"].append(s)
            rows["timestamp"].append(i)
            rows["x"].append(float(walk[i]))
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic extractors
# --------------------------------------------------------------------------- #
def causal_expanding(df: pl.DataFrame) -> pl.DataFrame:
    """Causal: expanding mean within symbol (only uses inputs at t' <= t)."""
    return (
        df.sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("x").cum_sum().over("symbol") / (pl.col("timestamp") + 1)).alias("feat")
        )
        .select("symbol", "timestamp", "feat")
    )


def leaky_full_mean(df: pl.DataFrame) -> pl.DataFrame:
    """Leaky: centre each value on the WHOLE-series mean (peeks at the future)."""
    return df.with_columns((pl.col("x") - pl.col("x").mean().over("symbol")).alias("feat")).select(
        "symbol", "timestamp", "feat"
    )


def leaky_last_value(df: pl.DataFrame) -> pl.DataFrame:
    """Leaky and order-sensitive: broadcast each symbol's LAST value backward.

    Permutation-based corruption (``shuffle``) catches this because reordering
    changes which value lands at the final timestamp.
    """
    return (
        df.sort(["symbol", "timestamp"])
        .with_columns(pl.col("x").last().over("symbol").alias("feat"))
        .select("symbol", "timestamp", "feat")
    )


class _JitterExtractor:
    """Nondeterministic-but-causal: expanding mean + fresh per-call jitter."""

    def __init__(self, scale: float = 1e-5) -> None:
        self._rng = np.random.default_rng(SEED)
        self._scale = scale

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.sort(["symbol", "timestamp"]).with_columns(
            (pl.col("x").cum_sum().over("symbol") / (pl.col("timestamp") + 1)).alias("feat")
        )
        jitter = self._rng.normal(0.0, self._scale, len(out))
        return out.with_columns((pl.col("feat") + pl.Series(jitter)).alias("feat")).select(
            "symbol", "timestamp", "feat"
        )


# --------------------------------------------------------------------------- #
# Core discrimination
# --------------------------------------------------------------------------- #
def test_causal_extractor_passes() -> None:
    report = audit_lookahead(causal_expanding, _panel())
    assert isinstance(report, CausalityReport)
    assert report.is_causal is True
    assert report.leaking_columns == {}
    assert report.determinism["is_deterministic"] is True
    assert report.determinism["noise_floor"] == 0.0
    assert report.feature_cols == ("feat",)


def test_leaky_extractor_is_flagged() -> None:
    report = audit_lookahead(leaky_full_mean, _panel())
    assert report.is_causal is False
    assert "feat" in report.leaking_columns
    info = report.leaking_columns["feat"]
    assert info["max_abs_delta"] > 0.0
    assert info["corruption"] in {"nan", "noise"}
    # Per-column, per-timestamp evidence is populated.
    assert report.leak_events
    assert all(isinstance(ev, LeakEvent) for ev in report.leak_events)
    assert info["first_leak"] is not None


def test_assert_causal_raises_with_report_attached() -> None:
    # Passing case returns the report.
    ok = assert_causal(causal_expanding, _panel())
    assert ok.is_causal is True

    # Failing case raises with the report attached.
    with pytest.raises(CausalityError) as exc:
        assert_causal(leaky_full_mean, _panel())
    assert isinstance(exc.value.report, CausalityReport)
    assert exc.value.report.is_causal is False
    assert "feat" in exc.value.report.leaking_columns


# --------------------------------------------------------------------------- #
# Determinism handling
# --------------------------------------------------------------------------- #
def test_nondeterministic_but_causal_is_not_flagged() -> None:
    # Robust across seeds: jitter must never masquerade as a leak.
    for seed in range(8):
        report = audit_lookahead(_JitterExtractor(), _panel(), seed=seed)
        assert report.is_causal is True, report.leaking_columns
        assert report.determinism["is_deterministic"] is False
        assert report.determinism["noise_floor"] > 0.0


def test_nondeterministic_leak_still_caught_above_floor() -> None:
    """A real leak on top of jitter must exceed the measured noise floor."""

    class _JitterLeak:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(SEED)

        def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
            out = df.with_columns((pl.col("x") - pl.col("x").mean().over("symbol")).alias("feat"))
            jitter = self._rng.normal(0.0, 1e-5, len(out))
            return out.with_columns((pl.col("feat") + pl.Series(jitter)).alias("feat")).select(
                "symbol", "timestamp", "feat"
            )

    report = audit_lookahead(_JitterLeak(), _panel(), seed=1)
    assert report.determinism["is_deterministic"] is False
    assert report.is_causal is False
    assert report.leaking_columns["feat"]["max_abs_delta"] > report.determinism["noise_floor"]


# --------------------------------------------------------------------------- #
# Corruption strategies + auto cutoffs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("corruption", ["nan", "shuffle", "noise"])
def test_each_corruption_leaves_causal_extractor_causal(corruption: str) -> None:
    report = audit_lookahead(causal_expanding, _panel(), corruptions=(corruption,))
    assert report.is_causal is True


@pytest.mark.parametrize(
    ("corruption", "extractor"),
    [
        ("nan", leaky_full_mean),
        ("noise", leaky_full_mean),
        ("shuffle", leaky_last_value),
    ],
)
def test_each_corruption_catches_a_matching_leak(corruption, extractor) -> None:
    report = audit_lookahead(extractor, _panel(), corruptions=(corruption,))
    assert report.is_causal is False
    assert report.corruptions == (corruption,)
    assert all(ev.corruption == corruption for ev in report.leak_events)


def test_auto_cutoffs_picks_inner_quantiles() -> None:
    report = audit_lookahead(causal_expanding, _panel(), cutoffs="auto")
    # Inner quantiles (0.4, 0.6, 0.8) of timestamps 0..39 -> round(q*39).
    assert report.cutoffs == (16, 23, 31)


def test_explicit_cutoffs_are_used() -> None:
    report = audit_lookahead(leaky_full_mean, _panel(), cutoffs=[10, 20, 30])
    assert report.cutoffs == (10, 20, 30)
    assert report.is_causal is False


# --------------------------------------------------------------------------- #
# Single-series (no symbol) support
# --------------------------------------------------------------------------- #
def test_single_series_keys() -> None:
    rng = np.random.default_rng(SEED)
    df = pl.DataFrame(
        {
            "timestamp": list(range(50)),
            "x": [float(v) for v in rng.standard_normal(50)],
        }
    )

    def leaky(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns((pl.col("x") - pl.col("x").mean()).alias("feat")).select(
            "timestamp", "feat"
        )

    report = audit_lookahead(leaky, df, keys=("timestamp",))
    assert report.is_causal is False


# --------------------------------------------------------------------------- #
# Reporting bridge
# --------------------------------------------------------------------------- #
def test_report_renders_all_formats() -> None:
    report = audit_lookahead(leaky_full_mean, _panel())

    html = report.to_html()
    assert isinstance(html, str) and "<html" in html.lower()

    md = report.to_markdown()
    assert isinstance(md, str) and "Lookahead Causality" in md

    import json

    payload = json.loads(report.to_json())
    assert payload["is_causal"] is False
    assert "feat" in payload["leaking_columns"]


def test_summary_reflects_verdict() -> None:
    assert "CAUSAL" in audit_lookahead(causal_expanding, _panel()).summary()
    assert "LEAK DETECTED" in audit_lookahead(leaky_full_mean, _panel()).summary()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_missing_keys_raise() -> None:
    df = pl.DataFrame({"timestamp": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing key columns"):
        audit_lookahead(causal_expanding, df)


def test_invalid_corruption_raises() -> None:
    with pytest.raises(ValueError, match="unknown corruption"):
        audit_lookahead(causal_expanding, _panel(), corruptions=("teleport",))
