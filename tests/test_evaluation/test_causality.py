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
    LeakEvent,
    assert_causal,
    audit_lookahead,
)
from ml4t.diagnostic.evaluation.causality import _abs_delta_frame, _corrupt

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


def test_single_timestamp_cannot_pass_vacuously() -> None:
    frame = _panel(n_per_symbol=1)

    with pytest.raises(ValueError, match="at least two distinct timestamps"):
        audit_lookahead(causal_expanding, frame)


@pytest.mark.parametrize("cutoff", [-1, 39, 40])
def test_explicit_cutoff_must_split_observed_timestamps(cutoff: int) -> None:
    with pytest.raises(ValueError, match="must have at least one row on each side"):
        audit_lookahead(causal_expanding, _panel(), cutoffs=[cutoff])


def test_single_future_row_shuffle_uses_value_destroying_fallback() -> None:
    report = audit_lookahead(
        causal_expanding,
        _panel(n_per_symbol=3),
        cutoffs=[1],
        corruptions=("shuffle",),
    )
    assert report.is_causal is True


def test_short_panel_with_auto_cutoffs_does_not_fail_on_shuffle() -> None:
    report = audit_lookahead(causal_expanding, _panel(n_per_symbol=8))
    assert report.is_causal is True


@pytest.mark.parametrize("corruption", ["shuffle", "noise"])
def test_corruptions_preserve_unsorted_input_order(corruption: str) -> None:
    frame = _panel(n_per_symbol=8).reverse()

    def row_position_feature(data: pl.DataFrame) -> pl.DataFrame:
        return data.select("symbol", "timestamp").with_columns(
            pl.arange(0, len(data), eager=True).cast(pl.Float64).alias("feat")
        )

    report = audit_lookahead(
        row_position_feature,
        frame,
        cutoffs=[3],
        corruptions=(corruption,),
    )
    assert report.is_causal is True


def test_keyless_output_aligns_to_corrupted_input_order() -> None:
    frame = _panel(n_per_symbol=8).reverse()

    def keyless_identity(data: pl.DataFrame) -> pl.DataFrame:
        return data.select(pl.col("x").alias("feat"))

    report = audit_lookahead(
        keyless_identity,
        frame,
        cutoffs=[3],
        corruptions=("shuffle",),
    )
    assert report.is_causal is True


def test_keyless_output_still_detects_a_leak() -> None:
    frame = _panel(n_per_symbol=8).reverse()

    def keyless_full_mean(data: pl.DataFrame) -> pl.DataFrame:
        return data.select((pl.col("x") - pl.col("x").mean().over("symbol")).alias("feat"))

    report = audit_lookahead(
        keyless_full_mean,
        frame,
        cutoffs=[3],
        corruptions=("nan",),
    )
    assert report.is_causal is False


@pytest.mark.parametrize("corruption", ["nan", "shuffle", "noise"])
def test_corruption_preserves_pre_cutoff_values_bit_for_bit(corruption: str) -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "timestamp": [0, 1, 2, 3],
            "large_int": [2**53 + 1, 2**53 + 3, 2**53 + 5, 2**53 + 7],
        }
    )
    corrupted = _corrupt(
        frame,
        cutoff=1,
        corruption=corruption,
        input_cols=("large_int",),
        group_cols=("symbol",),
        time_col="timestamp",
        rng=np.random.default_rng(SEED),
    )

    expected = frame.filter(pl.col("timestamp") <= 1)
    actual = corrupted.filter(pl.col("timestamp") <= 1)
    assert actual.equals(expected)


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
    assert payload["n_effective_probes"] == 9
    assert payload["n_skipped_probes"] == 0
    assert payload["skipped_probes"] == []
    assert payload["uncovered_pairs"] == []
    assert "Effective probes" in report.summary()


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


def test_duplicate_input_keys_raise() -> None:
    frame = pl.concat([_panel(n_per_symbol=4), _panel(n_per_symbol=4).head(1)])

    with pytest.raises(ValueError, match="frame.*duplicate keys"):
        audit_lookahead(causal_expanding, frame)


def test_duplicate_output_keys_raise() -> None:
    def duplicate_first_row(data: pl.DataFrame) -> pl.DataFrame:
        output = causal_expanding(data)
        return pl.concat([output, output.head(1)])

    with pytest.raises(ValueError, match="extractor output.*duplicate keys"):
        audit_lookahead(duplicate_first_row, _panel())


def test_atol_is_floor_for_nondeterministic_threshold() -> None:
    class TinyDrift:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
            initial_offsets = (0.0, 1e-12)
            offset = initial_offsets[self.calls] if self.calls < 2 else 1e-10
            self.calls += 1
            return data.select(
                "symbol", "timestamp", pl.lit(offset, dtype=pl.Float64).alias("feat")
            )

    report = audit_lookahead(
        TinyDrift(),
        _panel(n_per_symbol=4),
        cutoffs=[1],
        corruptions=("nan",),
        atol=1e-9,
    )
    assert report.is_causal is True
    assert report.determinism["noise_floor"] == pytest.approx(1e-12)


def test_missing_pre_cutoff_output_rows_are_a_leak() -> None:
    def drops_everything_if_future_is_null(data: pl.DataFrame) -> pl.DataFrame:
        if data.get_column("x").null_count():
            return pl.DataFrame(
                schema={"symbol": pl.String, "timestamp": pl.Int64, "feat": pl.Float64}
            )
        return data.select("symbol", "timestamp", pl.col("x").alias("feat"))

    report = audit_lookahead(
        drops_everything_if_future_is_null,
        _panel(n_per_symbol=4),
        cutoffs=[1],
        corruptions=("nan",),
    )
    assert report.is_causal is False
    assert report.leaking_columns["feat"]["max_abs_delta"] == float("inf")


def test_matching_nan_features_remain_causal() -> None:
    def stable_nan(data: pl.DataFrame) -> pl.DataFrame:
        return data.select(
            "symbol",
            "timestamp",
            pl.when(pl.col("timestamp") == 0)
            .then(float("nan"))
            .otherwise(pl.col("x"))
            .alias("feat"),
        )

    report = audit_lookahead(
        stable_nan,
        _panel(n_per_symbol=4),
        cutoffs=[1],
        corruptions=("nan",),
    )
    assert report.is_causal is True


def test_new_nan_in_pre_cutoff_features_is_a_leak() -> None:
    def future_null_poisoning(data: pl.DataFrame) -> pl.DataFrame:
        if data.get_column("x").null_count():
            feature = pl.lit(float("nan"), dtype=pl.Float64)
        else:
            feature = pl.col("x")
        return data.select("symbol", "timestamp", feature.alias("feat"))

    report = audit_lookahead(
        future_null_poisoning,
        _panel(n_per_symbol=4),
        cutoffs=[1],
        corruptions=("nan",),
    )
    assert report.is_causal is False
    assert report.leaking_columns["feat"]["max_abs_delta"] == float("inf")


def test_noise_preserves_integer_input_dtype() -> None:
    frame = _panel(n_per_symbol=8).select(
        "symbol", "timestamp", pl.col("timestamp").alias("volume")
    )

    def dtype_sensitive(data: pl.DataFrame) -> pl.DataFrame:
        value = 1.0 if data.schema["volume"] == pl.Int64 else 2.0
        return data.select("symbol", "timestamp").with_columns(pl.lit(value).alias("feat"))

    report = audit_lookahead(
        dtype_sensitive,
        frame,
        cutoffs=[3],
        corruptions=("noise",),
    )
    assert report.is_causal is True


def test_noise_corrupts_non_numeric_future_inputs() -> None:
    frame = _panel(n_per_symbol=8).with_columns(
        pl.when(pl.col("timestamp") == 7)
        .then(pl.lit("future"))
        .otherwise(pl.lit("past"))
        .alias("sector")
    )

    def categorical_last(data: pl.DataFrame) -> pl.DataFrame:
        return data.select(
            "symbol", "timestamp", pl.col("sector").last().over("symbol").alias("feat")
        )

    report = audit_lookahead(
        categorical_last,
        frame,
        cutoffs=[3],
        corruptions=("noise",),
    )
    assert report.is_causal is False
    assert "feat" in report.leaking_columns


def test_non_finite_determinism_floor_raises() -> None:
    class UnstableRows:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
            self.calls += 1
            output = data.select("symbol", "timestamp", pl.col("x").alias("feat"))
            return output if self.calls == 1 else output.head(len(output) - 1)

    with pytest.raises(ValueError, match="non-finite determinism noise floor.*feat"):
        audit_lookahead(UnstableRows(), _panel(n_per_symbol=4))


def test_empty_extractor_output_cannot_pass_vacuously() -> None:
    def empty_output(data: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(schema={"symbol": pl.String, "timestamp": pl.Int64, "feat": pl.Float64})

    with pytest.raises(ValueError, match="extractor output is empty"):
        audit_lookahead(empty_output, _panel(n_per_symbol=4))


def test_output_without_pre_cutoff_comparisons_cannot_pass() -> None:
    def future_only(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(pl.col("timestamp") > 2).select(
            "symbol", "timestamp", pl.col("x").alias("feat")
        )

    with pytest.raises(ValueError, match="no pre-cutoff comparisons.*feat"):
        audit_lookahead(
            future_only,
            _panel(n_per_symbol=4),
            cutoffs=[1],
            corruptions=("nan",),
        )


def test_detected_leak_is_reported_with_uncovered_cutoff_pair() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["A"] * 5,
            "timestamp": [0, 1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0, None],
        }
    )

    def late_leaky_feature(data: pl.DataFrame) -> pl.DataFrame:
        return (
            data.with_columns((pl.col("x") - pl.col("x").mean()).alias("feat"))
            .filter(pl.col("timestamp") >= 3)
            .select("symbol", "timestamp", "feat")
        )

    report = audit_lookahead(
        late_leaky_feature,
        frame,
        cutoffs=[1, 3],
        corruptions=("noise",),
    )
    assert report.is_causal is False
    assert "feat" in report.leaking_columns
    assert report.uncovered_pairs == ((1, "feat"),)

    with pytest.raises(CausalityError) as exc:
        assert_causal(
            late_leaky_feature,
            frame,
            cutoffs=[1, 3],
            corruptions=("noise",),
        )
    assert exc.value.report.uncovered_pairs == ((1, "feat"),)


def test_noop_probe_is_skipped_when_another_probe_is_effective() -> None:
    frame = _panel(n_per_symbol=4).with_columns(pl.lit(1.0).alias("x"))
    report = audit_lookahead(
        causal_expanding,
        frame,
        cutoffs=[1],
        corruptions=("shuffle", "nan"),
    )
    assert report.is_causal is True
    assert report.n_effective_probes == 1
    assert report.n_skipped_probes == 1
    assert report.skipped_probes == ((1, "shuffle"),)


def test_each_cutoff_requires_an_effective_probe_for_causal_verdict() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["A"] * 4,
            "timestamp": [0, 1, 2, 3],
            "x": [1.0, 2.0, 3.0, None],
        }
    )

    def identity(data: pl.DataFrame) -> pl.DataFrame:
        return data.select("symbol", "timestamp", pl.col("x").alias("feat"))

    with pytest.raises(ValueError, match="no effective corruption probes at cutoffs.*2"):
        audit_lookahead(
            identity,
            frame,
            cutoffs=[1, 2],
            corruptions=("nan",),
        )


def test_all_noop_probes_raise() -> None:
    frame = _panel(n_per_symbol=4).with_columns(pl.lit(1.0).alias("x"))
    with pytest.raises(ValueError, match="all requested corruption probes were ineffective"):
        audit_lookahead(
            causal_expanding,
            frame,
            cutoffs=[1],
            corruptions=("shuffle",),
        )


def test_feature_columns_cannot_overlap_keys() -> None:
    with pytest.raises(ValueError, match="feature columns must not overlap key columns"):
        audit_lookahead(causal_expanding, _panel(), feature_cols=("timestamp",))


def test_unsigned_feature_delta_does_not_wrap() -> None:
    base = pl.DataFrame({"timestamp": [0], "feat": pl.Series([5], dtype=pl.UInt64)})
    other = pl.DataFrame({"timestamp": [0], "feat": pl.Series([7], dtype=pl.UInt64)})
    delta = _abs_delta_frame(base, other, "feat", ("timestamp",), "timestamp")
    assert delta.get_column("__delta").item() == 2.0
