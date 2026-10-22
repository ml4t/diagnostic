"""The familiar interesting-times sheet must retain original event samples and options."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.compat import pyfolio as pf


def _inputs():
    dates = pd.date_range("2025-01-13", periods=5, tz="UTC")
    returns = pd.Series([0.01, np.nan, -0.02, 0.03, -0.04], index=dates)
    benchmark = pd.Series([0.005, 0.0, -0.01, 0.02, -0.03], index=dates)
    return returns, benchmark, {"observed": (dates[0], dates[3]), "absent": ("1990", "1990")}


def test_complete_familiar_stress_report_preserves_nan_membership_and_percent_statistics(tmp_path):
    returns, benchmark, periods = _inputs()
    target = tmp_path / "stress.html"
    figures = pf.tears.create_interesting_times_tear_sheet(
        returns, benchmark, periods, legend_loc="upper left", return_fig=True, output_file=target
    )
    assert len(figures) == 1
    figure = figures[0]
    assert figure.layout.meta["n_observations"] == 4
    assert figure.layout.meta["n_finite"] == 3
    assert figure.layout.meta["legend_loc"] == "upper left"
    np.testing.assert_allclose(figure.data[0].y, [0.01, 0.01, -0.0102, 0.019494])
    assert figure.layout.meta["statistics_percent"]["mean"] == pytest.approx(2 / 3)
    html = target.read_text(encoding="utf-8")
    assert "Stress Events" in html and "0.67%" in html and "absent" in html
    assert "Missing observations are not zero returns" in html


def test_familiar_empty_periods_warn_and_do_not_replace_existing_output(tmp_path):
    returns, _, _ = _inputs()
    target = tmp_path / "stress.html"
    target.write_text("existing", encoding="utf-8")
    with pytest.warns(UserWarning, match="overlap"):
        assert (
            pf.tears.create_interesting_times_tear_sheet(returns, periods={}, output_file=target)
            is None
        )
    assert target.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("location", ["best", "upper left", "lower right", 0, 3])
def test_legend_choices_are_explicit_native_display_mappings(location):
    returns, _, periods = _inputs()
    result = pf.tears.create_interesting_times_tear_sheet(
        returns, periods=periods, legend_loc=location, return_fig=True
    )
    assert result[0].layout.meta["legend_loc"] == location


def test_missing_benchmark_period_is_unavailable_not_a_fake_zero_path():
    returns, benchmark, periods = _inputs()
    benchmark.index = benchmark.index + pd.Timedelta(days=100)
    result = pf.tears.create_interesting_times_tear_sheet(
        returns, benchmark, periods, return_fig=True
    )
    assert len(result[0].data) == 1
    assert result[0].layout.meta["benchmark_status"] == "unobserved"


def test_actual_original_sheet_has_same_raw_curves_and_printed_statistics(monkeypatch):
    reference = pytest.importorskip("pyfolio.tears")
    returns, benchmark, periods = _inputs()
    captured = []
    monkeypatch.setattr(
        reference.utils, "print_table", lambda table, **kwargs: captured.append(table)
    )
    original = reference.create_interesting_times_tear_sheet(
        returns, benchmark, periods, return_fig=True
    )
    actual = pf.tears.create_interesting_times_tear_sheet(
        returns, benchmark, periods, return_fig=True
    )[0]
    for expected, trace in zip(original.axes[0].lines, actual.data, strict=True):
        np.testing.assert_allclose(expected.get_ydata(), trace.y, atol=1e-15)
    assert captured[0].loc["observed"].to_dict() == pytest.approx(
        actual.layout.meta["statistics_percent"]
    )
