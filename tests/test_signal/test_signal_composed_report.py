"""Complete signal reports must retain the shared prepared sample and real components."""

import runpy
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
from ml4t.diagnostic.signal import (
    PreparedSignalPanel,
    analyze_signal_information,
    analyze_signal_returns,
)


def _components():
    fixture = (
        Path(__file__).resolve().parents[1] / "test_compat/test_alphalens_information_sheet.py"
    )
    data = runpy.run_path(str(fixture))["_price_panel"]()
    panel = PreparedSignalPanel.from_prepared(data, return_columns=["1D", "2D"], calendar=None)
    returns = analyze_signal_returns(
        panel, annualization_factors={"1D": 252, "2D": 126}, cumulative_columns=["1D"]
    )
    information = analyze_signal_information(panel, min_obs=2)
    turnover = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=[1, 2]), panel.rank_autocorrelation(lags=[1, 2])
    )
    return panel, returns, information, turnover


@pytest.mark.parametrize("summary", [False, True])
def test_composed_signal_report_renders_every_real_component_and_original_factor_statistics(
    summary, tmp_path
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.signal import generate_signal_report, plot_signal_report

    panel, returns, information, turnover = _components()
    figures = plot_signal_report(returns, information, turnover, summary=summary)
    assert len(figures) == (2 if summary else 20)
    assert {figure.layout.meta["report_component"] for figure in figures} == (
        {"returns"} if summary else {"returns", "information", "turnover"}
    )
    target = tmp_path / "signal.html"
    generate_signal_report(returns, information, turnover, summary=summary, output_file=target)
    html = target.read_text(encoding="utf-8")
    assert html.count('class="plot-container"') == len(figures)
    frames = pd.read_html(StringIO(html), attrs={"id": "factor-statistics"})
    assert len(frames) == 1
    first = frames[0].iloc[0]
    assert first["quantile"] == 1 and first["count"] == 56
    assert first["min"] == 1 and first["max"] == 2 and first["mean"] == 1.5
    assert first["std"] == float(f"{np.sqrt(14 / 55):.6g}")
    assert first["count_percent"] == float(f"{100 / 3:.6g}")
    for identifier in (
        "return-regression",
        "quantile-means",
        "observed-spreads",
        "information-summary",
        "turnover-summary",
        "rank-summary",
    ):
        assert len(pd.read_html(StringIO(html), attrs={"id": identifier})) == 1
    assert "naive" in html and "dimensionless" in html
    assert returns.get_table("prepared_data").equals(panel.data)


def test_composed_report_refuses_information_from_another_real_factor_sample_without_replacing_output(
    tmp_path,
):
    pytest.importorskip("plotly")
    from ml4t.diagnostic.visualization.signal import generate_signal_report

    panel, returns, _, turnover = _components()
    different = PreparedSignalPanel.from_prepared(
        panel.data.with_columns(-pl.col("factor")), return_columns=["1D", "2D"], calendar=None
    )
    information = analyze_signal_information(different, min_obs=2)
    target = tmp_path / "retained.html"
    target.write_text("retained report", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared sample"):
        generate_signal_report(returns, information, turnover, output_file=target)
    assert target.read_text(encoding="utf-8") == "retained report"
