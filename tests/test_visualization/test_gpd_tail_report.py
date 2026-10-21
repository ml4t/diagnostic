"""Supplied GPD fits remain inspectable in real portfolio reports, including failures."""

import re
from xml.etree import ElementTree

import numpy as np
import pytest

from ml4t.diagnostic.metrics import estimate_gpd_tail_risk
from ml4t.diagnostic.visualization.backtest import (
    BacktestTearsheet,
    export_workspaces,
    generate_backtest_tearsheet,
)


def model_table(html):
    matches = re.findall(r'<section class="gpd-tail-fit".*?</section>', html, re.DOTALL)
    assert len(matches) == 1, "The supplied fitted model was silently omitted or duplicated"
    root = ElementTree.fromstring(matches[0])
    return {row.find("th").text: row.find("td").text for row in root.findall(".//tbody/tr")}


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_actual_portfolio_report_renders_supplied_fit_counts_units_and_boundary_without_refitting(
    route,
    monkeypatch,
):
    from ml4t.diagnostic.metrics import tail_risk

    returns = np.r_[
        np.repeat(0.002, 40), -np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2])
    ]
    fitted = estimate_gpd_tail_risk(returns, threshold=0.02, tail_probability=0.02)
    original_fit = tail_risk._fit
    refits = []

    def observe_fit(*args, **kwargs):
        refits.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(tail_risk, "_fit", observe_fit)
    supplied = {"gpd_tail_risk": fitted}
    if route == "function":
        html = generate_backtest_tearsheet(
            returns=returns, metrics=supplied, periods_per_year=12, include_statistical=False
        )
    elif route == "builder":
        html = (
            BacktestTearsheet(periods_per_year=12)
            .add_returns(returns)
            .add_metrics(supplied)
            .generate()
        )
    else:
        html = export_workspaces(
            ("performance",), returns=returns, metrics=supplied, periods_per_year=12
        )
    rows = model_table(html)
    assert rows["Method"] == "peaks_over_threshold"
    assert rows["Fit status"] == "ok"
    assert rows["Finite / supplied observations"] == "48 / 48"
    assert rows["Exceedances / probability denominator"] == "8 / 48"
    assert rows["Tail probability"] == "2.00%"
    assert rows["VaR (positive periodic loss)"] == "17.84%"
    assert rows["ES (positive periodic loss)"] == "18.92%"
    assert rows["At shape lower bound"] == "Yes"
    assert supplied["gpd_tail_risk"] is fitted
    assert not refits


def test_report_keeps_a_genuinely_infinite_fitted_mean_instead_of_null_or_zero():
    losses = np.array([0.021, 0.022, 0.023, 0.024, 0.026, 0.03, 0.04, 0.08, 0.2, 0.8])
    returns = np.r_[np.repeat(0.002, 40), -losses]
    fitted = estimate_gpd_tail_risk(returns, threshold=0.02)
    assert fitted.status == "ok" and fitted.shape > 1 and np.isposinf(fitted.expected_shortfall)
    html = generate_backtest_tearsheet(
        returns=returns, metrics={"gpd_tail_risk": fitted}, include_statistical=False
    )
    assert model_table(html)["ES (positive periodic loss)"] == "Positive infinity"


def test_report_discloses_unavailable_fitted_risk_instead_of_the_reference_zero_sentinel():
    returns = np.repeat(0.01, 20)
    fitted = estimate_gpd_tail_risk(returns)
    assert fitted.status == "insufficient_exceedances"
    html = generate_backtest_tearsheet(
        returns=returns, metrics={"gpd_tail_risk": fitted}, include_statistical=False
    )
    rows = model_table(html)
    assert rows["Fit status"] == "insufficient_exceedances"
    assert (
        rows["VaR (positive periodic loss)"] == rows["ES (positive periodic loss)"] == "Unavailable"
    )


@pytest.mark.parametrize("route", ["function", "builder", "export"])
def test_serialized_model_dictionary_is_rejected_before_report_metrics_or_output_are_changed(
    route, tmp_path
):
    from dataclasses import asdict

    returns = np.r_[
        np.repeat(0.002, 40), -np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2])
    ]
    supplied = {"gpd_tail_risk": asdict(estimate_gpd_tail_risk(returns, threshold=0.02))}
    before = dict(supplied)
    output = tmp_path / "report.html"
    output.write_text("Previous report")
    with pytest.raises(TypeError, match="GPDTailRiskResult"):
        if route == "function":
            generate_backtest_tearsheet(returns=returns, metrics=supplied, output_path=output)
        elif route == "builder":
            builder = BacktestTearsheet().add_returns(returns).add_metrics(supplied)
            builder.generate(output_path=output)
        else:
            export_workspaces(
                ("performance",), returns=returns, metrics=supplied, output_path=output
            )
    assert supplied == before
    if route == "builder":
        assert builder.metrics == before
    assert output.read_text() == "Previous report"
