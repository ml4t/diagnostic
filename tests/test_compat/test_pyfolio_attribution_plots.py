"""Public holdings workflows retain costs, component growth and exposure identities."""

import numpy as np
import pandas as pd
import pytest
from test_pyfolio_holdings_attribution import account_inputs

from ml4t.diagnostic.compat.pyfolio import perf_attrib as pf

pytest.importorskip("plotly", reason="Attribution figures require the optional viz extra")


@pytest.mark.parametrize("with_cost", [False, True])
def test_return_plot_subtracts_periodic_cost_once_before_total_compounding(with_cost):
    _, table = pf.perf_attrib(*account_inputs())
    original = table.copy(deep=True)
    cost = pd.Series([0.001, 0.002, 0.003], index=table.index) if with_cost else None
    figure = pf.plot_returns(table, cost=cost)
    first, second, third = (0.029, -0.012, 0.017) if with_cost else (0.03, -0.01, 0.02)
    np.testing.assert_allclose(
        figure.data[0].y,
        [
            first,
            (1 + first) * (1 + second) - 1,
            (1 + first) * (1 + second) * (1 + third) - 1,
        ],
        atol=1e-15,
    )
    np.testing.assert_allclose(figure.data[1].y, [0.01, 1.01 * 1.005 - 1, 1.01 * 1.005**2 - 1])
    np.testing.assert_allclose(figure.data[2].y, [0.02, 1.02 * 0.985 - 1, 1.02 * 0.985 * 1.015 - 1])
    assert len(figure.data) == (4 if with_cost else 3)
    if with_cost:
        np.testing.assert_allclose(
            figure.data[3].y, [-0.001, -(1.001 * 1.002 - 1), -(1.001 * 1.002 * 1.003 - 1)]
        )
    assert pd.DatetimeIndex(figure.data[0].x).equals(table.index)
    pd.testing.assert_frame_equal(table, original)


def test_factor_plot_retains_specific_return_and_omits_only_the_declared_aggregate_columns():
    _, table = pf.perf_attrib(*account_inputs())
    figure = pf.plot_factor_contribution_to_perf(table, title="Observed component paths")
    assert [trace.name for trace in figure.data] == ["market", "style", "specific_returns"]
    np.testing.assert_allclose(
        figure.data[0].y, [0.015, 1.015 * 0.98 - 1, 1.015 * 0.98 * 1.0075 - 1]
    )
    np.testing.assert_allclose(figure.data[1].y, [0.005, 1.005**2 - 1, 1.005**2 * 1.0075 - 1])
    assert figure.layout.title.text == "Observed component paths"


def test_exposure_plot_preserves_supplied_factor_order_values_and_observation_keys():
    exposures, _ = pf.perf_attrib(*account_inputs())
    exposures = exposures[["style", "market"]]
    figure = pf.plot_risk_exposures(exposures)
    assert [trace.name for trace in figure.data] == ["style", "market"]
    np.testing.assert_allclose(figure.data[0].y, [0.25, 0.5, 0.25])
    np.testing.assert_allclose(figure.data[1].y, [1.5, 1.0, 1.5])
    assert pd.DatetimeIndex(figure.data[0].x).equals(exposures.index)


def test_specific_return_histogram_renders_actual_ten_bin_counts_and_reports_missingness():
    _, table = pf.perf_attrib(*account_inputs())
    values = pd.concat([table["specific_returns"], pd.Series([np.nan, np.inf])], ignore_index=True)
    original = values.copy(deep=True)
    figure = pf.plot_alpha_returns(values)
    assert figure.data[0].type == "bar"
    np.testing.assert_array_equal(figure.data[0].y, [2, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(figure.data[0].width, [0.0005] * 10)
    assert figure.layout.meta["excluded_nonfinite"] == 2
    assert figure.layout.meta["n_observations"] == 5
    assert figure.layout.shapes[0].x0 == 0
    assert figure.layout.shapes[1].x0 == pytest.approx(1 / 150)
    pd.testing.assert_series_equal(values, original)


@pytest.mark.parametrize(
    "name",
    [
        "plot_returns",
        "plot_alpha_returns",
        "plot_factor_contribution_to_perf",
        "plot_risk_exposures",
    ],
)
def test_actual_reference_plot_payloads_reconcile_known_legend_errors_without_replacing_math(name):
    reference = pytest.importorskip("pyfolio.perf_attrib")
    plt = pytest.importorskip("matplotlib.pyplot")
    exposures, table = pf.perf_attrib(*account_inputs())
    values = (
        exposures
        if name == "plot_risk_exposures"
        else (table["specific_returns"] if name == "plot_alpha_returns" else table)
    )
    actual = getattr(pf, name)(values)
    figure, axis = plt.subplots()
    try:
        try:
            getattr(reference, name)(values.copy(deep=True), ax=axis)
        except AttributeError as error:
            assert name == "plot_alpha_returns" and "get_ydata" in str(error)
        except ValueError as error:
            assert name in ("plot_factor_contribution_to_perf", "plot_risk_exposures")
            assert "not enough values to unpack" in str(error)
        if name == "plot_alpha_returns":
            np.testing.assert_array_equal(
                actual.data[0].y, [bar.get_height() for bar in axis.patches]
            )
            np.testing.assert_allclose(
                actual.data[0].width, [bar.get_width() for bar in axis.patches]
            )
        else:
            for trace, line in zip(actual.data, axis.lines, strict=False):
                np.testing.assert_allclose(trace.y, line.get_ydata(), atol=1e-15)
                assert pd.DatetimeIndex(trace.x).equals(pd.DatetimeIndex(line.get_xdata()))
            assert len(actual.data) == len(axis.lines) - (
                name == "plot_factor_contribution_to_perf"
            )
    finally:
        plt.close(figure)
