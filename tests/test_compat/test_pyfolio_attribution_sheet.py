"""Complete attribution workflows retain partitions, statistics and offline output."""

import numpy as np
import pytest
from test_pyfolio_holdings_attribution import account_inputs

pytest.importorskip("plotly", reason="Attribution reports require the optional viz extra")


@pytest.mark.parametrize(
    "partitions,expected",
    [
        (None, [["market", "style", "specific_returns"], ["market", "style"]]),
        ({"selected": ["style", "market", "absent"]}, [["market", "style"], ["market", "style"]]),
        (
            {"first": ["style"], "second": ["market"]},
            [["style"], ["market"], ["style"], ["market"]],
        ),
        ({}, []),
    ],
)
def test_complete_sheet_preserves_original_partition_membership_and_input_factor_order(
    partitions, expected
):
    from ml4t.diagnostic.compat.pyfolio import tears

    figures = tears.create_perf_attrib_tear_sheet(
        *account_inputs(), factor_partitions=partitions, return_fig=True, periods_per_year=12
    )
    assert len(figures) == 1 + len(expected)
    assert [trace.name for trace in figures[0].data] == [
        "Total returns",
        "Cumulative specific returns",
        "Cumulative common returns",
    ]
    for figure, names in zip(figures[1:], expected, strict=True):
        assert [trace.name for trace in figure.data] == names
    np.testing.assert_allclose(figures[0].data[0].y, [0.03, 0.0197, 0.040094], atol=1e-15)


def test_default_sheet_emits_complete_statistics_and_all_canonical_partition_panels_offline(
    tmp_path,
):
    from ml4t.diagnostic.compat.pyfolio import tears

    output = tmp_path / "attribution.html"
    figures = tears.create_perf_attrib_tear_sheet(
        *account_inputs(), return_fig=True, output_file=output
    )
    assert len(figures) == 5
    html = output.read_text()
    assert "Annualized Specific Return" in html and "Average Risk Factor Exposure" in html
    assert "252 observations per year" in html
    assert "Cumulative common style returns attribution" in html
    assert "Daily sector factor exposures" in html
    assert '<script src="https://cdn.plot.ly/' not in html
    assert not list(tmp_path.glob("*.tmp"))


def test_html_return_is_explicit_and_invalid_partitions_cannot_replace_existing_report(tmp_path):
    from ml4t.diagnostic.compat.pyfolio import tears

    html = tears.create_perf_attrib_tear_sheet(
        *account_inputs(), factor_partitions=None, periods_per_year=12
    )
    assert html.startswith("<!DOCTYPE html>")
    assert "12 observations per year" in html
    path = tmp_path / "previous.html"
    path.write_text("verified previous report")
    with pytest.raises(TypeError, match="factor_partitions"):
        tears.create_perf_attrib_tear_sheet(
            *account_inputs(), factor_partitions={"style": "market"}, output_file=path
        )
    assert path.read_text() == "verified previous report"
