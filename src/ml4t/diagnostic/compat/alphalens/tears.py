"""Familiar signal sheets using the owned native analysis and report kernels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ml4t.specs import UNSET_CALENDAR

from ml4t.diagnostic.compat.alphalens.utils import _Keys, _profile, get_forward_returns_columns
from ml4t.diagnostic.signal import PreparedSignalPanel, analyze_signal_information

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["create_information_tear_sheet"]


def create_information_tear_sheet(
    factor_data: pd.DataFrame,
    group_neutral: bool = False,
    by_group: bool = False,
    *,
    reference_version: str = "0.5.0",
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Analyze a complete clean factor table without recomputing its buckets.

    Original group options select the native outcome demeaning and extra group
    panels. Both reference profiles share this IC definition. The renderer uses
    Plotly/HTML, not Matplotlib or a global seaborn context; set_context is an
    accepted presentation flag only. Default notebook calls display HTML and
    return None. The explicit return_fig extension returns native Plotly figures;
    output_file saves the complete native report atomically in either case.
    """
    _profile(reference_version)
    for name, value in (
        ("group_neutral", group_neutral),
        ("by_group", by_group),
        ("return_fig", return_fig),
        ("set_context", set_context),
    ):
        if not isinstance(value, bool | np.bool_):
            raise TypeError(f"{name} must be boolean")
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    keys = _Keys(factor_data["factor"])
    metadata = factor_data.attrs.get("diagnostic", {})
    panel = PreparedSignalPanel.from_prepared(
        keys.encode(factor_data),
        return_columns=list(get_forward_returns_columns(factor_data.columns)),
        feed_spec=metadata.get("feed_spec"),
        calendar=metadata.get("calendar_argument", UNSET_CALENDAR),
    )
    observed = factor_data.index.get_level_values("date")
    frequency = factor_data.index.levels[0].freq
    date_grid = (
        pd.DataFrame({"date": pd.date_range(observed.min(), observed.max(), freq=frequency)})
        if frequency is not None
        else None
    )
    result = analyze_signal_information(
        panel, group_adjust=bool(group_neutral), by_group=bool(by_group), date_grid=date_grid
    )
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_information_report,
        plot_signal_information,
    )

    if output_file is not None or not return_fig:
        report = generate_signal_information_report(result, output_file=output_file, theme=theme)
    if return_fig:
        return plot_signal_information(result, theme=theme)
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        return None
    if hasattr(get_ipython(), "kernel"):
        html = Path(report).read_text(encoding="utf-8") if output_file is not None else report
        display(HTML(html))
    return None
