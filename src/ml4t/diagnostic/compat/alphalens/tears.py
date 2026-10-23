"""Familiar signal sheets using the owned native analysis and report kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ml4t.specs import UNSET_CALENDAR

from ml4t.diagnostic.compat.alphalens.utils import (
    _Keys,
    _profile,
    get_forward_returns_columns,
    timedelta_strings_to_integers,
)
from ml4t.diagnostic.signal import PreparedSignalPanel, analyze_signal_information

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ml4t.diagnostic.signal import SignalReturnsResult

__all__ = [
    "create_information_tear_sheet",
    "create_returns_tear_sheet",
    "create_summary_tear_sheet",
    "create_full_tear_sheet",
    "create_turnover_tear_sheet",
    "create_event_returns_tear_sheet",
    "create_event_study_tear_sheet",
]


def _flags(**values: bool) -> None:
    for name, value in values.items():
        if not isinstance(value, bool | np.bool_):
            raise TypeError(f"{name} must be boolean")


def _prepared(factor_data: pd.DataFrame) -> tuple[PreparedSignalPanel, _Keys]:
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    keys = _Keys(factor_data["factor"])
    metadata = factor_data.attrs.get("diagnostic", {})
    return (
        PreparedSignalPanel.from_prepared(
            keys.encode(factor_data),
            return_columns=list(get_forward_returns_columns(factor_data.columns)),
            feed_spec=metadata.get("feed_spec"),
            calendar=metadata.get("calendar_argument", UNSET_CALENDAR),
        ),
        keys,
    )


def _reference_grid(factor_data: pd.DataFrame, profile: str) -> pd.DataFrame | None:
    observed = factor_data.index.get_level_values("date")
    frequency = factor_data.index.levels[0].freq
    if frequency is None and profile == "0.4.6":
        frequency = pd.offsets.Day()
    return (
        pd.DataFrame({"date": pd.date_range(observed.min(), observed.max(), freq=frequency)})
        if frequency is not None
        else None
    )


def _values(returns: pd.DataFrame, keys: _Keys) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a wide dated pandas DataFrame")
    return returns.loc[:, returns.columns.isin(keys.assets)].rename(columns=keys.codes)


def _display_report(report: str, output_file: str | Path | None) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        return
    if hasattr(get_ipython(), "kernel"):
        html = Path(report).read_text(encoding="utf-8") if output_file is not None else report
        display(HTML(html))


def _returns_analysis(
    factor_data: pd.DataFrame,
    profile: str,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
    annualization_factors: Mapping[str, float] | None,
) -> tuple[PreparedSignalPanel, SignalReturnsResult, pd.DataFrame | None]:
    from ml4t.diagnostic.signal import analyze_signal_returns

    panel, _ = _prepared(factor_data)
    columns = panel.metadata["return_columns"]
    durations = {column: pd.Timedelta(column).value for column in columns}
    if any(value <= 0 for value in durations.values()):
        raise ValueError("return-sheet horizons must be positive durations")
    factors = (
        {column: pd.Timedelta("252D").value / duration for column, duration in durations.items()}
        if annualization_factors is None
        else annualization_factors
    )
    grid = _reference_grid(factor_data, profile)
    result = analyze_signal_returns(
        panel,
        annualization_factors=factors,
        demeaned=bool(long_short),
        group_adjust=bool(group_neutral),
        by_group=bool(by_group),
        rate_conversion_factors={
            column: durations[columns[0]] / duration for column, duration in durations.items()
        },
        date_grid=grid,
        cumulative_columns=["1D"] if "1D" in columns else [],
        spread_exposure="unit_gross" if profile == "0.5.0" else "raw_difference",
    )
    return panel, result, grid


def _composed_sheet(
    factor_data: pd.DataFrame,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
    *,
    summary: bool,
    reference_version: str,
    annualization_factors: Mapping[str, float] | None,
    output_file: str | Path | None,
    return_fig: bool,
    theme: str | None,
    set_context: bool,
) -> list[go.Figure] | None:
    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
    from ml4t.diagnostic.visualization.signal import generate_signal_report, plot_signal_report

    profile = _profile(reference_version)
    _flags(
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        return_fig=return_fig,
        set_context=set_context,
    )
    panel, returns, grid = _returns_analysis(
        factor_data, profile, long_short, group_neutral, by_group, annualization_factors
    )
    information = analyze_signal_information(
        panel,
        group_adjust=False if summary else bool(group_neutral),
        by_group=False if summary else bool(by_group),
        date_grid=grid,
    )
    periods = get_forward_returns_columns(factor_data.columns, require_exact_day_multiple=True)
    lags = timedelta_strings_to_integers(periods)
    turnover = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=lags, date_grid=grid),
        panel.rank_autocorrelation(lags=lags, date_grid=grid),
    )
    options = {
        "summary": summary,
        "theme": theme,
        "rolling_window": 22,
        "bandwidth": 0.5,
        "ylim_percentiles": (1, 99),
    }
    if output_file is not None or not return_fig:
        report = generate_signal_report(
            returns, information, turnover, output_file=output_file, **options
        )
    if return_fig:
        return plot_signal_report(returns, information, turnover, **options)
    _display_report(report, output_file)
    return None


def create_summary_tear_sheet(
    factor_data: pd.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    *,
    reference_version: str = "0.5.0",
    annualization_factors: Mapping[str, float] | None = None,
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Compose familiar summary tables and bars, retaining unadjusted summary IC.

    As in both references, group_neutral adjusts returns but not this summary's
    information table. Use the full or information sheet for group-adjusted IC.
    """
    return _composed_sheet(
        factor_data,
        long_short,
        group_neutral,
        False,
        summary=True,
        reference_version=reference_version,
        annualization_factors=annualization_factors,
        output_file=output_file,
        return_fig=return_fig,
        theme=theme,
        set_context=set_context,
    )


def create_full_tear_sheet(
    factor_data: pd.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    *,
    reference_version: str = "0.5.0",
    annualization_factors: Mapping[str, float] | None = None,
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Compose familiar returns, information and persistence with actual factor stats.

    group_neutral adjusts both returns and IC here. Optional group means
    supplement the global analysis; event-return/event-study sheets remain
    separate routes, as in Alphalens. No execution account or data is inferred.
    """
    return _composed_sheet(
        factor_data,
        long_short,
        group_neutral,
        by_group,
        summary=False,
        reference_version=reference_version,
        annualization_factors=annualization_factors,
        output_file=output_file,
        return_fig=return_fig,
        theme=theme,
        set_context=set_context,
    )


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
    profile = _profile(reference_version)
    _flags(
        group_neutral=group_neutral,
        by_group=by_group,
        return_fig=return_fig,
        set_context=set_context,
    )
    panel, _ = _prepared(factor_data)
    date_grid = _reference_grid(factor_data, profile)
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
    _display_report(report, output_file)
    return None


def create_returns_tear_sheet(
    factor_data: pd.DataFrame,
    long_short: bool = True,
    group_neutral: bool = False,
    by_group: bool = False,
    *,
    reference_version: str = "0.5.0",
    annualization_factors: Mapping[str, float] | None = None,
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Render familiar returns options through the owned native return analysis.

    Legacy horizon durations establish the explicit 252-day alpha scales and
    first-horizon rate powers here, not in native defaults. Corrected0.5 uses
    unit-gross spread/errors; frozen0.4.6 retains raw gross-two differences.
    Only the supplied1D horizon creates familiar cumulative panels. Native
    callers can request arbitrary supplied-horizon mathematical wealth paths.
    """
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_returns_report,
        plot_signal_returns,
    )

    profile = _profile(reference_version)
    _flags(
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        return_fig=return_fig,
        set_context=set_context,
    )
    _, result, _ = _returns_analysis(
        factor_data, profile, long_short, group_neutral, by_group, annualization_factors
    )
    options = {"theme": theme, "rolling_window": 22, "bandwidth": 0.5, "ylim_percentiles": (1, 99)}
    if output_file is not None or not return_fig:
        report = generate_signal_returns_report(result, output_file=output_file, **options)
    if return_fig:
        return plot_signal_returns(result, **options)
    _display_report(report, output_file)
    return None


def create_turnover_tear_sheet(
    factor_data: pd.DataFrame,
    turnover_periods: Sequence[str] | None = None,
    *,
    reference_version: str = "0.5.0",
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Render familiar membership/rank panels using owned native dated observations.

    Whole-day duration labels select observation lags, as in Alphalens. They
    do not validate an exchange schedule. The frozen profile pads an undeclared
    frequency to civil days; the corrected profile retains observed dates.
    """
    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_turnover_report,
        plot_signal_turnover,
    )

    profile = _profile(reference_version)
    _flags(return_fig=return_fig, set_context=set_context)
    panel, _ = _prepared(factor_data)
    periods = (
        get_forward_returns_columns(factor_data.columns, require_exact_day_multiple=True)
        if turnover_periods is None
        else turnover_periods
    )
    lags = timedelta_strings_to_integers(periods)
    grid = _reference_grid(factor_data, profile)
    result = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=lags, date_grid=grid),
        panel.rank_autocorrelation(lags=lags, date_grid=grid),
    )
    if output_file is not None or not return_fig:
        report = generate_signal_turnover_report(result, output_file=output_file, theme=theme)
    if return_fig:
        return plot_signal_turnover(result, theme=theme)
    _display_report(report, output_file)
    return None


def create_event_returns_tear_sheet(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    avgretplot: tuple[int, int] = (5, 15),
    long_short: bool = True,
    group_neutral: bool = False,
    std_bar: bool = True,
    by_group: bool = False,
    *,
    reference_version: str = "0.5.0",
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Adapt the legacy supplied-level event sheet without rebasing event zero.

    Despite the historical argument name, Alphalens passes these values to
    common_start_returns(cumulative=True). Preserve supplied levels here; use
    the native panel.event_returns(input_kind='returns') for simple returns.
    Standard-deviation panels and group overviews reuse the native renderer.
    """
    from ml4t.diagnostic.visualization.signal import (
        generate_event_returns_report,
        plot_event_returns,
    )

    _profile(reference_version)
    _flags(
        long_short=long_short,
        group_neutral=group_neutral,
        std_bar=std_bar,
        by_group=by_group,
        return_fig=return_fig,
        set_context=set_context,
    )
    panel, keys = _prepared(factor_data)
    before, after = avgretplot
    result = panel.event_returns(
        _values(returns, keys),
        input_kind="levels",
        before=before,
        after=after,
        demeaned=bool(long_short),
        group_adjust=bool(group_neutral),
        by_group=bool(by_group),
    )
    if output_file is not None or not return_fig:
        report = generate_event_returns_report(
            result, output_file=output_file, std_bar=bool(std_bar), theme=theme
        )
    if return_fig:
        return plot_event_returns(result, std_bar=bool(std_bar), theme=theme)
    _display_report(report, output_file)
    return None


def create_event_study_tear_sheet(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame | None,
    avgretplot: tuple[int, int] | None = (5, 15),
    rate_of_ret: bool = True,
    n_bars: int = 50,
    *,
    reference_version: str = "0.5.0",
    output_file: str | Path | None = None,
    return_fig: bool = False,
    theme: str | None = None,
    set_context: bool = True,
) -> list[go.Figure] | None:
    """Compose event distribution, quantile observations and optional level paths.

    Familiar duration labels explicitly determine first-horizon display-rate
    powers. Native arbitrary labels continue to require declared conversion
    factors. No path is invented when returns or avgretplot is None.
    """
    from ml4t.diagnostic.signal import analyze_signal_event_study
    from ml4t.diagnostic.visualization.signal import (
        generate_signal_event_study_report,
        plot_signal_event_study,
    )

    _profile(reference_version)
    _flags(rate_of_ret=rate_of_ret, return_fig=return_fig, set_context=set_context)
    panel, keys = _prepared(factor_data)
    columns = panel.metadata["return_columns"]
    powers = None
    if rate_of_ret:
        durations = pd.to_timedelta(columns)
        if durations.isna().any() or (durations <= pd.Timedelta(0)).any():
            raise ValueError("rate conversion requires positive duration labels")
        nanoseconds = durations.as_unit("ns").asi8
        powers = {
            column: float(nanoseconds[0]) / float(duration)
            for column, duration in zip(columns, nanoseconds, strict=True)
        }
    result = analyze_signal_event_study(
        panel,
        _values(returns, keys) if returns is not None and avgretplot is not None else None,
        input_kind="levels",
        event_window=avgretplot,
        rate_of_return=bool(rate_of_ret),
        rate_conversion_factors=powers,
        num_bars=n_bars,
    )
    if output_file is not None or not return_fig:
        report = generate_signal_event_study_report(result, output_file=output_file, theme=theme)
    if return_fig:
        return plot_signal_event_study(result, theme=theme)
    _display_report(report, output_file)
    return None
