"""Shared episode segmentation for owned portfolio and raw-return reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .results import DrawdownPeriod

if TYPE_CHECKING:
    import pandas as pd


def _identify_drawdown_periods(
    underwater: np.ndarray,
    dates: pl.Series | pd.DatetimeIndex,
    threshold: float,
    initial_timestamp: Any = None,
) -> list[DrawdownPeriod]:
    periods = []
    in_drawdown = False
    peak_idx = -1
    valley_idx = 0
    valley_depth = 0.0
    for i, dd in enumerate(underwater):
        if not in_drawdown and dd >= 0:
            peak_idx = i
        elif dd < -threshold and not in_drawdown:
            in_drawdown = True
            valley_idx = i
            valley_depth = dd
        elif in_drawdown:
            if dd < valley_depth:
                valley_idx = i
                valley_depth = dd
            elif dd >= 0:
                periods.append(
                    DrawdownPeriod(
                        peak_date=initial_timestamp if peak_idx < 0 else dates[peak_idx],
                        valley_date=dates[valley_idx],
                        recovery_date=dates[i],
                        depth=valley_depth,
                        duration_days=valley_idx - peak_idx,
                        recovery_days=i - valley_idx,
                    )
                )
                in_drawdown = False
                peak_idx = i
    if in_drawdown:
        periods.append(
            DrawdownPeriod(
                peak_date=initial_timestamp if peak_idx < 0 else dates[peak_idx],
                valley_date=dates[valley_idx],
                recovery_date=None,
                depth=valley_depth,
                duration_days=valley_idx - peak_idx,
                recovery_days=None,
            )
        )
    return periods
