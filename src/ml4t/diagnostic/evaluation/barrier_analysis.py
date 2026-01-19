"""Barrier Analysis module for triple barrier outcome evaluation.

This module provides analysis of signal quality using triple barrier outcomes
(take-profit, stop-loss, timeout) instead of simple forward returns.

The BarrierAnalysis class computes:
- Hit rates by signal decile (% TP, % SL, % timeout)
- Profit factor by decile (sum TP returns / |sum SL returns|)
- Statistical tests for signal-outcome independence (chi-square)
- Monotonicity tests for signal strength vs outcome relationship

Triple barrier outcomes from ml4t.features:
- label: int (-1=SL hit, 0=timeout, 1=TP hit)
- label_return: float (actual return at exit)
- label_bars: int (bars from entry to exit)

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
    Chapter 3: Labeling (Triple Barrier Method)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats

from ml4t.diagnostic.config.barrier_config import BarrierConfig, BarrierLabel
from ml4t.diagnostic.results.barrier_results import (
    BarrierTearSheet,
    HitRateResult,
    PrecisionRecallResult,
    ProfitFactorResult,
    TimeToTargetResult,
)

if TYPE_CHECKING:
    pass


class BarrierAnalysis:
    """Analyze signal quality using triple barrier outcomes.

    This class evaluates how well a signal predicts barrier outcomes
    (take-profit hit, stop-loss hit, or timeout) rather than raw returns.

    Parameters
    ----------
    signal_data : pl.DataFrame
        DataFrame with columns: [date_col, asset_col, signal_col]
        Contains signal values for each asset-date pair.

    barrier_labels : pl.DataFrame
        DataFrame with columns: [date_col, asset_col, label_col, label_return_col, label_bars_col]
        Contains triple barrier outcomes from ml4t.features.triple_barrier_labels().

    config : BarrierConfig | None, optional
        Configuration for analysis. Uses defaults if not provided.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import BarrierAnalysis
    >>> from ml4t.diagnostic.config import BarrierConfig
    >>>
    >>> # Basic usage
    >>> analysis = BarrierAnalysis(signals_df, barriers_df)
    >>> hit_rates = analysis.compute_hit_rates()
    >>> print(hit_rates.summary())
    >>>
    >>> # With custom config
    >>> config = BarrierConfig(n_quantiles=5)
    >>> analysis = BarrierAnalysis(signals_df, barriers_df, config=config)
    >>> profit_factor = analysis.compute_profit_factor()
    """

    def __init__(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
        config: BarrierConfig | None = None,
    ) -> None:
        """Initialize BarrierAnalysis.

        Parameters
        ----------
        signal_data : pl.DataFrame
            Signal values with date, asset, signal columns.
        barrier_labels : pl.DataFrame
            Barrier outcomes with date, asset, label, label_return, label_bars columns.
        config : BarrierConfig | None
            Configuration object. Uses defaults if None.

        Raises
        ------
        ValueError
            If required columns are missing or data is invalid.
        """
        self.config = config or BarrierConfig()
        self._validate_inputs(signal_data, barrier_labels)

        # Store original data
        self._signal_data = signal_data
        self._barrier_labels = barrier_labels

        # Merge and prepare data
        self._merged_data = self._prepare_data(signal_data, barrier_labels)

        # Cache for computed results
        self._hit_rate_result: HitRateResult | None = None
        self._profit_factor_result: ProfitFactorResult | None = None
        self._precision_recall_result: PrecisionRecallResult | None = None
        self._time_to_target_result: TimeToTargetResult | None = None

    def _validate_inputs(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> None:
        """Validate input DataFrames have required columns and valid data.

        Raises
        ------
        ValueError
            If validation fails.
        """
        cfg = self.config

        # Check signal_data columns
        signal_required = {cfg.date_col, cfg.asset_col, cfg.signal_col}
        signal_cols = set(signal_data.columns)
        missing_signal = signal_required - signal_cols
        if missing_signal:
            raise ValueError(
                f"signal_data missing required columns: {missing_signal}. "
                f"Available columns: {signal_cols}"
            )

        # Check barrier_labels columns
        barrier_required = {cfg.date_col, cfg.asset_col, cfg.label_col, cfg.label_return_col}
        barrier_cols = set(barrier_labels.columns)
        missing_barrier = barrier_required - barrier_cols
        if missing_barrier:
            raise ValueError(
                f"barrier_labels missing required columns: {missing_barrier}. "
                f"Available columns: {barrier_cols}"
            )

        # Check for empty DataFrames
        if signal_data.height == 0:
            raise ValueError("signal_data is empty")
        if barrier_labels.height == 0:
            raise ValueError("barrier_labels is empty")

        # Validate label values
        valid_labels = {-1, 0, 1}
        unique_labels = set(barrier_labels[cfg.label_col].unique().to_list())
        invalid_labels = unique_labels - valid_labels
        if invalid_labels:
            raise ValueError(
                f"barrier_labels[{cfg.label_col}] contains invalid values: {invalid_labels}. "
                f"Expected values: {valid_labels} (-1=SL, 0=timeout, 1=TP)"
            )

    def _prepare_data(
        self,
        signal_data: pl.DataFrame,
        barrier_labels: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge signal data with barrier labels and prepare for analysis.

        Returns
        -------
        pl.DataFrame
            Merged DataFrame with signal values and barrier outcomes,
            plus computed quantile labels.
        """
        cfg = self.config

        # Merge on date and asset
        merged = signal_data.join(
            barrier_labels,
            on=[cfg.date_col, cfg.asset_col],
            how="inner",
        )

        if merged.height == 0:
            raise ValueError(
                "No matching rows after merging signal_data and barrier_labels. "
                "Check that date and asset columns match."
            )

        # Filter outliers if configured
        if cfg.filter_zscore is not None:
            signal_mean = merged[cfg.signal_col].mean()
            signal_std = merged[cfg.signal_col].std()
            if signal_std is not None and signal_std > 0:
                merged = merged.filter(
                    ((pl.col(cfg.signal_col) - signal_mean) / signal_std).abs() <= cfg.filter_zscore
                )

        # Drop NaN signals
        merged = merged.drop_nulls(subset=[cfg.signal_col])

        if merged.height == 0:
            raise ValueError("No valid observations after filtering NaN signals and outliers")

        # Add quantile labels
        merged = self._add_quantile_labels(merged)

        return merged

    def _add_quantile_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add quantile labels to DataFrame based on signal values.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with signal column.

        Returns
        -------
        pl.DataFrame
            DataFrame with added 'quantile' column.
        """
        cfg = self.config
        n_q = cfg.n_quantiles

        # Generate quantile labels (D1, D2, ..., D10 for deciles)
        quantile_labels = [f"D{i + 1}" for i in range(n_q)]

        if cfg.decile_method.value == "quantile":
            # Equal frequency bins (like pd.qcut)
            df = df.with_columns(
                pl.col(cfg.signal_col)
                .qcut(n_q, labels=quantile_labels, allow_duplicates=True)
                .alias("quantile")
            )
        else:
            # Equal width bins (like pd.cut)
            df = df.with_columns(
                pl.col(cfg.signal_col).cut(n_q, labels=quantile_labels).alias("quantile")
            )

        return df

    @property
    def merged_data(self) -> pl.DataFrame:
        """Get the merged and prepared data."""
        return self._merged_data

    @property
    def n_observations(self) -> int:
        """Total number of observations after merging."""
        return self._merged_data.height

    @property
    def n_assets(self) -> int:
        """Number of unique assets."""
        return self._merged_data[self.config.asset_col].n_unique()

    @property
    def n_dates(self) -> int:
        """Number of unique dates."""
        return self._merged_data[self.config.date_col].n_unique()

    @property
    def date_range(self) -> tuple[str, str]:
        """Date range (start, end) as ISO strings."""
        dates = self._merged_data[self.config.date_col]
        min_date = dates.min()
        max_date = dates.max()
        return (str(min_date), str(max_date))

    @property
    def quantile_labels(self) -> list[str]:
        """List of quantile labels used."""
        return [f"D{i + 1}" for i in range(self.config.n_quantiles)]

    def compute_hit_rates(self) -> HitRateResult:
        """Compute hit rates by signal decile.

        For each signal quantile, calculates the percentage of observations
        that hit TP, SL, or timeout barriers.

        Includes chi-square test for independence between signal strength
        and barrier outcome.

        Returns
        -------
        HitRateResult
            Results containing hit rates per quantile, chi-square test,
            and monotonicity analysis.

        Examples
        --------
        >>> result = analysis.compute_hit_rates()
        >>> print(result.summary())
        >>> df = result.get_dataframe("hit_rates")
        """
        if self._hit_rate_result is not None:
            return self._hit_rate_result

        cfg = self.config
        df = self._merged_data
        q_labels = self.quantile_labels

        # Initialize containers
        hit_rate_tp: dict[str, float] = {}
        hit_rate_sl: dict[str, float] = {}
        hit_rate_timeout: dict[str, float] = {}
        count_tp: dict[str, int] = {}
        count_sl: dict[str, int] = {}
        count_timeout: dict[str, int] = {}
        count_total: dict[str, int] = {}

        # Build contingency table for chi-square test
        # Rows: quantiles, Columns: outcomes (SL, Timeout, TP)
        contingency = np.zeros((cfg.n_quantiles, 3), dtype=np.int64)

        for i, q in enumerate(q_labels):
            q_data = df.filter(pl.col("quantile") == q)
            n_total = q_data.height

            if n_total == 0:
                # Handle empty quantile
                hit_rate_tp[q] = 0.0
                hit_rate_sl[q] = 0.0
                hit_rate_timeout[q] = 0.0
                count_tp[q] = 0
                count_sl[q] = 0
                count_timeout[q] = 0
                count_total[q] = 0
                continue

            # Count outcomes
            n_tp = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value).height
            n_sl = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.STOP_LOSS.value).height
            n_timeout = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TIMEOUT.value).height

            # Hit rates
            hit_rate_tp[q] = n_tp / n_total
            hit_rate_sl[q] = n_sl / n_total
            hit_rate_timeout[q] = n_timeout / n_total

            # Counts
            count_tp[q] = n_tp
            count_sl[q] = n_sl
            count_timeout[q] = n_timeout
            count_total[q] = n_total

            # Contingency table row
            contingency[i, 0] = n_sl
            contingency[i, 1] = n_timeout
            contingency[i, 2] = n_tp

        # Chi-square test for independence
        # H0: Signal quantile and barrier outcome are independent
        # H1: They are dependent (signal predicts outcome)

        # Remove rows/cols with all zeros to avoid chi2 issues
        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)
        valid_rows = row_sums > 0
        valid_cols = col_sums > 0

        if valid_rows.sum() < 2 or valid_cols.sum() < 2:
            # Not enough data for chi-square test
            chi2_stat = 0.0
            chi2_p = 1.0
            chi2_dof = 0
            warnings.warn(
                "Insufficient variation in data for chi-square test. "
                "Need at least 2 non-empty quantiles and 2 different outcomes.",
                UserWarning,
                stacklevel=2,
            )
        else:
            contingency_valid = contingency[valid_rows][:, valid_cols]
            chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency_valid)

        # Overall hit rates
        total_obs = df.height
        overall_tp = (
            df.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value).height / total_obs
        )
        overall_sl = (
            df.filter(pl.col(cfg.label_col) == BarrierLabel.STOP_LOSS.value).height / total_obs
        )
        overall_timeout = (
            df.filter(pl.col(cfg.label_col) == BarrierLabel.TIMEOUT.value).height / total_obs
        )

        # Monotonicity analysis for TP rate
        tp_rates = [hit_rate_tp[q] for q in q_labels]
        tp_monotonic, tp_direction, tp_spearman = self._analyze_monotonicity(tp_rates)

        self._hit_rate_result = HitRateResult(
            n_quantiles=cfg.n_quantiles,
            quantile_labels=q_labels,
            hit_rate_tp=hit_rate_tp,
            hit_rate_sl=hit_rate_sl,
            hit_rate_timeout=hit_rate_timeout,
            count_tp=count_tp,
            count_sl=count_sl,
            count_timeout=count_timeout,
            count_total=count_total,
            chi2_statistic=float(chi2_stat),
            chi2_p_value=float(chi2_p),
            chi2_dof=int(chi2_dof),
            is_significant=chi2_p < cfg.significance_level,
            significance_level=cfg.significance_level,
            overall_hit_rate_tp=overall_tp,
            overall_hit_rate_sl=overall_sl,
            overall_hit_rate_timeout=overall_timeout,
            n_observations=total_obs,
            tp_rate_monotonic=tp_monotonic,
            tp_rate_direction=tp_direction,
            tp_rate_spearman=tp_spearman,
        )

        return self._hit_rate_result

    def compute_profit_factor(self) -> ProfitFactorResult:
        """Compute profit factor by signal decile.

        Profit Factor = Sum(TP returns) / |Sum(SL returns)|

        A profit factor > 1 indicates the quantile is net profitable
        when trading based on the signal.

        Returns
        -------
        ProfitFactorResult
            Results containing profit factor per quantile and
            return statistics.

        Examples
        --------
        >>> result = analysis.compute_profit_factor()
        >>> print(result.summary())
        >>> df = result.get_dataframe()
        """
        if self._profit_factor_result is not None:
            return self._profit_factor_result

        cfg = self.config
        df = self._merged_data
        q_labels = self.quantile_labels
        eps = cfg.profit_factor_epsilon

        # Initialize containers
        profit_factor: dict[str, float] = {}
        sum_tp_returns: dict[str, float] = {}
        sum_sl_returns: dict[str, float] = {}
        sum_timeout_returns: dict[str, float] = {}
        sum_all_returns: dict[str, float] = {}
        avg_tp_return: dict[str, float] = {}
        avg_sl_return: dict[str, float] = {}
        avg_return: dict[str, float] = {}
        count_tp: dict[str, int] = {}
        count_sl: dict[str, int] = {}
        count_total: dict[str, int] = {}

        for q in q_labels:
            q_data = df.filter(pl.col("quantile") == q)
            n_total = q_data.height

            if n_total == 0:
                profit_factor[q] = 0.0
                sum_tp_returns[q] = 0.0
                sum_sl_returns[q] = 0.0
                sum_timeout_returns[q] = 0.0
                sum_all_returns[q] = 0.0
                avg_tp_return[q] = 0.0
                avg_sl_return[q] = 0.0
                avg_return[q] = 0.0
                count_tp[q] = 0
                count_sl[q] = 0
                count_total[q] = 0
                continue

            # TP returns
            tp_data = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value)
            n_tp = tp_data.height
            s_tp = tp_data[cfg.label_return_col].sum() if n_tp > 0 else 0.0

            # SL returns
            sl_data = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.STOP_LOSS.value)
            n_sl = sl_data.height
            s_sl = sl_data[cfg.label_return_col].sum() if n_sl > 0 else 0.0

            # Timeout returns
            timeout_data = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TIMEOUT.value)
            s_timeout = timeout_data[cfg.label_return_col].sum() if timeout_data.height > 0 else 0.0

            # Total returns
            s_all = q_data[cfg.label_return_col].sum()

            # Profit factor: PF = sum(TP) / |sum(SL)|
            # SL returns are typically negative, so we use abs
            denom = abs(s_sl) + eps if s_sl != 0 else eps
            pf = s_tp / denom if s_tp > 0 else 0.0

            # Store results
            profit_factor[q] = float(pf)
            sum_tp_returns[q] = float(s_tp) if s_tp is not None else 0.0
            sum_sl_returns[q] = float(s_sl) if s_sl is not None else 0.0
            sum_timeout_returns[q] = float(s_timeout) if s_timeout is not None else 0.0
            sum_all_returns[q] = float(s_all) if s_all is not None else 0.0
            avg_tp_return[q] = float(s_tp / n_tp) if n_tp > 0 and s_tp is not None else 0.0
            avg_sl_return[q] = float(s_sl / n_sl) if n_sl > 0 and s_sl is not None else 0.0
            avg_return[q] = float(s_all / n_total) if s_all is not None else 0.0
            count_tp[q] = n_tp
            count_sl[q] = n_sl
            count_total[q] = n_total

        # Overall metrics
        total_obs = df.height
        total_tp_returns = df.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value)[
            cfg.label_return_col
        ].sum()
        total_sl_returns = df.filter(pl.col(cfg.label_col) == BarrierLabel.STOP_LOSS.value)[
            cfg.label_return_col
        ].sum()

        total_tp_returns = float(total_tp_returns) if total_tp_returns is not None else 0.0
        total_sl_returns = float(total_sl_returns) if total_sl_returns is not None else 0.0

        overall_pf_denom = abs(total_sl_returns) + eps if total_sl_returns != 0 else eps
        overall_pf = total_tp_returns / overall_pf_denom if total_tp_returns > 0 else 0.0

        overall_sum = df[cfg.label_return_col].sum()
        overall_sum = float(overall_sum) if overall_sum is not None else 0.0
        overall_avg = overall_sum / total_obs

        # Monotonicity analysis for profit factor
        pf_values = [profit_factor[q] for q in q_labels]
        pf_monotonic, pf_direction, pf_spearman = self._analyze_monotonicity(pf_values)

        self._profit_factor_result = ProfitFactorResult(
            n_quantiles=cfg.n_quantiles,
            quantile_labels=q_labels,
            profit_factor=profit_factor,
            sum_tp_returns=sum_tp_returns,
            sum_sl_returns=sum_sl_returns,
            sum_timeout_returns=sum_timeout_returns,
            sum_all_returns=sum_all_returns,
            avg_tp_return=avg_tp_return,
            avg_sl_return=avg_sl_return,
            avg_return=avg_return,
            count_tp=count_tp,
            count_sl=count_sl,
            count_total=count_total,
            overall_profit_factor=overall_pf,
            overall_sum_returns=overall_sum,
            overall_avg_return=overall_avg,
            n_observations=total_obs,
            pf_monotonic=pf_monotonic,
            pf_direction=pf_direction,
            pf_spearman=pf_spearman,
        )

        return self._profit_factor_result

    def compute_precision_recall(self) -> PrecisionRecallResult:
        """Compute precision and recall metrics for barrier outcomes.

        For the top signal quantile (highest signals), computes:
        - Precision: P(TP | in quantile) = TP count / total in quantile
        - Recall: P(in quantile | TP) = TP in quantile / all TP

        Also computes cumulative metrics from the top quantile downward,
        and lift (precision relative to baseline TP rate).

        Returns
        -------
        PrecisionRecallResult
            Results containing precision, recall, F1, and lift metrics
            per quantile and cumulative from top down.

        Examples
        --------
        >>> result = analysis.compute_precision_recall()
        >>> print(result.summary())
        >>> df = result.get_dataframe("cumulative")
        """
        if self._precision_recall_result is not None:
            return self._precision_recall_result

        cfg = self.config
        df = self._merged_data
        q_labels = self.quantile_labels

        # Total TP count (baseline)
        total_tp = df.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value).height
        total_obs = df.height
        baseline_tp_rate = total_tp / total_obs if total_obs > 0 else 0.0

        # Per-quantile precision and recall
        precision_tp: dict[str, float] = {}
        recall_tp: dict[str, float] = {}
        lift_tp: dict[str, float] = {}

        # Count TP per quantile for cumulative calculations
        tp_counts: dict[str, int] = {}
        total_counts: dict[str, int] = {}

        for q in q_labels:
            q_data = df.filter(pl.col("quantile") == q)
            n_total = q_data.height
            n_tp = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value).height

            tp_counts[q] = n_tp
            total_counts[q] = n_total

            # Precision: P(TP | in this quantile)
            prec = n_tp / n_total if n_total > 0 else 0.0
            precision_tp[q] = prec

            # Recall: P(in this quantile | TP)
            rec = n_tp / total_tp if total_tp > 0 else 0.0
            recall_tp[q] = rec

            # Lift: precision / baseline
            lift = prec / baseline_tp_rate if baseline_tp_rate > 0 else 0.0
            lift_tp[q] = lift

        # Cumulative metrics (from top quantile down)
        # Reverse order: D10 is highest signal, then D9, etc.
        reversed_labels = list(reversed(q_labels))

        cumulative_precision_tp: dict[str, float] = {}
        cumulative_recall_tp: dict[str, float] = {}
        cumulative_f1_tp: dict[str, float] = {}
        cumulative_lift_tp: dict[str, float] = {}

        cum_tp = 0
        cum_total = 0

        best_f1 = 0.0
        best_f1_q = q_labels[-1]  # Default to top quantile

        for q in reversed_labels:
            cum_tp += tp_counts[q]
            cum_total += total_counts[q]

            # Cumulative precision
            cum_prec = cum_tp / cum_total if cum_total > 0 else 0.0
            cumulative_precision_tp[q] = cum_prec

            # Cumulative recall
            cum_rec = cum_tp / total_tp if total_tp > 0 else 0.0
            cumulative_recall_tp[q] = cum_rec

            # F1 score
            if cum_prec + cum_rec > 0:
                f1 = 2 * cum_prec * cum_rec / (cum_prec + cum_rec)
            else:
                f1 = 0.0
            cumulative_f1_tp[q] = f1

            # Track best F1
            if f1 > best_f1:
                best_f1 = f1
                best_f1_q = q

            # Cumulative lift
            cum_lift = cum_prec / baseline_tp_rate if baseline_tp_rate > 0 else 0.0
            cumulative_lift_tp[q] = cum_lift

        self._precision_recall_result = PrecisionRecallResult(
            n_quantiles=cfg.n_quantiles,
            quantile_labels=q_labels,
            precision_tp=precision_tp,
            recall_tp=recall_tp,
            cumulative_precision_tp=cumulative_precision_tp,
            cumulative_recall_tp=cumulative_recall_tp,
            cumulative_f1_tp=cumulative_f1_tp,
            lift_tp=lift_tp,
            cumulative_lift_tp=cumulative_lift_tp,
            baseline_tp_rate=baseline_tp_rate,
            total_tp_count=total_tp,
            n_observations=total_obs,
            best_f1_quantile=best_f1_q,
            best_f1_score=best_f1,
        )

        return self._precision_recall_result

    def compute_time_to_target(self) -> TimeToTargetResult:
        """Compute time-to-target metrics by signal decile.

        Analyzes how quickly different signal quantiles reach their barrier
        outcomes (TP, SL, or timeout). Uses the `label_bars` column from
        barrier labels to measure time to exit.

        Returns
        -------
        TimeToTargetResult
            Results containing mean, median, and std of bars to exit
            per quantile and outcome type.

        Raises
        ------
        ValueError
            If label_bars column is not available in barrier_labels.

        Examples
        --------
        >>> result = analysis.compute_time_to_target()
        >>> print(result.summary())
        >>> df = result.get_dataframe("detailed")
        """
        if self._time_to_target_result is not None:
            return self._time_to_target_result

        cfg = self.config
        df = self._merged_data
        q_labels = self.quantile_labels

        # Check if label_bars column exists
        if cfg.label_bars_col not in df.columns:
            raise ValueError(
                f"Time-to-target analysis requires '{cfg.label_bars_col}' column in barrier_labels. "
                f"Available columns: {df.columns}"
            )

        # Initialize containers
        mean_bars_tp: dict[str, float] = {}
        mean_bars_sl: dict[str, float] = {}
        mean_bars_timeout: dict[str, float] = {}
        mean_bars_all: dict[str, float] = {}
        median_bars_tp: dict[str, float] = {}
        median_bars_sl: dict[str, float] = {}
        median_bars_all: dict[str, float] = {}
        std_bars_tp: dict[str, float] = {}
        std_bars_sl: dict[str, float] = {}
        std_bars_all: dict[str, float] = {}
        count_tp: dict[str, int] = {}
        count_sl: dict[str, int] = {}
        count_timeout: dict[str, int] = {}
        tp_faster_than_sl: dict[str, bool] = {}
        speed_advantage_tp: dict[str, float] = {}

        for q in q_labels:
            q_data = df.filter(pl.col("quantile") == q)

            # TP outcomes
            tp_data = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value)
            n_tp = tp_data.height
            count_tp[q] = n_tp

            if n_tp > 0:
                tp_bars = tp_data[cfg.label_bars_col]
                mean_bars_tp[q] = float(tp_bars.mean() or 0.0)
                median_bars_tp[q] = float(tp_bars.median() or 0.0)
                std_bars_tp[q] = float(tp_bars.std() or 0.0)
            else:
                mean_bars_tp[q] = 0.0
                median_bars_tp[q] = 0.0
                std_bars_tp[q] = 0.0

            # SL outcomes
            sl_data = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.STOP_LOSS.value)
            n_sl = sl_data.height
            count_sl[q] = n_sl

            if n_sl > 0:
                sl_bars = sl_data[cfg.label_bars_col]
                mean_bars_sl[q] = float(sl_bars.mean() or 0.0)
                median_bars_sl[q] = float(sl_bars.median() or 0.0)
                std_bars_sl[q] = float(sl_bars.std() or 0.0)
            else:
                mean_bars_sl[q] = 0.0
                median_bars_sl[q] = 0.0
                std_bars_sl[q] = 0.0

            # Timeout outcomes
            timeout_data = q_data.filter(pl.col(cfg.label_col) == BarrierLabel.TIMEOUT.value)
            n_timeout = timeout_data.height
            count_timeout[q] = n_timeout

            if n_timeout > 0:
                mean_bars_timeout[q] = float(timeout_data[cfg.label_bars_col].mean() or 0.0)
            else:
                mean_bars_timeout[q] = 0.0

            # All outcomes
            n_all = q_data.height
            if n_all > 0:
                all_bars = q_data[cfg.label_bars_col]
                mean_bars_all[q] = float(all_bars.mean() or 0.0)
                median_bars_all[q] = float(all_bars.median() or 0.0)
                std_bars_all[q] = float(all_bars.std() or 0.0)
            else:
                mean_bars_all[q] = 0.0
                median_bars_all[q] = 0.0
                std_bars_all[q] = 0.0

            # Speed analysis: is TP reached faster than SL?
            if n_tp > 0 and n_sl > 0:
                tp_faster = mean_bars_tp[q] < mean_bars_sl[q]
                speed_adv = mean_bars_sl[q] - mean_bars_tp[q]
            elif n_tp > 0:
                tp_faster = True
                speed_adv = 0.0
            elif n_sl > 0:
                tp_faster = False
                speed_adv = 0.0
            else:
                tp_faster = False
                speed_adv = 0.0

            tp_faster_than_sl[q] = tp_faster
            speed_advantage_tp[q] = speed_adv

        # Overall statistics
        total_obs = df.height
        all_bars = df[cfg.label_bars_col]
        overall_mean_bars = float(all_bars.mean() or 0.0)
        overall_median_bars = float(all_bars.median() or 0.0)

        tp_all = df.filter(pl.col(cfg.label_col) == BarrierLabel.TAKE_PROFIT.value)
        overall_mean_bars_tp = (
            float(tp_all[cfg.label_bars_col].mean() or 0.0) if tp_all.height > 0 else 0.0
        )

        sl_all = df.filter(pl.col(cfg.label_col) == BarrierLabel.STOP_LOSS.value)
        overall_mean_bars_sl = (
            float(sl_all[cfg.label_bars_col].mean() or 0.0) if sl_all.height > 0 else 0.0
        )

        self._time_to_target_result = TimeToTargetResult(
            n_quantiles=cfg.n_quantiles,
            quantile_labels=q_labels,
            mean_bars_tp=mean_bars_tp,
            mean_bars_sl=mean_bars_sl,
            mean_bars_timeout=mean_bars_timeout,
            mean_bars_all=mean_bars_all,
            median_bars_tp=median_bars_tp,
            median_bars_sl=median_bars_sl,
            median_bars_all=median_bars_all,
            std_bars_tp=std_bars_tp,
            std_bars_sl=std_bars_sl,
            std_bars_all=std_bars_all,
            count_tp=count_tp,
            count_sl=count_sl,
            count_timeout=count_timeout,
            overall_mean_bars=overall_mean_bars,
            overall_median_bars=overall_median_bars,
            overall_mean_bars_tp=overall_mean_bars_tp,
            overall_mean_bars_sl=overall_mean_bars_sl,
            n_observations=total_obs,
            tp_faster_than_sl=tp_faster_than_sl,
            speed_advantage_tp=speed_advantage_tp,
        )

        return self._time_to_target_result

    def _analyze_monotonicity(
        self,
        values: list[float],
    ) -> tuple[bool, str, float]:
        """Analyze monotonicity of values across quantiles.

        Parameters
        ----------
        values : list[float]
            Values for each quantile (ordered by quantile rank).

        Returns
        -------
        tuple[bool, str, float]
            (is_monotonic, direction, spearman_correlation)
            direction is 'increasing', 'decreasing', or 'none'
        """
        if len(values) < 2:
            return False, "none", 0.0

        # Remove any NaN/inf values for correlation
        valid_values = [v for v in values if np.isfinite(v)]
        if len(valid_values) < 2:
            return False, "none", 0.0

        # Spearman correlation with rank
        ranks = list(range(len(valid_values)))
        try:
            spearman_corr, _ = stats.spearmanr(ranks, valid_values)
        except Exception:
            spearman_corr = 0.0

        spearman_corr = float(spearman_corr) if np.isfinite(spearman_corr) else 0.0

        # Check strict monotonicity
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        all_increasing = all(d >= 0 for d in diffs) and any(d > 0 for d in diffs)
        all_decreasing = all(d <= 0 for d in diffs) and any(d < 0 for d in diffs)

        if all_increasing:
            return True, "increasing", spearman_corr
        elif all_decreasing:
            return True, "decreasing", spearman_corr
        else:
            return False, "none", spearman_corr

    def create_tear_sheet(
        self,
        include_time_to_target: bool = True,
        include_figures: bool = True,
        theme: str | None = None,
    ) -> BarrierTearSheet:
        """Create comprehensive tear sheet with all analysis results.

        Parameters
        ----------
        include_time_to_target : bool, default=True
            If True, include time-to-target analysis. Requires `label_bars`
            column in barrier_labels. Set to False if column not available.
        include_figures : bool, default=True
            If True, generate Plotly figures for visualization.
            Set to False to skip figure generation (faster).
        theme : str | None
            Plot theme: 'default', 'dark', 'print', 'presentation'.
            If None, uses default theme.

        Returns
        -------
        BarrierTearSheet
            Complete results including hit rates, profit factor,
            precision/recall, time-to-target, figures, and metadata.

        Examples
        --------
        >>> tear_sheet = analysis.create_tear_sheet()
        >>> tear_sheet.save_html("barrier_analysis.html")
        >>> print(tear_sheet.summary())
        """
        # Compute all metrics
        hit_rate = self.compute_hit_rates()
        profit_factor = self.compute_profit_factor()
        precision_recall = self.compute_precision_recall()

        # Time-to-target is optional (requires label_bars column)
        time_to_target = None
        if include_time_to_target:
            try:
                time_to_target = self.compute_time_to_target()
            except ValueError:
                # label_bars column not available, skip
                pass

        # Generate figures if requested
        figures: dict[str, str] = {}
        if include_figures:
            figures = self._generate_figures(
                hit_rate=hit_rate,
                profit_factor=profit_factor,
                precision_recall=precision_recall,
                time_to_target=time_to_target,
                theme=theme,
            )

        return BarrierTearSheet(
            hit_rate_result=hit_rate,
            profit_factor_result=profit_factor,
            precision_recall_result=precision_recall,
            time_to_target_result=time_to_target,
            signal_name=self.config.signal_name,
            n_assets=self.n_assets,
            n_dates=self.n_dates,
            n_observations=self.n_observations,
            date_range=self.date_range,
            figures=figures,
        )

    def _generate_figures(
        self,
        hit_rate: HitRateResult,
        profit_factor: ProfitFactorResult,
        precision_recall: PrecisionRecallResult,
        time_to_target: TimeToTargetResult | None,
        theme: str | None = None,
    ) -> dict[str, str]:
        """Generate Plotly figures for the tear sheet.

        Parameters
        ----------
        hit_rate : HitRateResult
            Hit rate analysis results.
        profit_factor : ProfitFactorResult
            Profit factor analysis results.
        precision_recall : PrecisionRecallResult
            Precision/recall analysis results.
        time_to_target : TimeToTargetResult | None
            Time-to-target analysis results (optional).
        theme : str | None
            Plot theme.

        Returns
        -------
        dict[str, str]
            Dict mapping figure names to JSON-serialized Plotly figures.
        """
        import plotly.io as pio

        from ml4t.diagnostic.visualization.barrier_plots import (
            plot_hit_rate_heatmap,
            plot_precision_recall_curve,
            plot_profit_factor_bar,
            plot_time_to_target_box,
        )

        figures: dict[str, str] = {}

        # Hit Rate Heatmap
        try:
            fig = plot_hit_rate_heatmap(hit_rate, theme=theme)
            figures["hit_rate_heatmap"] = pio.to_json(fig)
        except Exception:
            pass  # Skip if visualization fails

        # Profit Factor Bar Chart
        try:
            fig = plot_profit_factor_bar(profit_factor, theme=theme)
            figures["profit_factor_bar"] = pio.to_json(fig)
        except Exception:
            pass

        # Precision/Recall Curve
        try:
            fig = plot_precision_recall_curve(precision_recall, theme=theme)
            figures["precision_recall_curve"] = pio.to_json(fig)
        except Exception:
            pass

        # Time-to-Target Box Plots (if available)
        if time_to_target is not None:
            try:
                fig = plot_time_to_target_box(
                    time_to_target, outcome_type="comparison", theme=theme
                )
                figures["time_to_target_comparison"] = pio.to_json(fig)
            except Exception:
                pass

        return figures
