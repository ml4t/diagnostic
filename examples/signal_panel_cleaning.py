"""How-to: audit one common supplied-return sample without a legacy dependency."""

import json
from importlib.metadata import version

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.metrics import compute_forward_returns
from ml4t.diagnostic.signal import prepare_signal_panel


def main() -> None:
    dates = pd.bdate_range("2025-01-02", periods=6, tz="UTC")
    assets = ["A", "B", "C", "D"]
    prices = pd.DataFrame(
        {
            asset: [100.0 + 11 * i + 3 * day + ((day * (i + 2)) % 7) for day in range(6)]
            for i, asset in enumerate(assets)
        },
        index=dates,
    )
    factor = prices.pct_change(fill_method=None).stack(future_stack=True).rename("factor")
    factor.index = factor.index.set_names(["date", "asset"])
    marks = (
        prices.stack(future_stack=True)
        .rename("close")
        .rename_axis(index=["date", "asset"])
        .reset_index()
    )
    forward = compute_forward_returns(marks, periods=[1, 2], group_col="asset")
    forward = forward.set_index(["date", "asset"])[["fwd_ret_1", "fwd_ret_2"]].rename(
        columns={"fwd_ret_1": "1D", "fwd_ret_2": "2D"}
    )
    panel = prepare_signal_panel(
        factor,
        forward,
        quantiles=2,
        groups={"A": "first", "B": "first", "C": "second", "D": "second"},
        max_loss=0.5,
    )
    assert panel.data.height == 12
    assert panel.metadata["loss_by_stage"] == {
        "factor": 4,
        "returns": 8,
        "groups": 0,
        "bucketing": 0,
    }
    result = panel.analyze(min_assets=2)
    assert result.n_dates == 3 and result.n_assets == 4
    assert sum(result.count_by_quantile.values()) == 12
    grouped_ic = panel.ic_by_date(group_adjust=True, by_group=True, min_obs=2)
    assert grouped_ic.height == 12
    assert grouped_ic["n_obs"].to_list() == [2] * 12
    adjusted = panel.demean_returns(by_group=True)
    unchanged = [name for name in panel.data.columns if name not in ["1D", "2D"]]
    assert adjusted.select(unchanged).equals(panel.data.select(unchanged))
    weights = panel.factor_weights(group_adjust=True)
    gross = weights.to_pandas().groupby("date")["weight"].agg(lambda values: values.abs().sum())
    np.testing.assert_allclose(gross, [1.0] * 3, atol=1e-14, rtol=0)
    portfolio_returns = panel.factor_returns(group_adjust=True)
    one_period = portfolio_returns.to_pandas()
    one_period = one_period[one_period["return_column"] == "1D"].set_index("date")[
        "weighted_return"
    ]
    portfolio_summary = PortfolioAnalysis(one_period, periods_per_year=252).compute_summary_stats()
    assert abs(portfolio_summary.total_return - ((1 + one_period).prod() - 1)) < 1e-14
    daily_quantiles = panel.quantile_statistics(by_date=True, demeaned=False)
    equal_date_quantiles = panel.quantile_statistics(demeaned=False)
    assert daily_quantiles.height == 12 and equal_date_quantiles.height == 4
    assert equal_date_quantiles["n_obs"].to_list() == [3] * 4
    print(
        json.dumps(
            {
                "versions": {
                    name: version(name) for name in ["ml4t-diagnostic", "pandas", "numpy", "polars"]
                },
                "input_rows": panel.metadata["input_rows"],
                "retained_rows": panel.metadata["retained_rows"],
                "loss_by_stage": panel.metadata["loss_by_stage"],
                "ic": result.ic,
                "group_adjusted_ic_by_date": grouped_ic.to_dicts(),
                "group_adjusted_returns": adjusted.to_dicts(),
                "factor_weights": weights.to_dicts(),
                "factor_returns": portfolio_returns.to_dicts(),
                "arithmetic_series_total_return": portfolio_summary.total_return,
                "daily_quantiles": daily_quantiles.to_dicts(),
                "equal_date_quantiles": equal_date_quantiles.to_dicts(),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
