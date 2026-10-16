"""How-to: audit one common supplied-return sample without a legacy dependency."""

import json
from importlib.metadata import version

import pandas as pd

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
    panel = prepare_signal_panel(factor, forward, quantiles=2, max_loss=0.5)
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
