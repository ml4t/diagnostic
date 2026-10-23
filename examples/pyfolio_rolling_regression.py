"""Continue a joint rolling-factor workflow without a Pyfolio installation."""

import json

import numpy as np
import pandas as pd

from ml4t.diagnostic.compat.pyfolio.timeseries import rolling_regression


def main() -> None:
    dates = pd.bdate_range("2025-01-02", periods=11, tz="UTC")
    market = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.04, 0.02, -0.03, 0.01, 0, 0.02])
    style = np.array([0.02, 0.01, -0.01, 0.03, 0.02, -0.02, 0.01, 0.04, -0.03, 0.02, 0.01])
    factors = pd.DataFrame({"market": market, "style": style}, index=dates)
    returns = pd.Series(0.001 + 1.5 * market - 0.75 * style, index=dates)
    returns.iloc[2] = np.nan
    factors.iloc[4, 0] = np.nan
    fits = rolling_regression(returns, factors, rolling_window=3, nan_threshold=0.25)
    print(
        json.dumps(
            {
                "fitted_windows": int(fits.notna().all(axis=1).sum()),
                "last_coefficients": fits.iloc[-1].to_dict(),
                "alpha_units": "periodic decimal return",
                "beta_units": "dimensionless",
                "window": "four surviving return observations with inclusive endpoints",
            }
        )
    )


if __name__ == "__main__":
    main()
