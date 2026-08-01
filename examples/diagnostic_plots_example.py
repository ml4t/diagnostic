"""Create diagnostic Plotly figures without opening a browser.

Run with the visualization extra installed:

    uv run --extra viz python examples/diagnostic_plots_example.py
"""

from __future__ import annotations

import numpy as np

from ml4t.diagnostic.evaluation.diagnostic_plots import (
    get_figure_data,
    plot_acf_pacf,
    plot_distribution,
    plot_qq,
    plot_volatility_clustering,
)


def main() -> None:
    """Build and inspect the four diagnostic figures."""
    rng = np.random.default_rng(42)
    returns = rng.standard_t(df=5, size=1_000) * 0.01

    figures = {
        "acf_pacf": plot_acf_pacf(returns, max_lags=40),
        "distribution": plot_distribution(returns, bins=40, fit_normal=True),
        "qq": plot_qq(returns),
        "volatility": plot_volatility_clustering(returns, window=20),
    }

    for name, figure in figures.items():
        data = get_figure_data(figure)
        html = figure.to_html(include_plotlyjs="cdn", full_html=False)
        if not html or data.empty:
            raise RuntimeError(f"{name} figure did not contain executable output")
        print(f"{name}: {len(figure.data)} traces, {len(data)} extracted rows")

    # In a notebook, display any figure by placing it on the final line of a cell.
    # In Streamlit, pass it to st.plotly_chart(figure, use_container_width=True).


if __name__ == "__main__":
    main()
