"""Factor analysis visualizations.

Provides plot functions for factor exposures, return/risk attribution,
and model diagnostics. All plots follow the library theme system.
"""

from .attribution_plots import (  # noqa: F401
    plot_return_attribution_area,
    plot_return_attribution_waterfall,
)
from .diagnostic_plots import (  # noqa: F401
    plot_factor_correlation_heatmap,
    plot_residual_diagnostics,
    plot_vif_bar,
)
from .exposure_plots import (  # noqa: F401
    plot_factor_betas_bar,
    plot_rolling_betas,
)
from .risk_plots import (  # noqa: F401
    plot_risk_attribution_bar,
    plot_risk_attribution_pie,
)

__all__: list[str] = [
    # Exposure plots
    "plot_factor_betas_bar",
    "plot_rolling_betas",
    # Attribution plots
    "plot_return_attribution_waterfall",
    "plot_return_attribution_area",
    # Risk plots
    "plot_risk_attribution_pie",
    "plot_risk_attribution_bar",
    # Diagnostic plots
    "plot_residual_diagnostics",
    "plot_factor_correlation_heatmap",
    "plot_vif_bar",
]
