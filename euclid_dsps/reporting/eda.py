"""EDA reporting entry points."""

from __future__ import annotations

from .core import (
    plot_color_distributions,
    plot_flux_distributions,
    plot_physical_parameters_distributions,
    plot_redshift_distributions,
    write_eda_outputs,
)

__all__ = [
    "plot_color_distributions",
    "plot_flux_distributions",
    "plot_physical_parameters_distributions",
    "plot_redshift_distributions",
    "write_eda_outputs",
]
