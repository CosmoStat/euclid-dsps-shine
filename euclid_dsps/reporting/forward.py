"""Forward-model reporting entry points."""

from __future__ import annotations

from .core import (
    ordered_bands,
    plot_batch_dashboard,
    plot_batch_observed_vs_model,
    plot_batch_parameter_truth,
    plot_batch_redshift_truth,
    plot_batch_residuals_by_band,
    plot_observed_model_scatter,
    plot_photometry_comparison,
    plot_redshift_scatter,
    plot_residual_boxplot,
    plot_sed,
    plot_sed_comparison,
    summarize_by_band,
    summarize_by_row,
    write_batch_outputs,
    write_run_outputs,
    write_sed_comparison_outputs,
)

__all__ = [
    "ordered_bands",
    "plot_batch_dashboard",
    "plot_batch_observed_vs_model",
    "plot_batch_parameter_truth",
    "plot_batch_redshift_truth",
    "plot_batch_residuals_by_band",
    "plot_observed_model_scatter",
    "plot_photometry_comparison",
    "plot_redshift_scatter",
    "plot_residual_boxplot",
    "plot_sed",
    "plot_sed_comparison",
    "summarize_by_band",
    "summarize_by_row",
    "write_batch_outputs",
    "write_run_outputs",
    "write_sed_comparison_outputs",
]
