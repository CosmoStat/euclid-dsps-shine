"""MAP and population fitting report entry points."""

from __future__ import annotations

from .core import (
    paired_fit_truth_frames,
    parameter_truth_metrics,
    plot_corner_overlay,
    plot_fit_trace,
    plot_map_population_chi2,
    plot_map_population_parameters,
    plot_population_parameter_histograms,
    plot_trace_truth_metrics,
    trace_truth_summary,
    workflow_fit_comparison,
    write_fit_diagnostic_outputs,
    write_fit_outputs,
    write_population_corner_outputs,
    write_trace_truth_outputs,
)

__all__ = [
    "paired_fit_truth_frames",
    "parameter_truth_metrics",
    "plot_corner_overlay",
    "plot_fit_trace",
    "plot_map_population_chi2",
    "plot_map_population_parameters",
    "plot_population_parameter_histograms",
    "plot_trace_truth_metrics",
    "trace_truth_summary",
    "workflow_fit_comparison",
    "write_fit_diagnostic_outputs",
    "write_fit_outputs",
    "write_population_corner_outputs",
    "write_trace_truth_outputs",
]
