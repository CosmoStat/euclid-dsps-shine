"""Truth-free architecture selection for population distribution projections."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

CORE_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)

TRAINED_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "realnvp_wide",
        "label": "RealNVP 16x384",
        "prior": {
            "type": "realnvp",
            "source": "joint_realnvp",
            "checkpoint": None,
            "train_jointly": True,
            "n_layers": 16,
            "hidden_size": 384,
            "permutation": "roll",
            "scale_clamp": 0.35,
            "shift_clamp": 4.0,
            "init": "identity",
            "init_scale": 0.0,
        },
    },
    {
        "name": "rq_spline_15d",
        "label": "RQ-spline 12x256",
        "prior": {
            "type": "rq_spline_coupling",
            "source": "rq_spline_coupling",
            "checkpoint": None,
            "train_jointly": True,
            "n_layers": 12,
            "hidden_size": 256,
            "n_bins": 16,
            "tail_bound": 12.0,
            "min_bin_width": 1.0e-3,
            "min_bin_height": 1.0e-3,
            "min_derivative": 1.0e-3,
            "permutation": "roll",
            "init": "identity",
            "init_scale": 0.0,
        },
    },
    {
        "name": "structured_rq_spline",
        "label": "Structured RQ-spline 5+10D",
        "prior": {
            "type": "structured_rq_spline",
            "source": "structured_rq_spline",
            "checkpoint": None,
            "train_jointly": True,
            "core_dim": 5,
            "core_layers": 10,
            "conditional_layers": 10,
            "hidden_size": 256,
            "n_bins": 16,
            "tail_bound": 12.0,
            "min_bin_width": 1.0e-3,
            "min_bin_height": 1.0e-3,
            "min_derivative": 1.0e-3,
            "permutation": "roll",
            "init": "identity",
            "init_scale": 0.0,
        },
    },
)

BASELINE_NAME = "source_realnvp"
MAXIMUM_VALIDATION_NLL_REGRESSION = 0.5

TRUTH_FREE_TOLERANCES = {
    "selected_redshift_cdf_supremum": 0.05,
    "selected_parent_redshift_cdf_supremum": 0.07,
    "parent_redshift_cdf_supremum": 0.07,
    "selected_core_cdf_supremum": 0.10,
    "selected_parent_core_cdf_supremum": 0.10,
    "parent_core_cdf_supremum": 0.10,
}


def config_for_candidate(
    base: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return a resolved config with only the population-prior architecture changed."""
    config = copy.deepcopy(base)
    config["amortized"]["prior"] = copy.deepcopy(candidate["prior"])
    return config


def summarize_truth_free_metrics(
    metrics: pd.DataFrame,
    *,
    parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build the predeclared architecture-selection score from validation CDFs."""
    expected = {
        "selected_flow_vs_q_aggregate",
        "parent_flow_vs_inverse_beta_q",
        "selected_parent_flow_vs_q_aggregate",
    }
    if set(metrics["comparison"].unique()) != expected:
        raise ValueError("truth-free projection comparisons are incomplete")
    if tuple(parameter_names[:5]) != CORE_PARAMETER_NAMES:
        raise ValueError(
            "population benchmark requires the canonical physical 5D prefix"
        )
    if set(metrics["parameter"].unique()) != set(parameter_names):
        raise ValueError("truth-free projection parameters are incomplete")

    by_comparison: dict[str, Any] = {}
    for comparison, group in metrics.groupby("comparison"):
        redshift = float(
            group.loc[group["parameter"].eq("z_obs"), "cdf_supremum"].iloc[0]
        )
        core = group.loc[group["parameter"].isin(CORE_PARAMETER_NAMES)]
        by_comparison[str(comparison)] = {
            "redshift_cdf_supremum": redshift,
            "maximum_core_5d_cdf_supremum": float(core["cdf_supremum"].max()),
            "mean_core_5d_cdf_supremum": float(core["cdf_supremum"].mean()),
            "mean_15d_cdf_supremum": float(group["cdf_supremum"].mean()),
        }

    selected = by_comparison["selected_flow_vs_q_aggregate"]
    selected_parent = by_comparison["selected_parent_flow_vs_q_aggregate"]
    parent = by_comparison["parent_flow_vs_inverse_beta_q"]
    ratios = {
        "selected_redshift": selected["redshift_cdf_supremum"]
        / TRUTH_FREE_TOLERANCES["selected_redshift_cdf_supremum"],
        "selected_parent_redshift": selected_parent["redshift_cdf_supremum"]
        / TRUTH_FREE_TOLERANCES["selected_parent_redshift_cdf_supremum"],
        "parent_redshift": parent["redshift_cdf_supremum"]
        / TRUTH_FREE_TOLERANCES["parent_redshift_cdf_supremum"],
        "selected_core": selected["maximum_core_5d_cdf_supremum"]
        / TRUTH_FREE_TOLERANCES["selected_core_cdf_supremum"],
        "selected_parent_core": selected_parent["maximum_core_5d_cdf_supremum"]
        / TRUTH_FREE_TOLERANCES["selected_parent_core_cdf_supremum"],
        "parent_core": parent["maximum_core_5d_cdf_supremum"]
        / TRUTH_FREE_TOLERANCES["parent_core_cdf_supremum"],
    }
    mean_core = sum(
        item["mean_core_5d_cdf_supremum"] for item in by_comparison.values()
    ) / len(by_comparison)
    return {
        "comparisons": by_comparison,
        "normalized_gate_ratios": ratios,
        "primary_score": float(max(ratios.values())),
        "secondary_mean_core_5d_cdf_supremum": float(mean_core),
        "passes_all_truth_free_distribution_gates": bool(max(ratios.values()) <= 1.0),
        "redshift_median_gate_used": False,
        "sfh_used_for_architecture_selection": False,
    }


def select_truth_free_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a candidate lexicographically without reading any truth diagnostic."""
    if not records:
        raise ValueError("no architecture candidates were evaluated")
    for record in records:
        if record.get("status") != "COMPLETE" or record.get("truth_used") is not False:
            raise ValueError(
                "architecture selection requires complete truth-free records"
            )
        if record.get("sfh_used_for_architecture_selection") is not False:
            raise ValueError("SFH cannot drive the core population architecture choice")
        if record.get("redshift_median_gate_used") is not False:
            raise ValueError("redshift medians cannot drive architecture selection")
        if not isinstance(record.get("passes_nll_non_regression_gate"), bool):
            raise ValueError("architecture selection requires an explicit NLL gate")
    return min(
        records,
        key=lambda item: (
            not item["passes_nll_non_regression_gate"],
            float(item["primary_score"]),
            float(item["secondary_mean_core_5d_cdf_supremum"]),
            float(item["fit_validation_weighted_nll_mean"]),
            str(item["candidate"]),
        ),
    )
