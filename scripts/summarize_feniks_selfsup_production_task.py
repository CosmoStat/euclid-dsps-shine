#!/usr/bin/env python3
"""Summarize the learned-prior Student-t2 production-candidate array."""

from pathlib import Path

import summarize_feniks_selfsup_rws_task as base

LABELS = (
    "selfsup_rws_k8_t2",
    "selfsup_rws_mix2_k8_t2",
    "selfsup_smcwake_mix2_k4_t2",
)

DESCRIPTIONS = {
    "selfsup_rws_k8_t2": (
        "Learned RealNVP prior; Gaussian-base posterior; Student-t2 sleep/wake RWS K=8"
    ),
    "selfsup_rws_mix2_k8_t2": (
        "Learned RealNVP prior; exact two-component posterior base; Student-t2 RWS K=8"
    ),
    "selfsup_smcwake_mix2_k4_t2": (
        "Learned RealNVP prior; two-component posterior; Student-t2 tempered SMC-Wake K=4"
    ),
}

_base_aggregate = base.aggregate


def _aggregate_with_lens(root: Path, expected: int) -> None:
    _base_aggregate(root, expected)
    report = root / "comparison" / "README.md"
    if not report.exists():
        return
    content = report.read_text(encoding="utf-8").replace(
        "The synthetic likelihood is Gaussian with the catalog flux errors and no extra floor.",
        "The robust likelihood and model-generated sleep noise are Student-t with two degrees of freedom, catalog flux-error scales, and no extra floor.",
    )
    lines = [
        "",
        "## Learned-prior and sampler diagnostics",
        "",
        "| run | prior marginal L1/IQR | prior Spearman error | mixture max weight | SMC stage ESS | MALA acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        metrics = base._read_json(root / label / "metrics.json")
        lines.append(
            "| "
            + " | ".join(
                str(value)
                for value in (
                    label,
                    metrics.get("prior_15d_mean_quantile_l1_iqr"),
                    metrics.get("prior_15d_mean_spearman_abs_delta"),
                    metrics.get("posterior_mixture_max_weight"),
                    metrics.get("smc_stage_ess_mean"),
                    metrics.get("smc_mala_acceptance_mean"),
                )
            )
            + " |"
        )
    lines.extend(["", "## Jacobian lens diagnostics", ""])
    for label in LABELS:
        lens = root / label / "jacobian_lens"
        if (lens / "jacobian_lens_summary.json").is_file():
            lines.append(
                f"- **{label}**: [Jacobian lens](../{label}/jacobian_lens/) | "
                "[15D prior correlation error]"
                f"(../{label}/inference/prior_vs_truth_correlation_error.png)"
            )
    report.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    base.LABELS = LABELS
    base.DESCRIPTIONS = DESCRIPTIONS
    base.aggregate = _aggregate_with_lens
    base.main()
