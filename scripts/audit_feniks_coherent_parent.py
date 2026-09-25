"""Read-only dataset qualification before adapting population/posterior training.

Test truth is used only for mechanical identity/weight/selection checks. Target
geometry diagnostics use parent TRAIN only and never fit a scientific model.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import SPLITS, complete, settings


def load_spec(path):
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

    payload = read(path)
    kwargs = {f.name: payload[f.name] for f in fields(LatentSpec) if f.name in payload}
    kwargs["names"] = tuple(kwargs["names"])
    if len(kwargs["names"]) != 15 or set(kwargs["names"]) != set(
        SPLINE15D_PARAMETER_NAMES
    ):
        raise ValueError("Expected the full 15D spline latent schema")
    for name, value in list(kwargs.items()):
        if isinstance(value, list):
            kwargs[name] = jnp.asarray(value)
    spec = LatentSpec(**kwargs)
    if spec.normalization not in {
        "identity",
        "standardized_logit",
        "bounded_mixed_warp",
    }:
        raise ValueError("This audit compares the historical bounded reference only")
    return spec


def geometry(frame, spec):
    import jax.numpy as jnp
    from scipy.stats import norm, wasserstein_distance

    from euclid_dsps.amortized.forward_population import PHYSICAL
    from euclid_dsps.amortized.latent import theta_to_x, x_to_theta

    theta = frame[list(spec.names)].to_numpy(np.float64)
    if theta.size == 0 or not np.isfinite(theta).all():
        raise ValueError("Empty or nonfinite 15D truth")
    lower, upper = np.asarray(spec.lower), np.asarray(spec.upper)
    x = np.asarray(theta_to_x(jnp.asarray(theta), spec))
    restored = np.asarray(x_to_theta(jnp.asarray(x), spec))
    if not np.isfinite(x).all() or not np.isfinite(restored).all():
        raise ValueError("Nonfinite latent transform")
    iqr = np.subtract(*np.percentile(theta, [75, 25], axis=0))
    scale = np.maximum(iqr, 1e-8)
    # Outside/boundary counts must be checked BEFORE accepting a clipped logit.
    outside = (theta < lower) | (theta > upper)
    boundary = (theta == lower) | (theta == upper)
    rows = []
    normal = norm.ppf((np.arange(len(frame)) + 0.5) / len(frame))
    for k, name in enumerate(spec.names):
        sfh = name not in PHYSICAL
        rows.append(
            dict(
                parameter=name,
                group="sfh" if sfh else "physical",
                lower=float(lower[k]),
                upper=float(upper[k]),
                minimum=float(theta[:, k].min()),
                maximum=float(theta[:, k].max()),
                outside_fraction=float(outside[:, k].mean()),
                boundary_fraction=float(boundary[:, k].mean()),
                zero_fraction=float(np.mean(theta[:, k] == 0)),
                distinct_values=int(np.unique(theta[:, k]).size),
                iqr=float(iqr[k]),
                max_roundtrip_over_iqr=float(
                    np.max(abs(restored[:, k] - theta[:, k])) / scale[k]
                ),
                x_mean=float(x[:, k].mean()),
                x_std=float(x[:, k].std()),
                # Only SFH coordinates are forced to N(0,1) in the old family.
                sfh_w1_to_standard_normal=float(wasserstein_distance(x[:, k], normal))
                if sfh
                else None,
            )
        )
    summary = pd.DataFrame(rows)
    correlations = pd.DataFrame(theta, columns=spec.names).corr(method="spearman")
    cross = correlations.loc[
        list(PHYSICAL), [n for n in spec.names if n not in PHYSICAL]
    ]
    finite_cross = cross.to_numpy()[np.isfinite(cross.to_numpy())]
    detail = dict(
        rows=len(frame),
        outside_rows=int(outside.any(axis=1).sum()),
        boundary_rows=int(boundary.any(axis=1).sum()),
        max_roundtrip_over_iqr=float(summary.max_roundtrip_over_iqr.max()),
        maximum_physical_sfh_abs_spearman=float(np.max(abs(finite_cross)))
        if len(finite_cross)
        else None,
        undefined_cross_correlations=int(np.sum(~np.isfinite(cross.to_numpy()))),
        sfh_coordinates_with_zero_fraction_above_1pct=summary.loc[
            (summary.group == "sfh") & (summary.zero_fraction > 0.01), "parameter"
        ].tolist(),
        diagnostics_scope="parent train only; no model fitted; correlations are descriptive",
        marginal_agreement_does_not_prove_conditional_agreement=True,
    )
    return summary, correlations, detail


def verify_catalogues(root):
    from euclid_dsps.synthetic_diffsky.coherent_parent import catalogue_checks

    m, cfg, digest = settings(root)
    if not complete(root / "report", digest):
        raise ValueError("Dataset report incomplete")
    contract = read(root / "dataset/CONTRACT.json")
    if contract.get("status") != "COHERENT_PARENT_DATASET_COMPLETE" or not contract.get(
        "ready_for_population_benchmark"
    ):
        raise ValueError("Dataset contract has not passed")
    source_ids, effective_seeds, summary, train = {}, {}, {}, None
    bands = read(root / "decoder.json")["bands"]
    for split in SPLITS:
        parent = pd.read_parquet(root / "dataset/parent" / f"{split}.parquet")
        selected = pd.read_parquet(root / "dataset/selected_r29" / f"{split}.parquet")
        if len(parent) != cfg["parent_rows"][split]:
            raise ValueError(f"Unexpected row count: {split}")
        pd.testing.assert_frame_equal(
            selected.reset_index(drop=True),
            parent.loc[parent.selected_r29].reset_index(drop=True),
        )
        if selected.empty or not parent.split.eq(split).all():
            raise ValueError(f"Empty selection or wrong split labels: {split}")
        source_ids[split] = set(parent.effective_proposal_key)
        effective_seeds[split] = set(parent.effective_source_seed)
        c = catalogue_checks(parent, bands, cfg["selection"])
        old = read(root / "report/checks.json")["splits"][split]
        if any(
            c[k] != old[k]
            for k in ("rows", "selected", "empirical_alpha", "analytic_alpha")
        ):
            raise ValueError("Recomputed catalogue checks differ from completed report")
        summary[split] = {
            k: c[k] for k in ("rows", "selected", "empirical_alpha", "analytic_alpha")
        }
        if split == "train":
            train = parent
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            if (
                source_ids[left] & source_ids[right]
                or effective_seeds[left] & effective_seeds[right]
            ):
                raise ValueError("Effective source leakage between splits")
    return train, summary, m


def run(root, spec_path, out):
    if out.exists() or out == root or root in out.parents:
        raise ValueError(
            "Use a new output directory outside the immutable dataset root"
        )
    spec_hash = sha(spec_path)
    train, counts, manifest = verify_catalogues(root)
    spec = load_spec(spec_path)
    table, correlations, detail = geometry(train, spec)
    if sha(spec_path) != spec_hash:
        raise ValueError("Latent spec changed during audit")
    blockers = []
    if detail["outside_rows"]:
        blockers.append("truth_outside_old_latent_support")
    if detail["boundary_rows"]:
        blockers.append("truth_at_bounded_transform_endpoints")
    if detail["max_roundtrip_over_iqr"] > 1e-4:
        blockers.append("old_transform_roundtrip_error")
    # A finite empirical catalogue always has repeated values. These warnings
    # flag possible structural atoms, not a proof about a continuous parent law.
    review = [
        "fixed_independent_standard_normal_SFH_is_not_learned_by_physical_weights"
    ]
    if detail["sfh_coordinates_with_zero_fraction_above_1pct"]:
        review.append("inspect_SFH_zero_pileups_before_continuous_flow_training")
    out.mkdir(parents=True)
    table.to_csv(out / "support_and_transform.csv", index=False)
    correlations.to_csv(out / "train_spearman.csv")
    result = dict(
        status="COHERENT_QUALIFICATION_COMPLETE",
        dataset_integrity_pass=True,
        root=str(root),
        source_manifest_sha256=sha(root / "MANIFEST.json"),
        latent_spec=str(spec_path),
        latent_spec_sha256=spec_hash,
        parent_scope=manifest["parent_scope"],
        splits=counts,
        geometry=detail,
        old_transform_compatible=not blockers,
        blockers=blockers,
        review_required=review,
        ready_for_population_training=False,
        ready_for_production=False,
        truth_role="mechanical checks all splits; geometry diagnostics train only",
        independent_decoder_accuracy_tested=False,
        next_action="adapt_reference_and_new_runtime_before_training",
        audit_script_sha256=sha(Path(__file__)),
        latent_implementation_sha256=sha(
            Path(__file__).resolve().parents[1] / "euclid_dsps/amortized/latent.py"
        ),
        artifacts={
            p.name: sha(p)
            for p in (out / "support_and_transform.csv", out / "train_spearman.csv")
        },
    )
    write(out / "FINAL.json", result)
    print("DATASET INTEGRITY: PASS")
    print("OLD LATENT SUPPORT/TRANSFORM:", "BLOCKED" if blockers else "PASS")
    print(
        table[
            [
                "parameter",
                "outside_fraction",
                "boundary_fraction",
                "zero_fraction",
                "max_roundtrip_over_iqr",
            ]
        ].to_string(index=False)
    )
    print("PHYSICAL-SFH MAX ABS SPEARMAN:", detail["maximum_physical_sfh_abs_spearman"])
    print("BLOCKERS:", blockers)
    print("REVIEW:", review)
    print("Saved:", out)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--latent-spec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.root.resolve(), a.latent_spec.resolve(), a.out.resolve())


if __name__ == "__main__":
    main()
