"""Audit utilization and geometry of trained FENIKS AVI mixture experts."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write


def audit(training: Path, arm_name: str, out: Path, *, platform: str = "gpu") -> None:
    import equinox as eqx
    import jax

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.avi_experiments import Arm
    from euclid_dsps.amortized.expert_audit import (
        evaluate_expert_geometry,
        gate_diagnostics,
        pairwise_mean_separation,
        responsibility_diagnostics,
    )
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.latent import x_to_theta
    from euclid_dsps.amortized.mira import (
        FENIKS_SPLINE15D_PARAMETERS,
        evaluate_feniks_mira,
    )
    from euclid_dsps.amortized.proposal_expressivity import (
        IndependentFlowMixture,
        sample_independent_mixture,
    )
    from euclid_dsps.amortized.train import load_checkpoint
    from euclid_dsps.config import load_config
    from scripts.feniks_avi_experiments import (
        check_inputs,
        initialize_candidate,
        initialize_transport,
    )

    manifest = check_inputs(training)
    arms = {record["name"]: Arm(**record) for record in manifest["arms"]}
    if arm_name not in arms or arms[arm_name].experts < 2:
        raise ValueError(f"trained multi-expert arm required: {arm_name}")
    arm_dir = training / "arms" / arm_name
    final = read(arm_dir / "FINAL.json")
    if final.get("status") != "TRAINING_COMPLETE":
        raise ValueError(f"incomplete arm: {arm_name}")
    if out.exists():
        raise FileExistsError(out)
    devices = tuple(jax.local_devices())
    if not devices or any(device.platform != platform for device in devices):
        raise ValueError(f"{platform} JAX device required")
    config = load_config(training / "source_config.yaml")
    model = load_checkpoint(manifest["source"]["checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    model = eqx.tree_at(
        lambda item: item.encoder, model, initialize_transport(model.encoder)
    )
    out.mkdir(parents=True)
    runtime = prepare_adaptive_training_runtime(
        config,
        out / "runtime",
        train_indices_file=training / "train.npy",
        validation_indices_file=training / "validation.npy",
        validation_catalog_path=manifest["validation_catalog"],
        fixed_feature_stats_path=manifest["source"]["feature_stats"],
        train_population_prior=False,
    )
    template = initialize_candidate(
        model, config, runtime.latent_spec, arms[arm_name], manifest["seed"]
    )
    candidate = eqx.tree_deserialise_leaves(arm_dir / "encoder.eqx", template)
    if not isinstance(candidate, IndependentFlowMixture):
        raise TypeError("mixture checkpoint did not reconstruct a mixture")
    arrays = runtime.validation_arrays
    features = make_encoder_features(
        arrays.flux, arrays.flux_err, runtime.feature_stats, arrays.mask
    )
    probabilities, responsibilities, means = evaluate_expert_geometry(
        model,
        candidate,
        features,
        jax.random.PRNGKey(manifest["seed"] + 83000000),
        draws=32,
    )
    probabilities, responsibilities, means = jax.device_get(
        (probabilities, responsibilities, means)
    )
    gate = gate_diagnostics(probabilities)
    resp = responsibility_diagnostics(responsibilities)
    separation = pairwise_mean_separation(means)
    pd.DataFrame(
        {
            "validation_index": np.arange(len(probabilities)),
            **{
                f"gate_probability_{i}": probabilities[:, i]
                for i in range(candidate.n_components)
            },
            "gate_entropy": -np.sum(
                np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0),
                axis=1,
            ),
            "gate_winner": np.argmax(probabilities, axis=1),
        }
    ).to_csv(out / "gate_by_object.csv", index=False)
    pd.DataFrame(separation).to_csv(out / "expert_pairwise_separation.csv", index=False)
    # Draw from the learned gate only after completing truth-free geometry metrics.
    raw = sample_independent_mixture(
        model,
        candidate,
        jax.random.PRNGKey(manifest["seed"] + 84000000),
        features,
        256,
    ).x
    theta = np.asarray(jax.device_get(x_to_theta(raw, runtime.latent_spec)))
    rows = np.asarray(arrays.row_index)
    names = tuple(runtime.latent_spec.names)
    samples = pd.DataFrame(theta.swapaxes(0, 1).reshape(-1, len(names)), columns=names)
    samples.insert(0, "sample_id", np.tile(np.arange(256), len(rows)))
    samples.insert(0, "row_index", np.repeat(rows, 256))
    samples.insert(0, "object_id", np.repeat(rows, 256))
    samples.to_parquet(out / "raw_joint_samples.parquet", index=False)
    truth = pd.read_parquet(
        manifest["validation_catalog"], columns=list(FENIKS_SPLINE15D_PARAMETERS)
    ).iloc[rows]
    truth = truth.copy()
    truth.insert(0, "row_index", rows)
    truth.insert(0, "object_id", rows)
    truth.to_parquet(out / "inference_truth.parquet", index=False)
    evaluate_feniks_mira(
        truth_path=out / "inference_truth.parquet",
        posterior_specs=[(arm_name, out / "raw_joint_samples.parquet")],
        out_dir=out / "mira",
        samples_per_object=256,
        seed=manifest["seed"] + 85000000,
        num_regions=100,
        num_bootstrap=1000,
    )
    mira_scores = pd.read_csv(out / "mira/mira_scores.csv")
    physical_mira = float(
        mira_scores.loc[mira_scores.group.eq("physical_5d"), "score"].iloc[0]
    )
    validation = pd.read_csv(arm_dir / "validation_final.csv")
    support = validation.groupby("validation_index").median(numeric_only=True)
    write(
        out / "EXPERT_AUDIT.json",
        {
            "status": "EXPERT_AUDIT_COMPLETE",
            "arm": arm_name,
            "encoder_sha256": sha(arm_dir / "encoder.eqx"),
            "validation_objects": len(probabilities),
            "coordinate_space": "normalized joint latent x; physical dimensions 0:5, SFH dimensions 5:15",
            "gate": gate,
            "responsibilities": resp,
            "pairwise_separation": separation,
            "support": {
                "median_ess_fraction_k512": float(support.ess_fraction.median()),
                "median_max_weight_k512": float(support.max_weight.median()),
                "fraction_ess_below_5_k512": float((support.ess < 5).mean()),
            },
            "physical_5d_mira_raw": physical_mira,
            "truth_used_for_gate_or_geometry": False,
            "truth_used_for_mira_only": True,
            "scientific_promotion": False,
        },
    )
    print(pd.DataFrame(separation).to_string(index=False), flush=True)
    print(gate, flush=True)
    print(resp, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.training, args.arm, args.out)


if __name__ == "__main__":
    main()
