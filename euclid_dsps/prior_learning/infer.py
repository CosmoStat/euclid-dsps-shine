"""Sampling and report helpers for supervised Diffsky priors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json

from .data import load_truth_dataset, prior_samples_frame
from .diagnostics import write_supervised_prior_diagnostics
from .train import load_prior_checkpoint, prior_learning_config


def sample_supervised_prior(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    n_samples: int,
    seed: int,
) -> None:
    """Sample theta values from a supervised RealNVP prior checkpoint."""
    del config
    out = ensure_dir(out_dir)
    prior, sidecar, latent_spec, _schema = load_prior_checkpoint(checkpoint)
    key = jax.random.PRNGKey(int(seed))
    x = np.asarray(prior.sample(key, int(n_samples)), dtype=np.float32)
    log_prob = np.asarray(prior.log_prob(jnp.asarray(x)), dtype=float)
    frame = prior_samples_frame(x, latent_spec, log_prob=log_prob)
    frame.to_parquet(out / "learned_prior_samples.parquet", index=False)
    write_json(
        out / "supervised_prior_sample_summary.json",
        {
            "checkpoint": str(checkpoint),
            "n_samples": int(n_samples),
            "seed": int(seed),
            "parameter_names": list(latent_spec.names),
            "schema": sidecar.get("schema"),
        },
    )


def write_supervised_prior_run_report(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    dataset_path: str | Path | None = None,
    schema_name: str | None = None,
    max_truth: int | None = None,
) -> dict[str, str]:
    """Regenerate truth-vs-prior diagnostics for an existing prior run."""
    cfg = prior_learning_config(config)
    run = Path(run_dir)
    out = ensure_dir(out_dir or run)
    prior_path = run / "learned_prior_samples.parquet"
    if not prior_path.exists():
        raise FileNotFoundError(f"Missing learned prior samples: {prior_path}")
    prior = pd.read_parquet(prior_path)
    if dataset_path is not None:
        truth = load_truth_dataset(
            dataset_path,
            schema_name=str(schema_name or cfg["schema"]),
            missing_policy=str(cfg.get("missing_policy", "reduce")),
            bounds=cfg.get("bounds", {}),
            limit=max_truth,
        )
        truth_frame = truth.theta_frame()
        names = truth.parameter_names
        summary = {
            "dataset": str(dataset_path),
            "schema": truth.schema.to_dict(),
            "n_truth_rows": int(len(truth_frame)),
            "dropped_nonfinite_rows": int(truth.dropped_rows),
        }
    else:
        truth_path = run / "truth_theta_samples.parquet"
        if not truth_path.exists():
            raise FileNotFoundError(
                "No dataset was supplied and the run has no truth_theta_samples.parquet"
            )
        truth_frame = pd.read_parquet(truth_path)
        names = tuple(
            col
            for col in truth_frame.columns
            if col not in {"object_id", "source_row"} and col in prior.columns
        )
        summary = {
            "dataset": "run_truth_theta_samples",
            "n_truth_rows": int(len(truth_frame)),
        }
    return write_supervised_prior_diagnostics(
        truth=truth_frame,
        prior=prior,
        parameter_names=tuple(names),
        out_dir=out,
        summary=summary,
    )
