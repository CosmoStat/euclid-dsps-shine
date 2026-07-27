#!/usr/bin/env python3
"""Audit the supervised spline15d prior, amortized training, and inference.

This also repairs, for diagnostics only, inference artifacts written before the
checkpoint-backed latent transform was used by ``amortized/infer.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from euclid_dsps.amortized.latent import latent_spec_from_config, theta_to_x, x_to_theta
from euclid_dsps.amortized.train import _latent_spec_for_amortized_config
from euclid_dsps.config import load_config


def _convert_legacy_theta(
    theta: np.ndarray,
    legacy_spec,
    checkpoint_spec,
    *,
    chunk_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    corrected = []
    for start in range(0, len(theta), chunk_size):
        old = jnp.asarray(theta[start : start + chunk_size], dtype=jnp.float32)
        x = theta_to_x(old, legacy_spec)
        xs.append(np.asarray(jax.device_get(x)))
        corrected.append(np.asarray(jax.device_get(x_to_theta(x, checkpoint_spec))))
    return np.concatenate(xs), np.concatenate(corrected)


def _posterior_metrics(
    summary: pd.DataFrame, truth: pd.DataFrame, names: tuple[str, ...]
) -> pd.DataFrame:
    joined = truth[["object_id", *names]].merge(
        summary, on="object_id", validate="one_to_one"
    )
    rows = []
    for name in names:
        y = joined[name].to_numpy(float)
        p = joined[f"{name}_median"].to_numpy(float)
        lo = joined[f"{name}_q16"].to_numpy(float)
        hi = joined[f"{name}_q84"].to_numpy(float)
        lo95 = joined[f"{name}_q025"].to_numpy(float)
        hi95 = joined[f"{name}_q975"].to_numpy(float)
        delta = p - y
        rows.append(
            {
                "parameter": name,
                "bias": np.mean(delta),
                "median_bias": np.median(delta),
                "rmse": np.sqrt(np.mean(delta**2)),
                "correlation": np.corrcoef(y, p)[0, 1],
                "coverage_68": np.mean((y >= lo) & (y <= hi)),
                "coverage_95": np.mean((y >= lo95) & (y <= hi95)),
                "median_width_68": np.median(hi - lo),
            }
        )
    return pd.DataFrame(rows)


def _prior_metrics(
    prior: pd.DataFrame, truth: pd.DataFrame, names: tuple[str, ...]
) -> pd.DataFrame:
    rows = []
    for name in names:
        p = prior[name].to_numpy(float)
        y = truth[name].to_numpy(float)
        pq = np.quantile(p, [0.16, 0.5, 0.84])
        yq = np.quantile(y, [0.16, 0.5, 0.84])
        rows.append(
            {
                "parameter": name,
                "ks": ks_2samp(y, p).statistic,
                "quantile_l1": np.mean(np.abs(pq - yq)),
                "median_delta": pq[1] - yq[1],
                "prior_q16": pq[0],
                "truth_q16": yq[0],
                "prior_median": pq[1],
                "truth_median": yq[1],
                "prior_q84": pq[2],
                "truth_q84": yq[2],
            }
        )
    return pd.DataFrame(rows)


def _plot_marginals(
    truth: pd.DataFrame,
    prior: pd.DataFrame,
    posterior: pd.DataFrame,
    names: tuple[str, ...],
    out: Path,
) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=True):
        y = truth[name].to_numpy(float)
        p = prior[name].to_numpy(float)
        q = posterior[f"{name}_median"].to_numpy(float)
        lo, hi = np.quantile(y, [0.001, 0.999])
        bins = np.linspace(lo, hi, 45)
        ax.hist(y, bins=bins, density=True, histtype="step", lw=2, label="truth")
        ax.hist(p, bins=bins, density=True, histtype="step", lw=1.5, label="prior")
        ax.hist(
            q,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            label="posterior median",
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlim(lo, hi)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_training(training: Path, out: Path) -> None:
    frame = pd.read_csv(training / "training_epoch_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for split, group in frame.groupby("split"):
        group = group.sort_values("epoch")
        axes[0].plot(group.epoch, group.negative_loglike, label=split)
        axes[1].plot(group.epoch, group.kl_mc_mean, label=split)
        axes[2].plot(group.epoch, group.posterior_median_log_std, label=split)
    axes[0].set(title="Negative log-likelihood", xlabel="epoch")
    axes[1].set(title="Monte Carlo KL", xlabel="epoch")
    axes[2].set(title="Posterior median log std", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    saved_config = json.loads((args.inference / "normalized_config.json").read_text())
    config["amortized"]["prior"]["checkpoint"] = saved_config["amortized"]["prior"][
        "checkpoint"
    ]
    legacy_spec = latent_spec_from_config(config)
    checkpoint_spec = _latent_spec_for_amortized_config(config)
    names = checkpoint_spec.names

    prior_old = pd.read_parquet(args.inference / "learned_prior_samples.parquet")
    x_prior, theta_prior = _convert_legacy_theta(
        prior_old[list(names)].to_numpy(float), legacy_spec, checkpoint_spec
    )
    prior = pd.DataFrame(theta_prior, columns=names)
    stored_x = prior_old[[f"x_{i:02d}" for i in range(len(names))]].to_numpy(float)
    prior_x_recovery_max_abs = float(np.max(np.abs(stored_x - x_prior)))

    frames = []
    for path in sorted((args.inference / "posterior_samples").glob("batch_*.parquet")):
        frame = pd.read_parquet(path, columns=["object_id", "sample_id", *names])
        _, corrected = _convert_legacy_theta(
            frame[list(names)].to_numpy(float), legacy_spec, checkpoint_spec
        )
        frame.loc[:, list(names)] = corrected
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    grouped = samples.groupby("object_id", sort=False)
    pieces = []
    for name in names:
        quantiles = grouped[name].quantile([0.025, 0.16, 0.5, 0.84, 0.975]).unstack()
        quantiles.columns = [
            f"{name}_q025",
            f"{name}_q16",
            f"{name}_median",
            f"{name}_q84",
            f"{name}_q975",
        ]
        pieces.append(quantiles)
    posterior = pd.concat(pieces, axis=1).reset_index()
    truth = pd.read_parquet(args.inference / "inference_truth.parquet")

    prior_metrics = _prior_metrics(prior, truth, names)
    posterior_metrics = _posterior_metrics(posterior, truth, names)
    prior_metrics.to_csv(args.out / "corrected_prior_vs_truth.csv", index=False)
    posterior_metrics.to_csv(args.out / "corrected_posterior_vs_truth.csv", index=False)
    posterior.to_parquet(args.out / "corrected_posterior_summary.parquet", index=False)
    _plot_marginals(
        truth,
        prior,
        posterior,
        names,
        args.out / "corrected_truth_prior_posterior.png",
    )
    _plot_training(args.training, args.out / "amortized_training_history.png")

    payload = {
        "legacy_inference_normalization": legacy_spec.normalization,
        "checkpoint_normalization": checkpoint_spec.normalization,
        "prior_x_recovery_max_abs": prior_x_recovery_max_abs,
        "n_prior": len(prior),
        "n_posterior_samples": len(samples),
        "n_objects": len(posterior),
        "median_prior_ks": float(prior_metrics.ks.median()),
        "max_prior_ks": float(prior_metrics.ks.max()),
        "median_posterior_coverage_68": float(posterior_metrics.coverage_68.median()),
        "median_posterior_coverage_95": float(posterior_metrics.coverage_95.median()),
    }
    (args.out / "audit_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
