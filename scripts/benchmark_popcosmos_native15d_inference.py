#!/usr/bin/env python3
"""Benchmark native COSMOS encoder, posterior draws, and DSPS prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.config import require_equinox
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.decoder import model_flux_from_x
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.posterior import (
    posterior_encoder_state,
    sample_posterior_from_state,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.model import dynamic_model_args, load_context

eqx = require_equinox()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--row-indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--decoder-sample-chunk-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260805)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def summarize_timings(records: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Summarize repeated synchronized timings without mixing compilation."""
    summary: dict[str, dict[str, float | int]] = {}
    for phase, group in records.groupby("phase", sort=False):
        seconds = group["seconds"].to_numpy(float)
        summary[str(phase)] = {
            "n_repeats": int(len(seconds)),
            "median_seconds": float(np.median(seconds)),
            "p16_seconds": float(np.quantile(seconds, 0.16)),
            "p84_seconds": float(np.quantile(seconds, 0.84)),
            "median_seconds_per_object": float(
                np.median(group["seconds_per_object"].to_numpy(float))
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.posterior_samples <= 0 or args.repeats <= 0:
        raise ValueError("limit, posterior-samples, and repeats must be positive")
    if args.decoder_sample_chunk_size <= 0:
        raise ValueError("decoder-sample-chunk-size must be positive")
    for path in (
        args.config,
        args.dataset,
        args.checkpoint,
        args.feature_stats,
        args.row_indices,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")

    setup_start = time.perf_counter()
    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    indices = np.asarray(np.load(args.row_indices), dtype=np.int64)[: args.limit]
    if len(indices) != args.limit:
        raise RuntimeError(f"Requested {args.limit} rows, found {len(indices)}")
    stats = read_feature_stats(args.feature_stats)
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10_000, row_indices=indices
    )
    feature_start = time.perf_counter()
    batch = next(
        iter_photometry_batches_from_arrays(
            arrays, batch_size=args.limit, feature_stats=stats
        )
    )
    jax.block_until_ready(batch.features)
    feature_seconds = time.perf_counter() - feature_start
    model = load_checkpoint(args.checkpoint, config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    latent_spec = _latent_spec_for_amortized_config(config)
    setup_seconds = time.perf_counter() - setup_start

    encode = eqx.filter_jit(lambda features: posterior_encoder_state(model, features))
    state, encoder_compile_seconds = _timed(encode, batch.features)

    sample = eqx.filter_jit(
        lambda key, encoder_state: sample_posterior_from_state(
            model,
            key,
            encoder_state,
            args.posterior_samples,
        )
    )
    key = jax.random.PRNGKey(args.seed)
    key, compile_key = jax.random.split(key)
    posterior, sampling_compile_seconds = _timed(sample, compile_key, state)

    decode_chunk = eqx.filter_jit(
        lambda x: model_flux_from_x(
            x,
            latent_spec,
            context,
            model_args,
            latent_spec.names,
        )
    )
    chunk_size = int(args.decoder_sample_chunk_size)
    _flux, predictive_compile_seconds = _timed(
        decode_chunk, posterior.x[:chunk_size]
    )

    records: list[dict[str, float | int | str]] = []
    for repeat in range(args.repeats):
        state, seconds = _timed(encode, batch.features)
        records.append(_record("encoder_only", repeat, seconds, args.limit))

        key, sample_key = jax.random.split(key)
        posterior, seconds = _timed(sample, sample_key, state)
        records.append(_record("posterior_draws", repeat, seconds, args.limit))

        start = time.perf_counter()
        checksum = None
        for offset in range(0, args.posterior_samples, chunk_size):
            flux = decode_chunk(posterior.x[offset : offset + chunk_size])
            checksum = flux.sum() if checksum is None else checksum + flux.sum()
        jax.block_until_ready(checksum)
        seconds = time.perf_counter() - start
        records.append(
            _record("posterior_predictive", repeat, seconds, args.limit)
        )
        print(
            "[cosmos-timing] "
            f"repeat={repeat + 1}/{args.repeats} encoder={records[-3]['seconds']:.6f}s "
            f"samples={records[-2]['seconds']:.6f}s predictive={seconds:.6f}s",
            flush=True,
        )

    frame = pd.DataFrame(records)
    summary = summarize_timings(frame)
    compilation = {
        "encoder_only_seconds": float(encoder_compile_seconds),
        "posterior_draws_seconds": float(sampling_compile_seconds),
        "posterior_predictive_first_chunk_seconds": float(
            predictive_compile_seconds
        ),
        "posterior_predictive_compile_shape": list(
            posterior.x[:chunk_size].shape
        ),
    }
    device = jax.devices()[0]
    payload = {
        "status": "complete",
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", str(device)),
        "n_devices_visible": int(len(jax.devices())),
        "n_objects": int(args.limit),
        "posterior_samples": int(args.posterior_samples),
        "decoder_sample_chunk_size": chunk_size,
        "setup_seconds": float(setup_seconds),
        "feature_construction_seconds": float(feature_seconds),
        "compilation": compilation,
        "steady_state": summary,
        "latency_contract": {
            "encoder_only": "precomputed features to posterior encoder state",
            "posterior_draws": (
                "precomputed encoder state to the configured number of latent "
                "samples and exact densities"
            ),
            "posterior_predictive": (
                "all configured latent samples through DSPS to all configured "
                "band fluxes"
            ),
            "synchronization": "jax.block_until_ready after every timed phase",
            "compilation_excluded_from_steady_state": True,
            "io_excluded_from_steady_state": True,
        },
        "inputs": {
            "config": {"path": str(args.config), "sha256": _sha256(args.config)},
            "dataset": {"path": str(args.dataset), "sha256": _sha256(args.dataset)},
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": _sha256(args.checkpoint),
            },
            "feature_stats": {
                "path": str(args.feature_stats),
                "sha256": _sha256(args.feature_stats),
            },
            "row_indices": {
                "path": str(args.row_indices),
                "sha256": _sha256(args.row_indices),
            },
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    out = args.out
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {out}")
    out.mkdir(parents=True)
    frame.to_csv(out / "timing_repeats.csv", index=False)
    (out / "timing_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "DONE").touch()
    print(f"[cosmos-timing] complete -> {out}")


def _record(phase: str, repeat: int, seconds: float, n_objects: int) -> dict:
    return {
        "phase": phase,
        "repeat": int(repeat),
        "seconds": float(seconds),
        "n_objects": int(n_objects),
        "seconds_per_object": float(seconds / n_objects),
        "objects_per_second": float(n_objects / seconds),
    }


if __name__ == "__main__":
    main()
