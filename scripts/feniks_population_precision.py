"""Bounded, restartable metric/oracle and independent decoder checks.

The oracle observes true latent coordinates and is only a diagnostic reference.
Every photometric fit uses frozen forward classifiers, never posterior draws.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.forward_population import (
    PHYSICAL,
    PhysicalBasis,
    parent_from_selected,
)
from euclid_dsps.amortized.population_low_rank import fit_low_rank_selected_weights
from scripts.feniks_avi_experiments import read, sha, write

ARMS = ("noiseless_photometry", "noisy_photometry")
RATIOS = ("exact", "photometric")


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def receipt_valid(directory):
    path = directory / "FINAL.json"
    if not path.is_file():
        return False
    receipt = read(path)
    return (
        receipt.get("status") == "COMPLETE"
        and bool(receipt.get("artifacts"))
        and all(
            (directory / name).is_file() and sha(directory / name) == digest
            for name, digest in receipt.get("artifacts", {}).items()
        )
    )


def finish(directory, **values):
    artifacts = {
        p.name: sha(p)
        for p in directory.iterdir()
        if p.is_file() and p.name not in {"FINAL.json", "PROGRESS.json", "FAILURE.json"}
    }
    write(
        directory / "FINAL.json", dict(status="COMPLETE", artifacts=artifacts, **values)
    )


def config(root):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"changed precision-audit input: {item['path']}")
    return manifest, manifest["settings"]


def choose_fixed_candidate(frame):
    """Freeze a single heldout optimum, independent of truth and bootstraps."""
    selected = frame.sort_values(
        ["heldout_log_likelihood", "rank", "strength"],
        ascending=[False, True, False],
    ).iloc[0]
    return dict(rank=int(selected["rank"]), strength=float(selected.strength))


def prepare(source, root, settings_path):
    from scripts.feniks_population_low_rank_audit import contract

    source_manifest, _ = contract(source)
    settings = yaml.safe_load(settings_path.read_text())
    for key in (
        "bootstraps",
        "directions",
        "bootstrap_metric_draws",
        "decoder_objects",
        "decoder_batch",
    ):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    if (
        sorted(set(settings["metric_draws"])) != settings["metric_draws"]
        or settings["metric_draws"][0] < 16
    ):
        raise ValueError("metric_draws must be increasing positive sizes")
    if len(settings["metric_seeds"]) < 2:
        raise ValueError("at least two metric seeds required")
    if root.exists():
        raise FileExistsError(root)
    chosen, inputs = {}, {"source_manifest": source / "MANIFEST.json"}
    for arm in ARMS:
        path = source / arm / "candidate_path_prebootstrap.csv"
        chosen[arm] = choose_fixed_candidate(pd.read_csv(path))
        inputs[f"candidates_{arm}"] = path
    inputs["modes"] = source / "spectral_basis/modes.npz"
    inputs["latent_spec"] = source / ARMS[0] / "runtime/effective_latent_spec.json"
    convergence = Path(source_manifest["source_convergence"])
    for arm in ARMS:
        inputs[f"classifier_{arm}"] = convergence / arm / "best.eqx"
    root.mkdir(parents=True)
    for name in ("logs", "cache", "decoder", "metric", "bootstrap", "report"):
        (root / name).mkdir()
    shutil.copy2(settings_path, root / "precision.yaml")
    inputs["config"] = root / "precision.yaml"
    write(
        root / "MANIFEST.json",
        dict(
            source=str(source),
            convergence=str(convergence),
            settings=settings,
            candidates=chosen,
            inputs={k: dict(path=str(v), sha256=sha(v)) for k, v in inputs.items()},
            population_uses_q=False,
            production_prior_modified=False,
            truth_role="exact-latent oracle and evaluation only",
            candidate_selection="frozen heldout maximum, no new grid search",
        ),
    )
    roadmap(root)
    print(
        json.dumps(
            dict(candidates=chosen, resources=settings["resources"], new_training=0),
            indent=2,
        )
    )


def bootstrap_counts(size, seed, repeat):
    rng = np.random.default_rng(np.random.SeedSequence([seed, repeat]))
    return np.bincount(rng.integers(size, size=size), minlength=size)


def load_geometry(root):
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta
    from scripts.feniks_population_inversion_audit import _common_draws

    payload = read(root / "cache/latent_spec.json")
    array_names = {
        "lower",
        "upper",
        "raw_center",
        "raw_scale",
        "transform_family",
        "transform_location",
        "transform_lambda",
    }
    kwargs = {f.name: payload[f.name] for f in fields(LatentSpec) if f.name in payload}
    kwargs["names"] = tuple(kwargs["names"])
    for name in array_names:
        if kwargs.get(name) is not None:
            kwargs[name] = jnp.asarray(kwargs[name])
    spec = LatentSpec(**kwargs)
    with np.load(root / "cache/geometry.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    basis = PhysicalBasis(spec.names, data["centers"], data["scales"])

    def theta(weights, count, seed):
        # Separate generators keep the prefixes identical across sample sizes.
        uniforms = np.random.default_rng(seed).random(count)
        normals = np.random.default_rng(seed + 1).normal(size=(count, len(spec.names)))
        x = _common_draws(basis, weights, uniforms, normals)
        return np.asarray(x_to_theta(jnp.asarray(x), spec))[:, basis.indices]

    return data, theta


def fixed_metrics(left, right, scale, directions):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 5:
        raise ValueError("equal nonempty physical 5D sample shapes required")
    if not left.size or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("finite samples required")
    scale = np.asarray(scale)
    if scale.shape != (5,) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("positive frozen physical scales required")
    p, q = left / scale, right / scale
    projected = np.abs(
        np.sort(p @ directions, axis=0) - np.sort(q @ directions, axis=0)
    )
    result = dict(parent_sw=float(projected.mean()))
    marginal = np.mean(abs(np.sort(p, axis=0) - np.sort(q, axis=0)), axis=0)
    for name, value in zip(PHYSICAL, marginal, strict=True):
        result[f"w1_{name}"] = float(value)
    return result


def fit_cell(root, arm, ratio, counts=None):
    manifest, settings = config(root)
    out = root / "cache"
    with np.load(out / "geometry.npz") as data:
        frequencies, alpha, modes = data["frequencies"], data["alpha"], data["modes"]
    chosen = manifest["candidates"][arm]
    logc = np.load(
        out / f"{arm if ratio == 'photometric' else 'exact'}.npy", mmap_mode="r"
    )
    initial = None
    full_path = out / f"full_{arm}_{ratio}.npz"
    if counts is not None:
        with np.load(full_path) as saved:
            initial = saved["coefficients"]
    start = time.monotonic()
    selected, coefficients, diagnostics = fit_low_rank_selected_weights(
        logc,
        frequencies,
        modes[:, : chosen["rank"]],
        strength=chosen["strength"],
        observation_weights=counts,
        tolerance=settings["solver_tolerance"],
        maximum_iterations=settings["solver_maximum_iterations"],
        maximum_seconds=settings["solver_seconds"],
        initial_coefficients=initial,
    )
    parent = parent_from_selected(selected, alpha)
    return dict(selected=selected, parent=parent, coefficients=coefficients), dict(
        **diagnostics,
        elapsed_seconds=time.monotonic() - start,
        selected_sum=float(selected.sum()),
        parent_sum=float(parent.sum()),
    )


def cache(root):
    from scripts.feniks_population_basis_audit import _classifier_log_probabilities
    from scripts.feniks_ratio_followup import _context
    from scripts.feniks_ratio_ladder import exact_selected_log_classifier

    manifest, settings = config(root)
    out = root / "cache"
    if receipt_valid(out):
        return
    context = _context(Path(manifest["convergence"]))
    (
        _,
        _,
        ratio_manifest,
        _,
        _,
        target,
        basis,
        splits,
        frequencies,
        alpha,
        _,
        truth,
        _,
    ) = context
    source = Path(manifest["source"])
    with np.load(source / "spectral_basis/modes.npz") as data:
        modes = data["modes"]
        np.testing.assert_allclose(
            data["reference_selected"], frequencies, rtol=0, atol=1e-14
        )
    atomic_npz(
        out / "geometry.npz",
        centers=basis.centers,
        scales=basis.scales,
        alpha=alpha,
        frequencies=frequencies,
        modes=modes,
        true_parent=truth,
    )
    shutil.copy2(manifest["inputs"]["latent_spec"]["path"], out / "latent_spec.json")
    atomic_npz(
        out / "identities.npz",
        target_fit=splits["target_fit"],
        target_heldout=splits["target_heldout"],
    )
    write(out / "ratio_manifest.json", ratio_manifest)
    exact = exact_selected_log_classifier(
        target["x"][splits["target_fit"]], basis, alpha, frequencies
    )
    np.save(out / "exact.npy", exact)
    for arm in ARMS:
        write(out / "PROGRESS.json", dict(stage="frozen_classifier", arm=arm))
        logc, heldout, calibration = _classifier_log_probabilities(
            Path(manifest["convergence"]), arm, context, target=True
        )
        np.save(out / f"{arm}.npy", logc)
        write(out / f"{arm}_calibration.json", calibration)
        del heldout
    for arm in ARMS:
        for ratio in RATIOS:
            write(out / "PROGRESS.json", dict(stage="full_fit", arm=arm, ratio=ratio))
            weights_path = out / f"full_{arm}_{ratio}.npz"
            cert_path = out / f"full_{arm}_{ratio}.json"
            if (
                weights_path.is_file()
                and cert_path.is_file()
                and read(cert_path).get("weights_sha256") == sha(weights_path)
            ):
                continue
            arrays, diagnostics = fit_cell(root, arm, ratio)
            atomic_npz(weights_path, **arrays)
            write(cert_path, dict(**diagnostics, weights_sha256=sha(weights_path)))
    geometry, theta = load_geometry(root)
    reference = theta(
        geometry["true_parent"], max(settings["metric_draws"]), settings["seed"]
    )
    scale = np.maximum(
        np.quantile(reference, 0.75, axis=0) - np.quantile(reference, 0.25, axis=0),
        1e-8,
    )
    directions = np.random.default_rng(settings["seed"]).normal(
        size=(5, settings["directions"])
    )
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    atomic_npz(out / "metric_reference.npz", scale=scale, directions=directions)
    finish(out, objects=len(exact), alpha_conditioned_on_saved_bank=True)


def metric_reference(root):
    with np.load(root / "cache/metric_reference.npz") as payload:
        return payload["scale"], payload["directions"]


def metric(root, task):
    _, settings = config(root)
    arm = ARMS[task]
    out = root / "metric" / arm
    out.mkdir(exist_ok=True)
    if receipt_valid(out):
        return
    geometry, theta = load_geometry(root)
    scale, directions = metric_reference(root)
    weights = {"truth": geometry["true_parent"]}
    for ratio in RATIOS:
        with np.load(root / f"cache/full_{arm}_{ratio}.npz") as saved:
            weights[ratio] = saved["parent"]
    rows = []
    for seed in settings["metric_seeds"]:
        for count in settings["metric_draws"]:
            samples = {key: theta(w, count, seed) for key, w in weights.items()}
            for left, right in (
                ("exact", "truth"),
                ("photometric", "truth"),
                ("photometric", "exact"),
                ("truth", "truth"),
            ):
                rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        draws=count,
                        left=left,
                        right=right,
                        **fixed_metrics(
                            samples[left], samples[right], scale, directions
                        ),
                    )
                )
            pd.DataFrame(rows).to_csv(out / "precision.csv", index=False)
            write(out / "PROGRESS.json", dict(stage="metric", seed=seed, draws=count))
    finish(out, simulations=0)


def bootstrap(root, task):
    _, settings = config(root)
    if not 0 <= task < settings["bootstraps"]:
        raise ValueError("bootstrap index out of range")
    geometry, theta = load_geometry(root)
    scale, directions = metric_reference(root)
    size = np.load(root / "cache/exact.npy", mmap_mode="r").shape[0]
    counts = bootstrap_counts(size, settings["seed"], task)
    errors = []
    for arm in ARMS:
        for ratio in RATIOS:
            out = root / "bootstrap" / f"repeat_{task:03d}" / f"{arm}_{ratio}"
            out.mkdir(parents=True, exist_ok=True)
            if receipt_valid(out):
                continue
            try:
                write(
                    out / "PROGRESS.json",
                    dict(stage="refit", repeat=task, arm=arm, ratio=ratio),
                )
                weights_path, cert_path = out / "weights.npz", out / "FIT.json"
                if (
                    weights_path.is_file()
                    and cert_path.is_file()
                    and read(cert_path).get("weights_sha256") == sha(weights_path)
                ):
                    with np.load(weights_path) as saved:
                        arrays = {key: saved[key] for key in saved.files}
                    diagnostics = read(cert_path)
                else:
                    arrays, diagnostics = fit_cell(root, arm, ratio, counts)
                    # Persist each expensive result before metric work.
                    atomic_npz(weights_path, **arrays)
                    diagnostics["weights_sha256"] = sha(weights_path)
                    write(cert_path, diagnostics)
                with np.load(root / f"cache/full_{arm}_{ratio}.npz") as saved:
                    full = saved["parent"]
                n, seed = (
                    settings["bootstrap_metric_draws"],
                    settings["metric_seeds"][0],
                )
                values = fixed_metrics(
                    theta(arrays["parent"], n, seed),
                    theta(full, n, seed),
                    scale,
                    directions,
                )
                finish(out, repeat=task, arm=arm, ratio=ratio, **values, **diagnostics)
            except Exception as error:
                write(out / "FAILURE.json", dict(error=repr(error), stage="bootstrap"))
                errors.append(str(error))
    if errors:
        raise RuntimeError("; ".join(errors))


def decoder(root):
    import equinox as eqx
    import jax.numpy as jnp

    from euclid_dsps.amortized.decoder import model_flux_from_x
    from euclid_dsps.amortized.posterior_target import (
        _apply_model_calibration,
        safe_decoder_inputs,
    )
    from euclid_dsps.photometry import abmag_to_fnu_cgs
    from scripts.feniks_failure_modes import _catalogue, _truth_x
    from scripts.feniks_ratio_ladder import _runtime, conditional_normal_score

    manifest, settings = config(root)
    out = root / "decoder"
    if receipt_valid(out):
        return
    convergence_manifest = read(Path(manifest["convergence"]) / "MANIFEST.json")
    source_manifest = read(Path(convergence_manifest["source"]) / "MANIFEST.json")
    for entry in source_manifest["inputs"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"decoder source changed: {entry['path']}")
    model, runtime, _, cfg = _runtime(source_manifest, out)
    bands = tuple(runtime.feature_stats.band_names)
    all_ids = np.load(source_manifest["inputs"]["audit_indices"]["path"])
    positions = np.linspace(
        0, len(all_ids) - 1, min(len(all_ids), settings["decoder_objects"]), dtype=int
    )
    ids = all_ids[positions]
    truth, x = _truth_x(source_manifest, runtime, ids)
    cat = _catalogue(source_manifest, bands, ids)
    saved = cat[[f"flux_true_{b}" for b in bands]].to_numpy()
    noisy = cat[[f"flux_{b}" for b in bands]].to_numpy()
    errors = cat[[f"fluxerr_{b}" for b in bands]].to_numpy()
    masks = cat[[f"mask_{b}" for b in bands]].to_numpy(bool)
    if np.any(masks & (~np.isfinite(errors) | (errors <= 0))):
        raise ValueError("invalid catalogue error")
    if np.any(masks & (~np.isfinite(saved) | ~np.isfinite(noisy))):
        raise ValueError("non-finite unmasked catalogue flux")
    np.save(out / "row_indices.npy", ids)
    write(
        out / "decoder_model.json",
        dict(model=cfg["model"], bands=cfg["bands"], inputs=source_manifest["inputs"]),
    )
    rows = []
    variants = (
        (
            "baseline",
            runtime.context.model_config.get(
                "photometry_integrator", "legacy_trapezoid_v1"
            ),
        ),
        ("merged", "merged_gauss4_v1"),
    )
    for label, integrator in variants:
        context = copy.copy(runtime.context)
        context.model_config = dict(
            runtime.context.model_config, photometry_integrator=integrator
        )
        current = replace(runtime, context=context)

        @eqx.filter_jit
        def decode_batch(value, current=current):
            safe, valid = safe_decoder_inputs(value, current.latent_spec)
            flux = model_flux_from_x(
                safe,
                current.latent_spec,
                current.context,
                current.model_args,
                current.parameter_names,
            )
            return (
                _apply_model_calibration(model, flux, current.calibration_config),
                valid,
            )

        chunks = []
        for start in range(0, len(ids), settings["decoder_batch"]):
            path = out / f"{label}_{start:04d}.npz"
            batch_ids = ids[start : start + settings["decoder_batch"]]
            if path.is_file():
                with np.load(path) as chunk:
                    flux, valid = chunk["flux"], chunk["valid"]
                    np.testing.assert_array_equal(chunk["row_indices"], batch_ids)
            else:
                write(
                    out / "PROGRESS.json",
                    dict(stage=label, complete=start, total=len(ids)),
                )
                flux, valid = decode_batch(
                    jnp.asarray(x[start : start + settings["decoder_batch"]])
                )
                flux, valid = np.asarray(flux), np.asarray(valid)
                atomic_npz(path, flux=flux, valid=valid, row_indices=batch_ids)
            if (
                flux.shape != (len(batch_ids), len(bands))
                or valid.shape != (len(batch_ids),)
                or not valid.all()
                or not np.isfinite(flux).all()
            ):
                raise ValueError(f"invalid decoded truth batch: {path}")
            chunks.append(flux)
        decoded = np.concatenate(chunks)
        for j, band in enumerate(bands):
            for i in np.flatnonzero(masks[:, j]):
                rows.append(
                    dict(
                        variant=label,
                        row=int(ids[i]),
                        band=band,
                        z=float(truth[i, runtime.latent_spec.names.index("z_obs")]),
                        snr=float(abs(saved[i, j]) / errors[i, j]),
                        residual_sigma=float(
                            (decoded[i, j] - saved[i, j]) / errors[i, j]
                        ),
                        residual_relative=float(
                            (decoded[i, j] - saved[i, j])
                            / max(abs(saved[i, j]), errors[i, j])
                        ),
                        reference_flux=float(saved[i, j]),
                        decoded_flux=float(decoded[i, j]),
                        reported_error=float(errors[i, j]),
                    )
                )
    long = pd.DataFrame(rows)
    long.to_csv(out / "residuals.csv", index=False)
    summary = (
        long.groupby(["variant", "band"])
        .residual_sigma.agg(
            median_abs=lambda x: np.median(abs(x)),
            p95_abs=lambda x: np.quantile(abs(x), 0.95),
            mean="mean",
            objects="size",
        )
        .reset_index()
    )
    summary.to_csv(out / "bands.csv", index=False)
    r = bands.index("lsst_r")
    cut = read(Path(source_manifest["parent"]) / "MANIFEST.json")["settings"]["cut"]
    threshold = float(abmag_to_fnu_cgs(cut))
    saved_selection = (
        pd.read_parquet(source_manifest["inputs"]["selection_identities"]["path"])
        .set_index("parent_row_index")
        .loc[ids, "selected"]
        .to_numpy(bool)
    )
    selected = masks[:, r] & (noisy[:, r] > threshold)
    noise_rows = []
    for j, band in enumerate(bands):
        use = masks[:, j]
        scores = (noisy[use, j] - saved[use, j]) / errors[use, j]
        if j == r:
            scores = conditional_normal_score(
                scores,
                (threshold - saved[use, j]) / errors[use, j],
                saved_selection[use],
            )
        noise_rows.append(
            dict(band=band, mean=float(scores.mean()), std=float(scores.std(ddof=1)))
        )
    pd.DataFrame(noise_rows).to_csv(out / "noise.csv", index=False)
    finish(
        out,
        objects=len(ids),
        diagnostic_screen_only=True,
        selection_identity_mismatches=int(
            np.count_nonzero(selected != saved_selection)
        ),
        baseline_p95_max=float(summary[summary.variant == "baseline"].p95_abs.max()),
        merged_p95_max=float(summary[summary.variant == "merged"].p95_abs.max()),
        full_decoder_qualified=False,
    )


def roadmap(root):
    manifest = read(root / "MANIFEST.json")
    settings = manifest["settings"]
    stages = [("ratio_cache", root / "cache"), ("decoder_screen", root / "decoder")]
    stages += [(f"metric_{arm}", root / "metric" / arm) for arm in ARMS]
    total = settings["bootstraps"] * len(ARMS) * len(RATIOS)
    done = sum(
        receipt_valid(root / "bootstrap" / f"repeat_{i:03d}" / f"{arm}_{ratio}")
        for i in range(settings["bootstraps"])
        for arm in ARMS
        for ratio in RATIOS
    )
    rows = [
        dict(
            stage=name,
            state="COMPLETE" if receipt_valid(path) else "PENDING_OR_PARTIAL",
        )
        for name, path in stages
    ]
    rows.append(
        dict(
            stage="paired_bootstrap",
            state="COMPLETE" if done == total else "PENDING_OR_PARTIAL",
            complete=done,
            total=total,
        )
    )
    rows += [
        dict(stage="decoder_full_qualification", state="NOT_VALIDATED"),
        dict(stage="additional_filters", state="WHAT_IF_ONLY"),
        dict(stage="final_parent_posterior", state="NOT_LAUNCHED"),
    ]
    write(root / "ROADMAP_STATUS.json", dict(stages=rows, production_ready=False))
    lines = [
        "# Population roadmap status",
        "",
        f"Run: `{root}`",
        "",
        "| Stage | Execution state |",
        "|---|---|",
    ]
    lines += [
        f"| {r['stage']} | {r['state']}"
        + (f" ({r['complete']}/{r['total']})" if "total" in r else "")
        + " |"
        for r in rows
    ]
    lines += [
        "",
        "Execution completion does not validate a scientific claim.",
        "The oracle sees latent truth; added-band experiments are not implemented.",
    ]
    (root / "ROADMAP_STATUS.md").write_text("\n".join(lines) + "\n")
    return rows


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, settings = config(root)
    out = root / "report"
    state = roadmap(root)
    records = []
    for path in sorted((root / "bootstrap").glob("repeat_*/*/FINAL.json")):
        if receipt_valid(path.parent):
            records.append(read(path))
    bootstrap = pd.DataFrame(records)
    precision_paths = [
        root / "metric" / arm / "precision.csv"
        for arm in ARMS
        if receipt_valid(root / "metric" / arm)
    ]
    precision = (
        pd.concat([pd.read_csv(p) for p in precision_paths], ignore_index=True)
        if precision_paths
        else pd.DataFrame()
    )
    precision.to_csv(out / "metric_precision.csv", index=False)
    if len(bootstrap):
        bootstrap.drop(columns=["artifacts"], errors="ignore").to_csv(
            out / "bootstrap.csv", index=False
        )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if len(precision):
        use = precision[
            (precision.left != precision.right) & (precision.right == "truth")
        ]
        for (arm, ratio), group in use.groupby(["arm", "left"]):
            table = group.groupby("draws").parent_sw.agg(["mean", "std"])
            axes[0].errorbar(
                table.index,
                table["mean"],
                yerr=table["std"],
                marker="o",
                label=f"{arm.split('_')[0]} {ratio}",
            )
        axes[0].set_xscale("log", base=2)
        axes[0].legend(fontsize=7)
    axes[0].set(
        title="Metric precision (seed variation)",
        xlabel="Evaluation draws",
        ylabel="Parent physical SW",
    )
    if len(bootstrap):
        groups = list(bootstrap.groupby(["arm", "ratio"]))
        axes[1].boxplot(
            [g.parent_sw for _, g in groups],
            tick_labels=[f"{a.split('_')[0]}\n{r}\nN={len(g)}" for (a, r), g in groups],
        )
    axes[1].axhline(
        settings["contracts"]["historical_bootstrap_sw"], color="black", ls="--"
    )
    axes[1].set(
        title="Paired bootstrap, fixed models", ylabel="SW to full-catalogue fit"
    )
    if receipt_valid(root / "decoder"):
        table = pd.read_csv(root / "decoder/bands.csv").pivot(
            index="band", columns="variant", values="p95_abs"
        )
        table.plot.bar(ax=axes[2], rot=90)
        axes[2].axhline(
            settings["contracts"]["decoder_p95_abs_sigma"], color="black", ls="--"
        )
    axes[2].set(title="Decoder screen", ylabel="p95 absolute residual / sigma")
    fig.tight_layout()
    fig.savefig(out / "precision_decoder_summary.png", dpi=160)
    plt.close(fig)
    all_complete = all(row["state"] == "COMPLETE" for row in state[:5])
    decisions = dict(execution_complete=all_complete, production_ready=False)
    if len(precision):
        merged = precision.merge(
            precision[precision.draws == precision.draws.max()],
            on=["arm", "seed", "left", "right"],
            suffixes=("", "_largest"),
        )
        delta = float(
            abs(
                merged[merged.draws == min(settings["metric_draws"])].parent_sw
                - merged[
                    merged.draws == min(settings["metric_draws"])
                ].parent_sw_largest
            ).max()
        )
        decisions["max_4096_to_largest_delta"] = delta
        decisions["metric_resolved_at_4096"] = (
            delta <= settings["contracts"]["metric_resolution_tolerance"]
        )
    if len(bootstrap):
        summary = bootstrap.groupby(["arm", "ratio"]).parent_sw.agg(
            ["count", "median", "max"]
        )
        summary.to_csv(out / "bootstrap_summary.csv")
    write(
        out / "FINAL.json",
        dict(status="COMPLETE" if all_complete else "PARTIAL", decisions=decisions),
    )
    (out / "REPORT.md").write_text(
        "# Precision and decoder audit\n\n"
        + f"Execution: {'complete' if all_complete else 'partial; missing cells remain missing'}.\n\n"
        + "See `precision_decoder_summary.png`, `bootstrap_summary.csv` and `metric_precision.csv`.\n\n"
        + "The same catalogue bootstrap counts are used for exact and learned ratios. "
        + "Rank and penalty are fixed from the previous heldout fit. Alpha and classifier uncertainty "
        + "are not resampled. Eight replicates are an initial diagnostic, not publication certification.\n\n"
        + "Oracle success does not distinguish intrinsic photometric degeneracy from classifier error. "
        + "The decoder check is a bounded screen on 128 fixed objects; full qualification remains separate.\n\n"
        + "No production prior, posterior or extra-band experiment was trained.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "cache",
            "metric",
            "bootstrap",
            "decoder",
            "report",
            "status",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(args.source.resolve(), root, args.config.resolve())
    elif args.mode == "status":
        print(json.dumps(roadmap(root), indent=2))
    else:
        try:
            if args.mode in ("metric", "bootstrap"):
                globals()[args.mode](root, args.task)
            else:
                globals()[args.mode](root)
        except Exception as error:
            write(
                root / "logs" / f"failure_{args.mode}_{args.task}.json",
                dict(error=repr(error), mode=args.mode, task=args.task),
            )
            raise


if __name__ == "__main__":
    main()
