"""Read-only audit of decoder, observation, nuisance, posterior and population failure modes."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_avi_experiments import read, sha, write

TASKS = (
    "decoder_observation",
    "nuisance_sensitivity",
    "posterior_original",
    "posterior_continued",
    "population_identifiability",
)


def _weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("weighted quantiles require finite nonnegative weights")
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(np.asarray(quantiles), positions, values)


def _status(path, expected):
    payload = read(path)
    if payload.get("status") != expected:
        raise ValueError(f"incomplete input {path}: {payload.get('status')}")
    return payload


def _selection_identities(parent_manifest):
    return (
        Path(parent_manifest["blind_truth_parent"]).parent
        / "selection_identities.parquet"
    )


def _stratified_indices(truth, selection, count, seed):
    frame = truth[["parent_row_index", "z_obs"]].merge(
        selection[["parent_row_index", "selected"]],
        on="parent_row_index",
        validate="one_to_one",
    )
    frame["z_bin"] = pd.cut(frame.z_obs, np.linspace(0.0, 5.5, 12), include_lowest=True)
    groups = list(frame.groupby(["selected", "z_bin"], observed=True))
    rng = np.random.default_rng(seed)
    chosen = []
    quota = max(1, count // max(len(groups), 1))
    for _, group in groups:
        chosen.extend(rng.choice(group.index, min(quota, len(group)), replace=False))
    remaining = np.setdiff1d(frame.index, np.asarray(chosen, dtype=int))
    if len(chosen) < min(count, len(frame)):
        chosen.extend(
            rng.choice(
                remaining, min(count - len(chosen), len(remaining)), replace=False
            )
        )
    result = frame.loc[chosen[:count], "parent_row_index"].to_numpy(np.int64)
    if len(result) != min(count, len(frame)) or len(np.unique(result)) != len(result):
        raise ValueError("failed to construct unique stratified audit cohort")
    return result


def prepare(parent, diagnostics, root, config):
    if root.exists():
        raise FileExistsError(root)
    spec = yaml.safe_load(config.read_text())
    parent_manifest = read(parent / "MANIFEST.json")
    _status(parent / "population/FINAL.json", "FORWARD_PARENT_COMPLETE")
    _status(parent / "posterior/FINAL.json", "FORWARD_POSTERIOR_COMPLETE")
    _status(parent / "report/FINAL.json", "FORWARD_POPULATION_REPORT_COMPLETE")
    _status(diagnostics / "FINAL.json", "FORWARD_DIAGNOSTICS_COMPLETE")
    refined = _status(
        diagnostics / "classifier/population/FINAL.json", "FORWARD_PARENT_COMPLETE"
    )
    continued = _status(
        diagnostics / "posterior/posterior/FINAL.json", "FORWARD_POSTERIOR_COMPLETE"
    )
    for path, digest in (
        (
            parent / "posterior/best.eqx",
            read(parent / "posterior/FINAL.json")["checkpoint_sha256"],
        ),
        (diagnostics / "posterior/posterior/best.eqx", continued["checkpoint_sha256"]),
        (diagnostics / "classifier/population/parent.json", refined["parent_sha256"]),
        (diagnostics / "classifier/population/best.eqx", refined["classifier_sha256"]),
    ):
        if sha(path) != digest:
            raise ValueError(f"input integrity failed: {path}")
    truth_path = Path(parent_manifest["blind_truth_parent"])
    selection_path = _selection_identities(parent_manifest)
    truth = pd.read_parquet(truth_path)
    selection = pd.read_parquet(selection_path)
    indices = _stratified_indices(
        truth,
        selection,
        max(spec["decoder_objects"], spec["nuisance_objects"]),
        spec["seed"],
    )
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    for task in TASKS:
        (root / task).mkdir()
    np.save(root / "audit_indices.npy", indices)
    shutil.copy2(config, root / "audit.yaml")
    inputs = {
        "parent_manifest": parent / "MANIFEST.json",
        "parent_checkpoint": parent / "posterior/best.eqx",
        "continued_checkpoint": diagnostics / "posterior/posterior/best.eqx",
        "refined_parent": diagnostics / "classifier/population/parent.json",
        "refined_ratios": diagnostics / "classifier/population/observed_ratios.npz",
        "truth_parent": truth_path,
        "selection_identities": selection_path,
        "audit_indices": root / "audit_indices.npy",
        "audit_config": root / "audit.yaml",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    hashes = {
        name: {"path": str(path.resolve()), "sha256": sha(path)}
        for name, path in inputs.items()
    }
    write(
        root / "MANIFEST.json",
        dict(
            parent=str(parent.resolve()),
            diagnostics=str(diagnostics.resolve()),
            source=parent_manifest["source"],
            settings=spec,
            inputs=hashes,
            tasks=list(TASKS),
            read_only_inputs=True,
            truth_role="diagnostic closure only",
        ),
    )
    print(
        json.dumps(
            dict(tasks=list(TASKS), resources=spec["resources"], new_training=0),
            indent=2,
        )
    )


def contract(root):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"audit input changed: {path}")
    return manifest, manifest["settings"]


def _runtime(manifest, settings, destination):
    from euclid_dsps.amortized.forward_population_runtime import load_forward_runtime

    parent_manifest = read(Path(manifest["parent"]) / "MANIFEST.json")
    return load_forward_runtime(
        Path(manifest["source"]), destination, parent_manifest["settings"]
    )


def _catalogue(manifest, bands, indices):
    source_manifest = read(Path(manifest["source"]) / "MANIFEST.json")
    catalogue = Path(source_manifest["validation_catalog"])
    names = []
    for band in bands:
        names.extend(
            [f"flux_true_{band}", f"flux_{band}", f"fluxerr_{band}", f"mask_{band}"]
        )
    frame = (
        pd.read_parquet(catalogue, columns=list(dict.fromkeys(names)))
        .iloc[indices]
        .copy()
    )
    frame.insert(0, "parent_row_index", indices)
    return frame


def _decode(model, runtime, x, batch=128):
    import equinox as eqx
    import jax.numpy as jnp

    from euclid_dsps.amortized.decoder import model_flux_from_x
    from euclid_dsps.amortized.posterior_target import (
        _apply_model_calibration,
        safe_decoder_inputs,
    )

    @eqx.filter_jit
    def one(value):
        safe, valid = safe_decoder_inputs(value, runtime.latent_spec)
        flux = model_flux_from_x(
            safe,
            runtime.latent_spec,
            runtime.context,
            runtime.model_args,
            runtime.parameter_names,
        )
        return _apply_model_calibration(model, flux, runtime.calibration_config), valid

    fluxes, valid = [], []
    for start in range(0, len(x), batch):
        f, v = one(jnp.asarray(x[start : start + batch]))
        fluxes.append(np.asarray(f))
        valid.append(np.asarray(v))
    return np.concatenate(fluxes), np.concatenate(valid)


def _truth_x(manifest, runtime, indices):
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import theta_to_x

    truth = pd.read_parquet(Path(manifest["inputs"]["truth_parent"]["path"])).set_index(
        "parent_row_index"
    )
    theta = truth.loc[indices, list(runtime.latent_spec.names)].to_numpy()
    return theta, np.asarray(theta_to_x(jnp.asarray(theta), runtime.latent_spec))


def decoder_observation(root):
    from scipy.stats import kstest

    from euclid_dsps.photometry import abmag_to_fnu_cgs

    manifest, settings = contract(root)
    out = root / "decoder_observation"
    model, runtime, observation, _ = _runtime(manifest, settings, out / "runtime")
    bands = tuple(runtime.feature_stats.band_names)
    parent_settings = read(Path(manifest["parent"]) / "MANIFEST.json")["settings"]
    indices = np.load(manifest["inputs"]["audit_indices"]["path"])[
        : settings["decoder_objects"]
    ]
    _, x = _truth_x(manifest, runtime, indices)
    decoded, valid = _decode(model, runtime, x)
    if not valid.all():
        raise ValueError("truth decoder cohort contains invalid parameters")
    catalogue = _catalogue(manifest, bands, indices)
    saved_true = catalogue[[f"flux_true_{b}" for b in bands]].to_numpy()
    noisy = catalogue[[f"flux_{b}" for b in bands]].to_numpy()
    reported = catalogue[[f"fluxerr_{b}" for b in bands]].to_numpy()
    masks = catalogue[[f"mask_{b}" for b in bands]].to_numpy(bool)
    from euclid_dsps.amortized.train import _sleep_m5_flux_error

    recomputed_true = np.asarray(_sleep_m5_flux_error(saved_true, observation))
    recomputed_decoded = np.asarray(_sleep_m5_flux_error(decoded, observation))
    scale = np.maximum(reported, 1e-40)
    records = []
    for j, band in enumerate(bands):
        good = masks[:, j] & np.isfinite(scale[:, j]) & (scale[:, j] > 0)
        z = (noisy[good, j] - saved_true[good, j]) / scale[good, j]
        for row, i in enumerate(np.flatnonzero(good)):
            records.append(
                dict(
                    parent_row_index=int(indices[i]),
                    band=band,
                    decoder_residual_sigma=(decoded[i, j] - saved_true[i, j])
                    / scale[i, j],
                    decoder_relative_residual=(decoded[i, j] - saved_true[i, j])
                    / max(abs(saved_true[i, j]), scale[i, j], 1e-40),
                    error_relative_residual_true=(
                        recomputed_true[i, j] - reported[i, j]
                    )
                    / scale[i, j],
                    error_relative_residual_decoded=(
                        recomputed_decoded[i, j] - reported[i, j]
                    )
                    / scale[i, j],
                    standardized_noise=z[row],
                )
            )
    long = pd.DataFrame(records)
    long.to_parquet(out / "per_object_band.parquet", index=False)
    summary = []
    for band, group in long.groupby("band", sort=False):
        summary.append(
            dict(
                band=band,
                objects=len(group),
                decoder_median_abs_sigma=np.median(abs(group.decoder_residual_sigma)),
                decoder_p95_abs_sigma=np.quantile(
                    abs(group.decoder_residual_sigma), 0.95
                ),
                decoder_p99_abs_sigma=np.quantile(
                    abs(group.decoder_residual_sigma), 0.99
                ),
                error_true_p95_abs_relative=np.quantile(
                    abs(group.error_relative_residual_true), 0.95
                ),
                error_decoded_p95_abs_relative=np.quantile(
                    abs(group.error_relative_residual_decoded), 0.95
                ),
                noise_mean=group.standardized_noise.mean(),
                noise_std=group.standardized_noise.std(ddof=1),
                noise_normal_ks=kstest(group.standardized_noise, "norm").statistic,
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "band_summary.csv", index=False)
    r = bands.index("lsst_r")
    threshold = float(abmag_to_fnu_cgs(parent_settings["cut"]))
    selection = pd.read_parquet(
        Path(manifest["inputs"]["selection_identities"]["path"]),
        columns=["parent_row_index", "selected"],
    ).set_index("parent_row_index")
    saved_selection = selection.loc[indices, "selected"].to_numpy(bool)
    recomputed_selection = masks[:, r] & (noisy[:, r] > threshold)
    write(
        out / "FINAL.json",
        dict(
            status="DECODER_OBSERVATION_AUDIT_COMPLETE",
            objects=len(indices),
            bands=len(bands),
            selection_identity_mismatches=int(
                np.sum(saved_selection != recomputed_selection)
            ),
            max_decoder_p95_abs_sigma=float(summary.decoder_p95_abs_sigma.max()),
            max_error_true_p95_abs_relative=float(
                summary.error_true_p95_abs_relative.max()
            ),
            max_noise_mean_abs=float(abs(summary.noise_mean).max()),
            max_noise_std_abs_from_one=float(abs(summary.noise_std - 1).max()),
        ),
    )


def nuisance_sensitivity(root):

    from euclid_dsps.amortized.train import _sleep_m5_flux_error
    from euclid_dsps.photometry import abmag_to_fnu_cgs

    manifest, settings = contract(root)
    out = root / "nuisance_sensitivity"
    model, runtime, observation, _ = _runtime(manifest, settings, out / "runtime")
    bands = tuple(runtime.feature_stats.band_names)
    indices = np.load(manifest["inputs"]["audit_indices"]["path"])[
        : settings["nuisance_objects"]
    ]
    theta, x = _truth_x(manifest, runtime, indices)
    rng = np.random.default_rng(settings["seed"] + 1)
    physical = np.array(
        [
            runtime.latent_spec.names.index(name)
            for name in (
                "z_obs",
                "log10_stellar_mass",
                "log10_stellar_metallicity",
                "dust_av",
                "dust_delta",
            )
        ]
    )
    nuisance = np.setdiff1d(np.arange(x.shape[1]), physical)
    reference = np.repeat(x[None], settings["nuisance_replicates"], axis=0)
    reference[:, :, nuisance] = rng.normal(
        size=(settings["nuisance_replicates"], len(x), len(nuisance))
    )
    true_flux, valid_true = _decode(model, runtime, x)
    reference_flux, valid_reference = _decode(
        model, runtime, reference.reshape(-1, x.shape[1])
    )
    reference_flux = reference_flux.reshape(settings["nuisance_replicates"], len(x), -1)
    if not (valid_true.all() and valid_reference.all()):
        raise ValueError("nuisance sensitivity contains invalid decoded parameters")
    errors = np.asarray(_sleep_m5_flux_error(true_flux, observation))
    shift = (reference_flux - true_flux[None]) / np.maximum(errors[None], 1e-40)
    eps = rng.normal(size=(settings["noise_replicates"], *true_flux.shape))
    r = bands.index("lsst_r")
    threshold = float(
        abmag_to_fnu_cgs(
            read(Path(manifest["parent"]) / "MANIFEST.json")["settings"]["cut"]
        )
    )
    ref_errors = np.asarray(
        _sleep_m5_flux_error(
            reference_flux.reshape(-1, true_flux.shape[1]), observation
        )
    ).reshape(reference_flux.shape)
    true_selected = (
        true_flux[None, :, r] + eps[:, :, r] * errors[None, :, r]
    ) > threshold
    ref_selected = (
        reference_flux[None, :, :, r] + eps[:, None, :, r] * ref_errors[None, :, :, r]
    ) > threshold
    rows = []
    for j, band in enumerate(bands):
        rows.append(
            dict(
                band=band,
                median_abs_flux_shift_sigma=np.median(abs(shift[:, :, j])),
                p90_abs_flux_shift_sigma=np.quantile(abs(shift[:, :, j]), 0.90),
                p99_abs_flux_shift_sigma=np.quantile(abs(shift[:, :, j]), 0.99),
                fraction_shift_gt_1sigma=np.mean(abs(shift[:, :, j]) > 1),
            )
        )
    pd.DataFrame(rows).to_csv(out / "band_sensitivity.csv", index=False)
    pd.DataFrame(
        dict(
            parent_row_index=indices,
            true_selection_probability=true_selected.mean(axis=0),
            reference_nuisance_selection_probability=ref_selected.mean(axis=(0, 1)),
        )
    ).to_parquet(out / "per_object_selection.parquet", index=False)
    z_edges = np.asarray(settings["conditional_bins"]["z_obs"])
    z_bin = pd.cut(theta[:, physical[0]], z_edges, include_lowest=True)
    selection_by_z = []
    for label in z_bin.categories:
        in_bin = np.asarray(z_bin == label)
        if not in_bin.any():
            continue
        true_probability = float(true_selected[:, in_bin].mean())
        reference_probability = float(ref_selected[:, :, in_bin].mean())
        selection_by_z.append(
            dict(
                z_bin=str(label),
                objects=int(in_bin.sum()),
                true_probability=true_probability,
                reference_probability=reference_probability,
                abs_difference=abs(true_probability - reference_probability),
            )
        )
    selection_by_z = pd.DataFrame(selection_by_z)
    selection_by_z.to_csv(out / "selection_by_redshift.csv", index=False)
    true_population_probability = float(true_selected.mean())
    reference_population_probability = float(ref_selected.mean())
    write(
        out / "FINAL.json",
        dict(
            status="NUISANCE_SENSITIVITY_COMPLETE",
            objects=len(indices),
            nuisance_replicates=settings["nuisance_replicates"],
            noise_replicates=settings["noise_replicates"],
            true_population_selection_probability=true_population_probability,
            reference_population_selection_probability=reference_population_probability,
            population_selection_probability_abs_shift=abs(
                true_population_probability - reference_population_probability
            ),
            max_zbin_selection_probability_abs_shift=float(
                selection_by_z.abs_difference.max()
            ),
            fraction_selection_probability_shift_gt_0p1=float(
                np.mean(
                    abs(true_selected.mean(axis=0) - ref_selected.mean(axis=(0, 1)))
                    > 0.1
                )
            ),
            max_band_p90_abs_flux_shift_sigma=float(
                np.max(np.quantile(abs(shift), 0.90, axis=(0, 1)))
            ),
            interpretation="oracle sensitivity only; not a production nuisance estimator",
        ),
    )


def _posterior_metrics(draws, truth, metadata, names, cohort, arm):
    from euclid_dsps.amortized.forward_population import PHYSICAL

    q16, med, q84 = np.quantile(draws, [0.16, 0.5, 0.84], axis=1)
    q025, q975 = np.quantile(draws, [0.025, 0.975], axis=1)
    rank = np.mean(draws <= truth[:, None, :], axis=1)
    rows = []
    for i in range(len(truth)):
        for j, name in enumerate(names):
            rows.append(
                dict(
                    arm=arm,
                    cohort=cohort,
                    object_index=i,
                    parameter=name,
                    group="physical" if name in PHYSICAL else "sfh",
                    truth=truth[i, j],
                    posterior_median=med[i, j],
                    median_bias=med[i, j] - truth[i, j],
                    width68=q84[i, j] - q16[i, j],
                    width95=q975[i, j] - q025[i, j],
                    coverage68=(q16[i, j] <= truth[i, j] <= q84[i, j]),
                    coverage95=(q025[i, j] <= truth[i, j] <= q975[i, j]),
                    rank=rank[i, j],
                    **{key: value[i] for key, value in metadata.items()},
                )
            )
    return pd.DataFrame(rows)


def _metadata(flux, errors, truth_z, bands, cut):
    from euclid_dsps.photometry import abmag_to_fnu_cgs

    r = bands.index("lsst_r")
    f = flux[:, r]
    e = errors[:, r]
    threshold = float(abmag_to_fnu_cgs(cut))
    magnitude = -2.5 * np.log10(np.maximum(f, np.finfo(float).tiny)) - 48.6
    return dict(
        z_obs=np.asarray(truth_z),
        r_mag=magnitude,
        log10_snr=np.log10(np.maximum(abs(f) / e, 1e-6)),
        selection_margin_sigma=(f - threshold) / e,
    )


def posterior_evaluation(root, continued):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.data import load_photometry_arrays_from_config
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.forward_population_runtime import posterior_template
    from euclid_dsps.amortized.latent import x_to_theta
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from scripts.feniks_forward_population import attach_frozen_parent, load_banks

    manifest, settings = contract(root)
    arm = "continued" if continued else "original"
    out = root / f"posterior_{arm}"
    parent = Path(manifest["parent"])
    parent_manifest = read(parent / "MANIFEST.json")
    model, runtime, _, config = _runtime(manifest, settings, out / "runtime")
    model = attach_frozen_parent(parent, model)
    candidate = posterior_template(model, runtime, config, parent_manifest["settings"])
    checkpoint = (
        Path(manifest["diagnostics"]) / "posterior/posterior/best.eqx"
        if continued
        else parent / "posterior/best.eqx"
    )
    candidate = eqx.tree_deserialise_leaves(checkpoint, candidate)
    draws_count = settings["posterior_draws"]

    @eqx.filter_jit
    def infer(features, key):
        x = sample_independent_mixture(model, candidate, key, features, draws_count).x
        return jnp.swapaxes(x_to_theta(x, runtime.latent_spec), 0, 1)

    def draw_all(features, seed):
        parts = []
        for start in range(0, len(features), settings["posterior_batch"]):
            part = np.asarray(
                infer(
                    jnp.asarray(features[start : start + settings["posterior_batch"]]),
                    jax.random.fold_in(jax.random.PRNGKey(seed), start),
                )
            )
            parts.append(part)
        return np.concatenate(parts)

    cfg = parent_manifest["settings"]
    bank = load_banks(
        parent,
        "posterior_bank",
        [cfg["posterior_shards"]],
        ("theta", "features", "selected", "flux", "errors"),
    )
    candidates = np.flatnonzero(
        bank["selected"]
        & (np.arange(len(bank["selected"])) >= len(bank["selected"]) // 2)
    )[: settings["posterior_simulation_objects"]]
    if len(candidates) < settings["posterior_simulation_objects"]:
        raise ValueError(
            f"only {len(candidates)} independent selected simulations; "
            f"requested {settings['posterior_simulation_objects']}"
        )
    sim_truth = bank["theta"][candidates]
    sim_draws = draw_all(bank["features"][candidates], settings["seed"] + 200)
    bands = tuple(runtime.feature_stats.band_names)
    sim = _posterior_metrics(
        sim_draws,
        sim_truth,
        _metadata(
            bank["flux"][candidates],
            bank["errors"][candidates],
            sim_truth[:, 0],
            bands,
            cfg["cut"],
        ),
        runtime.latent_spec.names,
        "simulation",
        arm,
    )
    keep = min(settings["posterior_persist_objects"], len(candidates))
    np.savez(
        out / "simulation_dense_posteriors.npz",
        draws=sim_draws[:keep].astype(np.float32),
        truth=sim_truth[:keep],
        bank_row_index=candidates[:keep],
        names=np.asarray(runtime.latent_spec.names),
    )
    del sim_draws
    indices = np.load(Path(manifest["diagnostics"]) / "evaluation_indices.npy")[
        : settings["posterior_catalogue_objects"]
    ]
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10000, row_indices=indices
    )
    if not np.array_equal(arrays.row_index, indices):
        raise ValueError("catalogue evaluation identity mismatch")
    features = np.asarray(
        make_encoder_features(
            arrays.flux, arrays.flux_err, runtime.feature_stats, arrays.mask
        )
    )
    catalogue_draws = draw_all(features, settings["seed"] + 300)
    truth_frame = pd.read_parquet(
        Path(read(Path(manifest["source"]) / "MANIFEST.json")["validation_catalog"]),
        columns=list(runtime.latent_spec.names),
    )
    catalogue_truth = truth_frame.iloc[indices].to_numpy()
    observed = _posterior_metrics(
        catalogue_draws,
        catalogue_truth,
        _metadata(
            arrays.flux, arrays.flux_err, catalogue_truth[:, 0], bands, cfg["cut"]
        ),
        runtime.latent_spec.names,
        "catalogue",
        arm,
    )
    keep = min(settings["posterior_persist_objects"], len(indices))
    np.savez(
        out / "catalogue_dense_posteriors.npz",
        draws=catalogue_draws[:keep].astype(np.float32),
        truth=catalogue_truth[:keep],
        row_index=indices[:keep],
        names=np.asarray(runtime.latent_spec.names),
    )
    del catalogue_draws
    frame = pd.concat([sim, observed], ignore_index=True)
    frame.to_parquet(out / "per_object_parameter.parquet", index=False)
    summary = (
        frame.groupby(["arm", "cohort", "group", "parameter"], sort=False)
        .agg(
            objects=("object_index", "size"),
            coverage68=("coverage68", "mean"),
            coverage95=("coverage95", "mean"),
            median_abs_bias=("median_bias", lambda x: np.median(abs(x))),
            median_width68=("width68", "median"),
            median_width95=("width95", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out / "marginal_summary.csv", index=False)
    conditional = []
    physical = frame[frame.group == "physical"]
    for variable, edges in settings["conditional_bins"].items():
        values = physical[variable]
        labels = pd.cut(values, edges, include_lowest=True)
        grouped = physical.assign(bin=labels).groupby(
            ["arm", "cohort", "parameter", "bin"], observed=True
        )
        part = grouped.agg(
            objects=("object_index", "size"),
            coverage68=("coverage68", "mean"),
            coverage95=("coverage95", "mean"),
            median_width68=("width68", "median"),
            median_abs_bias=("median_bias", lambda x: np.median(abs(x))),
        ).reset_index()
        part.insert(3, "conditioning_variable", variable)
        conditional.append(part)
    pd.concat(conditional, ignore_index=True).to_csv(
        out / "conditional_calibration.csv", index=False
    )
    write(
        out / "FINAL.json",
        dict(
            status="POSTERIOR_CONDITIONAL_AUDIT_COMPLETE",
            arm=arm,
            draws_per_object=draws_count,
            simulation_objects=len(candidates),
            catalogue_objects=len(indices),
            checkpoint_sha256=sha(checkpoint),
            dense_draws_persisted=int(settings["posterior_persist_objects"]),
        ),
    )


def population_identifiability(root):
    import jax.numpy as jnp
    from scipy.stats import wasserstein_distance

    from euclid_dsps.amortized.forward_population import (
        PHYSICAL,
        PhysicalBasis,
        fit_selected_weights,
        parent_from_selected,
    )
    from euclid_dsps.amortized.latent import x_to_theta

    manifest, settings = contract(root)
    out = root / "population_identifiability"
    diagnostics = Path(manifest["diagnostics"])
    parent = read(diagnostics / "classifier/population/parent.json")
    with np.load(diagnostics / "classifier/population/observed_ratios.npz") as saved:
        logc = saved["log_classifier"]
        row_index = saved["row_index"]
    c, alpha = np.asarray(parent["c"]), np.asarray(parent["alpha"])
    component_selection = pd.read_csv(
        diagnostics / "classifier/population/component_selection.csv"
    )
    eligible = component_selection.eligible.to_numpy(bool)
    with np.load(Path(manifest["parent"]) / "basis.npz") as saved:
        names = tuple(saved["names"].tolist())
        basis = PhysicalBasis(names, saved["centers"], saved["scales"])
    truth = pd.read_parquet(Path(manifest["inputs"]["truth_parent"]["path"]))
    truth_values = truth[list(names)].to_numpy()
    _, runtime, _, _ = _runtime(manifest, settings, out / "runtime")

    def to_theta(x):
        return np.asarray(x_to_theta(jnp.asarray(x), runtime.latent_spec))

    selection = pd.read_parquet(
        Path(manifest["inputs"]["selection_identities"]["path"]),
        columns=["parent_row_index", "selected"],
    ).set_index("parent_row_index")
    selected = selection.loc[truth.parent_row_index, "selected"].to_numpy(bool)
    truth_weight = truth.population_weight.to_numpy(np.float64)
    empirical_alpha = float(selected.mean())
    weighted_empirical_alpha = float(np.average(selected, weights=truth_weight))
    rng = np.random.default_rng(settings["seed"] + 4)
    rows, weights = [], []
    for replicate in range(settings["population_bootstraps"]):
        count = rng.multinomial(len(logc), np.full(len(logc), 1 / len(logc)))
        v, certificate = fit_selected_weights(
            logc,
            c,
            alpha=alpha,
            eligible=eligible,
            weak_parent_mass=read(Path(manifest["parent"]) / "MANIFEST.json")[
                "settings"
            ]["weak_parent_mass_cap"],
            observation_weights=count,
        )
        u = parent_from_selected(v, alpha)
        draw_x, _ = basis.sample(
            rng, settings["population_draws_per_bootstrap"], weights=u
        )
        draw = to_theta(draw_x)
        weights.append(u)
        for name in PHYSICAL:
            j = names.index(name)
            q25, q75 = _weighted_quantile(
                truth_values[:, j], [0.25, 0.75], truth_weight
            )
            scale = q75 - q25
            rows.append(
                dict(
                    replicate=replicate,
                    parameter=name,
                    w1_over_truth_iqr=wasserstein_distance(
                        draw[:, j], truth_values[:, j], v_weights=truth_weight
                    )
                    / max(scale, 1e-8),
                    mean=draw[:, j].mean(),
                    alpha=float(u @ alpha),
                    kkt_gap=certificate["kkt_gap"],
                )
            )
    weights = np.asarray(weights)
    np.savez(out / "bootstrap_weights.npz", u=weights, row_index=row_index)
    pd.DataFrame(rows).to_csv(out / "bootstrap_metrics.csv", index=False)
    v = np.asarray(parent["v"])
    logd = logc - np.log(c)
    logd -= logd.max(axis=1, keepdims=True)
    d = np.exp(np.maximum(logd, -700))
    denom = np.maximum(d @ v, 1e-300)
    hessian = (d / denom[:, None]).T @ (d / denom[:, None]) / len(d)
    q = np.eye(len(v)) - np.ones((len(v), len(v))) / len(v)
    eigenvalues = np.linalg.eigvalsh(q @ hessian @ q)
    positive = eigenvalues[eigenvalues > max(eigenvalues.max(), 0.0) * 1e-10]
    if not len(positive):
        raise ValueError(
            "selected-mixture Hessian has no identifiable tangent direction"
        )
    known = pd.read_csv(diagnostics / "classifier/population/simulation_closure.csv")
    true_draw_x, _ = basis.sample(rng, 100000, weights=known.true_parent.to_numpy())
    inferred_draw_x, _ = basis.sample(
        rng, 100000, weights=known.inferred_parent.to_numpy()
    )
    true_draw, inferred_draw = to_theta(true_draw_x), to_theta(inferred_draw_x)
    known_rows = []
    for name in PHYSICAL:
        j = names.index(name)
        scale = np.quantile(true_draw[:, j], 0.75) - np.quantile(true_draw[:, j], 0.25)
        known_rows.append(
            dict(
                parameter=name,
                w1_over_true_iqr=wasserstein_distance(
                    true_draw[:, j], inferred_draw[:, j]
                )
                / max(scale, 1e-8),
            )
        )
    pd.DataFrame(known_rows).to_csv(
        out / "known_mixture_density_closure.csv", index=False
    )
    write(
        out / "FINAL.json",
        dict(
            status="POPULATION_IDENTIFIABILITY_COMPLETE",
            bootstrap_replicates=len(weights),
            median_weight_l1_from_fit=float(
                np.median(np.abs(weights - np.asarray(parent["u"])).sum(axis=1))
            ),
            max_weight_l1_from_fit=float(
                np.max(np.abs(weights - np.asarray(parent["u"])).sum(axis=1))
            ),
            tangent_identifiable_rank=int(len(positive)),
            tangent_curvature_condition=float(positive.max() / positive.min()),
            predicted_parent_alpha=float(np.asarray(parent["u"]) @ alpha),
            empirical_parent_alpha=empirical_alpha,
            weighted_empirical_parent_alpha=weighted_empirical_alpha,
            weighted_alpha_abs_error=float(
                abs(np.asarray(parent["u"]) @ alpha - weighted_empirical_alpha)
            ),
            known_mixture_weight_l1=float(
                np.abs(known.true_parent - known.inferred_parent).sum()
            ),
            known_mixture_max_physical_w1_over_iqr=float(
                pd.DataFrame(known_rows).w1_over_true_iqr.max()
            ),
        ),
    )


def report(root):
    import matplotlib.pyplot as plt

    manifest, settings = contract(root)
    finals = {task: read(root / task / "FINAL.json") for task in TASKS}
    decoder = pd.read_csv(root / "decoder_observation/band_summary.csv")
    nuisance = pd.read_csv(root / "nuisance_sensitivity/band_sensitivity.csv")
    conditional = pd.concat(
        [
            pd.read_csv(root / f"posterior_{arm}/conditional_calibration.csv")
            for arm in ("original", "continued")
        ],
        ignore_index=True,
    )
    bootstrap = pd.read_csv(root / "population_identifiability/bootstrap_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(decoder.band, decoder.decoder_p95_abs_sigma)
    axes[0, 0].axhline(
        settings["gates"]["decoder_p95_abs_residual_sigma"], color="black", ls="--"
    )
    axes[0, 0].set(title="Decoder residual: p95 |delta flux| / sigma", ylabel="sigma")
    axes[0, 0].tick_params(axis="x", rotation=90)
    axes[0, 1].bar(nuisance.band, nuisance.p90_abs_flux_shift_sigma)
    axes[0, 1].set(title="Replace nuisance conditional: p90 flux shift", ylabel="sigma")
    axes[0, 1].tick_params(axis="x", rotation=90)
    z = conditional[
        (conditional.parameter == "z_obs")
        & (conditional.conditioning_variable == "z_obs")
        & (conditional.cohort == "catalogue")
    ]
    for arm, frame in z.groupby("arm"):
        axes[1, 0].plot(frame.bin.astype(str), frame.coverage68, marker="o", label=arm)
    axes[1, 0].axhline(0.68, color="black", ls="--")
    axes[1, 0].set(title="Catalogue redshift coverage by true z", ylabel="Coverage 68%")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend()
    b = bootstrap[bootstrap.parameter == "z_obs"]
    axes[1, 1].hist(b.w1_over_truth_iqr, bins=12, histtype="step")
    axes[1, 1].set(title="Bootstrap learned-parent redshift W1/IQR")
    fig.tight_layout()
    fig.savefig(root / "failure_modes_summary.png", dpi=160)
    plt.close(fig)
    gates = settings["gates"]
    original = pd.read_csv(root / "posterior_original/marginal_summary.csv")
    continued = pd.read_csv(root / "posterior_continued/marginal_summary.csv")

    def coverage_pass(frame, cohort, level, tolerance):
        values = frame[(frame.cohort == cohort) & (frame.group == "physical")][
            f"coverage{level}"
        ]
        return bool(
            len(values) == 5
            and np.isfinite(values).all()
            and np.all(abs(values - level / 100) <= tolerance)
        )

    checks = dict(
        decoder=bool(
            (
                decoder.decoder_p95_abs_sigma <= gates["decoder_p95_abs_residual_sigma"]
            ).all()
        ),
        selection_identity=bool(
            finals["decoder_observation"]["selection_identity_mismatches"] == 0
        ),
        error_contract=bool(
            (
                decoder.error_true_p95_abs_relative
                <= gates["error_p95_relative_residual"]
            ).all()
        ),
        noise_contract=bool(
            (abs(decoder.noise_mean) <= gates["noise_mean_abs"]).all()
            and (abs(decoder.noise_std - 1) <= gates["noise_std_abs_from_one"]).all()
        ),
        nuisance_conditional=bool(
            finals["nuisance_sensitivity"]["population_selection_probability_abs_shift"]
            <= gates["nuisance_population_selection_abs"]
            and finals["nuisance_sensitivity"][
                "max_zbin_selection_probability_abs_shift"
            ]
            <= gates["nuisance_zbin_selection_abs"]
        ),
        original_simulation_68=coverage_pass(
            original, "simulation", 68, gates["simulation_coverage68_abs"]
        ),
        original_simulation_95=coverage_pass(
            original, "simulation", 95, gates["simulation_coverage95_abs"]
        ),
        continued_simulation_68=coverage_pass(
            continued, "simulation", 68, gates["simulation_coverage68_abs"]
        ),
        continued_simulation_95=coverage_pass(
            continued, "simulation", 95, gates["simulation_coverage95_abs"]
        ),
        original_catalogue_68=coverage_pass(
            original, "catalogue", 68, gates["catalogue_coverage68_abs"]
        ),
        original_catalogue_95=coverage_pass(
            original, "catalogue", 95, gates["catalogue_coverage95_abs"]
        ),
        continued_catalogue_68=coverage_pass(
            continued, "catalogue", 68, gates["catalogue_coverage68_abs"]
        ),
        continued_catalogue_95=coverage_pass(
            continued, "catalogue", 95, gates["catalogue_coverage95_abs"]
        ),
        parent_alpha=bool(
            finals["population_identifiability"]["weighted_alpha_abs_error"]
            <= gates["alpha_abs"]
        ),
        classifier_density_closure=bool(
            finals["population_identifiability"][
                "known_mixture_max_physical_w1_over_iqr"
            ]
            <= gates["known_mixture_max_w1_over_iqr"]
        ),
        population_bootstrap_stability=bool(
            bootstrap.groupby("replicate").alpha.first().std(ddof=1)
            <= gates["bootstrap_alpha_std"]
        ),
    )
    failure_modes = []
    if not all(
        checks[key]
        for key in ("decoder", "selection_identity", "error_contract", "noise_contract")
    ):
        failure_modes.append("decoder_or_observation_contract")
    if not checks["nuisance_conditional"]:
        failure_modes.append("parent_nuisance_selection_contract")
    if not checks["classifier_density_closure"]:
        failure_modes.append("classifier_ratio_or_component_identifiability")
    if not checks["parent_alpha"] or not checks["population_bootstrap_stability"]:
        failure_modes.append("parent_selection_reconstruction")
    if checks["continued_simulation_68"] and not checks["continued_catalogue_68"]:
        failure_modes.append("simulation_to_catalogue_posterior_shift")
    production_checks = (
        "decoder",
        "selection_identity",
        "error_contract",
        "noise_contract",
        "nuisance_conditional",
        "continued_simulation_68",
        "continued_simulation_95",
        "continued_catalogue_68",
        "continued_catalogue_95",
        "parent_alpha",
        "classifier_density_closure",
        "population_bootstrap_stability",
    )
    ready = all(checks[key] for key in production_checks)
    write(
        root / "DECISION.json",
        dict(
            checks=checks,
            production_checks=list(production_checks),
            check_interpretation={
                "nuisance_conditional": (
                    "population prior selection-normalization contract; not "
                    "individual SFH reconstruction accuracy"
                )
            },
            failure_modes=failure_modes,
            ready_for_production=ready,
            next_action=(
                "launch_corrected_production"
                if ready
                else "do_not_retrain_until_failed_contract_is_corrected"
            ),
        ),
    )
    write(
        root / "FINAL.json",
        dict(
            status="FENIKS_FAILURE_MODE_AUDIT_COMPLETE",
            ready_for_production=ready,
            tasks=finals,
            scientific_promotion=False,
        ),
    )
    (root / "REPORT.md").write_text(
        "# FENIKS failure-mode audit\n\n"
        "Read DECISION.json and failure_modes_summary.png first. Gates were predeclared in audit.yaml. "
        "A failed gate localizes a contract or calibration failure; it does not prove a unique causal mechanism. "
        "Truth and nuisance replacements are diagnostic only. No model was trained or modified.\n"
    )


def run(root, task):
    if task < 0 or task >= len(TASKS):
        raise ValueError(f"invalid task {task}")
    write(
        root / TASKS[task] / "PROGRESS.json",
        dict(stage="started", task=task, name=TASKS[task]),
    )
    if task == 0:
        decoder_observation(root)
    elif task == 1:
        nuisance_sensitivity(root)
    elif task == 2:
        posterior_evaluation(root, False)
    elif task == 3:
        posterior_evaluation(root, True)
    elif task == 4:
        population_identifiability(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "task", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_failure_modes_r29.yaml"),
    )
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.parent is None or args.diagnostics is None:
            parser.error("prepare requires --parent and --diagnostics")
        prepare(args.parent, args.diagnostics, args.root, args.config)
    elif args.stage == "task":
        run(args.root, args.task)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
