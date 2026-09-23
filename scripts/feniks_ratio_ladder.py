"""Localize FENIKS parent recovery across exact and learned ratio levels.

This is a diagnostic closure. It never uses q samples and never modifies a
production prior. All four arms share the same 15D simulator and noisy-r
selection; only the information available to the ratio estimator changes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import log_ndtr, logsumexp, ndtri_exp
from scipy.stats import kstest, spearmanr, wasserstein_distance

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_weighted_truth_flow_capacity import _status

ARMS = ("exact_theta", "physical_a", "noiseless_photometry", "noisy_photometry")


def contract(root: Path):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"ratio-ladder input changed: {path}")
    return manifest, manifest["settings"]


def prepare(clean: Path, root: Path, config: Path):
    _status(clean / "report/FINAL.json", "CLEAN_PARENT_REPORT_COMPLETE")
    _status(clean / "closure/FINAL.json", "CLEAN_MATCHED_CLOSURE_COMPLETE")
    cm = read(clean / "MANIFEST.json")
    parent = Path(cm["parent"])
    _status(parent / "population/FINAL.json", "FORWARD_PARENT_COMPLETE")
    settings = yaml.safe_load(config.read_text())
    with np.load(parent / "basis.npz") as payload:
        components = len(payload["centers"])
    if root.exists():
        raise FileExistsError(root)
    if settings["reference_simulations"] % settings["reference_shards"]:
        raise ValueError("reference simulations must divide reference shards")
    if settings["reference_simulations"] < components * 512:
        raise ValueError(
            f"reference bank is too small for {components} selected classes"
        )
    for key in (
        "reference_shards",
        "reference_min_selected_per_component",
        "target_simulations",
        "metric_draws",
        "bootstraps",
    ):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    root.mkdir(parents=True)
    for name in (
        "logs",
        "observation",
        "decoder_observation",
        "bank",
        "report",
        *ARMS,
    ):
        (root / name).mkdir()
    shutil.copy2(config, root / "ratio_ladder.yaml")
    inputs = {
        "clean_manifest": clean / "MANIFEST.json",
        "clean_weights": clean / "closure/weights.csv",
        "parent_manifest": parent / "MANIFEST.json",
        "basis": parent / "basis.npz",
        "classifier_weights": parent / "population/component_weights.csv",
        "truth_parent": Path(cm["inputs"]["truth_parent"]["path"]),
        "selection_identities": Path(cm["inputs"]["selection_identities"]["path"]),
        "audit_indices": Path(cm["inputs"]["audit_indices"]["path"]),
        "config": root / "ratio_ladder.yaml",
    }
    write(
        root / "MANIFEST.json",
        dict(
            clean=str(clean.resolve()),
            parent=str(parent.resolve()),
            source=cm["source"],
            settings=settings,
            inputs={
                key: dict(path=str(path.resolve()), sha256=sha(path))
                for key, path in inputs.items()
            },
            arms=list(ARMS),
            components=components,
            population_uses_q=False,
            production_prior_modified=False,
            target=(
                "same known 15D parent and noisy observed-r selection; only ratio "
                "information changes across arms"
            ),
        ),
    )
    print(yaml.safe_dump(settings, sort_keys=False))
    print("Banks: uniform reference plus known-parent target. No posterior training.")


def _runtime(manifest, destination):
    from euclid_dsps.amortized.forward_population_runtime import load_forward_runtime

    parent_settings = read(Path(manifest["parent"]) / "MANIFEST.json")["settings"]
    return load_forward_runtime(
        Path(manifest["source"]), destination / "runtime", parent_settings
    )


def conditional_normal_score(z, threshold_z, selected):
    """Map selection-truncated Gaussian residuals back to standard normal."""
    z = np.asarray(z, np.float64)
    threshold_z = np.asarray(threshold_z, np.float64)
    selected = np.asarray(selected, bool)
    if not (z.shape == threshold_z.shape == selected.shape):
        raise ValueError("conditional-noise arrays differ")
    u = np.empty_like(z)
    # For z > t, F(z | z>t) = 1 - SF(z)/SF(t). log_ndtr(-x)=log SF(x).
    ratio = np.exp(log_ndtr(-z[selected]) - log_ndtr(-threshold_z[selected]))
    u[selected] = 1.0 - ratio
    # For z <= t, F(z | z<=t) = Phi(z)/Phi(t).
    u[~selected] = np.exp(log_ndtr(z[~selected]) - log_ndtr(threshold_z[~selected]))
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return ndtri_exp(np.log(u))


def observation(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from euclid_dsps.amortized.forward_population import PHYSICAL
    from euclid_dsps.photometry import abmag_to_fnu_cgs
    from scripts.feniks_failure_modes import (
        _catalogue,
        _truth_x,
        decoder_observation,
    )

    manifest, settings = contract(root)
    out = root / "observation"
    write(out / "PROGRESS.json", dict(stage="decoder_replay"))
    # The existing routine writes the auditable per-object residual table. It
    # intentionally retains the raw selected-r noise result for comparison.
    decoder_observation(root)
    source = root / "decoder_observation"
    for name in ("FINAL.json", "band_summary.csv", "per_object_band.parquet"):
        shutil.copy2(source / name, out / name)
    raw_decoder_final = read(source / "FINAL.json")
    model, runtime, _, _ = _runtime(manifest, out)
    del model
    bands = tuple(runtime.feature_stats.band_names)
    indices = np.load(manifest["inputs"]["audit_indices"]["path"])[
        : settings["decoder_objects"]
    ]
    theta, _ = _truth_x(manifest, runtime, indices)
    truth = pd.DataFrame(theta, columns=runtime.latent_spec.names)
    truth.insert(0, "parent_row_index", indices)
    catalogue = _catalogue(manifest, bands, indices)
    selection = pd.read_parquet(
        manifest["inputs"]["selection_identities"]["path"],
        columns=["parent_row_index", "selected"],
    ).set_index("parent_row_index")
    selected = selection.loc[indices, "selected"].to_numpy(bool)
    rows = []
    r_index = bands.index("lsst_r")
    cut = read(Path(manifest["parent"]) / "MANIFEST.json")["settings"]["cut"]
    threshold = float(abmag_to_fnu_cgs(cut))
    for j, band in enumerate(bands):
        true = catalogue[f"flux_true_{band}"].to_numpy()
        noisy = catalogue[f"flux_{band}"].to_numpy()
        error = catalogue[f"fluxerr_{band}"].to_numpy()
        mask = catalogue[f"mask_{band}"].to_numpy(bool)
        good = mask & np.isfinite(error) & (error > 0)
        z = (noisy[good] - true[good]) / error[good]
        score = (
            conditional_normal_score(
                z, (threshold - true[good]) / error[good], selected[good]
            )
            if j == r_index
            else z
        )
        rows.append(
            dict(
                band=band,
                transform="selection_conditional" if j == r_index else "marginal",
                objects=len(score),
                mean=float(score.mean()),
                std=float(score.std(ddof=1)),
                normal_ks=float(kstest(score, "norm").statistic),
            )
        )
    conditional = pd.DataFrame(rows)
    conditional.to_csv(out / "conditional_noise.csv", index=False)

    long = pd.read_parquet(out / "per_object_band.parquet").merge(
        truth, on="parent_row_index", validate="many_to_one"
    )
    long = long.merge(
        selection[["selected"]], left_on="parent_row_index", right_index=True
    )
    snr = []
    for band in bands:
        block = catalogue[["parent_row_index", f"flux_true_{band}", f"fluxerr_{band}"]]
        block = block.assign(
            band=band,
            log10_snr=np.log10(
                np.maximum(
                    np.abs(block[f"flux_true_{band}"])
                    / np.maximum(block[f"fluxerr_{band}"], 1e-40),
                    1e-8,
                )
            ),
        )[["parent_row_index", "band", "log10_snr"]]
        snr.append(block)
    long = long.merge(pd.concat(snr), on=["parent_row_index", "band"])
    stratifiers = ("log10_snr", *PHYSICAL)
    strata = []
    for band, group in long.groupby("band", sort=False):
        for key in stratifiers:
            bins = pd.qcut(group[key], 6, duplicates="drop")
            for label, values in group.groupby(bins, observed=True):
                strata.append(
                    dict(
                        band=band,
                        stratifier=key,
                        bin=str(label),
                        center=float(values[key].median()),
                        objects=len(values),
                        signed_mean_sigma=float(values.decoder_residual_sigma.mean()),
                        median_abs_sigma=float(
                            abs(values.decoder_residual_sigma).median()
                        ),
                        p95_abs_sigma=float(
                            abs(values.decoder_residual_sigma).quantile(0.95)
                        ),
                    )
                )
        for flag, values in group.groupby("selected"):
            strata.append(
                dict(
                    band=band,
                    stratifier="selected",
                    bin=str(bool(flag)),
                    center=float(flag),
                    objects=len(values),
                    signed_mean_sigma=float(values.decoder_residual_sigma.mean()),
                    median_abs_sigma=float(abs(values.decoder_residual_sigma).median()),
                    p95_abs_sigma=float(
                        abs(values.decoder_residual_sigma).quantile(0.95)
                    ),
                )
            )
    pd.DataFrame(strata).to_csv(out / "decoder_strata.csv", index=False)
    correlations = []
    for band, group in long.groupby("band", sort=False):
        for key in stratifiers:
            value = spearmanr(group[key], group.decoder_residual_sigma).statistic
            correlations.append(dict(band=band, variable=key, spearman=value))
    pd.DataFrame(correlations).to_csv(out / "decoder_correlations.csv", index=False)
    long.loc[abs(long.decoder_residual_sigma).nlargest(200).index].to_csv(
        out / "decoder_largest_residuals.csv", index=False
    )

    summary = pd.read_csv(out / "band_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].bar(summary.band, summary.decoder_p95_abs_sigma)
    axes[0, 0].axhline(
        settings["contracts"]["decoder_p95_abs_residual_sigma"], color="black", ls="--"
    )
    axes[0, 0].set(title="Decoder p95 residual", ylabel="|residual| / sigma")
    axes[0, 0].tick_params(axis="x", rotation=90)
    axes[0, 1].bar(conditional.band, conditional["mean"])
    axes[0, 1].set(title="Noise after selection-conditional transform", ylabel="mean")
    axes[0, 1].tick_params(axis="x", rotation=90)
    focus = long[long.band == "roman_F087"]
    axes[1, 0].scatter(focus.z_obs, focus.decoder_residual_sigma, s=4, alpha=0.25)
    axes[1, 0].set(title="Roman F087 residual vs redshift", xlabel="z", ylabel="sigma")
    axes[1, 1].scatter(focus.log10_snr, focus.decoder_residual_sigma, s=4, alpha=0.25)
    axes[1, 1].set(
        title="Roman F087 residual vs S/N", xlabel="log10 S/N", ylabel="sigma"
    )
    fig.tight_layout()
    fig.savefig(out / "observation_contract.png", dpi=160)
    plt.close(fig)
    noise_pass = bool(
        (
            abs(conditional["mean"])
            <= settings["contracts"]["conditional_noise_mean_abs"]
        ).all()
        and (
            abs(conditional["std"] - 1)
            <= settings["contracts"]["conditional_noise_std_abs_from_one"]
        ).all()
    )
    decoder_pass = bool(
        (
            summary.decoder_p95_abs_sigma
            <= settings["contracts"]["decoder_p95_abs_residual_sigma"]
        ).all()
    )
    write(
        out / "FINAL.json",
        dict(
            status="RATIO_LADDER_OBSERVATION_COMPLETE",
            conditional_noise_pass=noise_pass,
            decoder_pass=decoder_pass,
            selection_conditional_band="lsst_r",
            selection_identity_mismatches=raw_decoder_final.get(
                "selection_identity_mismatches", 0
            ),
        ),
    )


def _basis(root):
    from scripts.feniks_forward_population import basis_for

    manifest, _ = contract(root)
    return basis_for(Path(manifest["parent"]))


def bank(root: Path, task: int):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.forward_population_runtime import simulator

    manifest, settings = contract(root)
    shards = settings["reference_shards"]
    if not 0 <= task <= shards:
        raise ValueError("bank task outside reference plus target range")
    kind = "reference" if task < shards else "target"
    out = root / "bank" / (f"reference_{task:03d}" if kind == "reference" else "target")
    out.mkdir()
    basis = _basis(root)
    count = (
        settings["reference_simulations"] // shards
        if kind == "reference"
        else settings["target_simulations"]
    )
    rng = np.random.default_rng(settings["seed"] + 1000 + task)
    if kind == "reference":
        labels = np.arange(count) % basis.components
        rng.shuffle(labels)
        x, labels = basis.sample(rng, count, labels=labels)
    else:
        weights = pd.read_csv(manifest["inputs"]["clean_weights"]["path"])
        u_true = weights.true_parent.to_numpy()
        x, labels = basis.sample(rng, count, weights=u_true)
    model, runtime, observation_config, _ = _runtime(manifest, out)
    simulate = simulator(model, runtime, observation_config, diagnostic_features=True)
    block = settings["simulation_batch"]
    checkpoint = settings["bank_checkpoint_batches"]
    partial = out / "partial.npz"
    buffers = {}
    start = 0
    if partial.exists():
        with np.load(partial) as data:
            buffers = {key: data[key] for key in data.files}
        start = len(buffers["selected"])
    chunks = {key: [value] for key, value in buffers.items()}
    keep = ("x", "features", "noiseless_features", "selected")
    for offset in range(start, count, block):
        indices = np.minimum(np.arange(offset, offset + block), count - 1)
        generated = jax.device_get(
            simulate(
                jnp.asarray(x[indices]),
                jax.random.fold_in(jax.random.PRNGKey(settings["seed"] + task), offset),
            )
        )
        n = min(block, count - offset)
        if not np.asarray(generated["valid"][:n]).all():
            raise ValueError("invalid ratio-ladder simulation")
        for key in keep:
            chunks.setdefault(key, []).append(np.asarray(generated[key][:n]))
        if (offset // block + 1) % checkpoint == 0:
            values = {key: np.concatenate(value) for key, value in chunks.items()}
            temporary = out / "partial.next.npz"
            np.savez(temporary, **values)
            temporary.replace(partial)
            chunks = {key: [value] for key, value in values.items()}
            write(out / "PROGRESS.json", dict(done=offset + n, total=count))
            print(f"{kind} task={task} {offset + n}/{count}", flush=True)
    values = {key: np.concatenate(value) for key, value in chunks.items()}
    values["component"] = labels
    temporary = out / "bank.next.npz"
    np.savez(temporary, **values)
    temporary.replace(out / "bank.npz")
    partial.unlink(missing_ok=True)
    write(
        out / "FINAL.json",
        dict(
            status="RATIO_LADDER_BANK_COMPLETE",
            kind=kind,
            count=count,
            selected=int(values["selected"].sum()),
            bank_sha256=sha(out / "bank.npz"),
        ),
    )


def _load_bank(root, kind):
    _, settings = contract(root)
    paths = (
        [
            root / "bank" / f"reference_{i:03d}"
            for i in range(settings["reference_shards"])
        ]
        if kind == "reference"
        else [root / "bank/target"]
    )
    arrays = {}
    for path in paths:
        receipt = read(path / "FINAL.json")
        if receipt.get("status") != "RATIO_LADDER_BANK_COMPLETE":
            raise ValueError(f"incomplete bank {path}")
        if sha(path / "bank.npz") != receipt["bank_sha256"]:
            raise ValueError(f"changed bank {path}")
        with np.load(path / "bank.npz") as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def exact_selected_log_classifier(x, basis, alpha, frequencies):
    """Classifier-form logits whose C_j/c_j equal exact selected densities."""
    alpha = np.asarray(alpha, np.float64)
    frequencies = np.asarray(frequencies, np.float64)
    logits = basis.component_log_prob(x) - np.log(alpha) + np.log(frequencies)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def _stratified_split(labels, selected, seed):
    rng = np.random.default_rng(seed)
    groups = ([], [], [])
    for label in range(labels.max() + 1):
        idx = np.flatnonzero(selected & (labels == label))
        if len(idx) < 7:
            raise ValueError(
                f"component {label} has only {len(idx)} selected rows; enlarge bank"
            )
        rng.shuffle(idx)
        a, b = int(0.7 * len(idx)), int(0.85 * len(idx))
        groups[0].extend(idx[:a])
        groups[1].extend(idx[a:b])
        groups[2].extend(idx[b:])
    result = tuple(np.asarray(value, np.int64) for value in groups)
    if any(len(value) == 0 for value in result):
        raise ValueError("empty selected classifier split")
    for value in result:
        rng.shuffle(value)
    return result


def _mixture_objective(logc, frequencies, weights):
    return float(
        np.mean(
            logsumexp(
                logc
                - np.log(frequencies)[None]
                + np.log(np.maximum(weights, 1e-300))[None],
                axis=1,
            )
        )
    )


def _ece(logc, labels, bins=15):
    probability = np.exp(logc)
    predicted = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = predicted == labels
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        use = (confidence >= low) & (
            confidence < high if high < 1 else confidence <= high
        )
        if use.any():
            value += use.mean() * abs(confidence[use].mean() - correct[use].mean())
    return float(value)


def ratio(root: Path, task: int):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.forward_population import (
        fit_selected_weights,
        parent_from_selected,
        selected_from_parent,
    )
    from euclid_dsps.amortized.latent import x_to_theta
    from scripts.feniks_clean_parent import likelihood_geometry
    from scripts.feniks_forward_population import (
        classifier_template,
        classify,
        supervised_fit,
    )
    from scripts.report_feniks_forward_population import population_metrics

    if task not in range(len(ARMS)):
        raise ValueError("ratio task must be 0..3")
    arm = ARMS[task]
    manifest, settings = contract(root)
    out = root / arm
    reference, target = _load_bank(root, "reference"), _load_bank(root, "target")
    basis = _basis(root)
    table = pd.read_csv(manifest["inputs"]["classifier_weights"]["path"])
    attempts = np.bincount(reference["component"], minlength=basis.components).astype(
        float
    )
    successes = np.bincount(
        reference["component"][reference["selected"]], minlength=basis.components
    ).astype(float)
    alpha = successes / attempts
    # Keep the production support declaration, but evaluate selection efficiency
    # from this matched bank so stale Monte-Carlo alpha cannot contaminate the
    # analytic rung of the ladder.
    eligible = (
        table.eligible.to_numpy(bool)
        & (successes >= settings["reference_min_selected_per_component"])
        & (alpha > 0)
    )
    u_true = pd.read_csv(
        manifest["inputs"]["clean_weights"]["path"]
    ).true_parent.to_numpy()
    v_true = selected_from_parent(u_true, alpha)
    target_parent_empirical = np.bincount(
        target["component"], minlength=basis.components
    ) / len(target["component"])
    target_selected_empirical = np.bincount(
        target["component"][target["selected"]], minlength=basis.components
    ) / np.sum(target["selected"])
    train, validation, test = _stratified_split(
        reference["component"], reference["selected"], settings["seed"]
    )
    labels = reference["component"]
    frequencies = np.bincount(labels[train], minlength=basis.components) / len(train)
    if np.any(frequencies <= 0):
        raise ValueError("empty selected classifier class")
    target_selected = np.flatnonzero(target["selected"])
    rng = np.random.default_rng(settings["seed"] + task)
    rng.shuffle(target_selected)
    midpoint = len(target_selected) // 2
    fit_ids, heldout = target_selected[:midpoint], target_selected[midpoint:]
    if arm == "exact_theta":
        logc_fit = exact_selected_log_classifier(
            target["x"][fit_ids], basis, alpha, frequencies
        )
        logc_held = exact_selected_log_classifier(
            target["x"][heldout], basis, alpha, frequencies
        )
        logc_test = exact_selected_log_classifier(
            reference["x"][test], basis, alpha, frequencies
        )
    else:
        if arm == "physical_a":
            feature = reference["x"][:, basis.indices]
            target_feature = target["x"][:, basis.indices]
        elif arm == "noiseless_photometry":
            feature = reference["noiseless_features"]
            target_feature = target["noiseless_features"]
        else:
            feature = reference["features"]
            target_feature = target["features"]
        cs = {**settings["classifier"], "seed": settings["seed"] + task}
        candidate = classifier_template(feature.shape[1], basis.components, cs)

        def loss(net, f, t):
            prediction = jax.nn.log_softmax(jax.vmap(net)(f), axis=-1)
            return -jnp.take_along_axis(prediction, t[:, None], axis=1)[:, 0]

        classifier = supervised_fit(
            candidate,
            loss,
            feature[train],
            labels[train],
            (feature[validation], labels[validation]),
            cs,
            out,
        )
        logc_test = classify(classifier, feature[test])
        logc_fit = classify(classifier, target_feature[fit_ids])
        logc_held = classify(classifier, target_feature[heldout])
    ratio_moment = np.exp(logc_test - np.log(frequencies)[None]).mean(axis=0)
    per_component = []
    for component in range(basis.components):
        member = labels[test] == component
        per_component.append(
            dict(
                component=component,
                test_objects=int(member.sum()),
                nll=float(-logc_test[member, component].mean()),
                recall=float((logc_test[member].argmax(axis=1) == component).mean()),
                ratio_moment=float(ratio_moment[component]),
                alpha=float(alpha[component]),
                eligible=bool(eligible[component]),
            )
        )
    pd.DataFrame(per_component).to_csv(
        out / "classifier_component_metrics.csv", index=False
    )
    classifier_metrics = dict(
        classifier_trained=arm != "exact_theta",
        classifier_nll=float(-logc_test[np.arange(len(test)), labels[test]].mean()),
        classifier_null_nll=float(-np.log(frequencies[labels[test]]).mean()),
        classifier_accuracy=float((logc_test.argmax(axis=1) == labels[test]).mean()),
        classifier_ece=float(_ece(logc_test, labels[test])),
        ratio_moment_median_abs_error=float(np.median(abs(ratio_moment - 1))),
        ratio_moment_min=float(ratio_moment.min()),
        ratio_moment_max=float(ratio_moment.max()),
        classifier_sha256=None if arm == "exact_theta" else sha(out / "best.eqx"),
    )
    np.savez_compressed(out / "heldout_ratios.npz", logc=logc_held)
    v, diagnostics = fit_selected_weights(
        logc_fit,
        frequencies,
        alpha=alpha,
        eligible=eligible,
        weak_parent_mass=read(Path(manifest["parent"]) / "MANIFEST.json")["settings"][
            "weak_parent_mass_cap"
        ],
    )
    u = parent_from_selected(v, alpha)
    pd.DataFrame(
        dict(
            component=np.arange(basis.components),
            true_parent=u_true,
            target_parent_empirical=target_parent_empirical,
            fitted_parent=u,
            true_selected=v_true,
            target_selected_empirical=target_selected_empirical,
            fitted_selected=v,
            alpha=alpha,
            alpha_production=table.alpha.to_numpy(),
            alpha_attempts=attempts,
            alpha_successes=successes,
            eligible=eligible,
            frequency=frequencies,
        )
    ).to_csv(out / "weights.csv", index=False)
    write(
        out / "likelihood_geometry.json", likelihood_geometry(logc_fit, frequencies, v)
    )

    model, runtime, _, _ = _runtime(manifest, out)
    del model
    draws = settings["metric_draws"]
    learned_x, _ = basis.sample(rng, draws, u)
    truth_x, _ = basis.sample(rng, draws, u_true)
    learned_theta = np.asarray(x_to_theta(jnp.asarray(learned_x), runtime.latent_spec))
    truth_theta = np.asarray(x_to_theta(jnp.asarray(truth_x), runtime.latent_spec))
    parent_marginal, parent_joint = population_metrics(
        learned_theta,
        truth_theta,
        tuple(runtime.latent_spec.names),
        settings["metric_seed"],
    )
    parent_marginal.to_csv(out / "parent_marginal.csv", index=False)
    parent_joint.to_csv(out / "parent_joint.csv", index=False)

    terms = basis.component_log_prob(target["x"][heldout])
    lr = logsumexp(terms + np.log(np.maximum(u, 1e-300)), axis=1) - logsumexp(
        terms + np.log(np.maximum(u_true, 1e-300)), axis=1
    )
    importance = np.exp(lr - lr.max())
    importance /= importance.sum()
    chosen_fit = rng.choice(heldout, draws, replace=True, p=importance)
    chosen_true = rng.choice(heldout, draws, replace=True)
    selected_fit = np.asarray(
        x_to_theta(jnp.asarray(target["x"][chosen_fit]), runtime.latent_spec)
    )
    selected_true = np.asarray(
        x_to_theta(jnp.asarray(target["x"][chosen_true]), runtime.latent_spec)
    )
    selected_marginal, selected_joint = population_metrics(
        selected_fit,
        selected_true,
        tuple(runtime.latent_spec.names),
        settings["metric_seed"],
    )
    selected_marginal.to_csv(out / "selected_marginal.csv", index=False)
    selected_joint.to_csv(out / "selected_joint.csv", index=False)

    feature_rows = []
    held_features = target["features"][heldout]
    baseline_size = min(8192, len(heldout) // 2)
    baseline = np.empty((settings["bootstraps"], held_features.shape[1]))
    for repeat in range(settings["bootstraps"]):
        left = rng.choice(len(heldout), baseline_size, replace=False)
        remaining = np.setdiff1d(np.arange(len(heldout)), left)
        right = rng.choice(remaining, baseline_size, replace=False)
        for feature_index in range(held_features.shape[1]):
            baseline[repeat, feature_index] = wasserstein_distance(
                held_features[left, feature_index], held_features[right, feature_index]
            )
    for feature_index in range(held_features.shape[1]):
        observed = wasserstein_distance(
            held_features[:, feature_index],
            held_features[:, feature_index],
            v_weights=importance,
        )
        median = float(np.median(baseline[:, feature_index]))
        feature_rows.append(
            dict(
                feature=feature_index,
                learned_w1=float(observed),
                finite_sample_w1_median=median,
                finite_sample_w1_q90=float(
                    np.quantile(baseline[:, feature_index], 0.9)
                ),
                learned_over_baseline=observed / median if median > 0 else np.nan,
            )
        )
    pd.DataFrame(feature_rows).to_csv(out / "observable_closure.csv", index=False)

    boot = []
    bootstrap_draws = min(draws, 4096)
    bootstrap_truth_x, _ = basis.sample(rng, bootstrap_draws, u_true)
    bootstrap_truth = np.asarray(
        x_to_theta(jnp.asarray(bootstrap_truth_x), runtime.latent_spec)
    )
    for repeat in range(settings["bootstraps"]):
        sample = rng.integers(len(logc_fit), size=len(logc_fit))
        vb, _ = fit_selected_weights(
            logc_fit[sample],
            frequencies,
            alpha=alpha,
            eligible=eligible,
            weak_parent_mass=read(Path(manifest["parent"]) / "MANIFEST.json")[
                "settings"
            ]["weak_parent_mass_cap"],
        )
        ub = parent_from_selected(vb, alpha)
        bx, _ = basis.sample(rng, bootstrap_draws, ub)
        bt = np.asarray(x_to_theta(jnp.asarray(bx), runtime.latent_spec))
        _, bj = population_metrics(
            bt,
            bootstrap_truth,
            tuple(runtime.latent_spec.names),
            settings["metric_seed"],
        )
        boot.append(
            dict(
                replicate=repeat,
                parent_weight_l1=float(abs(ub - u_true).sum()),
                selected_weight_l1=float(abs(vb - v_true).sum()),
                alpha_fitted=float(ub @ alpha),
                parent_physical_sliced_wasserstein=float(
                    bj.loc[bj.group == "physical", "sliced_wasserstein"].iloc[0]
                ),
            )
        )
    pd.DataFrame(boot).to_csv(out / "bootstrap.csv", index=False)
    np.savez_compressed(
        out / "draws.npz",
        learned_parent=learned_theta,
        true_parent=truth_theta,
        learned_selected=selected_fit,
        true_selected=selected_true,
        names=np.asarray(runtime.latent_spec.names),
    )
    parent_physical = float(
        parent_joint.loc[parent_joint.group == "physical", "sliced_wasserstein"].iloc[0]
    )
    selected_physical = float(
        selected_joint.loc[
            selected_joint.group == "physical", "sliced_wasserstein"
        ].iloc[0]
    )
    write(
        out / "FINAL.json",
        dict(
            status="RATIO_LADDER_ARM_COMPLETE",
            arm=arm,
            observations_fit=len(fit_ids),
            observations_heldout=len(heldout),
            parent_weight_l1=float(abs(u - u_true).sum()),
            selected_weight_l1=float(abs(v - v_true).sum()),
            target_parent_sampling_l1=float(
                abs(target_parent_empirical - u_true).sum()
            ),
            target_selected_sampling_l1=float(
                abs(target_selected_empirical - v_true).sum()
            ),
            parent_physical_sliced_wasserstein=parent_physical,
            selected_physical_sliced_wasserstein=selected_physical,
            alpha_true=float(u_true @ alpha),
            alpha_fitted=float(u @ alpha),
            alpha_reference_bank=float(reference["selected"].mean()),
            alpha_target_bank=float(target["selected"].mean()),
            alpha_max_abs_delta_from_production=float(
                np.max(abs(alpha - table.alpha.to_numpy()))
            ),
            predictive_weight_ess=float(1 / np.sum(importance**2)),
            heldout_fit_log_likelihood=_mixture_objective(logc_held, frequencies, v),
            heldout_true_log_likelihood=_mixture_objective(
                logc_held, frequencies, v_true
            ),
            population_uses_q=False,
            production_prior_modified=False,
            **diagnostics,
            **classifier_metrics,
        ),
    )


def _localize_failure(summary, contracts):
    values = summary.set_index("arm")
    parent = values.parent_physical_sliced_wasserstein
    selected = values.selected_physical_sliced_wasserstein
    exact_pass = bool(
        parent.exact_theta <= contracts["exact_parent_physical_sw"]
        and selected.exact_theta <= contracts["exact_selected_physical_sw"]
    )
    pairs = {
        "physical_classifier": ("exact_theta", "physical_a"),
        "photometric_projection": ("physical_a", "noiseless_photometry"),
        "observation_noise": ("noiseless_photometry", "noisy_photometry"),
    }
    increments = {}
    stage_increments = {}
    for key, (before, after) in pairs.items():
        increments[f"{key}_parent"] = float(parent[after] - parent[before])
        increments[f"{key}_selected"] = float(selected[after] - selected[before])
        stage_increments[key] = max(
            increments[f"{key}_parent"], increments[f"{key}_selected"]
        )
    passes = {
        "exact_ratio_and_selection": exact_pass,
        **{
            key: bool(value <= contracts["maximum_stage_sw_increase"])
            for key, value in stage_increments.items()
        },
    }
    if not passes["exact_ratio_and_selection"]:
        conclusion = "selection_correction_or_simplex_objective"
    elif not passes["physical_classifier"]:
        conclusion = "classifier_ratio_or_component_identifiability"
    elif not passes["photometric_projection"]:
        conclusion = "decoder_or_photometric_information_loss"
    elif not passes["observation_noise"]:
        conclusion = "noise_and_selection_information_loss"
    else:
        conclusion = "matched_ratio_pipeline_closes"
    return passes, increments, conclusion


def report(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, settings = contract(root)
    out = root / "report"
    rows = []
    for arm in ARMS:
        final = read(root / arm / "FINAL.json")
        if final.get("status") != "RATIO_LADDER_ARM_COMPLETE":
            raise ValueError(f"incomplete ratio arm {arm}")
        rows.append(final)
    summary = pd.DataFrame(rows)
    summary["heldout_fit_minus_true"] = (
        summary.heldout_fit_log_likelihood - summary.heldout_true_log_likelihood
    )
    summary.to_csv(out / "ratio_ladder_summary.csv", index=False)
    decisions, increments, conclusion = _localize_failure(
        summary, settings["contracts"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels = summary.arm.str.replace("_", "\n")
    axes[0, 0].bar(labels, summary.parent_physical_sliced_wasserstein)
    axes[0, 0].set(title="Parent physical density", ylabel="sliced Wasserstein")
    axes[0, 1].bar(labels, summary.selected_physical_sliced_wasserstein)
    axes[0, 1].set(title="Selected physical density", ylabel="sliced Wasserstein")
    axes[1, 0].bar(labels, summary.parent_weight_l1)
    axes[1, 0].set(title="Component weights (diagnostic)", ylabel="parent L1")
    axes[1, 1].bar(labels, summary.heldout_fit_minus_true)
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set(title="Held-out pseudo-likelihood", ylabel="fit minus true")
    fig.tight_layout()
    fig.savefig(out / "ratio_ladder_summary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for arm in ARMS[1:]:
        history = pd.read_json(root / arm / "training.jsonl", lines=True)
        axes[0, 0].plot(history.epoch, history.train_nll, label=f"{arm} train")
        axes[0, 0].plot(
            history.epoch,
            history.validation_nll,
            linestyle="--",
            label=f"{arm} validation",
        )
    axes[0, 0].set(title="Classifier convergence", xlabel="epoch", ylabel="NLL")
    axes[0, 0].legend(fontsize=7, ncol=2)
    for arm in ARMS:
        component = pd.read_csv(root / arm / "classifier_component_metrics.csv")
        axes[0, 1].scatter(
            component.component,
            component.ratio_moment,
            s=10,
            alpha=0.7,
            label=arm,
        )
    axes[0, 1].axhline(1, color="black", lw=1)
    axes[0, 1].set(
        title="Reference ratio normalization",
        xlabel="component",
        ylabel="E_ref[C_j/c_j]",
    )
    axes[0, 1].legend(fontsize=7)
    weights = pd.read_csv(root / "exact_theta/weights.csv")
    axes[1, 0].scatter(weights.alpha_production, weights.alpha, s=12, alpha=0.7)
    limits = [min(weights.alpha_production.min(), weights.alpha.min()), 1]
    axes[1, 0].plot(limits, limits, color="black", lw=1)
    axes[1, 0].set(
        title="Fresh vs production selection efficiency",
        xlabel="production alpha",
        ylabel="fresh alpha",
    )
    bootstrap = []
    for arm in ARMS:
        frame = pd.read_csv(root / arm / "bootstrap.csv")
        bootstrap.append(frame.parent_physical_sliced_wasserstein.to_numpy())
    axes[1, 1].boxplot(bootstrap, tick_labels=labels)
    axes[1, 1].set(
        title="Catalogue-bootstrap parent closure", ylabel="physical sliced Wasserstein"
    )
    fig.tight_layout()
    fig.savefig(out / "ratio_ladder_diagnostics.png", dpi=160)
    plt.close(fig)

    names = None
    payload = {}
    for arm in ARMS:
        with np.load(root / arm / "draws.npz") as data:
            names = tuple(data["names"].astype(str))
            payload[arm] = {key: data[key] for key in data.files if key != "names"}
    fig, axes = plt.subplots(2, 5, figsize=(19, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ARMS)))
    for column, name in enumerate(names[:5]):
        all_truth = payload[ARMS[0]]["true_parent"][:, column]
        bins = np.linspace(*np.quantile(all_truth, [0.001, 0.999]), 60)
        axes[0, column].hist(
            all_truth,
            bins=bins,
            density=True,
            histtype="step",
            color="black",
            lw=2,
            label="truth",
        )
        selected_truth = payload[ARMS[0]]["true_selected"][:, column]
        sbins = np.linspace(*np.quantile(selected_truth, [0.001, 0.999]), 60)
        axes[1, column].hist(
            selected_truth,
            bins=sbins,
            density=True,
            histtype="step",
            color="black",
            lw=2,
            label="truth",
        )
        for color, arm in zip(colors, ARMS, strict=True):
            axes[0, column].hist(
                payload[arm]["learned_parent"][:, column],
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                label=arm,
            )
            axes[1, column].hist(
                payload[arm]["learned_selected"][:, column],
                bins=sbins,
                density=True,
                histtype="step",
                color=color,
                label=arm,
            )
        axes[0, column].set_title(name)
    axes[0, 0].set_ylabel("parent")
    axes[1, 0].set_ylabel("selected")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ratio_ladder_physical.png", dpi=160)
    plt.close(fig)
    observation_final = read(root / "observation/FINAL.json")
    (out / "REPORT.md").write_text(
        "# FENIKS ratio ladder\n\n"
        "All arms use the same known 15D parent, simulator and noisy observed-r "
        "selection. Compare exact theta, physical a, noiseless photometry and noisy "
        "photometry in ratio_ladder_summary.csv. Weight L1 is secondary to parent "
        "and selected physical-density closure.\n\n"
        "The lsst_r noise gate uses a selection-conditional truncated-normal PIT. "
        "Decoder localization is in observation/decoder_strata.csv.\n\n"
        f"Automated localization: `{conclusion}`. Threshold decisions are "
        "diagnostic gates, not scientific promotion criteria.\n"
    )
    write(
        out / "FINAL.json",
        dict(
            status="RATIO_LADDER_REPORT_COMPLETE",
            observation=observation_final,
            arms=list(ARMS),
            decisions=decisions,
            physical_sw_increments=increments,
            conclusion=conclusion,
            production_ready=False,
            population_uses_q=False,
            production_prior_modified=False,
            source_clean_manifest=manifest["inputs"]["clean_manifest"]["sha256"],
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("prepare", "observation", "bank", "ratio", "report")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.clean is None or args.config is None:
            parser.error("prepare requires --clean and --config")
        prepare(args.clean.resolve(), args.root.resolve(), args.config.resolve())
    elif args.mode == "observation":
        observation(args.root)
    elif args.mode == "bank":
        bank(args.root, args.task)
    elif args.mode == "ratio":
        ratio(args.root, args.task)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
