"""Decoupled forward population likelihood -> frozen parent -> supervised NPE.

No posterior samples enter population fitting or supervised training targets.
Run stages through submit_feniks_forward_population.sh; source is runtime/r29
from the SBEB campaign (only immutable decoder/config/observations are reused).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.forward_population import (
    PhysicalBasis,
    fit_selected_weights,
    parent_from_selected,
    selection_efficiencies,
)
from scripts.feniks_avi_experiments import read, runtime_asset_paths, sha, write


def contract(root):
    m = read(root / "MANIFEST.json")
    for path, digest in m["hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"changed input {path}")
    return m, m["settings"]


def basis_for(root):
    with np.load(root / "basis.npz") as z:
        return PhysicalBasis(tuple(z["names"].tolist()), z["centers"], z["scales"])


def preflight(root):
    """Small real-decoder contract check, not a scientific pilot."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import log_prob
    from euclid_dsps.amortized.forward_population_runtime import (
        PopulationPrior,
        load_forward_runtime,
        posterior_template,
        simulator,
    )

    m, cfg = contract(root)
    out = root / "preflight"
    out.mkdir(exist_ok=True)
    basis = basis_for(root)
    model, rt, obs, config = load_forward_runtime(
        Path(m["source"]), out / "runtime", cfg
    )
    x, _ = basis.sample(
        np.random.default_rng(cfg["seed"]),
        basis.components * 2,
        labels=np.tile(np.arange(basis.components), 2),
    )
    generate = simulator(model, rt, obs)
    parts = []
    for start in range(0, len(x), cfg["simulation_batch"]):
        parts.append(
            jax.device_get(
                generate(
                    jnp.asarray(x[start : start + cfg["simulation_batch"]]),
                    jax.random.PRNGKey(cfg["seed"] + start),
                )
            )
        )
    if not all(np.all(p["valid"]) for p in parts):
        raise ValueError("preflight decoder validity failed")
    candidate = posterior_template(model, rt, config, cfg)
    f, t = jnp.asarray(parts[0]["features"][:2]), jnp.asarray(parts[0]["x"][:2])
    value, grads = eqx.filter_jit(
        eqx.filter_value_and_grad(
            lambda net: -jnp.mean(log_prob(model, net, f, t[None]))
        )
    )(candidate)
    if not np.isfinite(value) or not all(
        np.isfinite(g).all()
        for g in jax.tree_util.tree_leaves(grads)
        if eqx.is_array(g)
    ):
        raise ValueError("preflight supervised 15D gradient failed")
    prior = PopulationPrior(basis, np.ones(basis.components))
    np.testing.assert_allclose(
        prior.log_prob(jnp.asarray(x)),
        basis.log_prob(x, np.ones(basis.components)),
        rtol=1e-6,
    )
    finish(
        out,
        "FORWARD_PREFLIGHT_PASS",
        simulations=len(x),
        dimensions=15,
        selected=sum(int(p["selected"].sum()) for p in parts),
        supervised_nll=float(value),
    )


def finish(directory, status, **kwargs):
    write(directory / "FINAL.json", dict(status=status, **kwargs))


def attach_frozen_parent(root, model):
    import equinox as eqx

    from euclid_dsps.amortized.forward_population_runtime import PopulationPrior

    receipt = read(root / "population/FINAL.json")
    if receipt["parent_sha256"] != sha(root / "population/parent.json"):
        raise ValueError("parent integrity failed")
    parent = read(root / "population/parent.json")
    if parent["basis_sha256"] != sha(root / "basis.npz"):
        raise ValueError("parent basis integrity failed")
    return eqx.tree_at(
        lambda tree: tree.prior, model, PopulationPrior(basis_for(root), parent["u"])
    )


def prepare(
    source, root, config, truth_parent=None, truth_selected=None, baseline=None
):
    from euclid_dsps.amortized.train import _latent_spec_for_amortized_config
    from euclid_dsps.config import load_config

    if root.exists():
        raise FileExistsError(root)
    source = source.resolve()
    m = read(source / "MANIFEST.json")
    settings = yaml.safe_load(config.read_text())
    spec = _latent_spec_for_amortized_config(load_config(source / "source_config.yaml"))
    if spec.normalization not in {"standardized_logit", "bounded_mixed_warp"}:
        raise ValueError("main run requires an invertible bounded 15D latent transform")
    basis = PhysicalBasis.create(spec.names, seed=settings["seed"], **settings["basis"])
    source_config = load_config(source / "source_config.yaml")
    cut = source_config["amortized"]["objective"]["sleep"]["selection"]["max_mag_ab"]
    if float(cut) != settings["cut"]:
        raise ValueError("source cohort cut must equal experiment cut")
    for count, shards in [
        (settings["reference_simulations"], settings["reference_shards"]),
        (settings["posterior_simulations"], settings["posterior_shards"]),
    ]:
        if count % shards or (count // shards) % basis.components:
            raise ValueError("simulation counts must divide into shards and components")
    paths = [
        source / "MANIFEST.json",
        source / "source_config.yaml",
        source / "train.npy",
        source / "validation.npy",
        Path(m["source"]["checkpoint"]),
        Path(m["source"]["feature_stats"]),
        config.resolve(),
    ]
    paths += runtime_asset_paths(source_config)
    paths += [Path(source_config["catalog_path"]), Path(m["validation_catalog"])]
    normalization = (
        source_config["amortized"].get("latent", {}).get("normalization_checkpoint")
    )
    if normalization:
        paths.append(Path(normalization))
        paths.append(Path(str(normalization) + ".json"))
    for checkpoint in (Path(m["source"]["checkpoint"]),):
        paths.extend(
            p for p in checkpoint.parent.glob(checkpoint.stem + "*.json") if p.is_file()
        )
    root.mkdir(parents=True)
    for name in (
        "logs",
        "reference",
        "population",
        "posterior_bank",
        "posterior",
        "report",
    ):
        (root / name).mkdir()
    np.savez(
        root / "basis.npz",
        names=basis.names,
        centers=basis.centers,
        scales=basis.scales,
    )
    paths.append(root / "basis.npz")
    write(
        root / "MANIFEST.json",
        dict(
            version=1,
            source=str(source),
            settings=settings,
            hashes={str(p.resolve()): sha(p) for p in paths},
            blind_truth_parent=str(truth_parent.resolve()) if truth_parent else None,
            blind_truth_selected=(
                str(truth_selected.resolve()) if truth_selected else None
            ),
            baseline=str(baseline.resolve()) if baseline else None,
            reference_prior="analytic joint physical Gaussian mixture times identity-reference SFH",
            population_uses_q=False,
            posterior_target_source="simulator latent draw",
            selection_in_individual_weights=False,
            parent_scope="configured C0 support domain",
        ),
    )
    shutil.copy2(config, root / "experiment.yaml")
    total = (
        settings["reference_simulations"]
        + settings["posterior_simulations"]
        + settings["evaluation_simulations"]
    )
    print(
        json.dumps(
            dict(
                components=basis.components,
                basis="Sobol joint-5D centers, Gaussian overlap + broad tail",
                nuisance="10 standard-normal reference latent coordinates, all decoded normally",
                selection="saved noisy r cut; component binomial efficiencies including rejected draws",
                total_forward_simulations=total,
                classifier=settings["classifier"],
                posterior={
                    **settings["posterior"],
                    "effective_encoder": {
                        **source_config["amortized"]["encoder"],
                        **settings["posterior"]["architecture"],
                    },
                },
                resources=settings["resources"],
                outputs="banks, classifier, v/u/alpha, frozen parent, 15D flow, draws, calibration and population report",
            ),
            indent=2,
        )
    )


def bank(root, task, kind):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.forward_population_runtime import (
        load_forward_runtime,
        simulator,
    )

    m, cfg = contract(root)
    directory = root / ("reference" if kind == "reference" else "posterior_bank")
    out = directory / f"shard_{task:03d}"
    out.mkdir(exist_ok=True)
    if (out / "FINAL.json").exists():
        receipt = read(out / "FINAL.json")
        if receipt["bank_sha256"] != sha(out / "bank.npz"):
            raise ValueError("bank checkpoint hash mismatch")
        if kind != "reference" and receipt["parent_sha256"] != sha(
            root / "population/parent.json"
        ):
            raise ValueError("existing bank was generated under a different parent")
        return
    basis = basis_for(root)
    weights = None
    if kind == "reference":
        count = cfg["reference_simulations"] // cfg["reference_shards"]
        if not 0 <= task < cfg["reference_shards"]:
            raise ValueError("invalid reference shard")
    else:
        population = read(root / "population/FINAL.json")
        if population["status"] != "FORWARD_PARENT_COMPLETE":
            raise ValueError("frozen parent required")
        weights = read(root / "population/parent.json")["u"]
        if sha(root / "population/parent.json") != population["parent_sha256"]:
            raise ValueError("parent changed after freezing")
        count = (
            cfg["evaluation_simulations"]
            if task == cfg["posterior_shards"]
            else cfg["posterior_simulations"] // cfg["posterior_shards"]
        )
        if not 0 <= task <= cfg["posterior_shards"]:
            raise ValueError("invalid posterior shard")
    seed = cfg["seed"] + (10000 if kind == "reference" else 20000) + task
    rng = np.random.default_rng(seed)
    model, runtime, observation, _ = load_forward_runtime(
        Path(m["source"]), out / "runtime", cfg
    )
    simulate = simulator(model, runtime, observation)
    block = cfg["simulation_batch"]
    x, labels = basis.sample(
        rng,
        count,
        weights=weights,
        labels=np.arange(count) % basis.components if kind == "reference" else None,
    )
    file = out / "partial.npz"
    start = 0
    buffers = {}
    if file.exists():
        with np.load(file) as cached:
            buffers = {k: cached[k] for k in cached.files}
        start = len(buffers["selected"])
    chunks = {k: [v] for k, v in buffers.items()}
    for offset in range(start, count, block):
        indices = np.minimum(np.arange(offset, offset + block), count - 1)
        generated = jax.device_get(
            simulate(
                jnp.asarray(x[indices]),
                jax.random.fold_in(jax.random.PRNGKey(seed), offset),
            )
        )
        n = min(block, count - offset)
        if not np.all(generated["valid"][:n]):
            raise ValueError(
                "invalid reference simulation: do not silently renormalize the parent"
            )
        for k, v in generated.items():
            chunks.setdefault(k, []).append(np.asarray(v[:n]))
        if (offset // block + 1) % cfg["bank_checkpoint_batches"] == 0:
            buffers = {k: np.concatenate(v) for k, v in chunks.items()}
            temp = out / "partial.next.npz"
            np.savez(temp, **buffers)
            temp.replace(file)
            chunks = {k: [v] for k, v in buffers.items()}
            write(out / "PROGRESS.json", dict(completed=offset + n, total=count))
            print(f"{kind} shard={task} simulated={offset+n}/{count}", flush=True)
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    arrays["component"] = labels
    np.savez(out / "bank.npz", **arrays)
    file.unlink(missing_ok=True)
    finish(
        out,
        "FORWARD_BANK_COMPLETE",
        count=count,
        selected=int(arrays["selected"].sum()),
        bank_sha256=sha(out / "bank.npz"),
        kind=kind,
        seed=seed,
        parent_sha256=None if weights is None else sha(root / "population/parent.json"),
    )


def load_banks(root, kind, tasks, keys):
    arrays = {key: [] for key in keys}
    for task in tasks:
        folder = root / kind / f"shard_{task:03d}"
        receipt = read(folder / "FINAL.json")
        if receipt["bank_sha256"] != sha(folder / "bank.npz"):
            raise ValueError("bank changed")
        if kind == "posterior_bank" and receipt["parent_sha256"] != sha(
            root / "population/parent.json"
        ):
            raise ValueError("posterior bank does not match frozen parent")
        with np.load(folder / "bank.npz") as data:
            for key in keys:
                arrays[key].append(data[key])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def classifier_template(features, components, settings):
    import equinox as eqx
    import jax

    return eqx.nn.MLP(
        features,
        components,
        settings["width"],
        settings["depth"],
        activation=jax.nn.gelu,
        key=jax.random.PRNGKey(settings["seed"]),
    )


def supervised_fit(
    candidate, loss_function, features, targets, validation, settings, out
):
    """Shared optimizer; targets are component labels or FORWARD simulator theta."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    batch = settings["batch_size"]
    initial_checkpoint = settings.get("initial_checkpoint")
    if initial_checkpoint and not (out / "RESUME.json").exists():
        candidate = eqx.tree_deserialise_leaves(initial_checkpoint, candidate)
    scheduled = bool(settings.get("lr_decay_every"))
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(1.0 if scheduled else settings["learning_rate"], weight_decay=1e-6),
    )
    state = optimizer.init(eqx.filter(candidate, eqx.is_inexact_array))
    start = 0
    best = float("inf")
    if (out / "RESUME.json").exists():
        r = read(out / "RESUME.json")
        state_path = out / r["state_file"]
        if sha(state_path) != r["state_sha256"]:
            raise ValueError("optimizer checkpoint integrity failed")
        candidate, state = eqx.tree_deserialise_leaves(state_path, (candidate, state))
        start, best = r["epoch"], r["best_nll"]

    @eqx.filter_jit
    def step(net, opt, f, t, valid, rate):
        def loss(nn):
            return jnp.sum(loss_function(nn, f, t) * valid) / jnp.sum(valid)

        value, grad = eqx.filter_value_and_grad(loss)(net)
        updates, opt = optimizer.update(
            grad, opt, eqx.filter(net, eqx.is_inexact_array)
        )
        if scheduled:
            updates = jax.tree_util.tree_map(lambda value: rate * value, updates)
        return eqx.apply_updates(net, updates), opt, value

    @eqx.filter_jit
    def evaluate(net, f, t):
        return loss_function(net, f, t)

    rng = np.random.default_rng(settings["seed"])
    vf, vt = validation
    fixed_vi = None
    if settings.get("fixed_validation"):
        fixed_vi = rng.choice(
            len(vf), min(len(vf), settings["validation_limit"]), replace=False
        )
        np.save(out / "validation_positions.npy", fixed_vi)
    if initial_checkpoint and start == 0 and not (out / "RESUME.json").exists():
        vi = (
            fixed_vi
            if fixed_vi is not None
            else np.arange(min(len(vf), settings["validation_limit"]))
        )
        values = np.concatenate(
            [
                np.asarray(
                    evaluate(candidate, jnp.asarray(vf[idx]), jnp.asarray(vt[idx]))
                )
                for idx in np.array_split(vi, max(1, int(np.ceil(len(vi) / batch))))
            ]
        )
        best = float(values.mean())
        if not np.isfinite(best):
            raise FloatingPointError("nonfinite initial validation loss")
        eqx.tree_serialise_leaves(out / "best.eqx", candidate)
        if settings.get("save_validation_losses"):
            np.savez(out / "best_validation_losses.npz", positions=vi, nll=values)
        write(
            out / "INITIAL_VALIDATION.json",
            dict(nll=best, checkpoint=str(initial_checkpoint), optimizer_reset=True),
        )
    for epoch in range(start, settings["epochs"]):
        rate = max(
            settings.get("minimum_lr", 0.0),
            settings["learning_rate"]
            * settings.get("lr_decay_factor", 0.5)
            ** (epoch // settings.get("lr_decay_every", settings["epochs"])),
        )
        order = np.random.default_rng(settings["seed"] + epoch).permutation(
            len(features)
        )
        losses = []
        for offset in range(0, len(order), batch):
            idx = order[offset : offset + batch]
            n = len(idx)
            idx = np.pad(idx, (0, batch - n), mode="wrap")
            candidate, state, value = step(
                candidate,
                state,
                jnp.asarray(features[idx]),
                jnp.asarray(targets[idx]),
                jnp.arange(batch) < n,
                jnp.asarray(rate),
            )
            value = float(value)
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite supervised loss")
            losses.append(value)
        vi = (
            fixed_vi
            if settings.get("fixed_validation")
            else rng.choice(
                len(vf), min(len(vf), settings["validation_limit"]), replace=False
            )
        )
        values = []
        for offset in range(0, len(vi), batch):
            idx = vi[offset : offset + batch]
            values.extend(
                np.asarray(
                    evaluate(candidate, jnp.asarray(vf[idx]), jnp.asarray(vt[idx]))
                ).tolist()
            )
        score = float(np.mean(values))
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite validation loss")
        if score < best:
            best = score
            eqx.tree_serialise_leaves(out / "best.next.eqx", candidate)
            (out / "best.next.eqx").replace(out / "best.eqx")
            if settings.get("save_validation_losses"):
                np.savez(out / "best_validation_losses.npz", positions=vi, nll=values)
        state_path = out / f"state_{(epoch+1)%2}.eqx"
        eqx.tree_serialise_leaves(state_path, (candidate, state))
        write(
            out / "RESUME.next.json",
            dict(
                epoch=epoch + 1,
                best_nll=best,
                state_file=state_path.name,
                state_sha256=sha(state_path),
            ),
        )
        (out / "RESUME.next.json").replace(out / "RESUME.json")
        record = dict(
            epoch=epoch + 1,
            train_nll=float(np.mean(losses)),
            validation_nll=score,
            best_nll=best,
            learning_rate=rate if scheduled else settings["learning_rate"],
            validation_nll_q50=float(np.median(values)),
            validation_nll_q99=float(np.quantile(values, 0.99)),
            validation_nll_max=float(np.max(values)),
        )
        with (out / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        write(out / "PROGRESS.json", record)
        print(json.dumps(record), flush=True)
    return eqx.tree_deserialise_leaves(out / "best.eqx", candidate)


def classify(net, features, batch=4096):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    predict = eqx.filter_jit(lambda n, f: jax.nn.log_softmax(jax.vmap(n)(f), axis=-1))
    return np.concatenate(
        [
            np.asarray(predict(net, jnp.asarray(features[i : i + batch])))
            for i in range(0, len(features), batch)
        ]
    )


def population(root):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.data import load_photometry_arrays_from_config
    from euclid_dsps.amortized.features import make_encoder_features, read_feature_stats
    from euclid_dsps.config import load_config

    m, cfg = contract(root)
    out = root / "population"
    basis = basis_for(root)
    bank = load_banks(
        root,
        "reference",
        range(cfg["reference_shards"]),
        ("component", "selected", "features"),
    )
    efficiency = selection_efficiencies(
        bank["component"],
        bank["selected"],
        basis.components,
        **cfg["selection_support"],
    )
    if np.any(efficiency["alpha"] == 0):
        raise ValueError(
            "zero selected reference component: parent unidentifiable, no inverse floor allowed"
        )
    selected = np.flatnonzero(bank["selected"])
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(selected)
    split, split_test = int(0.7 * len(selected)), int(0.85 * len(selected))
    train, val, test = (
        selected[:split],
        selected[split:split_test],
        selected[split_test:],
    )
    labels = bank["component"]
    frequencies = np.bincount(labels[train], minlength=basis.components) / len(train)
    if np.any(frequencies == 0):
        raise ValueError("empty classifier training class")
    cs = {**cfg["classifier"], "seed": cfg["seed"]}
    classifier = classifier_template(bank["features"].shape[1], basis.components, cs)

    def loss(net, f, t):
        logp = jax.nn.log_softmax(jax.vmap(net)(f), axis=-1)
        return -jnp.take_along_axis(logp, t[:, None], axis=1)[:, 0]

    classifier = supervised_fit(
        classifier,
        loss,
        bank["features"][train],
        labels[train],
        (bank["features"][val], labels[val]),
        cs,
        out,
    )
    held = classify(classifier, bank["features"][test])
    nll = float(-held[np.arange(len(test)), labels[test]].mean())
    null = float(-np.log(frequencies[labels[test]]).mean())
    if nll >= null:
        raise RuntimeError(
            "classifier does not improve held-out selected-frequency predictor"
        )
    np.savez(
        out / "classifier_validation.npz",
        log_prob=held,
        component=labels[test],
        frequencies=frequencies,
    )
    source = Path(m["source"])
    source_m = read(source / "MANIFEST.json")
    config = load_config(source / "source_config.yaml")
    config["truth"] = {"parameter_columns": {}}
    stats = read_feature_stats(source_m["source"]["feature_stats"])
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10000, row_indices=np.load(source / "train.npy")
    )
    features = np.asarray(
        make_encoder_features(arrays.flux, arrays.flux_err, stats, arrays.mask)
    )
    band = stats.band_names.index("lsst_r")
    flux_min = 10 ** (-0.4 * (cfg["cut"] + 48.6))
    if not np.all(arrays.mask[:, band] & (arrays.flux[:, band] > flux_min)):
        raise ValueError(
            "population fit contains observations outside declared selection"
        )
    logc = classify(classifier, features)
    # Retain solver inputs even if the independent convergence gate fails.
    np.savez(
        out / "observed_ratios.npz", log_classifier=logc, row_index=arrays.row_index
    )
    pd.DataFrame(
        dict(component=np.arange(basis.components), c=frequencies, **efficiency)
    ).to_csv(out / "component_selection.csv", index=False)
    write(
        out / "PROGRESS.json",
        dict(stage="selected_mixture_fit", observations=len(features)),
    )
    print(
        f"[population] selected mixture fit objects={len(features)} KKT tolerance=2e-6",
        flush=True,
    )
    v, diagnostics = fit_selected_weights(
        logc,
        frequencies,
        alpha=efficiency["alpha"],
        eligible=efficiency["eligible"],
        weak_parent_mass=cfg["weak_parent_mass_cap"],
    )
    u = parent_from_selected(v, efficiency["alpha"])
    frame = pd.DataFrame(
        dict(
            component=np.arange(basis.components), v=v, u=u, c=frequencies, **efficiency
        )
    )
    frame.to_csv(out / "component_weights.csv", index=False)
    write(
        out / "parent.json",
        dict(
            u=u.tolist(),
            v=v.tolist(),
            alpha=efficiency["alpha"].tolist(),
            c=frequencies.tolist(),
            alpha_parent=float(u @ efficiency["alpha"]),
            basis_sha256=sha(root / "basis.npz"),
            **diagnostics,
        ),
    )
    print(f"[population] selected fit certified: {diagnostics}", flush=True)
    write(out / "PROGRESS.json", dict(stage="synthetic_family_closure", **diagnostics))
    # Held-out synthetic family closure. Its draws never train the classifier.
    known_u = rng.dirichlet(np.full(basis.components, 2.0))
    eligible = efficiency["eligible"]
    if not np.all(eligible):
        known_u *= eligible
        known_u /= known_u.sum()
    known_v = known_u * efficiency["alpha"]
    known_v /= known_v.sum()
    val_frequency = np.bincount(labels[test], minlength=basis.components) / len(test)
    if np.any(val_frequency == 0):
        raise ValueError("empty independent classifier test class")
    reweight = known_v[labels[test]] / val_frequency[labels[test]]
    recovered, closure = fit_selected_weights(
        held,
        frequencies,
        alpha=efficiency["alpha"],
        eligible=eligible,
        weak_parent_mass=cfg["weak_parent_mass_cap"],
        observation_weights=reweight,
    )
    reconstructed = parent_from_selected(recovered, efficiency["alpha"])
    pd.DataFrame(
        dict(
            component=np.arange(basis.components),
            true_parent=known_u,
            inferred_parent=reconstructed,
            true_selected=known_v,
            inferred_selected=recovered,
        )
    ).to_csv(out / "simulation_closure.csv", index=False)
    finish(
        out,
        "FORWARD_PARENT_COMPLETE",
        parent_sha256=sha(out / "parent.json"),
        classifier_sha256=sha(out / "best.eqx"),
        classifier_validation_nll=nll,
        classifier_null_nll=null,
        synthetic_parent_weight_l1=float(abs(known_u - reconstructed).sum()),
        synthetic_kkt_gap=closure["kkt_gap"],
        observations=len(features),
        population_uses_q=False,
        simulation_bank_target="parent",
        scientific_promotion=False,
        **diagnostics,
    )


def train_posterior(root):
    from euclid_dsps.amortized.avi_experiments import log_prob
    from euclid_dsps.amortized.forward_population_runtime import (
        load_forward_runtime,
        posterior_template,
    )

    m, cfg = contract(root)
    out = root / "posterior"
    frozen_hash = sha(root / "population/parent.json")
    model, runtime, _, config = load_forward_runtime(
        Path(m["source"]), out / "runtime", cfg
    )
    model = attach_frozen_parent(root, model)
    candidate = posterior_template(model, runtime, config, cfg)
    train = load_banks(
        root,
        "posterior_bank",
        range(cfg["posterior_shards"]),
        ("x", "features", "selected"),
    )
    # Separate forward shard: no fitted-catalogue truths and no reused train draws.
    val = load_banks(
        root, "posterior_bank", [cfg["posterior_shards"]], ("x", "features", "selected")
    )
    ti = train["selected"]
    vi = val["selected"] & (np.arange(len(val["selected"])) < len(val["selected"]) // 2)
    if min(ti.sum(), vi.sum()) < cfg["posterior"]["batch_size"]:
        raise ValueError("not enough selected supervised simulations")

    def loss(net, f, t):
        return -log_prob(model, net, f, t[None])[0]

    supervised_fit(
        candidate,
        loss,
        train["features"][ti],
        train["x"][ti],
        (val["features"][vi], val["x"][vi]),
        {**cfg["posterior"], "seed": cfg["seed"]},
        out,
    )
    if sha(root / "population/parent.json") != frozen_hash:
        raise RuntimeError("population changed during posterior training")
    finish(
        out,
        "FORWARD_POSTERIOR_COMPLETE",
        checkpoint_sha256=sha(out / "best.eqx"),
        parent_sha256=frozen_hash,
        selected_training_pairs=int(ti.sum()),
        supervised_target="15D forward simulator latent",
        rws_updates=0,
        population_feedback=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "preflight",
            "reference",
            "population",
            "posterior-bank",
            "train",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_forward_population_r29.yaml"),
    )
    parser.add_argument("--truth-parent", type=Path)
    parser.add_argument("--truth-selected", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if args.source is None:
            parser.error("--source required")
        prepare(
            args.source,
            root,
            args.config,
            args.truth_parent,
            args.truth_selected,
            args.baseline,
        )
    elif args.mode == "preflight":
        preflight(root)
    elif args.mode in {"reference", "posterior-bank"}:
        bank(root, args.task, "reference" if args.mode == "reference" else "posterior")
    elif args.mode == "population":
        population(root)
    elif args.mode == "train":
        train_posterior(root)
    else:
        from scripts.report_feniks_forward_population import report

        report(root)


if __name__ == "__main__":
    main()
