"""Two bounded experiments: density representation and matched forward closure.

No production prior is changed. Truth training is confined to the representation
oracle; closure estimation sees only selected simulated observation features.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_weighted_truth_flow_capacity import _normalized_weights, _status


def contract(root):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input changed: {item['path']}")
    if "parent_manifest" in manifest["inputs"]:
        parent = read(Path(manifest["inputs"]["parent_manifest"]["path"]))
        for path, digest in parent["hashes"].items():
            if sha(Path(path)) != digest:
                raise ValueError(
                    f"upstream simulator/observation input changed: {path}"
                )
    return manifest, manifest["settings"]


def prepare(capacity, root, config):
    from scripts.feniks_weighted_truth_flow_capacity import (
        contract as capacity_contract,
    )

    cm, _ = capacity_contract(capacity)
    parent = Path(cm["parent"])
    _status(
        capacity / "report/FINAL.json", "WEIGHTED_TRUTH_FLOW_CAPACITY_REPORT_COMPLETE"
    )
    _status(parent / "population/FINAL.json", "FORWARD_PARENT_COMPLETE")
    settings = yaml.safe_load(config.read_text())
    if root.exists():
        raise FileExistsError(root)
    if settings["replicas"] != 2:
        raise ValueError("this controlled comparison uses two replicas per arm")
    for key in ("exact_repeats", "metric_draws", "bootstrap_reference"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("parent_simulations", "shard_size", "minimum_selected", "bootstraps"):
        if (
            not isinstance(settings["closure"][key], int)
            or settings["closure"][key] <= 0
        ):
            raise ValueError(f"closure.{key} must be a positive integer")
    if (
        not isinstance(settings["contracts"]["decoder_objects"], int)
        or settings["contracts"]["decoder_objects"] <= 0
    ):
        raise ValueError("contracts.decoder_objects must be a positive integer")
    parent_manifest = read(parent / "MANIFEST.json")
    from scripts.feniks_failure_modes import _selection_identities, _stratified_indices

    selection_path = _selection_identities(parent_manifest)
    truth_path = Path(cm["inputs"]["truth_parent"]["path"])
    truth = pd.read_parquet(truth_path)
    selection = pd.read_parquet(selection_path)
    audit_indices = _stratified_indices(
        truth,
        selection,
        settings["contracts"]["decoder_objects"],
        settings["seed"],
    )
    root.mkdir(parents=True)
    for path in (
        "logs",
        "contracts",
        "decoder_observation",
        "report",
        "closure",
        "joint/replica_0",
        "joint/replica_1",
        "structured/replica_0",
        "structured/replica_1",
    ):
        (root / path).mkdir(parents=True)
    np.save(root / "audit_indices.npy", audit_indices)
    inputs = {
        "capacity_manifest": capacity / "MANIFEST.json",
        "truth_parent": Path(cm["inputs"]["truth_parent"]["path"]),
        "selection_identities": selection_path,
        "audit_indices": root / "audit_indices.npy",
        "split": capacity / "split.npz",
        "parent_manifest": parent / "MANIFEST.json",
        "basis": parent / "basis.npz",
        "classifier": parent / "population/best.eqx",
        "component_weights": parent / "population/component_weights.csv",
        "config": config,
    }
    write(
        root / "MANIFEST.json",
        dict(
            source=cm["source"],
            parent=str(parent),
            settings=settings,
            inputs={
                name: dict(path=str(path.resolve()), sha256=sha(path))
                for name, path in inputs.items()
            },
            production_prior_modified=False,
            population_uses_q=False,
            truth_role="representation oracle only; closure fitting uses selected features",
            closure_scope="existing component family, matched simulator, frozen classifier",
            contracts_scope=(
                "weighted target, theta/x transform, decoder, photometric errors, "
                "noise, masks and observed-r selection"
            ),
        ),
    )
    print(yaml.safe_dump(settings, sort_keys=False))
    print(
        "One contract audit; four representation tasks; one matched-closure task; "
        "no production promotion."
    )


def _runtime(m, out):
    from euclid_dsps.amortized.forward_population_runtime import load_forward_runtime

    pm = read(Path(m["parent"]) / "MANIFEST.json")
    return load_forward_runtime(Path(m["source"]), out / "runtime", pm["settings"])


def audit_contracts(root):
    """Close data/transform/observation contracts before interpreting fits."""
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import theta_to_x, x_to_theta
    from scripts.feniks_failure_modes import decoder_observation

    m, cfg = contract(root)
    out = root / "contracts"
    write(out / "PROGRESS.json", dict(stage="transform_and_weight_contract"))
    model, runtime, _, _ = _runtime(m, out)
    del model
    names = tuple(runtime.latent_spec.names)
    truth = pd.read_parquet(m["inputs"]["truth_parent"]["path"])
    theta = truth[list(names)].to_numpy(np.float64)
    weights = _normalized_weights(truth.population_weight)
    x = np.asarray(theta_to_x(jnp.asarray(theta), runtime.latent_spec))
    restored = np.asarray(x_to_theta(jnp.asarray(x), runtime.latent_spec))
    iqr = np.maximum(np.subtract(*np.percentile(theta, [75, 25], axis=0)), 1e-8)
    relative = np.max(np.abs(restored - theta) / iqr, axis=0)
    bounds = np.mean(
        (theta <= np.asarray(runtime.latent_spec.lower))
        | (theta >= np.asarray(runtime.latent_spec.upper)),
        axis=0,
    )
    pd.DataFrame(
        dict(
            parameter=names,
            max_roundtrip_over_iqr=relative,
            at_or_outside_bounds=bounds,
        )
    ).to_csv(out / "transform_contract.csv", index=False)
    selection = pd.read_parquet(m["inputs"]["selection_identities"]["path"]).set_index(
        "parent_row_index"
    )
    selected = selection.loc[truth.parent_row_index, "selected"].to_numpy(bool)
    weight_summary = dict(
        objects=len(truth),
        weight_sum=float(truth.population_weight.sum()),
        normalized_weight_sum=float(weights.sum()),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
        weight_ess=float(1 / np.sum(weights**2)),
        unweighted_selected_fraction=float(selected.mean()),
        weighted_selected_fraction=float(weights[selected].sum()),
        representation_target="population_weight-normalized parent truth",
        observed_catalogue_target="one row per selected catalogue object",
        targets_are_interchangeable=False,
    )
    write(out / "weight_contract.json", weight_summary)
    if (
        not np.isfinite(x).all()
        or not np.isclose(weights.sum(), 1.0, atol=1e-12)
        or np.max(relative) > cfg["transform_tolerance"]
    ):
        raise ValueError("transform/weight numerical contract failed")

    # Reuse the already-audited implementation against the historical catalogue.
    # It writes detailed per-band residuals and selection identity checks. Its
    # scientific gates are reported below rather than raising, so tests 2/3 can
    # still isolate representation and matched-simulator recovery.
    cfg["decoder_objects"] = cfg["contracts"]["decoder_objects"]
    original = copy.deepcopy(m["settings"])
    try:
        m["settings"] = cfg
        write(root / "MANIFEST.json", m)
        decoder_observation(root)
    finally:
        m["settings"] = original
        write(root / "MANIFEST.json", m)
    decoder = read(root / "decoder_observation/FINAL.json")
    bands = pd.read_csv(root / "decoder_observation/band_summary.csv")
    gates = cfg["contracts"]
    decisions = dict(
        target_weighting=True,
        transform=bool(np.max(relative) <= cfg["transform_tolerance"]),
        decoder=bool(
            (
                bands.decoder_p95_abs_sigma <= gates["decoder_p95_abs_residual_sigma"]
            ).all()
        ),
        reported_error=bool(
            (
                bands.error_true_p95_abs_relative
                <= gates["error_p95_relative_residual"]
            ).all()
        ),
        noise=bool(
            (abs(bands.noise_mean) <= gates["noise_mean_abs"]).all()
            and (abs(bands.noise_std - 1) <= gates["noise_std_abs_from_one"]).all()
        ),
        selection_identity=decoder["selection_identity_mismatches"] == 0,
    )
    write(
        out / "FINAL.json",
        dict(
            status="CLEAN_CONTRACT_AUDIT_COMPLETE",
            decisions=decisions,
            all_scientific_gates_pass=all(decisions.values()),
            numerical_integrity_pass=True,
            historical_observation_contract_pass=all(
                decisions[k]
                for k in ("decoder", "reported_error", "noise", "selection_identity")
            ),
            decoder_summary=decoder,
            **weight_summary,
        ),
    )


def fit(root, task):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.forward_population import PHYSICAL
    from euclid_dsps.amortized.forward_population_runtime import posterior_template
    from euclid_dsps.amortized.latent import theta_to_x, x_to_theta
    from euclid_dsps.amortized.proposal_expressivity import count_parameters
    from euclid_dsps.amortized.structured_population import (
        StructuredPopulation,
        factor_log_prob,
        factor_sample,
        factor_template,
        transport_audit,
    )
    from scripts.feniks_forward_population import supervised_fit
    from scripts.feniks_weighted_truth_flow_followup import _weighted_loss_scale
    from scripts.report_feniks_forward_population import population_metrics

    m, cfg = contract(root)
    if task not in range(4):
        raise ValueError("representation task must be 0..3")
    arm, replica = ("joint" if task < 2 else "structured"), task % 2
    out = root / arm / f"replica_{replica}"
    write(out / "PROGRESS.json", dict(stage="loading"))
    model, runtime, _, source_config = _runtime(m, out)
    names = tuple(runtime.latent_spec.names)
    a = np.array([names.index(n) for n in PHYSICAL])
    b = np.array([i for i in range(15) if i not in a])
    truth = pd.read_parquet(m["inputs"]["truth_parent"]["path"])
    theta = truth[list(names)].to_numpy(np.float64)
    weights = _normalized_weights(truth.population_weight)
    x = np.asarray(theta_to_x(jnp.asarray(theta), runtime.latent_spec))
    back = np.asarray(x_to_theta(jnp.asarray(x), runtime.latent_spec))
    iqr = np.maximum(np.subtract(*np.percentile(theta, [75, 25], axis=0)), 1e-8)
    residual = np.max(np.abs(theta - back) / iqr, axis=0)
    pd.DataFrame(
        dict(
            parameter=names,
            max_roundtrip_over_iqr=residual,
            at_or_outside_bounds=np.mean(
                (theta <= np.asarray(runtime.latent_spec.lower))
                | (theta >= np.asarray(runtime.latent_spec.upper)),
                axis=0,
            ),
        )
    ).to_csv(out / "transform_contract.csv", index=False)
    if not np.isfinite(x).all() or np.max(residual) > cfg["transform_tolerance"]:
        raise ValueError(
            "truth transform round trip failed; inspect transform_contract.csv"
        )
    with np.load(m["inputs"]["split"]["path"]) as f:
        train, val, test = (f[k] for k in ("train", "validation", "test"))
    if len(np.unique(np.concatenate([train, val, test]))) != len(theta):
        raise ValueError("identity split is not a partition")
    ids = np.tile(train, cfg["exact_repeats"])
    seed = cfg["seed"] + replica
    parent_settings = copy.deepcopy(
        read(Path(m["parent"]) / "MANIFEST.json")["settings"]
    )
    parent_settings["seed"] = seed
    # Preserve production encoder overrides for joint and factor templates.
    architecture = copy.deepcopy(source_config["amortized"]["encoder"])
    architecture.update(parent_settings["posterior"]["architecture"])
    fs = dict(
        experts=parent_settings["posterior"]["experts"], architecture=architecture
    )
    fit_settings = dict(
        cfg["flow"],
        seed=seed,
        fixed_validation=True,
        validation_limit=len(val),
        save_validation_losses=True,
    )

    def train_factor(candidate, context, target, directory):
        directory.mkdir(exist_ok=True)
        targets = np.column_stack([target[ids], _weighted_loss_scale(weights, ids)])
        validation = np.column_stack([target[val], _weighted_loss_scale(weights, val)])

        def loss(net, f, t):
            return -factor_log_prob(net, f, t[:, :-1]) * t[:, -1]

        return supervised_fit(
            candidate,
            loss,
            context[ids],
            targets,
            (context[val], validation),
            fit_settings,
            directory,
        )

    zero = np.zeros((len(x), 1), np.float32)
    if arm == "joint":
        candidate = posterior_template(model, runtime, source_config, parent_settings)
        zero = np.zeros((len(x), candidate.input_dim), np.float32)
        net = train_factor(candidate, zero, x, out / "joint")
        audits = {
            "joint": transport_audit(
                net,
                jnp.asarray(zero[test[:16]]),
                jax.random.PRNGKey(seed),
                values=jnp.asarray(x[test[:16]]),
            )
        }

        def sample(key, count):
            return factor_sample(net, jnp.zeros((count, net.input_dim)), key)

        def lp(values):
            return factor_log_prob(net, jnp.zeros((len(values), net.input_dim)), values)

        checkpoints = {"joint": sha(out / "joint/best.eqx")}
    else:
        physical = train_factor(
            factor_template(fs, 5, 1, seed), zero, x[:, a], out / "physical"
        )
        conditional = train_factor(
            factor_template(fs, 10, 5, seed + 100),
            x[:, a],
            x[:, b],
            out / "conditional",
        )
        net = StructuredPopulation(physical, conditional, names)
        audits = {
            "physical": transport_audit(
                physical,
                jnp.asarray(zero[test[:16]]),
                jax.random.PRNGKey(seed),
                values=jnp.asarray(x[test[:16]][:, a]),
            ),
            "conditional": transport_audit(
                conditional,
                jnp.asarray(x[test[:16]][:, a]),
                jax.random.PRNGKey(seed),
                values=jnp.asarray(x[test[:16]][:, b]),
            ),
        }
        sample, lp = net.sample, net.log_prob
        checkpoints = {
            k: sha(out / k / "best.eqx") for k in ("physical", "conditional")
        }
    write(out / "transport_contract.json", audits)
    if not all(v["passed"] for v in audits.values()):
        raise ValueError("trained transport failed; not a valid capacity verdict")
    draw = eqx.filter_jit(lambda key: sample(key, 512))
    count = cfg["metric_draws"]
    flow_x = np.concatenate(
        [
            np.asarray(draw(jax.random.PRNGKey(seed + 10000 + i)))
            for i in range((count + 511) // 512)
        ]
    )[:count]
    flow_theta = np.asarray(x_to_theta(jnp.asarray(flow_x), runtime.latent_spec))
    rng = np.random.default_rng(cfg["metric_seed"])
    target_ids = rng.choice(test, count, p=_normalized_weights(weights[test]))
    marginal, joint = [], []
    for space, pred, target in (
        ("latent_x", flow_x, x[target_ids]),
        ("physical_theta", flow_theta, theta[target_ids]),
    ):
        mm, jj = population_metrics(pred, target, names, cfg["metric_seed"])
        mm["space"], jj["space"] = space, space
        marginal.append(mm)
        joint.append(jj)
    pd.concat(marginal).to_csv(out / "marginal.csv", index=False)
    pd.concat(joint).to_csv(out / "joint.csv", index=False)
    evaluate = eqx.filter_jit(lp)
    nll = np.concatenate(
        [
            -np.asarray(evaluate(jnp.asarray(x[t])))
            for t in np.array_split(test, max(1, (len(test) + 511) // 512))
        ]
    )
    np.savez_compressed(
        out / "draws.npz",
        flow_x=flow_x,
        flow_theta=flow_theta,
        target_x=x[target_ids],
        target_theta=theta[target_ids],
        names=names,
    )
    write(
        out / "FINAL.json",
        dict(
            status="CLEAN_REPRESENTATION_COMPLETE",
            arm=arm,
            replica=replica,
            heldout_weighted_nll=float(np.average(nll, weights=weights[test])),
            checkpoints=checkpoints,
            parameter_count=count_parameters(net),
            training_identities=len(train),
            heldout_identities=len(test),
            truth_used_for_training=True,
            dimensions=15,
            production_prior_modified=False,
        ),
    )


def known_parent_weights(basis, eligible, seed):
    """Predeclared physical tilt, independent of catalogue/fit and q."""
    rng = np.random.default_rng(seed)
    tilt = basis.centers @ rng.normal(size=5)
    tilt = np.exp(tilt / max(np.std(tilt), 1e-8))
    weights = 0.5 / basis.components + 0.5 * tilt / tilt.sum()
    # Restrict the experiment to declared reconstructible support, not a floor
    # on inverse selection. Removed mass is reported separately.
    removed = float(weights[~eligible].sum())
    weights[~eligible] = 0
    return _normalized_weights(weights), removed


def likelihood_geometry(logc, frequencies, v):
    """Curvature in the simplex tangent; small eigenvalues imply weak weights."""
    from scipy.linalg import null_space

    d = np.exp(
        logc
        - np.log(frequencies)[None]
        - np.max(logc - np.log(frequencies)[None], axis=1, keepdims=True)
    )
    score = d / (d @ v)[:, None]
    tangent = null_space(np.ones((1, len(v))))
    information = tangent.T @ (score.T @ score / len(score)) @ tangent
    eigenvalues = np.linalg.eigvalsh(information)
    return dict(
        eigenvalues=eigenvalues.tolist(),
        near_null_directions=int(
            np.sum(eigenvalues < max(eigenvalues[-1], 1e-15) * 1e-6)
        ),
        boundary_solution=bool(np.any(v < 1e-8)),
        interpretation="weight curvature, not physical-density uncertainty",
    )


def closure(root):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.forward_population import (
        fit_selected_weights,
        parent_from_selected,
    )
    from euclid_dsps.amortized.forward_population_runtime import simulator
    from euclid_dsps.amortized.latent import x_to_theta
    from scripts.feniks_forward_population import (
        basis_for,
        classifier_template,
        classify,
    )
    from scripts.report_feniks_forward_population import population_metrics

    m, cfg = contract(root)
    out = root / "closure"
    parent = Path(m["parent"])
    pm = read(parent / "MANIFEST.json")
    model, runtime, observation, _ = _runtime(m, out)
    basis = basis_for(parent)
    table = pd.read_csv(m["inputs"]["component_weights"]["path"])
    if not np.array_equal(table.component.to_numpy(), np.arange(basis.components)):
        raise ValueError("component ordering mismatch")
    c, alpha = table.c.to_numpy(), table.alpha.to_numpy()
    eligible = table.eligible.to_numpy(bool)
    u_true, removed = known_parent_weights(basis, eligible, cfg["seed"] + 200)
    simulate = simulator(model, runtime, observation)
    bankdir = out / "bank"
    bankdir.mkdir(exist_ok=True)
    count = cfg["closure"]["parent_simulations"]
    shard = cfg["closure"]["shard_size"]
    for offset in range(0, count, shard):
        path = bankdir / f"{offset:09d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng(cfg["seed"] + 100000 + offset)
        values, _ = basis.sample(rng, min(shard, count - offset), u_true)
        chunks = []
        batch = pm["settings"]["simulation_batch"]
        for start in range(0, len(values), batch):
            xx = values[start : start + batch]
            n = len(xx)
            xx = np.pad(xx, ((0, batch - n), (0, 0)), mode="wrap")
            data = simulate(
                jnp.asarray(xx),
                jax.random.PRNGKey(cfg["seed"] + 200000 + offset + start),
            )
            if not np.asarray(data["valid"])[:n].all():
                raise ValueError(
                    "invalid forward draws: cannot silently redefine the parent"
                )
            chunks.append(
                {
                    k: np.asarray(data[k])[:n]
                    for k in ("features", "selected", "theta", "x")
                }
            )
        arrays = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
        temporary = path.with_suffix(".next.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        write(
            out / "PROGRESS.json",
            dict(stage="forward", done=offset + len(values), total=count),
        )
    banks = [
        np.load(p) for p in sorted(bankdir.glob("*.npz")) if ".next." not in p.name
    ]
    data = {
        k: np.concatenate([p[k] for p in banks])
        for k in ("features", "selected", "theta", "x")
    }
    for p in banks:
        p.close()
    if len(data["theta"]) != count:
        raise ValueError("bank row count mismatch")
    # Disjoint fresh forward halves: estimation and predictive validation.
    midpoint = count // 2
    ids = np.flatnonzero(data["selected"][:midpoint])
    held = np.flatnonzero(data["selected"][midpoint:]) + midpoint
    if min(len(ids), len(held)) < cfg["closure"]["minimum_selected"]:
        raise ValueError("insufficient selected objects for matched closure")
    template = classifier_template(
        data["features"].shape[1],
        basis.components,
        {**pm["settings"]["classifier"], "seed": pm["settings"]["seed"]},
    )
    classifier = eqx.tree_deserialise_leaves(
        m["inputs"]["classifier"]["path"], template
    )
    logc = classify(classifier, data["features"][ids])
    logc_held = classify(classifier, data["features"][held])
    np.savez_compressed(out / "ratios.npz", train=logc, heldout=logc_held)
    v, diagnostics = fit_selected_weights(
        logc,
        c,
        alpha=alpha,
        eligible=eligible,
        weak_parent_mass=pm["settings"]["weak_parent_mass_cap"],
    )
    u = parent_from_selected(v, alpha)
    v_true = _normalized_weights(u_true * alpha)
    pd.DataFrame(
        dict(
            component=np.arange(len(u)),
            true_parent=u_true,
            fitted_parent=u,
            true_selected_using_reference_alpha=v_true,
            fitted_selected=v,
            alpha=alpha,
        )
    ).to_csv(out / "weights.csv", index=False)
    write(out / "likelihood_geometry.json", likelihood_geometry(logc, c, v))
    rng = np.random.default_rng(cfg["metric_seed"])
    draws = cfg["metric_draws"]
    learned_x, _ = basis.sample(rng, draws, u)
    learned = np.asarray(x_to_theta(jnp.asarray(learned_x), runtime.latent_spec))
    truth = data["theta"][midpoint:]
    target = truth[rng.choice(len(truth), draws)]
    mm, jj = population_metrics(
        learned, target, tuple(runtime.latent_spec.names), cfg["metric_seed"]
    )
    mm.to_csv(out / "parent_marginal.csv", index=False)
    jj.to_csv(out / "parent_joint.csv", index=False)
    # Independent forward validation weighted p_fit/p_true; all nuisance draws
    # stay intact. Diagnostic IS is not a q-based population update.
    from scipy.special import logsumexp

    vx = data["x"][held]
    terms = basis.component_log_prob(vx)
    lr = logsumexp(terms + np.log(np.maximum(u, 1e-300)), axis=1) - logsumexp(
        terms + np.log(np.maximum(u_true, 1e-300)), axis=1
    )
    iw = np.exp(lr - lr.max())
    iw /= iw.sum()
    selected_fit = data["theta"][rng.choice(held, draws, p=iw)]
    selected_true = data["theta"][rng.choice(held, draws)]
    mm, jj = population_metrics(
        selected_fit,
        selected_true,
        tuple(runtime.latent_spec.names),
        cfg["metric_seed"],
    )
    mm.to_csv(out / "selected_marginal.csv", index=False)
    jj.to_csv(out / "selected_joint.csv", index=False)
    np.savez_compressed(
        out / "draws.npz",
        learned_parent=learned,
        true_parent=target,
        learned_selected=selected_fit,
        true_selected=selected_true,
        names=runtime.latent_spec.names,
    )
    from scipy.stats import wasserstein_distance

    features = data["features"][held]
    pd.DataFrame(
        [
            dict(
                feature=i,
                w1=wasserstein_distance(features[:, i], features[:, i], v_weights=iw),
            )
            for i in range(features.shape[1])
        ]
    ).to_csv(out / "observable_closure.csv", index=False)
    boot = []
    for i in range(cfg["closure"]["bootstraps"]):
        vb, db = fit_selected_weights(
            logc[rng.integers(len(logc), size=len(logc))],
            c,
            alpha=alpha,
            eligible=eligible,
            weak_parent_mass=pm["settings"]["weak_parent_mass_cap"],
        )
        ub = parent_from_selected(vb, alpha)
        shift = logc_held - np.log(c)
        held_ll = np.mean(logsumexp(shift + np.log(np.maximum(vb, 1e-300)), axis=1))
        bx, _ = basis.sample(rng, min(draws, 4096), ub)
        bt = np.asarray(x_to_theta(jnp.asarray(bx), runtime.latent_spec))
        boot.append(
            dict(
                replicate=i,
                heldout_mean_log_ratio=float(held_ll),
                parent_weight_l1=float(np.abs(ub - u_true).sum()),
                **{
                    f"parent_mean_{runtime.latent_spec.names[j]}": float(
                        bt[:, j].mean()
                    )
                    for j in basis.indices
                },
                **{f"u_{j}": w for j, w in enumerate(ub)},
            )
        )
        write(
            out / "PROGRESS.json",
            dict(stage="bootstrap", done=i + 1, total=cfg["closure"]["bootstraps"]),
        )
    pd.DataFrame(boot).to_csv(out / "bootstrap.csv", index=False)
    write(
        out / "FINAL.json",
        dict(
            status="CLEAN_MATCHED_CLOSURE_COMPLETE",
            **diagnostics,
            parent_simulations=count,
            fitting_selected=len(ids),
            heldout_selected=len(held),
            sum_u=float(u.sum()),
            sum_v=float(v.sum()),
            true_parent_removed_support_mass=removed,
            alpha_true_fresh=float(data["selected"][midpoint:].mean()),
            alpha_true_reference=float(u_true @ alpha),
            alpha_fitted=float(u @ alpha),
            predictive_weight_ess=float(1 / (iw @ iw)),
            classifier_retrained=False,
            uncertainty_scope="catalogue bootstrap only, conditional on classifier and alpha",
            production_ready=False,
            population_uses_q=False,
        ),
    )


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from euclid_dsps.amortized.forward_population import PHYSICAL
    from scripts.feniks_avi_overnight import _corner
    from scripts.report_feniks_forward_population import population_metrics

    m, cfg = contract(root)
    out = root / "report"
    _status(root / "contracts/FINAL.json", "CLEAN_CONTRACT_AUDIT_COMPLETE")
    rows = []
    corner_series = {"latent_x": {}, "physical_theta": {}}
    corner_truth = {}
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for arm in ("joint", "structured"):
        for replica in range(2):
            directory = root / arm / f"replica_{replica}"
            _status(directory / "FINAL.json", "CLEAN_REPRESENTATION_COMPLETE")
            frame = pd.read_csv(directory / "joint.csv")
            frame["arm"], frame["replica"] = arm, replica
            rows.append(frame)
            with np.load(directory / "draws.npz") as d:
                names = d["names"].tolist()
                physical = [names.index(n) for n in PHYSICAL]
                for space, fk, tk in (
                    ("latent_x", "flow_x", "target_x"),
                    ("physical_theta", "flow_theta", "target_theta"),
                ):
                    corner_series[space][f"{arm}_{replica}"] = d[fk][:, physical]
                    corner_truth[space] = d[tk][:, physical]
                for i, ax in enumerate(axes.flat):
                    bounds = np.quantile(d["target_theta"][:, i], [0.001, 0.999])
                    bins = np.linspace(*bounds, 60)
                    ax.hist(
                        d["flow_theta"][:, i],
                        bins=bins,
                        density=True,
                        histtype="step",
                        label=f"{arm} {replica}",
                        alpha=0.8,
                    )
                    if arm == "joint" and replica == 0:
                        ax.hist(
                            d["target_theta"][:, i],
                            bins=bins,
                            density=True,
                            histtype="step",
                            color="black",
                            label="Held-out truth",
                        )
                    ax.set_title(names[i], fontsize=9)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Representation only: weighted held-out truth; central 99.8% plotting range"
    )
    fig.tight_layout()
    fig.savefig(out / "representation_theta_15d.png", dpi=160)
    plt.close(fig)
    for space in corner_series:
        _corner(
            out / f"representation_{space}_corner.png",
            corner_series[space],
            corner_truth[space],
            PHYSICAL,
        )
    pd.concat(rows).to_csv(out / "representation_joint.csv", index=False)
    # This reference reflects finite truth-object sampling, not 32k independent
    # galaxies invented by resampling a 1.5k-object empirical held-out catalogue.
    truth = pd.read_parquet(m["inputs"]["truth_parent"]["path"])
    with np.load(m["inputs"]["split"]["path"]) as split:
        ids = split["test"]
    weights = _normalized_weights(truth.population_weight.to_numpy()[ids])
    values = truth[names].to_numpy()[ids]
    rng = np.random.default_rng(cfg["metric_seed"])
    null = []
    for i in range(cfg["bootstrap_reference"]):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        wb = _normalized_weights(weights * counts)
        p = values[rng.choice(len(ids), cfg["metric_draws"], p=wb)]
        t = values[rng.choice(len(ids), cfg["metric_draws"], p=weights)]
        _, joint = population_metrics(p, t, names, cfg["metric_seed"])
        joint["replicate"] = i
        null.append(joint)
    pd.concat(null).to_csv(out / "truth_object_bootstrap_reference.csv", index=False)
    _status(root / "closure/FINAL.json", "CLEAN_MATCHED_CLOSURE_COMPLETE")
    with np.load(root / "closure/draws.npz") as d:
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
        for ax, name in zip(axes, PHYSICAL, strict=True):
            i = names.index(name)
            bins = np.linspace(*np.quantile(d["true_parent"][:, i], [0.001, 0.999]), 60)
            for k in (
                "true_parent",
                "learned_parent",
                "true_selected",
                "learned_selected",
            ):
                ax.hist(d[k][:, i], bins=bins, density=True, histtype="step", label=k)
            ax.set_title(name)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "matched_parent_selected.png", dpi=160)
        plt.close(fig)
    (out / "REPORT.md").write_text(
        "# Clean parent tests\n\n"
        "Contracts: inspect contracts/FINAL.json and decoder_observation/band_summary.csv. "
        "A failed historical observation gate is a scientific result and does not "
        "invalidate the independent representation/matched-simulator tests.\n\n"
        "Representation: joint 15D vs p(a)p(b|a), disjoint weighted truth identities.\n"
        "Compare representation_joint.csv to truth_object_bootstrap_reference.csv; "
        "the reference is finite-object variation, not a universal acceptance threshold.\n\n"
        "Matched closure: fresh selected forward observations under a known member "
        "of the existing component family. Classifier and reference alpha are frozen. "
        "Inspect closure/weights.csv, parent_joint.csv, observable_closure.csv, "
        "likelihood_geometry.json and bootstrap.csv. Near-null weight directions do "
        "not automatically imply different physical densities. Bootstrap excludes "
        "classifier/selection-efficiency uncertainty.\n\n"
        "Neither test certifies the historical observation law, catalogue inference, "
        "or learning the new continuous structured parent from photometry. "
        "No production prior was modified.\n"
    )
    write(
        out / "FINAL.json",
        dict(status="CLEAN_PARENT_REPORT_COMPLETE", production_ready=False),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "contracts", "fit", "closure", "report")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capacity", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.capacity, args.root, args.config)
    elif args.mode == "contracts":
        audit_contracts(args.root)
    elif args.mode == "fit":
        fit(args.root, args.task)
    elif args.mode == "closure":
        closure(args.root)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
