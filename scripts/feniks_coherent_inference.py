"""Decoupled population inference and two parallel/controlled 15D NPE branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

from euclid_dsps.amortized.coherent_coordinates import (
    fit_coordinates,
    to_theta,
    to_x,
    validate_theta,
)
from euclid_dsps.amortized.features import (
    compute_feature_stats,
    feature_stats_from_json,
    feature_stats_to_json,
    make_encoder_features,
)
from euclid_dsps.amortized.forward_population import (
    parent_from_selected,
    selection_efficiencies,
    simplex,
)
from euclid_dsps.amortized.native_reference import (
    choose_penalty,
    make_basis,
    sample_basis,
    supervised_weights,
)
from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish, photometer
from scripts.feniks_coherent_parent import settings as base_settings

NAMES = list(SPLINE15D_PARAMETER_NAMES)
STAGES = ("reference", "oracle", "population", "posterior", "report")


def settings(root):
    m, cfg, digest = base_settings(root)
    for path, expected in m["source_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Immutable source changed: {path}")
    return m, cfg, digest


def prepare(source, root, config):
    from scripts.audit_feniks_coherent_parent import verify_catalogues

    if root.exists() or source == root or source in root.parents:
        raise ValueError("Use a new root outside the immutable dataset")
    _, counts, original = verify_catalogues(source)
    cfg = yaml.safe_load(config.read_text())
    b, r = cfg["bank"], cfg["reference"]
    if b["shards"] < 2 or b["rows_per_shard"] % r["components"]:
        raise ValueError("At least two balanced bank shards required")
    if b["checkpoint_rows"] % r["components"] or r["anchors"] < r["components"]:
        raise ValueError("Invalid balanced blocks/anchor counts")
    for section in ("resources", "bank", "evaluation"):
        if any(not isinstance(v, int) or v <= 0 for v in cfg[section].values()):
            raise ValueError(f"Positive integer {section} required")
    for name in ("classifier", "posterior"):
        if not cfg[name]["fixed_validation"] or cfg[name]["epochs"] < 1:
            raise ValueError("Fixed validation and bounded training required")
    if cfg["stopping"]["block_epochs"] < 1 or cfg["stopping"]["patience_blocks"] < 1:
        raise ValueError("Invalid stopping budget")
    if any(p <= 0 for p in cfg["population"]["penalties"]):
        raise ValueError("Positive convex regularization penalties required")
    paths = [source / name for name in ("MANIFEST.json", "decoder.json", "noise.json")]
    paths += sorted((source / "dataset").rglob("*.parquet"))
    files = {str(p): sha(p) for p in paths}
    files.update(original["assets"])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "experiment.yaml").write_text(config.read_text())
    write(
        root / "MANIFEST.json",
        dict(
            source=str(source),
            settings=cfg,
            source_files=files,
            frozen_files={"experiment.yaml": sha(root / "experiment.yaml")},
            counts=counts,
            selection=original["settings"]["selection"],
            native_source=original["sources"]["train"],
            population_uses_q=False,
            population_uses_target_truth=False,
            oracle_uses_target_train_truth=True,
            production_promotion=False,
        ),
    )
    print(
        json.dumps(
            dict(
                basis="soft physical gates over disjoint unweighted native anchors; 15D Gaussian kernels",
                reference=r,
                nuisance="joint native anchor SFHs plus full-15D kernel variability",
                simulations=b["shards"] * b["rows_per_shard"],
                selection="same saved noisy r<29; binomial alpha including rejected parents",
                classifier=cfg["classifier"],
                posterior=cfg["posterior"],
                posterior_targets="simulator theta; final bank weights u_j / r_j, never q draws",
                resources=cfg["resources"],
                artifacts="basis, banks, classifiers, v/u/alpha, 15D posteriors, dense draws, comparison, roadmap",
            ),
            indent=2,
        )
    )


def observation_columns(bands):
    return [f"{prefix}_{b}" for prefix in ("flux", "fluxerr", "mask") for b in bands]


def observed(root, split):
    """Population readers have no access to target theta or population weights."""
    m = read(root / "MANIFEST.json")
    source = Path(m["source"])
    bands = [b["name"] for b in read(source / "decoder.json")["bands"]]
    return pd.read_parquet(
        source / f"dataset/selected_r29/{split}.parquet",
        columns=observation_columns(bands),
    ), bands


def features(frame, bands, stats):
    arrays = [
        frame[[f"{p}_{b}" for b in bands]].to_numpy()
        for p in ("flux", "fluxerr", "mask")
    ]
    return np.asarray(make_encoder_features(arrays[0], arrays[1], stats, arrays[2]))


def uniform_anchors(record, excluded, count, seed):
    """Uniform bottom-k reservoir: native weight magnitudes NEVER enter sampling."""
    from euclid_dsps.synthetic_diffsky.coherent_parent import eligible_proposals

    rng, pool, hashes, eligible = np.random.default_rng(seed), None, {}, 0
    for path in record["paths"]:
        p = Path(path)
        hashes[str(p)] = sha(p)
        frame = eligible_proposals(p, record["source_split"], record["source_seed"])
        frame = frame.loc[~frame.effective_proposal_key.isin(excluded)].copy()
        eligible += len(frame)
        frame["_priority"] = rng.random(len(frame))
        pool = (
            pd.concat([pool, frame], ignore_index=True) if pool is not None else frame
        )
        pool = pool.nsmallest(count, "_priority")
    if pool is None or len(pool) < count or not pool.effective_proposal_key.is_unique:
        raise ValueError("Insufficient unique independent reference proposals")
    pool = pool.drop(columns="_priority").reset_index(drop=True)
    pool["object_id"] = np.arange(len(pool), dtype=np.int64)
    return pool, dict(
        source_files=hashes,
        eligible_after_exclusion=eligible,
        sampling="uniform without replacement, not galaxy_weight",
    )


def reference(root):
    from euclid_dsps.prior_learning.spline15d import project_diffsky_frame_to_spline15d

    m, cfg, digest = settings(root)
    out = root / "reference"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    source, r = Path(m["source"]), cfg["reference"]
    excluded = set(
        pd.read_parquet(
            source / "dataset/parent/train.parquet", columns=["effective_proposal_key"]
        ).effective_proposal_key
    )
    pool, provenance = uniform_anchors(
        m["native_source"], excluded, r["anchors"], cfg["seed"]
    )
    decoder = read(source / "decoder.json")
    projected, _ = project_diffsky_frame_to_spline15d(
        pool, n_sfh_bins=decoder["model"]["n_sfh_bins"], batch_size=2048
    )
    if not np.array_equal(projected.object_id, pool.object_id):
        raise ValueError("Projection changed identities")
    theta = projected[NAMES].to_numpy()
    bounds = [r["bounds"].get(n, [-1, 1]) for n in NAMES]
    spec = fit_coordinates(theta, NAMES, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    spec.update(
        fitted_on="independent_unweighted_native_reference", role="population_reference"
    )
    basis = make_basis(
        to_x(theta, spec),
        r["components"],
        r["kernel_bandwidth"],
        r["gate_width"],
        r["broad_fraction"],
        cfg["seed"],
    )
    obs, bands = observed(root, "train")
    stats = compute_feature_stats(
        obs[[f"flux_{b}" for b in bands]].to_numpy(),
        obs[[f"fluxerr_{b}" for b in bands]].to_numpy(),
        obs[[f"mask_{b}" for b in bands]].to_numpy(),
        tuple(bands),
        append_mask=True,
    )
    write(out / "coordinates.json", spec)
    write(out / "feature_stats.json", feature_stats_to_json(stats))
    write(out / "provenance.json", provenance)
    pool[["object_id", "effective_proposal_key"]].to_csv(
        out / "identities.csv", index=False
    )
    np.savez(out / "basis.npz", **basis)
    finish(
        out,
        [
            out / n
            for n in (
                "coordinates.json",
                "feature_stats.json",
                "provenance.json",
                "identities.csv",
                "basis.npz",
            )
        ],
        digest,
        anchors=len(pool),
        components=r["components"],
        dimensions=15,
        target_truth_used=False,
        population_weight_magnitudes_used=False,
        reference_assumption="finite smoothed native family; not an unrestricted SFH conditional",
    )


def require_reference(root, digest):
    if not complete(root / "reference", digest):
        raise ValueError("Reference incomplete")
    with np.load(root / "reference/basis.npz") as f:
        basis = dict(f)
    stats = feature_stats_from_json(read(root / "reference/feature_stats.json"))
    return basis, read(root / "reference/coordinates.json"), stats


def bank(root, task):
    from euclid_dsps.synthetic_diffsky.coherent_parent import observe

    m, cfg, digest = settings(root)
    b = cfg["bank"]
    if not 0 <= task < b["shards"]:
        raise ValueError("Invalid bank task")
    basis, spec, stats = require_reference(root, digest)
    out = root / "banks" / f"shard_{task:03d}"
    out.mkdir(exist_ok=True, parents=True)
    if complete(out, digest):
        return
    source = Path(m["source"])
    decoder, noise = read(source / "decoder.json"), read(source / "noise.json")
    predict = photometer(decoder, b["decoder_batch_size"])
    bands, paths = [v["name"] for v in decoder["bands"]], []
    k = cfg["reference"]["components"]
    for start in range(0, b["rows_per_shard"], b["checkpoint_rows"]):
        block = out / f"block_{start:07d}"
        block.mkdir(exist_ok=True)
        if not complete(block, digest):
            n = min(b["checkpoint_rows"], b["rows_per_shard"] - start)
            seed = cfg["seed"] + task * 10000000 + start + 1000
            labels = np.arange(n) % k
            x = sample_basis(basis, labels, seed)
            theta = np.asarray(to_theta(x, spec))
            validate_theta(theta, spec)
            frame = observe(
                pd.DataFrame(index=np.arange(n)),
                predict(theta),
                decoder["bands"],
                noise,
                seed=seed + 1,
                selection=m["selection"],
            )
            # Five independently randomized row roles, assigned before selection.
            roles = np.random.default_rng(seed + 2).choice(
                5, n, p=[0.7, 0.1, 0.05, 0.05, 0.1]
            )
            np.savez(
                block / "bank.npz",
                x=x,
                theta=theta,
                component=labels,
                selected=frame.selected_r29.to_numpy(),
                role=roles,
                features=features(frame, bands, stats),
                flux=frame[[f"flux_{band}" for band in bands]].to_numpy(),
            )
            finish(block, [block / "bank.npz"], digest, rows=n)
        paths.append(block / "FINAL.json")
        write(
            out / "PROGRESS.json",
            dict(
                done=min(start + b["checkpoint_rows"], b["rows_per_shard"]),
                total=b["rows_per_shard"],
            ),
        )
    finish(out, paths, digest, rows=b["rows_per_shard"])


def load_bank(root, cfg, digest):
    arrays = {}
    for task in range(cfg["bank"]["shards"]):
        out = root / "banks" / f"shard_{task:03d}"
        if not complete(out, digest):
            raise ValueError(f"Incomplete bank {task}")
        for block in sorted(out.glob("block_*")):
            if not complete(block, digest):
                raise ValueError(f"Incomplete block {block}")
            with np.load(block / "bank.npz") as f:
                for key in f.files:
                    arrays.setdefault(key, []).append(f[key])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def bounded_fit(model, loss, train, validation, cfg, stop, out):
    """Resume full optimizer state; never declare max epoch to be convergence."""
    import equinox as eqx

    from scripts.feniks_forward_population import supervised_fit

    out.mkdir(exist_ok=True, parents=True)
    if (out / "STOP.json").exists():
        return eqx.tree_deserialise_leaves(out / "best.eqx", model)
    block = stop["block_epochs"]
    milestones = sorted(set([*range(block, cfg["epochs"] + 1, block), cfg["epochs"]]))
    start = read(out / "RESUME.json")["epoch"] if (out / "RESUME.json").exists() else 0
    for epoch in milestones:
        if epoch < start:
            continue
        model = supervised_fit(
            model, loss, *train, validation, dict(cfg, epochs=epoch), out
        )
        hist = pd.read_json(out / "training.jsonl", lines=True).drop_duplicates(
            "epoch", keep="last"
        )
        best = hist.set_index("epoch").best_nll
        ends = [e for e in milestones if e <= epoch and e in best.index]
        values = best.loc[ends].to_numpy()
        gains = -np.diff(values)[-stop["patience_blocks"] :]
        plateau = (
            epoch >= stop["minimum_epochs"]
            and len(gains) == stop["patience_blocks"]
            and np.all(gains < stop["minimum_gain"])
        )
        if plateau or epoch == cfg["epochs"]:
            write(
                out / "STOP.json",
                dict(
                    epoch=epoch,
                    plateau=bool(plateau),
                    reason="validation_plateau" if plateau else "maximum_epoch",
                    recent_best_gains=gains.tolist(),
                    best_nll=float(values[-1]),
                ),
            )
            return model
    raise RuntimeError("Optimizer checkpoint beyond configured budget")


def population(root):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.population_regularization import fit_selected_weights_kl
    from scripts.feniks_forward_population import classifier_template, classify
    from scripts.feniks_ratio_followup import (
        apply_logit_offsets,
        fit_marginal_logit_offsets,
    )

    _, cfg, digest = settings(root)
    out = root / "population"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    _, _, stats = require_reference(root, digest)
    data = load_bank(root, cfg, digest)
    k = cfg["reference"]["components"]
    masks = [(data["role"] == r) & data["selected"] for r in range(5)]
    counts = [np.bincount(data["component"][mask], minlength=k) for mask in masks]
    if any(np.any(c == 0) for c in counts):
        raise ValueError(
            "Missing selected classes in independent bank roles; no alpha flooring"
        )
    p = cfg["population"]
    efficiency = selection_efficiencies(
        data["component"],
        data["selected"],
        k,
        min_selected=p["min_selected"],
        min_alpha=p["min_alpha"],
    )
    train, val, cal, audit, _ = masks
    net_cfg = dict(cfg["classifier"], seed=cfg["seed"] + 1)
    model = classifier_template(data["features"].shape[1], k, net_cfg)

    def loss(net, f, labels):
        return -jax.nn.log_softmax(jax.vmap(net)(f))[jnp.arange(len(labels)), labels]

    model = bounded_fit(
        model,
        loss,
        (data["features"][train], data["component"][train]),
        (data["features"][val], data["component"][val]),
        net_cfg,
        cfg["stopping"],
        out,
    )
    c = simplex(counts[2])
    offset, offset_info = fit_marginal_logit_offsets(
        classify(model, data["features"][cal]), c
    )
    target_logs = []
    for split in ("train", "validation"):
        frame, bands = observed(root, split)
        target_logs.append(
            apply_logit_offsets(classify(model, features(frame, bands, stats)), offset)
        )
    audit_logc = apply_logit_offsets(classify(model, data["features"][audit]), offset)
    rows, candidates, heldout = [], [], []
    for strength in p["penalties"]:
        v, diagnostics = fit_selected_weights_kl(
            target_logs[0],
            c,
            strength=strength,
            alpha=efficiency["alpha"],
            eligible=efficiency["eligible"],
            weak_parent_mass=p["weak_parent_mass"],
        )
        candidates.append(v)
        heldout.append(logsumexp(target_logs[1] - np.log(c) + np.log(v), axis=1))
        rows.append(diagnostics)
    selected, accepted, degradation, se = choose_penalty(heldout, p["penalties"])
    for i, row in enumerate(rows):
        row.update(
            heldout_one_se=bool(accepted[i]),
            heldout_degradation=float(degradation[i]),
            paired_se=float(se[i]),
            selected=i == selected,
        )
    pd.DataFrame(rows).to_csv(out / "regularization.csv", index=False)
    v = candidates[selected]
    u = parent_from_selected(v, efficiency["alpha"])
    pd.DataFrame(
        dict(component=np.arange(k), selected_v=v, parent_u=u, **efficiency)
    ).to_csv(out / "weights.csv", index=False)
    write(
        out / "parent.json",
        dict(
            u=u.tolist(),
            v=v.tolist(),
            alpha=efficiency["alpha"].tolist(),
            classifier_reference_frequencies=c.tolist(),
            logit_offsets=offset.tolist(),
            reference_sha256=sha(root / "reference/FINAL.json"),
            classifier_sha256=sha(out / "best.eqx"),
            selected_penalty=p["penalties"][selected],
            population_uses_q=False,
            target_truth_used=False,
            population_scope="parent",
            alpha_parent=float(u @ efficiency["alpha"]),
        ),
    )
    write(
        out / "classifier_audit.json",
        dict(
            calibration=offset_info,
            audit_nll=float(
                -audit_logc[np.arange(audit.sum()), data["component"][audit]].mean()
            ),
            audit_null_nll=float(-np.log(c[data["component"][audit]]).mean()),
            independent_ratio_moment_median_error=float(
                np.median(abs(np.exp(audit_logc).mean(axis=0) / c - 1))
            ),
            role_counts=[int(mask.sum()) for mask in masks],
        ),
    )
    finish(
        out,
        [
            out / n
            for n in (
                "best.eqx",
                "STOP.json",
                "parent.json",
                "weights.csv",
                "regularization.csv",
                "classifier_audit.json",
            )
        ],
        digest,
        dimensions=15,
        population_uses_q=False,
        target_truth_used=False,
        scientific_promotion=False,
    )


def posterior(root, oracle=False):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        factor_log_prob,
        factor_template,
        transport_audit,
    )

    m, cfg, digest = settings(root)
    stage = "oracle" if oracle else "posterior"
    out = root / stage
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    _, spec, stats = require_reference(root, digest)
    pairs = []
    diagnostics = {}
    if oracle:
        for split in ("train", "validation"):
            frame, bands = observed(root, split)
            theta = pd.read_parquet(
                Path(m["source"]) / f"dataset/selected_r29/{split}.parquet",
                columns=NAMES,
            ).to_numpy()
            validate_theta(theta, spec)
            pairs.append(
                (
                    features(frame, bands, stats),
                    np.column_stack([to_x(theta, spec), np.ones(len(theta))]),
                )
            )
    else:
        if not complete(root / "population", digest):
            raise ValueError("Frozen parent incomplete")
        data = load_bank(root, cfg, digest)
        parent = read(root / "population/parent.json")
        u = np.asarray(parent["u"])
        r = np.ones(len(u)) / len(u)  # exact stratified parent sampling law
        for role in (0, 1):
            mask = (data["role"] == role) & data["selected"]
            weights = supervised_weights(data["component"][mask], u, r)
            diagnostics[str(role)] = dict(
                rows=int(mask.sum()),
                effective_rows=float(weights.sum() ** 2 / np.sum(weights**2)),
                maximum_weight=float(weights.max()),
            )
            pairs.append(
                (data["features"][mask], np.column_stack([data["x"][mask], weights]))
            )
    net_cfg = dict(cfg["posterior"], seed=cfg["seed"] + 2)
    model = factor_template(net_cfg, 15, pairs[0][0].shape[1], net_cfg["seed"])
    model = bounded_fit(
        model,
        lambda net, f, t: -factor_log_prob(net, f, t[:, :15]) * t[:, 15],
        pairs[0],
        pairs[1],
        net_cfg,
        cfg["stopping"],
        out,
    )
    transport = transport_audit(
        model,
        jnp.asarray(pairs[1][0][:8]),
        jax.random.PRNGKey(cfg["seed"] + 90),
        values=jnp.asarray(pairs[1][1][:8, :15]),
    )
    if any(
        not np.isfinite(value) for row in transport["experts"] for value in row.values()
    ):
        raise FloatingPointError(f"Nonfinite trained flow transport audit: {transport}")
    write(out / "transport.json", transport)
    if not transport["passed"]:
        raise FloatingPointError(
            "Trained flow inverse/Jacobian audit failed; inspect transport.json"
        )
    write(
        out / "training_measure.json",
        dict(
            oracle=oracle,
            targets="forward theta, never q draws",
            weighting="unit"
            if oracle
            else "selected bank pairs weighted by u_j / reference_parent_j",
            dimensions=15,
            diagnostics=diagnostics,
            parent_sha256=None if oracle else sha(root / "population/parent.json"),
        ),
    )
    evaluate_posterior(root, model, out, cfg, spec, stats)
    if not oracle:
        mask = (data["role"] == 4) & data["selected"]
        w = supervised_weights(data["component"][mask], u, r)
        ids = np.random.default_rng(cfg["seed"] + 77).choice(
            np.flatnonzero(mask), cfg["evaluation"]["objects"], p=w / w.sum()
        )
        evaluate_posterior(
            root,
            model,
            out,
            cfg,
            spec,
            stats,
            bank_evaluation=(data["features"][ids], data["theta"][ids], ids),
        )
    files = [
        p
        for p in out.iterdir()
        if p.is_file()
        and p.name
        in (
            "best.eqx",
            "STOP.json",
            "training_measure.json",
            "transport.json",
            "calibration.csv",
            "draws.npz",
            "in_model_calibration.csv",
            "in_model_draws.npz",
        )
    ]
    finish(out, files, digest, oracle=oracle, dimensions=15, scientific_promotion=False)


def evaluate_posterior(root, model, out, cfg, spec, stats, bank_evaluation=None):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from euclid_dsps.amortized.structured_population import DensityModel
    from scripts.report_feniks_forward_population import calibration

    prefix = "" if bank_evaluation is None else "in_model_"
    if bank_evaluation is None:
        frame, bands = observed(root, "test")
        source = Path(read(root / "MANIFEST.json")["source"])
        indices = np.random.default_rng(cfg["seed"] + 9).choice(
            len(frame), min(len(frame), cfg["evaluation"]["objects"]), replace=False
        )
        f = features(frame.iloc[indices], bands, stats)
        truth = pd.read_parquet(
            source / "dataset/selected_r29/test.parquet", columns=NAMES
        ).to_numpy()[indices]
    else:
        f, truth, indices = bank_evaluation
    count = cfg["evaluation"]["draws_per_object"]
    sample = eqx.filter_jit(
        lambda net, x, key: (
            sample_independent_mixture(
                DensityModel(net.experts[0]), net, key, x, count
            ).x
        )
    )
    chunks = []
    for start in range(0, len(f), 16):
        current = f[start : start + 16]
        padded = np.pad(current, ((0, 16 - len(current)), (0, 0)), mode="edge")
        x = np.asarray(
            sample(model, jnp.asarray(padded), jax.random.PRNGKey(cfg["seed"] + start))
        )
        chunks.append(x[:, : len(current)].transpose(1, 0, 2))
    draws_x = np.concatenate(chunks)
    draws = np.asarray(to_theta(draws_x.reshape(-1, 15), spec)).reshape(draws_x.shape)
    if not np.isfinite(draws).all():
        raise FloatingPointError("Invalid posterior draws")
    metrics, ranks, _ = calibration(draws, truth, NAMES)
    metrics["unique_evaluation_rows"] = len(np.unique(indices))
    metrics.to_csv(out / f"{prefix}calibration.csv", index=False)
    np.savez(
        out / f"{prefix}draws.npz",
        draws=draws,
        truth=truth,
        ranks=ranks,
        evaluation_positions=indices,
    )


def report(root):
    from scripts.report_feniks_coherent_inference import run

    run(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "prepare",
            "reference",
            "bank",
            "oracle",
            "population",
            "posterior",
            "report",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(args.source.resolve(), root, args.config.resolve())
    elif args.mode == "bank":
        bank(root, args.task)
    elif args.mode in ("oracle", "posterior"):
        posterior(root, args.mode == "oracle")
    else:
        globals()[args.mode](root)


if __name__ == "__main__":
    main()
