"""Prepare, preflight, train, resume and summarize the full-catalogue AVI array."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_rows(train, validation, same_catalog, expected):
    for rows in (train, validation):
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
            raise ValueError("nonempty one-dimensional integer row indices required")
        if len(np.unique(rows)) != len(rows) or np.min(rows) < 0:
            raise ValueError("duplicate or negative row indices")
    if len(train) != expected:
        raise ValueError(f"full catalogue required: {len(train)} != {expected}")
    if same_catalog and np.intersect1d(train, validation).size:
        raise ValueError("training and validation identities overlap")


def source_config_text(config, checkpoint):
    from euclid_dsps.amortized.latent import latent_spec_hash
    from euclid_dsps.amortized.train import _latent_spec_for_amortized_config

    # Free-parameter insertion order defines encoder coordinates, not just display.
    serialized = yaml.safe_dump(config, sort_keys=False)
    original = latent_spec_hash(_latent_spec_for_amortized_config(config))
    restored = latent_spec_hash(
        _latent_spec_for_amortized_config(yaml.safe_load(serialized))
    )
    if original != restored:
        raise ValueError("source config serialization changed latent coordinates")
    sidecar = read(Path(str(checkpoint) + ".json"))
    recorded = sidecar.get("latent_spec_hash", sidecar.get("latent_transform_hash"))
    if recorded != restored:
        raise ValueError(
            f"source checkpoint/config latent hash mismatch: {recorded} != {restored}"
        )
    return serialized


def prepare(args):
    from euclid_dsps.amortized.avi_experiments import ARMS
    from euclid_dsps.config import load_config

    root, ref, manifest_root = args.root, args.reference, args.manifest_root
    if root.exists():
        raise ValueError("new root required; use resume for existing experiments")
    source_manifest = read(ref / "RUN_MANIFEST.json")
    config = load_config(ref / "config.yaml")
    source = source_manifest["source"]
    config["catalog_path"] = source_manifest["dataset"]["path"]
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import check_config

    check_config(config)
    # Fail on source incompatibility before hashing catalogues or submitting GPUs.
    source_config_text(config, source["checkpoint"])
    train_path = manifest_root / "full_train_indices.npy"
    val_path = manifest_root / "confirmation_indices.npy"
    train = np.load(train_path, allow_pickle=False)
    # Preserve the upstream validation identities; cap only validation cost.
    validation = np.load(val_path, allow_pickle=False)[:512]
    val_catalog = args.validation_catalog or Path(config["catalog_path"]).with_name(
        "test.parquet"
    )
    validate_rows(
        train,
        validation,
        Path(config["catalog_path"]).resolve() == val_catalog.resolve(),
        args.expected_rows,
    )
    inputs = [
        ref / "RUN_MANIFEST.json",
        ref / "config.yaml",
        manifest_root / "manifest.json",
        train_path,
        val_path,
        Path(source["checkpoint"]),
        Path(source["feature_stats"]),
        Path(config["catalog_path"]),
        val_catalog,
    ]
    if Path(source["checkpoint"] + ".json").exists():
        inputs.append(Path(source["checkpoint"] + ".json"))
    hashes = {str(p.resolve()): sha(p) for p in inputs}
    # Build before creating the output root so missing teachers fail before submission.
    teachers, teacher_meta, teacher_inputs = build_teachers(
        args.nuts_root, train, source, ref
    )
    hashes.update(teacher_inputs)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    np.save(root / "train.npy", train, allow_pickle=False)
    np.save(root / "validation.npy", validation, allow_pickle=False)
    np.savez(root / "teachers.npz", **teachers)
    config["truth"] = {"parameter_columns": {}}
    config["extra_columns"] = []
    config["amortized"].setdefault("data", {}).update(
        use_redshift_for_split=False, stratify_column=None, redshift_bins=[]
    )
    (root / "source_config.yaml").write_text(
        source_config_text(config, source["checkpoint"])
    )
    write(root / "teachers.json", teacher_meta)
    hashes.update(
        {
            str((root / name).resolve()): sha(root / name)
            for name in (
                "train.npy",
                "validation.npy",
                "teachers.npz",
                "teachers.json",
                "source_config.yaml",
            )
        }
    )
    for folder in ("euclid_dsps", "scripts", "configs"):
        for p in Path(folder).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".yaml", ".slurm", ".sh"):
                hashes[str(p.resolve())] = sha(p)
    write(
        root / "MANIFEST.json",
        {
            "version": 1,
            "arms": [asdict(a) for a in ARMS],
            "source": source,
            "hashes": hashes,
            "seed": 260911,
            "epochs": args.epochs,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "validation_catalog": str(val_catalog.resolve()),
            "global_batch": 256,
            "local_microbatch": 32,
            "accumulation": 2,
            "gpus": 4,
            "particles": 128,
            "decoder_draw_block": 8,
            "validation_particles": 512,
            "learning_rate": 2e-5,
            "warmup_fraction": 0.05,
            "bootstrap_epochs": 6,
            "cycle": ["sleep", "sleep", "wake"],
            "prior_frozen": True,
            "decoder_frozen": True,
            "truth_used": False,
            "scientific_promotion": False,
            "new_nuts": False,
            "target_role": "frozen learned-prior encoder comparison; NOT population training",
            "code_snapshot_sha256": sha(root.parent / (root.name + ".code.tar"))
            if (root.parent / (root.name + ".code.tar")).exists()
            else None,
        },
    )
    print(
        f"Prepared {root}: {len(train)} observed training rows, {len(validation)} validation rows",
        flush=True,
    )


def build_teachers(nuts_root, train, source, reference):
    """Keep all joint retained draws; only upstream observed TRAIN identities."""
    nm = read(nuts_root / "MANIFEST.json")
    if Path(nm["reference"]).resolve() != reference.resolve():
        raise ValueError("teacher reference differs from frozen AVI reference")
    source_key = str(Path(source["checkpoint"]))
    if nm["inputs"].get(source_key) != sha(source_key):
        raise ValueError("teacher learned-prior checkpoint mismatch")
    cohort = pd.read_csv(nuts_root / "OBSERVED_COHORT.csv")
    if cohort.source_row.duplicated().any():
        raise ValueError("duplicate teacher galaxy identities")
    banks, fluxes, errors, masks, meta = [], [], [], [], []
    inputs = {
        str(p.resolve()): sha(p)
        for p in (nuts_root / "MANIFEST.json", nuts_root / "OBSERVED_COHORT.csv")
    }
    for path, digest in nm["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"teacher target provenance changed: {path}")
        inputs[str(Path(path).resolve())] = digest
    for row in cohort.itertuples():
        if row.source_row not in set(train.tolist()):
            raise ValueError(f"teacher {row.case} is not a training identity")
        group = nuts_root / "nuts" / row.case / "B_dense_depth6"
        final = group / "FINAL.json"
        receipt = read(final)
        if receipt.get("status") != "SAMPLING_COMPLETE":
            raise ValueError(f"teacher sampling incomplete: {group}")
        inputs[str(final.resolve())] = sha(final)
        chains = []
        for chain in range(8):
            files = sorted((group / f"chain_{chain}" / "chunks").glob("part_*.parquet"))
            files = [p for p in files if not p.stem.endswith("_info")]
            if not files:
                raise ValueError(f"missing teacher chain {chain}: {group}")
            pieces = []
            for p in files:
                df = pd.read_parquet(p)
                columns = sorted(c for c in df if c.startswith("x_"))
                if not columns:
                    raise ValueError(f"no joint latent coordinates: {p}")
                pieces.append(df[columns].to_numpy(dtype=np.float64))
                inputs[str(p.resolve())] = sha(p)
            chains.append(np.concatenate(pieces))
        if len({len(c) for c in chains}) != 1 or not all(
            np.isfinite(c).all() for c in chains
        ):
            raise ValueError(
                "teacher chains must have equal lengths and finite joint coordinates"
            )
        obs_path = nuts_root / row.case / "observation.npz"
        obs = np.load(obs_path, allow_pickle=False)
        inputs[str(obs_path.resolve())] = sha(obs_path)
        banks.append(np.concatenate(chains))
        fluxes.append(obs["flux"].reshape(-1))
        errors.append(obs["flux_err"].reshape(-1))
        masks.append(obs["mask"].reshape(-1))
        meta.append(
            {
                "case": row.case,
                "source_row": int(row.source_row),
                "final": receipt,
                "role": "empirical equal-length-chain distribution, not certified mode masses",
            }
        )
    if not banks or len({b.shape for b in banks}) != 1:
        raise ValueError("aligned nonempty teacher banks required")
    return (
        dict(
            x=np.stack(banks),
            flux=np.stack(fluxes),
            flux_err=np.stack(errors),
            mask=np.stack(masks),
            rows=cohort.source_row.to_numpy(),
        ),
        meta,
        inputs,
    )


def check_inputs(root):
    m = read(root / "MANIFEST.json")
    for path, digest in m["hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"changed input: {path}")
    return m


def initialize_candidate(model, config, latent_spec, arm, seed):
    import equinox as eqx
    import jax

    from euclid_dsps.amortized.proposal_expressivity import IndependentFlowMixture
    from euclid_dsps.amortized.train import build_amortized_model

    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    first = (
        build_amortized_model(config, keys[0], latent_spec=latent_spec).encoder
        if arm.scratch
        else model.encoder
    )
    if arm.experts == 1:
        return first
    candidate = IndependentFlowMixture(keys[4], first, n_components=4)
    if arm.scratch:
        es = tuple(
            build_amortized_model(config, k, latent_spec=latent_spec).encoder
            for k in keys[:4]
        )
    else:
        # Independent small head perturbations break symmetry without discarding the source.
        es = tuple(
            eqx.tree_at(
                lambda e: e.base.mean_head.bias,
                first,
                first.base.mean_head.bias
                + 0.05 * jax.random.normal(k, first.base.mean_head.bias.shape),
            )
            for k in keys[:4]
        )
    return eqx.tree_at(lambda c: c.experts, candidate, es)


def phase_at(epoch, arm, manifest):
    bootstrap = manifest["bootstrap_epochs"] if arm.scratch else 0
    return "sleep" if epoch < bootstrap else manifest["cycle"][(epoch - bootstrap) % 3]


def save_state(out, candidate, optimizer_state, next_step, contract):
    import equinox as eqx
    import jax

    path = out / f"state_{next_step:08d}.eqx"
    tmp = path.with_suffix(".tmp")
    eqx.tree_serialise_leaves(tmp, jax.device_get((candidate, optimizer_state)))
    tmp.replace(path)
    # Pointer is updated only after the complete state is durable and hashed.
    write(
        out / "RESUME.json",
        {
            "next_step": next_step,
            "path": path.name,
            "sha256": sha(path),
            "contract": contract,
        },
    )
    old = sorted(out.glob("state_*.eqx"))
    for p in old[:-2]:
        p.unlink()


def run(args, *, required_platform="gpu"):
    import fcntl

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        _replicate_model_for_pmap,
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.avi_experiments import (
        ARMS,
        enumerated_elbo,
        log_prob,
        make_parallel_steps,
        normalized_weights,
        stratified_proposal,
        weighted_nll,
    )
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.posterior_target import posterior_log_target
    from euclid_dsps.amortized.train import (
        LossBatch,
        _model_generated_sleep_loss,
        load_checkpoint,
    )
    from euclid_dsps.config import load_config

    root = args.root
    m = check_inputs(root)
    arm = ARMS[args.task]
    preflight = args.mode == "preflight"
    out = root / ("preflight" if preflight else "arms") / arm.name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "FINAL.json").exists():
        print(f"Already complete: {out}", flush=True)
        return
    if not preflight:
        pf = read(root / "preflight" / arm.name / "FINAL.json")
        if pf.get("status") != "PREFLIGHT_PASS" or pf.get("manifest_sha256") != sha(
            root / "MANIFEST.json"
        ):
            raise ValueError("matching successful per-arm preflight required")
    devices = tuple(jax.local_devices())
    if len(devices) != m["gpus"] or any(
        d.platform != required_platform for d in devices
    ):
        raise ValueError("this profile requires exactly four visible GPU devices")
    if not jax.config.jax_enable_x64:
        raise ValueError("JAX_ENABLE_X64=true required")
    start = time.monotonic()
    write(
        out / "PROGRESS.json",
        {"stage": "loading", "arm": arm.name, "devices": [str(d) for d in devices]},
    )
    config = load_config(root / "source_config.yaml")
    # Loading validates the source numerical contract before enabling qualified transport.
    model = load_checkpoint(m["source"]["checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    model = eqx.tree_at(lambda x: x.encoder, model, initialize_transport(model.encoder))
    rt = prepare_adaptive_training_runtime(
        config,
        out,
        train_indices_file=root / "train.npy",
        validation_indices_file=root / "validation.npy",
        validation_catalog_path=m["validation_catalog"],
        fixed_feature_stats_path=m["source"]["feature_stats"],
    )
    if str(rt.likelihood_config.get("type", "gaussian")) != "gaussian":
        raise ValueError("frozen comparison requires the qualified Gaussian target")
    candidate = initialize_candidate(model, config, rt.latent_spec, arm, m["seed"])
    frozen_digest = tree_digest((model.prior, model.sed_scale, model.band_calibration))
    write(
        out / "FROZEN_MODEL.json",
        {
            "source": m["source"],
            "prior_and_calibration_sha256": frozen_digest,
            "prior_initialization": "loaded learned source, unchanged in all arms",
            "coordinate_geometry_plots": "configured identity reference, not the active learned prior",
            "decoder_frozen": True,
            "population_training_started": False,
        },
    )
    geometry_path = out / "effective_latent_spec.json"
    if geometry_path.exists():
        geometry = read(geometry_path)
        geometry["population_density_initialization"] = "loaded_learned_source_frozen"
        geometry["source_checkpoint"] = m["source"]["checkpoint"]
        write(geometry_path, geometry)
    nb = math.ceil(m["train_rows"] / m["global_batch"])
    total = m["epochs"] * nb
    schedule = optax.warmup_cosine_decay_schedule(
        m["learning_rate"] * 0.1,
        m["learning_rate"],
        max(1, int(total * m["warmup_fraction"])),
        total,
        end_value=m["learning_rate"] * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0), optax.adamw(schedule, weight_decay=1e-6)
    )
    state = optimizer.init(eqx.filter(candidate, eqx.is_inexact_array))
    contract = sha(root / "MANIFEST.json")
    first_step = 0
    if (out / "RESUME.json").exists() and not preflight:
        resume = read(out / "RESUME.json")
        if (
            resume["contract"] != contract
            or sha(out / resume["path"]) != resume["sha256"]
        ):
            raise ValueError("resume contract/checksum mismatch")
        candidate, state = eqx.tree_deserialise_leaves(
            out / resume["path"], (candidate, state)
        )
        first_step = resume["next_step"]
        if (out / "training.csv").exists():
            history = pd.read_csv(out / "training.csv")
            history[history.step <= first_step].to_csv(
                out / "training.csv", index=False
            )
    (out / "PAUSED.json").unlink(missing_ok=True)

    def batch(flux, error, mask):
        return LossBatch(
            jnp.asarray(flux),
            jnp.asarray(error),
            jnp.asarray(mask),
            make_encoder_features(flux, error, rt.feature_stats, mask),
            jnp.zeros((len(flux), 0), jnp.float32),
        )

    train = batch(rt.train_arrays.flux, rt.train_arrays.flux_err, rt.train_arrays.mask)
    validation = batch(
        rt.validation_arrays.flux,
        rt.validation_arrays.flux_err,
        rt.validation_arrays.mask,
    )
    train = jax.device_get(train)
    validation = jax.device_get(validation)
    validation_rows = getattr(rt.validation_arrays, "row_index", None)
    if validation_rows is not None:
        pd.DataFrame(
            {
                "validation_index": np.arange(len(validation.flux)),
                "catalogue_row": validation_rows,
            }
        ).to_csv(out / "validation_identities.csv", index=False)
    with np.load(root / "teachers.npz", allow_pickle=False) as archive:
        bank = {k: archive[k] for k in archive.files}
    teacher = batch(bank["flux"], bank["flux_err"], bank["mask"])
    teacher = jax.device_get(teacher)
    # Enforce observational identity, not merely the integer source row label.
    if len(train.flux) != m["train_rows"] or rt.train_arrays.row_index is None:
        raise ValueError(
            "full catalogue loader changed count or omitted row identities"
        )
    lookup = {int(r): i for i, r in enumerate(rt.train_arrays.row_index)}
    if set(lookup) != set(np.load(root / "train.npy").tolist()):
        raise ValueError("loaded training rows differ from frozen manifest")
    ti = [lookup[int(r)] for r in bank["rows"]]
    for name in ("flux", "flux_err", "mask"):
        if not np.array_equal(
            np.asarray(getattr(train, name))[ti], getattr(teacher, name), equal_nan=True
        ):
            raise ValueError(
                f"teacher observation differs from training catalogue: {name}"
            )

    def target_values(x, b):
        def decode(xx):
            return posterior_log_target(
                model,
                xx,
                b,
                rt.latent_spec,
                rt.context,
                rt.model_args,
                rt.parameter_names,
                rt.likelihood_config,
                rt.calibration_config,
            )

        block = min(m.get("decoder_draw_block", 8), x.shape[0])
        if x.shape[0] % block:
            raise ValueError("draw count must divide into fixed decoder blocks")
        chunks = x.reshape((-1, block) + x.shape[1:])
        result = jax.lax.map(decode, chunks)
        return jax.tree_util.tree_map(
            lambda a: a.reshape((x.shape[0],) + a.shape[2:]), result
        )

    evaluator = make_evaluator(model, target_values, m, devices)

    def loss_for(phase):
        def loss(c, payload, key):
            b, tb, tx = payload
            k, ek, nk = jax.random.split(key, 3)
            teacher_ess = jnp.array(0.0, jnp.float64)
            if phase == "sleep":
                value, aux = _model_generated_sleep_loss(
                    model,
                    b,
                    rt.latent_spec,
                    rt.context,
                    rt.model_args,
                    rt.parameter_names,
                    k,
                    rt.likelihood_config,
                    rt.calibration_config,
                    rt.sleep_objective_config,
                    log_prob_fn=lambda f, x: log_prob(model, c, f, x),
                )
                ess, mw, valid = jnp.array(0.0), jnp.array(0.0), aux["finite_fraction"]
            else:
                x, logr = stratified_proposal(model, c, b.features, k, m["particles"])
                weights, usable, ess_values = normalized_weights(
                    target_values(x, b).logtarget - logr
                )
                weights = jax.lax.stop_gradient(weights)
                value = weighted_nll(log_prob(model, c, b.features, x), weights, usable)
                ess, mw, valid = (
                    jnp.mean(ess_values),
                    jnp.mean(jnp.max(weights, axis=0)),
                    jnp.mean(usable),
                )
                if arm.elbo_weight:
                    value += arm.elbo_weight * enumerated_elbo(
                        model,
                        c,
                        b.features,
                        ek,
                        lambda x: target_values(x, b).logtarget,
                    )
            if arm.teacher_weight:
                # tx arrives [object, draw, latent]; particle-major density convention.
                tx = jnp.swapaxes(tx, 0, 1)
                weights = jnp.ones(tx.shape[:2], jnp.float64) / tx.shape[0]
                usable = jnp.ones(tx.shape[1], dtype=bool)
                if arm.neighbour_sigma:
                    nf = (
                        tb.flux
                        + arm.neighbour_sigma
                        * tb.flux_err
                        * jax.random.normal(nk, tb.flux.shape)
                    )
                    neighbour = tb._replace(
                        flux=nf,
                        features=make_encoder_features(
                            nf, tb.flux_err, rt.feature_stats, tb.mask
                        ),
                    )
                    delta = (
                        target_values(tx, neighbour).loglike
                        - target_values(tx, tb).loglike
                    )
                    weights, usable, te = normalized_weights(delta)
                    # An empirical reweighting with one surviving draw is not a new teacher.
                    usable &= te >= 16
                    tb = neighbour
                    teacher_ess = jnp.mean(te)
                else:
                    teacher_ess = jnp.asarray(tx.shape[0], jnp.float64)
                value += arm.teacher_weight * weighted_nll(
                    log_prob(model, c, tb.features, tx),
                    jax.lax.stop_gradient(weights),
                    usable,
                )
            return value.astype(jnp.float64), jnp.array(
                [ess, mw, valid, teacher_ess], jnp.float64
            )

        return loss

    steps = {
        p: make_parallel_steps(loss_for(p), optimizer, devices=devices)
        for p in ("sleep", "wake")
    }
    cr, sr = (
        _replicate_model_for_pmap(candidate, devices),
        _replicate_model_for_pmap(state, devices),
    )
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def payload(step):
        epoch, ib = divmod(step, nb)
        order = np.random.default_rng(m["seed"] + epoch).permutation(m["train_rows"])
        # Cycle the permutation to complete the last fixed-shape batch. No row is dropped.
        indices = np.resize(order, nb * m["global_batch"])[
            ib * m["global_batch"] : (ib + 1) * m["global_batch"]
        ]
        b = jax.tree_util.tree_map(
            lambda x: (
                None
                if x is None
                else np.asarray(x)[indices].reshape((4, 2, 32) + x.shape[1:])
            ),
            train,
            is_leaf=lambda x: x is None,
        )
        rng = np.random.default_rng(m["seed"] + 1000000 + step)
        ids = rng.integers(len(bank["x"]), size=16)
        draws = rng.integers(bank["x"].shape[1], size=(16, 64))
        tx = bank["x"][ids[:, None], draws].reshape(4, 2, 2, 64, -1)
        tb = jax.tree_util.tree_map(
            lambda x: (
                None
                if x is None
                else np.asarray(x)[ids].reshape((4, 2, 2) + x.shape[1:])
            ),
            teacher,
            is_leaf=lambda x: x is None,
        )
        return b, tb, tx

    if not preflight and first_step == 0:
        save_state(out, candidate, state, 0, contract)
        evaluate(out, "source", model.encoder, validation, evaluator, m)
    planned = range(4) if preflight else range(first_step, total)
    last_step = first_step
    history_path = out / "training.csv"
    last_checkpoint = time.monotonic()
    for step in planned:
        phase = (
            ("sleep", "sleep", "wake", "wake")[step]
            if preflight
            else phase_at(step // nb, arm, m)
        )
        before = tree_digest(unreplicate(cr)) if preflight else None
        print(
            f"[avi] {arm.name} step={step + 1}/{total} phase={phase} start", flush=True
        )
        t0 = time.monotonic()
        key = jax.random.fold_in(jax.random.PRNGKey(m["seed"]), step)
        cr, sr, losses, metrics, finite = steps[phase](
            cr, sr, payload(step), jax.random.split(key, 4)
        )
        values = np.asarray(jax.device_get(metrics))[0]
        lv = float(np.asarray(jax.device_get(losses))[0])
        if not np.asarray(jax.device_get(finite)).all():
            raise RuntimeError(f"nonfinite {phase} update rejected at step {step}")
        if preflight and tree_digest(unreplicate(cr)) == before:
            raise RuntimeError(f"preflight {phase} did not change encoder parameters")
        last_step = step + 1
        record = dict(
            step=last_step,
            total_steps=total,
            epoch=step // nb + 1,
            phase=phase,
            loss=lv,
            ess=float(values[0]),
            max_weight=float(values[1]),
            valid_fraction=float(values[2]),
            teacher_ess=float(values[3]),
            seconds=time.monotonic() - t0,
            lr=float(schedule(step)),
        )
        pd.DataFrame([record]).to_csv(
            history_path, mode="a", header=not history_path.exists(), index=False
        )
        write(
            out / "PROGRESS.json",
            {
                **record,
                "stage": "update_complete",
                "arm": arm.name,
                "progress_percent": 100 * last_step / total,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        print(
            f"[avi] {arm.name} step={last_step}/{total} done loss={lv:.5g} seconds={record['seconds']:.1f}",
            flush=True,
        )
        if preflight:
            if values[2] <= 0:
                raise RuntimeError("preflight has no finite training signal")
            if arm.neighbour_sigma and values[3] < 16:
                raise RuntimeError(
                    "neighbour preflight has inadequate empirical support"
                )
        if not preflight and (
            last_step % 25 == 0
            or time.monotonic() - last_checkpoint > 600
            or stop[0]
            or last_step == total
        ):
            save_state(out, unreplicate(cr), unreplicate(sr), last_step, contract)
            last_checkpoint = time.monotonic()
        if not preflight and last_step % (3 * nb) == 0 and not stop[0]:
            evaluate(
                out,
                f"epoch_{last_step // nb:03d}",
                unreplicate(cr),
                validation,
                evaluator,
                m,
            )
        if stop[0] or (
            not preflight and time.monotonic() - start > args.max_hours * 3600
        ):
            if not preflight:
                save_state(out, unreplicate(cr), unreplicate(sr), last_step, contract)
            write(
                out / "PAUSED.json",
                {
                    "next_step": last_step,
                    "reason": "allocation budget or signal",
                    "resumable": True,
                },
            )
            return
    if (
        tree_digest((model.prior, model.sed_scale, model.band_calibration))
        != frozen_digest
    ):
        raise RuntimeError("frozen model changed")
    if not preflight:
        evaluate(out, "final", unreplicate(cr), validation, evaluator, m)
        eqx.tree_serialise_leaves(out / "encoder.eqx", unreplicate(cr))
    timing = pd.read_csv(history_path)
    steady = {
        p: float(timing[timing.phase.eq(p)].iloc[-1].seconds)
        for p in ("sleep", "wake")
        if timing.phase.eq(p).any()
    }
    estimate = (
        nb * sum(steady[phase_at(e, arm, m)] for e in range(m["epochs"]))
        if preflight
        else None
    )
    write(
        out / "FINAL.json",
        {
            "status": "PREFLIGHT_PASS" if preflight else "TRAINING_COMPLETE",
            "manifest_sha256": contract,
            "steps": last_step,
            "arm": asdict(arm),
            "gpus": 4,
            "elapsed_seconds": time.monotonic() - start,
            "prior_frozen_sha256": frozen_digest,
            "scientific_promotion": False,
            "population_training_started": False,
            "last_step_seconds_by_phase": steady,
            "rough_seconds_excluding_validation_io_and_compilation": estimate,
            "timing_caveat": "preflight estimate only; later states may cost differently",
        },
    )


def initialize_transport(encoder):
    import copy

    encoder = copy.copy(encoder)
    object.__setattr__(encoder, "transport_float64", True)
    return encoder


def tree_digest(tree):
    import equinox as eqx
    import jax

    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array)):
        a = np.asarray(jax.device_get(x))
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def unreplicate(tree):
    import equinox as eqx
    import jax

    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x))[0] if eqx.is_array(x) else x, tree
    )


def make_evaluator(model, target_values, m, devices):
    """Compile once per encoder architecture, reuse across all validation epochs."""
    from functools import partial

    import equinox as eqx
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import (
        log_prob,
        normalized_weights,
        with_encoder,
    )
    from euclid_dsps.amortized.posterior import sample_posterior
    from euclid_dsps.amortized.proposal_expressivity import (
        IndependentFlowMixture,
        sample_independent_mixture,
    )

    @partial(eqx.filter_pmap, in_axes=(None, 0, 0), devices=devices)
    def one(c, b, key):
        x = (
            sample_independent_mixture(
                model, c, key, b.features, m["validation_particles"]
            ).x
            if isinstance(c, IndependentFlowMixture)
            else sample_posterior(
                with_encoder(model, c), key, b.features, m["validation_particles"]
            ).x
        )
        q = log_prob(model, c, b.features, x)
        tv = target_values(x, b)
        w, valid, ess = normalized_weights(tv.logtarget - q)
        residual = jnp.where(
            b.mask[None],
            (tv.model_flux - b.flux[None]) / jnp.maximum(b.flux_err[None], 1e-30),
            0.0,
        )
        rms = jnp.sqrt(
            jnp.mean(
                jnp.sum(residual**2, axis=-1)
                / jnp.maximum(jnp.sum(b.mask, axis=-1), 1),
                axis=0,
            )
        )
        return jnp.stack(
            [ess, jnp.max(w, axis=0), rms, jnp.mean(q - tv.logtarget, axis=0), valid],
            axis=-1,
        )

    return one


def evaluate(out, label, candidate, validation, evaluator, m):
    """Independent fixed randomness, all four GPUs; no checkpoint selection."""
    import jax
    import jax.numpy as jnp

    records = []
    for start in range(0, len(validation.flux), 32):
        idx = np.minimum(np.arange(start, start + 32), len(validation.flux) - 1)
        b = jax.tree_util.tree_map(
            lambda x, idx=idx: (
                None if x is None else jnp.asarray(x[idx]).reshape((4, 8) + x.shape[1:])
            ),
            validation,
            is_leaf=lambda x: x is None,
        )
        for replica in range(2):
            key = jax.random.fold_in(
                jax.random.PRNGKey(m["seed"] + 90000000), start * 2 + replica
            )
            stats = np.asarray(
                jax.device_get(evaluator(candidate, b, jax.random.split(key, 4)))
            ).reshape(32, 5)
            for i, row in enumerate(stats[: min(32, len(validation.flux) - start)]):
                records.append(
                    dict(
                        validation_index=start + i,
                        replica=replica,
                        ess=row[0],
                        ess_fraction=row[0] / m["validation_particles"],
                        max_weight=row[1],
                        raw_predictive_rms=row[2],
                        negative_elbo=row[3],
                        finite=bool(row[4]),
                    )
                )
    pd.DataFrame(records).to_csv(out / f"validation_{label}.csv", index=False)
    print(
        f"[avi] validation={label} complete objects={len(validation.flux)}", flush=True
    )


def summarize(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    source_files = sorted((root / "arms").glob("*/validation_source.csv"))
    if source_files:
        source = (
            pd.read_csv(source_files[0])
            .groupby("validation_index")
            .median(numeric_only=True)
        )
        for metric, ax in zip(
            ("ess_fraction", "max_weight", "raw_predictive_rms"), axes, strict=True
        ):
            values = source[metric].sort_values().to_numpy()
            ax.plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                "k--",
                label="Frozen source",
            )
    for out in sorted((root / "arms").glob("*")):
        if not out.is_dir():
            continue
        files = sorted(out.glob("validation_*.csv"))
        files = [p for p in files if "source" not in p.name]
        if not files:
            continue
        path = (
            out / "validation_final.csv"
            if (out / "validation_final.csv").exists()
            else files[-1]
        )
        df = pd.read_csv(path)
        for metric, ax in zip(
            ("ess_fraction", "max_weight", "raw_predictive_rms"), axes, strict=True
        ):
            values = (
                df.groupby("validation_index")[metric].median().sort_values().to_numpy()
            )
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=out.name)
            ax.set_xlabel(metric)
            ax.set_ylabel("Fraction of validation galaxies")
        rows.append(
            {
                "arm": out.name,
                "artifact": path.name,
                **df[["ess_fraction", "max_weight", "raw_predictive_rms"]]
                .median()
                .to_dict(),
            }
        )
    if rows:
        axes[0].set_xscale("log")
        axes[2].set_xscale("log")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "avi_comparison.png", dpi=180)
        pd.DataFrame(rows).to_csv(root / "avi_comparison.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("No validation results yet.")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    curves = 0
    for path in sorted((root / "arms").glob("*/training.csv")):
        history = pd.read_csv(path)
        for phase, ax in zip(("sleep", "wake"), axes, strict=True):
            part = history[history.phase.eq(phase)]
            if len(part):
                ax.plot(
                    part.step,
                    part.loss.rolling(20, min_periods=1).mean(),
                    label=path.parent.name,
                )
                ax.set_title(phase + " training objective (auxiliary terms included)")
                ax.set_xlabel("Optimizer step")
                curves += 1
    if curves:
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "avi_training.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "preflight", "train", "summarize"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--manifest-root", type=Path)
    p.add_argument("--nuts-root", type=Path)
    p.add_argument("--validation-catalog", type=Path)
    p.add_argument("--expected-rows", type=int, default=37641)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--task", type=int, choices=range(7))
    p.add_argument("--max-hours", type=float, default=18.0)
    args = p.parse_args()
    args.root = args.root.resolve()
    if args.mode == "prepare":
        if any(
            getattr(args, n) is None
            for n in ("reference", "manifest_root", "nuts_root")
        ):
            p.error("prepare requires --reference, --manifest-root, --nuts-root")
        if args.epochs < 9:
            p.error("at least nine epochs required to exercise scratch wake")
        prepare(args)
    elif args.mode == "summarize":
        summarize(args.root)
    else:
        if args.task is None:
            p.error("--task required")
        run(args)


if __name__ == "__main__":
    main()
