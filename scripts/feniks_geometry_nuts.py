"""Frozen photometric target geometry and separately launched NUTS references."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.exact_posterior import (
    NUTSSettings,
    autocorrelation_ess,
    combine_chain_diagnostics,
    run_batched_nuts_chains,
)
from euclid_dsps.amortized.features import make_encoder_features, read_feature_stats
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.local_transport_precision import (
    DiagnosticTransport64,
    promote,
)
from euclid_dsps.amortized.local_vi_diagnostic import initialize, log_prob, sample
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    posterior_log_target,
)
from euclid_dsps.amortized.train import _array_tree_sha256, build_prior_from_config
from euclid_dsps.config import load_config
from scripts.run_feniks_exact_posterior_benchmark import _load_runtime_rows
from scripts.run_feniks_sc_drws_local_vi_diagnostic import check_config

CASES = ("observed_000", "observed_005", "simulated_003", "simulated_004")
GROUPS = ("A", "B", "C")
NUTS_RECOVERY_PROFILE = {
    "name": "float64_depth4_parallel_v1",
    "chains": 8,
    "warmup": 1000,
    "chunks": [512] * 8,
    "max_num_doublings": 4,
    "target_accept": 0.9,
    "array_tasks": len(CASES) * len(GROUPS),
    "recommended_array_concurrency": len(CASES) * len(GROUPS),
}


def prepare_nuts(reference, root):
    """Import verified geometry without mutating its original code/receipt."""
    reference = reference.resolve()
    m = json.loads((reference / "MANIFEST.json").read_text())
    done = json.loads((reference / "GEOMETRY_COMPLETE.json").read_text())
    if done["status"] != "GEOMETRY_COMPLETE" or done["manifest_sha256"] != sha256_file(
        reference / "MANIFEST.json"
    ):
        raise ValueError("geometry receipt does not match manifest")
    for name, expected in done["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe artifact path")
        if sha256_file(reference / name) != expected:
            raise ValueError(f"geometry artifact changed: {name}")
    for path, expected in m["inputs"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source input changed: {path}")
    root.mkdir(parents=True, exist_ok=False)
    for name, expected in done["artifacts"].items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / name, dest)
        if sha256_file(dest) != expected:
            raise ValueError("copied artifact mismatch")
    m["geometry_import"] = dict(
        path=str(reference),
        code_commit=m["code_commit"],
        receipt_sha256=sha256_file(reference / "GEOMETRY_COMPLETE.json"),
    )
    m["code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    m["geometry_proposed_nuts_settings"] = {
        key: m[key]
        for key in ("chains", "warmup", "chunks", "max_num_doublings")
        if key in m
    }
    m.update(
        chains=NUTS_RECOVERY_PROFILE["chains"],
        warmup=NUTS_RECOVERY_PROFILE["warmup"],
        chunks=NUTS_RECOVERY_PROFILE["chunks"],
        max_num_doublings=NUTS_RECOVERY_PROFILE["max_num_doublings"],
        target_accept=NUTS_RECOVERY_PROFILE["target_accept"],
        nuts_execution_profile=NUTS_RECOVERY_PROFILE,
    )
    m["nuts_target_dtype"] = "float64"
    write(root / "MANIFEST.json", m)
    done["manifest_sha256"] = sha256_file(root / "MANIFEST.json")
    done["imported_from"] = m["geometry_import"]
    write(root / "GEOMETRY_COMPLETE.json", done)


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def weights(logtarget, logq):
    value = np.asarray(logtarget) - np.asarray(logq)
    if np.any(np.isnan(value) | np.isposinf(value)) or not np.any(np.isfinite(value)):
        raise ValueError("invalid importance weights")
    w = np.exp(value - np.max(value))
    return w / w.sum()


def prepare(reference, root):
    reference = reference.resolve()
    manifest = json.loads((reference / "RUN_MANIFEST.json").read_text())
    final = json.loads((reference / "FINAL.json").read_text())
    if final["status"] != "OBJECTIVE_PILOT_COMPLETE" or final["cases_complete"] != 16:
        raise ValueError("complete 16-case objective pilot required")
    if final.get("scientific_promotion") is not False:
        raise ValueError("unexpected source promotion contract")
    from scripts.feniks_qualified_local_vi import verify_night

    ref = manifest["qualified_night"]
    verified = verify_night(Path(ref["path"]), ref["arm"])
    if verified["inventory_sha256"] != ref["inventory_sha256"]:
        raise ValueError("qualified source changed")
    for key in ("checkpoint", "feature_stats"):
        if (
            sha256_file(Path(manifest["source"][key]))
            != verified["source"][key + "_sha256"]
        ):
            raise ValueError(f"source does not match qualified arm: {key}")
    if manifest.get("transport_contract") != "conditional_transport_float64_v1":
        raise ValueError("qualified transport64 reference required")
    files = [
        reference / name
        for name in (
            "RUN_MANIFEST.json",
            "FINAL.json",
            "config.yaml",
            "observed_rows.npy",
            "SIMULATED_INPUTS.npz",
        )
    ]
    files += [Path(manifest["source"][k]) for k in ("checkpoint", "feature_stats")]
    files += [Path(manifest["dataset"]["path"])]
    for path, expected in (
        (reference / "config.yaml", manifest["config_sha256"]),
        (reference / "observed_rows.npy", manifest["rows_sha256"]),
        (files[-1], manifest["dataset"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"changed reference: {path}")
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "MANIFEST.json",
        dict(
            reference=str(reference),
            cases=list(CASES),
            groups=list(GROUPS),
            inputs={str(p): sha256_file(p) for p in files},
            code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            chains=8,
            warmup=1000,
            chunks=[512] * 8,
            max_num_doublings=10,
            seed=260910,
            draws_per_replica=4096,
            replicas=2,
            scientific_promotion=False,
            truth_used=False,
            initial_prior="configured identity density, same physical coordinate transform",
            case_selection="prespecified diagnostic contrasts; not representative population",
            groups_description={
                "A": "learned prior / dispersed initial-prior draws",
                "B": "learned prior / encoder regions",
                "C": "initial prior / same starts as A",
            },
        ),
    )


def load(root):
    m = json.loads((root / "MANIFEST.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != m["code_commit"]:
        raise ValueError("use the frozen code checkout")
    for path, expected in m["inputs"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"input changed: {path}")
    return m


def runtime(m, case):
    reference = Path(m["reference"])
    old = json.loads((reference / "RUN_MANIFEST.json").read_text())
    config = load_config(reference / "config.yaml")
    check_config(config)
    config["catalog_path"] = old["dataset"]["path"]
    index = int(case.rsplit("_", 1)[1])
    rows = np.load(reference / "observed_rows.npy", allow_pickle=False)
    r = _load_runtime_rows(
        SimpleNamespace(
            checkpoint=Path(old["source"]["checkpoint"]),
            feature_stats=Path(old["source"]["feature_stats"]),
        ),
        config,
        rows[index : index + 1],
    )
    obs = PosteriorObservation(r.batch.flux, r.batch.flux_err, r.batch.mask)
    if case.startswith("simulated"):
        with np.load(reference / "SIMULATED_INPUTS.npz", allow_pickle=False) as data:
            # Deliberately do not read generated_x (simulation truth).
            obs = PosteriorObservation(
                *[
                    jnp.asarray(data[k][index : index + 1])
                    for k in ("flux", "flux_err", "mask")
                ]
            )
    stats = read_feature_stats(Path(old["source"]["feature_stats"]))
    features = make_encoder_features(obs.flux, obs.flux_err, stats, obs.mask)
    parameters, context = initialize(r.model, features)
    transport = DiagnosticTransport64(r.model.encoder)
    parameters = promote(parameters)
    prior_cfg = config["amortized"]["prior"]
    if prior_cfg["source"] != "joint_realnvp" or prior_cfg.get("init") != "identity":
        raise ValueError(
            "initial identity prior provenance not supported by source config"
        )
    if prior_cfg.get("checkpoint"):
        raise ValueError(
            "initial prior cannot be reconstructed from a loaded-prior config"
        )
    initial = build_prior_from_config(
        config,
        jax.random.PRNGKey(m["seed"]),
        latent_dim=len(r.latent_spec.names),
        active_spec=r.latent_spec,
    )
    test = jax.random.normal(
        jax.random.PRNGKey(0), (32, len(r.latent_spec.names)), dtype=jnp.float64
    )
    expected = -0.5 * jnp.sum(test**2 + np.log(2 * np.pi), axis=-1)
    np.testing.assert_allclose(initial.log_prob(test), expected, rtol=1e-6, atol=1e-5)
    initial_model = eqx.tree_at(lambda model: model.prior, r.model, initial)

    def target(model):
        @eqx.filter_jit
        def evaluate(x):
            return posterior_log_target(
                model,
                x[:, None, :],
                obs,
                r.latent_spec,
                r.context,
                r.model_args,
                r.latent_spec.names,
                r.likelihood,
                {"calibration": config.get("calibration", {})},
            )

        return evaluate

    return (
        r,
        obs,
        transport,
        parameters,
        context,
        target(r.model),
        target(initial_model),
        initial,
    )


def evaluate(target, x):
    parts = [target(jnp.asarray(x[i : i + 16])) for i in range(0, len(x), 16)]
    return {
        k: np.concatenate([np.asarray(getattr(p, k))[:, 0] for p in parts])
        for k in ("loglike", "logprior", "logtarget", "model_flux")
    }


def geometry(root):
    m = load(root)
    if (root / "GEOMETRY_COMPLETE.json").exists():
        raise FileExistsError("geometry already complete")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inventory = {}
    for ci, case in enumerate(m["cases"]):
        out = root / case
        out.mkdir(exist_ok=False)
        r, obs, enc, params, ctx, target, target_initial, initial = runtime(m, case)
        key = jax.random.PRNGKey(m["seed"] + ci)
        xs, centers, reports = [], [], []
        for replica in range(m["replicas"]):
            key, drawkey = jax.random.split(key)
            x, q = sample(enc, params, ctx, drawkey, m["draws_per_replica"])
            x, q = np.asarray(x)[:, 0], np.asarray(q)[:, 0]
            inverse = np.asarray(log_prob(enc, params, ctx, jnp.asarray(x[:, None])))[
                :, 0
            ]
            np.testing.assert_allclose(q, inverse, atol=5e-4, rtol=0)
            values = evaluate(target, x)
            np.testing.assert_allclose(
                values["logtarget"], values["loglike"] + values["logprior"], atol=1e-7
            )
            w = weights(values["logtarget"], q)
            np.savez_compressed(
                out / f"bank_{replica}.npz", x=x, logq=q, weight=w, **values
            )
            centers.extend([x[int(np.argmax(w))], x[0]])
            xs.append(x)
            reports.append(
                dict(
                    replica=replica,
                    ess=float(1 / np.sum(w * w)),
                    max_weight=float(w.max()),
                )
            )
        x = np.concatenate(xs)
        scale = np.maximum(np.std(x, axis=0), 1e-4)
        rng = np.random.default_rng(m["seed"] + ci)
        directions = np.concatenate(
            [np.eye(x.shape[-1]), rng.normal(size=(4, x.shape[-1]))]
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        positive_distances = np.array(
            [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0]
        )
        distances = np.concatenate(
            [-positive_distances[::-1], [0.0], positive_distances]
        )
        gradient_checks = []
        for label, fn in (("learned", target), ("initial", target_initial)):

            def scalar(p, fn=fn):
                return fn(p[None]).logtarget[0, 0]

            for center_id, center in enumerate(centers):
                direction = jnp.asarray(directions[-1] * scale)
                ad = float(jnp.dot(jax.grad(scalar)(jnp.asarray(center)), direction))
                for step in (0.01, 0.003, 0.001, 0.0003):
                    fd = float(
                        (
                            scalar(center + step * direction)
                            - scalar(center - step * direction)
                        )
                        / (2 * step)
                    )
                    gradient_checks.append(
                        dict(prior=label, center=center_id, step=step, ad=ad, fd=fd)
                    )
        pd.DataFrame(gradient_checks).to_csv(
            out / "target_gradient_checks.csv", index=False
        )
        records = []
        for center_id, center in enumerate(centers):
            fig, axes = plt.subplots(4, 5, figsize=(18, 12), squeeze=False)
            for di, direction in enumerate(directions):
                points = center + distances[:, None] * direction * scale
                values = evaluate(target, points)
                other = evaluate(target_initial, points)
                q = np.asarray(
                    log_prob(enc, params, ctx, jnp.asarray(points[:, None]))
                )[:, 0]
                ax = axes.flat[di]
                for name, val in [
                    ("loglike", values["loglike"]),
                    ("learned prior", values["logprior"]),
                    ("target", values["logtarget"]),
                    ("proposal", q),
                    ("initial-prior target", other["logtarget"]),
                ]:
                    ax.plot(distances, val - val[len(positive_distances)], label=name)
                ax.set_xscale("symlog", linthresh=0.0001)
                ax.set_title(
                    r.latent_spec.names[di]
                    if di < x.shape[-1]
                    else f"oblique {di - x.shape[-1]}",
                    fontsize=9,
                )
                ax.set_ylim(-50, 20)
                for j, distance in enumerate(distances):
                    records.append(
                        dict(
                            center=center_id,
                            direction=di,
                            distance=distance,
                            logq=q[j],
                            loglike=values["loglike"][j],
                            logprior=values["logprior"][j],
                            logtarget=values["logtarget"][j],
                            initial_logtarget=other["logtarget"][j],
                        )
                    )
            axes.flat[0].legend(fontsize=6)
            for ax in list(axes.flat)[len(directions) :]:
                ax.set_visible(False)
            fig.suptitle(
                f"{case}: center {center_id} (even: dominant, odd: ordinary draw). Relative log density; y clipped [-50,20]"
            )
            fig.supxlabel(
                "Signed distance in proposal-standardized latent coordinates (symlog)"
            )
            fig.tight_layout()
            fig.savefig(out / f"slices_{center_id}.png", dpi=130)
            plt.close(fig)
        pd.DataFrame(records).to_csv(out / "slices.csv", index=False)
        # A and C share starts exactly. B uses four distinct sampled regions,
        # including ordinary controls, not eight copies of the winning particle.
        cold = np.asarray(
            initial.sample(jax.random.PRNGKey(m["seed"] + 100 + ci), m["chains"])
        )
        warm = (
            np.asarray(centers)[np.arange(m["chains"]) % len(centers)]
            + rng.normal(size=cold.shape) * scale * 0.01
        )
        for group, starts in [("A", cold), ("B", warm), ("C", cold)]:
            fn = target_initial if group == "C" else target
            vals, grads = jax.vmap(
                jax.value_and_grad(lambda p, fn=fn: fn(p[None]).logtarget[0, 0])
            )(jnp.asarray(starts))
            if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(grads)):
                raise ValueError(
                    f"{case}/{group}: invalid starts; no silent replacement"
                )
            np.save(out / f"starts_{group}.npy", starts)
        np.savez_compressed(
            out / "observation.npz", flux=obs.flux, flux_err=obs.flux_err, mask=obs.mask
        )
        write(
            out / "AUDIT.json",
            dict(
                status="PASS",
                banks=reports,
                decoder_arrays_hash=_array_tree_sha256(r.model_args),
                learned_prior_hash=_array_tree_sha256(r.model.prior),
                initial_prior_hash=_array_tree_sha256(initial),
                initial_identity_density_verified=True,
                interpretation="slices are not posterior mass or mode coverage",
            ),
        )
        for p in out.iterdir():
            inventory[str(p.relative_to(root))] = sha256_file(p)
        write(root / "PROGRESS.json", dict(case=case, cases_complete=ci + 1))
        print(f"[geometry] {case} complete: {reports}", flush=True)
    write(
        root / "GEOMETRY_COMPLETE.json",
        dict(
            status="GEOMETRY_COMPLETE",
            artifacts=inventory,
            manifest_sha256=sha256_file(root / "MANIFEST.json"),
            scientific_promotion=False,
        ),
    )


def nuts(root, task):
    m = load(root)
    if m.get("nuts_target_dtype") != "float64":
        raise ValueError("prepare a new float64 NUTS root from the completed geometry")
    if m.get("nuts_execution_profile") != NUTS_RECOVERY_PROFILE:
        raise ValueError("prepare a new bounded-depth NUTS recovery root")
    done = json.loads((root / "GEOMETRY_COMPLETE.json").read_text())
    if done["manifest_sha256"] != sha256_file(root / "MANIFEST.json"):
        raise ValueError("manifest changed")
    for name, expected in done["artifacts"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"geometry artifact changed: {name}")
    if not 0 <= task < len(m["cases"]) * 3:
        raise ValueError("invalid array task")
    case, group = m["cases"][task // 3], GROUPS[task % 3]
    r, obs, _, _, _, learned, initial, initial_prior = runtime(m, case)
    audit = json.loads((root / case / "AUDIT.json").read_text())
    if (
        audit["learned_prior_hash"] != _array_tree_sha256(r.model.prior)
        or audit["initial_prior_hash"] != _array_tree_sha256(initial_prior)
        or audit["decoder_arrays_hash"] != _array_tree_sha256(r.model_args)
    ):
        raise ValueError("target arrays changed since geometry")
    with np.load(root / case / "observation.npz") as saved:
        for name in ("flux", "flux_err", "mask"):
            np.testing.assert_array_equal(saved[name], np.asarray(getattr(obs, name)))
    target = initial if group == "C" else learned
    starts = np.load(root / case / f"starts_{group}.npy", allow_pickle=False)
    out = root / "nuts" / case / group
    out.mkdir(parents=True, exist_ok=True)
    if (out / "FINAL.json").exists():
        print(f"{case}/{group}: already complete; preserved", flush=True)
        return
    write(
        out / "TARGET.json",
        dict(
            case=case,
            group=group,
            description=m["groups_description"][group],
            geometry_sha256=sha256_file(root / "GEOMETRY_COMPLETE.json"),
        ),
    )
    run_batched_nuts_chains(
        lambda x: target(x[None]).logtarget[0, 0],
        jnp.asarray(starts),
        seeds=tuple(m["seed"] + 10000 + task * 100 + i for i in range(m["chains"])),
        settings=NUTSSettings(
            warmup_steps=m["warmup"],
            sample_chunks=tuple(m["chunks"]),
            target_accept=m["target_accept"],
            max_num_doublings=m["max_num_doublings"],
        ),
        out_dirs=tuple(out / f"chain_{i}" for i in range(m["chains"])),
        resume=True,
        target_dtype="float64",
    )
    receipt = summarize_chains(out, r.latent_spec, root / case, m)
    # Fixed evenly spaced retained draws, never chosen for goodness of fit.
    frames = [
        pd.read_parquet(p)
        for i in range(m["chains"])
        for p in sorted((out / f"chain_{i}" / "chunks").glob("part_*.parquet"))
        if not p.name.endswith("_info.parquet")
    ]
    x = pd.concat(frames)[
        [f"x_{i:02d}" for i in range(len(r.latent_spec.names))]
    ].to_numpy()
    flux = evaluate(target, x[np.linspace(0, len(x) - 1, 128, dtype=int)])["model_flux"]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    residual = (flux - np.asarray(obs.flux)[0]) / np.asarray(obs.flux_err)[0]
    ax.boxplot(
        residual,
        tick_labels=[str(b["name"]) for b in r.config["bands"]],
        showfliers=False,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("(Predicted flux - measured flux) / measurement error")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "photometric_residuals.png", dpi=140)
    plt.close(fig)
    write(out / "FINAL.json", receipt)


def summarize_chains(out, spec, case_root, m):
    import matplotlib.pyplot as plt

    chains, infos = [], []
    columns = [f"x_{i:02d}" for i in range(len(spec.names))]
    for i in range(m["chains"]):
        directory = out / f"chain_{i}" / "chunks"
        paths = sorted(
            p
            for p in directory.glob("part_*.parquet")
            if not p.name.endswith("_info.parquet")
        )
        chains.append(
            pd.concat([pd.read_parquet(p) for p in paths])[columns].to_numpy()
        )
        infos.append(
            pd.concat(
                [
                    pd.read_parquet(p)
                    for p in sorted(directory.glob("part_*_info.parquet"))
                ]
            )
        )
    if len({len(c) for c in chains}) != 1 or len(chains[0]) != sum(m["chunks"]):
        raise ValueError("incomplete chains; do not truncate or discard chains")
    x = np.stack(chains)
    theta = np.asarray(x_to_theta(jnp.asarray(x), spec))
    summary, diagnostic = combine_chain_diagnostics(
        [out / f"chain_{i}" for i in range(m["chains"])], parameter_names=spec.names
    )
    summary["mcse_mean_x"] = [
        np.std(x[:, :, i], ddof=1) / np.sqrt(autocorrelation_ess(x[:, :, i]))
        for i in range(len(spec.names))
    ]
    summary.to_csv(out / "diagnostics.csv")
    divergent = int(sum(v.is_divergent.sum() for v in infos))
    saturated = int(
        sum(
            (v.num_integration_steps >= 2 ** m["max_num_doublings"] - 1).sum()
            for v in infos
        )
    )
    fig, axes = plt.subplots(len(spec.names), 1, figsize=(14, 24))
    for i, ax in enumerate(axes):
        ax.plot(theta[:, :, i].T, linewidth=0.4)
        ax.set_ylabel(spec.names[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "traces.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(3, 5, figsize=(17, 9))
    bank = np.load(case_root / "bank_0.npz")
    base = np.asarray(x_to_theta(jnp.asarray(bank["x"]), spec))
    for i, ax in enumerate(axes.flat):
        ax.hist(base[:, i], bins=40, density=True, histtype="step", label="AVI raw")
        ax.hist(
            base[:, i],
            weights=bank["weight"],
            bins=40,
            density=True,
            histtype="step",
            label="AVI weighted (learned prior)",
        )
        ax.hist(
            theta[:, :, i].ravel(),
            bins=40,
            density=True,
            histtype="step",
            label="NUTS (check diagnostics)",
        )
        ax.set_title(spec.names[i], fontsize=9)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "marginals.png", dpi=130)
    plt.close(fig)
    # Pair plots retain joint draws, unlike independent marginal reconstructions.
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    joint = theta.reshape(-1, theta.shape[-1])[::8]
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
            elif i == j:
                ax.hist(joint[:, i], bins=40, density=True)
            else:
                ax.scatter(joint[:, j], joint[:, i], s=1, alpha=0.15)
            if i == 4:
                ax.set_xlabel(spec.names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(spec.names[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "corner_first5.png", dpi=130)
    plt.close(fig)
    passed = bool(
        diagnostic["passes_rhat_1_01"]
        and diagnostic["passes_bulk_ess_400"]
        and diagnostic["passes_tail_ess_400"]
        and divergent == 0
        and saturated == 0
    )
    return dict(
        status="SAMPLING_COMPLETE",
        diagnostics_pass=passed,
        divergences=divergent,
        integration_limit_hits=saturated,
        scientific_promotion=False,
        interpretation="conditional reference, not a known true posterior; compare A and B across initializations",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["prepare", "prepare-nuts", "geometry", "nuts"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.mode in {"prepare", "prepare-nuts"}:
        if args.reference is None:
            parser.error("--reference required")
        if args.mode == "prepare-nuts":
            prepare_nuts(args.reference, args.root.resolve())
        else:
            prepare(args.reference, args.root.resolve())
    else:
        if not jax.config.x64_enabled:
            raise ValueError("JAX_ENABLE_X64=true required")
        if args.mode == "geometry":
            geometry(args.root.resolve())
        else:
            if args.task is None:
                parser.error("--task required")
            with (args.root / f".nuts_{args.task}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                nuts(args.root.resolve(), args.task)


if __name__ == "__main__":
    main()
