"""Controlled restart diagnostics. Truth-only capacity arms NEVER update the prior."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_forward_population import basis_for, contract, supervised_fit


def link(source, destination):
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def evaluation_indices(frame):
    return frame.loc[
        frame.selected.astype(bool)
        & frame.split_bucket.ge(7000)
        & frame.split_bucket.lt(8500),
        "parent_row_index",
    ].to_numpy(np.int64)


def prepare(parent, root, config):
    original, settings = contract(parent)
    spec = yaml.safe_load(config.read_text())
    for stage, status in (
        ("population", "FORWARD_PARENT_COMPLETE"),
        ("posterior", "FORWARD_POSTERIOR_COMPLETE"),
    ):
        if read(parent / stage / "FINAL.json")["status"] != status:
            raise ValueError(f"incomplete parent stage: {stage}")
    for path, digest in (
        (
            parent / "population/parent.json",
            read(parent / "posterior/FINAL.json")["parent_sha256"],
        ),
        (
            parent / "posterior/best.eqx",
            read(parent / "posterior/FINAL.json")["checkpoint_sha256"],
        ),
        (
            parent / "population/best.eqx",
            read(parent / "population/FINAL.json")["classifier_sha256"],
        ),
    ):
        if sha(path) != digest:
            raise ValueError(f"checkpoint integrity: {path}")
    cohort = (
        Path(original["blind_truth_parent"]).parent / "selection_identities.parquet"
    )
    indices = evaluation_indices(pd.read_parquet(cohort))
    if len(indices) < 512 or len(np.unique(indices)) != len(indices):
        raise ValueError("invalid full validation cohort")
    if len(indices) > spec["report_objects"]:
        raise ValueError(
            "increase report_objects to include the full validation cohort"
        )
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    np.save(root / "evaluation_indices.npy", indices)
    shutil.copy2(config, root / "diagnostics.yaml")
    hashes = {
        str(p.resolve()): sha(p)
        for p in (
            cohort,
            parent / "population/parent.json",
            parent / "population/best.eqx",
            parent / "posterior/best.eqx",
            root / "evaluation_indices.npy",
            root / "diagnostics.yaml",
        )
    }
    write(
        root / "MANIFEST.json",
        dict(
            parent=str(parent.resolve()),
            settings=spec,
            hashes=hashes,
            evaluation_objects=len(indices),
            no_new_dsps_simulations=True,
            capacity_truth_used_only_for_diagnostics=True,
        ),
    )
    for arm in (
        "baseline",
        "classifier",
        "posterior",
        "capacity_analytic",
        "capacity_truth",
    ):
        branch = root / arm
        branch.mkdir()
        m = copy.deepcopy(original)
        m["hashes"].update(hashes)
        m["evaluation_indices"] = str((root / "evaluation_indices.npy").resolve())
        m["settings"]["report_objects"] = spec["report_objects"]
        m["settings"]["truth_weighting"] = "uniform"
        # Keep bank seed and split unchanged across comparisons.
        if arm in ("classifier", "posterior"):
            m["settings"][arm].update(spec[arm])
            source_stage = "population" if arm == "classifier" else "posterior"
            m["settings"][arm]["initial_checkpoint"] = str(
                (parent / source_stage / "best.eqx").resolve()
            )
        write(branch / "MANIFEST.json", m)
        for name in ("basis.npz", "reference", "posterior_bank"):
            link(parent / name, branch / name)
        for name in ("population", "posterior"):
            if (arm == "classifier" and name == "population") or (
                arm == "posterior" and name == "posterior"
            ):
                (branch / name).mkdir()
            else:
                link(parent / name, branch / name)
        (branch / "report").mkdir()
    print(
        json.dumps(
            dict(
                evaluation_objects=len(indices),
                stages=[
                    "baseline evaluation",
                    "classifier continuation and population refit",
                    "posterior continuation at frozen old parent",
                    "analytic prior capacity",
                    "held-out truth capacity",
                ],
                new_dsps_simulations=0,
                resources=spec["resources"],
            ),
            indent=2,
        )
    )


def ratio_checks(branch):
    with np.load(branch / "population/classifier_validation.npz") as saved:
        c, labels = saved["frequencies"], saved["component"]
        probabilities = np.exp(saved["log_prob"])
        test_nll = float(-saved["log_prob"][np.arange(len(labels)), labels].mean())
        f = np.bincount(labels, minlength=len(c)) / len(labels)
        if np.any(f == 0):
            raise ValueError("missing test class")
        means = np.einsum("i,ij->j", c[labels] / f[labels] / len(labels), probabilities)
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == labels
        bins = np.minimum((confidence * 15).astype(int), 14)
        ece = sum(
            np.mean(bins == b)
            * abs(correct[bins == b].mean() - confidence[bins == b].mean())
            for b in range(15)
            if np.any(bins == b)
        )
    return dict(
        test_objects=len(labels),
        test_nll=test_nll,
        accuracy=float(correct.mean()),
        confidence_ece_15bins=float(ece),
        ratio_reference_min=float((means / c).min()),
        ratio_reference_max=float((means / c).max()),
        ratio_reference_median_abs_error=float(np.median(abs(means / c - 1))),
    )


def capacity(root, task):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import log_prob
    from euclid_dsps.amortized.forward_population_runtime import (
        load_forward_runtime,
        posterior_template,
    )
    from euclid_dsps.amortized.latent import theta_to_x, x_to_theta
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from scripts.report_feniks_forward_population import population_metrics

    name = ("capacity_analytic", "capacity_truth")[task]
    branch = root / name
    manifest, cfg = contract(branch)
    settings = read(root / "MANIFEST.json")["settings"]
    fit = {**settings["capacity"], "seed": settings["seed"] + task}
    out = branch / "report"
    if (out / "FINAL.json").exists():
        return
    model, runtime, _, source_config = load_forward_runtime(
        Path(manifest["source"]), out / "runtime", cfg
    )
    candidate = posterior_template(model, runtime, source_config, cfg)
    rng = np.random.default_rng(fit["seed"])
    basis = basis_for(branch)
    if task == 0:
        u = read(branch / "population/parent.json")["u"]
        train, _ = basis.sample(rng, fit["training_draws"], weights=u)
        val, _ = basis.sample(rng, fit["validation_limit"], weights=u)
        test, _ = basis.sample(rng, fit["test_draws"], weights=u)
        exact_test_logp = basis.log_prob(test, u)
        unique_train = len(train)
    else:
        truth = pd.read_parquet(manifest["blind_truth_parent"])[
            list(basis.names)
        ].to_numpy()
        if not np.all(
            (truth > np.asarray(runtime.latent_spec.lower))
            & (truth < np.asarray(runtime.latent_spec.upper))
        ):
            raise ValueError(
                "truth capacity target outside flow support; do not silently clip"
            )
        order = rng.permutation(len(truth))
        split1, split2 = int(0.6 * len(order)), int(0.8 * len(order))
        raw = np.asarray(theta_to_x(jnp.asarray(truth), runtime.latent_spec))
        train_ids, val_ids, test_ids = (
            order[:split1],
            order[split1:split2],
            order[split2:],
        )
        np.savez(
            out / "truth_split.npz", train=train_ids, validation=val_ids, test=test_ids
        )
        unique_train = len(train_ids)
        train = raw[rng.choice(train_ids, fit["training_draws"])]
        val, test = raw[val_ids], raw[test_ids]
        exact_test_logp = None
    # Same 15D conditional architecture at a constant context: pure density capacity.
    dim = candidate.input_dim

    def constant(n):
        return np.zeros((n, dim), np.float32)

    def loss(net, f, x):
        return -log_prob(model, net, f, x[None])[0]

    net = supervised_fit(
        candidate,
        loss,
        constant(len(train)),
        train,
        (constant(len(val)), val),
        fit,
        out,
    )
    evaluate = eqx.filter_jit(lambda f, x: -log_prob(model, net, f, x[None])[0])
    nll = np.concatenate(
        [
            np.asarray(evaluate(jnp.asarray(constant(len(chunk))), jnp.asarray(chunk)))
            for chunk in np.array_split(test, max(1, int(np.ceil(len(test) / 512))))
        ]
    )
    draw = eqx.filter_jit(
        lambda key: sample_independent_mixture(
            model, net, key, jnp.zeros((1, dim)), 1024
        ).x[:, 0]
    )
    samples = np.concatenate(
        [
            np.asarray(draw(jax.random.PRNGKey(fit["seed"] + i + 10000)))
            for i in range(int(np.ceil(fit["test_draws"] / 1024)))
        ]
    )[: fit["test_draws"]]
    theta = np.asarray(x_to_theta(jnp.asarray(samples), runtime.latent_spec))
    target = np.asarray(x_to_theta(jnp.asarray(test), runtime.latent_spec))
    marginal, joint = population_metrics(theta, target, basis.names)
    marginal.to_csv(out / "capacity_marginal.csv", index=False)
    joint.to_csv(out / "capacity_joint.csv", index=False)
    np.savez(out / "test_draws.npz", learned=theta, target=target, test_nll=nll)
    from scripts.feniks_avi_overnight import _marginals

    _marginals(
        out / "capacity_15d.png", {"Flow": theta, "Target": target}, None, basis.names
    )
    result = dict(
        status="FLOW_CAPACITY_COMPLETE",
        target=name,
        dimensions=15,
        context="constant",
        unique_training_objects=unique_train,
        training_pairs=len(train),
        heldout_objects=len(test),
        test_nll=float(nll.mean()),
        production_prior_modified=False,
        truth_weighting="uniform",
    )
    if exact_test_logp is not None:
        kl = exact_test_logp + nll
        result.update(
            kl_target_flow=float(kl.mean()),
            kl_standard_error=float(kl.std(ddof=1) / np.sqrt(len(kl))),
        )
    write(out / "FINAL.json", result)


def run(root, stage, task=0):
    from scripts.feniks_forward_population import population, train_posterior
    from scripts.report_feniks_forward_population import report

    m = read(root / "MANIFEST.json")
    for path, digest in m["hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"diagnostic input changed: {path}")
    if stage == "baseline":
        report(root / "baseline")
    elif stage == "classifier":
        population(root / "classifier")
        write(
            root / "classifier/COMPARISON.json",
            dict(
                before=ratio_checks(Path(m["parent"])),
                after=ratio_checks(root / "classifier"),
            ),
        )
    elif stage == "posterior":
        train_posterior(root / "posterior")
        report(root / "posterior")
    elif stage == "capacity":
        capacity(root, task)
    else:
        results = {
            arm: read(root / arm / "report/FINAL.json")
            for arm in ("baseline", "posterior", "capacity_analytic", "capacity_truth")
        }
        results["classifier"] = read(root / "classifier/population/FINAL.json")
        results["ratios"] = read(root / "classifier/COMPARISON.json")
        write(root / "RESULTS.json", results)
        rows = []
        for arm in ("baseline", "posterior"):
            for cohort in ("simulation", "observed_blind"):
                frame = pd.read_csv(root / arm / "report" / f"{cohort}_calibration.csv")
                frame["arm"], frame["cohort"] = arm, cohort
                rows.append(frame)
        pd.concat(rows).to_csv(root / "calibration_comparison.csv", index=False)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, relative in zip(
            axes.flat,
            (
                "classifier/population",
                "posterior/posterior",
                "capacity_analytic/report",
                "capacity_truth/report",
            ),
            strict=True,
        ):
            history = pd.read_json(root / relative / "training.jsonl", lines=True)
            ax.plot(history.epoch, history.train_nll, label="Training")
            ax.plot(history.epoch, history.validation_nll, label="Fixed validation")
            initial = root / relative / "INITIAL_VALIDATION.json"
            if initial.exists():
                ax.axhline(
                    read(initial)["nll"],
                    ls="--",
                    color="black",
                    label="Starting checkpoint",
                )
            ax.set(title=relative, xlabel="Additional epoch", ylabel="NLL")
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / "training_comparison.png", dpi=160)
        plt.close(fig)
        (root / "REPORT.md").write_text(
            "# Forward diagnostics\n\nNo new DSPS simulations; original run untouched.\n\n- baseline/report: full validation cohort, original q and parent.\n- posterior/report: identical cohort and parent, continued q.\n- classifier/COMPARISON.json: independent ratio checks before/after.\n- classifier/population: candidate parent ONLY; not fed to continued q.\n- capacity_analytic/report: unconditional fit to exact learned 15D prior (KL).\n- capacity_truth/report: truth-only held-out 15D density fit, not a production model.\n\nRead RESULTS.json and calibration_comparison.csv. Completion is not scientific convergence.\n"
        )
        write(
            root / "FINAL.json",
            dict(status="FORWARD_DIAGNOSTICS_COMPLETE", scientific_promotion=False),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=(
            "prepare",
            "baseline",
            "classifier",
            "posterior",
            "capacity",
            "summary",
        ),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_forward_diagnostics.yaml"),
    )
    p.add_argument("--task", type=int, default=0, choices=(0, 1))
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.parent, a.root, a.config)
    else:
        run(a.root, a.stage, a.task)


if __name__ == "__main__":
    main()
