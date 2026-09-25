import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from euclid_dsps.amortized.forward_population import PHYSICAL, PhysicalBasis
from scripts import feniks_population_precision as audit


def test_fixed_metrics_keep_joint_information_and_exact_zero():
    rng = np.random.default_rng(6)
    left = rng.normal(size=(4096, 5))
    left[:, 1] = left[:, 0]
    right = left.copy()
    right[:, 1] = right[::-1, 1]
    directions = rng.normal(size=(5, 64))
    directions /= np.linalg.norm(directions, axis=0)
    zero = audit.fixed_metrics(left, left, np.ones(5), directions)
    assert zero["parent_sw"] == 0
    changed = audit.fixed_metrics(left, right, np.ones(5), directions)
    assert changed["parent_sw"] > 0.04
    assert all(changed[f"w1_{name}"] == 0 for name in PHYSICAL)


def test_bootstrap_counts_pair_exact_and_photometric_without_dropping_objects():
    a = audit.bootstrap_counts(2000, 9, 0)
    b = audit.bootstrap_counts(2000, 9, 0)
    np.testing.assert_array_equal(a, b)
    assert a.sum() == 2000
    assert not np.array_equal(a, audit.bootstrap_counts(2000, 9, 1))


def test_candidate_is_frozen_without_access_to_truth_or_bootstrap():
    table = pd.DataFrame(
        dict(
            rank=[8, 64],
            strength=[0.1, 0.01],
            heldout_log_likelihood=[0.0, 1.0],
            parent_physical_sliced_wasserstein=[0.0, 1.0],
            bootstrap_parent_sw_median=[0.0, 1.0],
        )
    )
    assert audit.choose_fixed_candidate(table) == dict(rank=64, strength=0.01)


def test_receipt_detects_corruption(tmp_path):
    audit.atomic_npz(tmp_path / "weights.npz", x=np.arange(5))
    audit.finish(tmp_path, parent_sum=1.0)
    assert audit.receipt_valid(tmp_path)
    (tmp_path / "weights.npz").write_bytes(b"broken")
    assert not audit.receipt_valid(tmp_path)


def test_cached_pipeline_reports_partial_then_complete_and_resumes(
    tmp_path, monkeypatch
):
    from scripts import feniks_population_basis_audit, feniks_ratio_followup

    source, root = tmp_path / "source", tmp_path / "audit"
    (source / "spectral_basis").mkdir(parents=True)
    for path in ("cache", "metric", "bootstrap", "decoder", "report", "logs"):
        (root / path).mkdir(parents=True, exist_ok=True)
    names = (*PHYSICAL, *(f"sfh_{i}" for i in range(10)))
    basis = PhysicalBasis.create(names, components=4)
    rng = np.random.default_rng(30)
    x, _ = basis.sample(rng, 256, weights=np.array([0.2, 0.3, 0.2, 0.3]))
    frequencies = np.full(4, 0.25)
    raw = rng.normal(size=(4, 2))
    modes = raw - raw.mean(axis=0)
    audit.atomic_npz(
        source / "spectral_basis/modes.npz", modes=modes, reference_selected=frequencies
    )
    spec = dict(
        names=names,
        lower=[0.0] * 15,
        upper=[10.0] * 15,
        normalization="identity",
        raw_center=[0.0] * 15,
        raw_scale=[1.0] * 15,
    )
    (source / "latent.json").write_text(json.dumps(spec))
    settings = dict(
        seed=5,
        bootstraps=2,
        metric_draws=[32, 64],
        metric_seeds=[8, 9],
        directions=4,
        bootstrap_metric_draws=64,
        solver_tolerance=2e-6,
        solver_maximum_iterations=200,
        solver_seconds=10.0,
        contracts=dict(
            historical_bootstrap_sw=0.015,
            metric_resolution_tolerance=0.002,
            decoder_p95_abs_sigma=0.25,
        ),
    )
    manifest = dict(
        inputs=dict(
            latent_spec=dict(
                path=str(source / "latent.json"),
                sha256=audit.sha(source / "latent.json"),
            )
        ),
        settings=settings,
        convergence=str(source),
        source=str(source),
        candidates={arm: dict(rank=2, strength=0.1) for arm in audit.ARMS},
    )
    audit.write(root / "MANIFEST.json", manifest)
    splits = dict(target_fit=np.arange(200), target_heldout=np.arange(200, 256))
    context = (
        {},
        {},
        {},
        {},
        {},
        dict(x=x),
        basis,
        splits,
        frequencies,
        np.ones(4),
        np.ones(4, bool),
        np.full(4, 0.25),
        np.full(4, 0.25),
    )
    monkeypatch.setattr(feniks_ratio_followup, "_context", lambda root: context)
    logc = basis.component_log_prob(x)
    logc -= logsumexp(logc, axis=1, keepdims=True)
    monkeypatch.setattr(
        feniks_population_basis_audit,
        "_classifier_log_probabilities",
        lambda *args, **kwargs: (logc[:200], logc[200:], {}),
    )
    audit.cache(root)
    assert audit.receipt_valid(root / "cache")
    audit.metric(root, 0)
    audit.report(root)
    assert audit.read(root / "report/FINAL.json")["status"] == "PARTIAL"
    audit.metric(root, 1)
    for repeat in range(2):
        audit.bootstrap(root, repeat)
    # A completed diagnostic screen remains different from a full qualification.
    pd.DataFrame(
        dict(variant=["baseline", "merged"], band=["r", "r"], p95_abs=[0.3, 0.1])
    ).to_csv(root / "decoder/bands.csv", index=False)
    audit.finish(root / "decoder", full_decoder_qualified=False)
    audit.report(root)
    assert audit.read(root / "report/FINAL.json")["status"] == "COMPLETE"
    assert not audit.read(root / "ROADMAP_STATUS.json")["production_ready"]

    def forbidden(*args, **kwargs):
        raise AssertionError("completed fits must not be rerun")

    monkeypatch.setattr(audit, "fit_cell", forbidden)
    audit.bootstrap(root, 0)
    # If final metric work was interrupted, reuse the independently saved fit.
    (root / "bootstrap/repeat_000/noisy_photometry_exact/FINAL.json").unlink()
    audit.bootstrap(root, 0)
    assert audit.receipt_valid(root / "bootstrap/repeat_000/noisy_photometry_exact")


def test_job_resources_use_cpu_for_repeated_fits_and_partial_reporting():
    script = (
        Path(__file__).parents[1] / "scripts/submit_feniks_population_precision.sh"
    ).read_text()
    assert "submit bootstrap 0" in script
    assert "submit decoder 1" in script
    assert "afterany:" in script
    assert "kill-on-invalid-dep=yes" in script


def test_submit_dependencies_and_selective_resume(tmp_path):
    repo = Path(__file__).parents[1]
    root, commands = tmp_path / "audit", tmp_path / "bin"
    root.mkdir()
    commands.mkdir()
    resources = dict(
        cache_minutes=25,
        decoder_minutes=25,
        metric_minutes=15,
        bootstrap_minutes=15,
        report_minutes=5,
        bootstrap_concurrency=4,
        cpu_threads=4,
    )
    audit.write(
        root / "MANIFEST.json",
        dict(inputs={}, settings=dict(resources=resources, bootstraps=2)),
    )
    (root / "CODE_DIR").write_text(str(repo))
    (root / "CODE_SHA256").write_text("original scientific snapshot")
    frozen = {
        name: (root / name).read_bytes()
        for name in ("MANIFEST.json", "CODE_DIR", "CODE_SHA256")
    }
    log = tmp_path / "submissions.jsonl"
    (commands / "sbatch").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "assert not any(a.split('=')[0] in ('--mem', '--mem-per-cpu', '--mem-per-gpu') for a in sys.argv[1:]), 'Jean-Zay rejects explicit memory flags'\n"
        "p = Path(os.environ['MOCK_SUBMISSIONS'])\n"
        "n = len(p.read_text().splitlines()) if p.exists() else 0\n"
        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(1000+n)\n"
    )
    (commands / "squeue").write_text("#!/bin/sh\nexit 0\n")
    for path in commands.iterdir():
        path.chmod(0o755)
    env = dict(os.environ, MOCK_SUBMISSIONS=str(log))
    # Obsolete overrides must not reintroduce the forbidden flags.
    env.update(FENIKS_CPU_MEM="16G", FENIKS_GPU_MEM="60G")
    env["PATH"] = f"{commands}:{Path(sys.executable).parent}:{env['PATH']}"
    command = [
        "bash",
        str(repo / "scripts/submit_feniks_population_precision.sh"),
        "--resume",
        str(root),
    ]
    # Preparation succeeded but the first sbatch failed: no JOBS.env exists yet.
    assert not (root / "JOBS.env").exists()
    subprocess.run(command, cwd=repo, env=env, capture_output=True, check=True)
    jobs = [json.loads(row) for row in log.read_text().splitlines()]
    assert len(jobs) == 5
    assert "--gres=gpu:1" in jobs[0] and "--gres=gpu:1" in jobs[1]
    assert not any(arg.startswith("--dependency") for arg in jobs[1])
    assert "--dependency=afterok:1000" in jobs[2]
    assert "--dependency=afterok:1000" in jobs[3]
    assert "--dependency=afterany:1000:1001:1002:1003" in jobs[4]
    assert "--array=0,1%4" in jobs[3]
    assert "--gres=gpu:1" not in jobs[3]

    done = [root / "cache", root / "decoder"]
    done += [root / "metric" / arm for arm in audit.ARMS]
    done += [
        root / "bootstrap/repeat_000" / f"{arm}_{ratio}"
        for arm in audit.ARMS
        for ratio in audit.RATIOS
    ]
    for directory in done:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.txt").write_text("complete")
        audit.finish(directory)
    subprocess.run(command, cwd=repo, env=env, capture_output=True, check=True)
    jobs = [json.loads(row) for row in log.read_text().splitlines()][5:]
    assert len(jobs) == 2
    assert "--array=1%4" in jobs[0]
    assert not any(arg.startswith("--dependency") for arg in jobs[0])
    assert "--dependency=afterany:1005" in jobs[1]
    for name, payload in frozen.items():
        assert (root / name).read_bytes() == payload


def test_decoder_screen_preserves_runtime_and_resumes_batches(tmp_path, monkeypatch):
    import jax.numpy as jnp

    from euclid_dsps.amortized import decoder, posterior_target
    from scripts import feniks_failure_modes, feniks_ratio_ladder

    @dataclass(frozen=True)
    class Runtime:
        context: object
        latent_spec: object
        feature_stats: object
        model_args: object = None
        parameter_names: object = None
        calibration_config: object = None

    root = tmp_path / "audit"
    for name in ("decoder", "source", "convergence", "parent"):
        (root / name).mkdir(parents=True)
    ids = np.arange(4)
    np.save(root / "ids.npy", ids)
    pd.DataFrame(dict(parent_row_index=ids, selected=[True] * 4)).to_parquet(
        root / "selection.parquet"
    )
    inputs = {
        name: dict(path=str(path), sha256=audit.sha(path))
        for name, path in (
            ("audit_indices", root / "ids.npy"),
            ("selection_identities", root / "selection.parquet"),
        )
    }
    audit.write(
        root / "source/MANIFEST.json", dict(inputs=inputs, parent=str(root / "parent"))
    )
    audit.write(root / "parent/MANIFEST.json", dict(settings=dict(cut=29)))
    audit.write(root / "convergence/MANIFEST.json", dict(source=str(root / "source")))
    audit.write(
        root / "MANIFEST.json",
        dict(
            inputs={},
            convergence=str(root / "convergence"),
            settings=dict(decoder_objects=4, decoder_batch=2),
        ),
    )
    runtime = Runtime(
        context=SimpleNamespace(model_config={}),
        latent_spec=SimpleNamespace(names=("z_obs",)),
        feature_stats=SimpleNamespace(band_names=("lsst_r", "lsst_i")),
    )
    monkeypatch.setattr(
        feniks_ratio_ladder,
        "_runtime",
        lambda *args: (None, runtime, None, dict(model={}, bands=["lsst_r", "lsst_i"])),
    )
    monkeypatch.setattr(
        feniks_failure_modes,
        "_truth_x",
        lambda *args: (np.ones((4, 1)), np.ones((4, 1))),
    )
    data = {}
    for band in ("lsst_r", "lsst_i"):
        data.update(
            {
                f"flux_true_{band}": np.ones(4),
                f"flux_{band}": np.array([1.01, 0.99, 1.02, 0.98]),
                f"fluxerr_{band}": np.full(4, 0.1),
                f"mask_{band}": np.ones(4, bool),
            }
        )
    monkeypatch.setattr(
        feniks_failure_modes, "_catalogue", lambda *args: pd.DataFrame(data)
    )
    monkeypatch.setattr(
        posterior_target,
        "safe_decoder_inputs",
        lambda x, spec: (x, jnp.ones(len(x), dtype=bool)),
    )
    monkeypatch.setattr(
        posterior_target, "_apply_model_calibration", lambda model, flux, cfg: flux
    )

    def flux(x, spec, context, *args):
        value = (
            1.0
            if context.model_config["photometry_integrator"] == "merged_gauss4_v1"
            else 1.2
        )
        return jnp.full((len(x), 2), value)

    monkeypatch.setattr(decoder, "model_flux_from_x", flux)
    audit.decoder(root)
    final = audit.read(root / "decoder/FINAL.json")
    assert final["merged_p95_max"] == 0
    assert abs(final["baseline_p95_max"] - 2) < 1e-5
    assert final["selection_identity_mismatches"] == 0
    assert not final["full_decoder_qualified"]
    assert runtime.context.model_config == {}
    (root / "decoder/FINAL.json").unlink()

    def forbidden(*args):
        raise AssertionError("completed batches must not be decoded again")

    monkeypatch.setattr(decoder, "model_flux_from_x", forbidden)
    audit.decoder(root)
    assert audit.receipt_valid(root / "decoder")
