"""Controlled diagnostics must not mix a refitted parent with an old simulation bank."""

import json

import numpy as np
import pandas as pd


def test_full_validation_indices():
    from scripts.feniks_forward_diagnostics import evaluation_indices

    frame = pd.DataFrame(
        dict(
            selected=[True, True, False, True, True],
            split_bucket=[7000, 8499, 7100, 8500, 6999],
            parent_row_index=[45000, 49999, 4, 5, 6],
        )
    )
    np.testing.assert_array_equal(evaluation_indices(frame), [45000, 49999])


def test_prepare_preserves_parent_and_bank_contract(tmp_path, monkeypatch):
    import yaml

    from scripts import feniks_forward_diagnostics as diag

    parent = tmp_path / "parent"
    for folder in ("population", "posterior", "reference", "posterior_bank"):
        (parent / folder).mkdir(parents=True)
    for filename in ("population/best.eqx", "posterior/best.eqx", "basis.npz"):
        (parent / filename).write_bytes(b"checkpoint")
    diag.write(parent / "population/parent.json", {"u": [1]})
    diag.write(
        parent / "population/FINAL.json",
        dict(
            status="FORWARD_PARENT_COMPLETE",
            classifier_sha256=diag.sha(parent / "population/best.eqx"),
        ),
    )
    diag.write(
        parent / "posterior/FINAL.json",
        dict(
            status="FORWARD_POSTERIOR_COMPLETE",
            checkpoint_sha256=diag.sha(parent / "posterior/best.eqx"),
            parent_sha256=diag.sha(parent / "population/parent.json"),
        ),
    )
    pd.DataFrame(
        dict(
            selected=np.ones(600, bool),
            split_bucket=np.full(600, 7500),
            parent_row_index=np.arange(600),
        )
    ).to_parquet(parent / "selection_identities.parquet")
    original = dict(
        hashes={},
        blind_truth_parent=str(parent / "true_parent.parquet"),
        settings=dict(seed=42, classifier=dict(epochs=60), posterior=dict(epochs=100)),
    )
    monkeypatch.setattr(diag, "contract", lambda _: (original, original["settings"]))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                report_objects=8192,
                classifier=dict(epochs=180),
                posterior=dict(epochs=150),
                resources={},
            )
        )
    )
    root = tmp_path / "diagnostics"
    diag.prepare(parent, root, config)
    assert (root / "posterior/population").resolve() == parent / "population"
    assert not (root / "classifier/population").is_symlink()
    assert not (root / "posterior/posterior").is_symlink()
    assert (root / "baseline/posterior").resolve() == parent / "posterior"
    for arm in ("baseline", "classifier", "posterior"):
        m = diag.read(root / arm / "MANIFEST.json")
        assert m["settings"]["seed"] == 42
        assert len(np.load(m["evaluation_indices"])) == 600
    assert original["settings"]["posterior"]["epochs"] == 100


def test_warm_start_fixed_validation_and_decay(tmp_path):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from scripts.feniks_forward_population import sha, supervised_fit

    net = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(7))
    checkpoint = tmp_path / "initial.eqx"
    eqx.tree_serialise_leaves(checkpoint, net)
    digest = sha(checkpoint)
    x = np.arange(32, dtype=float).reshape(-1, 1) / 32
    y = 2 * x

    def loss(model, f, t):
        return jnp.mean((jax.vmap(model)(f) - t) ** 2, axis=-1)

    cfg = dict(
        initial_checkpoint=str(checkpoint),
        batch_size=8,
        seed=4,
        fixed_validation=True,
        validation_limit=16,
        epochs=3,
        learning_rate=0.01,
        lr_decay_every=1,
        lr_decay_factor=0.5,
        save_validation_losses=True,
    )
    out = tmp_path / "fit"
    out.mkdir()
    best = supervised_fit(net, loss, x, y, (x, y), cfg, out)
    rows = [
        json.loads(line) for line in (out / "training.jsonl").read_text().splitlines()
    ]
    assert [r["learning_rate"] for r in rows] == [0.01, 0.005, 0.0025]
    positions = np.load(out / "validation_positions.npy")
    initial = json.loads((out / "INITIAL_VALIDATION.json").read_text())["nll"]
    assert (
        float(loss(best, jnp.asarray(x[positions]), jnp.asarray(y[positions])).mean())
        <= initial
    )
    np.testing.assert_array_equal(
        np.load(out / "best_validation_losses.npz")["positions"], positions
    )
    assert sha(checkpoint) == digest


def test_ratio_reference_moments(tmp_path):
    from scripts.feniks_forward_diagnostics import ratio_checks

    (tmp_path / "population").mkdir()
    np.savez(
        tmp_path / "population/classifier_validation.npz",
        frequencies=[0.25, 0.75],
        component=[0, 1],
        log_prob=np.log([[0.25, 0.75], [0.25, 0.75]]),
    )
    assert ratio_checks(tmp_path)["ratio_reference_median_abs_error"] < 1e-12


def test_submission_independent_arms_and_summary_dependency(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    script = Path("scripts/submit_feniks_forward_diagnostics.sh").resolve()
    repo, parent, root, binaries = [
        tmp_path / p for p in ("repo", "parent", "run", "bin")
    ]
    for directory in (repo, parent, binaries):
        directory.mkdir()
    for directory in ("scripts", "configs", "euclid_dsps", "filters", "Data"):
        (repo / directory).mkdir()
    (repo / "pyproject.toml").touch()
    config = repo / "settings.yaml"
    config.write_text("resources: {}")
    commands = {
        "python": 'if [[ "$1" == "-m" ]]; then mkdir -p "$FORWARD_ROOT/logs"; '
        'else printf "10\\n12\\n20\\n20\\n1\\n"; fi',
        "sbatch": 'printf "%s %s\\n" "$FORWARD_MODE" "$*" >> "$TRACE"\n'
        'printf "%s\\n" "$((100 + $(wc -l < "$TRACE")))"',
    }
    for name, body in commands.items():
        executable = binaries / name
        executable.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        executable.chmod(0o755)
    trace = tmp_path / "trace"
    result = subprocess.run(
        ["bash", str(script), str(parent), str(root), str(config)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            PATH=f"{binaries}:{os.environ['PATH']}",
            SCRATCH=str(tmp_path / "scratch"),
            TRACE=str(trace),
        ),
    )
    assert result.returncode == 0, result.stderr
    submitted = trace.read_text().splitlines()
    assert len(submitted) == 5
    assert all("--dependency" not in line for line in submitted[:4])
    assert "--array=0-1%2" in submitted[3]
    assert "--dependency=afterok:101:102:103:104" in submitted[4]
    assert "83 GPU-hours" in result.stdout
