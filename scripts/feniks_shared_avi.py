"""Prepare and run a small shared AVI continuation, without posterior teachers."""

import argparse
import copy
import subprocess
from pathlib import Path

import numpy as np
import yaml

from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.config import load_config
from scripts.feniks_qualified_local_vi import verify_night
from scripts.run_feniks_sc_drws_balanced_npe import read, write


def avi_config(source):
    cfg = copy.deepcopy(source)
    cfg["truth"] = {"parameter_columns": {}}
    cfg["extra_columns"] = []
    a = cfg["amortized"]
    a["objective"] = {
        "mode": "stochastic_elbo",
        "selection_correction": {"enabled": False},
        "sleep": {"enabled": False},
    }
    a.setdefault("prior", {})["train_jointly"] = False
    a.pop("truth_free_validation", None)
    a.setdefault("data", {}).update(
        use_redshift_for_split=False, stratify_column=None, redshift_bins=[]
    )
    a.setdefault("training", {}).update(
        learning_rate=3e-6,
        gradient_clip_norm=5.0,
        kl_annealing_epochs=0,
        kl_weight_max=1.0,
        best_checkpoint_metric="validation_loss",
        best_checkpoint_min_epoch=1,
    )
    a.setdefault("output", {}).update(checkpoint_every=1, save_posterior_preview=False)
    return cfg


def prepare(source, root):
    verified = verify_night(source)
    cfg = avi_config(load_config(verified["source"]["config"]))
    train = np.load(source / "cohorts/train.npy", allow_pickle=False)
    validation = np.load(source / "cohorts/tracking.npy", allow_pickle=False)
    if np.intersect1d(train, validation).size or not len(train) or not len(validation):
        raise ValueError("nonempty disjoint training/validation cohorts required")
    root.mkdir(parents=True, exist_ok=False)
    # Bound the first actual training run; these rows are not a confirmation cohort.
    np.save(root / "train.npy", train[:512], allow_pickle=False)
    np.save(root / "validation.npy", validation[:128], allow_pickle=False)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    write(
        root / "RUN_MANIFEST.json",
        dict(
            source=verified["source"],
            precision_root=str(source),
            code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            hashes={
                name: sha256_file(root / name)
                for name in ("train.npy", "validation.npy", "config.yaml")
            },
            seeds=[267001, 267002],
            epochs=4,
            truth_used=False,
            population_training_started=False,
            scientific_promotion=False,
            transport="source native global training path; not local transport64 wrapper",
        ),
    )


def run(root, task):
    m = read(root / "RUN_MANIFEST.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != m["code_commit"]:
        raise ValueError("run from the prepared commit")
    verify_night(Path(m["precision_root"]))
    for name, digest in m["hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"changed input: {name}")
    cfg = root / "config.yaml"
    out = root / f"seed_{task}"
    if out.exists():
        raise ValueError("refusing to overwrite a training attempt")
    import sys

    command = [
        sys.executable,
        "-m",
        "euclid_dsps.cli",
        "--config",
        str(cfg),
        "amortized-train-diffsky",
        "--runtime",
        "gpu",
        "--dataset",
        str(load_config(cfg)["catalog_path"]),
        "--out",
        str(out),
        "--train-indices-file",
        str(root / "train.npy"),
        "--validation-indices-file",
        str(root / "validation.npy"),
        "--initial-checkpoint",
        m["source"]["checkpoint"],
        "--fixed-feature-stats",
        m["source"]["feature_stats"],
        "--epochs",
        str(m["epochs"]),
        "--batch-size",
        "32",
        "--jax-batch-size",
        "8",
        "--n-samples",
        "8",
        "--kl-annealing-epochs",
        "0",
        "--kl-weight-max",
        "1",
        "--validation-every",
        "1",
        "--best-checkpoint-min-epoch",
        "1",
        "--data-parallel",
        "single",
        "--freeze-prior",
        "--seed",
        str(m["seeds"][task]),
        "--no-progress",
    ]
    subprocess.run(command, check=True)
    checkpoint = out / "checkpoints/last.eqx"
    if not checkpoint.is_file():
        raise ValueError("training returned without final checkpoint")
    write(
        out / "AVI_TRAINING_COMPLETE.json",
        dict(
            status="TRAINING_COMPLETE",
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            scientific_promotion=False,
            population_training_started=False,
            posterior_evaluation="NOT_RUN",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "run"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--task", type=int, choices=[0, 1])
    args = p.parse_args()
    if args.mode == "prepare":
        if args.source is None:
            p.error("--source required")
        prepare(args.source.resolve(), args.root.resolve())
    else:
        if args.task is None:
            p.error("--task required")
        run(args.root.resolve(), args.task)
