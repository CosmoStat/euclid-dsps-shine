#!/usr/bin/env python3
"""Stage runner for the final no-truth FENIKS SC-ASMC-EM workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-devices", type=int)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--estep-shards", type=int, required=True)

    sleep = sub.add_parser("sleep")
    sleep.add_argument("--resume-state", type=Path)
    sleep.add_argument("--smoke", action="store_true")

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--objects", type=int, default=8)
    smoke.add_argument("--row-offset", type=int, default=0)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--parallel-shards", type=int, required=True)

    estep = sub.add_parser("estep")
    estep.add_argument("--iteration", type=int, choices=(1, 2), required=True)
    estep.add_argument("--shard-id", type=int, required=True)
    estep.add_argument("--shard-count", type=int, required=True)

    merge_estep = sub.add_parser("merge-estep")
    merge_estep.add_argument("--iteration", type=int, choices=(1, 2), required=True)
    merge_estep.add_argument("--shard-count", type=int, required=True)

    mstep = sub.add_parser("prior-mstep")
    mstep.add_argument("--iteration", type=int, choices=(1, 2), required=True)

    reweight = sub.add_parser("reweight")
    reweight.add_argument("--iteration", type=int, choices=(1, 2), required=True)
    reweight.add_argument("--shard-id", type=int, required=True)
    reweight.add_argument("--shard-count", type=int, required=True)

    merge_reweight = sub.add_parser("merge-reweight")
    merge_reweight.add_argument("--iteration", type=int, choices=(1, 2), required=True)
    merge_reweight.add_argument("--shard-count", type=int, required=True)

    repair = sub.add_parser("repair-final")
    repair.add_argument("--shard-id", type=int, required=True)
    repair.add_argument("--shard-count", type=int, required=True)

    merge_repair = sub.add_parser("merge-repair-final")
    merge_repair.add_argument("--shard-count", type=int, required=True)

    distill = sub.add_parser("q-distill")
    distill.add_argument("--iteration", type=int, choices=(1,), required=True)

    sub.add_parser("report")
    sub.add_parser("validate")
    sub.add_parser("mark-training-complete")

    single = sub.add_parser("run-single-node")
    single.add_argument("--estep-shards", type=int, default=1)

    sub.add_parser("status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from euclid_dsps.config import load_config
    from euclid_dsps.jax_runtime import apply_jax_runtime_env

    config = load_config(args.config)
    if args.catalog is not None:
        config["catalog_path"] = str(args.catalog.resolve())
    apply_jax_runtime_env(config.get("runtime", {}) or {})
    import jax

    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")
    if args.expected_devices is not None and len(jax.local_devices()) != int(
        args.expected_devices
    ):
        raise RuntimeError(
            f"expected {args.expected_devices} local devices, got {jax.local_devices()}"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        _prepare(config, args.out, int(args.estep_shards))
    elif args.command == "sleep":
        _sleep(config, args.out, args.resume_state, bool(args.smoke))
    elif args.command == "smoke":
        _smoke(
            config,
            args.out,
            objects=int(args.objects),
            row_offset=int(args.row_offset),
        )
    elif args.command == "preflight":
        _preflight(config, args.out, int(args.parallel_shards))
    elif args.command == "estep":
        _estep(
            config,
            args.out,
            iteration=int(args.iteration),
            shard_id=int(args.shard_id),
            shard_count=int(args.shard_count),
        )
    elif args.command == "merge-estep":
        _merge_estep(
            args.out,
            iteration=int(args.iteration),
            shard_count=int(args.shard_count),
        )
    elif args.command == "prior-mstep":
        _prior_mstep(config, args.out, iteration=int(args.iteration))
    elif args.command == "reweight":
        _reweight(
            config,
            args.out,
            iteration=int(args.iteration),
            shard_id=int(args.shard_id),
            shard_count=int(args.shard_count),
        )
    elif args.command == "merge-reweight":
        _merge_reweight(
            args.out,
            iteration=int(args.iteration),
            shard_count=int(args.shard_count),
        )
    elif args.command == "q-distill":
        _distill(config, args.out, iteration=int(args.iteration))
    elif args.command == "repair-final":
        _repair_final(
            config,
            args.out,
            shard_id=int(args.shard_id),
            shard_count=int(args.shard_count),
        )
    elif args.command == "merge-repair-final":
        _merge_repair_final(args.out, shard_count=int(args.shard_count))
    elif args.command == "report":
        _report(config, args.out)
    elif args.command == "validate":
        _validate(config, args.out)
    elif args.command == "mark-training-complete":
        _mark_training_complete(args.out)
    elif args.command == "run-single-node":
        _run_single_node(config, args.out, int(args.estep_shards))
    elif args.command == "status":
        print(json.dumps(_status(args.out), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


def _prepare(config: dict, root: Path, estep_shards: int) -> dict:
    from euclid_dsps.amortized.sc_asmc_config import validate_sc_asmc_em_config
    from euclid_dsps.amortized.sc_asmc_manifest import prepare_sc_asmc_manifest
    from euclid_dsps.amortized.sc_asmc_training import (
        initialize_sc_model,
        prepare_sc_runtime,
    )

    validate_sc_asmc_em_config(config)
    manifest = prepare_sc_asmc_manifest(
        config,
        root / "manifest",
        n_estep_shards=int(estep_shards),
        resume=True,
    )
    runtime = prepare_sc_runtime(
        config,
        root / "runtime",
        feature_train_rows=manifest["artifacts"]["feature_train_rows"]["path"],
        heldout_rows=manifest["artifacts"]["heldout_rows"]["path"],
    )
    initialization = initialize_sc_model(config, runtime, root / "initialization")
    payload = {
        "status": "PASS",
        "manifest": str((root / "manifest" / "run_manifest.json").resolve()),
        "selected_objects": manifest["objects"]["selected"],
        "estep_shards": int(estep_shards),
        "initialization": initialization,
    }
    _write_json(root / "PREPARE_PASS.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def _sleep(
    config: dict,
    root: Path,
    resume_state: Path | None,
    smoke: bool,
) -> dict:
    from euclid_dsps.amortized.sc_asmc_training import (
        prepare_sc_runtime,
        train_sleep_bootstrap,
    )

    manifest = _manifest(root)
    runtime = prepare_sc_runtime(
        config,
        root / "runtime",
        feature_train_rows=manifest["artifacts"]["feature_train_rows"]["path"],
        heldout_rows=manifest["artifacts"]["heldout_rows"]["path"],
    )
    automatic_state = root / "sleep" / "states" / "sleep_latest.eqx"
    if resume_state is None and automatic_state.is_file():
        resume_state = automatic_state
    result = train_sleep_bootstrap(
        config,
        runtime,
        root / "sleep",
        resume_state=resume_state,
        smoke=smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _preflight(config: dict, root: Path, parallel_shards: int) -> dict:
    from euclid_dsps.amortized.sc_asmc_active import run_bounded_active_bootstrap
    from euclid_dsps.amortized.sc_asmc_estep import run_integrated_budget_preflight

    manifest = _manifest(root)
    runtime = _runtime(config, root, worker_label="preflight")
    q_raw, q_ema, p0 = _q0_p0(root)
    rows = np.load(manifest["artifacts"]["preflight_rows"]["path"])
    first = run_integrated_budget_preflight(
        config,
        runtime,
        manifest,
        preflight_rows=rows,
        out_dir=root / "preflight",
        attempt=1,
        q_checkpoint=q_raw,
        q_ema_checkpoint=q_ema,
        prior_checkpoint=p0,
        seed=int(manifest["seed"]) + 1_000,
        parallel_shards=int(parallel_shards),
    )
    if first["continue_full_catalogue"]:
        print(json.dumps(first, indent=2, sort_keys=True), flush=True)
        return first
    if not first["active_bootstrap_required"]:
        raise SystemExit("SC-ASMC-EM preflight aborted before full-catalogue work")
    active = run_bounded_active_bootstrap(
        config,
        runtime,
        manifest,
        failed_preflight_bank_root=root / "preflight" / "attempt_1" / "bank",
        q_checkpoint=q_raw,
        q_ema_checkpoint=q_ema,
        prior_checkpoint=p0,
        out_dir=root / "preflight" / "active_bootstrap",
        seed=int(manifest["seed"]) + 2_000,
    )
    second = run_integrated_budget_preflight(
        config,
        runtime,
        manifest,
        preflight_rows=rows,
        out_dir=root / "preflight",
        attempt=2,
        q_checkpoint=active["q_raw_checkpoint"],
        q_ema_checkpoint=active["q_ema_checkpoint"],
        prior_checkpoint=p0,
        seed=int(manifest["seed"]) + 1_000,
        parallel_shards=int(parallel_shards),
    )
    if not second["continue_full_catalogue"]:
        raise SystemExit("SC-ASMC-EM preflight failed twice; full run aborted cleanly")
    print(json.dumps(second, indent=2, sort_keys=True), flush=True)
    return second


def _estep(
    config: dict,
    root: Path,
    *,
    iteration: int,
    shard_id: int,
    shard_count: int,
) -> dict:
    from euclid_dsps.amortized.sc_asmc_estep import run_sc_estep_rows

    _require_preflight(root)
    manifest = _manifest(root)
    if int(manifest["e_step_shards"]["count"]) != int(shard_count):
        raise ValueError("launcher shard count differs from immutable run manifest")
    rows = np.load(manifest["artifacts"][f"estep_shard_{int(shard_id):02d}"]["path"])
    runtime = _runtime(config, root, worker_label=f"estep_{iteration}_{shard_id}")
    q_raw, q_ema, prior = _components_for_estep(root, iteration)
    result = run_sc_estep_rows(
        config,
        runtime,
        manifest,
        row_indices=rows,
        bank_root=root / "banks" / f"em{iteration}",
        worker_id=int(shard_id),
        iteration=int(iteration),
        q_checkpoint=q_raw,
        q_ema_checkpoint=q_ema,
        prior_checkpoint=prior,
        seed=int(manifest["seed"]) + 10_000 * int(iteration) + int(shard_id),
        resume=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _merge_estep(root: Path, *, iteration: int, shard_count: int) -> dict:
    from euclid_dsps.amortized.posterior_bank import merge_posterior_bank_shards

    bank = root / "banks" / f"em{iteration}"
    _require_worker_receipts(bank, shard_count)
    paths = sorted((bank / "shards").glob("shard_*"))
    expected = np.load(_manifest(root)["artifacts"]["selected_rows"]["path"])
    result = merge_posterior_bank_shards(
        bank,
        [str(path) for path in paths],
        expected_row_indices=expected,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _prior_mstep(config: dict, root: Path, *, iteration: int) -> dict:
    from euclid_dsps.amortized.sc_asmc_mstep import run_prior_mstep

    manifest = _manifest(root)
    runtime = _runtime(config, root, worker_label=f"mstep_{iteration}")
    q_raw, q_ema, prior = _components_for_estep(root, iteration)
    result = run_prior_mstep(
        config,
        runtime,
        input_bank_manifest=root
        / "banks"
        / f"em{iteration}"
        / "posterior_bank_manifest.json",
        heldout_rows=np.load(manifest["artifacts"]["heldout_rows"]["path"]),
        q_checkpoint=q_ema,
        old_prior_checkpoint=prior,
        out_dir=root / f"mstep{iteration}",
        iteration=int(iteration),
        seed=int(manifest["seed"]) + 20_000 * int(iteration),
        resume=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _reweight(
    config: dict,
    root: Path,
    *,
    iteration: int,
    shard_id: int,
    shard_count: int,
) -> dict:
    from euclid_dsps.amortized.sc_asmc_reweight import (
        reweight_and_refresh_bank_worker,
    )

    manifest = _manifest(root)
    runtime = _runtime(config, root, worker_label=f"reweight_{iteration}_{shard_id}")
    q_raw, q_ema, old_prior = _components_for_estep(root, iteration)
    new_prior = _mstep_receipt(root, iteration)["prior_checkpoint"]
    result = reweight_and_refresh_bank_worker(
        config,
        runtime,
        manifest,
        input_bank_manifest=root
        / "banks"
        / f"em{iteration}"
        / "posterior_bank_manifest.json",
        output_bank_root=root / "banks" / f"em{iteration}_p{iteration}",
        worker_id=int(shard_id),
        worker_count=int(shard_count),
        q_checkpoint=q_raw,
        q_ema_checkpoint=q_ema,
        old_prior_checkpoint=old_prior,
        new_prior_checkpoint=new_prior,
        seed=int(manifest["seed"]) + 30_000 * int(iteration) + int(shard_id),
        resume=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _merge_reweight(root: Path, *, iteration: int, shard_count: int) -> dict:
    from euclid_dsps.amortized.posterior_bank import merge_posterior_bank_shards

    bank = root / "banks" / f"em{iteration}_p{iteration}"
    _require_worker_receipts(bank, shard_count)
    paths = sorted((bank / "shards").glob("shard_*"))
    expected = np.load(_manifest(root)["artifacts"]["selected_rows"]["path"])
    result = merge_posterior_bank_shards(
        bank,
        [str(path) for path in paths],
        expected_row_indices=expected,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _repair_final(
    config: dict,
    root: Path,
    *,
    shard_id: int,
    shard_count: int,
) -> dict:
    """Retry unresolved final rows with frozen q1 and p2; do not run another EM."""
    from euclid_dsps.amortized.sc_asmc_reweight import (
        reweight_and_refresh_bank_worker,
    )

    manifest = _manifest(root)
    runtime = _runtime(config, root, worker_label=f"repair_final_{shard_id}")
    distill = json.loads(
        (root / "distill1" / "q_distillation_em1_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    p2 = _mstep_receipt(root, 2)["prior_checkpoint"]
    result = reweight_and_refresh_bank_worker(
        config,
        runtime,
        manifest,
        input_bank_manifest=root / "banks" / "em2_p2" / "posterior_bank_manifest.json",
        output_bank_root=root / "banks" / "em2_p2_repaired",
        worker_id=int(shard_id),
        worker_count=int(shard_count),
        q_checkpoint=distill["q_raw_checkpoint"],
        q_ema_checkpoint=distill["q_ema_checkpoint"],
        old_prior_checkpoint=p2,
        new_prior_checkpoint=p2,
        seed=int(manifest["seed"]) + 90_000 + int(shard_id),
        resume=True,
        refresh_unresolved=True,
        refresh_low_ess=False,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _merge_repair_final(root: Path, *, shard_count: int) -> dict:
    from euclid_dsps.amortized.posterior_bank import merge_posterior_bank_shards

    bank = root / "banks" / "em2_p2_repaired"
    _require_worker_receipts(bank, shard_count)
    paths = sorted((bank / "shards").glob("shard_*"))
    expected = np.load(_manifest(root)["artifacts"]["selected_rows"]["path"])
    result = merge_posterior_bank_shards(
        bank,
        [str(path) for path in paths],
        expected_row_indices=expected,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _distill(config: dict, root: Path, *, iteration: int) -> dict:
    from euclid_dsps.amortized.sc_asmc_distill import distill_q_from_full_bank

    manifest = _manifest(root)
    runtime = _runtime(config, root, worker_label=f"distill_{iteration}")
    _q_raw, q_ema, _old_prior = _components_for_estep(root, iteration)
    prior = _mstep_receipt(root, iteration)["prior_checkpoint"]
    result = distill_q_from_full_bank(
        config,
        runtime,
        input_bank_manifest=root
        / "banks"
        / f"em{iteration}_p{iteration}"
        / "posterior_bank_manifest.json",
        heldout_rows=np.load(manifest["artifacts"]["heldout_rows"]["path"]),
        q_checkpoint=q_ema,
        prior_checkpoint=prior,
        out_dir=root / f"distill{iteration}",
        iteration=int(iteration),
        seed=int(manifest["seed"]) + 40_000 * int(iteration),
        resume=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _run_single_node(config: dict, root: Path, estep_shards: int) -> None:
    if int(estep_shards) != 1:
        raise ValueError("run-single-node requires --estep-shards 1")
    _prepare(config, root, 1)
    _sleep(config, root, None, False)
    _preflight(config, root, 1)
    _estep(config, root, iteration=1, shard_id=0, shard_count=1)
    _merge_estep(root, iteration=1, shard_count=1)
    _prior_mstep(config, root, iteration=1)
    _reweight(config, root, iteration=1, shard_id=0, shard_count=1)
    _merge_reweight(root, iteration=1, shard_count=1)
    _distill(config, root, iteration=1)
    _estep(config, root, iteration=2, shard_id=0, shard_count=1)
    _merge_estep(root, iteration=2, shard_count=1)
    _prior_mstep(config, root, iteration=2)
    _reweight(config, root, iteration=2, shard_id=0, shard_count=1)
    _merge_reweight(root, iteration=2, shard_count=1)
    _mark_training_complete(root)
    _report(config, root)
    _validate(config, root)


def _smoke(
    config: dict,
    root: Path,
    *,
    objects: int,
    row_offset: int = 0,
) -> dict:
    import jax

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        _selection_alpha_gradient_preflight,
    )
    from euclid_dsps.amortized.posterior_bank import (
        merge_posterior_bank_shards,
        sha256_file,
    )
    from euclid_dsps.amortized.sc_asmc_estep import (
        run_sc_estep_rows,
        summarize_bank_shards,
    )
    from euclid_dsps.amortized.sc_asmc_training import load_sc_model

    manifest = _manifest(root)
    rows = np.load(manifest["artifacts"]["preflight_rows"]["path"])
    offset = max(int(row_offset), 0)
    if offset >= len(rows):
        raise ValueError(
            f"smoke row offset {offset} exceeds preflight cohort of {len(rows)}"
        )
    count = min(max(int(objects), 1), len(rows) - offset)
    rows = np.sort(rows[offset : offset + count])
    runtime = _runtime(config, root, worker_label="smoke")
    q_raw, q_ema, prior = _q0_p0(root)
    bank_root = root / "smoke" / "bank"
    worker = run_sc_estep_rows(
        config,
        runtime,
        manifest,
        row_indices=rows,
        bank_root=bank_root,
        worker_id=0,
        iteration=0,
        q_checkpoint=q_raw,
        q_ema_checkpoint=q_ema,
        prior_checkpoint=prior,
        seed=int(manifest["seed"]) + 500,
        resume=True,
    )
    shard_paths = [record["path"] for record in worker["shards"]]
    bank = merge_posterior_bank_shards(
        bank_root,
        shard_paths,
        expected_row_indices=rows,
    )
    model = load_sc_model(
        config,
        runtime,
        q_checkpoint=q_ema,
        prior_checkpoint=prior,
    )
    selection = _selection_alpha_gradient_preflight(
        runtime,
        model,
        jax.random.PRNGKey(int(manifest["seed"]) + 501),
    )
    if not selection["finite"] or not selection["nonzero"]:
        raise RuntimeError("4-GPU smoke selection score gradient failed")
    payload = {
        "status": "PASS",
        "phase": "four_device_smoke",
        "c0_scope_statement": manifest["c0_scope_statement"],
        "truth_used": False,
        "objects": count,
        "row_offset": offset,
        "devices": worker["devices"],
        "posterior_bank": str((bank_root / "posterior_bank_manifest.json").resolve()),
        "posterior_bank_sha256": sha256_file(
            bank_root / "posterior_bank_manifest.json"
        ),
        "run_manifest_sha256": sha256_file(root / "manifest" / "run_manifest.json"),
        "dataset_sha256": manifest["dataset"]["sha256"],
        "workflow_config_hash": manifest["config_sha256"],
        "code_commit": manifest["code"]["commit"],
        "bank_objects": int(bank["object_count"]),
        "hierarchy": summarize_bank_shards(shard_paths),
        "selection_gradient": selection,
        "scientific_result": False,
        "authorizes_big_run": False,
    }
    smoke_receipt = root / "smoke" / "SMOKE_PASS.json"
    _write_json(smoke_receipt, payload)
    _write_text(root / "smoke" / "SMOKE_PASS", sha256_file(smoke_receipt) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def _mark_training_complete(root: Path) -> dict:
    required = (
        root / "mstep2" / "prior_mstep_2_receipt.json",
        root / "banks" / "em2_p2" / "posterior_bank_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot complete training; missing final phases: {missing}")
    payload = {
        "status": "complete",
        "phase": "training_complete",
        "c0_scope_statement": (
            "We infer the parent distribution within the predefined FENIKS "
            "refinement and catalogue-support domain, while explicitly correcting "
            "the additional observed r<25 selection."
        ),
        "outer_iterations": 2,
        "truth_used": False,
        "nuts_used": False,
        "static_rws_used": False,
        "next_action": "generate report and run no-truth validation",
    }
    _write_json(root / "TRAINING_COMPLETE.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def _report(config: dict, root: Path) -> dict:
    from euclid_dsps.amortized.sc_asmc_report import generate_sc_asmc_report

    runtime = _runtime(config, root, worker_label="report")
    result = generate_sc_asmc_report(
        config,
        runtime,
        run_root=root,
        out_dir=root / "report",
        resume=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _validate(config: dict, root: Path) -> dict:
    from euclid_dsps.amortized.sc_asmc_validate import (
        validate_and_write_final_receipt,
    )

    result = validate_and_write_final_receipt(config, run_root=root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _runtime(config: dict, root: Path, *, worker_label: str):
    from euclid_dsps.amortized.sc_asmc_training import prepare_sc_runtime

    manifest = _manifest(root)
    return prepare_sc_runtime(
        config,
        root / ".runtime_cache" / worker_label,
        feature_train_rows=manifest["artifacts"]["feature_train_rows"]["path"],
        heldout_rows=manifest["artifacts"]["heldout_rows"]["path"],
    )


def _manifest(root: Path) -> dict:
    from euclid_dsps.amortized.sc_asmc_manifest import validate_sc_asmc_manifest

    return validate_sc_asmc_manifest(root / "manifest" / "run_manifest.json")


def _q0_p0(root: Path) -> tuple[str, str, str]:
    sleep = json.loads(
        (root / "sleep" / "sleep_receipt.json").read_text(encoding="utf-8")
    )
    active_path = (
        root / "preflight" / "active_bootstrap" / "active_bootstrap_receipt.json"
    )
    if active_path.is_file():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        q_raw = active["q_raw_checkpoint"]
        q_ema = active["q_ema_checkpoint"]
    else:
        q_raw = sleep["q_raw_checkpoint"]
        q_ema = sleep["q_ema_checkpoint"]
    initialization = json.loads(
        (root / "initialization" / "initialization_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    return q_raw, q_ema, initialization["prior_p0"]["path"]


def _components_for_estep(root: Path, iteration: int) -> tuple[str, str, str]:
    if int(iteration) == 1:
        return _q0_p0(root)
    distill = json.loads(
        (root / "distill1" / "q_distillation_em1_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    prior = _mstep_receipt(root, 1)["prior_checkpoint"]
    return distill["q_raw_checkpoint"], distill["q_ema_checkpoint"], prior


def _mstep_receipt(root: Path, iteration: int) -> dict:
    return json.loads(
        (
            root
            / f"mstep{int(iteration)}"
            / f"prior_mstep_{int(iteration)}_receipt.json"
        ).read_text(encoding="utf-8")
    )


def _require_preflight(root: Path) -> None:
    path = root / "preflight" / "PREFLIGHT_PASS.json"
    if not path.is_file():
        raise RuntimeError("full E-step refused: integrated preflight has not passed")


def _require_worker_receipts(bank: Path, count: int) -> None:
    missing = [
        str(bank / f"worker_{index:02d}_receipt.json")
        for index in range(int(count))
        if not (bank / f"worker_{index:02d}_receipt.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"posterior-bank workers are incomplete: {missing}")


def _status(root: Path) -> dict:
    checks = {
        "prepare": root / "PREPARE_PASS.json",
        "sleep": root / "sleep" / "sleep_receipt.json",
        "preflight": root / "preflight" / "PREFLIGHT_PASS.json",
        "bank_em1": root / "banks" / "em1" / "posterior_bank_manifest.json",
        "prior_p1": root / "mstep1" / "prior_mstep_1_receipt.json",
        "bank_em1_p1": root / "banks" / "em1_p1" / "posterior_bank_manifest.json",
        "q1": root / "distill1" / "q_distillation_em1_receipt.json",
        "bank_em2": root / "banks" / "em2" / "posterior_bank_manifest.json",
        "prior_p2": root / "mstep2" / "prior_mstep_2_receipt.json",
        "bank_em2_p2": root / "banks" / "em2_p2" / "posterior_bank_manifest.json",
        "bank_em2_p2_repaired": root
        / "banks"
        / "em2_p2_repaired"
        / "posterior_bank_manifest.json",
        "training_complete": root / "TRAINING_COMPLETE.json",
        "report": root / "report" / "report_receipt.json",
        "final_receipt": root / "FINAL_RECEIPT.json",
        "final_pass": root / "FINAL_PASS",
    }
    return {
        "root": str(root.resolve()),
        "phases": {
            name: {"complete": path.is_file(), "path": str(path.resolve())}
            for name, path in checks.items()
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
