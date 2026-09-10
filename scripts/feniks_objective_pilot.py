"""Audit-gated local objective comparison with immutable parent and source."""

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.features import make_encoder_features
from euclid_dsps.amortized.local_vi_diagnostic import (
    BudgetExceeded,
    initialize,
    make_step,
)
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.amortized.train import _array_tree_sha256
from scripts.feniks_support_probe import check_simulations, verify_source
from scripts.run_feniks_sc_drws_balanced_npe import write


def run_pilot(root, manifest, model, stats, spec, cases, target, budget):
    from euclid_dsps.amortized.local_vi_objective_audit import audit_objective
    from euclid_dsps.amortized.local_wake_diagnostic import make_wake_step, wake_batch
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import (
        evaluate_distribution,
        finite_json,
    )

    source = manifest["objective_pilot"]
    if "wake_descent_reference" in manifest:
        verify_source(manifest["wake_descent_reference"])
    if "wake_forensic_reference" in manifest:
        verify_source(manifest["wake_forensic_reference"])
    recipe = manifest.get("objective_execution_recipe", manifest["objective_recipe"])
    transport64 = (
        manifest.get("transport_contract") == "conditional_transport_float64_v1"
    )
    if bool(manifest.get("transport_contract")) != transport64:
        raise ValueError("unsupported transport contract")
    if transport64 and "transport_precision_reference" not in manifest:
        raise ValueError("transport64 pilot requires pinned qualification")
    encoder = model.encoder
    if transport64:
        from euclid_dsps.amortized.local_transport_precision import (
            DiagnosticTransport64,
            promote,
        )
        from scripts.feniks_transport_precision import pin_reference

        reference = manifest["transport_precision_reference"]
        if pin_reference(reference["path"], manifest, qualified=True) != reference:
            raise ValueError("transport qualification changed since preparation")
        encoder = DiagnosticTransport64(encoder)
    check_simulations(
        Path(source["path"]) / "SIMULATED_INPUTS.npz", root / "SIMULATED_INPUTS.npz"
    )
    fingerprint = _array_tree_sha256(model)
    prepared, audits = [], []

    def progress(stage, **fields):
        write(
            root / "PROGRESS.json",
            dict(stage=stage, **fields, budget=budget.snapshot()),
        )

    def unchanged():
        if "wake_descent_reference" in manifest:
            verify_source(manifest["wake_descent_reference"])
        verify_source(source)
        if "transport_precision_reference" in manifest:
            verify_source(manifest["transport_precision_reference"])
        if "night_gate_reference" in manifest:
            verify_source(manifest["night_gate_reference"])
        if "wake_forensic_reference" in manifest:
            verify_source(manifest["wake_forensic_reference"])
        if fingerprint != _array_tree_sha256(model):
            raise ValueError("frozen parent changed during objective pilot")
        for audit in audits:
            for name, digest in audit["artifacts"].items():
                if sha256_file(root / name) != digest:
                    raise ValueError(f"objective audit artifact changed: {name}")

    def finish(status, **fields):
        unchanged()
        final = dict(
            status=status,
            **fields,
            budget=budget.snapshot(),
            scientific_promotion=False,
            truth_used=False,
            population_training_started=False,
            prior_bitwise_unchanged=True,
            transport_contract=manifest.get("transport_contract", "historical_native"),
            adaptation_contract=manifest.get("adaptation_contract", "historical_wake"),
            artifacts={
                name: dict(sha256=sha256_file(root / name))
                for name in ("OBJECTIVE_AUDIT.json", "SIMULATED_INPUTS.npz")
            },
        )
        if "wake_backtracking" in manifest:
            final["wake_history_hashes"] = {
                str(path.relative_to(root)): sha256_file(path)
                for path in root.glob("cases/*/wake_*/optimization.csv")
            }
        write(root / "FINAL.json", finite_json(final))
        return final

    # Audit every prescribed start before constructing any optimizer.
    for number, (group, index, observation, generated) in enumerate(cases):
        case = f"{group}_{index:03d}"
        folder = root / "cases" / case
        features = make_encoder_features(
            observation.flux, observation.flux_err, stats, observation.mask
        )
        anchor, context = initialize(model, features)
        starts = []
        for start in (0, 1):
            path = (
                Path(source["path"])
                / "cases"
                / case
                / f"start_{start}"
                / "parameters.eqx"
            )
            parameters = eqx.tree_deserialise_leaves(path, anchor)
            audit_options = {}
            if transport64:
                from euclid_dsps.amortized.local_vi_objective_audit import _direction

                seed = 50000000 + number * 100000 + start * 10000
                audit_options = dict(
                    noise=jax.random.normal(
                        jax.random.PRNGKey(seed),
                        (recipe["audit_draws"],) + parameters.mean.shape,
                        dtype=parameters.mean.dtype,
                    ),
                    directions=[
                        (
                            block,
                            i,
                            _direction(
                                parameters,
                                block,
                                np.random.default_rng(seed + 10 * j + i),
                            ),
                        )
                        for j, block in enumerate(("mean", "log_std", "layers"))
                        for i in range(2)
                    ],
                )
                parameters = promote(parameters)
            starts.append(parameters)
            progress(
                "full_vi_objective_audit",
                case=case,
                start=start,
                audits_complete=len(audits),
            )
            audit = audit_objective(
                encoder,
                parameters,
                context,
                observation,
                target,
                budget,
                seed=50000000 + number * 100000 + start * 10000,
                draws=recipe["audit_draws"],
                **audit_options,
            )
            location = folder / f"audit_start_{start}"
            location.mkdir(parents=True, exist_ok=True)
            write(location / "AUDIT.json", finite_json(audit))
            pd.DataFrame(audit.get("rows", [])).to_csv(
                location / "stencils.csv", index=False
            )
            extra, names = {}, ["AUDIT.json", "stencils.csv"]
            if "transport_precision_reference" in manifest and not transport64:
                from euclid_dsps.amortized.local_transport_precision import (
                    compare_transport,
                )

                previous = json.loads(
                    (
                        Path(manifest["transport_precision_reference"]["path"])
                        / "cases"
                        / case
                        / f"audit_start_{start}"
                        / "AUDIT.json"
                    ).read_text()
                )
                if (audit["seed"], audit["noise_sha256"]) != (
                    previous["seed"],
                    previous["noise_sha256"],
                ):
                    raise ValueError(
                        "precision replay must reuse the original audit noise and seed"
                    )
                progress(
                    "transport_precision_audit",
                    case=case,
                    start=start,
                    audits_complete=len(audits),
                )
                comparison = compare_transport(
                    model.encoder,
                    parameters,
                    context,
                    observation,
                    target,
                    budget,
                    seed=audit["seed"],
                    draws=recipe["audit_draws"],
                )
                pd.DataFrame(comparison.pop("traces")).to_csv(
                    location / "transport_trace.csv", index=False
                )
                np.savez_compressed(
                    location / "transport_values.npz", **comparison.pop("arrays")
                )
                pd.DataFrame(comparison["transport64_audit"]["rows"]).to_csv(
                    location / "transport64_stencils.csv", index=False
                )
                comparison["native_reference_status"] = previous["status"]
                comparison["native_replay_status"] = audit["status"]
                write(location / "TRANSPORT_PRECISION.json", finite_json(comparison))
                names += [
                    "transport_trace.csv",
                    "transport64_stencils.csv",
                    "TRANSPORT_PRECISION.json",
                    "transport_values.npz",
                ]
                extra = dict(
                    transport64_status=comparison["transport64_audit"]["status"],
                    native_reference_status=previous["status"],
                )
            audits.append(
                dict(
                    case=case,
                    start=start,
                    status=audit["status"],
                    source_checkpoint_sha256=sha256_file(path),
                    artifacts={
                        str((location / name).relative_to(root)): sha256_file(
                            location / name
                        )
                        for name in names
                    },
                    **extra,
                )
            )
            write(
                root / "OBJECTIVE_AUDIT.json",
                dict(status="RUNNING", audits=audits, optimization_started=False),
            )
        prepared.append(
            (
                case,
                observation,
                generated,
                promote(anchor) if transport64 else anchor,
                context,
                starts,
            )
        )
    passed = all(item["status"] == "PASS" for item in audits)
    write(
        root / "OBJECTIVE_AUDIT.json",
        dict(
            status="PASS" if passed else "NOT_PASSED",
            audits=audits,
            optimization_started=False,
            scientific_promotion=False,
        ),
    )
    if "transport_precision_reference" in manifest and not transport64:
        return finish(
            "TRANSPORT_PRECISION_DIAGNOSTIC_COMPLETE",
            audits_complete=len(audits),
            cases_complete=0,
            optimization_started=False,
        )
    if not passed:
        return finish(
            "OBJECTIVE_AUDIT_NOT_PASSED", cases_complete=0, optimization_started=False
        )
    unchanged()
    if "wake_forensic_reference" in manifest:
        from scripts.feniks_wake_forensics import run as run_forensics

        result = run_forensics(root, manifest, encoder, prepared, target, spec, budget)
        unchanged()
        write(root / "FINAL.json", finite_json(result))
        return result
    reverse_optimizer, reverse_step = make_step(
        encoder,
        target,
        draws=recipe["reverse_draws"],
        learning_rate=recipe["learning_rate"],
    )
    wake_factory = make_wake_step
    if "wake_backtracking" in manifest:
        from euclid_dsps.amortized.local_wake_backtracking import make_guarded_wake_step

        wake_factory = make_guarded_wake_step
    wake_optimizer, wake_step = wake_factory(
        encoder,
        learning_rate=recipe["learning_rate"],
        minimum_ess=recipe["minimum_ess"],
        maximum_weight=recipe["maximum_weight"],
        **manifest.get("wake_backtracking", {}),
    )
    completed, began = [], time.monotonic()
    for number, (case, observation, generated, anchor, context, starts) in enumerate(
        prepared
    ):
        folder = root / "cases" / case
        eval_seed = (
            60000000 + number * 100000 + manifest.get("experiment_seed_offset", 0)
        )
        evaluate_distribution(
            folder / "amortized",
            encoder,
            anchor,
            context,
            observation,
            target,
            spec,
            budget,
            eval_seed,
            recipe["final_evaluation_draws"],
            generated,
        )
        outcomes = []
        for start, original in enumerate(starts):
            evaluate_distribution(
                folder / f"source_{start}",
                encoder,
                original,
                context,
                observation,
                target,
                spec,
                budget,
                eval_seed + 100 + start * 10,
                recipe["final_evaluation_draws"],
                generated,
            )
            for arm in ("reverse", "wake"):
                parameters = original
                optimizer, step = (
                    (reverse_optimizer, reverse_step)
                    if arm == "reverse"
                    else (wake_optimizer, wake_step)
                )
                state = optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
                draws = recipe[f"{arm}_draws"]
                local = folder / f"{arm}_{start}"
                local.mkdir(parents=True, exist_ok=True)
                history = []
                for iteration in range(recipe["decoder_draw_budgets"][-1] // draws):
                    key = jax.random.PRNGKey(
                        70000000
                        + number * 100000
                        + start * 10000
                        + (5000 if arm == "wake" else 0)
                        + iteration
                        + manifest.get("experiment_seed_offset", 0)
                    )
                    if arm == "reverse":
                        budget.charge(draws, gradient=True)
                        parameters, state, metrics = step(
                            parameters, state, context, observation, key
                        )
                    else:
                        x, logweights, batch_info = wake_batch(
                            encoder,
                            parameters,
                            anchor,
                            context,
                            observation,
                            target,
                            budget,
                            key,
                            draws=draws,
                        )
                        parameters, state, metrics = step(
                            parameters, state, context, x, logweights
                        )
                        metrics = dict(
                            metrics, **{f"batch_{k}": v for k, v in batch_info.items()}
                        )
                    metrics = {
                        k: np.asarray(v).item()
                        for k, v in jax.device_get(metrics).items()
                    }
                    used = (iteration + 1) * draws
                    history.append(
                        dict(attempt=iteration + 1, decoder_draws=used, **metrics)
                    )
                    pd.DataFrame(history).to_csv(
                        local / "optimization.csv", index=False
                    )
                    if not metrics["finite"]:
                        raise ValueError(
                            f"nonfinite objective update: {case} {arm} {start}"
                        )
                    progress(
                        "objective_pilot",
                        case=case,
                        arm=arm,
                        start=start,
                        decoder_draws=used,
                        cases_complete=len(completed),
                    )
                    if used in recipe["decoder_draw_budgets"]:
                        checkpoint = local / f"draws_{used:05d}"
                        checkpoint.mkdir(parents=True, exist_ok=True)
                        eqx.tree_serialise_leaves(
                            checkpoint / "parameters.eqx", parameters
                        )
                        if transport64:
                            write(
                                checkpoint / "TRANSPORT_CONTRACT.json",
                                dict(
                                    version=manifest["transport_contract"],
                                    checkpoint_sha256=sha256_file(
                                        checkpoint / "parameters.eqx"
                                    ),
                                    manifest_sha256=sha256_file(
                                        root / "RUN_MANIFEST.json"
                                    ),
                                    interpretation="Local parameters require the versioned conditional transport; not a native encoder checkpoint.",
                                ),
                            )
                        restored = eqx.tree_deserialise_leaves(
                            checkpoint / "parameters.eqx", original
                        )
                        final = used == recipe["decoder_draw_budgets"][-1]
                        result = evaluate_distribution(
                            checkpoint,
                            encoder,
                            restored,
                            context,
                            observation,
                            target,
                            spec,
                            budget,
                            eval_seed + 100 + start * 10,
                            recipe["final_evaluation_draws"]
                            if final
                            else recipe["intermediate_evaluation_draws"],
                            generated,
                        )
                        outcomes.append(
                            dict(
                                arm=arm,
                                start=start,
                                decoder_draws=used,
                                checkpoint_sha256=sha256_file(
                                    checkpoint / "parameters.eqx"
                                ),
                                applied_updates=sum(
                                    h.get("update_applied", True) for h in history
                                ),
                                attempts=len(history),
                                summary=result,
                            )
                        )
                # Rejected wake batches stay in the trajectory; never retry until accepted.
        unchanged()
        write(
            folder / "COMPLETE.json",
            finite_json(
                dict(
                    case=case,
                    outcomes=outcomes,
                    prior_bitwise_unchanged=True,
                    scientific_promotion=False,
                )
            ),
        )
        completed.append(case)
        if len(completed) == 2:
            estimate = (time.monotonic() - began) / 2 * (len(prepared) - 2)
            remaining = budget.seconds - (time.monotonic() - budget.started)
            write(
                root / "COST_PREFLIGHT.json",
                dict(
                    cases_measured=2,
                    estimated_remaining_seconds=estimate,
                    remaining_budget_seconds=remaining,
                    safety_factor=1.25,
                ),
            )
            if 1.25 * estimate > remaining:
                raise BudgetExceeded("objective pilot exceeds remaining allocation")
    return finish(
        "OBJECTIVE_PILOT_COMPLETE",
        cases_complete=len(completed),
        optimization_started=True,
    )
