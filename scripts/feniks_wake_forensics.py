"""Replay wake trajectories and inspect first accepted updates without selection."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.local_wake_diagnostic import (
    make_wake_step,
    wake_batch,
    wake_loss,
)
from scripts.feniks_objective_night import digest, read, write
from scripts.feniks_objective_night import prepare as prepare_base
from scripts.feniks_support_probe import verify_source


def prepare(pilot, root):
    pilot = pilot.resolve()
    final = read(pilot / "FINAL.json")
    if final["status"] != "OBJECTIVE_PILOT_COMPLETE" or final["cases_complete"] != 16:
        raise ValueError("requires completed transport64 pilot")
    source = read(pilot / "RUN_MANIFEST.json")
    if (
        final.get("scientific_promotion") is not False
        or final.get("transport_contract") != source.get("transport_contract")
        or "wake_forensic_reference" in source
    ):
        raise ValueError("invalid pilot contract")
    audit = read(pilot / "OBJECTIVE_AUDIT.json")
    if (
        audit["status"] != "PASS"
        or len(audit["audits"]) != 32
        or digest(pilot / "OBJECTIVE_AUDIT.json")
        != final["artifacts"]["OBJECTIVE_AUDIT.json"]["sha256"]
    ):
        raise ValueError("pilot audit missing or changed")
    for group in ("observed", "simulated"):
        for index in range(8):
            case = pilot / "cases" / f"{group}_{index:03d}"
            read(case / "COMPLETE.json")
            for start in range(2):
                saved = case / f"wake_{start}" / "draws_04096"
                contract = read(saved / "TRANSPORT_CONTRACT.json")
                if (
                    contract["checkpoint_sha256"] != digest(saved / "parameters.eqx")
                    or contract["manifest_sha256"]
                    != digest(pilot / "RUN_MANIFEST.json")
                    or contract["version"] != source["transport_contract"]
                ):
                    raise ValueError("wake checkpoint contract changed")
                if len(pd.read_csv(saved.parent / "optimization.csv")) != 16:
                    raise ValueError("expected 16 wake attempts")
    prepare_base(pilot, root)
    manifest = read(root / "RUN_MANIFEST.json")
    manifest.pop("night_extension")
    manifest.pop("objective_execution_recipe")
    manifest.pop("experiment_seed_offset")
    names = [
        p
        for p in pilot.rglob("*")
        if p.is_file() and p.suffix in (".json", ".csv", ".eqx")
    ]
    manifest["wake_forensic_reference"] = dict(
        path=str(pilot), hashes={str(p.relative_to(pilot)): digest(p) for p in names}
    )
    manifest["forensic_parameter_atol"] = 1e-9
    manifest["interpretation"] = (
        "Replay all wake starts and inspect first accepted update at fixed scales 0, .01, .1, 1. No selection, no population training."
    )
    write(root / "RUN_MANIFEST.json", manifest)


def interpolate(before, after, factor):
    return jax.tree.map(
        lambda a, b: a + factor * (b - a) if eqx.is_inexact_array(a) else a,
        before,
        after,
    )


def inspect_update(encoder, before, after, context, x, weights):
    arrays, static = eqx.partition(before, eqx.is_inexact_array)
    end = eqx.filter(after, eqx.is_inexact_array)
    direction = jax.tree.map(lambda a, b: b - a, arrays, end)
    objective = eqx.filter_jit(
        lambda p: wake_loss(encoder, eqx.combine(p, static), context, x, weights)
    )
    grad = eqx.filter_grad(objective)(arrays)
    ad = sum(
        float(jnp.sum(g * d))
        for g, d in zip(jax.tree.leaves(grad), jax.tree.leaves(direction), strict=True)
    )
    rows = []
    for h in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        plus = jax.tree.map(lambda a, d, h=h: a + h * d, arrays, direction)
        minus = jax.tree.map(lambda a, d, h=h: a - h * d, arrays, direction)
        rows.append(
            dict(
                step=h, ad=ad, fd=float((objective(plus) - objective(minus)) / (2 * h))
            )
        )
    deltas = {}
    for block in ("mean", "log_std", "layers"):
        a = jax.tree.leaves(eqx.filter(getattr(before, block), eqx.is_inexact_array))
        b = jax.tree.leaves(eqx.filter(getattr(after, block), eqx.is_inexact_array))
        deltas[block] = dict(
            l2=float(
                np.sqrt(
                    sum(
                        np.sum(np.asarray(y - x) ** 2)
                        for x, y in zip(a, b, strict=True)
                    )
                )
            ),
            max_abs=max(
                (
                    float(np.max(np.abs(np.asarray(y - x))))
                    for x, y in zip(a, b, strict=True)
                ),
                default=0.0,
            ),
        )
    return dict(
        directional_stencils=rows,
        parameter_deltas=deltas,
        before_loss=float(wake_loss(encoder, before, context, x, weights)),
        after_loss=float(wake_loss(encoder, after, context, x, weights)),
        interpretation="AD/FD along actual update, fixed stopped samples/weights; descriptive, not an automatic gradient PASS.",
    )


def run(root, manifest, encoder, prepared, target, spec, budget):
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import (
        evaluate_distribution,
        finite_json,
    )

    reference = manifest["wake_forensic_reference"]
    verify_source(reference)
    recipe = manifest["objective_recipe"]
    optimizer, step = make_wake_step(
        encoder,
        learning_rate=recipe["learning_rate"],
        minimum_ess=recipe["minimum_ess"],
        maximum_weight=recipe["maximum_weight"],
    )
    cases = []
    for number, (case, observation, generated, anchor, context, starts) in enumerate(
        prepared
    ):
        for start, original in enumerate(starts):
            folder = root / "cases" / case / f"wake_forensic_{start}"
            folder.mkdir(parents=True)
            parameters = original
            state = optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
            history = []
            inspected = False
            draws = recipe["wake_draws"]
            for iteration in range(recipe["decoder_draw_budgets"][-1] // draws):
                key = jax.random.PRNGKey(
                    70000000 + number * 100000 + start * 10000 + 5000 + iteration
                )
                x, logweights, info = wake_batch(
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
                candidate, next_state, metrics = step(
                    parameters, state, context, x, logweights
                )
                metrics = {
                    k: np.asarray(v).item() for k, v in jax.device_get(metrics).items()
                }
                if not metrics["finite"]:
                    raise ValueError("nonfinite replay update")
                if metrics["update_applied"] and not inspected:
                    detail = inspect_update(
                        encoder, parameters, candidate, context, x, logweights
                    )
                    np.savez_compressed(
                        folder / "fixed_batch.npz",
                        x=np.asarray(x),
                        logweights=np.asarray(logweights),
                    )
                    eqx.tree_serialise_leaves(folder / "before.eqx", parameters)
                    eqx.tree_serialise_leaves(folder / "after.eqx", candidate)
                    eqx.tree_serialise_leaves(folder / "adam_before.eqx", state)
                    detail["transport_contract"] = manifest.get("transport_contract")
                    detail["checkpoint_hashes"] = {
                        name: digest(folder / name)
                        for name in ("before.eqx", "after.eqx", "adam_before.eqx")
                    }
                    detail["attempt"] = iteration + 1
                    detail["evaluations"] = {}
                    for label, factor in (
                        ("zero", 0.0),
                        ("p01", 0.01),
                        ("p10", 0.1),
                        ("full", 1.0),
                    ):
                        q = interpolate(parameters, candidate, factor)
                        detail["evaluations"][label] = evaluate_distribution(
                            folder / label,
                            encoder,
                            q,
                            context,
                            observation,
                            target,
                            spec,
                            budget,
                            180000000 + number * 100000 + start * 10000,
                            2048,
                            generated,
                        )
                    write(folder / "UPDATE_AUDIT.json", finite_json(detail))
                    inspected = True
                parameters, state = candidate, next_state
                history.append(dict(attempt=iteration + 1, **metrics))
                pd.DataFrame(history).to_csv(folder / "replay.csv", index=False)
                write(
                    root / "PROGRESS.json",
                    dict(
                        stage="wake_forensics",
                        case=case,
                        start=start,
                        attempt=iteration + 1,
                        budget=budget.snapshot(),
                    ),
                )
            old = pd.read_csv(
                Path(reference["path"])
                / "cases"
                / case
                / f"wake_{start}"
                / "optimization.csv"
            )
            restored = eqx.tree_deserialise_leaves(
                Path(reference["path"])
                / "cases"
                / case
                / f"wake_{start}"
                / "draws_04096/parameters.eqx",
                original,
            )
            error = max(
                float(np.max(np.abs(np.asarray(a - b))))
                for a, b in zip(
                    jax.tree.leaves(eqx.filter(parameters, eqx.is_inexact_array)),
                    jax.tree.leaves(eqx.filter(restored, eqx.is_inexact_array)),
                    strict=True,
                )
            )
            result = dict(
                case=case,
                start=start,
                first_update_inspected=inspected,
                accepted=sum(h["update_applied"] for h in history),
                decisions_match=old.update_applied.tolist()
                == [h["update_applied"] for h in history],
                final_parameter_max_abs_delta=error,
            )
            result["replay_matches"] = bool(
                result["decisions_match"]
                and np.isfinite(error)
                and error <= manifest["forensic_parameter_atol"]
            )
            write(folder / "REPLAY.json", result)
            cases.append(result)
            verify_source(reference)
    write(
        root / "WAKE_FORENSICS.json",
        dict(status="DESCRIPTIVE_ONLY", replays=cases, scientific_promotion=False),
    )
    hashes = {
        str(p.relative_to(root)): digest(p)
        for p in (root / "cases").rglob("*")
        if p.is_file()
    }
    hashes["WAKE_FORENSICS.json"] = digest(root / "WAKE_FORENSICS.json")
    return dict(
        status="WAKE_FORENSICS_COMPLETE"
        if all(c["replay_matches"] for c in cases)
        else "WAKE_REPLAY_MISMATCH",
        replays_complete=len(cases),
        cases_complete=0,
        artifacts=hashes,
        scientific_promotion=False,
        population_training_started=False,
        transport_contract=manifest.get("transport_contract"),
        budget=budget.snapshot(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args.pilot, args.root)
