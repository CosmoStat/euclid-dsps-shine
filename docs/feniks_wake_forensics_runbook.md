# Wake update forensics

## Evidence and question

Operator output from pilot 1962310 reports 32/32 objective audits PASS and
16 completed cases. This qualifies the tested numerical objective, not posterior
quality or the wake update. At final K4096 every reverse/wake trajectory has
`bad_k=1`. Wake changes only 9/32 trajectories; all nine have lower ESS fractions
than their paired source evaluations. Several deteriorate after one update.
These observations do not yet establish a gradient bug or its cause.

The longer extension remains resource-gated. Do not lower its thresholds to
force training. Its remote final receipt has not been inspected here.

## Prescribed experiment

One node, one H100, at most 10 hours (9-hour internal budget), no multi-GPU
training. Re-audit all 32 starts under the existing transport64 contract. Replay
all original 16 wake attempts with the original seeds, batch size, optimizer and
rejection rules, starting from the same frozen long-VI checkpoints.

For each trajectory's first accepted update, save samples, stopped log weights,
parameters before/after and the pre-update Adam state. Measure the fixed-batch
wake loss, parameter displacements, and directional AD/FD along the actual update.
Evaluate amplitudes 0, 0.01, 0.1 and 1 with common fresh base noise, two independent
2048-draw replicates per amplitude (pooled K4096). Preserve joint draws. These
are fixed counterfactuals, not candidate checkpoint selection. The original
trajectory continues unchanged, not from a scaled candidate.

Compare all acceptance decisions and final parameters against the pilot.
Maximum absolute parameter discrepancy must be <=1e-9; otherwise the receipt
is `WAKE_REPLAY_MISMATCH`, not successful reproduction. Only the first accepted
update is inspected, not every later update. The AD/FD table is descriptive,
not a new automatic PASS gate. No scientific promotion is possible.

This separates three questions: does the update descend on its training batch;
does the finite difference agree with AD; does a smaller prescribed displacement
avoid the independent-evaluation deterioration? None alone validates RWS.

## Launch from Jean-Zay

Run after pulling the commit containing this script. A fresh output root is
required; existing experiments are never overwritten.

```bash
cd "$WORK/dsps-popcosmos"
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_wake_forensics.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_objective_transport64_pilot_v1" \
  "$BASE/frozen_parent_wake_forensics_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

This is an additional single-GPU job, independent of job 1962505. If that job
is running concurrently, peak combined allocation is two GPUs. Check its
`FINAL.json` rather than assuming the resource gate refused it.

Read back after completion:

```bash
python scripts/summarize_feniks_wake_forensics.py \
  "$BASE/frozen_parent_wake_forensics_v1"
```

The CPU readback verifies output hashes. Inspect `FINAL.json`,
`WAKE_FORENSICS.json`, all `REPLAY.json`, and `UPDATE_AUDIT.json` reports. A
completed Slurm job without a matching scientific receipt is not a result.
