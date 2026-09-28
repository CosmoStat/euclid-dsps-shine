# Targeted recovery and posterior continuation

## Re-audit after the 2026-09-28 numerical stop

Reported jobs: parent 263486 completed; audit 263487 blocked on target replay
error 0.00116167 > 0.001; GPU 263488 was cancelled. Both expert transport audits
passed. This alone does not establish CPU/GPU differences or a defective flow.
The old file retained theta but not the original x. Reconstructing x through a
bounded logit can amplify float64 rounding; this failure mode is locally reproduced.

The new `float64_theta_replay_v2` audit retains the 0.001 latent and 1e-5
transport thresholds. Each exception must be a bounded coordinate whose forward
replay agrees with stored theta within eight `nextafter` neighbours AND whose x
lies inside that interval mapped through the exact same monotone transform.
The actual saved/replayed samples are never clipped. Saturated endpoints, NaNs,
SFH discrepancies and transport failures still block. Directly saved x gets no
rounding exemption. Legacy theta compatibility does not certify exact latent
identity, calibration or the physical plausibility of the tails.

`audit/*/replay_coordinates.csv` records each extreme object's coordinate,
theta replay, latent replay, intrinsic roundtrip error, rounding compatibility
and verdict. Future evaluations save `latent_x` alongside `draws`, with unchanged
joint samples and calibration. Additional storage: 15 float64 values per draw
(about 120 MiB for 2048 objects x 512 draws); lightweight rsync still excludes NPZ.

Use the NEW code with `--reaudit OLD_RECOVERY NEW_RECOVERY`: source identities and
completed tied/report receipts are verified. Small reports/fits are copied with
identical content hashes so lightweight rsync works; source data stay untouched.
Only audit, conditional posterior continuation and report are submitted. The old failure
stays immutable. `--resume` still uses its original frozen code, so it does not
apply this repair. Existing banks and classifier are never regenerated.

The WORK inode quota can be avoided for code deployment by downloading the pinned
GitHub source archive into a new directory on fsn1, extracting only `euclid_dsps`,
`scripts`, `configs`, `pyproject.toml`, and linking the original `Data`. Do not
create another WORK worktree, delete results, or alter the dirty original checkout.

```bash
source "$BASE/avi_overnight_recovery_latest.env"
PREVIOUS="$RECOVERY"
NEW="$BASE/avi_overnight_recovery_$(date +%Y%m%d_%H%M%S)"
# Run from the newly deployed code directory, with shine active.
bash scripts/submit_feniks_overnight_recovery.sh --reaudit "$PREVIOUS" "$NEW"
```

Inspect the new audit before interpreting success: only coordinate diagnostics
can confirm the remote mismatch was bounded serialization, rather than real replay
inconsistency. The completed parent's joint-physical and classifier gates still
fail. No automatic scientific promotion or change to the posterior's frozen parent.

Local repair verification: 26 pipeline/regression and two coordinate tests pass;
four affected tests rerun after final hardening. Tests cover genuine corruption,
saturated endpoints, saved-latent disagreement, immutable source/report copies,
and the three-job re-audit dependency chain. Ruff, compileall, Bash syntax and
selective archive extraction pass. This is not a new Jean-Zay science result.

## What runs, and what does not

Use the two completed/partially completed branches ending in
`20260927_234702`. This is a new recovery root; neither old run is edited.

1. **CPU parent recovery**, in parallel with step 2. Reuse the expanded parent,
   trained classifier, SFH split and replayed banks. Recompute only the tied
   control and its report. Normalizing split masses within 1e-12 and computing
   a convex interpolation of efficiencies prevents `alpha=1+1.6e-15` without
   accepting genuinely invalid probabilities. All numerical/scientific gates
   and the target catalogue are unchanged.
2. **CPU posterior-tail audit**. Read saved joint draws, reconstruct their exact
   evaluation contexts/IDs, measure full-support tails in theta and latent x,
   replay the largest draws with the same RNG keys, identify their expert/gate,
   and test each expert's inverse and Jacobian at extreme values. CPU/GPU
   replay uses an explicit 0.001 absolute latent tolerance; independent transport
   tolerance is 1e-5. No clipping, draw removal, dimension removal or new DSPS.
3. **One H100 continuation**, automatically after a successful numerical audit.
   Load the source's best validation checkpoint (epoch 181 in the reviewed run),
   explicitly reset AdamW and lower LR to 6.25e-6, halved at 50 additional epochs
   to 3.125e-6. At most 100 additional epochs, early plateau checking, same
   537106 unit-weight simulation training pairs and fixed validation subset.
   Save/evaluate the best-so-far checkpoint every 20 additional epochs on 512
   fixed validation galaxies with 256 joint draws each. Final evaluation repeats
   the original 2048-object/512-draw cohorts with the same IDs and RNG seeds.
4. **CPU comparison report**, also produced as BLOCKED if an upstream stage
   fails. Compare baseline/continued calibration, tails and loss; link to the
   recovered baseline/tied/expanded parent comparison.

The numerical audit **does not require the already-failed tail quality to pass**.
It checks whether the saved extremes are reproducible samples from the trained
density with consistent transport. If they are, further optimization is an
exploratory attempt to improve that density. If sampling/transport is inconsistent,
training is blocked until the implementation fault is understood. No automatic
scientific approval occurs in either case. A `100 IQR` distance is a descriptive
tail counter, not a new physical bound or a publication gate.

The posterior still uses the **old frozen parent**. It is not silently switched
to the new expanded parent. Population fitting stays photometry-only; simulator
truth supervises q, never parent fitting. No q feedback, RWS, MCMC or SMC.

Default requests: 4 CPU threads/task; CPU parent 60 min, audit 30 min and report
15 min; one H100 train/evaluate job capped at 240 min. Peak one H100, allocation
ceiling 4 H100-hours, not an expected runtime. No `--mem*` options.
No new classifier training and zero new DSPS simulations.

Local verification: 36 distinct focused/regression tests pass, including real
small classifier/flow training, numerical failure/NaN reporting, an interrupted
milestone followed by optimizer-state recovery, immutable sources, guarded
small-file download and mocked Slurm dependencies. Ruff, Black, compileall and
shell syntax pass; diagnostic smoke plots inspected. DSPS is mocked in these
local tests; legacy CLI fit configs named in AGENTS are absent. No remote
scientific success is inferred from local smoke tests.

## Launch on Jean-Zay

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
CONDITIONAL="$BASE/avi_conditional_parent_20260927_234702"
POSTERIOR_RUN="$BASE/avi_parent_to_posterior_20260927_234702"
RECOVERY="$BASE/avi_overnight_recovery_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_overnight_recovery.sh \
  "$CONDITIONAL" "$POSTERIOR_RUN" "$RECOVERY"
)
```

The subshell prevents a failed check from exiting the interactive SSH session.
Source completion/provenance checks must pass before submission. The script
refuses active source/recovery jobs and never cancels old jobs.

## Watch and resume

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_overnight_recovery_latest.env"
bash scripts/watch_feniks_overnight_recovery.sh "$RECOVERY"
```

After an SSH disconnect, only rerun the watcher. The jobs are independent of
the terminal. `--once` shows one snapshot. Saved state is not live SLURM state.

For a timeout after inspecting stderr/accounting, with no active recovery jobs:

```bash
bash scripts/submit_feniks_overnight_recovery.sh --resume "$RECOVERY"
```

This uses the frozen code, skips completed branches, and resumes the new
optimizer state. It does not reset again, regenerate banks or retrain the
classifier. Numerical audit FAIL is not automatically bypassed. A code fix
needs a separately recorded recovery, not an edit to a frozen snapshot.

## Artifacts and checklist

- `parent/report/`: baseline/tied/expanded parent, selected and observable closure;
  `DECISION.json` retains failed gates and inherited limitations.
- `audit/{in_model,coherent_target}/tails.csv`: exact full-sample extrema, tail
  frequencies, physical/latent W1; `extreme_draws.csv`: IDs and replayed expert.
- `audit/DECISION.json`: numerical permission to optimize, NOT tail-quality PASS.
- `posterior/`: new optimizer/checkpoints/history, initial validation loss and
  explicit optimizer-reset provenance.
- `milestones/epoch_*/`: fixed validation calibration, tails and best checkpoints.
- `evaluation/`: paired original cohorts, dense 15D draws, calibration and tails.
- `report/calibration_comparison.png`, `tails.png`, `training.png`, numerical CSVs
  and `DECISION.json`; root `ROADMAP_STATUS.md`.

Next decision: determine whether conditional flexibility actually helps the
blind parent and whether continued q improves both central calibration and tails.
Lower NLL alone never validates q. Good in-model calibration alone never validates
the target parent. Coherent-target data remain a development benchmark, not a
new untouched paper test or real-sky validation.

Small results only, from the local machine:

```bash
cd /home/maxime/src/DSPS
bash scripts/rsync_feniks_coherent_results.sh overnight_recovery
```

No banks/checkpoints/NPZ are downloaded; max 25 MiB per included file.
