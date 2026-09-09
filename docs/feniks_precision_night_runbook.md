# Precision-corrected fixed-parent overnight pilot

Date: 2026-09-09. Prepared implementation, not an executed Jean-Zay result.

## What changed

Job 1922455 isolated a numerically stable z-dependent path. This implementation
ports the arithmetic into `run_spline15d_model_jax`, the nonlinear latent
transform and the canonical likelihood, behind explicit version flags:

```yaml
model:
  photometry_integrator: merged_gauss4_v1
  mdf_weight_precision: float64_v1
  spline_precision: float64_v1
amortized:
  latent:
    arithmetic_precision: float64_v1
  likelihood:
    arithmetic_precision: float64_v1
```

Legacy defaults remain unchanged. The new mode requires spline15d, no AGN,
fixed-scatter MDF and JAX x64. Dust, IGM, stellar-mass and SFH conventions remain
the same. Their arithmetic is promoted; no noise inflation, new bounds or truth
calibration is introduced. Static SSP assets retain their stored values.
The parent and encoder arrays are imported explicitly and checked after a new
checkpoint save/reload. The new sidecar records the numerical contract and
rejects loading under a different contract. Old checkpoints are untouched.

## Sequential Stages

1. Verify the completed same-source redshift-precision receipt and immutable
   source artifacts. Replay six prior/q parameter points and observed contexts.
2. Qualify the integrated candidate in all 15 latent coordinates: per-band FD
   plateaus, canonical and centered likelihoods, prior derivatives and forward/
   reverse gradient identities. Existing tolerances are retained. All six points
   must pass; a missing, FAIL or INCONCLUSIVE check stops before new training.
3. Migrate identical model arrays to a new versioned checkpoint. Generate a
   separate 1024-point smoke bank and perform one epoch on 128 training and 32
   validation rows with sleep plus observed ELBO. Require finite applied updates,
   nonzero q gradients, zero masked prior gradients and bitwise frozen arrays.
4. B: four pure-sleep epochs on 8192 uniformly sampled training rows, starting
   from the migrated source B checkpoint. Generate a NEW 16384-point noiseless
   DSPS bank; refresh noise and conditioning masks. Do not initialize from smoke.
5. C: four further epochs from B, sleep plus observed ELBO (lambda=0.001,
   four reparameterized q draws/observation). Prior and calibration remain frozen.
6. A/B/C: matched 64-object K256 diagnostics, then 64-object internal simulated
   and held-out-band checks with 64 draws. A means unchanged source-B weights
   under the NEW decoder, not the historical arm A of older experiments.

The validation subset for checkpoint selection excludes the monitoring cohort.
All three arms use the same monitoring objects and random seeds. These are
previously examined observed validation rows, not a fresh independent final test.
The simulated parameters come from the model, never from catalogue truth columns.

## Budget

- One node, **one H100**, 16 CPU threads. All stages serialized. Peak one GPU.
- Slurm ceiling **10 GPU-hours**; internal subprocess deadline 9.5 hours after
  subtracting qualification time. Qualification itself has an 80-minute /6000
  component-evaluation ceiling. No automatic retry or array fan-out.
- JAX training and cache-generation batch size 8; inference objects/batch 2,
  decoder sample chunk 1. No pmap. No changes to Slurm memory directives.
- About 131072 observed-ELBO decoder evaluations for C, plus smoke, cache,
  generated validation and diagnostics. Actual counters and elapsed time are
  written by each training phase; backward calls are not forward-equivalent cost.
- The ceiling is not a runtime prediction. Compilation and H100 timing of the
  integrated candidate are not yet measured remotely. Timeout preserves partial
  outputs and prevents further stages.

## Launch on Jean-Zay

Use a clean tracked checkout containing this patch. Do not modify snapshots or
old receipts. The last argument MUST be a new root.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git checkout feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="$SCRATCH/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_precision_night.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_redshift_precision_v1" \
  "$BASE/frozen_parent_precision_night_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

If `$SCRATCH` does not match the existing fsn1 project root, set BASE explicitly
to `/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111`.
Submission creates an immutable source worktree. Closing the terminal or stopping
the monitor does not cancel the Slurm job.

## Read Back

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
sacct -X -j "$DIAGNOSTIC_JOB" --format=JobID,State,Elapsed,Timelimit,ExitCode
tail -n 100 "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.err"
python - "$DIAGNOSTIC_ROOT" <<'PY'
import json, hashlib, sys
from pathlib import Path
r = Path(sys.argv[1])
for name in ('FAILED.json', 'NIGHT_PROGRESS.json', 'NIGHT_FINAL.json'):
    p = r/name
    if p.exists():
        x = json.loads(p.read_text())
        print(name, {k:v for k,v in x.items() if k != 'summaries'})
p = r/'NIGHT_ARTIFACTS.json'
if p.exists():
    for name, expected in json.loads(p.read_text()).items():
        assert hashlib.sha256((r/name).read_bytes()).hexdigest() == expected, name
    print('Night artifact hashes: PASS')
for arm in 'ABC':
    p = r/'validation'/arm/'tracking_k256/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json'
    if p.exists():
        x = json.loads(p.read_text())
        print(arm, x['technical_gate'], x['support'])
PY
```

`FINAL.json` covers numerical qualification only. `NIGHT_FINAL.json` distinguishes
`BLOCKED_NUMERICAL_QUALIFICATION` from `PRECISION_NIGHT_DIAGNOSTIC_COMPLETE`.
An exception writes `FAILED.json`; its latest stage is in `NIGHT_PROGRESS.json`.
Do not interpret a qualification FINAL as completion of training.

## Decision After the Night

This is a bounded retraining experiment, not a production population run.
Numerical success does not prove a good posterior. Review raw ESS/K, finite and
nonfinite Pareto tails separately, maximum weights, integration stability,
robust q-direct residuals, simulated coverage and held-out predictions for each
arm. Check catastrophic cases, not only averages. A better ELBO or sleep NLL
cannot select a scientific winner by itself.

Catalogue simulator compatibility is still NOT VERIFIED. All models are tested
under the explicitly changed decoder; new simulated validation is internally
matched to that decoder. The experiment cannot certify historical catalogue
generation conventions or population identifiability. No population learning,
MCMC, SMC, AIS, nested sampling or automatic scientific promotion follows.
