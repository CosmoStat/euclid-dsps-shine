# FENIKS transport precision localization

## Evidence and purpose

Operator logs for job 1960443 report 31/32 native audits PASS and one
INCONCLUSIVE: simulated_004, start 1, log_std direction 0. No optimizer ran.
The likelihood FD approaches AD near h=0.0025 and 0.00125, then becomes noisier.
This is compatible with a resolution problem, but does not establish its cause.
The prior and density checks pass; the likelihood term is the unresolved part.

Parameter promotion alone does not remove explicit float32 casts in the
conditional coupling transport and rational-quadratic spline inputs. This new
diagnostic compares historical native arithmetic with a diagnostic float64
transport. Production defaults and serialized checkpoint structures stay intact.

All 32 starts are replayed, without selecting a favorable step or case. The
original fixed noise values, seeds and parameter directions are retained.
The target, observations and context are frozen. This is NOT a full-float64
target: internal prior/decoder arithmetic is not changed. Three-step plateau
requirements and tolerances are unchanged. Even all-PASS results cannot start
an optimizer or override the original audit. There is no scientific promotion.

## Launch on Jean-Zay

Update the feature branch first. Execute from `$WORK/dsps-popcosmos` with `shine`
active, `REPO_DIR="$PWD"`, and `CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"`.
Use a new output directory; keep the failed pilot as immutable evidence.

```bash
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_transport_precision.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/frozen_parent_long_local_vi_v1" \
  "$BASE/frozen_parent_objective_pilot_v1" \
  "$BASE/frozen_parent_transport_precision_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

One Slurm job, one node, one H100 GPU, serial cases (peak allocation: one GPU).
The inherited limits are three hours Slurm, 9900 seconds diagnostic budget,
1,200,000 counted evaluations. Local CPU tests do not predict H100 wall time.

## Readback and decision

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_objective_pilot.py "$DIAGNOSTIC_ROOT"
```

Expected execution receipt: `TRANSPORT_PRECISION_DIAGNOSTIC_COMPLETE`,
`optimization_started=false`, `cases_complete=0`, `audits_complete=32`.
Completion is separate from the native and transport64 PASS/FAIL/INCONCLUSIVE
statuses. The monitor may still show 0/8 optimization cases; that is intentional.

Each `cases/<case>/audit_start_<start>/` contains native `AUDIT.json` and
`stencils.csv`, plus `TRANSPORT_PRECISION.json`, `transport64_stencils.csv`,
`transport_trace.csv`, and `transport_values.npz`. The archive preserves actual
joint latent draws, fluxes, logq and loglike at centers and both sides of every
log_std stencil. Keys identify variant, direction, step index and side; step
indices correspond to the audit JSON step list. Hashes pin all evidence.

If native replay reproduces the issue and transport64 resolves it, investigate
a versioned precision contract before any optimizer rerun. If both remain
inconclusive, inspect per-draw flux/likelihood stencils for target resolution or
branch crossings. If a stable FD disagrees with AD, investigate that derivative.
Do not relax tolerances or pick the closest FD to AD to obtain a PASS.
