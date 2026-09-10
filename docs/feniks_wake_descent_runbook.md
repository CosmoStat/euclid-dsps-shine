# Descent-controlled wake pilot

Contract: `wake_armijo_v1`, conditional transport `conditional_transport_float64_v1`.
Original behavior remains available for the pinned historical pilot.
This experiment uses the original 16 development objects, two starts, seeds and
4096 decoder draws per arm/start. Reverse is unchanged; wake uses 12 Armijo
trials, halving the displacement. No evaluation metric selects the step.
Rejected updates preserve Adam state. Accepted updates commit its proposed
moments and the scaled parameter displacement. Fresh full objective audits
remain mandatory. This does not modify global AVI or population RWS training.

Allocation: one node, one H100, 16 CPUs; walltime 10 hours, internal limit 9 hours.
The previous pilot took about one hour; this version adds density evaluations,
so its runtime must be measured. Decoder-work counters exclude those extra
network-only evaluations; `line_search_evaluations` records them per attempt.

```bash
(
set -e
cd "$WORK/dsps-popcosmos"
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_wake_descent.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_objective_transport64_pilot_v1" \
  "$BASE/frozen_parent_wake_descent_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
)
```

Morning readback:

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_objective_pilot.py "$DIAGNOSTIC_ROOT"
python scripts/summarize_feniks_wake_descent.py "$DIAGNOSTIC_ROOT"
```

Required result: `OBJECTIVE_PILOT_COMPLETE`, 32 objective audits PASS, every
applied wake step descending and satisfying Armijo, plus independent support
and residual tables. A successful optimizer invariant is not posterior
qualification. Do not reuse the old night-extension launcher with this root:
the new adaptation contract requires a separate resource decision.
