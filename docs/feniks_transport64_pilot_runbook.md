# Versioned transport64 reverse/wake pilot

## State after job 1961888

Operator output: 32/32 transport64 audits PASS. Native replay reproduces 31
PASS and one INCONCLUSIVE. This supports a transport-resolution explanation
for that unresolved stencil, not posterior accuracy or population readiness.

The new contract `conditional_transport_float64_v1` uses the previously tested
conditional transport for the entire local pilot: forward sampling, inverse
logq, fresh objective audits, reverse updates, wake mixture draws and wake
updates, checkpoint readback and independent evaluation. Mean, log_std and
coupling parameters are promoted without changing their initial values.
Native source checkpoints stay immutable; context and posterior target remain
frozen. This is not a full64 prior/decoder contract.

Preparation requires the completed precision diagnostic and all 32 hashed
transport64 PASS reports. The same qualification is checked at runtime.
Fresh audits on all prescribed starts must pass before either optimizer is
constructed. Old native audits are neither overwritten nor relabeled PASS.

## Run

From the updated feature branch in `$WORK/dsps-popcosmos`, activate `shine` and
set `REPO_DIR="$PWD"`, `CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"`.

```bash
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_transport64_pilot.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/frozen_parent_long_local_vi_v1" \
  "$BASE/frozen_parent_transport_precision_v1" \
  "$BASE/frozen_parent_objective_transport64_pilot_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

One job, one node, one H100, serial cases; peak allocation one GPU. Slurm limit
3 hours, diagnostic limit 9900 seconds and 1,200,000 counted evaluations.
No global NPE, population training, automatic continuation or promotion.

The prescribed comparison uses eight observed and eight simulated objects,
two starts, reverse batches of 32 and wake batches of 256, checkpoints at
1024 and 4096 decoder draws per arm/start. Counts match decoder draws, not
FLOPs or applied updates. Wake uses fresh 50/50 local/anchor joint-mixture
draws and its full mixture density; samples and weights are stopped for the
weighted density objective. Rejected batches (ESS <16 or max weight >0.2)
consume budget but preserve parameters and optimizer state. They are never
retried until acceptance. No pointwise substitute is introduced.

## Readback

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_objective_pilot.py "$DIAGNOSTIC_ROOT"
```

Expected successful execution: `OBJECTIVE_PILOT_COMPLETE`, 16 completed cases,
contract `conditional_transport_float64_v1`. An inconclusive fresh audit still
blocks all updates. Each saved parameter checkpoint has a hashed
`TRANSPORT_CONTRACT.json`; do not load it with the historical native encoder.
CPU summary checks these contracts. Evaluations use two independent replicates,
K1024 total at intermediate checkpoints and K4096 at final checkpoints.

Before a larger overnight run, inspect applied wake updates versus rejected
attempts, paired ESS/weights/Pareto diagnostics, independent evidence stability,
and photometric residuals for all cases and both starts. A completed pilot is
not a support gate. If wake is mostly rejected, simply extending its budget is
not justified. If support improves consistently, the next useful expansion is
fixed-parent wake on an independent cohort with a predeclared budget and
validation protocol, not immediate population RWS. No overnight job is chained
automatically from this pilot.
