# Fixed long local VI diagnostic

## Evidence as of 2026-09-09

Operator-provided support probe 1952467 completed with no optimization.
All 64 observed proposals and 63/64 simulated proposals had bad Pareto-k.
Uniform base broadening and the tested mixture did not recover general support.
The concentration audit of controlled-run start_4 has ESS 1.000003,
1.155043 and 1.151040 for observed_005, simulated_003 and simulated_004.
Their weighted-minus-unweighted mean log likelihood is about 401, 8.55 and
824 respectively. Weighted coordinates at ESS near one are not posterior
estimates or training targets. These are operator transcriptions, not a local
readback of Jean-Zay artifacts.

## Question and protocol

Does a longer, lower-noise local optimization improve support, or does it
plateau despite improved ELBO? This experiment cannot prove missing modes absent.

- Start again from the frozen precision-night arm C, not a selected local fit.
- Same deterministic 8 observed and 8 simulated development cases; seed 260910.
- Two starts, one fixed regime: Adam 1e-4, MC32, 512 updates, existing clipping.
- Save/evaluate steps 64, 128, 256, 512. No early/best checkpoint selection.
- Each evaluation uses two independent K512 replicates (pooled K1024).
  Evaluation draws are independent of optimizer draws and common across steps.
  Separate starts and cases have disjoint optimizer seed ranges.
- Prior, decoder, calibration, feature statistics remain frozen. No truth tuning.
- One H100 on one node, serial cases, maximum 3 hours / 900000 decoder evaluations.
  Existing first-two-case cost preflight and fail-closed numerical audit remain.
- No automatic subsequent job, teacher, NPE or population promotion.

MC32 and duration change together relative to the previous MC16 pilot; this is
not a clean duration-only causal comparison. The trajectory within this fixed
regime addresses progression versus plateau. Larger K alone can lower ESS/K
by discovering rare large weights; compare replicate stability and Pareto-k too.

## Launch from the updated checkout

```bash
cd "$WORK/dsps-popcosmos"
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_long_local_vi.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/frozen_parent_long_local_vi_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

After completion, require the new root, Slurm exit zero, contract PASS, all
16 cases and FINAL.json. Completion is not a scientific pass.

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_controlled_local_vi.py "$DIAGNOSTIC_ROOT"
python scripts/audit_feniks_saved_draws.py "$DIAGNOSTIC_ROOT" \
  --out "${DIAGNOSTIC_ROOT}_concentration"
```

Keep all checkpoints and failed cases. If support remains poor, do not extend
the run automatically or choose the single best ESS checkpoint.
