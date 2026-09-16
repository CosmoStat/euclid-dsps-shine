# Independent final-checkpoint replay

Operator readback, job 1957394: all 16 cases complete, contract PASS; final bad-k
counts 13/16 observed and 15/16 simulated. CPU audit completed on 144 saved
distributions. Weighted coordinates at ESS near one are not training targets.

Next question: how stable are these support diagnostics under new draws?
This is a diagnostic, not a correction or a new optimization objective.

- All 8 observed and 8 simulated development cases, no ranking or filtering.
- Fixed final step-512 checkpoints, both starts, plus unchanged C anchors.
- Two independent K2048 replicates per distribution; pooled K4096.
- New seeds disjoint from long-run evaluation and optimizer ranges.
- Source checkpoints checked against case receipts, hashes pinned at preparation
  and verified on replay; simulated contexts must match exactly.
- No optimizer construction, no training, no teacher or scientific promotion.
- One H100, one node, serial execution, 3h allocation, 250000 evaluations cap.
  Existing contract audit may account for a few gradients, not optimization.
- Existing budget preflight remains fail-closed. Existing roots are refused.

Changing K can expose rarer high weights: a lower ESS/K is not by itself proof
of a worse proposal. Inspect both replicate summaries and Pareto-k, plus paired
ELBO/residuals. Even stable evidence estimates do not prove complete support.
Do not select a start or checkpoint from this development replay.

```bash
cd "$WORK/dsps-popcosmos"
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_long_replay.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/frozen_parent_long_local_vi_v1" \
  "$BASE/frozen_parent_long_replay_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

After Slurm completion, check the new root and 16 cases, then:

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_long_replay.py "$DIAGNOSTIC_ROOT"
```

This reader compares only prespecified final checkpoints, never the best stage.
Raw draws and independent replicate metrics remain in each SUMMARY.json and
direct_draws.npz. No population or subsequent job is submitted automatically.
