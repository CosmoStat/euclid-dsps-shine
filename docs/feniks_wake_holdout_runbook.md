# Independent wake-batch check

The completed guarded pilot passed all 32 numerical checks. All accepted steps
reduced their own batch loss, but 23/32 trajectories remained unchanged and
only three of the nine changed trajectories improved final ESS in the supplied
readback. This is not a reason to extend the same training unchanged.

## What the next test does

Recreate the guarded pilot from the ORIGINAL completed transport64 pilot, with
the same training keys, starting parameters and budgets. After every wake
attempt, score the before/after parameters on two fresh batches of 256 joint
draws. Each batch comes from the pre-update 50/50 local/anchor mixture. Its
samples and exact importance weights remain fixed for both scores.

These batches never select a step, change its scale, or update optimizer state.
Every attempt, including rejection, is retained. ESS>=16 and maximum weight
<=0.2 label a batch informative; this is a diagnostic screen, not proof of
posterior correctness. Poor-weight batches must not be interpreted as reliable
validation. The two replicates are not two independent galaxies.

One node, one H100, 16 CPUs, maximum 10 hours (internal limit 9 hours).
No concurrent jobs are submitted. 512 attempts x 2 x 256 adds 262,144 decoder
forward evaluations. Adaptation draws remain unchanged; total decoder work is
charged to the existing budget. Network-only loss evaluations also take time.
There is no global NN training or automatic overnight extension.

## Launch on Jean-Zay

```bash
(
set -e
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_wake_holdout.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_objective_transport64_pilot_v1" \
  "$BASE/frozen_parent_wake_holdout_v1"
)
cd "$WORK/dsps-popcosmos"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

Use a new output root; never overwrite either completed pilot. The input is
the original pilot, NOT `frozen_parent_wake_descent_v1`: the preparer reapplies
the descent correction and adds the independent measurements.

## Read back

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
set -o pipefail
python -m scripts.summarize_feniks_wake_holdout \
  "$BASE/frozen_parent_wake_holdout_v1" | \
  tee "$BASE/frozen_parent_wake_holdout_v1/holdout_summary.txt"
```

The summary checks completed status, all 32 histories, descent and saved-batch
hashes before printing accepted-update comparisons. Negative loss_delta is
improvement on that batch. Training improvement with repeated independent
worsening suggests the update does not generalize beyond its training batch.
Mostly uninformative batches means this experiment cannot settle that question;
it instead documents the lack of reliable weighted examples. Independent loss
improvement still does not establish posterior coverage or global RWS readiness.

Saved `holdout_ATTEMPT_REPLICA.npz` files contain joint samples, frozen logweights,
before/after loss and weight diagnostics. These are diagnostic data, not a
posterior bank. Historical reverse trajectories need not reproduce bitwise
across environments; conclusions use within-run before/after comparisons.
