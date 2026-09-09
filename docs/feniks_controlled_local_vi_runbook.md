# Controlled local VI trajectories

## Evidence and question

User-supplied job 1938818 completed in 7m20s. The numerical contract passed,
but 15 of 16 observed local fits decreased ESS/K. Observed median ESS/K was
0.019134 amortized, 0.004227 local start 0 and 0.005909 local start 1.
The corresponding simulated medians were 0.009241, 0.007065 and 0.005179.
Lower residual RMS and improved negative ELBO did not recover support.
Observed case 5 had nonfinite Pareto-k in both local fits. A finite fraction
column named nonfinite_k does not mean the underlying k values are finite.

The next experiment separates learning-rate sensitivity from gradient-draw
count sensitivity. It does not establish family inadequacy, mode completeness
or catalogue compatibility, and is not a new global NPE run.

## Fixed protocol

- Same qualified night C, same eight observed contexts and eight fresh
  deterministic model-generated cases as the previous recipe (seed 260910).
- Three regimes: original Adam 0.001 / 4 draws, Adam 0.0001 / 4 draws,
  Adam 0.0001 / 16 draws. All use the existing global gradient clip of 5.
- Two matched initializations per regime, 64 updates each. The two starts
  remain nearby and are not a multimodality search.
- Save distributions at updates 8, 16, 32 and 64. Each has two independent
  K128 direct-draw replicates, pooled K256. Evaluation random keys are shared
  across steps/regimes for each initialization, independent of training keys.
  Different gradient draw shapes need not yield nested random samples.
- Persist every optimization row (loss decomposition, gradient and update
  norms), checkpoint, direct draws, latent covariance/std and density means.
  No evaluation metric enters optimization or checkpoint selection.
- One H100, one node, sequential regimes/cases; three-hour allocation ceiling,
  9900-second internal budget and 180000 charged decoder evaluations.
  Gradient evaluations include backward work, not forward-equivalent cost.
  First observed/simulated pair provides the actual cost preflight.
- All source checks, frozen prior/calibration and observed-only contracts stay
  active. Historical receipts are untouched. No population promotion.

## Launch on Jean-Zay

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_controlled_local_vi.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/frozen_parent_controlled_local_vi_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

The launcher refuses dirty tracked source and an existing output directory.
It submits an immutable Git snapshot. Do not resubmit after ambiguous sbatch
output; inspect squeue and the submission receipts first.

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_controlled_local_vi.py "$DIAGNOSTIC_ROOT"
```

This CPU-only readback is usable during execution. Inspect each case and
initialization, not only pooled medians. If slow updates preserve support
while the original loses it, optimization speed is implicated. If MC16 helps
relative to slow/4, sample-count sensitivity is implicated, not a measured
proof of gradient variance. If all improve ELBO but lose support, investigate
the objective/distribution geometry before extending training. If all remain
unchanged, 64 slower steps may simply be insufficient. None of these outcomes
alone authorizes population training or selecting a teacher.
