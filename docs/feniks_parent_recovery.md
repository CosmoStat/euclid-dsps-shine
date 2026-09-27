# Frozen-classifier parent recovery check

This is one bounded diagnostic with two parallel cases, not a new population
training campaign. It reuses the completed reference-to-parent run and writes
to a new immutable root. No DSPS, neural training, posterior or GPU is requested.

## What is fixed and what is tested

- Known parents: the saved blind parent and the saved physical-capacity control.
  These weights generate diagnostic targets only, never replace the production
  prior. The coherent catalogue is unchanged.
- Frozen network: the exact best classifier checkpoint, selected reference
  frequencies and calibrated offsets. Source hashes are recorded and checked.
- Frozen inversion: source-selected KL penalty, component selection efficiencies,
  weak-support constraint and 2e-6 KKT certificate. There is no penalty sweep or
  new validation-based tuning in this diagnostic.
- Full 15D reference samples: all physical/SFH coordinates remain stochastic in
  the existing bank. No SFH fixing, q samples or per-galaxy inference.
- Fit rows: reserved role 4. Evaluation and ratio-moment rows: role 3. Neither
  was used for classifier training (role 0), checkpoint validation (role 1) or
  offset fitting (role 2). Global bank row IDs are checked for disjointness.

For each known u, the selected pseudo-catalogue is represented by ALL its dense
bank rows with stratified importance weights, not medians or point predictions:

`w_i = u[j_i] / n_parent_role4[j_i]`, normalized AFTER selection.

Using selected counts in the denominator would accidentally remove selection.
Parent counts include rejected objects. The fitted selected weights v become
`u_fit = normalize(v / saved_alpha)`, exactly as in the original inversion.
Finite-bank alpha noise is not silently replaced by true probabilities.
The source alpha estimates used the full bank, including roles 3 and 4. Thus
these roles are disjoint for the diagnostic fit/evaluation and classifier
training, but are not independent of every frozen source parameter. This is
a conditional debugging check, not an untouched final validation cohort.

Three estimates are compared with the generating mixture:

1. Classifier-based inversion with the frozen selected penalty.
2. Label-known penalized inversion: same penalty and constraint, a compressed
   K-row one-hot component likelihood with weighted selected counts.
3. Label-known unpenalized counts followed by the same selection correction.

The label controls isolate sampling/selection bookkeeping and reveal penalty
effects in an upper-information problem. They do NOT provide the exact unknown
photometric likelihood; classifier error, weak identifiability and penalty bias
can still interact. Weight L1 is not an acceptance criterion for overlapping
components. Parent/selected weighted marginal and joint density distances use
independent role 3 and the original physical SW=0.05 and W1/IQR=0.1 limits.

Known/recovered densities are evaluated on common support within role 3 to
avoid unnecessary resampling noise. This is finite-bank quadrature of a
continuous 15D mixture, not an independent high-statistics sky-validation claim.
Effective fit/evaluation support is reported; fewer than 2,000 effective rows
makes the decision support-limited rather than establishing recovery.

## Additional diagnostics without refitting

- Ratio moment: 128 cheap resamples of 64 row-block means, not 128 optimizer
  jobs. Interval conditions on the frozen classifier/calibration; it excludes
  uncertainty from training or offset estimation and cannot certify all ratios.
- Observable prediction against development validation: per-band CDF/tail
  metrics and joint SW across standardized asinh flux projections. The latter
  is descriptive with no invented retrospective PASS threshold.
- r-band tail attribution: each component's probability contribution and excess
  flux contribution, plus descriptive physical/SFH averages in/out of the tail.
  These are associations, not causal interventions. No clipping is applied to
  simulations or inference; upper capping remains a labelled diagnostic metric.

## Launch on Jean-Zay

Run in a subshell so a failed check does not terminate the SSH session:

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$BASE/avi_reference_to_parent_20260927_184817"
RECOVERY="$BASE/avi_parent_recovery_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_parent_recovery.sh "$SOURCE" "$RECOVERY"
)
```

The DAG is CPU cache -> CPU array of two cases -> CPU report (`afterany`).
Eight CPU cores per job, maximum 16 during the array, zero GPUs. Limits are
25 minutes for cache, 25 minutes per case, 5 for report, not runtime predictions;
queue delays are additional. No forbidden `--mem*` options are used.

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_parent_recovery_latest.env"
bash scripts/watch_feniks_parent_recovery.sh "$RECOVERY"
```

Resume after a failed/timed-out job, not merely an SSH disconnection:

```bash
bash scripts/submit_feniks_parent_recovery.sh --resume "$RECOVERY"
```

This refuses active jobs and scheduler-query failures. Completed cache/cases and
per-case fits are retained; the same frozen source snapshot is reused. Source
artifacts and fitted scientific parameters must not be edited to force a retry.
The watcher reports both live SLURM and saved stage state; a failed stage has a
small `FAILED.json` and its normal traceback in `logs/`.

## Results and decision

- `report/DECISION.json`, `report/REPORT.md`, `ROADMAP_STATUS.md`.
- `learned_parent/` and `capacity_control/`: `parent_physical.png`, `joint.csv`,
  `marginals.csv`, `fit/weights.json`, `observable_predictive.csv`,
  `observable_joint.csv`, `tail_components.csv`, `tail_physical_sfh.csv`.
- `cache/ratio_moment.json` is also embedded in `report/DECISION.json` so the
  large cache directory does not need transfer.

On the local workstation, small results only:

```bash
cd /home/maxime/src/DSPS
bash scripts/rsync_feniks_coherent_results.sh parent_recovery
```

If label sampling controls fail, inspect bank/selection uncertainty first. If
label controls pass but photometric inversion fails, investigate ratios,
regularization and identifiability before new simulations. If both cases pass,
focus on why the coherent target differs from the joint reference, including
conditional SFH, bright tails and possible target-domain ratio error. Passing
two in-family cases does not prove that the classifier works for all mixtures.
No outcome automatically authorizes posterior training or paper production.

## Local verification versus remote execution

Tests load a genuine tiny classifier checkpoint with synthetic bank fixtures,
forbid DSPS/training/q calls, verify all 15D evaluation, independent roles, source
immutability, partial-fit resume and mocked SLURM dependencies. They validate
implementation invariants, not recovery for the real saved 256-component run.
No remote scientific result is claimed before this diagnostic is executed.
