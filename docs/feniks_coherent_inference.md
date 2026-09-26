# Parallel coherent inference benchmark

## Decision and scope

The user-reported physical refinement ended with test SW 0.0282 -> 0.0308,
despite lower NLL. Do not spend another serial run on this density oracle.
The new workflow tests population recovery and posterior calibration directly.
It reads the completed coherent catalogue without modifying or rephotometering it.
It does not use a representation/refinement checkpoint as a blind reference prior.

One DAG, no hyperparameter training sweep:

```
reference preparation (CPU)
  |-- supervised catalogue posterior control (1 H100) ------------|
  |-- reference simulation bank (4 tasks, <=4 H100)               |
        -> classifier + population (1 H100)                      |
        -> weighted supervised learned-parent posterior (1 H100)|
                                                                -> report (CPU)
```

The control is explicitly truth-supervised on TRAIN only, never claimed to be
blind recovery. Its held-out calibration answers whether the amortized model
works when the training joint distribution is correct. Population fitting reads
only photometry/errors/masks from selected target train/validation catalogues.
Target test truth is used only for evaluation, never tuning or checkpoint selection.

## Reference distribution

32768 anchors are sampled uniformly without replacement from eligible native
proposals, not according to the magnitude of their galaxy weights. Only the
canonical TRAIN generator-realization partition is used. Every target TRAIN
proposal identity is excluded; target validation/test use other realizations.
Eligibility retains positive-weight native proposals and the saved unclipped
metallicity support. This is a same-generator, disjoint-object reference, not
an externally independent astrophysical population model.

Normalize all 15 coordinates using these REFERENCE anchors only. Redshift,
metallicity and dust slope retain explicitly configured open bounds, not bounds
enlarged from target truth. Dust Av uses log(Av), with support (0, infinity),
because native proposals can exceed the former artificial cap of 6. Mass/SFH
have unbounded asinh coordinates. No target clipping, jitter or zero removal.
The Av inverse is exp(raw), with log-Jacobian raw + log(scale); normalization
is preserved by the coordinate change. Nonpositive Av fails explicitly, rather
than introducing an epsilon floor or silently removing objects. This continuous
positive model does not model an exact Av=0 atom. Old v1 coordinate files retain
their original transform and remain readable without reinterpretation.
`reference/support.csv` records actual anchor ranges before coordinate fitting.

64 joint physical centers are fitted to the reference anchors; Gaussian soft
gates (width 1 in normalized physical coordinates) plus 2% uniform gate mass
give anchor probabilities A_lj, normalized separately for every component j.

```
g_j(x) = sum_l A_lj Normal_15(x; anchor_l, 0.15^2 I)
sum_l A_lj = 1
p_parent(x) = sum_j u_j g_j(x), sum_j u_j = 1
p_parent(theta) = p_parent(T(theta)) |det dT/dtheta|
```

Thus every g_j is normalized, every one of the 15 coordinates varies, and
reference physical-SFH associations are retained through joint anchors.
Gaussian kernels smooth the REFERENCE, not stored truth. SFH atoms remain in
the target catalogue. A continuous model may approximate their mass imperfectly.
This is a finite family with a fixed smoothing scale, NOT a fully flexible
conditional SFH learner or an exact multiplicative tilt r(a) after smoothing.
An inability to close the target parent remains a valid possible result.

## Simulations, selection, inference

Generate exactly 1048576 parent simulations (4 x 262144), balanced across
components. Apply the frozen current projected-15D DSPS decoder, saved Gaussian
m5 errors, flux masks and noisy r<29 cut. Invalid draws stop the job; they are
not silently rejected and renormalized. Every 8192-draw block has a receipt and
hash, so a timeout resumes from missing blocks, not the whole arm.

Assign independent random row roles BEFORE selection: 70% training, 10%
validation, 5% classifier intercept calibration, 5% classifier audit, 10%
posterior/predictive audit. Selected class counts must be nonzero in every role.
The classifier is a 3-hidden-layer width-256 GELU MLP on flux/error/mask features
(54 inputs for 18 bands). Feature scales use target TRAIN observed values only.

Class intercept offsets are fitted on the separate calibration role. With c
equal to selected calibration frequencies, component ratios are C_j(x)/c_j.
Validation NLL is for checkpointing; independent audit NLL and ratio moments
remain diagnostic checks rather than assumed classifier correctness.

The existing convex simplex solver fits v using selected target TRAIN observations,
with KKT tolerance 2e-6 and KL regularization. Four cheap convex fits, not four
neural trainings, use penalties 0.003, 0.01, 0.03, 0.1. Choose the strongest
within one paired standard error of the best VALIDATION observable likelihood.
Target truth, SW and test data do not choose this penalty.

```
alpha_j = selected reference attempts_j / all reference attempts_j
u_j = (v_j / alpha_j) / sum_l(v_l / alpha_l)
```

Wilson lower bounds and selected counts identify weak support. Existing linear
constraints cap weak-component parent mass at 5%; unresolved zero alpha/classes
fail closed, not through an arbitrary alpha floor. Both u and v are exported.

Both posteriors reuse the existing continuous full-15D two-expert RQ-spline
architecture: 12 layers, width 256, 16 bins, float64 transport, residual
photometry trunk (512 wide, 3 blocks). The control trains on existing selected
catalogue simulation pairs. The final posterior uses the SAME reference bank,
with the joint-label training weight u_j/r_j, where r_j=1/64 is the reference
PARENT sampling probability. This importance-weights simulator truth pairs to
the selected learned-parent joint distribution; there is no 1/beta correction
in individual posteriors, RWS, or feedback from q to population fitting.
It is a finite weighted-bank approximation, not a newly simulated iid final bank.
Parent weights are plug-in empirical-Bayes estimates; this run does not propagate
their sampling uncertainty into individual posteriors. Calibration must be
assessed empirically, not assumed from the training objective. Each trained flow
also undergoes a small inverse/log-Jacobian consistency check before evaluation.

Training checks validation improvement every 20 epochs: at least 80 epochs,
three blocks with best-NLL gain below 0.001 for an operational plateau. Maximum
600 classifier / 200 posterior epochs are explicit finite budgets, not proof
of convergence. Optimizer state resumes every epoch; best-NLL model is retained.

## Evidence to inspect

- `report/training.png`: plateau versus maximum budget, not just DONE.
- `population/weights.csv`: selected v, reconstructed parent u, alpha, support counts.
- `population/classifier_audit.json`: independent reference density-ratio audit.
- `report/parent_selected_reference_15d.png`, `population_joint.csv`: population closure,
  physical/SFH separately, and finite-catalogue validation/test baseline.
- `report/parent_physical_corner.png`: joint physical correlations and reference support.
- `report/coverage_comparison.png`, `pit_comparison.png`: both networks on the SAME
  1024 held-out selected target galaxies with 512 joint draws each.
- `report/in_model_calibration.csv`: final q on a reserved reference-bank sample
  importance-resampled to the learned-parent selected joint. Unique counts are
  recorded. Failure here indicates amortization error even within the fitted model.
- `report/observable_predictive.csv`: selected photometry predictive W1/IQR.
- `ROADMAP_STATUS.md`: operational progress, explicit remaining scientific limits.

Keep the dense posterior NPZs on Jean-Zay. The small-result download excludes
them, model weights and simulation banks. No report automatically grants
production promotion or validates real-sky catalogue inference.

## Launch on Jean-Zay

Use a subshell: failures must not close the SSH login shell. Do not compare a
full commit hash with an abbreviated one. Update the existing checkout, no clone.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
COHERENT="$BASE/avi_coherent_parent_20260925_224312"
TAG=$(date +%Y%m%d_%H%M%S)
INFERENCE="$BASE/avi_coherent_inference_$TAG"
printf 'export INFERENCE=%q\n' "$INFERENCE" > "$BASE/avi_coherent_inference_latest.env"
bash scripts/submit_feniks_coherent_inference.sh "$COHERENT" "$INFERENCE" \
  configs/experiments/feniks_coherent_inference.yaml |
  tee "$BASE/avi_coherent_inference_submission_$TAG.txt"
)
```

Peak 5 H100s (4 bank tasks plus control), then at most 2 concurrent training jobs.
Allocated GPU ceiling 20 H100-hours: bank 4x120min, control 240min, classifier
180min, final NPE 300min. These are conservative TIME LIMITS, not predicted
runtimes. CPU preparation/report capped at 60/30min. No Jean-Zay `--mem` flags.

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_coherent_inference_latest.env"
bash scripts/watch_feniks_coherent_inference.sh "$INFERENCE"
```

After jobs have stopped, resume ONLY missing stages/blocks using the frozen code:

```bash
bash scripts/submit_feniks_coherent_inference.sh --resume "$INFERENCE"
```

The launcher refuses duplicate submission while recorded jobs are active. A
failed prerequisite cancels its afterok dependents; the afterany report explains
incomplete stages. Corrupt completed artifacts fail rather than being overwritten.

### Recovery of reference job 217287

This failed at `fit_coordinates`: the original reference config restricted
Av to (1e-6, 6). The local saved native TRAIN proposals contain 35 eligible
rows above 6 among 1345886 eligible rows, with maximum 7.5906 and minimum
6.17e-5. These local counts are not claimed to be the remote sampled-anchor
counts. The traceback identifies the same coordinate contract violation.

No reference bank, classifier or posterior was completed. The bounded-config
smoke fixture failed to exercise this legitimate native dust tail. Tests now
cover Av above 6, positive-coordinate inverses/Jacobians/normalization and the
actual reference-to-bank path, as well as safe frozen-code recovery.

Do NOT merely pull and use `--resume`: it deliberately reuses the old frozen
snapshot AND the old prepared settings. After all recorded jobs have stopped:

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
INFERENCE="$BASE/avi_coherent_inference_20260926_094557"
bash scripts/repair_feniks_coherent_inference.sh "$INFERENCE"
)
```

The repair keeps the SAME root, archives original configuration, code pointers,
job records and partial reference/blocked report under `recovery/`, freezes a
new code snapshot, records `REFERENCE_REPAIR.json`, and resubmits the DAG. It
changes only Av support, leaving seeds, counts, assets, datasets and all other
scientific settings untouched. It refuses any completed reference or downstream
bank/training artifact, any active job, or a failed scheduler query. The original
code archive is retained. An interrupted repair fails closed for inspection.
If repair succeeded but submission was interrupted, use ordinary `--resume`
after any submitted jobs stop; do not apply the support repair twice.

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
INFERENCE="$BASE/avi_coherent_inference_20260926_094557"
bash scripts/watch_feniks_coherent_inference.sh "$INFERENCE"
```

This fixes a preparation bug, not a demonstrated scientific failure or validation
of parent recovery. The next report must still establish population closure and
both in-model and target-catalogue posterior calibration.

## Download locally

After pulling this branch locally:

```bash
cd /home/maxime/src/DSPS
bash scripts/rsync_feniks_coherent_results.sh refinement
# Later, for the new experiment:
bash scripts/rsync_feniks_coherent_results.sh inference
```

The helper validates the exact remote path before rsync, never copies `/`, writes
nothing on the SSH relay, and only permits figures/tables/reports/configs/logs
up to 25 MiB per file. It excludes `.eqx`, `.npz`, `.npy`, parquet and banks.
It does not delete any previously downloaded files.

## Local verification

Support-repair verification: 29 focused tests pass, including positive-coordinate
normalization/Jacobians, old v1 roundtrips, Av>6 reference/bank generation, full
tiny training/report runs with both coordinate versions, and same-root frozen
code repair/resubmission. Active jobs and scheduler failures block repair.
Ruff, compileall, Bash syntax and diff checks pass. Native Av ranges were read
directly from saved parquet proposals. An additional native-SFH projection smoke
could not run in the current .venv because its optional Diffmah dependency is
absent; Jean-Zay had already completed projection before the support error.

Previous initial implementation checks:

27 focused and existing population tests pass, including full two-epoch NPE /
population / report execution on a toy observation model, trained flow transport,
classifier learning, optimizer interruption, block recovery, source immutability,
normalization/selection identities, SLURM dependencies and rsync path guards.
An additional actual native-projection -> joint-reference-kernel -> DSPS smoke
produced finite 18-band fluxes for eight draws from 32 historical native anchors.
These catch integration errors, not scientific convergence or H100 timing.
The full repository suite was not run; an unrelated pre-existing local edit in
`tests/test_audit_feniks_coherent_parent.py` currently has invalid syntax and was
left untouched. Production-size reference support and calibration remain remote
experimental outcomes, not guarantees from the small tests.
