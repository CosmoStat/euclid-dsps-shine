# Controlled forward-population diagnostics

This campaign diagnoses the completed forward run without overwriting it or
generating another DSPS bank. Completion is not a convergence certificate.

## Independent comparisons

1. **Baseline:** reevaluate the original best posterior on every selected
   validation object (split bucket 7000--8499), not the first 512. Preparation
   refuses truncation. The saved cohort has 6,903 objects. Report simulation
   validation also grows to at most 8,192 independent evaluation-bank objects.
2. **Classifier:** start from its best checkpoint, reset optimizer, run 180
   additional epochs at a decaying learning rate and fixed validation subset.
   Refit selected weights and selection-corrected parent weights with the
   existing KKT gate. Save independent test ratio moments before/after.
3. **Posterior:** start from its best checkpoint, reset optimizer, run 150
   additional supervised epochs. Keep the ORIGINAL parent and simulation bank
   fixed. Reevaluate on the identical cohort as baseline. This isolates the
   inference-network training error from changes in the population model.
4. **Analytic capacity:** train the same four-expert 15D flow at constant context
   on 262,144 fresh analytic samples from the known learned parent. Validate on
   32,768 independent samples and test on 32,768 more. Report
   `KL(p_parent || q) = E_parent[log p_parent - log q]` with Monte Carlo error.
   Both log densities use latent-x coordinates, so the Jacobians cancel.
5. **Truth capacity:** train the same constant-context flow on empirical true
   parent parameters, split 60/20/20 by object. Resample 262,144 training pairs
   only within the training partition; no validation/test leakage. Report
   held-out NLL, physical/SFH marginal W1 and joint sliced W1. This is a
   diagnostic oracle, never a production prior or posterior-training target.

Both capacity arms use 120 epochs and all 15 dimensions. They establish whether
this optimizer/architecture can represent the tested marginal density; they
do not prove it can learn all conditional posteriors. Failure alone does not
distinguish insufficient optimization from insufficient expressivity. Inspect
loss curves and held-out distributions together. Truth outside flow support
causes an explicit error, never silent clipping.

All comparisons use uniform catalogue-object weighting, matching the original
population objective. Historical population-weighted plots are a different
target and must not be overlaid without explicit labels.

The classifier's candidate parent is NOT passed to the posterior continuation.
If classifier refinement helps, a subsequent production run must generate
posterior simulations consistent with that new parent. No q-to-parent feedback
is introduced. No NUTS/SMC is involved.

## Training safeguards

Validation positions are fixed and recorded; the starting checkpoint competes
with later checkpoints on this same subset. LR is halved every 30 epochs.
Best-checkpoint per-object validation losses and tail quantiles are saved.
The epoch counts are additional budgets, not promises of convergence. Existing
training defaults and historical pipelines remain available.

## Launch on Jean-Zay

From the updated checkout and activated `shine` environment:

```bash
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
PARENT="$BASE/avi_forward_population_r29_20260920_162915"
DIAG="$BASE/avi_forward_diagnostics_$(date +%Y%m%d_%H%M%S)"
printf 'export DIAG=%q\n' "$DIAG" > "$BASE/avi_forward_diagnostics_latest.env"
bash scripts/submit_feniks_forward_diagnostics.sh "$PARENT" "$DIAG"
bash scripts/watch_feniks_forward_diagnostics.sh "$DIAG"
```

The launcher freezes a code snapshot and hashes checkpoints/cohort inputs.
Five independent one-H100 tasks run at most concurrently (capacity is a
two-element array). A dependent summary job follows. Default allocated-time
ceiling: 83 H100-hours, not an expected consumption estimate. No new DSPS
simulations. Original banks are linked read-only by convention; do not move or
modify the parent run while diagnostics are running.

## Read the results

- `training_comparison.png`: fixed-validation learning curves and starting NLL.
- `calibration_comparison.csv`: before/after coverage, widths and bias, separated
  into observed and held-out simulation cohorts.
- `baseline/report/`, `posterior/report/`: full joint draws, PIT plots, individual
  corners, aggregate distributions and population comparisons.
- `classifier/COMPARISON.json`: reference ratio moment departures from one.
- `classifier/population/component_weights.csv`, `parent.json`, `FINAL.json`:
  candidate selected/parent weights, alpha, KKT convergence diagnostics.
- `capacity_*/report/capacity_15d.png`, `capacity_marginal.csv`,
  `capacity_joint.csv`, `FINAL.json`: pure density-capacity diagnostics.
- `RESULTS.json`, `REPORT.md`: collected results and provenance guide.

A larger cohort tests the previous ordering/finite-sample concern. Better
simulation calibration at fixed parent implicates q training. Better classifier
test ratios and a changed parent implicate population ratio estimation. Neither
ESS nor an aggregate match alone certifies parent recovery. Parent/selected
distributions remain distinct: aggregation over selected data targets the
model-selected prior only under the corresponding model data distribution.
