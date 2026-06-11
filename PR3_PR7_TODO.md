# PR 3-7 TODO

This file tracks the remaining Diffsky HLTDS physical-validation PRs after:

- PR 1: dataset integrity and truth semantics.
- PR 2: supervised RealNVP prior learning from truth parameters.

Keep these PRs separate. Do not mix truth-prior learning, same-parameter
forward closure, and photometric posterior inference in one objective.

## Prompt Alignment

Scientific objective: validate a differentiable generative galaxy-population
model on Diffsky/HLTDS with photometry and physical ground truths.

The remaining work must preserve these three distinct levels:

- A. supervised prior learning on `theta_true` - already started in PR 2.
- B. same-parameter forward closure `theta_true -> photometry` - PR 3.
- C. photometric amortized inference `q(theta | flux)` - PR 4 and later.

Do not interpret a good photometric fit as physical recovery. Physical claims
require:

- same-parameter forward closure;
- supervised prior-vs-truth diagnostics;
- posterior calibration;
- comparison of derived quantities, not only raw latent parameters.

Dependency gates:

- PR 3 is the gatekeeper before physical interpretation of amortized results.
- PR 4 can implement supervised-prior loading, but scientific claims from it
  should be conditional on PR 3 closure quality.
- PR 5 redshift ablation needs outputs from PR 4 plus fixed-z/closure runs.
- PR 6 population realism diagnostics need PR 2 prior samples and PR 4
  posterior samples.
- PR 7 should clean docs/configs after PR 3-6 behavior and command names are
  stable.

## PR 3 - Same-Parameter Diffsky Forward Closure

Goal: test whether exported Diffsky truth parameters can reproduce HLTDS
photometry through the configured forward model. This is a simulator gatekeeper,
not an optimizer benchmark.

Implementation:

- Add CLI command `diffsky-forward-closure`.
- Add config `configs/diffsky_hltds_04_14_trueparam_closure_gpu.yaml`.
- Use `model.sfh_model: diffsky_basic`, not `popcosmos_bins`.
- Build forward-model parameters directly from truth columns:
  - `z_obs <- redshift_true`
  - `log10_stellar_mass <- logsm_true`
  - `diffstar_lgmcrit <- diffstar_lgmcrit`
  - `diffstar_lgy_at_mcrit <- diffstar_lgy_at_mcrit`
  - `diffstar_indx_lo <- diffstar_indx_lo`
  - `diffstar_indx_hi <- diffstar_indx_hi`
  - `diffstar_lg_qt <- diffstar_lg_qt`
  - `diffstar_qlglgdt <- diffstar_qlglgdt`
  - `diffstar_lg_drop <- diffstar_lg_drop`
  - `diffstar_lg_rejuv <- diffstar_lg_rejuv`
  - `diffmah_logm0 <- diffmah_logm0`
  - `diffmah_logtc <- diffmah_logtc`
  - `diffmah_early_index <- diffmah_early_index`
  - `diffmah_late_index <- diffmah_late_index`
  - `diffmah_t_peak <- diffmah_t_peak`
  - `dust_av <- dust_av or dust_av_true`
  - `dust_delta <- dust_delta`
- If stellar metallicity is unavailable, use fixed
  `log10_stellar_metallicity` and record it as `nuisance_fixed`, not truth.
- If Diffstar/Diffmah required columns are missing, fail clearly unless config
  says `allow_partial_truth: true`.
- Add a small internal decoder/parameter-builder module if needed, but keep
  DSPS-specific execution in or near `model.py` boundaries.

Example command:

```bash
python -m euclid_dsps.cli \
  --config configs/diffsky_hltds_04_14_trueparam_closure_gpu.yaml \
  diffsky-forward-closure \
  --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
  --limit 1024 \
  --out outputs/runs/diffsky_trueparam_forward_closure
```

Outputs:

- `forward_closure_photometry.parquet`
- `forward_closure_residuals_by_band.csv`
- `forward_closure_summary.json`
- `forward_closure_report.md`
- Optional plots for examples, redshift bins, mass bins, and color residuals.

Metrics:

- Residual magnitude by band.
- Median residual by band.
- RMS residual by band.
- Color residuals.
- Residuals by redshift bin.
- Residuals by stellar-mass bin.

Tests:

- Add `tests/test_diffsky_trueparam_closure.py`.
- Mock decoder closure gives zero residual.
- Missing Diffstar columns raise a clear error.
- Fixed nuisance metallicity is recorded in summary/report.

Acceptance:

- Command runs on a small HLTDS prepared parquet.
- Config uses `model.sfh_model: diffsky_basic`, not `popcosmos_bins`.
- Report states whether same-param forward closure is good enough to interpret
  later posterior results physically.

## PR 4 - Use Supervised Prior In Photometric Inference

Goal: let amortized photometric inference train an encoder under a chosen prior
source without always joint-training the RealNVP prior.

Implementation:

- Extend amortized prior config:
  - `source: standard_normal | supervised_checkpoint | joint_realnvp`
  - `checkpoint: <path>` for supervised checkpoint.
  - `train_jointly: false | true`
- Add config `configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml`.
- Add or rename public configs:
  - `configs/amortized_diffsky_hltds_standard_normal_gpu.yaml`
  - `configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml`
  - `configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml`
- Load supervised RealNVP checkpoints from PR 2 into the ELBO prior.
- If `train_jointly: false`, freeze prior parameters and update only encoder
  parameters.
- If `train_jointly: true`, preserve current joint encoder+prior behavior.
- Add standard normal baseline prior object with compatible `log_prob` and
  `sample`.
- Ensure latent spec compatibility between supervised prior checkpoint and
  amortized config; fail clearly on name/order/bounds mismatch.

Expected config block:

```yaml
amortized:
  prior:
    type: realnvp
    source: supervised_checkpoint
    checkpoint: outputs/runs/diffsky_supervised_prior_basic/checkpoints/best.eqx
    train_jointly: false
```

Outputs:

- `posterior_samples.parquet`
- `posterior_summary.parquet`
- `learned_or_loaded_prior_samples.parquet`
- `photoz_metrics.csv`
- `posterior_vs_truth_metrics.csv`
- Training summary that records `prior_source` and whether prior was frozen.

Tests:

- Add `tests/test_amortized_prior_source.py`.
- Frozen prior parameters are unchanged after one train step.
- Joint RealNVP prior has nonzero gradients/updates.
- Standard normal baseline works.
- Supervised checkpoint latent-spec mismatch fails clearly.

Acceptance:

- Same amortized code path supports standard normal, supervised frozen RealNVP,
  and joint RealNVP priors.

## PR 5 - Redshift Ablation

Goal: compare redshift recovery and posterior calibration across prior and
closure modes.

Implementation:

- Add CLI command `diffsky-redshift-ablation`.
- Compare:
  - standard normal prior;
  - supervised RealNVP prior fixed;
  - joint RealNVP prior;
  - fixed-z closure;
  - optional redshift-only mode.
- Read posterior samples/summaries from multiple run directories.
- Define a shared photo-z metric implementation usable by tests and reports.

Metrics:

- `delta_z = (z_pred - z_true) / (1 + z_true)`
- median bias
- `sigma_MAD`
- RMSE
- outlier fraction `|delta_z| > 0.15`
- PIT
- 68/95 coverage
- posterior width

Outputs:

- `redshift_ablation_summary.csv`
- `redshift_ablation_report.md`
- `z_pred_vs_z_true.png`
- `pit_histogram.png`
- `delta_z_histogram.png`

Tests:

- Add `tests/test_diffsky_redshift_ablation.py`.
- Toy posterior gives correct bias, sigma_MAD, RMSE, outlier fraction.
- Coverage and PIT are correct on a calibrated toy posterior.
- Command fails clearly if required truth redshift is missing.

Acceptance:

- Report compares posterior calibration, not only posterior median accuracy.
- Outputs include `redshift_ablation_summary.csv`,
  `redshift_ablation_report.md`, `z_pred_vs_z_true.png`,
  `pit_histogram.png`, and `delta_z_histogram.png`.

## PR 6 - Population Realism Diagnostics

Goal: compare the physical population implied by truth, learned prior, and
aggregate posterior.

Implementation:

- Extend `euclid_dsps/amortized/prior_overlap.py`.
- Add logSFR/logSSFR diagnostics when truth or derived posterior quantities are
  available.
- Add dust diagnostics when dust is fitted.
- Add Diffstar/Diffmah distributions for supervised prior runs.
- Compare:
  - `p_true(theta)`
  - supervised or learned `p_beta(theta)`
  - aggregate posterior `q_agg(theta)`
- Do not compare raw `dlog10_sfr_i` directly with `logsfr_true`.
- Add derived quantities where decoder support exists:
  - `SFR_at_obs`
  - `sSFR_at_obs`
  - `mass_weighted_age`
  - `recent_sfr_100myr`
  - `quenched_flag`
- If decoder can produce `sfr_at_obs_msun_per_yr`, export
  `log10_sfr_at_obs` in posterior samples and summaries.

Outputs:

- `prior_overlap_metrics.csv`
- `truth_vs_prior_corner.png`
- `truth_vs_qagg_corner.png`
- `truth_vs_prior_z_logm_logsfr.png`
- `population_realism_report.md`

Tests:

- Add `tests/test_population_realism_diagnostics.py`.
- Population overlap includes z/logM/logSFR when columns exist.
- Raw SFH-ratio-vs-logsfr comparisons are not emitted.
- Missing derived quantities are reported, not silently filled.

Acceptance:

- Report clearly separates direct truth, generated truth, learned prior, and
  posterior aggregate.

## PR 7 - Documentation And Public Config Cleanup

Goal: make the public workflow match the science-validation path and mark old
paths as secondary/deprecated.

Public config surface:

- Dataset:
  - `configs/diffsky_dataset_hltds_04_14.yaml`
- Supervised prior:
  - `configs/prior_diffsky_hltds_supervised_basic_realnvp.yaml`
  - `configs/prior_diffsky_hltds_supervised_extended_realnvp.yaml`
- Forward closure:
  - `configs/diffsky_hltds_04_14_trueparam_closure_gpu.yaml`
- Amortized inference:
  - `configs/amortized_diffsky_hltds_standard_normal_gpu.yaml`
  - `configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml`
  - `configs/amortized_diffsky_hltds_joint_realnvp_gpu.yaml`
- FS2 comparison:
  - `configs/fs2_gpu.yaml`
  - `configs/amortized_fs2_realnvp.yaml`

Docs to update:

- `README.md`
- `PLAN.md`
- `docs/source/diffsky_dataset.rst`
- `docs/source/amortized_inference.rst`
- `docs/source/run_setup.rst`
- `docs/source/prior_learning.rst`
- `docs/source/diffsky_forward_closure.rst`
- `docs/source/scientific_validation_plan.rst`

Documentation must explain:

- Dataset construction.
- Truth semantics.
- Supervised prior learning.
- Same-parameter forward closure.
- Photometric posterior inference.
- Redshift ablation.
- Population realism diagnostics.

Required warning:

```text
A good photometric fit is not evidence of physical recovery.
Physical claims require:
- same-param forward closure;
- supervised prior vs truth diagnostics;
- posterior calibration;
- comparison of derived quantities, not only raw latent parameters.
```

Tests/validation:

- Config load smoke for all public configs.
- Sphinx `-W --keep-going`.
- CLI `--help` includes public commands.
- Older configs are marked deprecated in docs or moved out of the public path
  without deleting code needed by tests.

Acceptance:

- A new user can identify the main Diffsky HLTDS science-validation path
  without reading old OpenUniverse-specific experiments.

## Global Acceptance Checklist

The full prompt is satisfied when:

- The prepared Diffsky dataset has a clear integrity report.
- A supervised prior can be trained on truth parameters with
  `diffsky-train-supervised-prior`.
- The supervised prior can be sampled and compared against truth.
- A true-parameter forward closure can be launched.
- Amortized inference supports:
  - standard normal prior;
  - supervised frozen RealNVP prior;
  - joint RealNVP prior.
- A redshift ablation report is generated.
- Docs clearly distinguish:
  - truth-prior learning;
  - same-parameter forward closure;
  - posterior inference.
- Existing FS2 and Diffsky tests pass.

Recommended PR order remains:

- PR 3 - Same-param Diffsky forward closure.
- PR 4 - Use supervised prior in amortized inference.
- PR 5 - Redshift ablation.
- PR 6 - Population realism diagnostics.
- PR 7 - Documentation and public config cleanup.
