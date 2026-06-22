# Plan

## 2026-06-22 PhotErr Error-Model Slides

- Status: in progress.
- Goal: create a Reveal.js slide deck explaining the PhotErr error model,
  the flux-space formula implemented in this repository, the regenerated error
  diagnostics, and how `fluxerr_*` plugs into the likelihood.
- Scope: use local diagnostic PNGs and MathJax equations; do not add a
  JavaScript build pipeline or package dependency.

## 2026-06-21 No-KL z<0.35 Rerun With Updated Error Model

- Status: implemented.
- Goal: rerun the pure autoencoder/DSPS reconstruction sanity check on the
  active continuous low-z Diffsky subset with the updated `m5_depth` +
  PhotErr-style systematic flux-error model.
- Dataset status: the local active subset
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  has been regenerated from the `m5_depth` source parquet with
  `sigma_sys_mag: 0.005`; it contains 78,651 objects with
  `redshift_true <= 0.35` and no high-redshift 0.4 feature.
- Slurm update: `scripts/diffsky_autoencoder_nokl_h100.slurm` now defaults to
  the full active subset (`LIMIT=`), 20 epochs, no KL
  (`--kl-weight-max 0.0`), and a 20-hour wall time. A smaller smoke run can
  still be requested explicitly with `LIMIT=10000`.
- Runtime guard: the script keeps the memory-safe posterior predictive
  inference knobs (`DECODER_SAMPLE_CHUNK_SIZE` and
  `PRIOR_PREDICTIVE_BATCH_SIZE`) so the post-training diagnostics can be run on
  the full subset without repeating the previous H100 OOM pattern.
- Follow-up adjustment: full-subset 30-epoch training is too slow for the
  current diagnostic loop. The Slurm wrapper now exposes `SIGMA_SYS_MAG` and
  `SELECTION_MODE` explicitly. The recommended short rerun is `LIMIT=20000`,
  `SELECTION_MODE=random`, `FORCE_REBUILD_DATASET=1`, `ERROR_MODEL=m5_depth`,
  and `SIGMA_SYS_MAG=0.005` so the parquet is rebuilt with the intended
  PhotErr-inspired systematic floor before training.
- Runtime fix: the H100 wrappers no longer require `$WORK` to be exported.
  They infer `REPO_DIR` from `SLURM_SUBMIT_DIR` when submitted from the repo,
  and infer `MINICONDA_PATH` from the parent directory of `REPO_DIR` when
  needed.
- Analysis completed for Jean-Zay job 730441
  `diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128`: the run used
  `m5_depth`, `sigma_sys_mag=0.005`, `FORCE_REBUILD_DATASET=1`, random
  20k-row selection, and `kl_weight_max=0.0`. Training is stable and improves
  through the run, with the best validation checkpoint at epoch 27.
- Reconstruction verdict: the no-KL encoder+DSPS path recovers the central
  photometry qualitatively better than the previous run, but it does not yet
  pass a likelihood-calibrated flux-reconstruction gate. On the 20k selected
  objects, `residual_sigma_median=(flux_in-flux_out)/sigma_eff` has global
  median `-0.38`, robust width `~1.90 sigma`, standard deviation `4.70`,
  `24.3%` of band residuals outside `|3 sigma|`, and `12.9%` outside
  `|5 sigma|`. Validation residuals are essentially the same as train
  residuals, so this is not simply a train/validation overfit artifact.
- Dominant residual failures remain band-dependent: `lsst_u` and `lsst_g`
  have median absolute residuals of roughly `6.8 sigma` and `5.4 sigma`, while
  `roman_F129`, `roman_F146`, and `roman_F158` are close to acceptable.
- Follow-up diagnostics update: future amortized inference outputs now add
  exact DSPS-derived `log10_sfr_at_obs` and `log10_ssfr_at_obs` quantities to
  posterior summaries and diagnostic prior samples. Extended truth comparisons
  and the truth/posterior/prior population corner include SFR and sSFR when
  `logsfr_true` and `logssfr_true` are available.

## 2026-06-21 PhotErr-Inspired Systematic Floor

- Status: implemented.
- Goal: align the synthetic `m5_depth` flux-error model with the core PhotErr
  point-source formulation without adding a runtime dependency on `photerr`.
- Planned model: keep the existing Rubin/PhotErr random term
  `sigma_rand^2 = (0.04 - gamma) * abs(flux) * f5 + gamma * f5^2`, then add
  the PhotErr-style irreducible systematic floor in quadrature,
  `sigma_sys^2 = (sys_frac * abs(flux))^2`, with
  `sys_frac = 10 ** (sigma_sys_mag / 2.5) - 1` and default
  `sigma_sys_mag = 0.005`.
- Scope: implement the formula in `photometric_uncertainty.py`, expose the
  default in manifests/config payloads, update tests and docs, and run targeted
  validation. Do not add `photerr` as a package dependency.
- Completed: added `DEFAULT_PHOTERR_SIGMA_SYS_MAG = 0.005` and the quadrature
  systematic term to the `m5_depth` model, exposed `--sigma-sys-mag` on
  Diffsky dataset/subset generation, and recorded `sigma_sys_mag` in manifests.
- Completed: regenerated
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet`
  and
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`.
  The current manifests now declare `sigma_sys_mag: 0.005`.
- Completed: regenerated subset flux-error diagnostics and analytical
  error-vs-flux curves, including the Sphinx static PNG copies.
- Validation completed: `pytest tests/test_photometry.py
  tests/test_diffsky_prepare_dataset.py -q`, `python -m compileall euclid_dsps
  scripts`, `diffsky-validate-dataset`, `pytest tests/test_config.py
  tests/test_cli.py tests/test_diffsky_validation.py -q`, strict Sphinx build,
  and `git diff --check`.

## 2026-06-19 Full-Dataset No-KL m5-Depth Rerun Setup

- Status: implemented.
- Goal: rerun the pure autoencoder/DSPS sanity check on the full active
  Diffsky dataset after regenerating or copying the new `m5_depth` `fluxerr_*`
  model.
- Added `FORCE_REBUILD_DATASET=1` support to the H100 autoencoder, inference,
  and full-validation Slurm scripts. When set, the scripts rebuild the active
  subset parquet from `SOURCE_DATASET` even if `DATASET` already exists; when
  unset, an existing copied parquet is used as-is.
- Fixed full-dataset overrides for the autoencoder and inference scripts:
  `LIMIT=` and `INFER_LIMIT=` now mean no row limit instead of falling back to
  the old 10k/5k defaults.
- Current default generated dataset contract in the H100 scripts:
  `DATASET=Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  rebuilt from
  `SOURCE_DATASET=Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet`
  with `ERROR_MODEL=m5_depth`, `REDSHIFT_MIN=0.0`, and
  `REDSHIFT_MAX=0.35`.

## 2026-06-19 No-KL z<0.35 Run Analysis

- Status: completed.
- Scope: analyze the completed Jean-Zay run
  `outputs/runs/diffsky_autoencoder_nokl_z035_retry1` and its inference output
  `outputs/runs/diffsky_autoencoder_nokl_z035_retry1_infer`.
- Checks to perform: confirm KL was disabled in the objective, inspect
  train/validation likelihood convergence, summarize posterior-predictive
  normalized residual distributions globally and per band, inspect redshift and
  truth diagnostics, and identify whether the no-KL autoencoder passes the
  flux-reconstruction sanity gate.
- Confirmed the run is genuinely no-KL in the objective: all logged rows have
  `kl_weight=0.0`, the phase is `joint_no_prior`, the prior source is
  `standard_normal`, `train_prior=false`, and `prior_grad_norm=0.0`. The logged
  `kl_mc_mean` is only a diagnostic and is not multiplied into the loss.
- Training/validation NLL improves through epoch 12, and the best checkpoint is
  epoch 12. The run is numerically stable, but the inference closure gate fails.
- Flux-reconstruction sanity verdict: fail under the current likelihood errors.
  The global normalized residual distribution has mean `1.47`, standard
  deviation `7.58`, median `-0.10`, `37.8%` outside `|3 sigma|`, and `24.5%`
  outside `|5 sigma|`.
- Dominant failing bands: `lsst_u`, `lsst_g`, `lsst_r`, `roman_F062`,
  `lsst_i`, and `roman_F213`. Median absolute residuals are especially large
  for `lsst_u` (`13.4 sigma`), `lsst_g` (`8.1 sigma`), `lsst_r`
  (`4.1 sigma`), and `roman_F062` (`4.1 sigma`).
- Inferred redshift collapses to the upper bound: median `z_obs_median` is
  `0.34994`, `97.4%` of objects have `z_obs_median > 0.345`, and `90.9%` have
  `z_obs_median > 0.349`. The median 68% posterior width in redshift is
  `7.3e-5`, so the posterior is extremely overconfident.
- Closure metrics fail on redshift: photo-z median bias `0.0804`, `z_obs`
  median bias `0.1004`, and `coverage_68=0.0`. Stellar mass median bias is
  moderate (`0.182 dex`), but this is not enough to accept the run.
- The scalar `posterior_predictive_chi2_median` is `inf` for all 10k inference
  objects while the per-band residual summary is finite. Treat that scalar as a
  diagnostics bug/gap for this run and use
  `posterior_predictive_residual_summary.parquet` plus
  `posterior_predictive_normalized_residual_tails.csv` instead.
- Recommended next gate: do not proceed to full KL/NF science training from
  this checkpoint. First isolate whether the issue is the DSPS/data/error model
  or the amortized encoder by running a redshift-fixed/truth-redshift no-KL
  closure and a small likelihood-only MAP/z-grid check on the same z<0.35
  subset.

## 2026-06-19 Synthetic Flux Error Implementation Plan

- Status: implemented.
- Goal: replace the main Diffsky `fractional_snr` synthetic errors with a
  simple band-depth error model that depends on flux, remains deterministic, and
  is usable for all LSST+Roman bands.
- Model: use `sigma_cat_b^2 = 0.04 * ((1 - eta_b) * abs(flux_b) * f5_b +
  eta_b * f5_b^2)`, where `f5_b = fnu(m5_b)` and `eta_b` is the background
  fraction of the 5-sigma variance. This is equivalent to the Rubin/LSST
  `m5,gamma` form with `gamma_b = 0.04 * eta_b`.
- LSST defaults: use the Rubin/LSST `m5,gamma` model for LSST bands. `m5_b`
  must be chosen from the intended scenario, e.g. single-visit, 10-year coadd,
  or HLTDS-like synthetic depth. `gamma_b` can be configured per band, with
  Science Book/rubin_sim values as defaults.
- Roman defaults: do not reuse LSST `gamma` as an official Roman model. Use
  Roman WFI 5-sigma AB sensitivities as `m5_b`; set `eta_b=1.0` for a pure
  depth floor or `eta_b~0.95` for a simple source-plus-background approximation
  until an ETC/Pandeia-derived SNR table is available.
- Integration: add a `depth_flux` or `m5_depth` error-model type in
  `photometric_uncertainty.py`, expose per-band `m5` and `eta/gamma` through
  `diffsky-redshift-subset`, update `configs/diffsky_dataset_hltds_04_14.yaml`
  and H100 Slurm defaults, regenerate the `continuous_lowz_fluxerr` parquet,
  and update docs/manifests to distinguish catalog/statistical `fluxerr_*` from
  the likelihood model floor.
- Likelihood cleanup: keep `fluxerr_*` as catalog noise and use a separate
  `flux_error_floor_frac` only for model/calibration tolerance. Apply the same
  floor reference convention in MAP/MCMC and amortized likelihoods before
  comparing likelihood values across methods.
- Completed: added the `m5_depth` model to `photometric_uncertainty.py` and
  threaded `band_name` through dataset generation and all observation-loading
  fallbacks (`io.py`, `observation_arrays.py`, and workflow batch arrays).
- Completed: changed Diffsky dataset generation defaults and H100 preflight
  scripts from `fractional_snr` to `m5_depth`; configs now point to
  `hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet` as the
  full source artifact.
- Completed: regenerated the full source parquet
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet`
  and the active low-z subset
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  with `m5_depth` `fluxerr_*` columns.
- Completed: generated error diagnostics under
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/`:
  `flux_error_summary.csv`, `flux_fractional_error_by_band.png`,
  `flux_snr_by_band.png`, and `flux_vs_fluxerr_by_band.png`.
- Completed: generated explanatory per-band error-vs-flux diagnostics:
  `flux_error_model_curves_by_band.png`,
  `flux_fractional_error_model_curves_by_band.png`, individual
  `flux_error_model_curves/flux_error_model_curve_<band>.png` files, and
  `flux_error_model_curve_summary.csv`.
- Completed: updated Roman synthetic noise from a pure depth approximation
  (`eta=1.0`, `gamma=0.04`) to `eta=0.95` (`gamma=0.038`) so Roman bands retain
  a non-zero source/Poisson-like term while remaining depth dominated.
  Regenerated the full source parquet, active low-z subset, manifests, standard
  error diagnostics, and model-curve plots. Rebuilt `docs/_build/html` with the
  explanation of absolute versus relative error.
- Validation completed: targeted tests
  `pytest tests/test_photometry.py tests/test_diffsky_prepare_dataset.py
  tests/test_diffsky_validation.py -q`, config/CLI tests
  `pytest tests/test_config.py tests/test_cli.py -q`, `python -m compileall
  euclid_dsps scripts`, `bash -n` on the H100 Slurm scripts,
  `diffsky-validate-dataset` on the regenerated low-z subset, strict Sphinx
  build, and `git diff --check`.

## 2026-06-19 Web Review of Synthetic Photometric Errors

- Status: completed.
- Scope: check external survey/photometry references for whether the current
  `fluxerr_* = abs(flux) / 50` model is scientifically appropriate.
- Finding: the formula is internally coherent only as a constant-SNR tolerance
  model. It follows directly from `SNR = flux / sigma`, but it assumes every
  band and every object is measured at the same fractional precision.
- Finding: it is not a realistic Rubin/Roman/HLS-style photometric error model.
  Survey references compute SNR from source counts plus sky/background,
  read/dark/instrumental noise, aperture/PSF footprint, exposure time, object
  size, and band-dependent limiting depth. Roman WFI documentation also gives
  band/source-size/integration-dependent 5-sigma AB limits.
- Recommended fix: replace the main science synthetic-errors path with a
  band-dependent depth model, e.g. `sigma_depth_b = fnu(m5_b) / 5`, then combine
  it in quadrature with an explicit calibration/model floor. Keep
  `abs(flux)/50` only for closure/smoke tests where the desired assumption is a
  fixed fractional tolerance.
- Recommended fix: avoid double-counting the same fractional uncertainty in
  both materialized `fluxerr_*` and `fit.flux_error_floor_frac`; decide whether
  `fluxerr_*` means catalog/statistical noise and `flux_error_floor_frac` means
  model/calibration tolerance, then apply the same convention in MAP/MCMC and
  amortized likelihoods.

## 2026-06-19 Documentation Contract Cleanup

- Status: completed.
- Scope: update repository guidance and public docs after the documentation,
  dataset, and flux-error audit. Keep the public path on the
  `continuous_lowz_fluxerr` Diffsky subset, document the deterministic
  `fluxerr_*` generation, and remove stale references to removed PopCosmos and
  FS2-only amortized paths.
- Completed: refreshed `AGENTS.md`, the top-level README, `configs/README.md`,
  and Sphinx pages for architecture, dataset, forward-model, and run setup.
  Public commands now build the no-error 04/14 source parquet, materialize the
  `z < 0.35` continuous subset with `fractional_snr` SNR 50 errors, and use the
  `continuous_lowz_fluxerr` dataset for Diffsky MAP, closure, supervised-prior,
  amortized, and diagnostics examples.
- Completed: documented that `flux_*` columns are AB-magnitude-derived
  `fnu_cgs` fluxes and that current `fluxerr_*` columns are deterministic
  likelihood inputs, `max(abs(flux) / 50, 1e-40)`, not native survey errors or
  per-fit random draws.
- Follow-up: code still uses observed-flux fractional floors for MAP/MCMC and
  model-flux fractional floors for amortized likelihood/residual diagnostics.
  The docs now state this explicitly; unify the implementation before using
  absolute MAP-vs-amortized likelihood values as a quantitative comparison.
- Validation completed: `git diff --check`, `python -m compileall euclid_dsps
  scripts`, and `uv run python -m sphinx -W --keep-going -b html docs/source
  outputs/doc_audit_html`.

## 2026-06-19 Documentation, Dataset, and Flux-Error Audit

- Status: completed.
- Scope: compare README/Sphinx/config documentation against the current
  checkout, local processed datasets, public configs, and implemented
  flux-error/likelihood paths.
- Questions to answer: whether dataset choices and artifacts are documented
  coherently, whether stale config names remain in the docs, and whether the
  per-band flux-error model is random or an explicit deterministic likelihood
  assumption.
- Findings: the main Diffsky dataset contract is coherent in code/configs and
  current Sphinx pages: active public configs point to
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  which exists locally with `78651` objects, `z=0.006876--0.334662`, 14
  `flux_*` bands, and 14 `fluxerr_*` bands documented as synthetic
  `fractional_snr` SNR 50 errors in both manifest and schema.
- Findings: the top-level README is partially stale because several example
  Diffsky commands still pass the source
  `*_photometry_truth_noerr.parquet` to closure/prior/overlap/redshift
  diagnostics, while `docs/source/run_setup.rst` and active configs use the
  `continuous_lowz_fluxerr` subset.
- Findings: `configs/README.md` and `docs/source/diffsky_dataset.rst` still
  describe `amortized_diffsky_hltds_04_14_realnvp_gpu.yaml` as the main
  Diffsky amortized config, but the public command docs promote
  `amortized_diffsky_hltds_joint_realnvp_gpu.yaml`, which extends the base and
  explicitly sets `prior.source: joint_realnvp`.
- Findings: `docs/source/architecture.rst` is stale for amortized inference:
  it still says `amortized/` is FS2-only and lists only the old narrow config
  set, while the current code/configs support Diffsky HLTDS amortized training,
  inference, prior overlap, sharded finalization, and H100 validation.
- Findings: the synthetic flux-error model is deterministic after dataset
  materialization, not random during fitting:
  `fluxerr_* = max(abs(flux) / 50, 1e-40)` for the main subset. The
  likelihood then adds a 2% fractional floor plus jitter in quadrature.
- Blocker/gap: the documentation overstates the effective uncertainty formula
  as using `max(abs(obs_flux), abs(model_flux))`; current code uses observed
  flux for MAP/MCMC/posterior-target paths and model flux for amortized
  likelihood/posterior-predictive diagnostics unless `error_floor_reference`
  is overridden.
- Blocker/gap: older local processed datasets with synthetic `fluxerr_*`
  columns (`*_photometry_truth.parquet` and `03_31`) have manifests that note
  synthetic SNR errors, but their schema JSON files do not list
  `flux_error_columns`; they should be treated as historical artifacts or
  regenerated if promoted again.
- Validation completed: public config-load smoke for Diffsky/FS2 configs,
  local parquet/manifest/schema inventory, CLI help checks, and
  `uv run python -m sphinx -W --keep-going -b html docs/source outputs/doc_audit_html`.

## 2026-06-19 Jean-Zay Dataset Preflight For z<0.35 Runs

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Root cause from Jean-Zay logs: the no-KL job used the intended main dataset
  path,
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  but that derived parquet was not present in the remote checkout, so training
  failed during catalog fingerprinting before any model step.
- Updated H100 Slurm scripts to treat the z<0.35 continuous subset as a
  first-class runtime dependency. Before training/inference/validation they now
  check `DATASET`, print its size when present, or rebuild it with
  `diffsky-redshift-subset` from `SOURCE_DATASET` using
  `REDSHIFT_MIN=0.0`, `REDSHIFT_MAX=0.35`, `ERROR_MODEL=fractional_snr`, and
  `ERROR_SNR=50`.
- The same preflight was added to:
  `scripts/diffsky_autoencoder_nokl_h100.slurm`,
  `scripts/diffsky_amortized_infer_h100.slurm`, and
  `scripts/diffsky_full_validation_h100.slurm`.
- Documentation updated to state that the main subset has `78651` objects and
  that H100 entry points can rebuild the derived subset on Jean-Zay when the
  source `*_photometry_truth_noerr.parquet` exists.
- Validation completed: `bash -n` on the three edited Slurm scripts, rebuilt
  Sphinx HTML in both documented output locations, and `git diff --check`.

## 2026-06-18 Remove 0.4 Redshift Island From Main Subset

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Recompute the main continuous-low-z Diffsky subset with
  `redshift_true <= 0.35` instead of `<= 0.5` to exclude the separated
  `z~0.4` island seen in the redshift histogram.
- Keep the main parquet path stable:
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  but update manifests, configs, Slurm defaults, docs, and rebuilt HTML docs to
  state the `0.0--0.35` subset contract.
- Update Jean-Zay run names to include `z035` so new no-KL and
  posterior-predictive runs are not confused with older `lowz` outputs.
- Recomputed the stable main parquet with `78651` objects,
  `redshift_true` range `0.006876--0.334662`, median `0.249744`, and the same
  `fractional_snr` SNR 50 `fluxerr_*` model.
- Copied the rebuilt redshift/truth plots into `docs/source/_static/` and
  rebuilt both `docs/build/html/diffsky_dataset.html` and
  `docs/_build/html/diffsky_dataset.html`; both HTML pages now show the z<0.35
  subset plots and no `z~0.4` island.
- Validation completed: `python -m compileall euclid_dsps scripts`, `bash -n`
  on H100 Slurm scripts, config-load smoke for active Diffsky configs,
  `uv run ruff check euclid_dsps tests`, targeted pytest (`63 passed`), and
  full `uv run pytest` (`349 passed, 5 skipped, 2 warnings`).

## 2026-06-18 Continuous Low-z Dataset As Main Diffsky Dataset

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Promote
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  to the default 04_14 Diffsky training/evaluation dataset. The old
  `*_photometry_truth_noerr.parquet` catalog remains a source/input artifact
  for rebuilding subsets, not the default science training catalog.
- Recompute the continuous redshift subset with materialized `fluxerr_*`
  columns using the shared flux-dependent error model and write companion
  manifest, schema, truth summary/report, and distribution plots.
- Clean active 04_14 configs so PopCosmos-like runs expose all SFH ratio bins
  in the latent/free-parameter vector. Low-dimensional legacy experiment
  filenames should no longer freeze `dlog10_sfr_2..6`.
- Update Jean-Zay Slurm defaults so no-KL autoencoder and posterior predictive
  residual diagnostics operate on the continuous low-z subset by default.
- Document in `.rst` the main dataset path, the uncertainty model,
  likelihood residual units, encoder flux/error feature standardization,
  standardized-logit latent coordinates, DSPS physical parameter decoding, and
  SSP/compressed SSP usage.
- Recomputed the main subset:
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  with `78651` objects, `redshift_true` range `0.006876--0.334662`,
  median `0.249744`, and 14 materialized `fluxerr_*` bands from the
  `fractional_snr` SNR 50 model. Companion manifest, schema, summary,
  truth summary/report, and redshift/truth distribution plots were written.
- Generated the SSP/dust audit under
  `outputs/reports/diffsky_dust_ssp_audit/`, including
  `dust_ssp_audit.md`, `dust_ssp_audit.json`, `dust_transmission_grid.csv`,
  and `dust_transmission_grid.png`.
- Validation completed after the documentation/config cleanup:
  `python -m compileall euclid_dsps scripts`, `bash -n` on the updated Slurm
  scripts, config-load smoke for active Diffsky configs,
  `uv run ruff check euclid_dsps tests`, targeted pytest (`63 passed`),
  `git diff --check`, and full `uv run pytest`
  (`349 passed, 5 skipped, 2 warnings`).

## 2026-06-18 Proposed Diffsky Model Sanity Plan

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Add likelihood-consistent normalized residual diagnostics for post-training
  inference: signed `(flux_in - flux_out) / sigma_eff` histograms globally and
  per band, box/violin summaries by band, Gaussian reference curves, and
  vertical markers at -3 and +3. Use the same `sigma_eff` definition as the
  training likelihood, including flux error model, floor, and jitter, and
  write both plots and machine-readable summary tables with tail fractions.
- Generalize the existing posterior-predictive residual plots so every
  residual labeled "sigma" is dimensionless and computed from flux units
  consistently. Keep the sign convention explicit: input/catalog flux minus
  predicted/model flux.
- Reactivate all PopCosmos-like SFH bins by making `dlog10_sfr_4`,
  `dlog10_sfr_5`, and `dlog10_sfr_6` free parameters in the relevant
  Diffsky HLTDS configs, with matching bounds, latent names, encoder latent
  dimension, RealNVP dimension, inference summaries, truth/proxy diagnostics,
  and MAP start generation updated together.
- Audit the current Prospector/FSPS dust parameters (`tau2`, `dust_index_n`,
  `tau1_over_tau2`) against the SSP/data assets and DSPS implementation:
  confirm physical meaning, allowed ranges, wavelength behavior, young/old
  stellar population split, and whether the current bounds are coherent for
  HLTDS closure tests.
- Define and implement an explicit per-galaxy/per-band uncertainty model as a
  function of flux for `*_noerr` catalogs. Candidate models to compare:
  fractional-SNR Gaussian, magnitude-tolerance propagation, flux floor plus
  fractional floor, and, only if survey exposure/depth metadata exists, a
  Poisson/depth-inspired model. The chosen model must materialize
  `fluxerr_*` columns or an equivalent runtime array and be used identically by
  training, inference diagnostics, and MAP diagnostics.
- Add normalized latent coordinates for the autoencoder path: train the
  encoder/NF in a standardized latent coordinate system, then denormalize or
  invert-transform before passing physical parameters to DSPS. Preserve the
  existing bounded-parameter safety checks and document the exact transform in
  run summaries.
- Add a no-KL autoencoder sanity experiment: train with `kl_weight=0` and the
  NF prior disabled or ignored, then check whether the encoder+DSPS decoder can
  reconstruct catalog fluxes on train/validation using the normalized residual
  diagnostics. This is the pass/fail gate before interpreting learned priors.
- Build a continuous-redshift 04_14 subset dataset for easier first training:
  identify the continuous-redshift slice, recompute a processed catalog for
  that subset, report object counts, redshift distribution, and ground-truth
  parameter distributions, then point dedicated smoke/science configs at this
  subset.
- Validation order after implementation: compileall and targeted tests,
  regenerate the subset manifest, run a no-KL autoencoder smoke, run normal
  KL/RealNVP training only if the no-KL flux reconstruction passes, and inspect
  normalized residual tails before scaling.
- Implemented `euclid_dsps.photometric_uncertainty` as the shared source for
  flux-dependent synthetic errors and likelihood `sigma_eff`; Diffsky no-error
  AB-mag presets now synthesize `fractional_snr` errors by default, while
  prepared/subset datasets can materialize `fluxerr_*` columns.
- Implemented likelihood-consistent posterior-predictive residual diagnostics:
  signed `(flux_in - flux_out) / sigma_eff`, global and per-band histograms
  with Gaussian reference and +/-3 markers, by-band box summaries, and
  machine-readable tail tables.
- Implemented full 12D PopCosmos-like HLTDS latent by freeing
  `dlog10_sfr_4..6`, updating encoder latent dimension, adding standardized
  logit latent coordinates, and preserving identity normalization for
  supervised-prior checkpoint configs.
- Implemented `diffsky-redshift-subset` to build continuous-redshift prepared
  parquet subsets with optional `fluxerr_*` materialization, object counts,
  redshift/truth summaries, manifests, and plots.
- Implemented `diffsky-dust-ssp-audit` for SSP wavelength/age summaries and
  Prospector/FSPS dust transmission curves over configured dust parameter
  bounds.
- Added configs for the continuous low-z subset and no-KL autoencoder sanity
  check, plus `scripts/diffsky_autoencoder_nokl_h100.slurm` for train+infer
  reconstruction validation.
- Validation completed locally: `python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, `bash -n` on the new Slurm script,
  `git diff --check`, `diffsky-redshift-subset` smoke on 100 objects, targeted
  amortized/config tests, and full `pytest` (`346 passed, 8 skipped`).

## 2026-06-18 Model/Likelihood Audit

- Current analysis slice: audit the current DSPS/amortized model rather than
  changing science behavior. Answer the outstanding questions on photometric
  error modeling, normalized flux residuals, latent/flux normalization,
  RealNVP prior initialization/evolution, and the exact DSPS parameterization.
- Inspect the code paths that load catalog errors, build likelihood weights,
  normalize observed fluxes/features, initialize/train the NF prior, and map
  latent vectors into DSPS inputs. Record gaps and follow-up implementation
  work in this plan after the audit.
- Audit completed: the active Diffsky HLTDS H100 configs use the `*_noerr`
  processed catalog with no native or synthetic flux-error columns, so the fit
  currently relies on the per-band `sigma_mag=0.10` model-tolerance fallback
  plus likelihood floor/jitter rather than a Poisson or survey-depth noise
  model. Older/non-noerr prepared catalogs and OpenUniverse helpers use a
  simple Gaussian fractional-SNR recipe (`sigma = abs(flux) / snr`, usually
  SNR 50), not a Poisson derivation.
- Audit completed: MAP comparison outputs already contain
  likelihood-space normalized residuals (`chi_likelihood`) including
  floor/jitter, while amortized posterior-predictive residual summaries use
  raw photometric errors, the opposite sign convention, and absolute-value
  histograms. Follow-up: add signed `(flux_in - flux_out) / sigma_eff`
  residual plots with Gaussian overlays and +/-3 lines using exactly the
  likelihood sigma.
- Audit completed: encoder photometry features are robustly normalized per
  band, but DSPS parameters are physical bounded parameters mapped through an
  unconstrained logit latent; they are not z-scored. Flux amplitude is set by
  the surviving stellar mass parameter plus trainable global/per-band
  calibration, not by a separate DSPS normalization parameter.
- Audit completed: the joint RealNVP prior is initialized from random
  RealNVP coupling networks over the unconstrained latent, with a standard
  normal base distribution and small scale clamp; current stable configs
  freeze the prior for early epochs and then alternate encoder/prior updates.
  Existing training logs expose `logprior_mean`, `kl_mc_mean`,
  `prior_grad_norm`, `update_phase`, entropy, and likelihood-temperature
  evolution, but a dedicated prior-evolution report/plot should be added for
  science runs.
- Audit completed: the current Diffsky HLTDS decoder is a PopCosmos-binned
  DSPS proxy with free `z_obs`, stellar mass, three SFH ratios, stellar
  metallicity, and dust parameters. Later SFH ratios, gas/AGN, several dust
  details, SSP metadata policy, and calibration priors are fixed by config.
  The runtime uses the configured SSP HDF5 and compressed SSP basis rather
  than plain DSPS package defaults.

## 2026-06-17 Diffsky Collapse-Fix Validation Ladder

- Current implementation slice: keep redshift free and keep training
  unsupervised. Truth columns are used only for closure diagnostics and plots.
- Added `PLAN_FIX.md` as the implementation checklist for phases A-E:
  extended truth diagnostics, actual `popcosmos_bins` decoder proxy closure,
  MAP-Adam multistart likelihood-landscape diagnostics, low-dimensional
  controlled configs, and automatic collapse gates.
- Implemented extended truth snapshots/diagnostics for inference and MAP runs:
  posterior/MAP/prior-vs-truth tables, object-level parquet outputs, truth
  vs estimate plots, bias plots, population overlays, and compact population
  corner plots for comparable/proxy parameters.
- Implemented `diffsky-popcosmos-proxy-closure` plus
  `scripts/diffsky_popcosmos_proxy_closure_h100.slurm` to validate the exact
  amortized `popcosmos_bins` decoder against Diffsky truth proxies before
  interpreting encoder/NF results.
- Implemented MAP-Adam start modes `encoder`, `prior`, `z_grid`, `lowz_grid`,
  `latin_hypercube`, and `mixed`, with per-start-family parquet/CSV outputs
  and plots. This supports likelihood-only debug runs via `PRIOR_WEIGHT=0.0`.
- Added H100 configs for full widened redshift bounds and low-dimensional
  `z + mass + SFH proxy + dust` diagnostics:
  `diffsky_hltds_popcosmos_full_bounds_h100.yaml`,
  `diffsky_hltds_popcosmos_lowdim_z_mass_dust_h100.yaml`, and
  `diffsky_hltds_popcosmos_lowdim_z_mass_dust_lowz_h100.yaml`.
- Added automatic training/inference/MAP collapse gates that write JSON
  artifacts and surface pass/warn/fail status in run summaries.

## 2026-06-17 Diffsky Unsupervised RealNVP Stabilization

- Current implementation slice: keep the science objective fully
  photometry-driven and unsupervised. Do not train on `redshift_true` and do
  not fix redshift; truth columns are closure diagnostics only.
- Stabilize the encoder/DSPS/RealNVP path so the learned NF prior captures
  population degeneracies instead of amplifying a collapsed amortized posterior.
  The target changes are likelihood-temperature scheduling, posterior entropy
  floors, delayed/alternating prior updates, prior-predictive distribution
  diagnostics, and checkpoint gates based on non-supervised collapse signals.
- Add dense inference batching so selected rows are packed into full
  `jax_batch_size` batches rather than hundreds of sparse catalog-window
  micro-shards.
- Add an inference finalizer that can combine shard outputs, report incomplete
  runs, compute diagnostics, and generate plots after a Slurm interruption.
- Add a MAP-Adam under learned RealNVP prior workflow so final science
  estimates can optimize the DSPS photometric likelihood plus learned prior
  log-density, with redshift remaining free.
- Implemented in the working tree:
  `configs/experiments/diffsky_hltds_joint_realnvp_unsup_stable_h100.yaml`,
  anti-collapse objective controls in the amortized ELBO/training loop,
  prior update schedules, prior-predictive color diagnostics, dense selected
  inference batches, an `amortized-finalize-inference` CLI, and
  `diffsky-map-adam-prior` plus `scripts/diffsky_map_adam_prior_h100.slurm`.
- Validation completed: `uv run python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, targeted amortized/redshift pytest
  (`18 passed`), CLI help smokes for train/infer/finalize/MAP, `bash -n` on
  the three H100 Slurm scripts, and `git diff --check`.
- Next Jean-Zay order: run a 5k/5-epoch RealNVP smoke, infer 1k with dense
  shards, run MAP-Adam 1k under the learned prior, then scale to 30k/20 epochs
  only if prior-predictive colors, entropy/log-std diagnostics, photo-z closure
  plots, and MAP optimizer traces look non-collapsed.

## 2026-06-16 Diffsky Photo-z RealNVP Calibration

- Current implementation slice: refocus the next Diffsky H100 run on
  recovering redshift from photometry and comparing against true redshift,
  with `joint_realnvp` as the main prior path rather than treating
  `standard_normal` as a science target.
- Drop fixed-z closure from the main decision gate. The gate is now
  free-redshift inference on a diagnostic subset: photo-z bias, RMSE,
  outlier fraction, PIT/coverage, residuals by band, chi2 outliers, and
  redshift bias by truth-redshift bin. Stellar mass remains a degeneracy
  symptom, not the primary pass/fail metric.
- Add trainable per-band flux calibration in the amortized path so the model
  can absorb small zero-point/color mismatches without forcing redshift or mass
  to compensate. The calibration is saved to JSON/CSV/PNG and included in
  training/inference summaries.
- Add a train-only H100 Slurm entry point for amortized Diffsky runs, and add
  a dedicated `joint_realnvp` photo-z H100 experiment config with per-band
  calibration enabled, post-annealing best-checkpoint selection, sharded
  inference defaults, and resumable outputs.
- Add `photoz_metrics_by_redshift_bin.csv` to the standard inference photo-z
  metrics so the diagnostic gate has an explicit redshift-bin bias/coverage
  table, not only per-object samples.
- Follow-up after the failed 10k/5k smoke: inference and metric outputs now
  preserve `row_index`, write `inference_truth.parquet`, prefer row-index
  truth joins, fail on duplicate legacy object-id joins, save catalog
  fingerprints, and support balanced/random/stratified inference selection
  through CLI and H100 Slurm exports. The photo-z H100 config now defaults to
  balanced redshift selection and a conservative first RealNVP
  `kl_weight_max=0.05`.
- Validation completed with `python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, targeted amortized pytest
  (`19 passed`), CLI help smokes for `amortized-train-diffsky` and
  `amortized-infer-diffsky`, and `bash -n`
  on the H100 train/infer/full-validation Slurm scripts.

## 2026-06-15 Diffsky H100 Pipeline Cleanup

- Current implementation slice: clean up the H100 Diffsky science pipeline after
  the partial Jean-Zay run showed supervised-prior collapse, bad
  `best.eqx` selection under KL annealing, and non-resumable monolithic
  inference writes.
- Remove supervised-prior stages from the default full-validation science path.
  Keep the supervised-prior code available as an explicit diagnostic/dev tool,
  but do not feed collapsed supervised priors into amortized science stages.
- Make the default science sequence: true-parameter forward closure,
  `standard_normal` amortized baseline, `joint_realnvp` amortized target,
  sharded inference, redshift diagnostics, population realism, and final report.
- Select amortized `best.eqx` on a stable post-annealing metric
  (`validation_negative_loglike` by default) instead of the annealed
  `validation_loss`.
- Add sharded, resumable amortized inference outputs so long runs write useful
  partial products batch by batch and avoid concatenating large posterior
  samples/flux tables in memory.
- Add Slurm entry points for cleaned full validation and checkpoint-only
  sharded inference from an existing `last.eqx`/epoch checkpoint.
- Completed in the working tree: H100 full validation defaults now skip
  supervised-prior stages and run `standard_normal` plus `joint_realnvp`;
  amortized `best.eqx` selection uses post-annealing
  `validation_negative_loglike`; inference can write resumable shards with
  compact combined summaries and optional large predictive/residual outputs;
  redshift and population reports can consume posterior sample shards; and
  `scripts/diffsky_amortized_infer_h100.slurm` launches checkpoint-only
  sharded inference from an existing training run.
- Validation completed with `uv run python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, targeted `pytest` for
  redshift/diagnostics/full-validation
  reporting, CLI help smokes for `diffsky-run-full-validation` and
  `amortized-infer-diffsky`, H100 config-load smoke, and `bash -n` on the H100
  Slurm scripts.
- Review follow-up completed in the working tree: standalone
  `amortized-infer-diffsky --jax-batch-size` now overrides
  `amortized.inference.jax_batch_size`, and shard resume metadata now includes
  run/batch partition details plus an object-id digest. Redshift and population
  diagnostics prefer `posterior_shards_manifest.json` so stale shards in the
  same output directory are ignored after a completed run.
- Review follow-up validation completed with `uv run python -m compileall
  euclid_dsps scripts`, `uv run ruff check euclid_dsps tests`, targeted
  `pytest` (`11 passed`), CLI help smokes, `bash -n` on the H100 Slurm scripts,
  and `git diff --check`.

## 2026-06-11 Documentation Style Refresh

- Current implementation slice: improve the public-facing documentation style
  without changing runtime behavior or scientific assumptions.
- Focus on the first surfaces a reader sees: top-level `README.md`, the
  Sphinx landing page, and lightweight HTML styling/configuration.
- Keep the existing Sphinx/RTD dependency set; avoid introducing new
  documentation tooling for a cosmetic pass.
- Completed in the working tree: rewrote the README entry point with workflow
  tables and clearer sectioning; rebuilt the Sphinx landing page around
  reader goals, workflow stages, boundaries, and guardrails; grouped the
  toctree into Getting Started, Science Workflows, and Reference; added a
  local RTD-theme CSS override; and normalized visible heading style in the
  installation/run setup pages.
- Validation completed with `uv run sphinx-build -W --keep-going docs/source
  docs/_build/html`.

## 2026-06-11 Diffsky Student-t and Global SED Scale Calibration

- Follow-up implementation slice: close the remaining review gaps by adding a
  real `diffsky-run-full-validation` entrypoint/report, prior-predictive
  photometry with raw/scaled fluxes, alpha/likelihood metadata in redshift
  metric outputs, conditional architecture metadata, and a stronger frozen
  alpha update test.
- Follow-up completed in the working tree: added
  `euclid_dsps.diffsky_full_validation`, CLI command
  `diffsky-run-full-validation`, report-only aggregation for existing stage
  outputs, prior-predictive DSPS photometry output
  `prior_predictive_flux.parquet`, likelihood/alpha metadata in
  `photoz_metrics.csv`, conditional architecture component reporting, and
  stronger tests for frozen alpha and full-validation report contents.
- Follow-up validation completed with `uv run ruff check euclid_dsps tests`,
  `uv run python -m compileall euclid_dsps tests`, `uv run sphinx-build -W
  --keep-going docs/source docs/_build/html`, CLI help smoke for
  `diffsky-run-full-validation`, and `uv run pytest` (`332 passed, 5 skipped,
  2 warnings`).
- Current implementation slice: make the Diffsky science path use a robust
  Student-t photometric likelihood by default and add a single global
  `alpha_sed` nuisance parameter for SED/flux normalization mismatch.
- Keep `alpha_sed` outside supervised truth-prior learning. It is a global
  decoder/likelihood calibration parameter, not a per-galaxy physical
  component of `theta`.
- Apply the global scale exactly once on model photometry paths, expose
  raw/scaled flux diagnostics, and report the stellar-mass degeneracy through
  raw and alpha-corrected mass summaries.
- Update forward closure with disabled/fixed/fit-global alpha modes and
  before/after residual outputs. Update amortized training/inference so
  `log_alpha_sed` can be trainable or frozen independently of the prior.
- Update public configs, tests, and docs so the new default is explicit and
  Gaussian remains available only as an explicit ablation choice.
- Implementation completed in the working tree: added
  `euclid_dsps.calibration`, Student-t defaults for Diffsky science configs,
  amortized trainable/frozen global `log_alpha_sed`, raw/scaled flux outputs,
  alpha-corrected mass summaries, forward-closure alpha modes and reports,
  H100 experiment config metadata, and documentation of the calibration
  nuisance parameter.
- Validation completed with `uv run ruff check euclid_dsps tests`,
  `uv run pytest` (`328 passed, 5 skipped, 2 warnings`),
  `uv run python -m compileall euclid_dsps tests`, config-load smoke for all
  public YAML files, and `uv run sphinx-build -W --keep-going docs/source
  docs/_build/html`.

## 2026-06-11 Diffsky Physical Validation PR 3-7

- Implemented the remaining Diffsky physical-validation scaffold from
  `PR3_PR7_TODO.md` while keeping the three scientific objectives separate:
  supervised truth-prior learning, same-parameter forward closure, and
  photometric amortized inference.
- PR 3 completed in the working tree: added
  `euclid_dsps.diffsky_forward_closure`, CLI command
  `diffsky-forward-closure`, config
  `configs/diffsky_hltds_04_14_trueparam_closure_gpu.yaml`, and tests for
  truth-column mapping, missing Diffstar/Diffmah failures, fixed nuisance
  metallicity recording, and mock zero-residual closure.
- PR 4 completed in the working tree: amortized inference now supports
  `standard_normal`, `supervised_checkpoint`, and `joint_realnvp` prior
  sources; frozen supervised priors have their gradients zeroed before the
  optimizer update; loaded supervised checkpoints validate latent names and
  bounds against the active amortized schema; new public configs cover the
  three Diffsky prior modes.
- PR 5 completed in the working tree: added
  `euclid_dsps.diffsky_redshift_ablation`, CLI command
  `diffsky-redshift-ablation`, per-run `photoz_metrics.csv` and
  `posterior_vs_truth_metrics.csv` outputs during inference, and ablation
  reports with bias, sigma MAD, RMSE, outlier fraction, PIT, coverage, and
  posterior-width metrics.
- PR 6 completed in the working tree: extended Diffsky population-realism
  diagnostics to include logSFR/logSSFR derived quantities when exported, dust
  terms when fitted, Diffstar/Diffmah/burst generated-truth marginals for
  supervised-prior diagnostics, q_agg/prior/truth plots, and
  `population_realism_report.md`. Raw `dlog10_sfr_i` ratios are intentionally
  not compared directly to `logsfr_true`.
- PR 7 completed in the working tree: added public configs and docs for the
  dataset, supervised priors, true-param closure, amortized prior-source modes,
  redshift ablation, and scientific validation plan. Older simple/fixed-z
  Diffsky configs remain documented as legacy/debug paths rather than the main
  physical-validation path.
- Validation completed in `conda run -n shine`: targeted Ruff on modified
  files, targeted PR3-PR7 pytest, `compileall euclid_dsps scripts`, CLI help,
  config-load smoke for seven public configs, Sphinx `-W --keep-going`, and
  full pytest (`318 passed, 4 skipped, 1 warning`). The skipped MCMC tests are
  due to an installed NumPyro/JAX incompatibility and now use the module's
  existing compatibility check.

## 2026-06-11 Diffsky Physical Validation PR 1/2

- Current implementation slice: split the new Diffsky/HLDTS validation path
  into PR 1 dataset-integrity/truth-semantics work and PR 2 supervised
  truth-prior learning. Keep both separate from photometric amortized
  inference and from true-parameter forward closure.
- PR 1 targets in progress: make prepared object ids globally unique while
  preserving `core_tag`, classify columns into truth/generated/derived/
  diagnostic/proxy/unavailable semantics, record the exact error model, and
  write a first-class dataset integrity report.
- PR 2 targets in progress: add an independent `prior_learning` package and
  CLI commands for supervised RealNVP density learning on truth parameters,
  with its own schema reduction/missing-column policy, logs, samples, summary,
  and truth-vs-prior diagnostics.
- PR 1 dataset-integrity slice completed in the working tree: prepared HLTDS
  parquet files now preserve `core_tag`, guarantee unique `object_id` values,
  add `global_object_id/source_file/source_row` when source ids are duplicated,
  classify columns by truth semantics in manifest/schema/reports, write
  explicit `error_model` provenance, and generate
  `diffsky_dataset_integrity_report.md`.
- PR 2 supervised-prior slice completed in the working tree: added
  `euclid_dsps.prior_learning`, configs for basic/extended supervised RealNVP
  priors, CLI commands `diffsky-train-supervised-prior`,
  `diffsky-sample-supervised-prior`, and `diffsky-supervised-prior-report`,
  plus toy/schema/output tests and documentation. This path uses truth columns
  directly and remains separate from photometric amortized inference.
- Validation completed in `conda run -n shine`: targeted Ruff, targeted
  pytest, full `compileall euclid_dsps scripts`, CLI help, config-load smoke
  for the two new prior configs, and Sphinx `-W --keep-going`.
- Added `PR3_PR7_TODO.md` as the remaining implementation checklist. Next
  priority is PR 3 same-parameter Diffsky forward closure; PR 4 should only
  start after PR 3 establishes whether truth parameters reproduce HLTDS
  photometry well enough for physical recovery claims.
- Updated `PR3_PR7_TODO.md` to align explicitly with the full prompt:
  A/B/C objective separation, PR dependency gates, command/config examples,
  expected test files, and global acceptance criteria.

## 2026-06-10 Prior-Learning Physical Audit

- Current audit: answer which parameters are actually fitted by the learned
  prior workflows, whether the compact Diffsky/PopCosmos setup is coherent
  with the DSPS forward model, which dataset columns are treated as truth, and
  which physical caveats remain before interpreting redshift inference under
  the learned degenerate prior.
- Audit outcome: Diffsky amortized prior learning fits a compact 9D
  PopCosmos-bin DSPS latent (`z_obs`, stellar mass, three SFR-ratio
  parameters, stellar metallicity, and three dust parameters) with
  `dlog10_sfr_4..6` fixed. FS2 keeps the full 16D PopCosmos-like latent. The
  learned RealNVP is a variational population prior in unconstrained latent
  space and inference currently samples the trained encoder approximation,
  not an exact reweighted/sampled posterior under a frozen learned prior. HLTDS
  truth columns are redshift, stellar mass, sSFR/SFR, halo/central/size fields;
  Diffstar/Diffmah/dust/burst columns are generated-truth diagnostics. Main
  caveats are decoder/data-model mismatch, no native HLTDS photometric errors,
  small truth tails outside configured bounds, and duplicated object IDs at
  shard boundaries.

## 2026-06-10 Diffsky HLTDS Amortized Prior Learning

- Objective validated: learn a joint degenerate population prior over redshift
  and compact DSPS physical parameters from HLTDS photometry, compare learned
  prior and aggregate posterior to direct/generative dataset distributions,
  then evaluate redshift recovery under that learned prior.
- Implemented public config
  `configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`.
  It uses HLTDS 04/14 LSST+Roman 14-band AB magnitudes, feature dimension 28,
  a 9D latent (`z_obs`, stellar mass, three SFH-ratio parameters, metallicity,
  and dust parameters), Gaussian encoder, RealNVP prior, fixed DSPS decoder,
  and the HLTDS SSP compressed basis asset.
- Compressed HLTDS SSP support validated. The builder now preserves physical
  axes at source precision while compressing the large basis/coefficient
  payload, avoiding wavelength-axis mismatch for the long HLTDS wavelength
  tail.
- The amortized data path is now generic in `B` and preserves large Diffsky
  `object_id` values as `int64` outside JAX, avoiding silent int32 truncation
  in posterior and validation outputs.
- The amortized decoder now merges compact latent free parameters with
  `model.fixed_parameters` before calling DSPS, so compact schemas can use the
  existing PopCosmos-bin forward model without exposing every nuisance
  parameter in the encoder.
- Added CLI paths `amortized-train-diffsky`,
  `amortized-infer-diffsky`, and `amortized-prior-overlap-diffsky`.
  The overlap report writes CSV/JSON/Markdown plus plots comparing direct truth
  to `q_agg` and the learned RealNVP prior for `z_obs` and
  `log10_stellar_mass`.
- Smoke validation completed on CPU with 8-object training, 4-object
  inference, and prior-overlap report. This validates code execution only; it
  is not a scientific-quality run.
- CUDA stability fix after first user GPU smoke: the Diffsky amortized config
  now upcasts compressed SSP arrays to resident `float32` and caps actual
  compiled DSPS/JAX batches with `amortized.training.jax_batch_size: 4` and
  `amortized.inference.jax_batch_size: 4`. User-facing `--batch-size` can stay
  larger; the command logs the internal cap.
- Scientific limit kept explicit: `logsfr_true` is not yet directly comparable
  to fitted `dlog10_sfr_*` ratios. A derived DSPS SFR diagnostic is still
  required before claiming SFR recovery.

## 2026-06-10 Public Pipeline Cleanup

- Public config surface reduced to:
  `configs/fs2_gpu.yaml`,
  `configs/diffsky_hltds_04_14_simple_gpu.yaml`,
  `configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml`, and
  `configs/amortized_fs2_realnvp.yaml`. The public surface now also includes
  `configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml` for HLTDS learned
  prior experiments.
- Deprecated public configs removed from `configs/`: old Diffstar variants,
  old OpenUniverse fit-ready experiments, old non-GPU Diffsky variants, and
  broad ablation configs.
- Main documented dataset is now
  `hltds_cosmos_260215_04_14_2026`, prepared as
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet`
  with `--no-synthetic-errors`.
- Main Diffsky fit model is deliberately simple: PopCosmos-bin DSPS, no AGN,
  fixed nebular SSP treatment, no Diffstar/Diffmah latent recovery, and direct
  comparison only to available basic truth columns (`redshift_true`,
  `logsm_true`, `logsfr_true`).
- Documentation reset completed for the public path: downloader, local data
  locations, Diffsky dataset contract, FS2 commands, GPU runtime, and
  amortized FS2 prior learning.

## 2026-06-08 Experimental SSP INR Compression

- Objective: investigate NeRF/implicit-neural-representation style compression
  for SSP HDF5 cubes as an experimental path, separate from the production
  dense and low-rank DSPS model paths.
- Scope for first implementation: add isolated training/evaluation/report CLI
  commands for direct coordinate MLPs and latent spectral-basis MLPs; compare
  against dense HDF5 payloads and existing low-rank compressed assets; write
  loss curves, reconstruction plots, metrics JSON/CSV, and Markdown reports.
- Integration policy: do not add a new `model.ssp_model` runtime mode until the
  asset-level benchmarks show a useful error/runtime/memory tradeoff.
- First implementation completed in the working tree:
  `euclid_dsps.experimental.ssp_inr` now provides HDF5 subset loading, direct
  Fourier/SIREN-style coordinate MLPs, latent spectral-basis MLPs, checkpoint
  serialization, evaluation against existing compressed assets, plots, CSV/JSON
  metrics, and Markdown reports.
- CLI commands added: `experimental-ssp-inr-train`,
  `experimental-ssp-inr-eval`, and `experimental-ssp-inr-report`.
- Validation completed with targeted smoke runs on the real Chabrier stellar
  SSP, `uv run pytest tests/test_experimental_ssp_inr.py -q`,
  `uv run ruff check euclid_dsps/experimental/ssp_inr euclid_dsps/cli.py
  tests/test_experimental_ssp_inr.py`, `uv run python -m compileall
  euclid_dsps scripts`, and `uv run python -m euclid_dsps.cli --help`.
- First benchmark review completed after running the quick/full SSP, gas, and
  AGN experiments. Direct coordinate MLPs are slower and less accurate than the
  latent path. The latent path is promising for stellar SSP in all-wavelength
  log metrics, but the existing low-rank assets remain much better on
  physically relevant wavelength/flux masks and are especially dominant for
  AGN. Next improvement should be metric design and a stronger latent residual
  architecture before any runtime integration.
- Second implementation pass in progress: add physically masked metrics,
  log-SVD baselines, end-to-end latent log-flux training, residual latent
  models over existing compressed assets, AGN `fagn` factorization, and zoomed
  reconstruction plots.
- Second implementation pass completed: `experimental-ssp-inr-eval` now writes
  `useful_wave`, `peak1em04`, `peak1em06`, and combined useful/significant
  metrics; supports repeatable `--log-svd-k`; and writes full, useful-wave, and
  UV/optical reconstruction plots plus masked error histograms. Latent training
  defaults to end-to-end Huber loss on weighted log-flux reconstruction, with
  optional coefficient loss, residual training over a compressed baseline, and
  AGN `fagn` factorization. Validation passed with targeted Ruff, targeted
  pytest, full `compileall`, CLI help checks, and a real-asset smoke run.
- Runtime follow-up: experimental SSP INR modules now apply the repository JAX
  runtime configuration before importing JAX and report a concise GPU backend
  installation error when `--runtime gpu` is requested but CUDA JAX is missing
  or version-mismatched.
- GPU run blocker diagnosed after the first full residual latent command:
  WSL sees an RTX 4060 Laptop GPU with driver 581.80 / CUDA 13.0, but the
  active `shine` environment reported an incompatible `jax_cuda13_plugin`
  version relative to `jaxlib`. The next full benchmark should only be rerun
  after reinstalling matching CUDA 13 JAX wheels in `shine`; CPU/auto commands
  remain usable for smoke/debug runs.
- Current phase: review the newly generated full SSP INR benchmark outputs,
  compare direct INR, latent residual, explicit log-SVD, and existing compressed
  baselines, then identify the next implementation improvements.
- Benchmark review outcome: for stellar SSP, `v2_stellar_residual_k32_full`
  is much better than the direct Fourier coordinate MLP and gives useful-wave
  p95 errors around a few percent, but it is still slower and less accurate
  than the existing low-rank compressed asset on useful/significant wavelengths.
  `v2_stellar_latent_k128_full` has lower training loss but worse useful-mask
  reconstruction, showing that the current loss is not aligned enough with the
  science metric. The direct coordinate model is currently disadvantaged by
  linear wavelength normalization over 100--9.95e7 Angstrom and should be
  retested with log-wavelength encoding plus masked/weighted sampling.
- Next priorities from the review: make evaluation plots/metrics mask-aware by
  default, add log-wavelength and useful-wave weighting for direct INR, add
  coefficient-table replacement experiments over existing low-rank bases,
  rerun high-dimensional gas/AGN with v2 objectives and factorization, and use
  actual photometry/filter wavelength coverage to define the reconstruction
  domain.
- Current implementation phase: add an experimental coefficient-MLP model that
  keeps an existing compressed spectral basis but replaces the coefficient table
  with a neural map from curve coordinates to coefficients; make comparison
  plots science-mask-first; add ECDF, per-wavelength, age/metal heatmap, and
  worst-spectrum plots for stronger benchmark diagnosis; prepare serious
  stellar/gas/AGN benchmark commands.
- Implementation completed: `compressed_coeff_mlp` is available through
  `experimental-ssp-inr-train`, using `--coeff-baseline`,
  `--coeff-loss={coeff,log_flux,mixed}`, `--coeff-log-weight`, and optional
  `--factor-agn-fagn`. Checkpoints store the kept basis plus coefficient
  standardization statistics and reconstruct spectra by predicting coefficients
  from curve coordinates. Evaluation now supports these checkpoints, uses
  useful/significant-wave error in the runtime-size plot, and writes ECDF,
  wavelength-profile, age/metal heatmap, and worst-case reconstruction plots.
  Validation passed with compileall, Ruff, targeted pytest, real stellar
  coefficient-MLP smoke train/eval, and real AGN factorized smoke train.

## 2026-06-05 OpenUniverse / Diffsky Validation Pivot

- Objective: change the validation priority from FS2-first to
  OpenUniverse/Diffsky-first, using LSST+Roman 14-band photometry as the first
  OpenUniverse target while keeping FS2 only as a domain-shift comparison and
  diagnostic dataset.
- Truth policy: label directly available OpenUniverse quantities as `truth`,
  future Diffsky/Diffstar exports as `generated_truth`, derived stand-ins as
  `proxy`, and absent quantities as `unavailable`. Do not describe FS2 catalog
  proxies as physical ground truth.
- First implementation slice (PR 1) in progress: add
  `euclid_dsps.openuniverse` schema, parquet IO/join, local/S3 path resolution,
  photon-rate unit contract, toy noise models, LSST+Roman subset preparation,
  `openuniverse-prepare` CLI, configs, docs, and synthetic unit tests. No bulk
  OpenUniverse download should be triggered by default.
- PR 1 completed in the working tree: mini OpenUniverse parquets can be joined
  and normalized into 14 truth fluxes, 14 noisy fluxes, 14 errors, and 14 masks;
  the CLI refuses implicit full-dataset processing; OpenUniverse units remain
  native photon-rate with explicit conversion TODOs; Diffsky extended truth
  extraction is a clear optional-dependency placeholder; toy photo-z and
  prior-overlap metrics are available for later reports; the amortized encoder
  now reads `input_dim` from config so B=14 feature tensors have shape `[N, 28]`.
- PR 1 validation completed with `uv run python -m compileall euclid_dsps scripts`,
  targeted Ruff, Sphinx `-W --keep-going`, targeted pytest (`75 passed`), and
  full pytest (`255 passed, 5 skipped`). Remaining warning: extended Diffsky
  truths are unavailable unless an optional Diffsky export path is provided.
- 2026-06-08 EDA completed for downloaded OpenUniverse preview hpix 10307:
  raw main/flux files each have 3,561,877 unique galaxy ids and join exactly;
  prepared subset has 10,000 rows, 65 columns, all 14 LSST+Roman masks valid,
  median subset redshift 0.983, and median log10 stellar mass 7.999. Report
  written under `outputs/reports/openuniverse_eda_10307/` with schema catalog,
  physical/flux stats, QA tables, and first diagnostic plots. Caveat: the
  10,000-row subset is a head/limit preview sample and not representative.
- 2026-06-08 PR 2/3 slice implemented on
  `feature/openuniverse-truth-sed-validation`: standalone
  `python -m euclid_dsps.openuniverse.cli` commands for truth inventory, basic
  truth export, photo-z metrics, prior-overlap metrics, and OpenUniverse
  B=14 feature stats; bounded SED HDF5 inventory/reader for
  `/meta/wave_list` plus `/galaxy/<prefix>/<galaxy_id>` datasets; prepared
  parquet loader into generic `PhotometryArrays`; and synthetic tests. Real
  hpix 10307 inventory found 312 wavelength bins, 3,561,877 SED datasets shaped
  `(3, 312)`, direct truth for redshift/redshiftHubble/stellar_mass/fluxes,
  generated-truth low-resolution SED, proxy MW/lensing columns, and unavailable
  SFH/Diffstar/dust-internal/metallicity/halo latents.
- 2026-06-08 data-first photon closure slice implemented without touching the
  DSPS decoder: OpenUniverse filter loading, SED Fnu to integrated photon-rate
  helpers, `sed-flux-closure` CLI, per-band robust calibration factors, and
  `merge-external-truth` for explicit future Diffsky latent tables. Real
  hpix 10307 LSST-only closure over 200 galaxies used exact repository LSST
  filters and found calibrated median residuals near zero, sigma_MAD about
  2.4--3.0%, and p95 absolute calibrated residuals about 6--8%. Roman closure
  remains blocked on exact Roman filter curves unless top-hat smoke filters are
  explicitly enabled.
- 2026-06-09 fit-ready OpenUniverse slice implemented data-side, without
  changing the DSPS decoder: `make-fit-ready` computes `mu_lensing` from
  convergence/shear, preserves public photon-rate fluxes in explicit lensed
  audit columns, writes unlensed photon columns, and converts the standard
  `flux_*`/`fluxerr_*` columns to DSPS-compatible `fnu_cgs`. The conversion
  uses per-band AB0 photon rates and defaults to `filter_response_mode:
  dsps_clipped` so Roman effective-area-like filters match
  `euclid_dsps.filters.load_ascii_filter`. Local hpix 10307 validation wrote
  `Data/openuniverse/processed/ou_lsst_roman_14_subset_fit_ready.parquet`,
  confirmed B=14 feature stats with feature_dim=28, and ran a one-object MAP
  smoke fit. Caveat: Roman zeropoints/response conventions still need deeper
  validation before MAP parameters should be interpreted scientifically.
- Follow-up slices: optional full Diffsky latent export, physical
  OpenUniverse train/infer commands after photon-rate decoder/conversion,
  redshift ablation on real posteriors, prior-overlap plots from model outputs,
  and FS2-vs-OpenUniverse comparison plots.

## 2026-06-02 Andrew Hearin Discussion Preparation

- Objective: prepare a didactic discussion package for a meeting with Andrew
  Hearin about the current DSPS/PopCosmos-like Euclid+LSST forward-modeling
  work.
- Deliverables completed under
  `outputs/report/andrew_hearin_discussion_2026-06-02/`: one explanatory
  Markdown document with Mermaid architecture diagrams, a RevealJS slide deck,
  a concise question list, and a metrics summary grounded in existing
  run/benchmark artifacts.
- Slide refresh completed: replaced the first pipeline diagram and SED
  construction diagram with static HTML diagrams, added detailed slides on
  model ingredients, `model.py`'s DSPS adapter role, active free parameters, and
  all current free-parameter bounds.
- Follow-up slide refresh completed after review: slide 4 now uses a vertical
  detail stack for physical ingredients and code changes relative to base DSPS;
  the dust slide now explains the old-star versus young-star attenuation model
  with equations and code mapping; the `model.py` slide now has vertical
  details for context loading, forward pass, mass normalization, and diagnostic
  versus likelihood paths.
- Added generated component figures from the real JAX model components and
  slides explaining emission-line construction, AGN component construction,
  AGN/IGM ordering, flux-space Student-t errors, and surviving-mass
  normalization.
- Second visual refresh completed: replaced the main component plot with
  `sed_component_build_row471_detail.png` because row 471 has visible AGN/IGM
  effects; added `emission_lines_before_after_row471.png`,
  `agn_igm_detail_row471.png`, and `igm_before_after_transmission.png` to show
  emission-line before/after, AGN ratios, and IGM transmission directly.
- Updated the AGN optical-depth bound explanation: `ln_tauagn` bounds map to
  tau 5.0 to 148.0, matching the PopCosmos/Prospector hard-limit convention and
  the FSPS tabulated `agn_tau` range before extrapolation.
- Rewrote the final Andrew questions to focus on decisions: minimal per-galaxy
  model, gas/line corrections, AGN/dust/IGM conventions, likelihood/model
  mismatch, and population-prior strategy.
- Amortized learned-prior slide update completed: added a vertical RevealJS
  stack for the FS2-only Gaussian encoder + RealNVP prior + fixed DSPS decoder
  prototype, with Mermaid architecture/output diagrams, ELBO details, caveats,
  and dedicated Andrew questions.
- Source policy: use existing code, configs, `PLAN.md`, documentation, and
  outputs as evidence; do not rerun expensive MAP/posterior jobs for this
  communication package unless explicitly needed.
- Main talk caveats to keep explicit: the current model is FSPS/Prospector-like
  rather than an official PopCosmos reproduction; PopCosmos learned emission-line
  corrections are not included; existing MAP SED diagnostics did not load the
  COSMOS proxy columns even though the FS2 parquet contains them; full-AGN MAP
  is under-constrained from 10 bands and shows redshift, gas, and AGN
  degeneracies.

## 2026-06-02 PopCosmos Learned Prior Preparation

- Branch: `feature/popcosmos-prior-learning`, created from the current
  `feature/mclmc-posterior` state.
- Feature objective: prepare for learning a POP-COSMOS-style prior from the
  existing PopCosmos/FSPS compressed workflow outputs and catalog-derived
  parameter distributions.
- Initial cleanup policy: keep scientific inputs in `Data/` and run products
  in `outputs/` untouched. Only remove local Python/test/tool caches during
  branch preparation.
- Preparation completed: removed local `__pycache__`, `.pytest_cache`, and
  `.ruff_cache` directories from source/test areas; deliberately did not run
  broad `git clean` because it would remove local `Data/` assets and
  `outputs/` products.
- Detailed implementation planning moved into local running plan
  `posterior_plan.md` after the updated prompt. This file is intentionally not
  for commit; it tracks PR 0 through PR 5, API contracts, tests, docs, and open
  implementation decisions for the FS2-only amortized posterior feature.
- Implementation breakdown:
  - PR 0: public `parameter_vectors.py`, array photometry bridge, and
    architecture docs.
  - PR 1: `amortized/latent.py`, `features.py`, and `likelihood.py`.
  - PR 2: Equinox Gaussian encoder, RealNVP prior, and optional dependency
    extra.
  - PR 3: DSPS decoder wrapper, Monte Carlo ELBO, and asset-free synthetic
    smoke.
  - PR 4: FS2 dataloader, joint training loop, config, and
    `amortized-train-fs2` CLI.
  - PR 5: inference, catalog export, diagnostics, and
    `amortized-infer-fs2` CLI.
- Implementation completed in the working tree:
  - Added public theta-vector helpers and array photometry extraction.
  - Added `euclid_dsps.amortized` with latent transforms, features,
    Student-t likelihood, Equinox encoder, RealNVP prior, decoder, ELBO,
    synthetic smoke, FS2 training, inference, catalog export, and diagnostics.
  - Added `configs/amortized_fs2_realnvp.yaml`, optional dependency extra
    `amortized`, CLI commands, tests, README/Sphinx docs, and lockfile update.
  - Runtime follow-up: a real DSPS inference run with
    `posterior_samples=32` and `batch_size=8` exhausted GPU memory because all
    posterior samples were decoded in one DSPS vmap. Inference now chunks the
    posterior predictive decode over the sample axis, defaults
    `decoder_sample_chunk_size` to 1, exposes
    `--decoder-sample-chunk-size`, and records the chunk size in
    `inference_summary.json`.
  - Follow-up from the first chunked inference run: posterior predictive files
    were finite, but bright low-redshift FS2 objects produced extreme linear
    encoder features and poor flux predictions. New feature stats now default
    to `flux_transform: asinh` while legacy stats without a transform remain
    linear for checkpoint compatibility. Inference now writes normalized
    posterior predictive residual tables, top chi-square objects, feature-scale
    diagnostics, redshift proxy comparisons when available, and diagnostic
    plots including a compact posterior corner plot.
  - Next diagnostic addition requested: treat the RealNVP as a learned prior
    product, not only a KL term. Inference should sample `p_beta(x)`, export
    learned-prior samples in theta space, compare learned prior versus aggregate
    amortized posterior with true corner contours, write redshift distribution
    comparisons, and add a redshift PIT diagnostic when FS2 redshift proxy
    columns are present.
  - Implemented learned-prior inference diagnostics: `learned_prior_samples`
    now stores both latent `x_00` ... `x_15` and physical `theta` samples with
    exact `logprior`; diagnostics write learned-prior quantiles, a log-density
    histogram, contour-style learned-prior and posterior-vs-prior corners,
    redshift distribution comparison, and redshift PIT tables/plots.
  - Added explicit FS2 catalog proxy diagnostics for the amortized outputs:
    `catalog_proxy_comparison.parquet/csv`, stellar-mass posterior/prior/proxy
    histograms, stellar-mass proxy residual histogram, catalog SFR proxy
    distribution, and proxy mass-SFR plane. These are deliberately labeled as
    catalog proxies; SFR is not overlaid against posterior samples until a
    model-derived `log10_sfr_at_obs` export is added.
  - Added anti-collapse training controls for Jean Zay/H100 runs: reproducible
    FS2 row selection with `sequential`, `random`, or `stratified_redshift`
    modes; balanced/proportional redshift stratification; epoch-level training
    shuffle; train/validation split artifacts; validation ELBO rows in
    `training_log.csv`; `validation_redshift_bin_metrics.csv`; validation loss
    plots by redshift bin; validation-based `best.eqx` checkpointing when a
    validation split exists; and explicit `kl_weight_max` plus longer default
    KL annealing.
  - Cleaned training history diagnostics after the first H100 run: raw
    `training_log.csv` remains batch-level, but plots are now recomputed from
    `training_epoch_summary.csv` using `epoch` on the x-axis, train/validation
    means with p16/p84 bands, a compact overview figure, and redshift-bin
    validation heatmaps for loss, negative loglike, chi-square, and KL.
  - Validation passed with `uv sync`, `uv sync --extra dev`,
    `uv sync --extra dev --extra amortized`,
    `uv run python -m compileall euclid_dsps scripts`, Black on the feature
    Python files, targeted Ruff on the feature diff,
    `uv run pytest tests -q` with amortized extras installed
    (`226 passed, 5 skipped`), Sphinx `-W --keep-going`,
    `uv run python -m euclid_dsps.cli --help`, and the mock-decoder
    `amortized-synthetic-smoke` CLI.
  - Equinox/Optax remain optional dependencies; CLI commands report the missing
    dependency clearly if the `amortized` extra is not installed.
- 2026-06-02 architecture and `conda shine` recheck:
  - Static audit confirms the amortized implementation stays FS2-only, does not
    import private `fit.py` helpers, does not call `predict_batch_mags`, does
    not construct `GalaxyObservation` in the training path, and keeps Pandas
    use to IO/reporting surfaces outside the hot JAX decoder/ELBO path.
  - `conda run -n shine python -c "import jax, equinox, optax"` passed with
    JAX `0.10.1`, Equinox `0.13.7`, and Optax `0.2.8`. In this shell, JAX
    falls back to CPU because the installed CUDA plugin is incompatible with
    the installed `jaxlib`; real FS2 DSPS training should be run after fixing
    the `shine` CUDA/JAX stack if GPU is required.
  - `conda run -n shine python -m euclid_dsps.cli --help` passed and shows the
    three amortized commands without breaking the existing `check`, `fit`, and
    `posterior` command registration.
  - `conda run -n shine python -m compileall euclid_dsps scripts` passed.
  - `conda run -n shine python -m pytest tests/test_amortized_encoder.py
    tests/test_amortized_flows.py tests/test_amortized_elbo.py
    tests/test_amortized_synthetic.py -q` passed: 6 passed.
  - `conda run -n shine python -m pytest tests/test_amortized_synthetic.py -q`
    passed after tightening the synthetic summary criterion.
  - `conda run -n shine python -m euclid_dsps.cli --config
    configs/amortized_fs2_realnvp.yaml amortized-synthetic-smoke
    --mock-decoder --n-objects 64 --epochs 2 --batch-size 8 --out
    outputs/runs/dev_amortized_synthetic_prompt_check_shine` passed and wrote
    `training_log.csv`, `training_summary.json`, `feature_stats.json`, best/last
    checkpoints, and diagnostics. The summary now reports
    `loss_decreased: true` using the best observed loss while retaining
    `last_loss_decreased` for the final-mini-batch comparison.
- 2026-06-02 DSPS e2e and progressive training outputs:
  - Added `tests/test_amortized_dsps_e2e.py`, which builds a tiny synthetic
    ten-filter PopCosmos-like `DspsContext`, runs the non-mock
    `model_flux_from_x` decoder through `model_mags_jax_dynamic`, evaluates the
    ELBO, and verifies nonzero encoder and RealNVP prior gradients from the
    same `eqx.filter_value_and_grad` call.
  - Training and synthetic smoke now write `training_progress.json`, update
    `checkpoints/last.eqx` during training, write epoch checkpoints such as
    `checkpoints/epoch_0001.eqx`, and regenerate diagnostics every
    `amortized.output.diagnostics_every` epochs.
  - `training_log.csv` now includes `encoder_grad_norm`, `prior_grad_norm`, and
    `joint_grad_norm`; the diagnostics writer plots those columns to make joint
    encoder/RealNVP optimization visible.
  - Checkpoint JSON sidecars now include an architecture summary covering the
    Gaussian MLP encoder, RealNVP prior, fixed DSPS decoder, and Monte Carlo
    `logq - logp` KL objective.
  - Validation passed:
    `uv run pytest tests/test_parameter_vectors.py tests/test_amortized_latent.py
    tests/test_amortized_features.py tests/test_amortized_likelihood.py
    tests/test_amortized_encoder.py tests/test_amortized_flows.py
    tests/test_amortized_elbo.py tests/test_amortized_synthetic.py
    tests/test_amortized_dsps_e2e.py -q` (`20 passed`),
    `uv run python -m sphinx -W --keep-going -b html docs/source
    /tmp/dsps_docs_amortized_arch_check`, and
    `conda run -n shine python -m pytest tests/test_amortized_dsps_e2e.py -q`
    (`1 passed`, with the known JAX CUDA plugin warning).
  - `conda run -n shine python -m euclid_dsps.cli --config
    configs/amortized_fs2_realnvp.yaml amortized-synthetic-smoke
    --mock-decoder --n-objects 32 --epochs 2 --batch-size 8 --out
    outputs/runs/dev_amortized_synthetic_progress_check_shine` passed and wrote
    epoch checkpoints plus `encoder_grad_norm.png`, `prior_grad_norm.png`, and
    `joint_grad_norm.png`.
  - Full test suite passed after the follow-up:
    `uv run pytest tests -q` (`227 passed, 5 skipped, 1 warning`).
    `git diff --check` passed. Full-repo Ruff currently reports an unrelated
    pre-existing `B007` unused loop variable in `tests/test_fsps_grid_scripts.py`;
    targeted Ruff on the amortized files passed.
  - Added default verbose console logging and per-epoch progress bars for
    `amortized-train-fs2`. The command now reports setup stages, JAX
    backend/devices, DSPS loading, architecture summary, epoch starts, and
    epoch summaries. The progress bar displays live loss, NLL, MC KL, encoder
    gradient norm, and RealNVP prior gradient norm. CLI flags `--quiet` and
    `--no-progress` disable these outputs when needed.
  - Validation for the verbose/progress follow-up passed:
    `uv run ruff check euclid_dsps/amortized/train.py euclid_dsps/cli.py`,
    `uv run python -m compileall euclid_dsps scripts`,
    `uv run python -m sphinx -W --keep-going -b html docs/source
    /tmp/dsps_docs_verbose_check`, `git diff --check`, and
    `conda run -n shine python -m euclid_dsps.cli --config
    configs/amortized_fs2_realnvp.yaml amortized-train-fs2 --help`.
  - Fixed the first real FS2 amortized-training NaN failure:
    flux-space likelihood now evaluates in stop-gradient normalized flux units
    to avoid float32 underflow of cgs variances near `1e-30`; encoder features
    now use scale-relative floors instead of an absolute `1e-8`; the FS2
    encoder initializes near a stable PopCosmos theta point (`z_obs` around
    0.8 rather than the midpoint of the broad redshift bound); and the training
    loop skips non-finite updates instead of applying corrupt gradients.
  - Added `loss_finite`, `grads_finite`, and `update_applied` columns to
    `training_log.csv`, plus `updates_applied`/`updates_skipped` in progress
    and summary JSON.
  - NaN-fix validation passed:
    first-batch inspection reports finite loglike and finite encoder/RealNVP
    gradients; `UV_CACHE_DIR=/tmp/uv-cache uv run pytest
    tests/test_amortized_features.py tests/test_amortized_likelihood.py
    tests/test_amortized_dsps_e2e.py tests/test_amortized_synthetic.py -q`
    passed (`8 passed`); `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall
    euclid_dsps scripts` passed; `git diff --check` passed; and a direct real
    FS2 mini-run with `limit=16`, `batch_size=8`, `epochs=2` applied 4/4
    finite updates with no NaNs.
- Remaining before production use: run the real `amortized-train-fs2` smoke in
  `shine` once the FS2/DSPS assets and desired CUDA-capable JAX stack are
  active; this was not run during the architecture audit.

## Current State

2026-06-01 MCLMC implementation phase:

- Branch `feature/mclmc-posterior` starts from committed MAP recovery fixes.
- Target implementation: single-galaxy 16D posterior sampling for compressed
  PopCosmos using experimental BlackJAX MCLMC while leaving NumPyro HMC/NUTS
  untouched.
- Implemented contract: `sample.sampler: mclmc` routes through a pure-JAX
  bounded posterior target over an unconstrained vector, with log-Jacobian,
  prior terms, existing Gaussian/Student-t distribution log-probs, and the
  PopCosmos gas metallicity constraint.
- Current limitation: first MCLMC backend is unadjusted MCLMC with configured
  or automatic `L` and `step_size`; it is an engineering diagnostic sampler
  until compared against HMC/NUTS on selected rows.
- Debug/progress update: the MCLMC path now reports JAX backend/devices,
  compile/warmup/sample phase timings, optional per-chunk diagnostics, and
  chunked progress bars controlled by `sample.mclmc_progress_chunk_size`.
- Stability update: the BlackJAX target now reuses the existing chi-based
  Student-t objective up to an additive constant, uses a triangular transform
  for the free `log10_stellar_metallicity`/`log10_gas_metallicity` pair, and
  avoids nesting a separately jitted MCLMC step inside `lax.scan`.
- Multi-chain update: MCLMC now supports `sample.num_chains` through sequential
  chains to keep VRAM bounded. Posterior predictive magnitudes are written in
  configurable chunks, chain metadata is exported, trace plots mark chain
  boundaries, and posterior summaries/corner truth plots include catalog
  redshift truth as `truth_z_obs`.
- Initialization update: MCLMC supports `sample.init_strategy` values `map`,
  `map_jitter`, `config`, and `random_uniform`. `map_jitter` is the preferred
  first diagnostic for multi-chain runs because it keeps chains near a good MAP
  mode but gives each chain a distinct unconstrained starting point.
- Guardrail update: MCLMC now raises a clear error if a warmup or sampling phase
  returns zero valid transitions (`nonans=0` everywhere), preventing frozen MAP
  repeats from being written as apparent posterior samples.

2026-06-01 large MAP finalization fix:

- Observed failure: a `configs/popcosmos_binned_compressed.yaml` MAP run with
  `--limit 10000 --batch-size 128` completed 78 full chunks, then crashed at
  9984/10000 when the final 16-row chunk triggered a new JAX/XLA GPU graph
  capture path (`cuda_blas_lt.cc` workspace failure).
- Fix: independent batch MAP now pads partial chunks internally to the requested
  static `batch_size`, then filters synthetic rows before checkpoint/final
  outputs. This keeps one stable JAX batch shape for long production runs. The
  padding is not applied to population/hierarchical fits because duplicated
  rows would change the statistical objective.
- Recovery tool: `scripts/finalize_fit_from_chunks.py` concatenates existing
  `_chunks` checkpoints and regenerates aggregate MAP tables, QC plots, and
  completion metadata without rerunning the optimizer. For the interrupted
  n=10000 run, it can produce a scientifically usable report over the 9984
  completed galaxies and explicitly records the missing row indices.

2026-05-30 speed optimization phase:

- Goal: improve PopCosmos compressed MAP throughput and GPU batch capacity for
  learned-prior/posterior production.
- Current measured bottleneck: `jax_optimize` dominates runtime on
  `configs/popcosmos_binned_compressed.yaml`; warm chunks take about 75 s for
  64 galaxies at 200 Adam iterations.
- Implementation target: remove per-iteration diagnostic forward passes from
  the MAP loop when configured, and use a likelihood-only PopCosmos
  `model_mags` path that avoids constructing diagnostic SED components during
  optimization.
- Validation target: `model_mags` fast path must match the full forward model
  magnitudes, and the default dense/science configs must remain compatible.
- Implemented slice: `fit.trace_mode` and `fit.trace_interval` now control
  expensive Adam trace diagnostics. `optimizer` mode records optimizer loss and
  gradient summaries without per-iteration photometric diagnostic forward
  passes; `none` disables trace rows. The compressed PopCosmos config defaults
  to `optimizer` every 20 iterations.
- Implemented slice: PopCosmos binned and Diffstar reduced6 now have
  likelihood-only JAX magnitude paths. MAP, population MAP, MCMC likelihoods,
  and benchmark code that ask only for model magnitudes no longer build the
  diagnostic SED result object.
- Implemented slice: population MAP uses the analytic bounded-transform chain
  rule instead of evaluating a second full loss gradient each Adam step.
- Smoke result on the local RTX 4060 Laptop GPU with
  `configs/popcosmos_binned_compressed.yaml`, `batch-size=64`, and 200 Adam
  iterations: the optimized path completed 128 galaxies in 188.9 s with a
  2.44 GiB GPU peak. The previous 512-galaxy compressed run used 4.66 GiB peak
  and 1.32 s/gal overall; warm-chunk runtime is now lower and peak GPU memory is
  roughly halved for the same compressed model family.

2026-05-31 forward/runtime optimization phase:

- Goal: continue reducing GPU memory pressure and wall time for production MAP
  batches used before learned posterior/prior training.
- New target: remove avoidable `age x wavelength` intermediates from
  likelihood-only forward passes. The current dust models apply one attenuation
  curve to old populations and one to young populations, so the fast path can
  sum young/old SSP contributions before applying dust while preserving the
  full diagnostic forward model unchanged.
- Benchmarking policy: save every smoke/runtime run under `outputs/runs/` and
  write a progress report under `outputs/report/` so the speed progression can
  be shown externally.
- Result: the young/old dust-sum rewrite and a hand-decomposed photometry
  rewrite were both rejected after GPU benchmarks because they increased XLA
  memory and runtime. They are documented as negative results, not kept in the
  model code.
- Kept optimization: the GPU runtime preset now sets
  `TF_GPU_ALLOCATOR=cuda_malloc_async`, which removed the large allocator/autotune
  memory spike on the RTX 4060 Laptop GPU.
- Kept optimization: `--reporting-level light` now actually skips plot-heavy
  batch reports while keeping tables, JSON summaries, and performance
  benchmarks.
- Current best production setting on the local GPU: compressed full-AGN binned
  model with `batch-size=128`, `maxiter=200`, `sed-samples=0`, and
  `reporting-level=light`. A two-chunk smoke run processed 256 galaxies in
  175.35 s with a 4.49 GiB GPU peak; the warm optimization chunk took 63.83 s
  for 128 galaxies, roughly twice the warm throughput of the previous
  `batch-size=64` run.

2026-05-31 JAX compilation/options investigation:

- Goal: test whether JAX writing/compilation knobs can improve the compressed
  PopCosmos MAP path beyond simply running on GPU.
- Implemented reproducible fit options for controlled benchmarking:
  `fit.scan_unroll`, `fit.donate_optimizer_inputs`, and
  `fit.remat_model_mags`, exposed through the CLI as `--fit-scan-unroll`,
  `--fit-donate-inputs`, and `--fit-remat-model-mags`.
- Result: all three options stay disabled by default. On the local RTX 4060
  Laptop GPU, `scan_unroll=2` gave no warm-speed improvement and much higher
  compile/memory pressure; input donation produced JAX donation warnings and
  was slower; rematerializing the magnitude forward was slower and did not
  reduce the real graph memory.
- Kept recommendation: use the compressed config defaults with the GPU runtime
  preset, which now sets `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Added `scripts/check_photometry_equivalence.py` to compare the full
  diagnostic forward path against the optimized likelihood-only magnitude path.
  The current compressed PopCosmos smoke check reports exactly zero magnitude
  difference for the sampled points.
- Report and plots:
  `outputs/report/jax_compilation_investigation_2026-05-31/`.

2026-05-31 advanced optimizer/MCMC investigation:

- Tested a deeper JAX autodiff rewrite for independent MAP batches:
  `vmap(value_and_grad(single_objective))` versus
  `value_and_grad(sum(vmap(single_objective)))`. The latter is exposed as
  `fit.batch_grad_mode: sum` and CLI `--fit-batch-grad-mode sum` for future
  hardware benchmarks.
- Result: rejected as production default on the local RTX 4060 Laptop GPU.
  Warm optimization time was essentially unchanged, while compile time and peak
  GPU memory roughly doubled. Keep `fit.batch_grad_mode: per_galaxy`.
- MCMC path now calls `model_mags_jax_dynamic` so large compressed DSPS arrays
  are passed as model arguments before future sampler experiments.
- Local `shine` has NumPyro but not `blackjax`, `jaxopt`, `optax`, or `flowMC`.
  MCLMC therefore requires a separate dependency/backend slice rather than a
  config-only switch.
- A compressed full-model HMC smoke with one warmup, one sample, and one
  leapfrog was terminated after more than two minutes of CPU-heavy compilation.
  Posterior acceleration should be benchmarked separately with a sampler
  backend designed for this use case.
- Report:
  `outputs/report/jax_advanced_optimization_2026-05-31/`.

2026-05-31 wavelet compression benchmark:

- Added `scripts/benchmark_wavelet_compression.py` to compare the current
  low-rank compressed assets against a sparse Haar-wavelet oracle on sampled
  dense stellar SSP, gas, and AGN spectra.
- The benchmark intentionally measures resident payload size and spectral
  reconstruction loss, not disk compression. It does not add a runtime wavelet
  mode yet.
- Result: keep the current SVD/factored-SVD compression as the production
  direction for gas and AGN. Sparse Haar is much worse for AGN and larger or
  worse for gas at the tested loss levels. It is only competitive for the base
  stellar SSP on the sampled median spectral metric, but its worst-tail error
  and sparse-index runtime complexity need a separate photometric/JAX benchmark
  before it can be considered.
- Report and plots:
  `outputs/report/wavelet_compression_benchmark_2026-05-31/`.

2026-05-31 compressed production documentation/config update:

- Promoted `configs/popcosmos_binned_compressed.yaml` as the documented
  production MAP config and added `configs/popcosmos_diffstar_compressed.yaml`
  for the compressed Diffstar comparison path.
- Added `docs/source/ssp_compression.rst` and linked it from the Sphinx index.
  The page documents the SVD `basis/coeff/scale` format, gas and AGN compressed
  parameter axes, the wavelet audit result, asset build commands, and the
  production compressed fit command.
- Updated README and Sphinx run/setup/forward-model/science/testing docs so
  dense configs are described as reference/closure paths while compressed
  configs are the recommended high-throughput runtime path.
- Validation run before commit: `compileall`, `pytest tests/test_config.py`,
  `pytest tests/test_model.py`, `pytest tests/test_fit_memory.py
  tests/test_mcmc.py`, Sphinx `-W --keep-going`, and compressed photometry
  equivalence smoke all pass locally in `conda shine`.

2026-05-31 MCLMC posterior planning:

- Added `PLAN_MCLMC.md` as the implementation plan for a BlackJAX MCLMC
  posterior backend.
- Current decision: do not put posterior sampling on the critical path for the
  next MAP sprint. First run compressed MAP `n=10k`, generate QC, test SSP
  `k128`, rerun dense-vs-compressed `n=500`, then scale to `n=100k+`.
- MCLMC should be developed in parallel as an experimental posterior backend
  over the compressed model. It needs a pure JAX log-density over an
  unconstrained parameter vector, a bounded-parameter transform with Jacobian,
  lazy BlackJAX dependency handling, and a benchmark against current NumPyro
  HMC/NUTS on selected MAP-QC rows.
- Use `mclmc` as the canonical spelling in config/code/docs. Treat unadjusted
  MCLMC as an engineering/diagnostic sampler until compared against NUTS/HMC;
  add adjusted MCLMC as the science candidate if the installed BlackJAX API
  supports it.

Branch objective: integrate the Diffstar SFH implementation from
`feature/diffstar` into the current PopCosmos FSPS gas/AGN workflow without
losing the generated-grid scripts, GPU/JIT memory fixes, or documentation
cleanup.

The repository currently has six active PopCosmos-like configurations:

```text
configs/popcosmos_binned.yaml
configs/popcosmos_binned_compressed.yaml
configs/popcosmos_diffstar.yaml
configs/popcosmos_diffstar_compressed.yaml
configs/popcosmos_binned_noagn.yaml
configs/popcosmos_diffstar_noagn.yaml
```

The compressed full AGN configs are now the recommended production path:
`popcosmos_binned_compressed` first for high-throughput GPU MAP batches, and
`popcosmos_diffstar_compressed` as the comparison. The dense full AGN configs
remain available for dense-vs-compressed and FSPS/Prospector closure checks.
The no-AGN configs remain available for controlled ablations and fallback
debugging.

They are standalone and represent the current PopCosmos-like DSPS/JAX paths:

- LSST `ugrizy` plus Euclid VIS/Y/J/H photometry.
- Flux-space likelihood with catalog per-band flux errors.
- Configurable photometric objective, with POP-COSMOS-style Student-t as the
  current default for both full AGN and no-AGN PopCosmos-like configs.
- Seven-bin PopCosmos-like SFH ratios in `popcosmos_binned.yaml`.
- Six-free-parameter Diffstar SFH in `popcosmos_diffstar.yaml`.
- Single stellar metallicity.
- Explicit age-dependent dust modes:
  `charlot_fall_powerlaw` preserves the previous behavior, while
  `prospector_fsps` is the current approximate Prospector/FSPS-like target mode.
- Madau95-style approximate IGM.
- PopCosmos-like assets must be explicit Chabrier assets with HDF5 metadata:
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5`,
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5`,
  `Data/popcosmos_chabrier_gas_ssp_grid.h5`, and
  `Data/popcosmos_chabrier_agn_component_ssp_grid.h5`.
- The first one-row `popcosmos_binned_noagn` smoke fit passes after generating
  those assets, with `EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0` and
  `JAX_PLATFORMS=cuda` in the `shine` environment.
- PopCosmos-like `z_sun` is `0.0142`; metallicity conversion is
  `log10(Zstar_abs) = log10(0.0142) + log10_stellar_metallicity`.
- Full configs keep the 16-parameter fit including SFH, gas, and AGN.
  No-AGN configs remove `ln_fagn` and `ln_tauagn`.
- `configs/popcosmos_binned_compressed.yaml` and
  `configs/popcosmos_diffstar_compressed.yaml` inherit their dense full AGN
  configs and swap only the resident spectral assets to compressed SSP, gas,
  and AGN component bases.

Older config variants, presets, examples, and partial smoke configs have been
removed from source. Tests now target the active configs plus low-level
synthetic fixtures.

## Scientific Caveats

- This is PopCosmos-like, not yet an audited reproduction of the full
  POP-COSMOS population model.
- Independent gas and stellar metallicity variation in FSPS is useful for
  fitting but not fully self-consistent for all nebular line ratios.
- A hard PopCosmos-like constraint now enforces
  `log10_gas_metallicity >= log10_stellar_metallicity`, with both quantities in
  `log10(Z/Zsun)`.
- `model.emission_line_corrections: none` means the gas grid is
  `uncalibrated_cloudy`. This remains usable for development but is non-final
  for a science prior. `popcosmos_table` support exists for enriched grids with
  `nebular_continuum_flux`, `emline_wavelengths`, and `line_flux_grid`.
- `prospector_fsps` dust is a JAX approximation to the FSPS/Prospector
  `dust_type=4` plus birth-cloud behavior. It uses `dust2=tau2`,
  `dust1=tau1_over_tau2*tau2`, `dust_tesc_logyr=7.0`, and
  `dust1_index=-1.0`, but still needs direct FSPS/Prospector benchmarking before
  it can be called an exact reproduction.
- Legacy `agn_model: template_grid` still follows the repository convention
  `fagn * integrated stellar Lbol` and remains an audit-only approximation.
  The active `agn_model: fsps_component_grid` path instead loads an FSPS-native
  AGN component SSP grid and convolves it with the same SFH weights as the
  stellar SSP.
- `agn_host_attenuation: fsps_diffuse_unit_tau` is the current FSPS-aligned
  AGN host attenuation convention: it applies the PopCosmos/Prospector-like
  diffuse `dust_type=4` attenuation curve at unit V-band optical depth to the
  AGN component, matching the local FSPS `agn_dust.f90` convention instead of
  multiplying by `dust2`. The older `diffuse` and `prospector_fsps` scaled AGN
  modes remain diagnostics only.
- The IGM model is a stable Madau95-style approximation.
- The fit and post-fit batch prediction paths pass large SSP/gas/AGN arrays as
  dynamic JAX arguments so JIT can stay enabled without compiling the gas grid
  as a multi-GiB XLA constant.
- Gas-grid interpolation uses direct four-corner bilinear interpolation in
  `(gas_lgmet, gas_lgu)`, avoiding a batch-scaled intermediate over the full
  gas-ionization axis.

## Standard Commands

Generate/validate the FSPS Chabrier SSP, gas, and AGN assets as documented in
`docs/source/data_download.rst`. The Chabrier SSP is generated locally with
`scripts/generate_fsps_ssp_grid.py`; it is not a legacy `manage_ssp.py`
download.

Short one-row fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_compressed_fullagn_one_short \
  --sed-samples 1
```

Batched fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  fit --limit 1000 \
  --batch-size 128 \
  --fit-maxiter 200 \
  --out outputs/runs/popcosmos_binned_compressed_map_n1000_bs128 \
  --sed-samples 0 \
  --reporting-level light
```

Diffstar short one-row fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_diffstar_compressed.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_diffstar_compressed_fullagn_one_short \
  --sed-samples 1
```

Verification:

```bash
uv run python -m compileall euclid_dsps scripts
uv run pytest tests
uv run python -m euclid_dsps.cli --config configs/popcosmos_binned_compressed.yaml fit --help
uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar_compressed.yaml fit --help
```

## Remaining Work

### SSP Shape And Compression Investigation

2026-05-28 SSP shape investigation phase started:

- Scope: inspect the Chabrier SSP HDF5 assets, generate plots by wavelength,
  age, and metallicity dimensions, and compare simple compression candidates
  for reducing SSP storage without erasing spectral information.
- Primary assets: `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5` and
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5`.
- Candidate diagnostics: sample spectra, age and metallicity tracks, spectral
  heatmaps, SVD energy/error, piecewise quadratic fits in log wavelength, and a
  simple Haar-wavelet proxy on a log-wavelength grid.
- Outputs should be written under `outputs/ssp_shape_investigation/` and kept
  out of source.

2026-05-28 SSP shape investigation phase completed:

- Added `scripts/investigate_ssp_shapes.py`, a reproducible HDF5 diagnostic
  script for SSP shape plots and simple compression-error metrics.
- Generated plots and metrics under `outputs/ssp_shape_investigation/`,
  including wavelength, age, and metallicity slices; heatmaps; noNE-vs-wNE
  ratios; SVD energy; reconstruction examples; and compression tradeoffs.
- The active Chabrier SSP grids are both `12 x 107 x 11149`; the default
  compression diagnostics focus on 900-30000 Angstrom, which contains 9239 of
  the 11149 wavelength samples.
- Initial result: a shared low-rank basis is the strongest first candidate for
  stellar-continuum compression. On a 2048-point log-wavelength grid, 32 SVD
  components give p95 log-flux errors around 0.013-0.015 dex at roughly 24x
  nominal compression, and 64 components give p95 errors around 0.004-0.005 dex
  at roughly 12x nominal compression.
- Piecewise quadratics are not competitive for narrow spectral structure:
  they only reach p95 below 0.05 dex at low compression, especially for the
  fixed-nebular SSP.
- Haar-style sparsity is useful as a baseline but not clearly better than SVD
  at the tested tolerances. If wavelets are pursued, use a real codec with
  quantization and explicit coefficient/index storage accounting.
- New priority if compression becomes implementation work: split continuum and
  emission-line information before compressing gas or fixed-nebular grids,
  rather than fitting narrow lines with smooth polynomial segments.

2026-05-28 SSP compression documentation extension started:

- Scope: make cleaner summary plots, add explicit interpolation/knot-storage
  tests, document candidate compressed representations, and clarify the VRAM
  tradeoff for larger galaxy batches.
- Additional diagnostics should distinguish file size, resident VRAM size,
  interpolation reconstruction error, and extra compute from on-the-fly
  decoding.

2026-05-28 SSP compression documentation extension completed:

- Extended `scripts/investigate_ssp_shapes.py` with log-wavelength knot
  interpolation tests and cleaner compression-frontier plots.
- Added `outputs/ssp_shape_investigation/popcosmos_asset_size_summary.png` to
  show that the gas SSP grid dominates the resident tensor size.
- Added `docs/source/ssp_compression.rst` and linked it from the docs index.
  The doc recommends a low-rank continuum representation plus separate sparse
  line storage, and records the key caveat that VRAM only drops if JAX consumes
  the compressed representation directly.
- Verification: `uv run python -m compileall scripts/investigate_ssp_shapes.py`,
  `uv run ruff check scripts/investigate_ssp_shapes.py`, and
  `uv run sphinx-build -b html docs/source /tmp/dsps_docs_ssp_check` pass.

2026-05-28 dedicated implementation planning update:

- Added `PLAN_SSP_COMPRESSION.md` as the complete execution plan for testing
  linear-flux compression, continuum/line separation, compressed HDF5 assets,
  JAX integration, age-dependent dust handling, and dense-vs-compressed
  benchmarks.
- Next step remains gated on explicit user approval before implementing the
  compressed-grid experiments or changing model code.

2026-05-29 VRAM-compression planning update:

- Reframed `PLAN_SSP_COMPRESSION.md` around resident GPU memory reduction, not
  disk-level HDF5 compression. The compressed representation must be consumed
  directly by JAX; decoding back to dense gas/AGN tensors is explicitly
  non-goal.
- First implementation step is now to create a clean branch from `dev`, without
  creating a separate worktree. Existing dense `Data/` assets are reference
  inputs and must not be modified in place.
- Priority order is AGN component compression first, then gas compression with
  continuum-plus-sparse-lines, then optional base SSP compression.
- Required gates are dense-vs-compressed benchmarks for AGN-only, gas-only,
  `full_noagn`, and `full_agn`, followed by FSPS/Prospector runs at `n=50` and
  `n=500`.

2026-05-29 VRAM-compression implementation slice:

- Added metadata-first inventory and VRAM-estimation scripts under
  `scripts/inventory_spectral_assets.py` and `scripts/profile_vram_batch.py`.
  They do not instantiate `load_context` by default.
- Baseline metadata outputs were generated under `outputs/ssp_compression/`.
  The current dense full-AGN PopCosmos config has about `6.66 GiB` of estimated
  float32 resident spectral payload before JAX overhead and batch intermediates.
- Added `scripts/build_compressed_agn_component_grid.py` and
  `agn_model: compressed_fsps_component_grid`, which loads `agn_basis`,
  `agn_coeff`, and `agn_scale` instead of the dense AGN component tensor.
- Added `scripts/build_compressed_gas_grid.py` and
  `nebular_model: compressed_gas_grid`, which loads `gas_basis`, `gas_coeff`,
  and `gas_scale` instead of the dense gas `ssp_flux` tensor.
- The first compressed gas mode is explicitly a full-spectrum low-rank
  prototype for VRAM and dense-vs-compressed testing. It is not yet the final
  continuum-plus-sparse-lines science representation.
- Added `scripts/validate_compressed_spectral_asset.py` for compressed AGN and
  gas assets. It validates compressed files without reading dense source grids.
- Added `scripts/benchmark_dense_vs_compressed_spectral_assets.py` for
  identical-parameter dense-vs-compressed photometry checks with progress
  reporting. It now loads one benchmark level at a time and uses lazy dense
  AGN HDF5 slice reads by default, so the default command does not preload the
  dense AGN grid twice.
- Added `scripts/benchmark_photometry_engines.py` for subprocess-isolated
  `dsps_dense_lazy`, `dsps_dense_resident`, `dsps_compressed`, and
  `fsps_prospector` comparison, including per-engine wall time and peak RSS.
  Dense-lazy keeps the dense FSPS AGN component physics but reads only the HDF5
  slices required for each point. Dense-resident has a memory guard and is
  skipped when its estimated static payload plus overhead is too close to
  `MemAvailable`.
- Updated the FSPS/Prospector benchmark harness so compressed gas and
  compressed AGN configs can be passed after the compressed assets exist.
- Targeted verification passed with `python -m compileall euclid_dsps scripts
  tests/test_config.py tests/test_model.py tests/test_fsps_grid_scripts.py` and
  `pytest -q tests/test_config.py tests/test_model.py
  tests/test_fsps_grid_scripts.py`.

2026-05-29 deeper VRAM-compression analysis:

- The current compressed gas asset is coefficient dominated:
  `gas_coeff` is about `15.36 MiB` raw, while `gas_basis` is about `2.72 MiB`.
  Further wins should target coefficient dtype/rank or a different gas
  representation, not HDF5 repacking.
- Sampled rank truncation indicates AGN can likely shrink from `k32` toward
  `k12-k16` with little information loss, pending broad-band benchmarks.
- Sampled dense AGN slices show `agn_lnu_per_mformed / fagn` is effectively
  invariant across `fagn_grid` at the `~1e-7` relative level. The next AGN
  compressed format should remove the explicit `fagn` axis and apply `fagn` as
  a runtime multiplier, after a dedicated benchmark/test gate.
- Gas still benefits from `k48-k64`; dropping full-spectrum gas to low ranks
  damages line-sensitive structure. The preferred final representation remains
  low-rank continuum plus sparse emission-line luminosities.
- After gas/AGN compression, the base SSP is now the largest resident tensor
  left, about `55 MiB`. Revisit compressed base SSP modes at `k64/k128`; this
  can save more memory than small AGN basis tweaks once the multi-GiB grids are
  gone.
- Naively storing all compressed arrays as `float16` is unsafe because the
  scale arrays are around `1e-22..1e-11`. Mixed precision should keep scale as
  `float32` or store `log10(scale)`, and reconstruct in `float32`.
- Next experiments should benchmark mixed-precision assets:
  AGN `basis=float32, coeff=float16, scale=float32` at `k12/k16`; gas
  `basis=float16, coeff=float16, scale=float32` at `k48/k64`; and optional
  coefficient-only int8 quantization with explicit scales.

2026-05-29 compression tradeoff audit and implementation update:

- Added `scripts/audit_compression_tradeoffs.py`. It samples dense gas/AGN
  spectra without materializing full dense tensors and writes
  `outputs/ssp_compression/tradeoffs/compression_tradeoff.csv`,
  `compression_tradeoff_summary.json`, `compression_factor_vs_loss.png`, and
  `candidate_payload_mib.png`.
- Audit highlights from `n=256` sampled dense spectra:
  - AGN `fagn_factored_svd_k12_coeff16_basis32`: estimated payload
    `0.82 MiB`, compression factor `~4800x`, median sampled p95 relative loss
    `2.4e-4`.
  - AGN `fagn_factored_svd_k8_coeff16_basis32`: estimated payload
    `0.56 MiB`, compression factor `~7000x`, median sampled p95 relative loss
    `5.8e-4`.
  - Gas `k64 mixed_f16`: estimated payload `9.28 MiB`, compression factor
    `~288x`, median sampled p95 relative loss `3.7e-3`.
  - Gas `k48 mixed_f16`: estimated payload `7.02 MiB`, compression factor
    `~381x`, median sampled p95 relative loss `6.2e-3`.
  - Stellar SSP `k64` remains the recommended first compact SSP point from
    existing log-flux diagnostics: estimated payload `4.47 MiB`, p95
    `0.00437 dex`; `k128` is safer at `8.90 MiB`, p95 `0.00135 dex`.
- Implemented explicit runtime support for:
  - `model.ssp_model: compressed_basis` with `model.compressed_ssp_path`;
  - compressed SSP payloads `ssp_basis`, `ssp_coeff`, `ssp_scale`;
  - compressed AGN assets with `fagn_handling: linear_runtime_multiplier`,
    meaning `fagn` is multiplied at runtime and the coefficient tensor drops
    the `fagn` axis;
  - mixed-precision compressed payloads where basis/coefficients can stay
    `float16` resident on device while selected slices are reconstructed in
    `float32`.
- Added `scripts/build_compressed_ssp_grid.py`; extended
  `scripts/build_compressed_agn_component_grid.py` with `--factor-fagn`,
  `--basis-dtype`, and `--coeff-dtype`; extended
  `scripts/build_compressed_gas_grid.py` with `--basis-dtype` and
  `--coeff-dtype`.
- Benchmark harnesses now accept `--compressed-ssp` in addition to compressed
  gas and AGN paths.

### Implementation Plan: PopCosmos Benchmark Closure

2026-05-28 forward-model closure phase started:

- Scope: implement the next correction slice after the noNE benchmark report:
  diagnose and fix verified DSPS/FSPS mismatches in the pure-stellar
  normalization, redshift/luminosity-distance/filter path, UV/IGM behavior,
  Prospector/FSPS-like dust, and then gas/line-continuum handling.
- Current gate remains unchanged: do not train the first learned prior until
  `stellar_only`, `stellar_plus_dust`, `stellar_plus_gas`, and `full_noagn`
  pass the benchmark criteria or any remaining approximation is explicitly
  named and accepted.

Goal: turn the current PopCosmos-like no-AGN path into a benchmarked prior-v0
forward model. The `n=500` FSPS/Prospector benchmark in
`outputs/benchmarks/popcosmos_binned_noagn_fsps_n500/` runs end to end, but its
residuals are not yet within the recommended prior-readiness thresholds.

Phase 1 - Pure-stellar baseline:

- Add a pure-stellar Chabrier SSP generation mode, with
  `add_neb_emission=0` and `add_neb_continuum=0`.
- Write it to an unambiguous path such as
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5`.
- Add HDF5 attrs making the no-nebular contract explicit:
  `imf_type=1`, `imf_name=chabrier`, `z_sun=0.0142`,
  `add_neb_emission=0`, `add_neb_continuum=0`, units, FSPS version, isochrones,
  and spectral library.
- Extend the benchmark so `stellar_only` and `stellar_plus_dust` use this
  pure-stellar SSP on the DSPS side. The current `fixed_ssp` path uses the base
  SSP and therefore includes fixed nebular emission if the base SSP HDF5 was
  generated that way.
- Tests: pure-stellar SSP validation passes; benchmark level selection uses the
  pure-stellar context for gas-free levels; the old fixed-nebular SSP remains
  usable only where explicitly requested.

Phase 2 - Strict asset metadata:

- Tighten PopCosmos-like IMF validation so contradictory metadata fails.
  If `imf_type` exists it must be `1`; if `imf_name` exists it must be
  `chabrier`; if both exist they must agree; if neither exists, fail.
- Add tests for inconsistent metadata, especially `imf_type=2` with
  `imf_name=chabrier` and `imf_type=1` with `imf_name=kroupa`.
- Keep legacy lognormal assets compatible outside PopCosmos-like configs.

Phase 3 - Benchmark diagnostics:

- Extend `benchmark_summary.json` to report residual statistics per
  `(level, band)` using finite-only counts and explicit non-finite counts.
- Add residual correlations per `(level, band)` instead of only pooled across
  all levels and bands.
- Add recent-SFR diagnostics derived from the PopCosmos SFH bins, at minimum
  the youngest-bin SFR and a recent-to-old SFR contrast.
- Add plots for residuals against `z_obs`, `tau2`, `dust_index_n`,
  `log10_stellar_metallicity`, `log10_gas_metallicity`,
  `log10_gas_ionization`, and recent SFR.

Phase 4 - Rerun staged benchmarks:

- Rerun a tiny benchmark after each code/data change:
  `--n 5`, then `--n 50`, then `--n 500`.
- Compare each stage against
  `outputs/benchmarks/popcosmos_binned_noagn_fsps_n500/`.
- Do not move to prior training until broad bands are close to the target:
  median `|Delta mag| < 0.02`, p95 `|Delta mag| < 0.05`, and no clear monotone
  residual trend with redshift, dust, metallicity, gas, or recent SFR.

Phase 5 - Dust/gas correction decisions:

- If pure-stellar residuals remain large, audit SSP normalization, mass units,
  age-bin mapping, luminosity distance, filter integration, and surviving-mass
  normalization before touching gas or dust.
- If pure-stellar passes but `stellar_plus_dust` fails, benchmark the
  `prospector_fsps` dust curve directly against FSPS/Prospector as a function
  of wavelength, `tau2`, `dust_index_n`, `dust1`, and age.
- If `stellar_plus_gas` or `full_noagn` fails after the baseline is clean,
  generate an enriched gas grid with separated continuum and line fluxes, then
  apply the PopCosmos emission-line correction table.
- Keep `emission_line_corrections: none` labeled as `uncalibrated_cloudy`.

Phase 6 - Later AGN and production scaling:

- Audit AGN normalization against FSPS internals or external CLUMPY convention
  before recommending full AGN configs.
- Scale CUDA batch fitting only after the no-AGN forward model passes the
  benchmark gates.
- Add synthetic recovery tests with known DSPS-generated parameters.

2026-05-28 forward-model closure implementation update:

- Fixed the dominant pure-stellar DSPS/Prospector mismatch by replacing the
  generic log-cosmic-time DSPS age-weight interpolation with a direct overlap
  integral from PopCosmos lookback bins onto the SSP age grid when
  `model.sfh_time_grid: prospector_step` is active.
- Ported FSPS Madau95 IGM attenuation into `igm_model: fsps_madau95`, replacing
  the earlier coarse UV approximation for PopCosmos-like configs.
- Ported the FSPS/Prospector `dust_type=4` diffuse attenuation shape into the
  JAX `prospector_fsps` dust mode while preserving the existing age-dependent
  architecture: diffuse dust applies to all stellar ages; birth-cloud dust
  applies only at ages `<= dust_tesc_logyr`.
- Added optional loading and use of `ssp_surviving_mstar` from FSPS-generated
  SSP HDF5 files. This removes the remaining few-percent normalization offset
  from using the DSPS analytic surviving-mass approximation for Chabrier FSPS
  SSPs.
- Updated `scripts/generate_fsps_ssp_grid.py` to write
  `ssp_surviving_mstar[stellar_lgmet, age]` plus units metadata, and
  regenerated:
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5` and
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5`.
- Latest staged benchmark:
  `outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n20/`.
  Finite overall residuals now show:
  `stellar_only` median `|Delta mag| = 0.0060`, p95 `0.0107`;
  `stellar_plus_gas` median `0.0060`, p95 `0.0183`.
  Excluding effectively-zero flux cases with either DSPS or reference magnitude
  `>80`, `stellar_plus_dust` median is `0.0058`, p95 `0.0154`, and
  `full_noagn` median is `0.0058`, p95 `0.0193`.
- Remaining blocker is the extreme-dust/tiny-flux tail where the reference
  remains finite at magnitudes `~180` while the float32 DSPS path either
  underflows or floors around magnitudes `~100-115`. Benchmark summaries must
  keep reporting these non-finite/effectively-faint counts separately; prior
  training should use flux-space likelihoods and avoid interpreting these
  magnitude deltas as normal broadband residuals.
- The larger `n=500` benchmark completed at
  `outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n500/`.
  It confirms the `n=20` result: `stellar_only` median `|Delta mag| = 0.0058`,
  p95 `0.0115`; `stellar_plus_gas` median `0.0059`, p95 `0.0178`.
  For bright finite rows with DSPS and reference magnitudes `<80`,
  `stellar_plus_dust` median is `0.0057`, p95 `0.0125`, and `full_noagn`
  median is `0.0057`, p95 `0.0135`. The remaining raw dust/full p99 outliers
  are still the expected extreme-faint magnitude tail.
- Wrote the benchmark analysis report:
  `outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n500/analysis_report.md`.

## Latest Verification

2026-05-28 forward-model closure verification:

- Regenerated and validated the Chabrier noNE and fixed-nebular SSP HDF5 assets
  with `ssp_surviving_mstar`.
- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py
  scripts/generate_fsps_ssp_grid.py scripts/fsps_grid_common.py
  tests/test_model.py tests/test_config.py tests/test_benchmark.py
  tests/test_fsps_grid_scripts.py` passed. The repo-wide ruff check is blocked
  by an unrelated pre-existing import-order issue in
  `scripts/investigate_ssp_shapes.py`.
- `uv run pytest -q tests/test_config.py tests/test_model.py tests/test_benchmark.py`
  passed: 72 passed, 3 skipped.
- `uv run pytest -q` passed: 140 passed, 3 skipped.
- `conda run -n shine bash -lc 'JAX_PLATFORMS=cpu python
  scripts/benchmark_against_fsps_prospector.py --config
  configs/popcosmos_binned_noagn.yaml --n 20 --seed 0 --out
  outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n20'` passed.

2026-05-28 AGN benchmark audit mode:

- Extended `scripts/benchmark_against_fsps_prospector.py` so configs with
  `model.agn_model: template_grid` automatically run two additional audit
  levels: `stellar_plus_agn` and `full_agn`.
- The AGN reference side now passes `fagn=exp(ln_fagn)` and
  `agn_tau=exp(ln_tauagn)` into FSPS/Prospector and enables
  `add_dust_emission` for AGN levels.
- The DSPS side uses the repository AGN template grid. The benchmark summary
  explicitly labels this as an AGN audit rather than a final PopCosmos AGN
  validation because the DSPS template-grid bolometric normalization convention
  is still approximate.
- Smoke run passed:
  `outputs/benchmarks/smoke_popcosmos_binned_agn_audit_n1/`.
  It confirms the no-AGN levels remain at `~0.005` mag while AGN levels expose
  larger UV/blue differences, especially `lsst_u`, which is the intended audit
  signal for the next AGN-normalization phase.
- Verification passed:
  `uv run python -m compileall scripts/benchmark_against_fsps_prospector.py`,
  `uv run ruff check scripts/benchmark_against_fsps_prospector.py
  tests/test_benchmark.py`, and `uv run pytest -q tests/test_benchmark.py`.
- User-run AGN audit benchmark analysed:
  `outputs/benchmarks/popcosmos_binned_agn_audit_n500/`.
  Report written to `outputs/report/popcosmos_binned_agn_audit_n500/report.md`
  with figures and CSV tables. The no-AGN levels remain benchmark-ready in
  bright finite bands, but `stellar_plus_agn` has a severe UV/blue tail and
  `full_agn` fails the readiness gates by a large margin.
- Current AGN verdict: keep `popcosmos_binned_noagn` as the prior-v0 candidate.
  Do not use the full AGN path for a scientific learned prior until a direct
  SED-level AGN normalization audit resolves the template-grid versus
  FSPS/Prospector `fagn`/`agn_tau` convention.

2026-05-28 proposed AGN SED-audit implementation plan:

- Do not tune `full_agn` photometry directly. First isolate the pure AGN
  component before dust, gas, IGM, and filters can hide the source of the
  mismatch.
- Add DSPS component outputs for AGN debugging: intrinsic stellar SED, dusted
  stellar SED, optional gas SED, AGN SED, pre-IGM SED, and post-IGM SED.
- Extend the AGN template generator to support signed finite-difference
  templates and grids over `agn_tau`, `fagn_normalization`, normalization age,
  and normalization metallicity. Keep the current single-age/single-metallicity
  template as the legacy approximate convention.
- Generate a new explicit audit asset whose filename and metadata state that it
  is an FSPS finite-difference AGN calibration grid, not yet a production
  PopCosmos AGN asset.
- Add an `agn_component_only` benchmark level comparing
  `FSPS(fagn, agn_tau) - FSPS(fagn=0)` against the DSPS AGN component before
  IGM and photometric integration.
- Add configurable AGN host attenuation experiments:
  `none` for the current behavior, plus at least one host-dust variant that
  applies the diffuse Prospector/FSPS-like dust curve to the AGN component.
- Rerun staged benchmarks in this order: tiny `agn_component_only`, `n=50`
  AGN SED audit, then `n=500` photometric AGN audit only if the component-level
  SED comparison is acceptable.

2026-05-28 AGN SED-audit implementation started:

- Scope: implement the proposed AGN component audit path, including DSPS
  component outputs, signed/multi-axis AGN finite-difference template support,
  an `agn_component_only` benchmark level, and configurable AGN host attenuation
  experiments.

2026-05-28 AGN SED-audit implementation completed:

- `JaxModelResult` now carries debug component SEDs:
  `stellar_intrinsic_sed`, `stellar_dusted_sed`, `gas_sed`, `agn_sed`,
  `pre_igm_sed`, and `post_igm_sed`.
- The PopCosmos binned and Diffstar forward paths now build the AGN component
  separately before IGM attenuation, so benchmark code can inspect it without
  going through filters.
- Added `model.agn_host_attenuation` with supported values `none`, `diffuse`,
  and `prospector_fsps`. The full AGN configs explicitly keep `none` for
  backward-compatible behavior until the audit chooses a physical convention.
- `scripts/generate_fsps_agn_grid.py` now supports audit grids over
  `fagn_grid`, `agn_tau_grid`, `tage_gyr_grid`, and `stellar_logzsol_grid`,
  with optional `--signed-delta`. The legacy 2D `agn_tau x wave` format still
  loads and validates.
- `scripts/benchmark_against_fsps_prospector.py` now accepts
  `--levels agn_component_only` and writes SED-ratio audit rows with
  `dsps_agn_lnu`, `reference_agn_lnu`, `delta_log10_lnu`, and equivalent
  `delta_mag = -2.5 log10(DSPS_AGN/FSPS_AGN)`.
- Added `--agn-template` to the benchmark CLI so AGN audit assets can be tested
  without editing the science config.
- Added `--agn-host-attenuation` to the benchmark CLI so the `none`, `diffuse`,
  and `prospector_fsps` AGN host-attenuation experiments can be compared without
  creating temporary YAML configs.
- Added component-only diagnostic plots under the benchmark `diagnostics/`
  directory, including SED ratio versus rest wavelength and point-level
  residual trends versus `ln_fagn`, `ln_tauagn`, `z_obs`, and `tau2`.
- Generated the first signed AGN audit asset:
  `Data/popcosmos_chabrier_agn_fspsdiff_audit_grid.h5`, shape
  `(5, 9, 4, 3, 11149)` for `fagn`, `agn_tau`, `tage_gyr`,
  `stellar_logzsol`, and wavelength. This is an audit asset, not yet the
  default science AGN template.
- Smoke `agn_component_only` run with the audit grid passed at
  `outputs/benchmarks/smoke_popcosmos_binned_agn_component_only_audit_grid/`.
  The `n=1` smoke has median absolute equivalent AGN-component residual
  `~0.33` mag and p95 `~5.26` mag over 64 sampled rest wavelengths, indicating
  that the component-level mismatch is now directly measurable and still needs
  science interpretation before enabling AGN prior training.
- User-run AGN audit results analysed:
  `outputs/benchmarks/popcosmos_binned_agn_component_only_audit_grid_n50`,
  `outputs/benchmarks/popcosmos_binned_agn_audit_grid_nohost_n500`, and
  `outputs/benchmarks/popcosmos_binned_agn_audit_grid_hostdust_n500`.
  Report written to
  `outputs/report/popcosmos_binned_agn_audit_grid_comparison_2026-05-28/report.md`.
- The signed 5D audit grid does not materially change AGN photometry in the
  no-host-dust convention: row-wise p95 absolute difference from the old 2D
  template is only `~0.017` mag. `full_agn` remains at bright p95 `~20` mag.
- `agn_host_attenuation: prospector_fsps` reduces the aggregate `full_agn`
  bright p95 from `~20.0` mag to `~10.4` mag, but shifts the failure mode from
  DSPS being too bright to DSPS often being too faint. This is not a final AGN
  convention.
- The `agn_component_only` `n=50` run shows that the component-level mismatch is
  wavelength dependent: median absolute equivalent residual is `~0.18` mag in
  rest `3000-10000 A`, but the p95 is `~7.1` mag globally and is dominated by
  far-UV/Lyman and IR wavelength regions.
- Updated verdict: the no-AGN path remains the only prior-v0 candidate. The AGN
  path now has the right audit instrumentation, but still needs a component
  convention/normalization fix before any scientific AGN prior training.
- Verification passed:
  `uv run python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py scripts/generate_fsps_agn_grid.py
  scripts/fsps_grid_common.py tests/test_model.py tests/test_benchmark.py
  tests/test_fsps_grid_scripts.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py tests/test_fsps_grid_scripts.py` with 88 passed and
  3 skipped; latest `uv run pytest -q` with 148 passed and 3 skipped.

2026-05-28 FSPS-native AGN component implementation completed:

- Added `scripts/generate_fsps_agn_component_grid.py`, which writes an
  SSP-shaped AGN component grid:
  `agn_lnu_per_mformed[fagn, agn_tau, Zstar, age, wave] =
  FSPS(fagn, agn_tau) - FSPS(fagn=0)`.
  Its default `fagn` and `agn_tau` axes cover the current PopCosmos AGN prior
  bounds, including `ln_fagn [-14, 1]` and `ln_tauagn [1.609438, 5.010635]`.
- Added `model.agn_model: fsps_component_grid` and
  `model.agn_component_grid_path`. This path does not use the legacy
  `fagn * Lbol * template` normalization; the AGN component is interpolated in
  `fagn`, `agn_tau`, and stellar metallicity, then summed over SSP age with the
  same PopCosmos/Diffstar SFH weights and formed mass as the stellar component.
- The AGN component grid loader validates the SSP wavelength, age, and
  metallicity axes against the active base SSP and keeps the strict PopCosmos
  Chabrier/z_sun metadata checks.
- `scripts/benchmark_against_fsps_prospector.py` now accepts
  `--agn-component-grid`, mutually exclusive with `--agn-template`, so the
  legacy template-grid audit and the FSPS-native component-grid audit can be
  compared without editing science YAML files.
- Added tests covering config validation, synthetic forward scaling with
  formed mass, benchmark level selection, and the component-grid generator
  using the fake FSPS test backend.
- Remaining AGN work is empirical: generate a production-size component grid,
  rerun `agn_component_only`, then rerun `stellar_plus_agn` and `full_agn`.
  If the component-level SED benchmark passes but photometric AGN levels still
  fail, the next suspect is host/torus attenuation order rather than AGN
  normalization.

2026-05-28 AGN component benchmark runtime fix:

- Fixed `scripts/benchmark_against_fsps_prospector.py` so
  `--levels agn_component_only` only loads the stellar-only DSPS context and no
  longer loads the gas-grid context.
- Added `--runtime config|auto|cpu|gpu`; use `--runtime cpu` for large
  component-grid audits that do not fit on the GPU.
- Updated runtime setup so a non-auto runtime platform forces
  `JAX_PLATFORMS`, while `auto` still clears stale platform requests.
- Smoke benchmark passed with the small AGN component grid:
  `outputs/benchmarks/smoke_popcosmos_binned_agn_component_grid_cpu_lazy_n1/`.

2026-05-28 full-AGN lazy component-grid loading implemented:

- `full_agn` and `stellar_plus_agn` benchmark levels with
  `model.agn_model: fsps_component_grid` no longer load the full 3.9 GB AGN
  component grid into JAX.
- The benchmark now loads the DSPS stellar/gas context with AGN disabled, then
  reads only the HDF5 slices needed for the sampled
  `(fagn, agn_tau, stellar metallicity)` point, interpolates those slices, and
  adds the AGN component before IGM and photometric integration.
- Axis interpolation is strict: sampled AGN values outside the component-grid
  axes now raise an explicit error instead of clipping silently.
- This is a benchmark-only memory fix. Production fitting with
  `fsps_component_grid` still needs a dedicated compressed/lazy model path
  before it is safe for large runs.

2026-05-29 AGN dust/gas isolation implementation started:

- Scope: add a tunable AGN host-attenuation scale and two benchmark levels that
  isolate dust+AGN and gas+AGN before attempting any `full_agn n=500` run.
- Motivation from `full_agn_hostdust_n50`: `agn_host_attenuation: none`
  under-attenuates the faint dusty tail, while `prospector_fsps` with full
  strength over-attenuates some high-redshift dusty points.

2026-05-29 AGN dust/gas isolation implementation completed:

- Added `model.agn_host_attenuation_scale`, defaulting to `1.0`, with
  non-negative config validation. The scale multiplies the host-dust optical
  depth applied to the AGN component, so `0.0` is equivalent to no host
  attenuation and `1.0` is the previous full-strength behavior.
- Added benchmark CLI option `--agn-host-attenuation-scale`.
- Added benchmark levels:
  `stellar_plus_dust_plus_agn` for pure-stellar+dust+AGN, and
  `stellar_plus_gas_plus_agn` for gas+AGN with dust disabled.
- Verification passed:
  `uv run python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py tests/test_model.py
  tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 87 passed and 3 skipped.

2026-05-29 FSPS AGN diffuse attenuation alignment started:

- Scope: replace the diagnostic interpretation of
  `agn_host_attenuation_scale` with a fixed FSPS-native AGN host attenuation
  mode. Local FSPS source inspection shows `agn_dust.f90` applies
  `exp(-attn_curve(...))` to the AGN dust template, without multiplying by
  `dust2`. The existing scaled modes remain audit diagnostics, but should not
  be used as learned science parameters.

2026-05-29 FSPS AGN diffuse attenuation alignment completed:

- Added `model.agn_host_attenuation: fsps_diffuse_unit_tau`. This mode applies
  the JAX Prospector/FSPS diffuse attenuation shape directly at unit optical
  depth and is intentionally independent of fitted `tau2`.
- Config validation now rejects `agn_host_attenuation_scale != 1.0` with
  `fsps_diffuse_unit_tau`, so the exact FSPS convention cannot be silently
  turned into a learned or tuned scale.
- The full AGN configs now use `agn_host_attenuation: fsps_diffuse_unit_tau`.
  They remain non-recommended for prior v0 until the component-grid AGN
  benchmark passes.
- Benchmark isolation keeps `stellar_plus_agn` and `stellar_plus_gas_plus_agn`
  explicitly host-dust-free; the unit-tau FSPS AGN attenuation is only used for
  dust+AGN and full-AGN levels.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py tests/test_model.py
  tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 89 passed and 3 skipped.

2026-05-29 FSPS AGN/IGM ordering alignment started:

- Scope: add an explicit `model.agn_igm_order` convention. The existing DSPS
  behavior applies IGM after adding AGN (`pre_igm`). Local FSPS source shows
  `compsp.f90` applies IGM before `agn_dust`, so the benchmark needs an
  `fsps_after_igm` mode where stellar/gas light is IGM-attenuated before the
  AGN component is added.

2026-05-29 FSPS AGN/IGM ordering alignment completed:

- Added `model.agn_igm_order` with supported values `pre_igm` and
  `fsps_after_igm`. The default remains `pre_igm` for backward compatibility.
- Full AGN PopCosmos-like configs now set `agn_igm_order: fsps_after_igm`.
- `combine_agn_and_igm_jax` centralizes the ordering in the production forward
  model. In `fsps_after_igm`, stellar/gas light is attenuated by IGM first and
  the already host-attenuated AGN component is added afterward.
- The lazy AGN component-grid benchmark path now uses the same helper, so
  `fsps_component_grid` audits test the same ordering as production.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py tests/test_model.py
  tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 92 passed and 3 skipped.

2026-05-29 FSPS AGN baked-attenuation replacement completed:

- Analysis of the after-IGM benchmark showed `stellar_plus_agn` is clean in the
  bright regime, while `stellar_plus_dust_plus_agn` and `full_agn` remain too
  faint in dusty blue bands. The active AGN component/template assets were
  generated through FSPS `agn_dust` with `dust_type=0`, so they already include
  a unit-tau power-law AGN attenuation before runtime applies the
  `fsps_diffuse_unit_tau` target curve.
- Added `model.agn_baked_attenuation` with supported values `none` and
  `fsps_powerlaw_unit_tau`, plus `model.agn_baked_dust_index` defaulting to
  `-0.7`, the FSPS default from `sps_vars.f90`.
- In `agn_host_attenuation: fsps_diffuse_unit_tau`, DSPS now replaces baked
  power-law attenuation by applying
  `exp(-(tau_fsps_dust_type4 - tau_baked_dust_type0))`. This avoids stacking
  the baked FSPS `dust_type=0` attenuation and the target `dust_type=4`
  attenuation.
- The full AGN PopCosmos-like configs now declare
  `agn_baked_attenuation: fsps_powerlaw_unit_tau` and
  `agn_baked_dust_index: -0.7`.
- AGN grid generators now write these baked-attenuation metadata keys for newly
  generated template/component assets.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py
  scripts/generate_fsps_agn_component_grid.py scripts/generate_fsps_agn_grid.py
  tests/test_model.py tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 94 passed and 3 skipped.

2026-05-29 FSPS AGN baked-attenuation benchmark analysed:

- User-run benchmarks completed:
  `outputs/benchmarks/popcosmos_binned_agn_afterigm_replace_baked_n50` and
  `outputs/benchmarks/popcosmos_binned_agn_afterigm_replace_baked_n500`.
- The replacement fix closes the previous dusty blue-band failure. At `n=500`,
  `full_agn` has median p95 `0.0123` mag and max p95 `0.0371` mag over the
  finite/effectively-bright summary, while `stellar_plus_dust_plus_agn` has
  median p95 `0.0114` mag and max p95 `0.0316` mag. Both satisfy the broad-band
  prior-readiness target for finite observable rows.
- Compared to the previous `fsps_unit_tau_afterigm_n500`, the `full_agn`
  `lsst_u` p95 drops from `~5.0` mag to below the global max of `0.0371` mag,
  confirming the failure was stacked baked `dust_type=0` plus target
  `dust_type=4` AGN attenuation.
- `stellar_plus_agn` still has a very faint `lsst_u` tail when all finite rows
  are included, but it is clean in the observable regime (`mag<35` max p95
  `0.0255` mag). This is a diagnostic faint-UV tail rather than the blocker for
  the full dusty AGN model.

2026-05-29 full forward FSPS/Prospector closure benchmark analysed:

- User-run benchmark completed:
  `outputs/benchmarks/popcosmos_binned_full_forward_fsps_closure_n500`.
  It covers 500 sampled points, 8 benchmark levels, and 10 bands, including
  `stellar_only`, `stellar_plus_dust`, `stellar_plus_gas`, `full_noagn`,
  `stellar_plus_agn`, `stellar_plus_dust_plus_agn`,
  `stellar_plus_gas_plus_agn`, and `full_agn`.
- Report written to
  `outputs/report/popcosmos_binned_full_forward_fsps_closure_n500/report.md`,
  with summary tables, worst residuals, threshold sensitivity, correlations,
  heatmaps, CDFs, and trend plots.
- Production broad-band levels now pass the configured bright finite
  FSPS/Prospector closure target. `full_noagn` has median band p95
  `0.0129` mag and max band p95 `0.0166` mag. `full_agn` has median band p95
  `0.0123` mag and max band p95 `0.0371` mag.
- `stellar_only`, `stellar_plus_dust`, `stellar_plus_gas`, and
  `stellar_plus_dust_plus_agn` also pass the bright finite target. Remaining
  non-finite rows in dusty/full levels are explicit magnitude-space
  zero-flux/effectively-faint cases and should be handled in flux space for
  inference.
- The only large residual tails are diagnostic component levels with dust
  disabled: `stellar_plus_agn` and `stellar_plus_gas_plus_agn`, mainly in
  `lsst_u/g` at very faint or high-redshift UV points. These remain documented
  component-isolation limitations, not a production `full_agn` blocker.
- Current verdict: the DSPS forward model is now FSPS/Prospector-like for
  broad-band prior training. It is still not an official PopCosmos reproduction
  because PopCosmos learned emission-line correction tables are not included;
  the gas path remains raw FSPS/CLOUDY with
  `emission_line_corrections: none`.

2026-05-29 documentation and default-config cleanup started:

- Promote the validated full AGN path to the default documented setup:
  `configs/popcosmos_binned.yaml` first, `configs/popcosmos_diffstar.yaml` as
  the comparison, with no-AGN configs retained only for fallback/ablation.
- Update the full AGN configs to use `agn_model: fsps_component_grid` and
  `Data/popcosmos_chabrier_agn_component_ssp_grid.h5`, matching the benchmark
  that closed against FSPS/Prospector.
- Rewrite README and Sphinx docs so the SED pipeline is understandable from
  SSP generation to photometry, including active assets, assumptions,
  implementation files, benchmark commands, and remaining scientific caveats.
- Reduce `docs/source/science_assessment.rst` to the current status and caveats
  rather than retaining stale historical AGN/no-AGN conclusions.
- Remove unused legacy/audit HDF5 files from `Data/`, while keeping
  `ssp_data_fsps_v3.2_lgmet_age.h5` because a legacy smoke test still
  references it.

2026-05-29 documentation and default-config cleanup completed:

- `README.md` now presents the validated full AGN path as the default, documents
  active assets, generation commands, fit commands, and the FSPS/Prospector
  closure benchmark command.
- Added `docs/source/forward_model.rst` as the step-by-step SED pipeline:
  HDF5 SSP axes, SFH weights, surviving mass, dust, gas, AGN component grid,
  IGM/order, photometry, implementation files, assumptions, and benchmark
  status.
- Rewrote `docs/source/data_download.rst`, `docs/source/run_setup.rst`, and
  `docs/source/science_assessment.rst` around the current full AGN model.
  `science_assessment.rst` is now intentionally short and only records current
  status, validated scope, and remaining caveats.
- Updated `docs/source/architecture.rst`, `catalog_columns.rst`,
  `installation.rst`, `testing.rst`, and `index.rst` to match the active assets
  and commands.
- Updated `configs/popcosmos_binned.yaml` and `configs/popcosmos_diffstar.yaml`
  to use `agn_model: fsps_component_grid`,
  `agn_component_grid_path: Data/popcosmos_chabrier_agn_component_ssp_grid.h5`,
  `stellar_only_ssp_path`, and Student-t photometric likelihood by default.
- Removed unused legacy/audit assets from `Data/`: old Kroupa SSPs, old gas
  grids, legacy AGN template grids, the AGN smoke/audit grids, and Windows zone
  identifier sidecar files. Retained the active Chabrier assets and
  `ssp_data_fsps_v3.2_lgmet_age.h5`.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `pytest tests/test_config.py tests/test_model.py tests/test_benchmark.py
  tests/test_fsps_grid_scripts.py` with 104 passed and 3 skipped;
  `uv run python -m sphinx -W --keep-going -b html docs/source
  docs/build/html`;
  `git diff --check`.

2026-05-29 didactic documentation pass started:

- Add short explanations of the main acronyms and model ingredients:
  SSP, SED, SFH, IMF, isochrones, C3K, IGM, CLOUDY, emission lines, AGN,
  FSPS, Prospector, and DSPS.
- Make the distinction between full AGN and no-AGN explicit in README,
  `forward_model.rst`, and run setup.
- Remove the Sphinx `SSP Compression Notes` page for now because compressed SSP
  execution is not implemented in the production model.

2026-05-29 didactic documentation pass completed:

- `docs/source/forward_model.rst` now includes a concise glossary for SED, SSP,
  SFH, IMF, isochrones, C3K, IGM, CLOUDY, emission lines, AGN, FSPS,
  Prospector, and DSPS. README links to this page rather than duplicating the
  glossary.
- `docs/source/forward_model.rst` now explains the same concepts before the
  step-by-step SED pipeline and records why the current FSPS/MIST/C3K/Chabrier
  choices are used.
- `docs/source/run_setup.rst` now defines full AGN versus no-AGN before listing
  commands.
- Removed `docs/source/ssp_compression.rst` from source and from the Sphinx
  toctree. Compression remains only a historical plan item, not user-facing
  production documentation.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run python -m sphinx -W --keep-going -b html docs/source
  docs/build/html`.

2026-05-28 fit-regression diagnosis started:

- Scope: compare the user-provided 512-galaxy runs
  `popcosmos_binned_chi2_512_b10`,
  `popcosmos_binned_student_t_512_b10`, and
  `popcosmos_binned_noagn_chabrier_student_t_512_b10` to explain why redshift
  recovery degrades and why some inferred parameters are flat across galaxies.
- Initial hypothesis to test from the generated reports: the fit is not simply
  optimizer noise; likely contributors are changed likelihood/physics
  contracts, forward-model mismatch in blue bands, redshift attractors, and
  poorly constrained or inactive parameters that sit at initialization/bounds.

2026-05-28 Reveal.js recap deck started:

- Scope: create a didactic French slide deck summarizing the code delta from
  `master`/`dev`/the current working tree, PopCosmos-like inference pre-work,
  new SSP generation, DSPS photometry flow, FSPS/Prospector benchmark results,
  and the next benchmark tests to implement.
- Target output:
  `outputs/reports/ssp_generation_assessment_2026-05-28/slides_popcosmos_ssp_benchmark_reveal.html`.
- The deck should reuse the generated report figures and cite the papers behind
  the main scientific choices.
- Completed the deck at the target path. It uses local figures from the SSP,
  Student-t, `n=500`, and noNE smoke benchmark reports, plus linked paper
  references for FSPS, Chabrier, MIST, DSPS, Prospector, dust, IGM, Diffstar,
  CLUMPY, and PopCosmos.
- Verified the HTML parses and that all seven local image paths resolve.

2026-05-28 PopCosmos benchmark-closure implementation started:

- Scope: implement the first next-step slice from the `n=500` benchmark review.
- Added `scripts/generate_fsps_ssp_grid.py --stellar-only`, which writes the
  pure-stellar Chabrier no-nebular-emission asset
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5`.
- Generated and validated `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5` in the
  `shine` conda environment. Metadata confirms `imf_type=1`,
  `imf_name=chabrier`, `z_sun=0.0142`, `add_neb_emission=0`, and
  `add_neb_continuum=0`.
- Added `model.stellar_only_ssp_path` to the no-AGN configs. The benchmark also
  accepts `--stellar-ssp` to override this path.
- Updated `scripts/benchmark_against_fsps_prospector.py` so
  `stellar_only` and `stellar_plus_dust` use the pure-stellar DSPS context,
  while `stellar_plus_gas` and `full_noagn` use the Chabrier gas-grid context.
- Tightened PopCosmos-like IMF validation so contradictory HDF5 metadata now
  fails instead of accepting `imf_type=1` or `imf_name=chabrier` independently.
- Extended benchmark summaries with `n_total`, `n_finite_both`,
  `n_nonfinite_dsps`, `n_nonfinite_reference`, and `n_nonfinite_delta` per
  `(level, band)`, plus per-level/per-band residual correlations including a
  recent-SFR proxy.
- Added per-level diagnostic plots under `diagnostics/` for residuals against
  redshift, dust, metallicity, gas, and recent SFR.
- Ran a smoke benchmark:
  `outputs/benchmarks/smoke_popcosmos_binned_noagn_noNE/`. It completes and
  writes `benchmark_points.csv`, `benchmark_summary.json`,
  `delta_mag_by_band.png`, and per-level diagnostics. Residuals remain too
  large for science use, so the next step is baseline debugging, not prior
  training.
- Fixed the standalone benchmark runtime setup so it applies the config
  `runtime` block before importing `euclid_dsps.model`. This lets
  `runtime: auto` clear a stale shell-level `JAX_PLATFORMS=cuda` request before
  JAX is imported. Verified that `--help` works with `JAX_PLATFORMS=cuda`, that
  a CPU-forced `shine` smoke benchmark with `--n 1` completes at
  `outputs/benchmarks/smoke_popcosmos_binned_noagn_noNE_cpu_check/`, and that
  `runtime: auto` clears a shell-level `JAX_PLATFORMS=cuda` for the `--n 1`
  smoke run at
  `outputs/benchmarks/smoke_popcosmos_binned_noagn_noNE_auto_runtime_check/`.
- Generated the comparison report for the old fixed-nebular benchmark versus
  the new pure-stellar noNE benchmark at
  `outputs/report/popcosmos_binned_noagn_fsps_noNE_n500/report.md`. The
  comparison confirms that `stellar_only` median residuals improve from
  `0.0718` to `0.0531` mag and `stellar_plus_dust` non-finite rows drop from
  199 to 192, but all benchmark levels still fail the prior-readiness gates.
  Next priority is a pure-stellar normalization/UV/IGM audit before changing
  gas or training the first learned prior.

2026-05-28 FSPS/Prospector `n=500` benchmark available:

- Completed benchmark directory:
  `outputs/benchmarks/popcosmos_binned_noagn_fsps_n500/`.
- Outputs present: `benchmark_points.csv`, `benchmark_summary.json`,
  `delta_mag_by_band.png`, and `run.log`.
- The benchmark has 20,000 rows: 500 sampled points x 4 levels x 10 bands.
- It runs end to end, but residuals do not meet the recommended prior-v0
  criteria. `stellar_only` already has broad residuals larger than target,
  especially in LSST `u/g/r` and Euclid VIS, so the next priority is a clean
  pure-stellar baseline before tuning dust or gas.
- For dust/gas levels, some DSPS magnitudes are non-finite in the current
  benchmark output while the FSPS/Prospector reference remains finite. The next
  benchmark summary should report finite and non-finite counts explicitly per
  `(level, band)`.
- Updated remaining-work plan to prioritize pure-stellar SSP generation,
  stricter IMF metadata validation, richer benchmark diagnostics, staged reruns,
  and then dust/gas decisions.

2026-05-28 SSP generation assessment report completed:

- Wrote `outputs/reports/ssp_generation_assessment_2026-05-28/report.md`.
- Generated companion diagnostics in the same directory:
  `ssp_axis_contract.png`, `ssp_sed_flux_ratios.png`,
  `ssp_broadband_offsets.png`, `gas_grid_reference_match.png`,
  `fsps_prospector_benchmark_residuals.png`, plus CSV/JSON summary tables.
- Findings: the new Chabrier base SSP is DSPS-layout compatible with the legacy
  Kroupa `logGasU=-2` asset; the older DSPS v3.2 SSP keeps the same age and
  stellar-metallicity axes but uses a different wavelength grid; the Chabrier
  and Kroupa spectra are not flux-identical, as expected from the IMF change.
- The Chabrier gas grid exactly reproduces the base SSP at
  `gas_logu=-2`, `gas_logz=0` in the local HDF5 assets.
- The existing `n=500` FSPS/Prospector benchmark still misses the desired
  photometry residual targets, so the report recommends a pure-stellar
  Chabrier SSP and a cleaner layered benchmark before calling the forward model
  science-ready.

2026-05-28 SSP generation assessment report started:

- Scope: write a didactic Markdown report under `outputs/reports/` explaining
  how the new locally generated SSP assets are produced, whether they match the
  previous SSP assets closely enough for the current PopCosmos-like workflow,
  and how DSPS turns physical parameters plus SSPs into broadband photometry.
- Planned checks: inspect `scripts/generate_fsps_ssp_grid.py`,
  `scripts/fsps_grid_common.py`, the active config asset paths, and the HDF5
  schemas/statistics for the Chabrier, legacy Kroupa, and old DSPS SSP grids;
  generate compact comparison plots for the report.

2026-05-27 PopCosmos forward-model gap correction completed:

- Added recommended no-AGN configs:
  `configs/popcosmos_binned_noagn.yaml` and
  `configs/popcosmos_diffstar_noagn.yaml`. They use
  `model.agn_model: none`, Student-t photometric likelihood, and do not expose
  `ln_fagn` or `ln_tauagn` as free parameters.
- Updated PopCosmos-like configs and FSPS generation scripts to explicit
  Chabrier asset names and metadata. PopCosmos-like `load_context` now rejects
  Kroupa-named assets, missing/ambiguous IMF metadata, and `z_sun` mismatches.
- PopCosmos-like `z_sun` is now `0.0142`; legacy lognormal defaults remain
  compatible with `0.0134`.
- Added `dust_model: charlot_fall_powerlaw` as the explicit name for the
  previous behavior and `dust_model: prospector_fsps` as the current approximate
  JAX target mode. Direct FSPS/Prospector benchmarking is still required before
  calling this exact.
- Added the hard `Zgas >= Zstar` constraint, strict gas-grid axis/unit
  validation, optional enriched-grid emission-line correction support, and tests
  for the correction affecting only the line-covered filter.
- Added `scripts/benchmark_against_fsps_prospector.py`. It defines the CLI,
  sampling, summary/output contract, and uses an independent
  `prospect.sources.FastStepBasis` + python-FSPS + sedpy reference. It refuses
  unsupported mappings rather than falling back to DSPS.
- Smoke benchmark with `--n 5` now runs and writes
  `outputs/benchmarks/smoke_popcosmos_binned_noagn/benchmark_points.csv`,
  `benchmark_summary.json`, and `delta_mag_by_band.png`. Current residuals do
  not meet the recommended prior-readiness criteria; the largest UV/dust/gas
  discrepancies should be audited before any final learned prior.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `pytest tests/test_config.py` (28 passed);
  `pytest tests/test_model.py` (33 passed, 3 skipped);
  `pytest tests/test_benchmark.py tests/test_fsps_grid_scripts.py` (6 passed);
  full `pytest` (123 passed, 4 skipped);
  `git diff --check`.

2026-05-27 PopCosmos forward-model gap correction phase started:

- Scope: remove avoidable PopCosmos-like mismatches before learning a first
  prior, without touching the already implemented manual Student-t likelihood.
- `configs/fs2_phz1_science.yaml` is not present in this checkout; active
  inputs are the binned and Diffstar PopCosmos-like configs.
- Current PopCosmos-like assets/configs still reference Kroupa-named SSP data
  and gas metadata with `imf_type=2`; this phase will switch PopCosmos-like
  generation/config contracts to explicit Chabrier assets and make ambiguous
  HDF5 metadata fail loudly.
- Planned outputs: no-AGN configs for prior v0, PopCosmos-like `z_sun=0.0142`,
  explicit dust modes, gas metallicity and grid-axis validation, optional line
  correction support, and an FSPS/Prospector benchmark harness.

2026-05-27 group meeting report drafting:

- Preparing a presentation-ready scientific report summarizing the `main` to
  `dev` changes, the two SFH fitting paths, the PopCosmos-like configuration,
  the cleaner SSP/gas/AGN generation, and the two comparison runs in
  `outputs/runs/`.
- Wrote `outputs/reports/group_meeting_popcosmos_dev_2026-05-28.md` with:
  codebase changes from `master`/`main` to `dev`, scientific justification,
  gas/AGN asset details, chi-square vs Student-t run comparison, caveats,
  suggested figures, next steps, and a short meeting talk track.
- Report uses the local `master` branch as the base because no local `main`
  branch exists in this checkout.

2026-05-27 likelihood comparison report in progress:

- Comparing `outputs/runs/popcosmos_binned_chi2_512_b10` against
  `outputs/runs/popcosmos_binned_student_t_512_b10`.
- Goal: supervisor-ready report with paired Gaussian vs Student-t diagnostics,
  fit quality, redshift truth/proxy residuals, band residuals, and runtime
  comparison.
- Added `scripts/compare_likelihood_runs.py` to generate paired comparison
  reports.
- Generated `outputs/reports/popcosmos_binned_chi2_vs_student_t_512/`.
- Paired 512 shared galaxies. Student-t improves median reduced Gaussian chi2
  from 15.76 to 1.56, median |dz| from 0.291 to 0.120, median |dz|/(1+z) from
  0.150 to 0.060, and median mean absolute magnitude residual from 0.136 to
  0.049.
- Galaxy-by-galaxy tables are written, including top redshift improvements and
  degradations for Student-t relative to Gaussian chi2.
- `uv run python -m compileall scripts/compare_likelihood_runs.py` passed.
- `uv run --extra dev ruff check scripts/compare_likelihood_runs.py` passed.

2026-06-10 Diffsky simple recovery reset:

- Decision: stop using the Diffsky/Diffstar generated latents as first-order
  MAP recovery targets from broad-band photometry. They remain useful
  generated truths for later population diagnostics, but not for the first
  differentiable DSPS closure.
- Added a recommended simple HLTDS 04/14 path based on `popcosmos_bins`, HLTDS
  SSP/filter assets, native AB magnitudes, and explicit model-tolerance
  magnitudes rather than synthetic flux errors.
- New configs:
  `configs/diffsky_hltds_04_14_simple.yaml`,
  `configs/diffsky_hltds_04_14_simple_gpu.yaml`,
  `configs/diffsky_hltds_04_14_fixedz_closure.yaml`, and
  `configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml`.
- `diffsky-prepare-dataset` now supports `--no-synthetic-errors`; the simple
  fit configs do not consume `fluxerr_*` and instead use explicit
  `sigma_mag` model tolerance.
- Fit targets are restricted to basic direct truths that DSPS can plausibly
  recover: redshift, stellar mass, and recent SFR proxy. Halo mass, centrality,
  and sizes are retained as context columns but are not DSPS photometric fit
  targets.
- Remote/data investigation: HLTDS 04/14 is still the best current science
  target; HLTDS 03/31 is a high-z slice with the same schema; `sparse_cosmos`
  is useful for fast debugging but has a very coarse SSP grid; `smdpl` is too
  small locally; `lsstdesc_diffsky_data` is testdata, not a population dataset.
- Smoke diagnostics: fixed-z simple fit on 8 HLTDS objects gives median
  residual 0.073 mag and reduced chi2 ~1.25 with 0.10 mag model tolerance.
  The same setup with free redshift still has poor z recovery, so the next
  blocker is redshift initialization/multi-start/photo-z strategy, not
  Diffstar latent fitting.

2026-05-27 Student-t likelihood switch:

- Confirmed from Thorp et al., *Scaleable inference of galaxy properties and
  redshifts with a data-driven population model*, section 2.2.1, that the
  POP-COSMOS photometric likelihood is a per-band flux Student-t with 2 degrees
  of freedom.
- Added `fit.photometric_likelihood` with `gaussian` and `student_t` modes,
  defaulting to Gaussian chi-square for backward compatibility.
- Added `fit.student_t_dof`, default `2.0`, and CLI override
  `--fit-likelihood student_t`.
- Added mode-aware `fit_quality`/`reduced_fit_quality` diagnostics that follow
  the configured photometric likelihood. `chi2`/`reduced_chi2` remain Gaussian
  comparison metrics at the final parameters.
- Student-t is wired through independent MAP, population MAP, and NumPyro
  posterior sampling.
- Batch dashboards, objective components, redshift attractor summaries, workflow
  MAP-vs-population plots, and SED sample ranking now prefer mode-aware fit
  quality over Gaussian chi-square when available.

2026-05-27 verification:

- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run --extra dev ruff check euclid_dsps tests` passed.
- `uv run pytest` passed: 106 passed, 2 skipped.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --help` passed and exposes `--fit-likelihood {gaussian,student_t}`.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml fit
  --help` passed and exposes `--fit-likelihood {gaussian,student_t}`.
- Student-t one-row smoke passed:
  `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --index 0 --fit-maxiter 1 --fit-likelihood student_t --out
  outputs/runs/dev_popcosmos_student_t_one_iter --sed-samples 0
  --reporting-level light`.
- Student-t batch smoke passed:
  `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --limit 1 --batch-size 1 --fit-maxiter 1 --fit-likelihood student_t --out
  outputs/runs/dev_popcosmos_student_t_batch1_quality2 --sed-samples 0
  --reporting-level light`.
- Added the `gpu` optional dependency extra for the `uv` environment using the
  official JAX CUDA wheel path (`jax[cuda12]`).
- `uv sync --extra gpu` installed the CUDA JAX stack in `.venv`.
- `uv run --extra gpu python -c "import jax; print(jax.devices());
  print(jax.default_backend())"` reports `[CudaDevice(id=0)]` and `gpu`.

2026-05-26 combined Diffstar branch:

- Created `feature/pop-cosmos-diffstar` from `feature/pop-cosmos`.
- Source Diffstar commit: `497895a Add Diffstar PopCosmos model path` on
  `feature/diffstar`.
- Ported the Diffstar SFH path while keeping the current production FSPS
  gas/AGN grid scripts, `gas_grid_path`/`agn_template_path` config contract,
  dynamic JAX context arguments, and four-corner gas-grid interpolation.
- Added `configs/popcosmos_diffstar.yaml` as the combined gas + AGN + Diffstar
  configuration. `configs/` now contains only:
  `configs/popcosmos_binned.yaml` and `configs/popcosmos_diffstar.yaml`.
- Added the optional packaging extra `diffstar` for `diffstar` and `diffmah`.
- Updated tests and docs to describe both production configs and the Diffstar
  caveat that halo assembly currently uses Diffmah defaults until catalog MAH
  inputs are wired.

2026-05-26 verification:

- `git diff --check` passed.
- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run pytest` passed after installing the optional Diffstar extra: 103
  passed.
- `uv run --extra diffstar pytest tests/test_model.py -k diffstar` passed: 2
  passed, 17 deselected.
- `uv run --extra dev ruff check euclid_dsps scripts tests` passed.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --help` passed.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml fit
  --help` passed.
- `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra
  diffstar python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml
  fit --index 0 --fit-maxiter 1 --out
  outputs/runs/dev_popcosmos_diffstar_one_short --sed-samples 0` passed and
  wrote the expected diagnostic outputs.
