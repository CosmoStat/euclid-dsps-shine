# DSPS Plan

Living plan for the simplified Euclid + LSST DSPS workflow.

## Active Goal

Build a DSPS-like pipeline that fits SED parameters from Euclid + LSST
photometry, compares fitted quantities to FS2 truth/proxy columns, and keeps
workflow/docs small enough to audit.

## Current Phase

Implementation phase after `science_fit_new_features_1000` and
`science_fit_calib_offsets_1000` diagnostics. Runtime cost diagnostics,
redshift-attractor summaries, diagnostic-only nebular line/filter crossings, and
Diffstar assessment docs are now implemented. MAP redshift stays free and
single-start for quick tests. MCMC/posterior, redshift grid/global search, and
Diffstar SFH implementation are deferred.

## Active Public Workflow

Use one science config:

```bash
configs/fs2_phz1_science.yaml
```

Use three commands:

```bash
euclid-dsps fit --limit 100 --batch-size 50 --sed-samples 8 --out outputs/runs/science_fit
euclid-dsps posterior --index 0 --num-warmup 100 --num-samples 200 --out outputs/runs/posterior_one
euclid-dsps check --kind cosmos --limit 20 --out outputs/check/cosmos
```

## Current Science Contract

- Free parameters: `z_obs`, `log10_formed_mass_msun`, `sfh_t_peak`,
  `sfh_tau`, `log10_metallicity`.
- Not free in main config: `log10_sfr`, `dust_av`.
- Redshift: no `phz_median` initialization and no photo-z prior in the science
  preset. `z_obs` is free in MAP/posterior; `redshift.initial: fixed` only
  supplies the MAP start value.
- Dust: COSMOS proxy dust is row-injected in the main config. Do not interpret
  this as a fitted DSPS dust parameter. Next plan may remove dust proxy from
  the main science likelihood if it keeps adding confusion.
- Metallicity truth: proxy only, because gas-phase O/H is not stellar
  metallicity.
- Emission-line columns: diagnostics only until a line model/assets exist. The
  new SSP contains `ssp_emline_luminosity`, `ssp_emline_name`, and
  `ssp_emline_wave`; current model still uses only `ssp_flux`.
- Priors: `weak_physical` for science, `flat_debug` for debugging,
  `popcosmos_like` reserved until exact mapping/units exist.

## Diagnostics From Latest Runs

- Fixed band-calibration offsets must be sign-audited. The first calibrated run
  used the wrong sign convention for LSST offsets and worsened LSST residuals.
- Redshift MAP shows attractor bands near repeated fitted redshift values. This
  is consistent with local MAP optimization on a multi-modal broad-band
  likelihood, weak population prior, missing/ambiguous nebular control, and
  mass/SFH/metallicity compensation.
- New SSP differs from old SSP in both resolution and nebular assets:
  `ssp_flux` moved from 5994 to 11149 wavelength samples and now ships
  explicit 166-line `ssp_emline_*` datasets. Strong-line ratios in `ssp_flux`
  changed substantially, especially near H-alpha and near-IR sulphur lines.
- COSMOS proxy SED height mismatch remains an audit target, not proof that DSPS
  normalization alone is wrong.

## Near-Term Implementation Plan

### Priority 1: Runtime and Throughput Diagnostics

Add per-run and per-batch timing so every science run reports cost.

- Save `fit_wall_time_s`, `compile_time_s` when measurable, `optimize_time_s`,
  `n_galaxies`, `batch_size`, `rows_per_second`, and `seconds_per_galaxy`.
- Save device metadata: JAX backend, device kind/name, requested runtime
  platform, and whether GPU was actually used.
- Save cost metrics:
  - `gpu_hours_total = wall_time_s * n_gpu / 3600` when backend is GPU;
  - `gpu_hours_per_galaxy = gpu_hours_total / n_galaxies`;
  - CPU runs should report backend `cpu` and leave GPU-hour metrics null.
- Add `performance_summary.json` and `performance_by_batch.csv`.
- Add optional benchmark mode later:
  `--benchmark-batch-sizes 128,256,512,1024,2048`, to measure scaling versus
  batch size/chunk size on the same catalog slice.

### Priority 2: Dust Handling Cleanup

Keep dust honest until a real dust prior/model exists.

- Main science config should keep `dust_av` inactive by default.
- Evaluate whether `dust_model: cosmos_proxy_fixed` should remain in the
  likelihood or become SED-diagnostic-only.
- Add a config switch for:
  - `dust_model: none`
  - `dust_model: cosmos_proxy_fixed`
  - `dust_model: dsps_free`
- If `cosmos_proxy_fixed`, do not plot dust as inferred and report COSMOS dust
  as proxy-only metadata.
- If `dsps_free`, require explicit prior bounds and plot active dust diagnostics.

### Priority 3: Simple Population Prior

Do not implement full POP-COSMOS yet. Implement a simple, explicit, auditable
population prior first.

- Add `prior: bounded_physical_v2`.
  - tighter redshift, mass, SFH, and metallicity bounds;
  - soft penalties near unphysical corners;
  - no learned density.
- Add `prior: fs2_empirical_simple`.
  - learn/use empirical `p(z)`;
  - learn/use `logM | z`;
  - learn/use `logSFR | logM,z` using catalog proxy only;
  - learn/use `logZ_proxy | logM`;
  - penalize rare combinations without calling this POP-COSMOS.
- Store prior components separately:
  `prior_z`, `prior_mass_z`, `prior_sfr_mass_z`, `prior_metallicity_mass`,
  `prior_total`, `chi2_flux`, `loss_total`.
- Keep truth/proxy errors out of the science loss.
- Reuse current POP-COSMOS-style diagnostic plots as prior-learning QA:
  color-redshift planes, mass-SFR by redshift bin, sSFR-mass, metallicity-mass,
  and dust proxy-mass if dust remains diagnostic.

### Priority 4: Nebular Diagnostics and Switch

This is medium difficulty, not trivial. The new SSP has line tables, but current
`ssp_flux` already contains line-like spikes, so a naive line-table addition may
double-count emission.

- Expose config:
  `nebular_emission: none | ssp_flux | emline_table`.
- Define current behavior as `ssp_flux`.
- Read and store `ssp_emline_luminosity`, `ssp_emline_name`,
  `ssp_emline_wave` in the DSPS context when present.
- First implement diagnostics, not a new likelihood:
  - line observed wavelength versus `z`;
  - filter overlap per line and per band;
  - expected line-contribution ranking by band and redshift;
  - compare candidate redshift attractors to strong-line/filter crossings.
- For `emline_table`, require a no-double-count strategy before using it in the
  forward model:
  - either a continuum-only SSP asset;
  - or a controlled subtraction/addition convention;
  - or multiple SSP grids over `logGasU`/`logGasZ` if gas parameters become
    free.
- Full free nebular model is deferred until assets expose variable gas
  ionization/metallicity, not just one fixed `logGasU=-2.0`, `logGasZ=0.0`
  table.

### Deferred: MCMC and Global Redshift Strategy

- For catastrophic galaxies, run NUTS later with free `z_obs` to identify
  multi-modal posterior structure.
- MAP-only remains a fast diagnostic, not final inference.
- Redshift grid/global strategy is deferred; user will handle later.

### Deferred: Diffstar SFH

- Current lognormal SFH is temporary.
- Keep model interface ready for replacing `sfh_t_peak` and `sfh_tau` by
  Diffstar parameters.
- Do not over-tune SFR recovery until Diffstar lands.

## DSPS Paper Parameter-Count Audit

The base DSPS paper is a differentiable SPS framework, not a single fixed
photometric-fitting pipeline with one canonical parameter count. It demonstrates
autodiff gradients for SFH, stellar metallicity, nebular emission, and dust
attenuation. Therefore the plan should not claim that the current 5-parameter
MAP fit matches "the DSPS paper".

Current repo fit count:

- 5 active MAP parameters in science config:
  `z_obs`, `log10_formed_mass_msun`, `sfh_t_peak`, `sfh_tau`,
  `log10_metallicity`.
- 0 active dust parameters under `cosmos_proxy_fixed`.
- 0 active nebular parameters.

Audit task before matching DSPS-literature parameterization:

- Identify the exact DSPS/Diffstar example to emulate.
- Count its active SFH parameters, metallicity parameters, dust parameters, and
  nebular parameters.
- Map each to available catalog proxies and SSP assets.
- Only then create a `dsps_paper_like` or `diffstar_science` preset.

## Completed In Current Phase

- Public CLI simplified to `fit`, `posterior`, `check`.
- Old configs moved to `configs/legacy/`.
- PHZ interval prior support removed.
- Fast-grid/warmstart public config removed.
- Science config removes PHZ interval priors and now avoids `phz_median`
  entirely for science fitting.
- Sample priors must match actual free parameters.
- Batch SED diagnostics now select worst-fit galaxies.
- Blank failed heatmaps no longer get written.
- Docs reduced to active workflow and science assumptions.
- Verification passed: `compileall`, full `pytest`, CLI CPU EDA smoke, and CLI
  CPU one-row/batch fit smoke with SED diagnostics.
- LSST query/config/docs updated for `*_el_model3_ext_odonnell_ext_error`
  columns.
- The 10-band preset now loads LSST and Euclid passbands from repository
  `filters/`.
- COSMOS noisy/forward target-set diagnostics now support LSST as well as
  Euclid.
- Verification passed after LSST update: `compileall`, `uv run ruff check
  euclid_dsps tests scripts`, and full `pytest`.
- Added `euclid_dsps.photometry` as the central AB/Jy/Fnu-cgs conversion module.
- Added explicit flux likelihood helpers in `euclid_dsps.likelihood`.
- MAP and MCMC now use `fit.likelihood_space`, defaulting to flux-space.
- Science config explicitly sets `fit.likelihood_space: flux` and a 2 percent
  flux-error floor.
- Fit photometry comparison rows now include the flux error used and `chi_flux`.
- Verification passed after PR 1: `compileall`, `uv run ruff check euclid_dsps
  tests scripts`, full `pytest`, and CLI smoke fit on `configs/smoke_test.yaml`.
- Science config now uses a single fixed MAP redshift start with
  `prior_z.mode: none`; redshift multi-start support was removed.
- Batch fit outputs report redshift initialization mode, prior mode, and initial
  z, without multi-start bookkeeping.
- Reduced chi2 now means `chi2 / dof`; the previous per-band quantity is
  exposed separately as `chi2_per_band`.
- Reporting summaries and objective components now use `chi_flux` when flux
  likelihood diagnostics are available.
- Verification passed after redshift/chi2 phase: `compileall`, full `pytest`,
  `uv run ruff check euclid_dsps tests scripts`, one-row CLI smoke, and
  two-row batch CLI smoke.
- Added `euclid_dsps.semantics` for inferred, active, and inactive parameters.
- COSMOS proxy dust marks `dust_av` as inactive and prevents fixed dust from
  appearing in inferred-vs-proxy plots/metrics.
- SED diagnostics now label photometry markers as model-anchored ratios.
- SED diagnostics now include flux-space residuals
  `(F_obs - F_model) / sigma_F` per band.
- Verification passed after free-redshift update: `compileall`, full `pytest`,
  `uv run ruff check euclid_dsps tests scripts`, and two-row CLI smoke with SED
  diagnostic output.
- Runtime preset `auto` no longer forces CUDA and clears stale
  `JAX_PLATFORMS=cuda` inside the process.
- Science config defaults to `runtime: auto`.
- Batch SED diagnostics split the configured sample budget between best and
  worst fits.
- COSMOS proxy SED outputs now include normalization factor and normalization
  residual metadata.
- Flux-space non-detection policy `gaussian_flux` keeps finite non-positive
  fluxes with valid errors; `upper_limit` is reserved and fails validation.
- Optional fixed per-band calibration offsets can be applied to the model
  during likelihood evaluation.
- Added POP-COSMOS-style color-redshift and physical-population diagnostics when
  coverage is sufficient.
- Added a minimal `Snakefile` for reproducible science fits.
- Current verification: `compileall`, full `pytest` (`70 passed, 1 skipped`),
  two-row science CLI smoke with `runtime:auto`, and one-row
  `conda run -n shine euclid-dsps fit` smoke. `ruff` is not installed in the
  current shell/conda base, so it could not be run this pass.
- Added per-run and per-batch performance summaries with seconds/galaxy,
  galaxies/second, device/backend metadata, and GPU-hour per galaxy when JAX
  uses a GPU backend.
- Added MAP redshift-attractor diagnostics:
  `*_redshift_attractors.csv` and `*_redshift_attractors.png`.
- Added diagnostic-only SSP nebular-line inventory and line/filter crossing
  outputs from `ssp_emline_*` when available. This does not change the
  likelihood.
- Dust is no longer compared to catalog truth/proxy in parameter metrics, even
  if `dust_av` is made free in a future config.
- Added DSPS parameter-count audit docs and Diffstar integration assessment.
- Removed API reference from the main docs navigation to keep science docs
  focused.
- Current verification after this pass: `compileall`, full `pytest`
  (`76 passed, 1 skipped`), and smoke fit on `configs/smoke_test.yaml`.

## Remaining Work

- Implement runtime throughput diagnostics before the next large science run.
- Decide whether `cosmos_proxy_fixed` dust remains in the main likelihood or
  becomes SED-diagnostic-only.
- Implement `bounded_physical_v2`, then `fs2_empirical_simple` as a simple
  population prior. Do not call either one POP-COSMOS.
- Validate exact POP-COSMOS parameter mapping and units before implementing any
  future `popcosmos_like`.
- Add true synthetic recovery workflow with DSPS-generated photometry and known
  parameters, independent of FS2/OpenUniverse catalog semantics.
- Add nebular diagnostics from `ssp_emline_*`, then decide whether a safe
  `emline_table` model is possible without double-counting lines already in
  `ssp_flux`.
- Audit exact DSPS/Diffstar paper parameterization before claiming a paper-like
  preset.
- Keep lognormal SFH as temporary; replace with Diffstar later.
- Add a small formal smoke dataset that exercises the full `fit` command
  without private FS2 data.
- Consider splitting `workflows/core.py` and `reporting/core.py` after science
  behavior stabilizes.
