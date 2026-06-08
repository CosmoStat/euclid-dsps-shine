# DSPS PopCosmos Forward Model

This repository contains a DSPS/JAX forward-model and fitting workflow for
multi-band photometry. The validation priority is now OpenUniverse/Diffsky:
use LSST+Roman OpenUniverse subsets as the main generative-truth validation
surface, and keep Euclid FS2 as a comparison/domain-shift diagnostic dataset.

The default PopCosmos-like DSPS model path remains the compressed full model
with gas and AGN enabled:

```text
configs/popcosmos_binned_compressed.yaml
configs/popcosmos_diffstar_compressed.yaml
```

The dense full-AGN configs remain available for dense-vs-compressed and
FSPS/Prospector closure checks:

```text
configs/popcosmos_binned.yaml
configs/popcosmos_diffstar.yaml
```

The no-AGN configs remain available for fallback/debug runs:

```text
configs/popcosmos_binned_noagn.yaml
configs/popcosmos_diffstar_noagn.yaml
```

The binned config uses PopCosmos-style step SFH bins. The Diffstar config keeps
the same redshift, dust, gas, and AGN surface but swaps the SFH model for the
reduced six-parameter Diffstar path. The compressed configs keep the same
science surface while replacing dense resident spectral grids by SVD
`basis/coeff/scale` assets.

## What The Model Does

At fit time the code does not call FSPS. It loads pre-generated HDF5 spectral
assets and evaluates the forward model in JAX:

1. Load a Chabrier FSPS SSP grid with MIST isochrones and C3K spectra.
2. Convert the sampled SFH into weights on the SSP age grid.
3. Interpolate stellar metallicity and multiply by formed stellar mass.
4. Apply the Prospector/FSPS-like dust model.
5. Add nebular emission from the compressed Chabrier gas SSP grid.
6. Add the compressed FSPS-native AGN component grid.
7. Apply the FSPS-like IGM/order convention.
8. Redshift, apply luminosity distance, integrate through LSST+Euclid filters.
9. Fit catalog fluxes with the configured flux-space likelihood.

The details and glossary are documented in `docs/source/forward_model.rst`.

## OpenUniverse / Diffsky Validation

The first OpenUniverse target is a compact 14-band LSST+Roman table:

```text
LSST u,g,r,i,z,y + Roman W146,R062,Z087,Y106,J129,H158,F184,K213
```

OpenUniverse SkyCatalog files are read per nside=32 HEALPix from
`galaxy_<hpix>.parquet` and `galaxy_flux_<hpix>.parquet`, joined on
`galaxy_id`, then normalized into a small parquet with truth fluxes, noisy
observed fluxes, flux errors, and masks:

```bash
python -m euclid_dsps.cli \
  --config configs/openuniverse_lsst_roman_14.yaml \
  openuniverse-prepare \
  --input-root Data/openuniverse/raw \
  --hpix 9812 9813 \
  --limit 10000 \
  --out Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet
```

OpenUniverse fluxes are currently kept in their native
`photon_per_sec_cm2` unit. The code deliberately does not fake a photon-rate
to `fnu_cgs` conversion; a filter-aware DSPS photon-rate decoder or validated
conversion remains a tracked TODO. Direct public columns such as `redshift`,
`redshiftHubble`, and `um_source_galaxy_obs_sm` are treated as truth when
present. Diffsky/Diffstar SFH, dust, metallicity, and halo latents are only
`generated_truth` after an actual Diffsky export exists.

Inventory downloaded truth fields and the optional low-resolution SED HDF5:

```bash
python -m euclid_dsps.openuniverse.cli inventory-truth \
  --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
  --input-root Data/openuniverse/raw \
  --hpix 10307 \
  --sed \
  --sed-sample-limit 3 \
  --out outputs/reports/openuniverse_truth_inventory_10307
```

Export the directly available basic truth table and compute B=14 encoder
feature stats:

```bash
python -m euclid_dsps.openuniverse.cli extract-truth \
  --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
  --out Data/openuniverse/processed/ou_truth_basic.parquet \
  --schema-out Data/openuniverse/processed/truth_schema.json

python -m euclid_dsps.openuniverse.cli feature-stats \
  --input Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
  --limit 10000 \
  --out outputs/runs/openuniverse_feature_stats_10307/feature_stats.json
```

Run a data-side LSST SED-to-flux closure without touching the DSPS decoder:

```bash
python -m euclid_dsps.openuniverse.cli sed-flux-closure \
  --catalog Data/openuniverse/processed/ou_lsst_roman_14_subset.parquet \
  --sed Data/openuniverse/raw/galaxy_sed_10307.hdf5 \
  --bands lsst_u lsst_g lsst_r lsst_i lsst_z lsst_y \
  --limit 200 \
  --out outputs/reports/openuniverse_sed_flux_closure_10307_lsst200
```

If a real Diffsky latent export becomes available, merge it explicitly with
`python -m euclid_dsps.openuniverse.cli merge-external-truth`; missing latents
are not inferred from the public files.

See `docs/source/openuniverse.rst` for the dataset contract, truth policy,
unit caveats, and next CLI phases.

## FS2 Amortized Posterior Prototype

The optional FS2 amortized workflow trains a JAX encoder and RealNVP prior
jointly on Euclid FS2 photometry while keeping DSPS fixed as the physical
decoder:

```text
10 FS2 fluxes + 10 FS2 errors -> q_psi(x | flux, err) -> theta = h(x) -> DSPS
```

`x` is the unconstrained 16D latent and `theta` is the bounded PopCosmos-like
16D parameter vector, including redshift. The encoder input dimension is 20 by
default because the per-band errors are part of the posterior information.
Flux features use robust per-band `asinh(flux / flux_scale)` normalization,
while errors use a log transform. The feature builder is generic in band count:
OpenUniverse LSST+Roman uses 14 fluxes + 14 errors = 28 encoder features.
The KL term is estimated by Monte Carlo as
`logq - logp`; the standard Gaussian/Gaussian closed-form VAE KL is not valid
because the prior is a RealNVP.

Asset-free smoke:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_fs2_realnvp.yaml \
  amortized-synthetic-smoke \
  --mock-decoder \
  --n-objects 64 \
  --epochs 2 \
  --out outputs/runs/dev_amortized_synthetic
```

Small FS2 training run, if the compressed assets exist:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_fs2_realnvp.yaml \
  amortized-train-fs2 \
  --limit 32 \
  --batch-size 8 \
  --epochs 2 \
  --n-samples 1 \
  --out outputs/runs/dev_amortized_fs2
```

This command is verbose by default and shows a per-epoch progress bar. Add
`--quiet` to reduce logs or `--no-progress` if your terminal does not render
progress bars cleanly.

Small FS2 inference run from the best checkpoint:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_fs2_realnvp.yaml \
  amortized-infer-fs2 \
  --checkpoint outputs/runs/dev_amortized_fs2/checkpoints/best.eqx \
  --limit 32 \
  --batch-size 8 \
  --posterior-samples 32 \
  --decoder-sample-chunk-size 1 \
  --out outputs/runs/dev_amortized_fs2_infer
```

`--decoder-sample-chunk-size 1` keeps the fixed DSPS posterior predictive
decode at the same peak-memory scale as training. Increase it only after a
small GPU memory check. Inference also writes normalized posterior predictive
diagnostics, including residuals by band, top chi-square objects, feature-scale
histograms, redshift proxy comparison when catalog columns are available, a
redshift PIT diagnostic, contour-style posterior corners, and learned RealNVP
prior diagnostics. The prior diagnostics include `learned_prior_samples.parquet`,
`learned_prior_summary.json`, `learned_prior_logprob_hist.png`,
`learned_prior_corner.png`, and `posterior_vs_learned_prior_corner.png`.
When FS2 catalog proxy columns are available, inference also writes
`catalog_proxy_comparison.parquet` plus proxy plots for stellar mass and SFR.

See `docs/source/amortized_inference.rst` for the ELBO, RealNVP KL rationale,
architecture, progressive checkpoints, training diagnostics, and scientific
limitations. During training, `training_log.csv` includes
`encoder_grad_norm` and `prior_grad_norm`, which verifies that the MLP encoder
and RealNVP prior are optimized jointly in the same ELBO step.

## Full AGN vs No-AGN

`configs/popcosmos_binned_compressed.yaml` is the default production full
model. It fits all stellar, dust, gas, redshift, and AGN parameters:

```text
ln_fagn    = log AGN luminosity fraction
ln_tauagn  = log AGN torus optical depth
```

`configs/popcosmos_binned_noagn.yaml` uses the same stellar, dust, gas,
redshift, filters, and likelihood setup, but sets `agn_model: none` and removes
`ln_fagn` and `ln_tauagn`. Use no-AGN only for ablations, memory debugging, or
science tests where AGN is intentionally excluded.

## Setup

```bash
conda activate shine
python -m pip install -e .
```

For Diffstar:

```bash
python -m pip install -e '.[diffstar]'
```

For the optional FS2 amortized posterior prototype:

```bash
python -m pip install -e '.[amortized]'
```

For documentation/tests in the `uv` environment:

```bash
uv sync
uv run python -m compileall euclid_dsps scripts
uv run pytest tests
```

## Required Data Assets

The active compressed full-AGN pipeline expects:

```text
Data/Euclid FS2 LC galaxy catalog_phz1.parquet
Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5
Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5
Data/popcosmos_chabrier_gas_ssp_grid.h5
Data/popcosmos_chabrier_agn_component_ssp_grid.h5
Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5
```

The PopCosmos-like configs reject ambiguous/Kroupa SSP metadata. Active assets
must declare Chabrier IMF metadata and `z_sun = 0.0142`.

## Generate Assets

Install FSPS/python-FSPS in the same environment used for generation:

```bash
cd "$HOME/src"
export SPS_HOME="$HOME/src/fsps"
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

conda activate shine
python -m pip install fsps
python -c "import fsps; sp=fsps.StellarPopulation(sfh=0); print(len(sp.wavelengths)); print(sp.isoc_library, sp.spec_library)"
```

Generate the fixed-nebular reference SSP:

```bash
python scripts/generate_fsps_ssp_grid.py \
  --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --overwrite
```

This SSP includes a fixed nebular setup and is the axis contract used by the gas
and AGN grids. It is not used for the gas-free benchmark levels.

Generate the pure-stellar SSP used by benchmark gas-free levels:

```bash
python scripts/generate_fsps_ssp_grid.py \
  --stellar-only \
  --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
  --overwrite
```

This SSP disables nebular emission and nebular continuum. It is the correct
asset when the benchmark asks for `stellar_only` or `stellar_plus_dust`.

Generate the gas grid:

```bash
python scripts/generate_fsps_gas_grid.py \
  --output Data/popcosmos_chabrier_gas_ssp_grid.h5 \
  --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --base-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --overwrite
```

This grid is raw FSPS/CLOUDY gas. It is physically motivated and benchmarked
against FSPS broad-band photometry, but it does not include the learned
PopCosmos line-by-line corrections.

Generate the FSPS-native AGN component grid:

```bash
python scripts/generate_fsps_agn_component_grid.py \
  --output Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
  --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --overwrite
```

This grid stores the FSPS AGN contribution as a component per SSP age and
metallicity. It is the current validated full-AGN path.

Build the compressed runtime assets:

```bash
python scripts/build_compressed_ssp_grid.py \
  --input Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --output Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5 \
  --k 64 --basis-dtype float32 --coeff-dtype float16 --overwrite

python scripts/build_compressed_gas_grid.py \
  --input Data/popcosmos_chabrier_gas_ssp_grid.h5 \
  --output Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5 \
  --k 64 --basis-dtype float16 --coeff-dtype float16 --overwrite

python scripts/build_compressed_agn_component_grid.py \
  --input Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
  --output Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5 \
  --k 12 --factor-fagn --basis-dtype float32 --coeff-dtype float16 --overwrite
```

Validate existing assets without importing python-FSPS:

```bash
python scripts/generate_fsps_ssp_grid.py \
  --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --validate-only
python scripts/generate_fsps_ssp_grid.py \
  --stellar-only \
  --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
  --validate-only
python scripts/generate_fsps_gas_grid.py \
  --output Data/popcosmos_chabrier_gas_ssp_grid.h5 \
  --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --base-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --validate-only
python scripts/generate_fsps_agn_component_grid.py \
  --output Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
  --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --validate-only
```

## Run Fits

CPU-safe runtime:

```bash
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

One-row full AGN smoke:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_compressed_fullagn_one_short \
  --sed-samples 1
```

Production compressed full AGN batch:

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

No-AGN fallback:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_noagn.yaml \
  fit --limit 20 \
  --batch-size 5 \
  --out outputs/runs/dev_popcosmos_noagn_batch20 \
  --sed-samples 4
```

Diffstar smoke:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_diffstar_compressed.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_diffstar_compressed_fullagn_one_short \
  --sed-samples 1
```

## FSPS/Prospector Benchmark

The current closure benchmark is:

```bash
mkdir -p outputs/matplotlib_cache

MPLCONFIGDIR=outputs/matplotlib_cache python scripts/benchmark_against_fsps_prospector.py \
  --runtime cpu \
  --config configs/popcosmos_binned.yaml \
  --agn-component-grid Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
  --agn-host-attenuation fsps_diffuse_unit_tau \
  --agn-igm-order fsps_after_igm \
  --agn-baked-attenuation fsps_powerlaw_unit_tau \
  --agn-baked-dust-index -0.7 \
  --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn stellar_plus_agn stellar_plus_dust_plus_agn stellar_plus_gas_plus_agn full_agn \
  --n 500 \
  --seed 0 \
  --out outputs/benchmarks/popcosmos_binned_full_forward_fsps_closure_n500
```

Report:

```text
outputs/report/popcosmos_binned_full_forward_fsps_closure_n500/report.md
```

Result: the production broad-band levels `full_noagn` and `full_agn` pass the
FSPS/Prospector-like bright finite target. This is not a claim that official
PopCosmos emission-line calibration has been reproduced.

## Caveats

- The gas grid is raw FSPS/CLOUDY. `emission_line_corrections: none` means no
  PopCosmos learned line-by-line corrections are applied.
- Production fitting should use the compressed config. Dense full-AGN configs
  remain useful for reference checks, but they require much more memory.
- Very faint magnitude-space rows can be non-finite. Inference is flux-space.

## Checks

```bash
python -m compileall euclid_dsps scripts
pytest tests/test_config.py tests/test_model.py tests/test_benchmark.py
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```
