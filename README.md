# DSPS PopCosmos Forward Model

This repository contains a DSPS/JAX forward-model and fitting workflow for
Euclid + LSST photometry. The default production path is now the compressed
full PopCosmos-like model with gas and AGN enabled:

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
