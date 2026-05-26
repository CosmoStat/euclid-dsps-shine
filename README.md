# DSPS PopCosmos

DSPS/JAX fitting workflow for Euclid + LSST photometry with one active
PopCosmos-like configuration:

```text
configs/popcosmos_binned.yaml
```

That config fits the full 16-parameter branch: redshift, stellar mass, six SFH
bin ratios, stellar metallicity, Charlot-Fall dust, gas metallicity, gas
ionization, AGN fraction, and AGN torus optical depth. The gas and AGN pieces
come from generated FSPS HDF5 assets in `Data/`.

## Setup

```bash
conda activate shine
python -m pip install -e .
```

Install FSPS/python-FSPS in the same environment:

```bash
cd "$HOME/src"
export SPS_HOME="$HOME/src/fsps"
git clone https://github.com/cconroy20/fsps.git "$SPS_HOME"

cd /home/maxime/src/DSPS-pop-cosmos
export SPS_HOME="$HOME/src/fsps"
uv pip install fsps
```

Sanity check:

```bash
python -c "import fsps; sp=fsps.StellarPopulation(sfh=0); print(len(sp.wavelengths)); print(sp.isoc_library, sp.spec_library)"
```

## Data And Assets

The config expects:

```text
Data/Euclid FS2 LC galaxy catalog_phz1.parquet
Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5
Data/popcosmos_gas_ssp_grid.h5
Data/popcosmos_agn_template_grid.h5
```

Download the reference SSP if needed:

```bash
python scripts/manage_ssp.py download fsps_v0.4.7_u-2.0
python scripts/manage_ssp.py test Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5
```

Generate and validate the gas grid:

```bash
export SPS_HOME="$HOME/src/fsps"

python scripts/generate_fsps_gas_grid.py \
  --output Data/popcosmos_gas_ssp_grid.h5 \
  --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --overwrite

python scripts/generate_fsps_gas_grid.py \
  --output Data/popcosmos_gas_ssp_grid.h5 \
  --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --validate-only
```

Generate and validate the AGN grid:

```bash
python scripts/generate_fsps_agn_grid.py \
  --output Data/popcosmos_agn_template_grid.h5 \
  --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --agn-tau-grid 5 10 20 30 40 60 80 100 150 \
  --fagn-normalization 1.0 \
  --tage-gyr 1.0 \
  --stellar-logzsol 0.0 \
  --overwrite

python scripts/generate_fsps_agn_grid.py \
  --output Data/popcosmos_agn_template_grid.h5 \
  --base-ssp Data/fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5 \
  --validate-only
```

Expected generated shapes:

```text
gas ssp_flux: (7, 7, 12, 107, 11149)
AGN template_lnu_per_lbol: (9, 11149)
```

## Run

Use CPU-safe JAX settings on WSL unless you have a CUDA-enabled JAX install:

```bash
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

One-row short fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_one_short \
  --sed-samples 1
```

Batched fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned.yaml \
  fit --limit 20 \
  --batch-size 5 \
  --out outputs/runs/dev_popcosmos_batch \
  --sed-samples 4
```

Posterior smoke:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned.yaml \
  posterior --index 0 \
  --num-warmup 10 \
  --num-samples 10 \
  --out outputs/runs/dev_popcosmos_posterior_one
```

## Checks

```bash
uv run python -m compileall euclid_dsps scripts
uv run pytest tests
uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit --help
```

The optimizer and post-fit batch prediction paths pass large SSP/gas/AGN arrays
as dynamic JAX arguments to jitted model calls, so XLA does not compile the full
gas grid as a closed-over constant. Gas-grid interpolation gathers only the four
bracketing gas-metallicity/gas-ionization slabs. For GPU runs, use a
CUDA-enabled JAX install and enough device memory for the ~2.6 GiB gas grid plus
optimizer/reporting buffers; reduce `--batch-size` first if memory is
exhausted.
