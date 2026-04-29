# Euclid DSPS SHINE

Standalone workflow for testing native `dsps` on simulated Euclid FS2 catalog data. The current target is practical: load the local parquet catalog, select one galaxy or stream many galaxies, run DSPS with the catalog redshift, convert the SED to VIS/Y/J/H photometry, compare to simulated fluxes, and fit a small parameter set.

## Data Contract

Default config: `configs/fs2_phz1.yaml`.

Default catalog: `Data/Euclid FS2 LC galaxy catalog_phz1.parquet`.

Default NISP passbands are converted from Euclid FITS throughput files into DSPS HDF5 files under `filters/converted/`. Rebuild them with:

```bash
python scripts/convert_euclid_filters.py filters/NISP-PHOTO-PASSBANDS-V1-*_throughput.fits
```

Required columns for the default workflow:

- `phz_mode_1`: redshift used by DSPS as `z_obs`.
- `true_redshift_halo`: truth redshift used only for diagnostics.
- `euclid_vis`, `euclid_nisp_y`, `euclid_nisp_j`, `euclid_nisp_h`: simulated fluxes.
- Morphology/halo columns listed in `extra_columns`: copied into EDA, `selected_galaxy.json`, and batch galaxy summaries.

Fluxes are interpreted as `Fnu` in `erg/s/cm^2/Hz`. The code also supports `microjy` for future Q1 MER-style catalogs. NISP Y/J/H use the provided Euclid throughput curves; VIS still uses the configured top-hat until a VIS throughput curve is added.

## Install

Use the existing `shine` conda environment:

```bash
conda activate shine
python -m pip install -e .
```

## Commands

Inspect schema, stats, flux/color distributions, and redshift diagnostics:

```bash
euclid-dsps --config configs/fs2_phz1.yaml eda --out outputs/eda_phz1
```

Run DSPS for one selected galaxy:

```bash
euclid-dsps --config configs/fs2_phz1.yaml run-one --out outputs/runs/phz1_one
```

Fit one galaxy with a pure-JAX Adam chi-square likelihood in AB magnitudes:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one --out outputs/runs/phz1_fit_one
```

Run the forward model over many galaxies with JAX-vmapped chunks:

```bash
euclid-dsps --config configs/fs2_phz1.yaml run-batch --limit 1000 --batch-size 500 --out outputs/runs/phz1_batch
```

Fit many galaxies with one JAX-vmapped Adam optimizer per parquet chunk. Increase `--batch-size` until GPU memory becomes the bottleneck:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --limit 1024 --batch-size 64 --out outputs/runs/phz1_fit_batch
```

Fit a chunked hierarchical population MAP model:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-population --limit 1024 --batch-size 64 --out outputs/runs/phz1_population_fit
```

Use `--all` instead of `--limit` to process the full parquet catalog.

## Outputs

Single-galaxy runs write `selected_galaxy.json`, `model_parameters.json`, `sed.csv`, `sed.png`, `photometry_comparison.csv`, and `photometry_comparison.png`.

Batch runs write the flat comparison table plus `*_summary_by_band.csv`, `*_summary_by_galaxy.csv`, `*_dashboard.png`, `*_residuals_by_band.png`, `*_observed_vs_model.png`, and `*_redshift_truth.png`.

`fit-batch` additionally writes `batch_fit_results.csv` and `batch_fit_trace.csv` with recovered parameters and optimizer diagnostics. `fit-population` writes `population_fit_results.csv`, `population_hyperparameters.csv`, and `population_fit_trace.csv`. Progress bars are shown for batch commands.

## Model Notes

`euclid_dsps/model.py` is the only module that calls native DSPS. It builds a lognormal SFH table, uses `calc_rest_sed_sfh_table_lognormal_mdf`, applies DSPS dust attenuation, and calls `calc_obs_mag` for each configured filter. The fit path keeps these operations in JAX until final report writing, so `jax.value_and_grad` can differentiate the photometric likelihood. Batch fitting uses `jax.vmap` over each parquet chunk and runs on the active JAX device, normally GPU when available.

The default fit follows the DSPS-paper process as closely as the current catalog allows:

- Physical parameters plus redshift define an SFH, metallicity, and dust model.
- Native DSPS maps those parameters to a rest-frame SED.
- DSPS photometry maps the SED to observed VIS/Y/J/H AB magnitudes.
- A Gaussian chi-square compares model magnitudes to simulated Euclid photometry.
- Adam optimizes bounded parameters through a smooth transform.
- `fit-population` adds a shared Gaussian population prior over the free parameters and jointly optimizes per-galaxy parameters plus population `mu`/`sigma`.

The default free parameters are `log10_sfr`, `dust_av`, and `log10_metallicity`. Bounds and optimizer settings live under `fit` in `configs/fs2_phz1.yaml`.

By default, `z_obs` is fixed per row from `phz_mode_1`. To fit redshift as well, add this under `fit.free_parameters`:

```yaml
z_obs:
  initial: from_base
  bounds: [0.001, 6.0]
```

The current catalog does not contain true stellar mass, SFH, metallicity, or dust parameters. Therefore parameter recovery can only be checked against simulated photometry and `true_redshift_halo`, not against true SFH/dust/metallicity. The hierarchical mode is a population-level MAP approximation, not full posterior sampling. If future catalogs include physical truth columns, add them under `truth.parameter_columns` in the YAML config and they will be propagated into batch summaries.
