# Euclid DSPS SHINE

Standalone workflow for testing native `dsps` on simulated Euclid FS2 catalog data. The current target is practical: load the local parquet catalog, select one galaxy or stream many galaxies, run DSPS with the catalog redshift, convert the SED to VIS/Y/J/H photometry, compare to simulated fluxes, and fit a small parameter set.

## Data Contract

Default config: `configs/fs2_phz1.yaml`.

Default catalog: `Data/Euclid FS2 LC galaxy catalog_phz1.parquet`.

Required columns for the default workflow:

- `phz_mode_1`: redshift used by DSPS as `z_obs`.
- `true_redshift_halo`: truth redshift used only for diagnostics.
- `euclid_vis`, `euclid_nisp_y`, `euclid_nisp_j`, `euclid_nisp_h`: simulated fluxes.
- Morphology/halo columns listed in `extra_columns`: copied into EDA, `selected_galaxy.json`, and batch galaxy summaries.

Fluxes are interpreted as `Fnu` in `erg/s/cm^2/Hz`. The code also supports `microjy` for future Q1 MER-style catalogs. The current Euclid filters are approximate top-hats; replace them with exact HDF5 transmission curves when available.

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

Fit one galaxy with a chi-square likelihood in AB magnitudes:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one --out outputs/runs/phz1_fit_one
```

Run the forward model over many galaxies without loading the full parquet:

```bash
euclid-dsps --config configs/fs2_phz1.yaml run-batch --limit 1000 --batch-size 500 --out outputs/runs/phz1_batch
```

Fit many galaxies. Keep `--limit` small until the model and filters are final:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --limit 25 --batch-size 25 --out outputs/runs/phz1_fit_batch
```

## Outputs

Single-galaxy runs write `selected_galaxy.json`, `model_parameters.json`, `sed.csv`, `sed.png`, `photometry_comparison.csv`, and `photometry_comparison.png`.

Batch runs write the flat comparison table plus `*_summary_by_band.csv`, `*_summary_by_galaxy.csv`, `*_dashboard.png`, `*_residuals_by_band.png`, `*_observed_vs_model.png`, and `*_redshift_truth.png`.

`fit-batch` additionally writes `batch_fit_results.csv` with recovered parameters per galaxy.

## Model Notes

`euclid_dsps/model.py` is the only module that calls native DSPS. It builds a lognormal SFH table, uses `calc_rest_sed_sfh_table_lognormal_mdf`, applies DSPS dust attenuation, and calls `calc_obs_mag` for each configured filter.

By default, `z_obs` is fixed per row from `phz_mode_1`. To fit redshift as well, add this under `fit.free_parameters`:

```yaml
z_obs:
  initial: from_base
  bounds: [0.001, 6.0]
```

The current catalog does not contain true stellar mass, SFH, metallicity, or dust parameters. Therefore "truth" diagnostics currently mean simulated photometry and `true_redshift_halo`. If future catalogs include physical truth columns, add them under `truth.parameter_columns` in the YAML config and they will be propagated into batch summaries.
