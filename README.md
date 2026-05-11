# Euclid DSPS SHINE

Standalone workflow for testing native `dsps` on simulated Euclid FS2 catalog data. The current target is practical: load the local parquet catalog, select one galaxy or stream many galaxies, run DSPS with the catalog redshift, convert the SED to VIS/Y/J/H photometry, compare to simulated fluxes, and fit a small parameter set.

## Data Contract

Default config: `configs/fs2_phz1.yaml`.

Optional 10-band config: `configs/fs2_phz1_10band.yaml`. This explicitly
activates LSST `ugrizy` in addition to Euclid VIS/Y/J/H. The default config
remains Euclid-only.

Default catalog: `Data/Euclid FS2 LC galaxy catalog_phz1.parquet`.

Default Euclid passbands (VIS/Y/J/H) are loaded from ASCII throughput files under `filters/`:
- `filters/Euclid_VIS.vis.dat`
- `filters/Euclid_NISP.Y.dat`
- `filters/Euclid_NISP.J.dat`
- `filters/Euclid_NISP.H.dat`

Required columns for the default workflow:

- `z_phz`: photometric redshift used by DSPS as fixed `z_obs`.
- `z_true`: truth redshift used only for diagnostics.
- `euclid_vis`, `euclid_nisp_y`, `euclid_nisp_j`, `euclid_nisp_h`: simulated fluxes.
- `sed_cosmos_1`, `sed_cosmos_2`, `ebv_cosmos_*`, `ext_curve_cosmos_*`: COSMOS template reconstruction inputs.
- `frac_cosmos_1`, `frac_cosmos_2`: component fractions used for COSMOS proxy SED reconstruction.
- `euclid_vis_abs`, `euclid_nisp_y_abs`, `euclid_nisp_j_abs`, `euclid_nisp_h_abs`: rest-frame 10 pc Euclid fluxes used to normalize COSMOS proxy SEDs.
- `log_sfr_true`: catalog log SFR, compared to derived DSPS `log10_sfr_at_obs`, not to the internal SFH amplitude.
- `metallicity_true`: gas-phase oxygen abundance, converted to a metallicity proxy as `metallicity_true - 10.61`.
- `dust_ebv_true`: catalog color excess, converted to an `A_V` proxy as `4.05 * E(B-V)`.
- Morphology/halo columns listed in `extra_columns`: copied into EDA, `selected_galaxy.json`, and batch galaxy summaries.

Fluxes are interpreted as `Fnu` in `erg/s/cm^2/Hz`. The code also supports `microjy` for future Q1 MER-style catalogs. All Euclid bands (VIS/Y/J/H) use the provided ASCII throughput curves.

## Install

Use the existing `shine` conda environment:

```bash
conda activate shine
python -m pip install -e .
```

For documentation and quality tooling:

```bash
python -m pip install black ruff sphinx sphinx-rtd-theme
```

## Documentation

Sphinx documentation lives in `docs/source/`:

```bash
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

Start with:

- `docs/source/architecture.rst` for project boundaries and refactor roadmap.
- `docs/source/data_download.rst` for the CosmoHub SQL query and data contract.
- `docs/source/run_setup.rst` for config parameters and CLI workflows.
- `docs/source/testing.rst` for unit tests, smoke fixtures, and CI checks.

## Quality Checks

```bash
find euclid_dsps scripts -name '*.py' -exec python -m black --check {} \;
python -m ruff check euclid_dsps scripts tests
python -m pytest tests
python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py
python -m sphinx -W --keep-going -b html docs/source docs/build/html
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

Opt into the ten-band diagnostic setup:

```bash
euclid-dsps --config configs/fs2_phz1_10band.yaml fit-one --out outputs/runs/phz1_10band_fit_one
```

Reconstruct COSMOS-template proxy SEDs from the local LePhare data:

```bash
euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed --limit 10 --plot-samples 12 --out outputs/runs/phz1_cosmos_sed
```

Compare COSMOS proxy SEDs against fitted DSPS SEDs for a small sample:

```bash
euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed --limit 3 --fit-dsps --out outputs/runs/phz1_cosmos_sed_fit_dsps
```

Compare COSMOS proxy SEDs against a chunked population MAP DSPS fit:

```bash
euclid-dsps --config configs/fs2_phz1.yaml cosmos-sed --limit 64 --batch-size 32 --population-dsps --plot-samples 16 --out outputs/runs/phz1_cosmos_sed_population_dsps
```

Sample one galaxy posterior. For quick debugging, use fixed-step HMC to cap the
number of DSPS gradient evaluations per draw:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one \
  --index 0 \
  --bayesian \
  --sampler hmc \
  --num-warmup 120 \
  --num-samples 400 \
  --num-chains 1 \
  --num-steps 8 \
  --target-accept-prob 0.8 \
  --no-progress \
  --out outputs/runs/phz1_mcmc_row_0_fast
```

For a more careful single-galaxy posterior, use NUTS with a modest tree depth:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-one \
  --index 0 \
  --bayesian \
  --sampler nuts \
  --num-warmup 300 \
  --num-samples 800 \
  --num-chains 1 \
  --max-tree-depth 6 \
  --target-accept-prob 0.8 \
  --no-progress \
  --out outputs/runs/phz1_mcmc_row_0_nuts
```

Run the forward model over many galaxies with JAX-vmapped chunks:

```bash
euclid-dsps --config configs/fs2_phz1.yaml run-batch --limit 1000 --batch-size 500 --out outputs/runs/phz1_batch
```

Fit many galaxies with one JAX-vmapped Adam optimizer per parquet chunk. Increase `--batch-size` until GPU memory becomes the bottleneck:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --limit 1024 --batch-size 64 --out outputs/runs/phz1_fit_batch
```

To rerun only a chosen subset of catalog rows, pass a text or CSV file containing one `row_index` per line:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --row-indices-file outputs/hmc_rows.txt --batch-size 32 --out outputs/runs/phz1_fit_subset
```

Sample a small batch of independent galaxy posteriors with NumPyro NUTS/HMC:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-batch --bayesian --limit 5 --batch-size 1 --out outputs/runs/phz1_sample_batch
```

This same `--row-indices-file` option works for `run-batch`, `fit-batch`, and `fit-population`, which makes the recommended workflow straightforward: MAP on a large batch, inspect `batch_fit_results.csv`, write the selected `row_index` values to a file, then run Bayesian sampling only on that subset.

Fit a chunked hierarchical population MAP model:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-population --limit 1024 --batch-size 64 --out outputs/runs/phz1_population_fit
```

Run the complete diagnostic workflow in one command: independent MAP on a large batch, automatic HMC subset selection, Bayesian sampling on that subset, population MAP initialized from the independent MAP result, and comparison plots:

```bash
euclid-dsps --config configs/fs2_phz1.yaml fit-workflow \
  --limit 1000 \
  --batch-size 64 \
  --hmc-n 20 \
  --hmc-select stratified \
  --num-warmup 300 \
  --num-samples 800 \
  --num-chains 4 \
  --max-tree-depth 7 \
  --out outputs/runs/phz1_fit_workflow
```

`fit-workflow` writes `map/`, `hmc/`, `population/`, and `comparison/` subdirectories under the chosen output path. The HMC row list is saved as `hmc_row_indices.txt`.

Use `--all` instead of `--limit` to process the full parquet catalog.

## Outputs

Single-galaxy runs write `selected_galaxy.json`, `model_parameters.json`, `sed.csv`, `sed.png`, `photometry_comparison.csv`, and `photometry_comparison.png`.

`cosmos-sed` writes `cosmos_sed_validation.json`, `cosmos_sed_diagnostics.csv`, `cosmos_seds.parquet`, `cosmos_sed_example.png`, `cosmos_sed_sample_set.png`, `cosmos_template_pair_heatmap.png`, `cosmos_fraction_diagnostics.png`, and `synthetic_vs_catalog_abs_flux.png`. With `--compare-dsps`, `--fit-dsps`, or `--population-dsps`, it also writes branch-1 rest-frame SED metrics and branch-2 observed photometry residual metrics. Fit modes add `cosmos_dsps_fit_results.csv`, `cosmos_dsps_fit_trace.csv`, and population hyperparameters when requested.

Batch runs write the flat comparison table plus `*_summary_by_band.csv`, `*_summary_by_galaxy.csv`, `*_dashboard.png`, `*_residuals_by_band.png`, `*_observed_vs_model.png`, and `*_redshift_truth.png`.

`fit-batch` additionally writes `batch_fit_results.csv` and `batch_fit_trace.csv` with recovered parameters, derived quantities, and optimizer diagnostics. `fit-population` writes `population_fit_results.csv`, `population_hyperparameters.csv`, and `population_fit_trace.csv`. Progress bars are shown for batch commands.

Bayesian single-galaxy runs first compute an Adam/MAP solution, then initialize NumPyro HMC/NUTS from that point unless `--no-map-init` is passed. They write `posterior_samples.csv`, `posterior_derived_samples.csv`, `posterior_comparable_samples.csv`, `posterior_summary.csv`, `posterior_corner.png`, `posterior_corner_with_truth.png`, `posterior_trace.png`, `posterior_predictive_photometry.csv`, and `posterior_predictive_photometry.png`. The truth-overlaid corner uses comparable quantities: derived `log10_sfr_at_obs`, fitted `dust_av`, and fitted `log10_metallicity`. Bayesian batch runs use the same per-galaxy MAP initialization and write `batch_posterior_summary.csv`, `batch_posterior_samples.csv`, `batch_posterior_predictive.csv`, and batch posterior diagnostic plots.

## Model Notes

`euclid_dsps/model.py` is the only module that calls native DSPS. It builds a lognormal SFH table, uses `calc_rest_sed_sfh_table_lognormal_mdf`, applies DSPS dust attenuation, and calls `calc_obs_mag` for each configured filter. The fit path keeps these operations in JAX until final report writing, so `jax.value_and_grad` can differentiate the photometric likelihood. Batch fitting uses `jax.vmap` over each parquet chunk and runs on the active JAX device, normally GPU when available.

The default fit follows the DSPS-paper process as closely as the current catalog allows:

- Physical parameters plus redshift define an SFH, metallicity, and dust model.
- Native DSPS maps those parameters to a rest-frame SED.
- DSPS photometry maps the SED to observed VIS/Y/J/H AB magnitudes.
- A Gaussian chi-square compares model magnitudes to simulated Euclid photometry.
- Adam optimizes bounded parameters through a smooth transform.
- `fit-population` adds a shared Gaussian population prior over the free parameters and jointly optimizes per-galaxy parameters plus population `mu`/`sigma`.
- `--bayesian` uses NumPyro HMC/NUTS to sample the posterior density after an Adam/MAP initialization instead of returning only one MAP point.
- `--sampler hmc` is the fast debug path. It uses a fixed number of leapfrog steps, controlled by `--num-steps`, so runtime is predictable.
- `--sampler nuts` is more adaptive and usually more robust, but it can be much slower because each draw may require many leapfrog steps up to the configured `--max-tree-depth`.
- `--chain-method vectorized` can run multiple chains on one accelerator, but it increases memory use and is mainly useful after a single-chain run looks healthy.

The default free parameters are `log10_sfr`, `dust_av`, and `log10_metallicity`. Bounds and optimizer settings live under `fit` in `configs/fs2_phz1.yaml`.

By default, `z_obs` is fixed per row from `z_phz`. This is intentional: with only VIS/Y/J/H photometry, freeing redshift lets redshift absorb dust/SFR/metallicity mismatch. To run a photo-z recovery experiment anyway, add this under `fit.free_parameters`:

```yaml
z_obs:
  initial: from_base
  bounds: [0.001, 6.0]
```

When physical truth columns are configured under `truth.parameter_columns`, batch and population fits write inferred-vs-truth diagnostics alongside the photometry reports. For `fit-population`, inspect `population_corner_parameters_with_truth.png`, `population_parameter_distributions_with_truth.png`, `population_parameter_truth_metrics.csv`, `population_fit_parameter_truth.png`, and `population_fit_trace_truth.png`.

Truth comparisons are only made between like-for-like quantities where possible. The internal fitted `log10_sfr` is an SFH amplitude, so `log_sfr_true` is compared to derived `fit_log10_sfr_at_obs`. Dust and metallicity are proxy comparisons: `dust_ebv_true` is converted to `A_V` with `scale: 4.05`, and gas-phase oxygen abundance is converted to a total-metallicity proxy. These are diagnostics, not strict proof of stellar metallicity or attenuation-law recovery.

The hierarchical mode is a population-level MAP approximation. Use `--bayesian` on small samples when posterior uncertainty and parameter degeneracies matter.

MCMC/NUTS is useful here because four Euclid bands cannot uniquely identify redshift, SFR, dust, and metallicity. MAP gives one best point; MCMC shows whether that point is well constrained or one of many degenerate solutions. Run it on selected validation rows, not the full catalog.
