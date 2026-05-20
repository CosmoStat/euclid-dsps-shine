# Euclid DSPS SHINE

DSPS-like SED inference from Euclid + LSST photometry.

Main goal:

- fit simple DSPS parameters from catalog photometry;
- compare recovered redshift, mass, SFR, metallicity, and dust proxies to FS2 truth/proxy columns;
- compare inferred DSPS SEDs to COSMOS-template proxy SEDs;
- keep MAP fast for batches and keep HMC/NUTS posterior sampling for selected galaxies.

## Quick Start

```bash
conda activate shine
python -m pip install -e .
```

Recommended science config:

```text
configs/fs2_phz1_science.yaml
```

Run MAP fits and save SED diagnostics:

```bash
euclid-dsps fit \
  --limit 1000 \
  --batch-size 512 \
  --sed-samples 16 \
  --out outputs/runs/science_fit
```

Run one galaxy:

```bash
euclid-dsps fit --index 0 --out outputs/runs/row0_fit
```

Run HMC/NUTS posterior checks on selected rows:

```bash
euclid-dsps posterior \
  --row-indices-file outputs/runs/science_fit/hmc_row_indices.txt \
  --num-warmup 300 \
  --num-samples 800 \
  --out outputs/runs/posterior_subset
```

Run sanity checks without fitting:

```bash
euclid-dsps check --kind eda --out outputs/check/eda
euclid-dsps check --index 0 --out outputs/check/row0_forward
euclid-dsps check --kind cosmos --limit 20 --out outputs/check/cosmos
```

## Simple CLI

Public commands:

- `fit`: MAP fit. Uses `--index` for one row, otherwise batch mode.
- `posterior`: HMC/NUTS for one row or a small row list.
- `check`: EDA, forward pass, or standalone COSMOS proxy SED checks.

Legacy commands still work for old scripts: `fit-one`, `fit-batch`, `run-one`, `run-batch`, `cosmos-sed`, `fit-population`, `fit-workflow`.

## Science Config

`configs/fs2_phz1_science.yaml` is intentionally short. It uses:

- `bands: lsst_euclid_10` for LSST `ugrizy` + Euclid VIS/Y/J/H;
- `column_groups` instead of a long `extra_columns` list;
- `dust_model: cosmos_proxy_fixed` so COSMOS dust columns are row-injected, not inferred as DSPS `dust_av`;
- broad non-circular priors for the current DSPS parameterization;
- `plot_ground_truth: true` so SED diagnostics overlay the COSMOS proxy when local resources exist.

Expanded normalized config is written in run outputs as `normalized_config.json`.

## Outputs

Batch MAP writes:

- `batch_fit_results.*`: recovered parameters, derived SFR, truth/proxy columns, optimizer diagnostics;
- `batch_fit_photometry_comparison.*`: observed vs model photometry;
- `batch_fit_parameter_audit.csv`: labels fixed, free, derived, or row-injected columns;
- `sed_diagnostics/`: DSPS SED, COSMOS proxy SED, filters, and photometry constraints for sampled rows;
- `normalized_config.json`: exact expanded config used for audit.

Posterior runs write `posterior_samples.csv`, `posterior_summary.csv`, posterior predictive photometry, and diagnostics.

## Priors

Current priors are `weak_physical`: broad, stabilizing, and non-circular.

Do not call them POP-COSMOS priors yet. POP-COSMOS uses learned population priors from rich COSMOS photometry; matching that requires exact parameter mapping, units, and selection treatment before implementation.

## Development Checks

```bash
uv run python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py
uv run euclid-dsps --help
uv run pytest
```

GPU JAX setup may differ between `conda shine` and `uv`. Verify long GPU runs with:

```bash
python scripts/check_jax_gpu.py --require-nvidia --hold-seconds 10
```
