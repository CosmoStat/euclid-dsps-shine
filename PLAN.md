# Plan

## Current State

The repository now has one active configuration:

```text
configs/popcosmos_binned.yaml
```

It is standalone and represents the most advanced PopCosmos-like DSPS/JAX path:

- LSST `ugrizy` plus Euclid VIS/Y/J/H photometry.
- Flux-space likelihood with catalog per-band flux errors.
- Seven-bin PopCosmos-like SFH ratios.
- Single stellar metallicity.
- Charlot-Fall age-dependent dust.
- Madau95-style approximate IGM.
- Generated FSPS gas SSP grid:
  `Data/popcosmos_gas_ssp_grid.h5`.
- Generated FSPS/CLUMPY AGN template grid:
  `Data/popcosmos_agn_template_grid.h5`.
- Full 16-parameter fit including gas and AGN parameters.

Older config variants, presets, examples, and partial smoke configs have been
removed from source. Tests now target the single active config plus low-level
synthetic fixtures.

## Scientific Caveats

- This is PopCosmos-like, not yet an audited reproduction of the full
  POP-COSMOS population model.
- Independent gas and stellar metallicity variation in FSPS is useful for
  fitting but not fully self-consistent for all nebular line ratios.
- AGN normalization follows the repository convention
  `fagn * integrated stellar Lbol`; exact FSPS/CLUMPY bolometric normalization
  still needs an independent audit.
- The IGM model is a stable Madau95-style approximation.
- The fit and post-fit batch prediction paths pass large SSP/gas/AGN arrays as
  dynamic JAX arguments so JIT can stay enabled without compiling the gas grid
  as a multi-GiB XLA constant.
- Gas-grid interpolation uses direct four-corner bilinear interpolation in
  `(gas_lgmet, gas_lgu)`, avoiding a batch-scaled intermediate over the full
  gas-ionization axis.

## Standard Commands

Generate/validate FSPS gas and AGN assets as documented in
`docs/source/data_download.rst`.

Short one-row fit:

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

Verification:

```bash
uv run python -m compileall euclid_dsps scripts
uv run pytest tests
uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit --help
```

## Remaining Work

- Audit AGN normalization against FSPS internals or external CLUMPY convention.
- Scale the CUDA batch run beyond the validated small batch once GPU memory and
  wall time are acceptable.
- Add synthetic recovery tests with known DSPS-generated parameters.

## Latest Verification

2026-05-26 15:00 CEST cleanup verification:

- `find configs -maxdepth 3 -type f` returns only
  `configs/popcosmos_binned.yaml`.
- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run pytest` passed: 92 passed, 1 skipped.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --help` passed.
- Sphinx build was not run because Sphinx is not installed in the current
  `uv` environment (`No module named sphinx`).

2026-05-26 JIT/GPU cleanup:

- Added dynamic model-argument binding for large context arrays in
  `euclid_dsps.model`.
- Updated single-row, independent batched Adam, and population Adam paths to
  pass SSP/gas/AGN/filter arrays as explicit JAX arguments to jitted functions.
- Updated post-fit batch prediction/reporting paths (`predict_batch_mags`,
  `predict_batch_derived`, `predict_batch_seds`) to use the same dynamic
  context-array arguments, so SED diagnostics do not re-capture the gas grid as
  an XLA constant after optimization.
- Reworked gas-grid interpolation to gather only the four bracketing
  gas-metallicity/gas-ionization slabs before interpolation, instead of
  materializing an intermediate across the full gas-ionization grid.
- Single-row JIT is enabled by default again; set `fit.jit: false` only for
  debugging.
- Verification passed: `uv run python -m compileall euclid_dsps scripts`, `uv
  run pytest tests/test_workflows_smoke.py tests/test_model.py
  tests/test_fit_memory.py` (23 passed), `uv run pytest tests/test_model.py
  tests/test_fit_memory.py tests/test_config.py` (38 passed), `uv run pytest`
  (94 passed, 1 skipped), and `uv run python -m euclid_dsps.cli --config
  configs/popcosmos_binned.yaml fit --help`.

2026-05-26 user CUDA run inspection:

- `outputs/runs/dev_popcosmos_gpu_batch4/batch_fit_results.parquet` exists with
  16 rows.
- `outputs/runs/dev_popcosmos_gpu_batch4/batch_fit_photometry_comparison.parquet`
  exists with 160 rows.
- The reported crash happened after primary fit outputs were written, during
  post-fit SED/report generation with `--sed-samples 2`.

2026-05-26 final pre-commit audit:

- Documentation and launch commands now reference only
  `configs/popcosmos_binned.yaml`; stale references to removed configs were
  removed from source docs and `AGENTS.md`.
- Gas and AGN HDF5 metadata/shape inspection passed without loading full
  datasets:
  gas `(7, 7, 12, 107, 11149)`, AGN `(9, 11149)`.
- AGN script default `agn_tau` grid now matches the documented/generated
  PopCosmos grid: `5, 10, 20, 30, 40, 60, 80, 100, 150`.
- Verification passed: `uv run python scripts/generate_fsps_gas_grid.py
  --help`, `uv run python scripts/generate_fsps_agn_grid.py --help`, `uv run
  python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit --help`,
  `uv run python -m compileall euclid_dsps scripts`, `uv run pytest` (95
  passed, 1 skipped), and `git diff --check`.
- `uv run ruff check euclid_dsps scripts tests` could not run because `ruff` is
  not installed in the current `uv` environment.
