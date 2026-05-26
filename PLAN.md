# Plan

## Current State

Branch objective: integrate the Diffstar SFH implementation from
`feature/diffstar` into the current PopCosmos FSPS gas/AGN workflow without
losing the generated-grid scripts, GPU/JIT memory fixes, or documentation
cleanup.

The repository currently has two active production configurations:

```text
configs/popcosmos_binned.yaml
configs/popcosmos_diffstar.yaml
```

They are standalone and represent the most advanced PopCosmos-like DSPS/JAX
paths:

- LSST `ugrizy` plus Euclid VIS/Y/J/H photometry.
- Flux-space likelihood with catalog per-band flux errors.
- Seven-bin PopCosmos-like SFH ratios in `popcosmos_binned.yaml`.
- Six-free-parameter Diffstar SFH in `popcosmos_diffstar.yaml`.
- Single stellar metallicity.
- Charlot-Fall age-dependent dust.
- Madau95-style approximate IGM.
- Generated FSPS gas SSP grid:
  `Data/popcosmos_gas_ssp_grid.h5`.
- Generated FSPS/CLUMPY AGN template grid:
  `Data/popcosmos_agn_template_grid.h5`.
- Full 16-parameter fit including SFH, gas, and AGN parameters in both configs.

Older config variants, presets, examples, and partial smoke configs have been
removed from source. Tests now target the active configs plus low-level
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

Diffstar short one-row fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_diffstar.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_diffstar_one_short \
  --sed-samples 1
```

Verification:

```bash
uv run python -m compileall euclid_dsps scripts
uv run pytest tests
uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit --help
uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml fit --help
```

## Remaining Work

- Audit AGN normalization against FSPS internals or external CLUMPY convention.
- Scale the CUDA batch run beyond the validated small batch once GPU memory and
  wall time are acceptable.
- Add synthetic recovery tests with known DSPS-generated parameters.

## Latest Verification

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
