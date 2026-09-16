# Repository Guidelines

## Project Structure & Module Organization

The standalone wrapper lives in `euclid_dsps/`. `model.py` owns the production DSPS forward-model boundary, `io.py` handles parquet rows and photometry units, `filters.py` loads or approximates transmission curves, `fit.py` contains optimization, `reporting/` writes tables and plots, and `workflows/` composes CLI workflows. Narrow DSPS utilities are also used by `filters.py`, `synthetic_diffsky/`, and numerical audit modules; do not introduce a second production forward model there. Configurations live in `configs/`. The current experiment surface is the controlled FENIKS data/prior pipeline, the spline-15D AVI/EM/factorial chain, exact-posterior geometry diagnostics, and COSMOS2020 external-data benchmarks using the repository's own inference methods. See `docs/source/active_workflows.rst` before changing launch scripts. Local data and DSPS assets are under `Data/`. Generated artifacts belong in `outputs/` and should not be treated as source.

## Build, Test, and Development Commands

Install in the existing environment:

```bash
conda activate shine
python -m pip install -e .
```

The project should also be kept compatible with a future `uv` workflow. When dependency or packaging changes are made, verify or update:

```bash
uv sync
uv run python -m compileall euclid_dsps scripts
uv run euclid-dsps --help
```

If GPU JAX setup differs between `conda shine` and `uv`, document the exact commands and caveats instead of assuming one install path works for both.

Run the repository checks:

```bash
python -m compileall euclid_dsps scripts
python -m ruff check euclid_dsps scripts tests
python -m black --check euclid_dsps scripts tests
python -m pytest
python -m euclid_dsps.cli --help
```

For a data-independent workflow smoke test, use:

```bash
python -m euclid_dsps.cli --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml amortized-synthetic-smoke --mock-decoder --n-objects 16 --epochs 1 --batch-size 8 --out outputs/runs/dev_amortized_synthetic
```

Use `fit` only with a small `--limit` while iterating because it runs one optimizer per galaxy. Record the exact dataset, checkpoint, config, and output path for data-dependent checks.

## Coding Style & Naming Conventions

Use Python 3.11+ with type hints and small, explicit functions. Keep production forward-model calls behind `model.py`; narrow DSPS utilities outside it must remain isolated and covered by numerical contract tests. Prefer snake_case for functions, variables, YAML keys, and output filenames. Keep comments short and focused on non-obvious scientific or data-contract choices.

## Testing Guidelines

For changes, run `compileall`, Ruff, Black, and the tests that cover the edited workflow. Run the full suite before merging. If touching the current AVI/EM chain, include:

```bash
python -m pytest tests/test_avi_experiments.py tests/test_avi_inference.py tests/test_avi_overnight.py tests/test_avi_em.py tests/test_avi_em_factorial.py tests/test_avi_next_validation.py tests/test_geometry_nuts.py
```

Posterior changes must preserve dense joint draws and valid weights; a displayed median is not a replacement for a posterior distribution. Keep truth out of observed training and checkpoint selection, and require saved chains plus convergence/support diagnostics before scientific promotion.

## Commit & Pull Request Guidelines

Use concise imperative commit messages, for example `Add FS2 redshift batch diagnostics`. Keep scientific-contract, runtime, and documentation cleanups in focused commits. PRs should describe the data/config used, commands run, output paths inspected, and any scientific limitations such as approximate filters or missing truth parameters.

## Planning Workflow

Keep `PLAN.md` as the chronological implementation ledger: add current work at the start and close it out at the end without deleting older completed entries. Use `docs/source/active_workflows.rst` for the maintained workflow map. Prefer small phase commits over broad mixed commits, especially when changing scientific assumptions, runtime behavior, or output formats.
