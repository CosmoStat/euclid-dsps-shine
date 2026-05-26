# Repository Guidelines

## Project Structure & Module Organization

The standalone wrapper lives in `euclid_dsps/`. `model.py` is the DSPS boundary, `io.py` handles parquet rows and photometry units, `filters.py` loads or approximates transmission curves, `fit.py` contains optimization, `reporting/` writes tables and plots, and `workflows/` composes CLI workflows. Configurations live in `configs/`; the single active science setup is `configs/popcosmos_binned.yaml`. Local data and DSPS assets are under `Data/`. Generated artifacts belong in `outputs/` and should not be treated as source.

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

Run the main checks:

```bash
python -m compileall euclid_dsps scripts
python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit --index 0 --out outputs/runs/dev_popcosmos_one --sed-samples 1
python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit --limit 20 --batch-size 5 --out outputs/runs/dev_popcosmos_batch --sed-samples 4
```

Use `fit` only with a small `--limit` while iterating because it runs one optimizer per galaxy.

## Coding Style & Naming Conventions

Use Python 3.11+ with type hints and small, explicit functions. Keep DSPS-specific calls isolated in `model.py`; other modules should use the wrapper dataclasses and CSV/JSON outputs. Prefer snake_case for functions, variables, YAML keys, and output filenames. Keep comments short and focused on non-obvious scientific or data-contract choices.

## Testing Guidelines

For changes, run `compileall`, `pytest`, one-row `fit`, and a small batch `fit`. If touching posterior logic, also run:

```bash
python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml posterior --index 0 --num-warmup 10 --num-samples 10 --out outputs/runs/dev_popcosmos_posterior_one
```

## Commit & Pull Request Guidelines

This checkout has no git history, so no existing commit convention can be inferred. Use concise imperative messages, for example `Add FS2 redshift batch diagnostics`. PRs should describe the data/config used, commands run, output paths inspected, and any scientific limitations such as approximate filters or missing truth parameters.

## Planning Workflow

Keep `PLAN.md` as the living implementation plan. At the start and end of each phase prompt, update it with completed work, changed priorities, and newly discovered blockers. Prefer small phase commits over broad mixed commits, especially when changing scientific assumptions, runtime behavior, or output formats.
