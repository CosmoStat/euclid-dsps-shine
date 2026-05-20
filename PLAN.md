# DSPS Plan

Living plan for the DSPS wrapper. Update this file at the start/end of each phase prompt so the project keeps one current source of truth.

## Operating Rules

- Work in small phases with regular commits.
- Keep production workflow simple before adding richer inference.
- Only implement features supported by local data columns and local SSP/filter assets.
- Keep hot paths JAX-first and GPU-friendly. Avoid pandas/NumPy inside model, loss, optimizer, and batched forward paths.
- Support current `conda` environment `shine`; add `uv` setup so future runs can use reproducible `uv` workflows too.
- Write outputs progressively during long runs, then compile plots/reports after compute.
- Every run that fits or forwards galaxies should be able to emit SED diagnostics for a sample.

## Environment Plan

Current supported path:

```bash
conda activate shine
python -m pip install -e .
```

Add `uv` path:

```bash
uv sync
uv run euclid-dsps --help
uv run python -m compileall euclid_dsps scripts/quickstart_one_galaxy.py
```

Tasks:

- Audit `pyproject.toml` dependencies for full `uv sync` support.
- Decide how GPU JAX is installed under `uv` without breaking `conda shine`.
- Document CPU-only vs GPU install commands.
- Add a smoke test command for each env path.

## Phase 0 - Baseline And Cleanup

Status: active cleanup.

Commit target: `Document current DSPS workflow state`

Tasks:

- Document commands that currently work.
- List active features vs experimental/inactive features.
- Mark removed quench, burst, PHZ-prior, and complex SFH terms as removed from
  active code/configs.
- Reduce dense science assessment prose into decisions, limitations, and next checks.
- Keep SSP path rationale documented: SSP tables are model inputs, not inferred by DSPS.

Done when:

- `README.md`, `docs/source/*`, and this plan agree.
- Sphinx docs build.

## Phase 1 - Mandatory SED Diagnostics

Status: in progress, baseline implemented.

Commit target: `Add SED fit diagnostics`

Goal: understand why inferred SEDs diverge from `cosmos_sed`.

Tasks:

- [x] Add SED output for forward pass and MAP fit.
- [x] Add ground-truth `cosmos_sed` overlay when local data provides it.
- [x] Plot sample diagnostics:
  - inferred DSPS SED
  - COSMOS/ground-truth SED, both scaled and unscaled diagnostic shape
  - observed photometric constraints on the SED panel
  - filter transmission curves placed near the lower plot axis
  - residuals by band
  - fitted/input parameters: redshift, mass, SFR, metallicity, dust
- [x] Add CLI options:
  - `--save-sed-samples N`
  - `--plot-filters`
  - `--plot-ground-truth`
- [ ] Add worst-fit automatic SED sampling, not only first-N rows.
- [ ] Add compact SED summary metrics directly to batch reports.

Done when:

- A small `fit-batch` run produces diagnostic SED plots.
- Missing ground truth is reported clearly, not faked.

## Phase 2 - Simple Workflow

Status: started.

Commit target: `Simplify user-facing workflows`

Goal: make commands obvious and consistent.

Workflows:

- [x] `forward`: no fit, catalog/config parameters in, photometry/SED/plots out.
- [x] `fit-batch`: batched MAP, progressive chunk outputs, optional SED sample plots.
- [x] `fit-population`: batched population summaries first; true hierarchical fit later.
- MCMC: deferred until forward model and optimizer are fast enough.

Tasks:

- Align CLI names/help text with these workflows.
- Ensure each workflow can optionally generate SED diagnostics.
- Remove duplicated or confusing reporting paths.

Done when:

- New user can run one forward command and one fit command without reading code.

## Phase 3 - Minimal Production Model

Status: active cleanup.

Commit target: `Reduce fitted parameter space`

Goal: fit few realistic parameters first.

Production defaults:

- Free: `z_obs`, `log10_formed_mass_msun`, simple SFH parameters if needed.
- Derived: SFR from fitted mass-normalized SFH.
- Fixed or weak-prior initially: metallicity, dust, depending on diagnostics.
- Removed from active model: quench terms, burst terms, non-parametric SFH bins.

Tasks:

- [x] Remove complex SFH/quench/burst from code and default configs.
- [x] Keep simple lognormal SFH table as the only active SFH path.
- Add flat-prior config for comparison.

Done when:

- Fast mode still fits redshift, mass, SFR-derived quantity, and minimal SFH.
- No default production run uses quench/burst accidentally.

## Phase 4 - Redshift Treatment

Status: baseline production config changed.

Commit target: `Remove naive PHZ interval prior`

Goal: fit redshift without circular hard photo-z truth.

Tasks:

- [x] Stop using `phz_min_70` / `phz_max_70` as hard truth-like bounds in current production configs.
- [x] Remove PHZ interval priors from MAP, fast-grid, population, sampling, and config validation.
- [x] Keep PHZ interval columns only as optional diagnostics.
- Test:
  - fixed catalog redshift
  - free redshift with flat broad prior
  - calibrated non-circular redshift prior, only if justified later
- Report redshift bias and mass coupling.

Done when:

- Configs make clear whether photo-z is ignored, soft prior, or fixed input.

## Phase 5 - Mass Offset Investigation

Commit target: `Add mass offset diagnostics`

Goal: identify source of fitted mass offset.

Checks:

- formed mass vs surviving stellar mass
- IMF/SSP mismatch
- luminosity distance/redshift normalization
- flux units and filter integration
- dust/mass degeneracy
- per-band residual pattern

Outputs:

- `fit_mass - catalog_mass` vs redshift, color, SNR, dust, true/catalog mass.
- Flux ratio by band.
- Summary table for worst outliers.

Done when:

- Mass offset has one or more falsifiable causes, not guesses.

## Phase 6 - SSP And Pop-COSMOS Research

Commit target: `Document SSP and Pop-COSMOS choices`

Tasks:

- Compare local SSP assets available to DSPS options.
- Document which SSP table is used and why.
- Verify what Pop-COSMOS does for SFH, dust, metallicity, priors, and photo-z.
- Decide whether current SSP is acceptable before changing it.

Done when:

- SSP choice is explicit and tied to local files.
- Pop-COSMOS comparison informs priors without copying unavailable data assumptions.

## Phase 7 - Performance And Benchmarks

Status: existing benchmark hooks kept; SED sample writes are explicit opt-in.

Commit target: `Benchmark forward and optimizer`

Goal: prepare for millions of galaxies.

Benchmarks:

- context load
- batched forward SED
- filter projection
- loss evaluation
- optimizer step
- full grid/MAP fit
- RAM, VRAM, CPU timing

Tasks:

- Keep arrays on GPU.
- Remove NumPy/pandas from hot path where possible.
- Use progressive parquet chunk writes.
- Generate plots after compute.
- Add `uv run` and `conda shine` benchmark commands.

Done when:

- Benchmark output explains runtime per galaxy and per major stage.

## Phase 8 - Priors

Commit target: `Add prior comparison configs`

Configs:

- `prior_flat.yaml`
- `prior_weak_informed.yaml`
- `prior_catalog_informed.yaml`

Goal: find informed but non-circular priors.

Rules:

- Do not use target truth directly as prior.
- Do not treat photo-z intervals as hard truth.
- Learn population priors only after forward/SED/mass diagnostics are stable.

Done when:

- Prior comparison report shows bias, scatter, and failure cases.

## Commit Cadence

- One commit per phase or coherent sub-phase.
- Commit message imperative and short.
- Each commit should mention commands run and output paths inspected.
- Avoid mixing science behavior change with unrelated formatting.

## Immediate Next Step

Start Phase 1: add SED diagnostics and ground-truth/filter overlays. This is blocker for understanding SED divergence and mass offset.
