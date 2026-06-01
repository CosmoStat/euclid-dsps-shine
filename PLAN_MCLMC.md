# MCLMC Posterior Implementation Plan

## Goal

Add an experimental BlackJAX MCLMC posterior backend for the compressed
PopCosmos DSPS model, then benchmark it against the current NumPyro HMC/NUTS
posterior path.

This is a posterior-sampling project, not a MAP replacement. The immediate
production path remains:

1. run compressed MAP batches;
2. generate MAP quality-control reports;
3. improve compressed SSP quality if needed;
4. use MCLMC on selected rows to characterize posterior geometry and train or
   validate learned-prior/posterior machinery.

## Current State

- The runtime model path is now fast enough for large MAP batches through
  `configs/popcosmos_binned_compressed.yaml`.
- `euclid_dsps.mcmc` now keeps NumPyro `nuts` and fixed-step `hmc`, and adds an
  experimental BlackJAX `mclmc` path for one-galaxy posterior diagnostics.
- Local `shine` has BlackJAX 1.5, with `blackjax.mclmc`,
  `blackjax.adjusted_mclmc`, and `blackjax.adjusted_mclmc_dynamic` available.
- `euclid_dsps.posterior_target` provides the pure-JAX bounded log-density used
  by MCLMC.
- Current limitation: the first implemented backend is unadjusted MCLMC with
  static configured/automatic `L` and `step_size`. It is not yet adapted or
  Metropolis-corrected, so it must be benchmarked against HMC/NUTS before
  science use.

## Naming

The algorithm is usually called `MCLMC`: Microcanonical Langevin Monte Carlo.
The user may write `MCMLC`; in code, config, docs, and reports use `mclmc`.

## External API To Verify

BlackJAX exposes MCLMC as a JAX-native sampler family. Current documentation
shows:

- `blackjax.mclmc` / `blackjax.mcmc.mclmc` for unadjusted MCLMC;
- `blackjax.adjusted_mclmc` and `blackjax.adjusted_mclmc_dynamic` for
  Metropolis-adjusted variants;
- MCLMC kernels are not JIT-compiled by default and should be wrapped with
  `jax.jit` by the caller;
- adaptation helpers exist under `blackjax.adaptation.mclmc_adaptation`,
  including `mclmc_find_L_and_step_size`.

Implementation must probe the installed BlackJAX version because the public
docs may track `main`, not necessarily the release installed by pip/conda.

## Scientific/Statistical Choice

There are two possible MCLMC tracks:

1. `mclmc` unadjusted:
   - likely fastest;
   - useful for engineering and posterior-shape diagnostics;
   - can have discretization bias, so do not use it as a final science sampler
     without a convergence/bias comparison.
2. `adjusted_mclmc_dynamic` or `adjusted_mclmc`:
   - slower;
   - Metropolis-corrected;
   - preferred target for science posterior comparisons if available and stable
     in the installed BlackJAX version.

Default plan: implement `mclmc` first as an experimental backend, then add the
adjusted variant as the science candidate once the log-density and transform
plumbing are validated.

## Target Log-Density Design

BlackJAX wants a pure JAX log-density function. The current NumPyro model hides
part of this in `numpyro.sample`, so implement a separate backend:

```text
unconstrained vector y[D]
  -> bounded physical parameters theta[name]
  -> log_prior(theta)
  -> log_likelihood(theta)
  -> log_abs_det_jacobian(y -> theta)
  -> log_posterior_y
```

Use the same bounded transform conventions as the MAP optimizer where possible:

```text
theta = low + (high - low) * sigmoid(y)
log|dtheta/dy| = log(high-low) + log(sigmoid(y)) + log1p(-sigmoid(y))
```

Why this matters:

- MCLMC operates most naturally on unconstrained real variables.
- Direct hard bounds in physical space produce `-inf` plateaus and bad
  gradients.
- The Jacobian is required so the posterior over physical parameters is the
  same posterior the NumPyro model samples.

## Likelihood And Priors

The log-density must reuse existing semantics:

- flux-space Student-t likelihood stays unchanged;
- Gaussian likelihood remains supported;
- `flux_error_floor_frac`, `flux_error_jitter`, band offsets, and finite masks
  match `fit.py`/`mcmc.py`;
- priors use `sample.priors` merged with fit bounds exactly as the current
  NumPyro path does;
- the PopCosmos hard gas constraint
  `log10_gas_metallicity >= log10_stellar_metallicity` returns `-inf` in the
  target log-density.

Longer-term: hard gas inequality may hurt gradient samplers near the boundary.
If it becomes a real MCLMC blocker, add a separately validated constrained
reparameterization rather than clipping.

## Implementation Phases

### Phase 0 - Dependency And API Probe

Add optional dependency support without forcing BlackJAX into the base install:

```toml
[project.optional-dependencies]
samplers = [
    "blackjax",
]
```

Add a small API probe script:

```text
scripts/inspect_blackjax_mclmc_api.py
```

It should print:

- installed BlackJAX version;
- whether `blackjax.mclmc` exists;
- whether `blackjax.adjusted_mclmc_dynamic` exists;
- whether `blackjax.adaptation.mclmc_adaptation.mclmc_find_L_and_step_size`
  exists;
- the callable signatures needed by the implementation.

Go criterion: the script runs cleanly in `shine` after installing the optional
sampler dependency.

### Phase 1 - Pure JAX Posterior Target

Create a small internal module, probably:

```text
euclid_dsps/posterior_target.py
```

Responsibilities:

- build ordered free-parameter metadata from `fit.free_parameters`;
- transform `y <-> theta`;
- compute log-Jacobian;
- compute log-prior in JAX;
- compute log-likelihood in JAX using `model_mags_jax_dynamic`;
- return `logposterior_y(y, model_args, observed, sigma, mask, ...)`;
- expose helper functions for posterior result conversion.

Tests:

- transform round-trip is finite and monotonic;
- log-Jacobian finite for interior values;
- log-density matches NumPyro target on a synthetic one-band model where both
  can be evaluated;
- gas metallicity violation gives `-inf`;
- gradients are finite at valid interior points.

### Phase 2 - BlackJAX MCLMC Backend

Add a separate path in `euclid_dsps.mcmc`:

```text
sample.sampler: nuts | hmc | mclmc | adjusted_mclmc
```

Keep NumPyro `nuts/hmc` untouched. For `mclmc`:

- lazily import BlackJAX and raise a clear error if missing;
- initialize from MAP when `sample.init_from_map: true`;
- convert MAP physical parameters to unconstrained `y0`;
- build a JIT-compiled MCLMC step function;
- run warmup/adaptation if configured;
- run sampling with `jax.lax.scan`;
- convert `y_samples` back to physical parameter samples;
- reuse `MCMCResult` output shape and reporting.

Suggested config keys:

```yaml
sample:
  sampler: mclmc
  num_warmup: 256
  num_samples: 512
  mclmc_l: auto
  mclmc_step_size: auto
  mclmc_adjusted: false
  mclmc_diagonal_preconditioning: true
  mclmc_desired_energy_var: 0.0005
  mclmc_num_effective_samples: 150
```

Use lowercase config names in code:

```text
sample.mclmc_l
sample.mclmc_step_size
sample.mclmc_adjusted
sample.mclmc_diagonal_preconditioning
sample.mclmc_desired_energy_var
sample.mclmc_num_effective_samples
```

Diagnostics to save:

- backend: `blackjax_mclmc` or `blackjax_adjusted_mclmc`;
- compile time;
- warmup time;
- sampling time;
- samples/sec;
- `L`;
- `step_size`;
- mean and p95 `energy_change`;
- mean and p95 `kinetic_change`;
- fraction of non-finite proposals/states if available;
- ESS/sec if implemented.

### Phase 3 - CLI, Config, And Docs

Update:

- `SUPPORTED_SAMPLERS`;
- CLI `--sampler` choices;
- sample config validation;
- `docs/source/run_setup.rst`;
- `docs/source/api.rst` if a new module is added;
- README posterior section.

Keep command examples explicit that this is experimental until benchmarked:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  posterior --index 0 \
  --sampler mclmc \
  --num-warmup 256 \
  --num-samples 512 \
  --out outputs/runs/dev_popcosmos_compressed_mclmc_one
```

### Phase 4 - Benchmark Harness

Add:

```text
scripts/benchmark_posterior_samplers.py
```

It should compare:

- NumPyro `hmc`;
- NumPyro `nuts`;
- BlackJAX `mclmc`;
- BlackJAX adjusted MCLMC if available.

Rows to test:

- one easy galaxy from MAP QC;
- one dusty/faint galaxy;
- one high-redshift galaxy;
- one AGN-dominated or high-`ln_fagn` galaxy;
- one poor-MAP/constrained-boundary case.

Metrics:

- compile time;
- warmup time;
- sampling time;
- samples/sec;
- ESS/sec per parameter, if feasible;
- posterior median and 16/84 intervals;
- posterior predictive residuals;
- fraction non-finite;
- agreement with NUTS on small cases where NUTS can run.

Output:

```text
outputs/benchmarks/posterior_samplers_YYYYMMDD/
  benchmark_summary.json
  posterior_sampler_comparison.csv
  per_parameter_ess.csv
  posterior_interval_comparison.csv
  plots/*.png
```

### Phase 5 - MAP Sprint Integration

Do not block the MAP sprint on MCLMC. The recommended order remains:

1. Run MAP `n=10k` compressed.
2. Generate MAP QC.
3. Build SSP `k128`.
4. Re-run dense-vs-compressed `n=500`.
5. If OK, run MAP `n=100k+`.
6. Use MCLMC on a stratified subset of MAP results to validate posterior shape
   and train/check learned posterior machinery.

MCLMC row selection should use the MAP QC output:

- high and low redshift;
- high dust;
- high gas ionization;
- high AGN fraction;
- near-boundary solutions;
- excellent and poor MAP fit-quality examples.

## Risks

- BlackJAX API instability: guard with explicit API probe and tests.
- Unadjusted MCLMC bias: benchmark against NUTS/HMC on small cases.
- Hard gas metallicity constraint: may cause sampler pathologies near the
  boundary.
- Full 16D posterior geometry: AGN/dust/gas degeneracies can require
  preconditioning and careful initialization.
- Compile time: use one-galaxy smoke tests before batch posterior runs.

## Initial Smoke Commands

After dependency install:

```bash
python scripts/inspect_blackjax_mclmc_api.py

python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  posterior --index 0 \
  --sampler mclmc \
  --num-warmup 32 \
  --num-samples 32 \
  --out outputs/runs/dev_popcosmos_compressed_mclmc_one_smoke
```

Then benchmark:

```bash
python scripts/benchmark_posterior_samplers.py \
  --config configs/popcosmos_binned_compressed.yaml \
  --indices outputs/runs/popcosmos_binned_compressed_map_n10000_bs128/qc_selected_rows.txt \
  --samplers hmc nuts mclmc adjusted_mclmc \
  --num-warmup 256 \
  --num-samples 512 \
  --out outputs/benchmarks/posterior_samplers_popcosmos_compressed
```

The benchmark command is intentionally future-facing: the QC-selected row file
should be produced by the MAP QC sprint.
