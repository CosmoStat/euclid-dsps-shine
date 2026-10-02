# DSPS sampling performance diagnosis

## Local acceleration continuation, 2026-10-02

The merged integrator now supports an explicit performance-only option:

```yaml
model:
  photometry_integrator: merged_gauss4_v1
  photometry_autodiff: scalar_redshift_jvp_v1
```

The default remains `reverse_v1`. The option preserves the quadrature primal,
stored assets and arithmetic precision. It uses linearity in the spectrum and
scalar forward-mode differentiation in redshift to reduce reverse-mode memory;
active wavelength/filter derivatives fall back to the reference JVP. Numerical
checkpoint receipts remain unchanged because the primal integrator is identical.
Record the runtime config and source hashes separately. This is first-derivative
validation, not certification of Hessians or arbitrary higher-order AD.

Retained GPU experiments from 2026-10-01, under
`outputs/analysis/sampling_speed_20261001/`:

| Full target gradient | Median batch ms | States/s | Compiler temporaries |
| --- | ---: | ---: | ---: |
| Standard, 8 states | 192.6 | 41.5 | 407 MB |
| Standard, 16 states | 230.4 | 69.5 | 820 MB |
| Standard, 32 states | 364.6 | 87.8 | 1759 MB |
| Specialized JVP, 64 states | 652.7 | 98.1 | 2755 MB |

Standard 64 states failed with RESOURCE_EXHAUSTED, also in a fresh process.
The specialized 64-state run matches standard eight-state reference blocks:
maximum target difference 1.59e-12, gradient difference 1.55e-9. States comprise
eight actual archived observations and up to eight independent starts per object,
not 64 distinct galaxies. Larger batching improves kernel throughput, not proof
of faster convergence for an individual posterior. The specialized eight-state
paired test had no latency gain (223.1 versus 226.6 ms), despite lower memory
(407 versus 326 MB). Ordered merging and zero-tail cropping were slower;
grouped dust's tentative 5% gain was not promoted.

`compare_feniks_affine_nuts.py` supports isolated `--arms raw_dense`,
`--arms affine_dense`, or `--arms flow_dense`. The latter uses the frozen
conditional coupling flow in float64 as an invertible coordinate map, with its
Jacobian added to the unchanged physical target. Checks cover inverse recovery,
Jacobian determinant and target gradient chain rule before MCMC. All arms use
the same physical starts, dense mass adaptation and seeds; independent-process
timings are not hardware-paired comparisons. Short probes require a longer
convergence gate before changing scientific defaults.

The previous raw arm completed (48 warmup, 64 draws, four chains, depth 4):
314.4 s execution, minimum bulk ESS 14.0, maximum R-hat 3.50, zero sampling
divergences, 100% depth saturation. The interruption left no saved affine draws;
that arm is rerun in a separate fresh output directory, preserving the raw files.

Completed affine arm:
`outputs/analysis/sampling_speed_20261002/affine_nuts_gpu_w48_s64/summary.json`.
Execution 365.5 s (164.8 s warmup, 200.7 s sampling), min bulk ESS 17.0,
max R-hat 2.67, zero sampling divergences, 100% depth saturation, exploratory
minimum ESS/execution second 0.0465 versus raw 0.0444. This small difference
does not establish an efficiency win: both probes are severely unconverged,
48-step adaptation is short, and hardware timing was not paired. Dense mass
adaptation was already present in the raw arm; linear whitening is not a cure.

The first flow-coordinate four-chain probe passed exact target/gradient and
Jacobian checks but terminated with signal 15 during warmup compilation.
No flow draws or convergence results were saved; no OOM evidence was found in
kernel logs. Do not treat the partial output directory as completed sampling.

A smaller flow feasibility run completed:
`outputs/analysis/sampling_speed_20261002/flow_nuts_gpu_small/summary.json`.
Two chains, 20 warmup, 16 draws/chain, depth 3: warmup 30.8 s, sampling 25.0 s;
compilation separately 97.5 s and 54.0 s. Zero sampling divergences, one warmup
divergence, max R-hat 2.69 and 100% depth saturation. The 55.8 s execution is
not comparable to the four-chain raw/affine probes with more draws and depth 4.
The tiny-run ESS/s is not evidence of acceleration or improved mixing. The
implementation is feasible and target-preserving, but remains diagnostic-only.

### Configured production-path validation

`outputs/analysis/sampling_speed_20261002/configured_jvp_gpu64_retry.json`
uses `--implementation configured`, not a monkeypatch. All 64 production
posterior values/gradients match standard eight-state microbatches:
max target difference 3.29e-11, max gradient difference 2.05e-9.
Fifteen synchronized repeats give median 603.2 ms, p95 656.1 ms,
106.1 states/s; compiler temporaries 2,755,118,640 bytes, compilation 54.6 s.
Laptop clocks are not locked: do not attribute the difference from yesterday's
98.1 states/s to a new arithmetic speedup. This run validates the production
configuration route and confirms its usable 64-state memory footprint.
An earlier attempt passed compilation/execution but failed the benchmark's
reference import after moving the helper; the import was restored and regression
tested. Its failure JSON is preserved and is superseded by the complete retry.

Reproduce from the repository root:

```bash
env EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 EUCLID_DSPS_REQUIRE_GPU=1 \
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/maxime/miniforge3/envs/shine/bin/python \
  -m scripts.benchmark_feniks_production_local \
  --out outputs/analysis/configured_jvp_fresh.json --implementation configured \
  --batches 64 --components target_gradient --repeats 15 --nuts-steps 0 \
  --verify-reference-chunk-size 8
```

Next sampling gate: matched-budget raw/flow probes with sufficient adaptation,
then longer dense draws assessed by bulk/tail ESS, R-hat, divergences and ESS/s.
No production sampler default, prior or scientific precision was changed.
Catalogue-based one-row/batch fit validation remains blocked: the catalogue is
absent, as are the historical smoke YAML files named in AGENTS. Actual archived
observations were used instead for forward and NUTS validation.

Final local checks: 40 focused tests passed (quadrature, production routing,
benchmark helpers, affine transforms, canonical target and exact posterior),
compileall, CLI help, focused Ruff and `git diff --check` passed. No remote job
was launched. All benchmark/sampler processes from this phase have exited.
Before publication, the same 40 tests, compileall, CLI help and focused Ruff
also passed in an isolated checkout of the staged index, excluding unrelated
dirty working-tree changes. Benchmark inputs/outputs remain local artifacts,
not source files included in the commit.

## Fresh production-asset benchmark, 2026-10-01

Entrypoint: `scripts/benchmark_feniks_production_local.py`. Actual learned
checkpoint SHA256 `bc14123c46c06e5898cd4ae4ef3733163077164ad9775a9f4ddcddcf4bf6a1b5`
matches the frozen reference. Reference config, checkpoint sidecar, manifest and
observed-row hashes also match the archived NUTS input provenance. The local
amortized catalogue directory is empty: this benchmark reads archived **real**
observations and latent starts, not a regenerated or synthetic catalogue.
Rows: 66, 68, 445, 945, 1437, 11617, 13012, 14615; one archived start per object.

Actual setup: 18 bands, 80 SFH bins, 11,149 SSP wavelengths, 64 compressed basis
components, 12 metallicities, 107 ages. Coefficients are float16 on disk and
float32 at runtime; basis/SSP arrays are float32. Loaded prior floating leaves
are float64. Spline, MDF and Gaussian likelihood use the configured float64
arithmetic. No precision setting or scientific sampler setting was changed.

Both devices use Python 3.12, JAX 0.10.1, BlackJAX 1.5, conda `shine`.
CPU is i5-13500H under WSL; GPU is RTX 4060 Laptop, 8 GiB. Device arrays remain
resident, every call is synchronized, compilation and three priming calls are
excluded. Thirty repeats per component; medians below, raw samples/p95 and
compilation seconds are in the JSON files. No CPU tests ran during the retained
CPU sweep or GPU sweep. This is a laptop measurement without locked clocks or
power, not a controlled H100 comparison.

| Component | Galaxies per batch | CPU batch ms | RTX batch ms |
| --- | ---: | ---: | ---: |
| Forward, 18 fluxes | 1 | 34.7 | 68.1 |
| Forward, 18 fluxes | 4 | 119.5 | 87.3 |
| Forward, 18 fluxes | 8 | 196.1 | 113.2 |
| Full posterior value + gradient | 1 | 69.0 | 154.5 |
| Full posterior value + gradient | 4 | 229.6 | 174.8 |
| Full posterior value + gradient | 8 | 487.3 | 258.8 |
| Likelihood + decoder value/gradient | 1 | 62.8 | 146.5 |
| Prior network alone value/gradient | 1 | 3.8 | 7.5 |
| Prior network alone value/gradient | 8 | 6.2 | 17.4 |

GPU full-gradient throughput: 6.47 / 22.88 / 30.92 evaluations per second for
1 / 4 / 8 galaxies. CPU: 14.49 / 17.42 / 16.42. The tested optimum is eight on
GPU and four on CPU; this is not a capacity ceiling. CPU singleton repetition
gave 34.2 ms forward / 62.9 ms target gradient, confirming the order of magnitude.
An earlier exploratory CPU sweep was substantially slower (113/170 ms singleton)
and overlapped tests during some component measurements. Retain it as provenance,
not as the comparison table or evidence of an optimization gain.

The canonical guarded prior component **includes forward evaluation**, because
it checks predicted-flux finiteness. Network-only prior is separately measured;
otherwise a decoder cost would incorrectly be attributed to the RealNVP prior.
Independent component graphs are not additive. These measurements locate the
dominant gradient cost in decoder/likelihood work, not in the prior network;
they do not yet isolate reconstruction, dust, photometry or cosmology kernels.
Source-level candidates include repeated merged-grid sorting/interpolation in
18 bands, dense reconstructed age/wavelength intermediates and FP64 operations.
Changing filter resolution or precision would require accuracy validation.

Shared dynamic SSP/filter arguments occupy about 3.8 MB for all tested batch
sizes. Compiler temporary allocation for full gradient, batches 1/4/8:
CPU 121/463/922 MB; GPU 61/211/407 MB. This excludes constants, driver/context
memory and allocator reservation, and is not measured peak VRAM. The cube is
not replicated in the function arguments; reconstructed spectra and gradients
grow per state. No separate-process or multi-GPU sharing claim follows.

Numerical checks pass across CPU/GPU and batch sizes: forward relative tolerance
1e-9 (absolute 1e-40), gradient relative 1e-8 (absolute 1e-7). Maximum posterior
gradient discrepancy is 7.15e-9 absolute. Likelihood plus raw prior matches the
full target value and gradient at every tested valid point.

Retained files under `outputs/analysis/sampling_speed_20261001/`:
`production_cpu_clean_sweep.json`, `production_cpu_isolated_one.json`,
`production_rtx4060.json`, `production_cpu_clean_nuts.json`,
`production_numerical_checks.json`, `production_source_provenance.json`.
The latter records the dirty local checkout commit and Python source hashes;
production numerical code stayed unchanged during the retained sweeps. These
are current local-code timings, not a replay of the historical remote commit.
Validation: 11 focused benchmark/quadrature tests, compileall, Ruff, CLI help
and diff checks pass. Full catalogue fitting is not exercised.

Reproduction, sequentially, with fresh output paths:

```bash
PY=/home/maxime/miniforge3/envs/shine/bin/python
env JAX_PLATFORMS=cpu EUCLID_DSPS_REQUIRE_GPU=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" \
  -m scripts.benchmark_feniks_production_local \
  --out outputs/analysis/production_cpu_repeat.json --nuts-steps 0 \
  --components forward target_gradient likelihood_gradient prior_network_gradient \
  --repeats 30
env EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 EUCLID_DSPS_REQUIRE_GPU=1 \
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" \
  -m scripts.benchmark_feniks_production_local \
  --out outputs/analysis/production_gpu_repeat.json --nuts-batches 1 \
  --components forward target_gradient likelihood_gradient prior_network_gradient \
  --repeats 30
```

Short NUTS probe: three transitions, fixed step .001, identity diagonal mass,
depth4, one chain per object, no adaptation. CPU singleton takes 3.10 s median
for 45 integration steps (compilation 51.7 s); RTX takes 7.18 s (50.2 s compile).
Both have zero divergences and acceptance approximately .9998. Identical seeds,
starts, step count and observations; CPU/GPU acceptance rates agree at numerical
roundoff. The intentionally small step and high acceptance do not
establish good tuning, mixing or convergence. This is not an ESS benchmark.
The historical dense-mass depth6 run still has a separate algorithmic problem:
nearly all trajectories saturate their depth while delivering little ESS.
Twenty-minute converged sampling is not established by this local cost probe.

## Evidence inspected locally

Retrieved artifacts under `outputs/feniks_nuts_20260911/` are historical
measurements, not a fresh GPU benchmark or a live Slurm accounting check.
The intended current run and scope of the tutor's 20-minute estimate are pending.

For `observed_000`, learned prior, encoder initialization, eight vectorized
chains, dense mass matrix, float64 geometry target:

| Run | Warmup | Stored draws across chains | Warmup minutes | Sampling minutes | Total minutes | Depth limit fraction | Minimum bulk ESS | Maximum Rhat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth5 probe | 500 | 4096 | 19.56 | 16.61 | 36.18 | 96.80% | 76.53 | 1.640 |
| depth6 probe | 500 | 4096 | 42.35 | 36.03 | 78.38 | 98.46% | 65.62 | 1.239 |
| depth6 long | 1500 | 32768 | 110.24 | 260.11 | 370.35 | 98.39% | 126.32 | 1.095 |

Sources: each run's `nuts_followup_summary.csv`, `MANIFEST.json`, and
`nuts/observed_000/B_dense_depth6/chain_0/chain_manifest.json` (depth5 uses
`B_dense_depth5`). Batch wall time is shared across chains, not summed eight times.
Reported times include compilation inside warmup/sampling; loading and target
validation occur outside those timers. All three fail their diagnostic gate.

Reading all long-run info Parquets gives a mean of 62.453 integration steps
per transition, with median 63, and divergence fraction 0.000824. Thus the long
sampling alone executes about 2.05 million integrator steps across eight chains.
Its minimum bulk ESS is 0.386% of stored draws, about 0.341 effective draws per
minute including warmup. This is an efficiency diagnostic, not a converged ESS
budget prediction. Weakest bulk ESS: `sfh_dlog_sfr_10`, `sfh_dlog_sfr_02`, `dust_av`.

## Interpretation

The measured algorithmic bottleneck is saturated trajectories with poor mixing.
Fifteen coordinates do not imply a cheap posterior: each gradient traverses the
canonical DSPS photometric decoder and learned prior. Eight chains are already
vectorized. The component cost split and accelerator utilization are unmeasured.
Saturation suggests trajectories are truncated before a natural U-turn, but
does not alone identify whether the difficulty comes from likelihood, prior,
coordinate scaling, nonlinear degeneracy or numerical behavior.

Simply shortening trajectories or discarding warmup is not an efficiency gain
unless the same posterior accuracy is retained. Increasing trajectory depth
already increases cost substantially in these probes without resolving mixing.

## Component benchmark

On the GPU node, from this checkout and its configured interpreter:

```bash
python -m scripts.profile_feniks_sampling_cost \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --row 0 --positions 8 --repeats 20 \
  --out "$SCRATCH/dsps_sampling_profile_8.json"
```

Set those four paths to the actual run's assets first; use fresh output paths.
Repeat with 1 and 32 positions to assess batching at the same object/seed.
The profiler measures compilation separately and synchronizes every execution.
It checks finiteness and measures target value, full value/gradient, likelihood
value/gradient and prior value/gradient at encoder-generated positions.
Component graphs compile independently and timings are not additive. Input
float64 does not promote loaded checkpoint or decoder arrays: use the declared
production config; this does not automatically reproduce the fully promoted
historical geometry target. No MCMC or convergence claim follows from this test.

Then measure NUTS warmup and repeated equal-size sampling chunks with integration
counts, GPU utilization and ESS per second on a fixed target. The existing
`scripts/benchmark_feniks_nuts_multigalaxy.py` covers distinct-object batching;
its 10-warmup/10-draw capacity defaults do not measure statistical efficiency.

Initial validation used `.venv` (JAX 0.10.0), which exposes CPU only. Subsequent
checks found a working CUDA backend in conda `shine` (JAX 0.10.1), and completed
local GPU performance probes as described below. No production scientific GPU
sampling or live Jean-Zay accounting was performed.

## Local optimization and measured results

The merged integrator now interpolates the SED and transmission at merged knots
once, then blends their endpoint values at each Gauss node. Each interpolant is
affine between consecutive merged knots, so this preserves the quadrature rule
and original grids. It reduces search work by roughly the quadrature order.
The legacy integrator is not affected.

Completed benchmarks use the real local SSP asset
`/home/maxime/src/test_codex/euclid_dsps/Data/ssp_data_fsps_v3.2_lgmet_age.h5`:
12 metallicities, 107 ages, 5,994 wavelength samples. This is an alternate dense
SSP asset, not the absent FENIKS production compressed basis. Full forward tests
use four actual Euclid filter curves, the 15D spline implementation, MDF/dust/IGM,
and float64 numerical settings. NUTS probes use the canonical target with a
standard Gaussian latent prior and synthetic Gaussian-noise observations.

| Measurement | Before | After | Interpretation |
| --- | --- | --- | --- |
| CPU isolated single-band projection gradient | 9.60 ms | 3.13 ms | 3.06x faster |
| CPU four-band full forward | 19.70 ms | 13.80 ms | 1.43x faster |
| CPU full gradient of summed magnitudes | 58.66 ms | 48.72 ms | 1.20x faster |
| CPU canonical target value/gradient | 58.41 ms | 65.78 ms | slower in this measurement |
| CPU 10 NUTS transitions | 9.075 s | 6.621 s | 1.37x faster |
| RTX 4060 canonical target value/gradient | 176.50 ms | 193.29 ms | slower in this measurement |
| RTX 4060 10 NUTS transitions | 25.110 s | 25.458 s | no demonstrated gain |

CPU full timings: `outputs/analysis/sampling_speed_20261001/spline_nuts_gaussian64.json`.
GPU timings: `outputs/analysis/sampling_speed_20261001/spline_nuts_rtx4060.json`.
Isolated timing: `outputs/analysis/sampling_speed_20261001/photometry_endpoint.json`.
Compilation and priming are excluded from execution timings. The microbenchmarks
use 30/50 repetitions; NUTS uses three repeats of the same keyed trajectory.
CPU/GPU runs use different JAX versions and direct/vmapped-one-chain wrappers;
do not interpret cross-backend ratios as an isolated hardware experiment.
Measurements are sequential and component p95 values show variability; they are
not a randomized interleaved benchmark or a production-wide performance claim.

Each probe performs 10 transitions, all with 15 integration steps, fixed step
size 0.01, diagonal identity mass matrix, no warmup and zero divergences. This
tests the cost of equivalent kernel work, not convergence, ESS or scientific
efficiency. Multiple-chain support in the script is for performance only and
starts chains at the same center with different RNG keys.

Center parity: full magnitudes match; summed-magnitude gradient maximum absolute
difference is 1.82e-13 on CPU and 2.44e-13 on GPU. Canonical target gradients differ
by at most 3.36e-12 / 6.98e-12. Regression tests cover randomized spectra, redshift
gradients, support boundaries, empty overlap, narrow lines and orders 4/8.

An age-group attenuation experiment was also tested and removed: despite correct
values/gradients, it did not improve full-gradient performance. Its intermediate
JSON is an experiment receipt, not the retained implementation's benchmark.
The preliminary `spline_nuts.json` uses default Student-t2/float32 likelihood and
is superseded by the explicit Gaussian64 probe.

Reproduce on CPU:

```bash
.venv/bin/python -m scripts.benchmark_spline_forward \
  --ssp /home/maxime/src/test_codex/euclid_dsps/Data/ssp_data_fsps_v3.2_lgmet_age.h5 \
  --repeats 50 --nuts-steps 10 --out outputs/analysis/sampling_speed_repeat_cpu.json
```

Reproduce on the local GPU from conda `shine`:

```bash
EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 EUCLID_DSPS_REQUIRE_GPU=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/maxime/miniforge3/envs/shine/bin/python -m scripts.benchmark_spline_forward \
  --ssp /home/maxime/src/test_codex/euclid_dsps/Data/ssp_data_fsps_v3.2_lgmet_age.h5 \
  --repeats 50 --nuts-steps 10 --out outputs/analysis/sampling_speed_repeat_gpu.json
```

The missing pure-Python dependency was added to `shine` with
`uv pip install --python /home/maxime/miniforge3/envs/shine/bin/python --no-deps jax-cosmo==0.1.0`.
No JAX packages, dependency manifest or lockfile changed. The existing `uv`
environment remains the CPU validation path. The production component profiler
also needs explicit GPU plugin discovery when launched outside the CLI wrappers.

Validation: 96 targeted tests pass, 3 skipped, compileall/Ruff/diff checks pass.
The legacy one-row/small-batch fit configurations named in AGENTS are absent;
the one-row attempt stops at config loading. No production posterior speedup or
20-minute convergence guarantee follows from these alternate-asset probes.
