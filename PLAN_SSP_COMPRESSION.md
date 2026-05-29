# SSP And Grid VRAM Compression Plan

## Goal

Reduce the resident GPU memory used by the PopCosmos-like SSP, gas, and AGN
spectral assets so much larger galaxy batches can fit on GPU.

This is not a disk-compression project. HDF5 gzip/LZF/shuffle can help storage
and transfer, but it does not solve the problem if `load_context` decodes the
asset into a dense `float32` JAX tensor. The compressed representation must be
the representation consumed by JAX.

Scientific compression is acceptable if it is explicit, benchmarked, and keeps
the final dense-vs-compressed and FSPS/Prospector residuals inside the agreed
budget.

## Current Resident Assets

The current full PopCosmos-like path uses:

```text
Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5
Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5
Data/popcosmos_chabrier_gas_ssp_grid.h5
Data/popcosmos_chabrier_agn_component_ssp_grid.h5
```

Main dense tensors:

```text
base SSP:
  ssp_flux[stellar_lgmet, age, wave]
          [12,            107, 11149]                 ~55 MiB

pure stellar SSP:
  ssp_flux[stellar_lgmet, age, wave]
          [12,            107, 11149]                 ~55 MiB

gas SSP grid:
  ssp_flux[gas_lgmet, gas_lgu, stellar_lgmet, age, wave]
          [7,         7,       12,            107, 11149]  ~2.6 GiB

AGN component grid:
  agn_lnu_per_mformed[fagn, agn_tau, stellar_lgmet, age, wave]
                       [8,    9,       12,            107, 11149]  ~3.8 GiB
```

The gas and AGN component grids dominate resident memory. The base SSPs are
small by comparison, but they should still be included in the long-term
compressed asset family so the final model is coherent.

## Current Compressed Runtime Entry Point

`configs/popcosmos_binned_compressed.yaml` is the CLI entry point for the full
16-parameter PopCosmos-like binned model with compressed resident assets. It
inherits `configs/popcosmos_binned.yaml` and switches:

```yaml
model:
  ssp_model: compressed_basis
  compressed_ssp_path: Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5
  nebular_model: compressed_gas_grid
  compressed_gas_grid_path: Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5
  agn_model: compressed_fsps_component_grid
  compressed_agn_component_grid_path: Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5
```

The dense config remains the reference. The compressed config is the operational
high-throughput path for MAP batches after a dense-vs-compressed residual check.

## Non-Goals

- Do not make lossless HDF5 repacking the main deliverable. It is optional IO
  cleanup only.
- Do not decode a compressed asset into the full dense gas or AGN tensor before
  entering JAX.
- Do not replace the dense production path until dense-vs-compressed
  photometry, gradients, and GPU batch-size benchmarks pass.
- Do not silently fall back to dense assets when a compressed config asks for a
  compressed mode.
- Do not judge scientific compression only from SED plots. The gate is
  broad-band photometry, gradients, and final FSPS/Prospector closure.

## Phase 0 - Clean Branch And Data Safety

Start implementation on a clean branch from `dev`. Do not create another
worktree. The existing local `Data/` directory is the source of dense reference
assets, and compression builders must treat it as read-only input unless the
output path is a new file.

### Work

1. Fetch `dev` and create a dedicated implementation branch:

   ```bash
   git fetch origin dev
   git switch -c feature/ssp-vram-compression origin/dev
   ```

2. Confirm the branch starts clean:

   ```bash
   git status --short --branch
   ```

3. Validate dense reference assets before changing model code:

   ```bash
   python scripts/generate_fsps_ssp_grid.py \
     --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only
   python scripts/generate_fsps_ssp_grid.py \
     --stellar-only \
     --output Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
     --validate-only
   python scripts/generate_fsps_gas_grid.py \
     --output Data/popcosmos_chabrier_gas_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --base-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
     --validate-only
   python scripts/generate_fsps_agn_component_grid.py \
     --output Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
     --reference-ssp Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
     --validate-only
   ```

### Outputs

- implementation branch: `feature/ssp-vram-compression`
- validation log under `outputs/ssp_compression/bootstrap_validation.log`

### Go Criteria

- `git status --short` is clean on the compression branch before edits.
- All dense baseline assets validate.
- No source dense `Data/*.h5` file is modified in place. Compressed outputs use
  new filenames such as `Data/popcosmos_chabrier_agn_component_basis_k32.h5`.

## Phase 1 - Baseline Memory And Science Inventory

Quantify what the dense path costs before optimizing it.

### Work

- Record asset shapes, dtypes, HDF5 size, logical tensor bytes, chunking, and
  metadata.
- Measure resident JAX context size for:
  - no gas, no AGN;
  - dense gas only;
  - dense AGN component only;
  - dense gas plus dense AGN.
- Measure actual GPU memory and max successful batch size for:
  - `predict`/SED batch;
  - fit batch with `jax_adam_vmap` if used;
  - one-row fit smoke to keep behavior stable.
- Separate static resident tensors from per-galaxy intermediates. A compressed
  grid lowers static memory; very large batches may still be limited by
  `(batch, age, wave)` or `(batch, wave)` intermediates.

### Scripts To Add

```text
scripts/inventory_spectral_assets.py
scripts/profile_vram_batch.py
```

2026-05-29 status: implemented. Both scripts are metadata-first by default and
do not instantiate `load_context` or copy the multi-GiB gas/AGN arrays to JAX.
Baseline outputs were written to:

```text
outputs/ssp_compression/baseline_asset_inventory.json
outputs/ssp_compression/baseline_vram_profile.json
outputs/ssp_compression/baseline_asset_sizes.png
```

The dense full-AGN PopCosmos config currently has an estimated float32 resident
spectral payload of about `6.66 GiB` for base SSP + gas grid + AGN component
grid, before JAX allocator overhead, compiled executables, optimizer state, and
batch intermediates.

### Outputs

```text
outputs/ssp_compression/baseline_asset_inventory.json
outputs/ssp_compression/baseline_vram_profile.json
outputs/ssp_compression/baseline_batch_capacity.json
outputs/ssp_compression/baseline_asset_sizes.png
```

### Go Criteria

- Dense baseline resident bytes are explicit.
- Current maximum batch size is measured, not guessed.
- The report distinguishes file size, host RAM, GPU resident arrays, and
  per-batch intermediates.

## Phase 2 - Compression Methods To Test

All candidate methods must be JAX-friendly and must support `jit`, `vmap`, and
gradients. The first implementation should reconstruct spectra per galaxy for
scientific validation. A later optimization can project directly to photometry.

### Base SSP Candidates

The base SSPs are not the main VRAM problem, but they are a useful low-risk
testbed.

Methods:

- linear-flux SVD basis:

  ```text
  ssp_flux[stellar_lgmet, age, wave]
    ~= coeff[stellar_lgmet, age, k] @ basis[k, wave]
  ```

- normalized linear SVD:

  ```text
  ssp_flux = scale[stellar_lgmet, age] *
             coeff[stellar_lgmet, age, k] @ basis[k, wave]
  ```

- non-negative or clipped-positive reconstruction only if negative tails are
  scientifically negligible and documented.

### AGN Component Candidates

The AGN component grid is the first serious target because it is large and
does not require nebular line separation.

Diagnostics:

- Test whether `agn_lnu_per_mformed` is close to linear in `fagn`. If yes,
  store one scaled amplitude axis instead of all eight `fagn` slices.
- Test low-rank bases over `(tau, stellar_lgmet, age)` and over the full
  `(fagn, tau, stellar_lgmet, age)` curve set.
- Test `k = 8, 16, 32, 64`.

Candidate format:

```text
ssp_wave[wave]
ssp_lgmet[stellar_lgmet]
ssp_lg_age_gyr[age]
agn_tau_grid[agn_tau]
fagn_grid[fagn]                       optional if not factored out

agn_basis[k, wave]
agn_coeff[fagn, agn_tau, stellar_lgmet, age, k]
agn_scale[...]                        optional
```

If fagn linearity passes, use:

```text
agn_basis[k, wave]
agn_coeff[agn_tau, stellar_lgmet, age, k]
fagn_normalization = linear
```

### Gas Grid Candidates

The gas grid is line-rich. Do not force the full line-contaminated spectrum
into a smooth continuum basis as the final science path.

Diagnostics:

- Decompose the dense grid as:

  ```text
  gas_grid = pure_stellar_ssp + nebular_delta
  ```

  and test whether `nebular_delta` compresses better than the full gas grid.

- Prefer enriched generation with separate continuum and lines:

  ```text
  nebular_continuum_flux[gas_lgmet, gas_lgu, stellar_lgmet, age, wave]
  line_flux_grid[gas_lgmet, gas_lgu, stellar_lgmet, age, line]
  emline_wavelengths[line]
  line_name[line]
  ```

- If enriched generation is not ready, prototype pseudo-continuum extraction
  from the dense grid only for algorithm development. Do not label that
  prototype as a science asset.

Candidate format:

```text
ssp_wave[wave]
ssp_lgmet[stellar_lgmet]
ssp_lg_age_gyr[age]
gas_lgmet_grid[gas_lgmet]
gas_lgu_grid[gas_lgu]

gas_continuum_basis[k, wave]
gas_continuum_coeff[gas_lgmet, gas_lgu, stellar_lgmet, age, k]
gas_continuum_scale[...] optional

line_wave[line]
line_luminosity[gas_lgmet, gas_lgu, stellar_lgmet, age, line]
line_name[line]
```

### Metrics

For every candidate:

- resident payload bytes and predicted GPU context bytes;
- compression ratio relative to dense resident tensor;
- reconstruction error in linear flux;
- broad-band dense-vs-compressed residuals by band and level;
- residuals versus redshift, dust, stellar metallicity, gas metallicity,
  ionization, AGN `fagn`, AGN `tau`, and recent SFR;
- negative reconstructed flux fraction;
- gradient finiteness;
- JIT compile time and wall time.

### Go Criteria

Initial candidate acceptance:

- at least `5x` resident tensor reduction for AGN or gas alone;
- target `10x+` reduction for the combined gas+AGN resident payload;
- no NaNs and finite gradients in valid parameter space;
- dense-vs-compressed broad-band residuals:
  - target median `|delta_mag| < 0.005`;
  - target p95 `|delta_mag| < 0.015`;
  - temporary prototype ceiling p95 `< 0.03` if FSPS/Prospector final closure
    still passes.

### 2026-05-29 Follow-Up: Smaller Scientific Representations

After the first `k64` gas and `k32` AGN compressed assets, the next objective is
not just “lower k”. The first assets already reduce the dense gas+AGN spectral
payload from about `6.66 GiB` to about `85.9 MiB`, but further reduction should
preserve information where photometry is sensitive:

- The compressed gas asset is dominated by `gas_coeff`: about `15.36 MiB`
  raw out of an `18.32 MiB` raw compressed payload. The basis is only about
  `2.72 MiB`; reducing coefficient storage has the largest payoff.
- The compressed AGN asset can likely be much smaller. Sampled spectral
  reconstruction showed AGN `k=12` is already near the `k32` error floor for
  the current basis family, while `k=8` is still acceptable for many smooth
  continuum checks but needs photometry gates before use.
- A sampled dense-grid audit found the AGN component is effectively linear in
  `fagn`: `spectrum / fagn` residuals across the `fagn_grid` were below
  `~2e-7` relative on the sampled slices. This means the compressed AGN format
  should drop the explicit `fagn` axis and multiply by `fagn` at runtime,
  after adding a full benchmark/test gate.
- After gas and AGN are compressed, the base SSP becomes the largest remaining
  resident tensor at about `55 MiB`. Earlier SSP diagnostics showed normalized
  low-rank stellar SSP compression is viable: `k64` had p95 log-flux errors of
  about `0.004-0.005 dex` on the tested log-wavelength grid, and `k128` pushed
  p95 to about `0.0015 dex`. This should be revisited because it can now save
  more VRAM than further micro-optimizing the already-small AGN basis.
- Gas is different: sampled reconstruction improves steadily through `k64`,
  and line-peak preservation is visibly better at `k48-k64` than at `k16-k32`.
  Therefore a lower-rank full-spectrum gas basis is unlikely to be the final
  solution.
- A naive `float16` conversion of all arrays is invalid because `gas_scale`
  and `agn_scale` live around `1e-22..1e-11`, below or near unsafe `float16`
  ranges. If mixed precision is used, keep scale as `float32` or store
  `log10(scale)`.
- Sampled mixed-precision checks suggest `basis16 + coeff16 + scale32` adds
  only small reconstruction error for gas, and `basis32 + coeff16 + scale32`
  is safer for AGN. This should become an explicit `mixed_float16` asset mode
  and must be benchmarked photometrically, not enabled by dtype casting.
- Global `int8` quantization of coefficients is plausible as an experiment,
  but `int8` basis quantization damages small spectral structure, especially
  for AGN. If int8 is pursued, keep the basis in float16/float32 and quantize
  coefficients with explicit per-component or per-block scales.

Updated optimization priority:

1. Add a mixed-precision compressed asset format:
   `basis_dtype`, `coeff_dtype`, `scale_storage`, quantization metadata, and a
   loader that reconstructs in `float32` inside JAX.
2. Add an AGN-linear-in-`fagn` compressed format:
   `agn_coeff[agn_tau, stellar_lgmet, age, k]` plus runtime multiplication by
   `fagn`. Benchmark it against the dense `fagn` axis before making it the
   default.
3. Build and benchmark:
   - AGN `k12/k16` with `coeff=float16`, `basis=float32`, `scale=float32`;
   - gas `k48/k64` with `coeff=float16`, `basis=float16`, `scale=float32`.
4. Add a compressed base SSP mode, initially `k64/k128`, and benchmark
   stellar-only, dust-only, and full photometry. This becomes high-leverage
   once gas/AGN are no longer multi-GiB tensors.
5. Implement the real gas science representation:
   low-rank continuum plus sparse line luminosities. Do not rely on a smaller
   full-spectrum basis to preserve all narrow line/filter crossings.
6. Add filter-aware compression as a second-stage optimization: optimize
   basis/coefficients against broad-band flux errors over sampled redshifts and
   filters, not only pointwise SED reconstruction.

Implemented follow-up slice:

- `scripts/audit_compression_tradeoffs.py` now generates the method/factor/loss
  audit from sampled dense gas/AGN spectra plus existing SSP diagnostics. The
  current output directory is:

  ```text
  outputs/ssp_compression/tradeoffs/
  ```

- Implemented `model.ssp_model: compressed_basis`, loaded from
  `model.compressed_ssp_path`. In this mode `load_context` does not copy the
  dense base `ssp_flux` tensor to JAX; it keeps `ssp_basis`, `ssp_coeff`, and
  `ssp_scale` as the resident representation and reconstructs only the
  metallicity-selected age-by-wavelength SSP.
- Implemented compressed AGN assets with
  `fagn_handling: linear_runtime_multiplier`, where `agn_coeff` has shape
  `[agn_tau, stellar_lgmet, age, k]` instead of
  `[fagn, agn_tau, stellar_lgmet, age, k]`.
- Implemented explicit mixed-precision storage for compressed SSP, gas, and
  AGN builders. The loaders preserve `float16` resident basis/coeff arrays and
  cast selected slices to `float32` for reconstruction. Scale arrays remain
  `float32`.

Current audit recommendations:

```text
AGN default experiment:
  fagn factored, k=12 or k=16, basis=float32, coeff=float16, scale=float32

Gas default experiment:
  k=64, basis=float16, coeff=float16, scale=float32
  k=48 is a smaller comparison point, but needs photometry and line checks

Base SSP default experiment:
  k=64 for aggressive memory savings
  k=128 for safer stellar-continuum validation
```

Build commands for the recommended first compact assets:

```bash
python scripts/build_compressed_ssp_grid.py \
  --input Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
  --output Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5 \
  --k 64 --basis-dtype float32 --coeff-dtype float16 --overwrite

python scripts/build_compressed_gas_grid.py \
  --input Data/popcosmos_chabrier_gas_ssp_grid.h5 \
  --output Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5 \
  --k 64 --basis-dtype float16 --coeff-dtype float16 --overwrite

python scripts/build_compressed_agn_component_grid.py \
  --input Data/popcosmos_chabrier_agn_component_ssp_grid.h5 \
  --output Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5 \
  --k 12 --factor-fagn --basis-dtype float32 --coeff-dtype float16 --overwrite
```

Use `k48` gas and `k128` SSP as comparison assets when exploring the final
memory/accuracy frontier.

## Phase 3 - Build Compressed Assets

Add builders that read dense assets and write compressed scientific assets.
The builders may use NumPy/SciPy-style CPU algorithms offline, but the output
format must be simple JAX arrays.

### Scripts To Add

```text
scripts/build_compressed_ssp_grid.py
scripts/build_compressed_agn_component_grid.py
scripts/build_compressed_gas_grid.py
scripts/validate_compressed_spectral_asset.py
```

2026-05-29 status:

- `scripts/build_compressed_agn_component_grid.py` implemented for the dense
  FSPS-native AGN component grid.
- `scripts/build_compressed_gas_grid.py` implemented as a first low-rank
  full-spectrum gas-grid prototype.
- `scripts/validate_compressed_spectral_asset.py` validates both compressed
  AGN and compressed gas assets without loading dense source grids.
- `scripts/benchmark_dense_vs_compressed_spectral_assets.py` added for
  identical-parameter dense-vs-compressed photometry checks with progress
  reporting. It loads one level at a time and defaults to lazy dense AGN slice
  reads to avoid preloading the multi-GiB dense AGN tensor twice.
- `scripts/benchmark_photometry_engines.py` added for subprocess-isolated
  comparison of `dsps_dense_lazy`, `dsps_dense_resident`, `dsps_compressed`,
  and `fsps_prospector`, with wall time and peak RSS recorded per engine/level.
  Dense-lazy reads dense AGN HDF5 slices per point; dense-resident is skipped by
  a memory guard when estimated static payload plus overhead is too close to
  available memory.
- `scripts/build_compressed_ssp_grid.py` is now implemented. It became useful
  after gas/AGN compression because the base SSP is then the largest remaining
  resident spectral tensor.

### Required Metadata

Every compressed asset must include:

```text
asset_kind
compression_kind
source_grid_path
source_grid_sha256 or source_grid_size_mtime
compression_version
basis_space
k_basis
normalization
line_handling
imf_type = 1
imf_name = chabrier
z_sun = 0.0142
fsps_version
python_fsps_version
isochrones
spectral_library
units_*
dense_validation_summary_json
generation_command
```

### Outputs

Candidate assets:

```text
Data/popcosmos_chabrier_stellar_ssp_basis_k{K}.h5
Data/popcosmos_chabrier_agn_component_basis_k{K}.h5
Data/popcosmos_chabrier_gas_grid_basis_k{K}.h5
```

Validation products:

```text
outputs/ssp_compression/assets/*_validation.json
outputs/ssp_compression/assets/*_reconstruction_examples.png
```

### Go Criteria

- Validation can reject wrong IMF, wrong `z_sun`, wrong axes, wrong units, and
  unsupported compression formats.
- Builders never overwrite dense sources in place.
- The compressed asset can be loaded without allocating the dense source grid.

## Phase 4 - JAX Model Integration

Add parallel compressed modes while keeping dense modes unchanged.

### Config

Dense modes stay:

```yaml
model:
  nebular_model: gas_grid
  gas_grid_path: Data/popcosmos_chabrier_gas_ssp_grid.h5
  agn_model: fsps_component_grid
  agn_component_grid_path: Data/popcosmos_chabrier_agn_component_ssp_grid.h5
```

Compressed modes:

```yaml
model:
  ssp_model: compressed_basis
  compressed_ssp_path: Data/popcosmos_chabrier_stellar_ssp_basis_k64.h5
  nebular_model: compressed_gas_grid
  compressed_gas_grid_path: Data/popcosmos_chabrier_gas_grid_basis_k64.h5
  agn_model: compressed_fsps_component_grid
  compressed_agn_component_grid_path: Data/popcosmos_chabrier_agn_component_basis_k32.h5
```

The exact key names can change during implementation, but the principle cannot:
if a config asks for a compressed mode, missing compressed arrays must raise a
clear error.

2026-05-29 status: the first compressed runtime modes exist:

```yaml
model:
  nebular_model: compressed_gas_grid
  compressed_gas_grid_path: Data/popcosmos_chabrier_gas_grid_basis_k64.h5
  agn_model: compressed_fsps_component_grid
  compressed_agn_component_grid_path: Data/popcosmos_chabrier_agn_component_basis_k32.h5
```

The dense gas/AGN fields remain `None` in these modes. There is no silent
fallback to the dense `ssp_flux` or `agn_lnu_per_mformed` grids.

### Context

Add compressed fields to `DspsContext`, for example:

```text
compressed_ssp_basis_jax
compressed_ssp_coeff_jax
compressed_gas_basis_jax
compressed_gas_coeff_jax
compressed_gas_scale_jax
compressed_gas_line_wave_jax          future continuum-plus-lines mode
compressed_gas_line_luminosity_jax    future continuum-plus-lines mode
compressed_agn_basis_jax
compressed_agn_coeff_jax
compressed_agn_scale_jax
```

For compressed modes, the dense fields for that component should remain `None`.
Tests should assert that the dense gas/AGN tensors are not loaded.

### Forward Model

Use coefficient-space interpolation and SFH summation:

```text
coeff[age, k] = interpolate physical axes in coefficient tensor
galaxy_coeff[k] = sum_age(age_weights[age] * coeff[age, k])
sed[wave] = galaxy_coeff[k] @ basis[k, wave]
```

For age-dependent dust, compute young and old components separately:

```text
young_coeff[k] = sum_young(age_weights * coeff[..., k])
old_coeff[k] = sum_old(age_weights * coeff[..., k])

young_sed = young_coeff @ basis
old_sed = old_coeff @ basis

dusted = young_sed * diffuse_dust * birth_cloud_dust
       + old_sed * diffuse_dust
```

For gas lines, apply dust at `line_wave` and rasterize/project line fluxes
without materializing the full dense gas tensor.

Current gas-compressed implementation note: the first `compressed_gas_grid`
mode reconstructs a low-rank approximation of the full gas spectrum after
interpolating in `(gas_lgmet, gas_lgu)`. This is useful for VRAM testing, but
it remains a prototype for science because it does not yet separate nebular
continuum and sparse emission-line luminosities.

For AGN, preserve the validated semantics:

- FSPS-native additive component per formed mass;
- same SFH age weights as the stellar path;
- same host attenuation mode;
- same IGM ordering.

### Go Criteria

- `load_context` for compressed modes does not allocate the dense gas or dense
  AGN component tensor.
- `jax.jit`, `vmap`, and gradients work.
- Dense and compressed modes can be selected by config with no hidden fallback.

## Phase 5 - Batch-Size Optimization

After resident tensors are compressed, profile whether per-galaxy wave
intermediates become the next bottleneck.

### Work

- Benchmark max batch size for dense versus compressed modes.
- Inspect whether `(batch, age, wave)` and `(batch, wave)` arrays dominate at
  large batch.
- If needed, add a second optimization that avoids storing full SEDs for every
  galaxy in a batch:
  - compute only photometry for fit batches;
  - keep SED reconstruction only for diagnostics;
  - chunk wavelength integration internally;
  - later test filter-projected basis photometry.

### Photometry-Direct Candidate

This is not the first compressed implementation because redshift, dust, and
IGM are continuous. It becomes attractive once basis-space spectra validate.

Potential target:

```text
basis_photometry[k, band, z] or dynamic projected_basis[k, band]
line_photometry[line, band, z]
```

Any redshift grid interpolation introduced here needs its own science
benchmark.

### Go Criteria

- compressed mode permits materially larger GPU batches than dense mode;
- wall time per galaxy does not regress enough to erase the batch-size gain;
- fit/predict APIs expose a mode that avoids saving full SEDs in production
  batch fitting.

## Phase 6 - Benchmarks

### Dense Versus Compressed

Required levels:

- AGN component only;
- gas only;
- `full_noagn`;
- `full_agn`.

Metrics:

- per-band `delta_mag` and `delta_flux/flux`;
- median, p95, p99, max by level and band;
- residuals versus `z_obs`, `tau2`, `dust_index_n`,
  `log10_stellar_metallicity`, `log10_gas_metallicity`,
  `log10_gas_ionization`, `ln_fagn`, `ln_tauagn`, and recent SFR;
- non-finite and effectively-faint counts kept explicit;
- gradient and optimizer smoke tests.

Initial command template:

```bash
python scripts/benchmark_dense_vs_compressed_spectral_assets.py \
  --dense-config configs/popcosmos_binned.yaml \
  --compressed-gas-grid Data/popcosmos_chabrier_gas_grid_basis_k64.h5 \
  --compressed-agn-component-grid Data/popcosmos_chabrier_agn_component_basis_k32.h5 \
  --levels stellar_plus_gas full_noagn stellar_plus_agn full_agn \
  --n 50 \
  --seed 0 \
  --runtime cpu \
  --out outputs/benchmarks/dense_vs_compressed_popcosmos_n50
```

Use `--runtime gpu` only after a small CPU residual check and after estimating
resident payloads with `scripts/profile_vram_batch.py`.
Use `--dense-agn-mode resident` only when deliberately measuring the true dense
resident AGN mode; the default `lazy` mode prevents avoidable OOM during
photometric residual checks.

Three-engine comparison template:

```bash
python scripts/benchmark_photometry_engines.py \
  --config configs/popcosmos_binned.yaml \
  --compressed-gas-grid Data/popcosmos_chabrier_gas_grid_basis_k64.h5 \
  --compressed-agn-component-grid Data/popcosmos_chabrier_agn_component_basis_k32.h5 \
  --engines dsps_dense_lazy dsps_compressed fsps_prospector \
  --levels stellar_plus_gas full_noagn stellar_plus_agn full_agn \
  --n 50 \
  --seed 0 \
  --runtime cpu \
  --out outputs/benchmarks/photometry_engines_n50
```

Add `dsps_dense_resident` to `--engines` only when explicitly auditing the true
resident mode. The memory guard skips runs that are likely to OOM; use
`--no-memory-guard` only for deliberate stress testing.

### FSPS/Prospector

After dense-vs-compressed passes:

```bash
python scripts/benchmark_against_fsps_prospector.py \
  --runtime cpu \
  --config configs/popcosmos_binned.yaml \
  --compressed-gas-grid Data/popcosmos_chabrier_gas_grid_basis_k64.h5 \
  --compressed-agn-component-grid Data/popcosmos_chabrier_agn_component_basis_k32.h5 \
  --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn \
           stellar_plus_agn stellar_plus_dust_plus_agn \
           stellar_plus_gas_plus_agn full_agn \
  --n 50 \
  --seed 0 \
  --out outputs/benchmarks/compressed_popcosmos_binned_n50

python scripts/benchmark_against_fsps_prospector.py \
  --runtime cpu \
  --config configs/popcosmos_binned.yaml \
  --compressed-gas-grid Data/popcosmos_chabrier_gas_grid_basis_k64.h5 \
  --compressed-agn-component-grid Data/popcosmos_chabrier_agn_component_basis_k32.h5 \
  --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn \
           stellar_plus_agn stellar_plus_dust_plus_agn \
           stellar_plus_gas_plus_agn full_agn \
  --n 500 \
  --seed 0 \
  --out outputs/benchmarks/compressed_popcosmos_binned_n500
```

These commands inject compressed asset paths without editing the science config.

### Runtime Benchmarks

Compare dense and compressed modes on the same GPU:

- max resident GPU memory;
- max successful `--batch-size`;
- wall time per batch;
- wall time per galaxy;
- JIT compile time;
- one-row fit stability;
- small batch fit stability.

### Go Criteria

- dense-vs-compressed science residuals pass the compression error budget;
- FSPS/Prospector `n=50`, then `n=500`, still pass the broad-band closure
  criteria for bright finite rows;
- compressed mode gives a clear max-batch-size improvement.

## Decision Matrix

| Candidate | Resident VRAM Gain | SFH Linear | JAX Friendly | Science Risk |
|---|---:|---|---|---|
| lossless HDF5 gzip/LZF only | none after load | yes | yes | low but not useful for VRAM |
| log-flux SVD | high | no | medium | high: does not commute with SFH sums |
| linear SVD full spectra | high | yes | high | medium: negative tails and line errors |
| normalized linear SVD | high | yes | high | medium: scale choice affects residuals |
| AGN fagn factoring + SVD | very high if valid | yes | high | low-medium: must prove fagn linearity |
| gas stellar+nebular delta | medium-high | yes | high | medium: needs careful dust/line handling |
| continuum SVD + sparse lines | high | yes | high | lowest gas path, needs enriched asset |
| wavelet/sparse coefficients | high | conditionally | medium | batching complexity |
| photometry-direct basis | highest batch gain | yes if validated | medium | redshift/IGM interpolation risk |

## Recommended Implementation Order

1. Create the clean branch from `dev` and validate the existing dense `Data/`.
2. Add baseline asset and GPU memory profiling.
3. Build and validate compressed AGN component assets first.
4. Add `compressed_fsps_component_grid` JAX mode and benchmark AGN-only and
   full-AGN dense-vs-compressed.
5. Build gas compression diagnostics around `stellar + nebular_delta` and
   continuum-plus-lines.
6. Add `compressed_gas_grid` JAX mode and benchmark gas-only and `full_noagn`.
7. Optionally compress base SSPs after gas/AGN are stable.
8. Profile large GPU batches and add photometry-only/chunked execution if wave
   intermediates become the next limit.
9. Run FSPS/Prospector `n=50`, then `n=500`, using the compressed configs.

The first production candidate should be:

```text
compressed AGN component grid:
  fagn factoring if validated + linear SVD basis, k=16 or k=32

compressed gas grid:
  linear continuum basis + sparse line luminosities, k=32 or k=64
```

This directly targets the two multi-GiB resident tensors and is the most likely
path to substantially larger GPU galaxy batches without hiding scientific
approximations.
