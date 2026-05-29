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

## Phase 0 - Clean Worktree And Data Isolation

Start implementation in a clean worktree so compression experiments do not
pollute the current dirty checkout or overwrite the existing runtime assets.

### Work

1. Freeze or commit the current forward-model state before branching. A clean
   worktree created from a dirty checkout will not include uncommitted code.
2. Create an isolated worktree:

   ```bash
   git worktree add ../DSPS-ssp-compression -b feature/ssp-vram-compression HEAD
   ```

3. Populate `../DSPS-ssp-compression/Data/` with the required local assets.
   Prefer a real copy for safety. Reflinks/hardlinks are acceptable only if all
   compression builders write to new temp files and never mutate inputs.

   ```bash
   mkdir -p ../DSPS-ssp-compression/Data
   rsync -a --ignore-existing Data/ ../DSPS-ssp-compression/Data/
   ```

4. Validate assets in the worktree before changing model code:

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

- clean worktree: `../DSPS-ssp-compression`
- copied local `Data/` assets
- validation log under `outputs/ssp_compression/bootstrap_validation.log`

### Go Criteria

- `git status --short` is clean in the compression worktree before edits.
- All dense baseline assets validate.
- No source `Data/*.h5` file is modified in place.

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
  ssp_representation: compressed_basis
  compressed_ssp_path: Data/popcosmos_chabrier_stellar_ssp_basis_k64.h5
  nebular_model: compressed_gas_grid
  compressed_gas_grid_path: Data/popcosmos_chabrier_gas_grid_basis_k64.h5
  agn_model: compressed_fsps_component_grid
  compressed_agn_component_grid_path: Data/popcosmos_chabrier_agn_component_basis_k32.h5
```

The exact key names can change during implementation, but the principle cannot:
if a config asks for a compressed mode, missing compressed arrays must raise a
clear error.

### Context

Add compressed fields to `DspsContext`, for example:

```text
compressed_ssp_basis_jax
compressed_ssp_coeff_jax
compressed_gas_basis_jax
compressed_gas_coeff_jax
compressed_gas_line_wave_jax
compressed_gas_line_luminosity_jax
compressed_agn_basis_jax
compressed_agn_coeff_jax
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

### FSPS/Prospector

After dense-vs-compressed passes:

```bash
python scripts/benchmark_against_fsps_prospector.py \
  --runtime cpu \
  --config configs/popcosmos_binned.yaml \
  --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn \
           stellar_plus_agn stellar_plus_dust_plus_agn \
           stellar_plus_gas_plus_agn full_agn \
  --n 50 \
  --seed 0 \
  --out outputs/benchmarks/compressed_popcosmos_binned_n50

python scripts/benchmark_against_fsps_prospector.py \
  --runtime cpu \
  --config configs/popcosmos_binned.yaml \
  --levels stellar_only stellar_plus_dust stellar_plus_gas full_noagn \
           stellar_plus_agn stellar_plus_dust_plus_agn \
           stellar_plus_gas_plus_agn full_agn \
  --n 500 \
  --seed 0 \
  --out outputs/benchmarks/compressed_popcosmos_binned_n500
```

The config used for these commands must point to compressed assets once the
compressed modes exist.

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

1. Create the clean worktree with copied `Data/`.
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
