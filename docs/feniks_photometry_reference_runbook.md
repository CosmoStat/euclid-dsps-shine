# Fixed-spectrum photometry reference

## Scope and decision

Job 1915987 showed the main point-0 AD/FD discrepancy persists in projection
with a fixed spectrum in float64. This experiment tests a correction candidate:
integrate on merged redshifted-spectrum and filter knots using 4/8-point Gauss
quadrature on each segment. It does NOT replace production `predict_mags_jax`.

The independent NumPy reference analytically integrates the quadratic product
of the two piecewise-linear interpolants divided by wavelength on each merged
segment. Small-interval series avoid cancellation. Same SSP values, transmission
values, AB constant and cosmology; no noise inflation, no smoothing spectral
features, no truth-derived inputs. This reference is not proof that the input
SSP/filter assets themselves resolve all physical structure.

Measure legacy float64, merged Gauss4, merged Gauss8, and merged Gauss8 with the
legacy AB normalization (to isolate numerator versus denominator effects).
Twenty step sizes extend the earlier physical-z stencil down by another factor
1024. Count filter samples crossing spectrum interpolation knots and report
the nearest knot distance. Never choose the FD step closest to AD: pick a
three-step plateau from the independent reference alone, then compare AD.

Integration checks require agreement at centers AND both sides of every stencil,
not just a central normalization match. Flux tolerance is 0.001 photometric
error plus 1e-10 times reference flux; gradient plateau/agreement tolerance is
0.001 sigma per physical-z plus relative 0.001. These are software qualification
checks, not posterior coverage/PIT gates. Nonfinite differences cannot pass.

The same three frozen generated cache points and first observed context are
used. Export compiled and eager spectra to expose their numerical differences.
The export is small: spectra, filters, generated parameters, fixed observational
flux/error context; no whole checkpoint or SSP bank, and no catalogue truths.

## Launch

One node, one H100, 16 CPU threads. Slurm ceiling 45 minutes (0.75 GPU-hours),
internal elapsed ceiling 40 minutes, 1000 component evaluation units. It
generates only three fixed spectrum exports; subsequent scans do not recompute
SFHs/SSPs. Compilation time counts; a JVP is not forward-equivalent cost.
No cluster performance claim has been made from CPU-only local tests.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
unset LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_REDSHIFT_DECOMPOSITION
unset LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
export LOCAL_VI_PHOTOMETRY_REFERENCE=1
export DIAGNOSTIC_ROOT="$(dirname "$BALANCED_ROOT")/frozen_parent_photometry_reference_v1"
bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

Existing roots are preserved. Mutually exclusive diagnostic flags fail before
preparation. No NPE or population job is submitted as a dependency.

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/analyze_feniks_sc_drws_photometry_reference.py "$DIAGNOSTIC_ROOT"
```

The reader verifies final SHA256 artifacts and prints numerical qualification,
forward differences in error units, precision/normalization controls, and point-0
stencils. Use `--point 1` or `--point 2` for other points.

## CPU replay without SSP assets

Transfer `FIXED_SPECTRA.npz` and `SNAPSHOT.json` from the completed export to a
local directory. Replay can also recover a completed export after a later scan
failed; it never overwrites the original diagnostic. Run from the checked-out
project environment, not on a cluster login node for a long analysis:

```bash
JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu \
EUCLID_DSPS_REQUIRE_GPU=0 JAX_ENABLE_X64=true \
python scripts/analyze_feniks_sc_drws_photometry_reference.py \
  /path/to/export --replay-out /path/to/new_cpu_analysis
```

## After the result

`PHOTOMETRY_REFERENCE_COMPLETE` means measurements completed. Even numerical
checks `PASS` only authorize considering FULL DECODER QUALIFICATION, not NPE.

1. If candidate quadrature does not agree/converge, investigate grids/support
   and arithmetic with the exported data; do not train around the discrepancy.
2. If it passes, integrate it behind a versioned opt-in in a separate patch and
   check the full decoder gradients, forward residuals, memory and runtime on a
   bounded generated/observed cohort. Verify how the synthetic observed catalogue
   was generated: numerical changes must not silently switch its forward model.
3. If flux changes are material, invalidate old sleep/noiseless-flux banks and
   simulated validation under the old integrator. Regenerate them under the
   qualified decoder and record its fingerprint. The old projected prior may
   also have absorbed numerical errors: keep it frozen initially but revalidate
   it; do not declare it scientifically correct because integration passed.
4. Then compare corrected pure sleep and sleep+observed ELBO under matched
   budgets, with observed truth-free support/PPC and independent simulated
   calibration. Only afterwards reconsider population updates.

This is the correction path, not a guarantee that photometry integration explains
all previous approximation, identifiability or population errors.

## Local verification

Targeted quadrature, decomposition, local-VI and model suite: 91 passed, three
asset-dependent skips. Final quadrature/CPU-replay subset: six passed, including
hash verification, altered-file rejection and refusal to overwrite outputs.
The DSPS smoke uses synthetic spectra and the real numerical kernels, not the
cluster SSP assets. Compileall, Ruff, CLI help and shell syntax passed. No real
catalogue fit, source-checkpoint run or H100 execution was performed locally.
