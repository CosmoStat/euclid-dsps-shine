# Redshift branch diagnostic

This follows gradient isolation job 1914143. It does not retrain anything and
does not modify a checkpoint, likelihood, noise model, or scientific gate.

## Measurement

Use cache entries 0, 1, 2 from the frozen balanced-v2 B parent-generated bank,
with the same first observed context selected by the existing local diagnostic.
No catalogue truth is read. These are numerical probes, not a representative
posterior evaluation. Entry 0 reproduces the previous audit's central input.

Split physical redshift into three inputs: stellar age/SFH, IGM, and observer
projection (filter integration plus distance). All other parameters remain
fixed. Record the full native function, its explicit recomposition, each branch,
their central flux differences, and the AD chain-rule sum. A separate comparison
to the canonical decoder must agree within 0.01 photometric error at the center;
this is an implementation identity check, NOT a posterior acceptance threshold.
The existing cache/live flux check is also retained at all three points.

Ten physical steps equal `abs(dz/dx) * 0.02 / 2**i`, `i=0..9`. Also record the
actual physical redshifts produced by the original latent-x stencil. This
distinguishes physical from latent derivatives; these stencils coincide only
to first order. Native input representability is recorded through actual spans.
No step is selected by closeness to AD. The report makes no convergence claim.

Two additional comparisons use float64 arithmetic: projection of the *fixed*
native post-IGM spectrum, and age/mass weights with fixed metallicity-derived
assets. Nested JAX traces must contain only float64 floating variables/constants
for those subgraphs. Stored SSP/filter values are promoted, not regenerated;
lost asset precision is not recovered. The complete decoder, IGM, dust, metallicity
interpolation, and latent transform are NOT certified float64 by these tests.
The only production helper change is an optional numerical dtype for two casts
in SFH mass normalization; its default remains float32.

Each spectrum and parameter point is evaluated separately. No object/sample
vmap, no full spectral Jacobian, and no additional arrays of posterior draws.
Only scalar-input JVPs and finite differences are used.

## Launch on Jean-Zay

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
export REPO_DIR="$PWD"
export DIAGNOSTIC_ROOT="$(dirname "$BALANCED_ROOT")/frozen_parent_redshift_decomposition_v1"
unset LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
export LOCAL_VI_REDSHIFT_DECOMPOSITION=1
bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
```

One node, one H100, 16 CPU threads, no array. Allocation ceiling 45 minutes
(0.75 GPU-hours), internal elapsed budget 40 minutes, 1000 component evaluation
units. A JVP is charged as one gradient evaluation, not as a forward-equivalent
cost. Eight curves per point imply 504 forwards + 24 JVPs, plus center/provenance
checks. Compilation time counts toward the elapsed budget. Runtime on the real
SSP bank has not been measured locally. Existing roots are never overwritten.

```bash
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
tail -F "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.out" \
  "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.err"
```

After completion, from the working checkout (not an older immutable snapshot):

```bash
cd "$WORK/dsps-popcosmos"
python scripts/summarize_feniks_sc_drws_redshift_decomposition.py "$DIAGNOSTIC_ROOT"
```

The reader verifies final artifact hashes. `--point 1` and `--point 2` show the
other stencils. Main artifacts are `REDSHIFT_DECOMPOSITION.json`,
`redshift_decomposition.csv`, `CACHE_FLUX_AUDIT.json`, and `FINAL.json`.
Partial tables and progress remain after failure. The monitor reports points
and branches, not misleading zero completed local-VI cases.

## Interpretation

- Stable float64 projection but unstable native projection implicates arithmetic
  precision in that sub-calculation; compare central forward differences too.
- Irregular projection in both precisions warrants spectral/filter interpolation
  resolution checks. AD/FD disagreement alone does not establish wrong AD.
- Irregular stellar branch with unstable age/mass weights localizes work upstream
  of photometry; compare precision and age-bin interpolation next.
- A branch center or chain-rule mismatch means the decomposition itself needs
  investigation before attributing failure to the physical model.
- Small error on these probes is not certification across the posterior support.

No automatic decoder correction, local VI continuation, or population run is
submitted after this diagnostic. Keep `scientific_promotion=false`.

## Local verification

The regression suite covering the model, spline, local VI and decomposition
passed (97 tests, three asset-dependent skips before adding the final reader
and tabulated-survival checks). The updated runner/diagnostic subset passed
25 tests; the final dedicated decomposition suite passed all seven tests.
Checks include the real DSPS kernels with small synthetic SSP spectra,
native branch center/chain identities, float64 trace validation including a
deliberately downcast negative control, and prepare/run receipts with mock physics.
Ruff, compileall, CLI help and Bash syntax checks also passed. No real SSP bank,
cluster checkpoint or GPU execution is available locally; catalogue fit smokes
were not run. This does not certify the Jean-Zay physical gradient.
