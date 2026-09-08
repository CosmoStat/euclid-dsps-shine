# Full decoder quadrature qualification

## What changes

`model.photometry_integrator: merged_gauss4_v1` uses the qualified merged-knot
Gauss4 integration and its matching AB normalization in `predict_mags_jax`.
Projection arithmetic is float64 and requires `JAX_ENABLE_X64=true`. Stored SSP
assets, SFH/metallicity/dust/IGM equations and their existing mixed-precision
casts are unchanged. This is NOT an all-float64 decoder. Missing configuration
means `legacy_trapezoid_v1`, preserving historical execution and checkpoints.
Unknown names fail. Sleep cache receipts now identify the integration contract;
historical receipts without this key mean legacy and cannot be reused as merged
banks. Never edit receipts to relabel a bank: regenerate the photometry instead.
The exported `candidate_config.yaml` describes the diagnostic target, not a
ready-to-launch training configuration: its inherited sleep bank is still old
and will be rejected by the new cache check.

## Experiment

The fixed-spectrum reference must have completed with numerical PASS, with
unchanged hashed artifacts, and originate from the same balanced experiment.
The runner rechecks the reference at execution. No catalogue truth is read.

Six points: first three frozen-parent cache parameters, followed by one direct
q draw for each of the first three fixed observed validation contexts. All
points and row identities are exported. No truth-, residual- or ESS-based case
selection. Cache fluxes are used only to check the LEGACY target; candidate
fluxes are freshly decoded. Neither posterior nor prior is optimized.

Compare legacy and candidate full canonical targets, sequentially, all fifteen
latent-x coordinate directions. Each uses ten FD steps from .02 to .0000390625.
JVPs include the full SFH/SED/IGM/projection and likelihood path. Report per-band
flux derivatives in sigma/x, centered Gaussian loglike, logprior and canonical
loglike. The centered loglike removes only the observed normalization constant.
Its AD must agree with canonical loglike AD; the latter's scalar FD is retained
as a diagnostic, not used to certify cancellation of a large normalization sum.
Also compare reverse-mode canonical logtarget gradients against the JVP sum
of logprior and loglike (.01 absolute plus .1% relative tolerance). Backward work
is included, but only for one point at a time, not a large NPE training batch.

FD steps are chosen from the finest resolved three-step plateau without looking
at AD. Flux tolerances: .01 sigma/x absolute and 1% relative. Density tolerances:
.1 absolute and 5% relative (existing local audit values). A conservative
float32 output-ULP screen is applied; it is not a bound on internal arithmetic.
Unresolved finite differences stay INCONCLUSIVE, not evidence of a wrong AD.
No thresholds are widened automatically. Native mixed precision can still block
the full target even when the fixed-spectrum integrator is sound.

Report forward changes and loglike changes on the same points, repeated
single-object steady forward times, total case time including compilation and
device allocator memory statistics (whole-process peaks, not isolated operator
memory). This does not certify throughput or memory for large training batches.
Forward differences are reported, not forced small by fitting calibration/noise.

## Launch on Jean-Zay

One node, one H100, 16 CPU threads; sequential variants. Slurm ceiling 90 minutes
(1.5 GPU-hours), internal ceiling 80 minutes and 6000 counted component calls.
JVP units are not forward-equivalent cost. No follow-up training is submitted.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark

source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
export LOCAL_VI_FULL_DECODER_REFERENCE="$(dirname "$BALANCED_ROOT")/frozen_parent_photometry_reference_v1"
unset LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_REDSHIFT_DECOMPOSITION
unset LOCAL_VI_PHOTOMETRY_REFERENCE LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
export DIAGNOSTIC_ROOT="$(dirname "$BALANCED_ROOT")/frozen_parent_full_decoder_qualification_v1"
bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

`LOCAL_VI_FULL_DECODER_REFERENCE` must identify the successful photometry-reference
run. Change this explicit path if it used another name. Existing output roots are never
overwritten. A retry needs a new root after inspecting the cause of failure.

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_decoder_qualification.py "$DIAGNOSTIC_ROOT"
```

## Decisions afterwards

`FULL_DECODER_QUALIFICATION_COMPLETE` is execution completion, not PASS.
If candidate checks remain unresolved, inspect coordinate/band FD curves and
remaining mixed precision. Do not relaunch NPE around a failed gradient audit.
If they pass, inspect forward changes and the numerical model used to create
the synthetic catalogue. This runner does NOT establish catalogue simulator
compatibility, posterior calibration or prior accuracy. Keep historical results
separate. Review/regenerate sleep and internal validation banks with the new
contract; measure a small actual training microbatch before any large run.
Then resume bounded local VI and matched frozen-parent NPE. Population training
remains subject to posterior support/predictive gates, not numerical checks alone.

## Local verification

Model/photometry/local diagnostic suite: 95 passed, three asset-dependent skips.
Final qualification, workflow, frozen-NPE and posterior regressions: 58 passed
(overlapping suites, not 153 distinct tests). Includes full synthetic SED
derivatives, an intentionally broken gradient, legacy output preservation,
cross-integrator cache rejection and final artifact integrity. Compileall,
Ruff, CLI help and shell syntax passed. No cluster job was submitted locally;
actual SSP assets, source checkpoint and catalogue fits were unavailable.
