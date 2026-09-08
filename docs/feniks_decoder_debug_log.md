# FENIKS frozen-parent NPE: numerical debug log

Updated: 2026-09-09. Branch: `feature/feniks-exact-posterior-benchmark`.

## Evidence policy

Historical cluster results below are transcribed from the user's receipts and
terminal outputs. They have not been independently fetched from Jean-Zay in this
implementation session. Local tests are identified separately. COMPLETED refers
to execution, not numerical acceptance or scientific validation. Never overwrite
an earlier run to make its receipt pass.

No catalogue truth enters the new diagnostic, training or checkpoint-selection
paths. Historical truth-based closure results are retrospective observations,
not targets for repairing the code. No MCMC, SMC, AIS or nested sampling is used.

## Current decision

The merged quadrature and MDF64 candidate now passes all metallicity checks at
six tested inputs. Four complete points pass. Five required checks remain
INCONCLUSIVE: four centered log-likelihood checks at point 1 and one redshift
flux check at point 4. No remaining required FAIL is reported for this candidate.

This is numerical progress, not evidence of improved posterior support after
retraining. The population prior is still frozen; population training is blocked.

## Step-by-step record

| Step | Cluster evidence / code | Finding | Decision |
| --- | --- | --- | --- |
| 1. Frozen-parent pure-sleep NPE | 2026-09-05/06; frozen winner `warm_start`, validation sleep NLL 19.433409 | Historical redshift PIT KS 0.2502 -> 0.1330; support K1024 median ESS 5.98 -> 13.28, but bad-k fraction 0.6875; joint posterior not validated | Preserve winner for comparison; no population promotion |
| 2. Conditional-flow topology | `ecf3fb2`; pilot B/C training jobs 1829245/1829247; final internal recovery 1890513, closure 1890514 | Old transform counts `[0,6,0,6,0,6,0,6,0,6,0,6,0,6,3]`; rebuilt counts `[3,3,3,3,3,2,4,2,4,3,3,3,2,4,3]`; prior identical | Structural defect corrected, but observed support gates still fail |
| 3. Balanced sleep/ELBO | single-GPU fix `1f5462f`; anchor 1893047, candidates 1893048, validation 1893049 | At K256, all four support gates FAIL. D lowers residual RMS to 8.64 from A's 17.50, but ESS/K remains about 0.016 and bad-k about 0.918 | Stop using training-loss/PPC improvements as posterior certificates |
| 4. Local-VI preflight | `09bb8c7`, job 1913341; multiscale audit `8a04db4`, job 1913854 | Gradient audit cannot establish convergence. No local optimization starts | Diagnose full-target derivatives before more VI |
| 5. Flux/likelihood isolation | `9660681`, job 1914143, 4m13 | Likelihood-only derivative agrees with analytic control. Largest remaining discrepancy involves redshift through DSPS | Separate projection, stellar-age and IGM branches |
| 6. Redshift decomposition | `7f48a0e`, job 1915987, 11m19 | Fixed-spectrum observer projection retains large AD/reference discrepancies even in float64; stellar branch is better behaved | Precision alone does not repair historical quadrature |
| 7. Photometry reference | job 1918919, about 7m | Merged-grid Gauss4/Gauss8 agree with independent piecewise reference and pass all 54 band-gradient checks at three points | Introduce versioned opt-in integration, then test full decoder |
| 8. Full decoder | `ca92735`, job 1920226, 33m55 | Merged quadrature leaves nine metallicity FAIL and many INCONCLUSIVE checks; gradient identities pass. Forward cost about 0.012 s versus legacy about 1 s at tested points | Isolate MDF arithmetic; do not extrapolate timing to NPE throughput |
| 9. MDF precision | `293d5a9`, job 1921589, 4m04 | MDF64 analytic reference PASS; all metallicity checks pass. Full points 0/2/3/5 PASS; 1/4 remain INCONCLUSIVE | Inspect the five unresolved stencil curves |
| 10. Residual audit | Implemented in this update; not yet executed on H100 | Correct output-resolution accounting and export denser, representable stencils for CPU replay | No training; preserve previous receipts and thresholds |

Operational errors were tracked separately: missing `git` on compute nodes,
runtime-commit authorization, missing `filters` argument, missing historical
validation config key, oversized allocations, and a one-GPU job requesting
`pmap`. Their fixes enabled execution; none were scientific validation.

## MDF64 result details

Maximum candidate-minus-MDF32 flux changes at points 0..5, in observed sigma:
`[0.003286, 0.004168, 0.001599, 0.001722, 0.003048, 0.000849]`.
These are not differences from the original legacy integrator. MDF64 weight and
analytic derivative discrepancies are around 1e-16. The old weights themselves
already agreed at about 1e-8; the controlled change includes induced SSP and
survival contractions, not just the stored weights. It does not establish that
every historical AD derivative was wrong.

The full path remains mixed precision. Inputs, SSP assets, SFH/dust operations
and downstream casts have not all been converted to float64. Do not label this
candidate a full-float64 decoder.

## Why five controls remain unresolved

At point 1, dust_delta and SFH08/09/10 have coarse-step FD/AD disagreements below
0.002, against tolerances around 0.13--0.23. The old audit nevertheless imposed
a float32 output ULP screen on a float64 likelihood sum near -4332, producing
screen values 0.097656, 0.195312, 0.390625 at the first three steps. That proxy
prevents an otherwise stable plateau from qualifying. It is not a rigorous
propagated bound on upstream arithmetic.

At point 4 in lsst_z, redshift AD is -0.579424. FD approaches -0.581997 near
h=0.000313, then becomes noisy. One matching step does not establish a plateau.
The new audit must distinguish truncation from numerical noise without selecting
the step closest to AD. This case is not declared fixed in advance.

## New residual-audit protocol

- Freeze the merged/MDF64 decoder, prior, likelihood and observation contexts.
  Require the completed MDF receipt, its hashes, and exact original points.
- Revisit every unresolved required source check, capped at eight. For this
  source that is five checks. Unsupported components fail closed.
- Evaluate 25 predetermined half-octave steps from 0.02 to 0.02/4096 in latent x.
  Record actual representable positive/negative steps and use the asymmetric
  three-point derivative formula when rounding makes them unequal.
- Compute Gaussian differences per band using
  `delta_logL = -delta_residual * (anchor_residual + delta_residual/2)`.
  This avoids subtracting two large sums and is invariant to a likelihood
  additive constant. It does not alter the likelihood used by inference.
- Propagate the actual flux-output dtype's ULP screen through this expression.
  This only accounts for output rounding and reduction arithmetic, not unknown
  upstream quantization. Therefore also require stable finite-difference curves.
- Compare second-order stencils to Richardson estimates from nested stencils.
  Require three consecutive values stable in both sequences and agreement
  between them within the existing tolerance. Select using FD only, then compare
  to AD. Do not enlarge atol/rtol or choose a lucky step using AD.
- Verify reconstructed AD agrees with the source check. Save all perturbed
  fluxes in `TARGET_RESOLUTION_SNAPSHOT.json`, with final artifact hashes, for
  offline CPU replay. Historical full qualification is not rewritten.

Richardson agreement is empirical convergence evidence, not a proof of a smooth
full target or a rigorous rounding-error bound. A residual PASS still requires
consistent full requalification and simulator compatibility review. There is no
automatic bank reuse, local VI, NPE or population submission.

## Launch and inspect

On Jean-Zay, after pulling this update:

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
export LOCAL_VI_TARGET_RESOLUTION_REFERENCE="$(dirname "$BALANCED_ROOT")/frozen_parent_mdf_precision_qualification_v1"
export DIAGNOSTIC_ROOT="$(dirname "$BALANCED_ROOT")/frozen_parent_target_resolution_v1"
unset DIAGNOSTIC_LOG_ROOT LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_REDSHIFT_DECOMPOSITION
unset LOCAL_VI_PHOTOMETRY_REFERENCE LOCAL_VI_FULL_DECODER_REFERENCE LOCAL_VI_MDF_PRECISION_REFERENCE
unset LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

One node, one H100, 16 CPU threads, no array. Slurm ceiling 45 minutes
(0.75 GPU-hours), internal time ceiling 40 minutes, 1000 decoder evaluations.
At five checks the new stencil work is 255 forward and 5 gradient evaluations,
plus three legacy cache checks. Backward work is counted separately, not asserted
to cost the same as a forward evaluation. No training jobs follow.

After completion (also usable on a synced results folder on CPU):

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_target_resolution.py "$DIAGNOSTIC_ROOT"
```

The summary verifies final hashes and recomputes decisions from the snapshot.
An existing output root is never overwritten; use a new version for a retry.

## Step 11: residual audit result, job 1922142

Operator-provided evidence, 2026-09-09 01:07: `TARGET_RESOLUTION_COMPLETE`,
1m01 Slurm elapsed, 258 forward and 5 gradient evaluations, CPU replay PASS.
The four point-1 density checks pass the original tolerances. Their selected
fine stencils differ from AD by about 3--8%; coarse stencils were more accurate.
This is not a high-precision certificate. The selector is unchanged and does
not choose steps using their agreement with AD.

Point 4 / z_obs / lsst_z remains INCONCLUSIVE. AD=-0.579424; FD=-0.581997
at h=0.0003125 and -0.573820 at h~0.000221, but -0.598115 at h=0.00015625
and -1.334922 at h~0.00001. Two favorable steps are insufficient. Output ULP
estimates near 1e-10 do not bound rounding inside the mixed-precision decoder.
Do not loosen tolerances or restart NPE from this evidence.

## Step 12: targeted redshift-dependent precision (prepared)

This mode reuses point 4, its observed context, source receipts and parameters.
It compares ten branches sequentially: canonical latent target; mixed full,
stellar, IGM and projection; float64-z-path full, stellar, IGM and projection;
and float64 age/mass weights followed by native SED casts. All bands and the
centered likelihood are exported, not just the one unresolved band.

The diagnostic promotes redshift-dependent age/mass/SFH arithmetic, contractions,
IGM and projection. MDF/SSP arrays and dust transmission are fixed at the central
non-redshift parameters, retaining stored/native precision. Float64 branches
are checked by traversing their JAX traces. No YAML option switches production
to this path. The IGM helper defaults remain historical float32.

Branches use physical z=z0+(dz/dx)*u; canonical_x retains the original nonlinear
latent mapping and its casts. All derivatives are per equivalent latent unit,
but finite stencils are not identical between those two coordinate paths.
Central flux shifts and sum-of-branches minus full derivatives are reported.
Old thresholds, FD-only plateau selection and receipts remain unchanged.

Launch from the updated branch, with a new output root:

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
export LOCAL_VI_REDSHIFT_PRECISION_REFERENCE="$(dirname "$BALANCED_ROOT")/frozen_parent_target_resolution_v1"
export DIAGNOSTIC_ROOT="$(dirname "$BALANCED_ROOT")/frozen_parent_redshift_precision_v1"
unset DIAGNOSTIC_LOG_ROOT LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_REDSHIFT_DECOMPOSITION
unset LOCAL_VI_PHOTOMETRY_REFERENCE LOCAL_VI_FULL_DECODER_REFERENCE
unset LOCAL_VI_MDF_PRECISION_REFERENCE LOCAL_VI_TARGET_RESOLUTION_REFERENCE
unset LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

Budget: one node, one H100, 16 CPU threads, maximum 45 minutes (0.75 GPU-hours),
internal 40 minutes / 1000 evaluations. Ten 25-step curves cost 510 forward
calls and 10 JVP calls, plus setup and three legacy cache checks. Setup work and
branch calls are not claimed to be equal-cost full decoder evaluations.
No arrays, optimizer, NPE or population submission follow.

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_redshift_precision.py "$DIAGNOSTIC_ROOT"
```

Artifacts: `REDSHIFT_PRECISION.json`, `redshift_precision.csv`, full flux/JVP
stencils in `REDSHIFT_PRECISION_SNAPSHOT.json`, parameter archive and hashed
final receipt. The summary verifies hashes and replays all decisions on CPU.
Even a successful float64 branch does not qualify the production full target:
review the localized cause, implement a versioned correction, then requalify.

## Conditions for restarting training

1. Remaining numerical checks resolved and full-target audit consistently
   qualified. Keep the corrected integrator/MDF version in every receipt.
2. Check simulator/likelihood compatibility: filter assets, flux units, AB
   normalization, noise, masks and selection. No catalogue truth for tuning.
3. Generate fresh versioned noiseless sleep banks under the frozen prior.
   Do not mix old fluxes with a changed numerical decoder.
4. Run bounded local VI and model-generated validation first. Check mode/support
   behavior and photometric consistency, not only a lower ELBO.
5. Only then run matched NPE continuation and held-out posterior validation.
   Population updates remain gated by stable importance integration and
   representative observed validation, not just decoder qualification.

We are closer to the numerical prerequisite for a bounded training experiment.
Neither its success nor usable population-level inference is guaranteed by this
debug progress. The already-seen validation cohort is not a fresh test set.

## Local verification (latest implementation)

103 tests passed, 3 skipped across the numerical/model/workflow suites. New
coverage includes three synthetic SED/IGM cases with exclusively float64
z-path traces, AD/FD and branch-chain checks, unchanged native outputs, an
analytic collector/source-AD guard and mock receipt-linked cluster workflow.
Compileall, Ruff, Bash syntax, CLI help and Sphinx HTML with `-W` pass.
No real checkpoint/H100 point-4 follow-up was run locally or submitted here.

### Previous residual-audit implementation

39 targeted tests pass, covering the Gaussian difference identity,
likelihood-offset invariance, actual float32 step sizes, asymmetric quadratic
stencils, wrong-gradient rejection, missing-plateau rejection, synthetic DSPS
SED evaluation and receipt-linked workflow/replay. Compileall, Ruff, CLI help,
Bash syntax and Sphinx HTML with warnings-as-errors pass. Cluster execution is
recorded above in step 11 from operator logs. The legacy fit smoke configurations named in AGENTS.md are
absent; no catalogue fit or H100 verification is claimed.

Related runbooks: [full decoder](feniks_full_decoder_qualification_runbook.md),
[MDF precision](feniks_mdf_precision_runbook.md).
