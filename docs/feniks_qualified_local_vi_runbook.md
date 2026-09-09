# Qualified Local VI Follow-up

## Evidence and question

Operator-provided results from Jean-Zay job 1923347 (September 9, 2026): the
precision-night run completed in 4:05:02, with six numerical qualification
points PASS, training smoke/B/C complete and all three K256 banks complete.
These are supplied cluster receipts, not locally reproduced catalogue results.

| Metric on the same 64 objects | A | B | C |
| --- | ---: | ---: | ---: |
| Median raw ESS/K | 0.02112 | 0.01128 | 0.01158 |
| k>0.7 or nonfinite fraction | 0.90625 | 0.953125 | 0.953125 |
| Residual absolute median | 2.0390 | 2.2547 | 1.8923 |
| Residual RMS | 20.6205 | 20.5113 | 7.5835 |
| Support gate | FAIL | FAIL | FAIL |

C improves residual tails but not importance support. Numerical qualification
is progress in the implementation, not evidence that q or the parent is correct.
No population update is justified. Relative held-out RMS gates are potentially
uninformative when the simulated-q reference has much larger residuals than the
observed cases. Small-cohort maximum-KS gates also need statistical context;
neither historical thresholds nor receipts are rewritten here.

The next question: can the *same conditional-flow family*, optimized separately
for each fixed observation, improve support under the unchanged qualified target?
Contrast observations with fresh simulations under that target. Do not attribute
a poor local result to the family alone: finite optimization and initialization
can also be limiting.

## Implemented protocol

- Requires the complete precision-night inventory, six-point qualification,
  successful explicit migration, C checkpoint/sidecar/config/stats hashes and
  unchanged frozen prior/calibration arrays. Every source file remains intact.
- Default source is C, chosen as a diagnostic starting point, not a promoted
  winner. The Python preparer can explicitly choose B for a later separate
  experiment; the standard launcher does not start a sweep.
- Eight observed development objects, chosen by the existing observed-photometry
  representative selector within the night's monitoring cohort. Check training
  disjointness and complete masks; abort instead of dropping invalid contexts.
  This is not a new independent test, nor a population-representative estimator.
- Eight new simulations: direct draws from the frozen parent, same observed
  errors/masks, same likelihood noise and noisy-r selection. Bounded rejection
  stops if no selected simulation is found. No catalogue parameter columns read.
- Local base mean/log-scale and coupling weights are optimized; the photometric
  context, topology, physical decoder, prior, calibration and feature stats stay
  fixed. The objective is mean(logq - logprior - loglike), not reconstruction.
- Two starts: original C and a 0.05-base-standard-deviation location perturbation.
  They are deliberately nearby, **not independent searches for all modes**.
  Each uses 64 Adam steps, four fresh reparameterized draws/step, learning rate
  0.001. Keep both final iterates, never select the better start or best draw.
- Evaluate the amortized baseline and both restored local checkpoints using two
  fresh independent sets of 128 direct draws each. Save x, flux and all densities.
  Report pooled K256 support **and** separate K128 ESS/max weights, evidence
  differences, moments/covariances, residuals and generated-parameter ranks.
  Optimization never sees the generated parameters as labels.
- Local prior/entropy gradients remain active through x, but no prior parameter
  is optimized. Source frozen-array fingerprint is checked at load and per case.
- Existing contract audit remains fail-closed, including live/cache agreement,
  sample/log_prob, serialization and AD-independent finite-difference checks.
  For the qualified target, perturbation inputs are also float64; do not round
  the finite-difference stencil in float32 before calling a float64 target.

Only the new precision64 bank is read, for a cache/live contract check. Fresh
simulated cases are decoded independently, not drawn from that bank. No old
mixed-precision flux bank is reused. Catalogue-simulator compatibility remains
unverified.

## Budget

One node, one H100, 16 CPUs, sequential cases, three-hour allocation ceiling.
Internal ceiling: 9900 seconds and 32000 decoder evaluations (gradient calls
count backward work separately, not as forward-equivalent timing). With eight
cases/group and two starts, the main loop uses 12288 evaluation forwards and
8192 gradient-sample evaluations, plus simulation/reference/contract checks.
The first observed/simulated pair measures actual cost. Stop with BUDGET_STOP
if the remaining projected cost exceeds the allocation; never extend it
automatically. Compilation or Slurm OOM can still stop the job.

## Launch on Jean-Zay

From the front end after updating the branch:

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git checkout feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
NIGHT="$BASE/frozen_parent_precision_night_v1"

# CPU-only artifact audit, no DSPS and no new GPU job.
python scripts/analyze_feniks_sc_drws_precision_night.py "$NIGHT" \
  --out "$BASE/frozen_parent_precision_night_readback_v1" &&
bash scripts/submit_feniks_sc_drws_qualified_local_vi.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$NIGHT" "$BASE/frozen_parent_qualified_local_vi_v1"
```

Both output directories must be new. Do not delete a partial diagnostic to
resubmit. If the CPU audit already exists, inspect it and run the submission
command separately with a new local-VI root. A readback completion is not a
scientific gate pass.

```bash
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
tail -n 80 "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.err"
python -m json.tool "$DIAGNOSTIC_ROOT/FINAL.json"
```

Results: `paired_comparison.csv` and `.png`, `cases/*/COMPLETE.json`,
`cases/*/{amortized,start_0,start_1}/SUMMARY.json`, `direct_draws.npz`, local
`parameters.eqx` and `optimization.csv`, `SIMULATED_INPUTS.npz`,
`CONTRACT_AUDIT.json`, `GRADIENT_AUDIT.json`, `COST_PREFLIGHT.json`.
The CPU audit writes `NIGHT_READBACK.json` with training loss/gradient evidence,
absolute observed/reference held-out residuals and a descriptive simultaneous
uniform-rank bound; it never changes old PASS/FAIL labels.

## Interpretation before any next run

- Clear local support improvement on observations and simulations supports
  working on amortized adaptation. It does not prove mode completeness.
- Failure on simulations calls for controlled family/optimization/simulation
  checks before attributing the problem to the observed population.
- Improvement on simulations but not observations motivates auditing context
  distributions and the generative model. It does not prove model mismatch.
- A better ELBO or photometric fit without better support is not success.
  Compare every paired case and both starts, not only a mean or best case.
- Keep the prior frozen. No teacher bank, global NPE, population job or automatic
  promotion follows this diagnostic. Analytic multimodal controls show that
  even excellent ESS can coexist with a missed mode.

## Verification status

50 targeted tests passed:

```bash
JAX_ENABLE_X64=true .venv/bin/python -m pytest -q \
  tests/test_qualified_local_vi.py tests/test_local_vi_diagnostic.py \
  tests/test_precision_night.py tests/test_npe_validation.py
```

Includes synthetic full-SED numerical controls, a real local optimizer under
mock physics, finite gradients/frozen arrays, sample/density/serialization,
independent evaluation draws, missing-mode analytic controls, receipt tampering,
training overlap and accidental YAML coordinate reordering. Compileall, Ruff,
Bash syntax, CLI help and Sphinx HTML with warnings-as-errors pass. Legacy fit
smoke configs named in AGENTS.md are absent; those catalogue fits were not run.
No MCMC or catalogue-truth validation was executed. No new Jean-Zay job is
submitted by this patch. Local mock-physics workflow tests validate plumbing,
not the SED inference conclusions.
