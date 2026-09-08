# Fixed-parent local-VI diagnostic

This implements the first decision experiment of the
[diagnostic reset](feniks_vi_diagnostic_attack_plan_20260908.md), not a production
posterior or a new population-training campaign.

## Question and comparison

Can the corrected conditional-flow family approximate individual likelihood
constraints better when its parameters are adapted separately for each object?

- Source: the completed **balanced B** checkpoint, with corrected topology.
  Historical A is not reused as the local family because its topology is known
  to be incomplete. This experiment does not select B as scientifically valid.
- Cases: 16 observed validation objects selected deterministically by observed
  r flux, SNR and mask count; 16 fresh selected simulations at the same noise
  and mask contexts, from the unchanged parent. The validation catalogue has
  already been examined: it is not a new independent test.
- Each case: original amortized distribution, then two local VI starts.
  Optimize the base location/log-scale and existing coupling parameters; keep
  the photometric trunk/context, permutations, masks, prior, calibration,
  feature statistics and physical decoder unchanged.
- Objective: mean `logq - logprior - loglike`, with four reparameterized direct
  draws per step. This is a local reverse-KL diagnostic, not a replacement
  sleep trainer or an importance-resampled teacher. No MCMC or population step.
- Run 64 steps and evaluate the final iterate on two fresh independent banks
  of 128 draws, using the complete generating density. Both starts are reported;
  no best-start selection using generated parameters or historical truth.

The second start perturbs the base mean by 0.05 base standard deviations. It
tests local optimization sensitivity, not widely separated modes. Failure
after 64 steps does not prove that the family cannot represent the target.
High ESS from both starts also cannot exclude an entirely missed mode.

## Fail-closed contracts

Before optimization: sample/log-prob and inverse parity, checkpoint round trip,
sleep/inference feature parity (including a masked band), theta/x round trip,
canonical-target/export parity, effective noise scale, source-cache/live flux
parity and a finite-difference check of the target gradient through the decoder.

The gradient audit writes `GRADIENT_AUDIT.json` before deciding whether to
continue. It records loglike, logprior and logtarget separately, their automatic
derivatives, both function values at six prespecified step sizes (0.02 down to
0.000625), and a floating-point resolution screen. Select a resolved three-step
finite-difference plateau without consulting AD, then compare AD at the same
absolute/relative tolerances (0.1/0.05). No plateau means `INCONCLUSIVE`, and a
resolved disagreement means `FAIL`; both stop before local optimization. The
resolution screen is not a rigorous bound on roundoff inside DSPS.

Job 1913341 in v1 failed the older two-step convergence check before optimization.
The log's absolute difference alone cannot distinguish truncation, roundoff or
an actual derivative problem. Preserve that run; use a distinct v2 root with
the instrumented audit. No physics or training settings are changed by this
recovery, and a passing result is not presumed.

Legacy catalogue truth/reporting references are removed from a separate resolved
configuration before loading any rows. The source files are preserved, all
physical settings retained, and the read columns recorded. Dataset, source
checkpoint, feature statistics, cache and configuration hashes are checked.

Initial supported contract is Gaussian noise with zero jitter and fractional
floor, observed-catalogue errors, a one-component residual conditional flow in
latent-x coordinates and complete finite photometry. Unsupported contexts stop
the diagnostic rather than silently filtering objects or changing the model.

Gaussian positive and prior-only negative controls check that likelihood-based
ranks detect a proposal ignoring data even when marginal/projection ranks pass.
Generated parameters are used only for diagnostics after local fitting.

## Budget

One node, **one H100**, 16 CPU cores, **three hours maximum (3 GPU-hours)**.
No arrays, follow-up jobs or population jobs. This allocation is additional to
any existing jobs; the launcher neither cancels nor changes those jobs.

Default work: 16,384 parameter-to-flux evaluations with gradients and 24,576
forward evaluation draws, plus bounded simulation/contract overhead. Hard cap:
45,000 decoder evaluations and 9,900 seconds inside the process. Gradient and
forward evaluations are recorded separately; they do not have equal cost.
The first observed/simulated pair measures runtime. Stop if its extrapolated
remaining cost, with a 1.25 safety factor, exceeds the remaining allocation.
Slurm provides the hard time limit, including an unexpectedly long compilation.

## Launch on Jean-Zay

The balanced environment must refer to the completed v2 B arm and its frozen
simulation cache. No need to wait for or rerun the large K1024 validations.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark

export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
unset DIAGNOSTIC_ROOT
unset LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env
```

The new root defaults to `frozen_parent_local_vi_diagnostic_v1` beside the
balanced run. Existing roots are never overwritten or implicitly resumed.
On a failure, inspect its receipt/log first; do not rerun the entire balanced
workflow or delete historical artifacts. A smaller diagnostic is possible with
`LOCAL_VI_OBJECTS=2 LOCAL_VI_STEPS=8 LOCAL_VI_DRAWS=32` and a distinct root.

## Monitor and results

```bash
cd "$WORK/dsps-popcosmos"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
tail -n 80 "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.err"
python -m json.tool "$DIAGNOSTIC_ROOT/FINAL.json"
```

- `CONTRACT_AUDIT.json`: executed contract and analytical-control results.
- `PROGRESS.json`, `COST_PREFLIGHT.json`: case/start/step and measured costs.
- `paired_comparison.csv` and `.png`: per-case support, residuals and evidence
  stability, with both local starts displayed. ESS is for the pooled two banks,
  K=256 by default; evidence stability compares the two separate K=128 banks.
- `cases/*/amortized/` and `cases/*/start_*/`: full latent joint draws and log
  densities, calibrated model fluxes, summaries, local parameters and optimizer
  histories. No conversion to pointwise posterior targets.
- `SIMULATED_INPUTS.npz`: generated parameters and noisy observations. Simulated
  summaries include data-dependent ranks and generating-flux residual references.
- `FINAL.json`: `DIAGNOSTIC_COMPLETE` or `BUDGET_STOP`; neither means posterior
  validation. Exceptions produce `FAILED.json`; hard Slurm kills may only leave
  partial artifacts and scheduler/log evidence. All promotion flags stay false.

## Interpretation after readback

1. Contract failure: repair that inconsistency before any new training.
2. Strong local gains on simulations and observations, stable across starts:
   evidence of amortization/optimization limitations; design a bounded amortized
   adaptation next, checking mode retention. Not proof of full posterior quality.
3. Local gains on simulations but not observations: investigate observed-context
   coverage and population/likelihood adequacy before expanding architecture.
4. Failure even on simulations: inspect gradients, optimization trajectories and
   representation conditioning; a longer global NPE run is not yet justified.
5. Lower photometric residuals but unstable weights or start disagreement:
   insufficient inference evidence; do not resume population learning.

Local tests replace DSPS with analytical flux maps where stated. They verify
software behavior, not FENIKS physics or H100 timing. No real SED run is claimed
until its cluster receipts have been inspected.
