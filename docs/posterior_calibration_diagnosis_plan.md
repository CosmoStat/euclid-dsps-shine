# Posterior calibration diagnosis plan

## Non-negotiable posterior contract

A vector of posterior medians is never a posterior and must never be used as a
surrogate for a posterior distribution.

This forbids using marginal medians for posterior aggregation, MIRA, TARP,
PIT, coverage, posterior predictive checks, population comparisons,
correlations, or forward decoding of a supposedly representative joint
parameter vector. All such analyses must consume weighted empirical
distributions, dense joint draws, or an explicit normalized density.

The only allowed exception is a `z_true` versus inferred-redshift point plot
and its point-estimate photo-z metrics. In that plot, the posterior median may
be used as the displayed redshift estimator, but the dense posterior draws and
intervals must still be preserved as artifacts. This exception does not turn
the median into a posterior.

## Goal

Diagnose the observed redshift under-calibration by separating three possible
causes while holding the evaluation cohort fixed:

1. amortized-posterior inference error;
2. an incorrectly learned population prior;
3. forward-model or likelihood misspecification.

These causes are not mutually exclusive. Each experiment must change one
component at a time and preserve object IDs, train/evaluation splits, feature
statistics, filters, likelihood settings, checkpoint hashes, and posterior
draw identities.

Run the complete diagnosis on both:

- the FENIKS synthetic held-out data, where all 15 latent truths are known and
  the data-generating forward model is controlled;
- the matched Pop-COSMOS photometry, using spectroscopic truth only for
  redshift evaluation and never for population-prior training.

## Experiment 1: importance-correct the amortized posterior

Freeze the forward model, likelihood, learned prior, and encoder. For samples
`theta_k ~ q_psi(theta | x)`, compute

```text
log w_k = log p(x | theta_k) + log p_phi(theta_k)
          - log q_psi(theta_k | x).
```

Use stable log normalization and report both raw importance sampling and PSIS.
The required artifacts are the joint samples, `loglike`, `logprior`, `logq`,
normalized weights, raw ESS, maximum-weight fraction, PSIS Pareto-k, and
terminal manifests. Evaluate increasing proposal budgets, initially 128, 512,
2,048, and 8,192 draws on a fixed diagnostic subset before choosing a catalog
budget.

Compute weighted redshift PIT, central coverage, interval widths, photo-z
metrics, and posterior predictive checks directly from the weighted empirical
distribution. For tools that require unweighted draws, use seeded stratified
resampling and retain both the original weighted bank and resampled draws.

Decision rule:

- if IW/PSIS improves calibration and predictive checks while ESS is adequate
  and Pareto-k is acceptable, amortization error is implicated;
- if weights collapse, the test is inconclusive because the proposal may miss
  modes or target support; importance weighting cannot repair mode seeking
  when no samples cover the missing modes;
- difficult or inconclusive objects must be checked with adjusted MCLMC, SMC,
  or another target-exact reference under the same prior and forward model.

## Experiment 2: post-hoc empirical-Bayes prior refinement

Start from the current learned prior `p_phi_0`. A valid EM-like update over
observed photometry is:

```text
E step:
    approximate p_phi_t(theta | x_i)
    proportional to p(x_i | theta) p_phi_t(theta)

M step:
    phi_(t+1) = argmax_phi sum_i
        E_{p_phi_t(theta | x_i)}[log p_phi(theta)].
```

Implement the E step using the encoder as a proposal and the importance
weights from Experiment 1. Implement the M step as weighted maximum likelihood
for the exact-density flow prior, with a trust-region or KL penalty toward the
previous prior, validation-based early stopping, and immutable
training/evaluation membership. This is empirical Bayes for the selected
catalog unless an explicit selection correction is added.

Training `p_phi_1` directly on unweighted draws from `q_0(theta | x)` is not a
valid EM step: it reproduces amortizer bias and ignores the likelihood/prior
correction. Generating simulations from `p_phi_0` and fitting the same family
back to them is also a fixed-point identity, not data-driven prior learning.

Two implementations should be tested:

1. cheap post-hoc generalized EM: keep `q_psi` fixed as a proposal, recompute
   importance weights under every updated prior, then refit the prior;
2. robust alternating refinement: after each weighted prior update, fine-tune
   or refresh the amortized proposal under the new prior before the next E
   step.

Track held-out importance-weighted evidence estimates, ESS/Pareto-k, prior
geometry, posterior calibration, posterior predictive checks, and changes in
astrophysical marginals. Stop if held-out evidence or calibration degrades, or
if proposal ESS collapses. A deliberately broad prior is only a final stress
test, not the default initialization.

This refinement is possible after the current training because the encoder,
exact prior density, and DSPS likelihood are available. The stored 128-draw
banks are useful for triage but may be too small for stable EM; larger proposal
banks should be generated from the existing checkpoint without retraining the
forward model.

## Experiment 3: forward-model and likelihood adequacy

Freeze inference as tightly as possible using high-budget IW where reliable
and target-exact chains on a small fixed cohort. Then test whether any latent
configuration can reproduce the photometry.

Required diagnostics:

- likelihood-only and prior-regularized multi-start MAP residuals;
- adjusted MCLMC or SMC posterior predictive residuals;
- reduced chi-square and Student-t residual tails by band, redshift, S/N, and
  magnitude;
- posterior predictive coverage for fluxes and colors;
- comparisons of 26 bands, 24 bands without IRAC, and targeted band-drop
  ablations;
- controlled nuisance tests for zero points, error floors, calibration terms,
  Student-t scale/degrees of freedom, and selection effects.

On synthetic data, first require exact self-closure, then inject known band
offsets, noise misspecification, and missing forward components to verify that
the diagnostics recover a forward-model failure. On Pop-COSMOS, persistent
structured residuals or poor best-fit likelihood after inference and prior
checks implicate the forward model or noise model. A flexible prior can absorb
forward-model error, so improved calibration after prior refinement alone does
not prove that the prior is physically correct.

## Controlled execution matrix

For each synthetic and Pop-COSMOS cohort, preserve one frozen baseline and
produce these rows:

| Row | Prior | Posterior | Forward model | Purpose |
| --- | --- | --- | --- | --- |
| A | current | raw amortized `q` | current | baseline |
| B | current | IW/PSIS-corrected `q` | current | inference error |
| C | current | target-exact subset | current | IW/support reference |
| D | EM-refined | IW-corrected/refreshed `q` | current | prior error |
| E | broad stress prior | target-exact subset | current | prior sensitivity only |
| F | fixed prior | target-exact subset | nuisance/ablated variants | forward-model error |

Promotion gates:

1. run A/B on synthetic held-out data and a small fixed Pop-COSMOS subset;
2. require acceptable IW diagnostics before catalog-scale weighted claims;
3. run C on representative reliable, borderline, and failed-IW objects;
4. run two to five EM iterations on synthetic data before Pop-COSMOS;
5. keep the spectroscopic evaluation cohort completely outside prior updates;
6. report redshift calibration on Pop-COSMOS and full 15D calibration only on
   synthetic truth;
7. retain every dense draw, weight, chain diagnostic, bootstrap replicate,
   input hash, decision gate, and completion marker.

## Interpretation table

| Observation | Supported interpretation |
| --- | --- |
| IW fixes calibration with healthy ESS/Pareto-k | amortized inference error |
| IW collapses but target-exact chains calibrate | proposal support or mode-seeking failure |
| EM improves held-out calibration/evidence after reliable E steps | population-prior error contributes |
| Target-exact inference still gives structured photometric residuals | forward/likelihood misspecification contributes |
| Synthetic passes but Pop-COSMOS fails after exact inference | real-data forward/noise/selection mismatch is likely |
| Several interventions improve different metrics | multiple failure sources coexist |

## Implemented workflows

The current implementation covers Experiments 1 and 2 only. Experiment 3 is
deliberately not launched by these wrappers.

### Importance correction

`scripts/importance_correct_posterior.py` consumes an inference directory, a
`posterior_samples` shard directory, or one parquet file. It writes:

- the original joint draws with raw and PSIS weights;
- a seeded joint PSIS-resampled bank for MIRA/TARP or plotting code that cannot
  consume weights directly;
- per-object ESS, maximum-weight, Pareto-k, and IS evidence diagnostics;
- weighted redshift PIT, coverage, widths, and photo-z metrics when truth is
  supplied;
- input hashes, a summary, and a terminal `DONE` marker.

The optional `--config --target-checkpoint` pair evaluates a different learned
prior on the cached proposal draws. Without those arguments, the stored
`logprior` is used and the correction isolates amortized-inference error under
the source prior.

The Jean-Zay array `scripts/submit_posthoc_importance_probes.sh` runs both the
synthetic FENIKS test set and held-out COSMOS cohort at configurable proposal
budgets. Its default matrix is `K={128,512,2048}` on 256 fixed objects.

### Generalized EM with proposal refresh

`scripts/train_posthoc_empirical_bayes_prior.py` recomputes the exact current
prior density at every E-step, freezes the per-object self-normalized weights
during each M-step, and updates only the population flow. A cross-entropy term
under draws from the preceding prior implements a `KL(old || new)` trust
penalty up to a constant. Each iteration writes its own checkpoint, E-step
diagnostics, held-out IS evidence estimate, and support gate.

Within one call the proposal bank is fixed. The Jean-Zay wrapper calls a
single E/M update at a time and regenerates the proposal bank between calls,
so it implements the alternating proposal-refresh variant.

The default support gate refuses an update when the median raw ESS fraction is
below 0.01 or more than half of objects have Pareto-k above 0.7. The
`--allow-low-ess` option exists only for technical diagnostics; a result made
under that override must not be promoted as an empirical-Bayes result.

`scripts/submit_posthoc_empirical_bayes.sh` runs one FENIKS and one COSMOS task.
Each outer iteration rebuilds a proposal bank only on frozen training indices,
performs one generalized-EM prior update, and then uses that checkpoint to
refresh the next proposal. After the last iteration it performs controlled
source-prior versus updated-prior inference on the same evaluation indices and
writes MIRA/TARP comparisons from the joint PSIS-resampled distributions. The
encoder is preserved, but because this posterior family is expressed in
learned-prior base coordinates, regenerating a bank with the updated checkpoint
refreshes the transported proposal and its exact `logq`.

### Execution order

Run the importance-budget matrix first. Promote generalized EM only when a
budget has adequate support diagnostics. Start EM with small training and
evaluation limits, then increase them in a new output root; never overwrite a
failed or completed root. Spectroscopic COSMOS truth is read only after
inference and is never passed to the prior M-step.
