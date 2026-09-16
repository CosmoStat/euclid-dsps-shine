# FENIKS Sleep-NPE / Defensive-Wake Analysis Handoff

## Repository and immutable run

- Repository: `CosmoStat/euclid-dsps-shine`
- Branch: `feature/feniks-exact-posterior-benchmark`
- Code commit used after recovery fixes: `5a56f94`
- Production job: `1247657`
- Production root:
  `outputs/runs/feniks_parentprior_sleepnpe_recovery2_20260821_150802`
- Training root:
  `outputs/runs/feniks_parentprior_sleepnpe_recovery2_20260821_150802/train`
- Warm restart checkpoint:
  `outputs/runs/feniks_parentprior_sleepnpe_20260821_111314/train/checkpoints/epoch_0024.eqx`
- Restart epoch: 25; model parameters and feature statistics were restored,
  optimizer state was intentionally reinitialized.

## Scientific objective

The intended factorization was:

1. Train the amortized posterior `q_psi(x | f_obs, sigma_obs)` only with
   inclusive sleep/NPE.
2. Freeze `q` during wake.
3. Use a defensive importance proposal during wake to update only the learned
   parent population prior `p_eta(x)`.
4. Correct the selected-catalog prior loss with `+log(alpha_eta)` while keeping
   normalized per-object importance weights unchanged.

The latent is the existing 15-dimensional FENIKS spline latent. DSPS,
PhotoErr, passbands and zero-points are fixed. No FENIKS truth is used for
training or checkpoint selection.

## Sleep objective

For each observed catalog error vector `sigma_i`:

```text
x ~ p_eta(x)
f_model = DSPS(x)
f_sim = f_model + likelihood_noise(sigma_i)
retain f_sim only when its noisy observed lsst_r flux passes r_obs < 25
minimize -log q_psi(x | f_sim, sigma_i)
```

This is intended to minimize the inclusive direction

```text
E_(f,sigma) KL[p_eta(x | f,sigma) || q_psi(x | f,sigma)].
```

The training manifest contains 5,526 distinct observed-selected rows per
epoch. Pmap pads these to 5,632 rows (22 batches of 256) with 106 duplicate
rows. Observed rows provide photometric error covariates; physical truth labels
are not used.

## Wake target and proposal

The canonical target is

```text
log target(x) = log p(f_obs | x, sigma_obs) + log p_eta(x),
```

with the same physical support mask in wake, IS, MAP, NUTS and MCLMC. The
working likelihood is Student-t with 2 degrees of freedom.

The 32-particle defensive proposal is

```text
r(x | y) = 0.50 q_T=1 + 0.25 q_T=2 + 0.15 q_T=4 + 0.10 p_eta.
```

Every sampled particle is scored under the complete mixture density. The
normalized weights remain exactly

```text
softmax(loglike + logprior - logproposal).
```

Neither `beta(x)` nor `log(alpha_eta)` is added to these weights.

For a supported wake batch, the parent-prior objective is

```text
J_eta = -mean_i sum_k stop_gradient(w_ik) log p_eta(x_ik)
        + log(alpha_eta),
```

where

```text
beta(x) = Phi((model_flux_r(x) - flux_limit) / sigma_r(x))
alpha_eta = E_(x~p_eta)[beta(x)].
```

`alpha_eta` uses Gaussian m5/PhotoErr survey noise, 1,024 fixed common-random
number prior draws in chunks of 256. A real prior-to-DSPS-to-PhotoErr gradient
preflight passed with 64 draws and gradient norm `0.918089`.

## Support gates

A prior update is rejected when proposal support is inadequate. Relevant
thresholds include:

- median `ESS/K >= 0.10`;
- per-object eligibility requires `ESS/K >= 0.10` and maximum weight `<= 0.80`;
- batch failure also uses the configured fraction of maximum weights above
  0.8.

The gate is intentionally not relaxed merely to force a prior update.

## Observed remote results

### Sleep validation

- Epoch 24 validation sleep NLL: `7.71941`
- Around epoch 33 best validation sleep NLL: `6.47187`
- Epoch 76 validation sleep NLL: `2.65908`
- Epoch 80 validation sleep NLL: `2.51289`

The synthetic held-out sleep task improved substantially.

### Defensive wake support

Aggregated training-log diagnostics:

| Epoch | Median ESS/K | Fraction max weight > 0.8 | Eligible fraction | Updates |
|---:|---:|---:|---:|---:|
| 25 | 0.031250 | 0.944602 | 0.0 | 0/22 |
| 33 | 0.031250 | 0.943892 | 0.0 | 0/22 |
| 41 | 0.031251 | 0.940874 | 0.0 | 0/22 |

`0.03125 = 1/32`, the theoretical single-effective-particle floor. Logs from
the final scheduled wake at epoch 73 also show every inspected batch rejected
by the support gate with finite gradients. No prior update was applied during
the run. Consequently, `+log(alpha_eta)` was never part of an optimizer update
and this run did not learn a selection-corrected parent prior.

### Engineering failures already fixed

1. The first run computed the selection branch before masking rejected wake
   updates, permitting `0 * NaN` gradients. The differentiable prior and alpha
   computations now live inside the accepted branch of `jax.lax.cond`.
2. The first recovery smoke exposed a float32/float64 branch mismatch under
   x64 pmap. All branch outputs are now cast to the wake-loss dtype.
3. The recovered run has finite gradients. Its failure is proposal support,
   not a NaN or compilation failure.
4. Local validation after the fixes: ruff, compileall, shell checks and 100
   posterior/selection/workflow tests passed, including x64 plus pmap.

## What is established

1. The current defensive wake proposal is unusable for prior learning on the
   observed FENIKS rows: nearly every object is dominated by one particle.
2. The support gate correctly prevents a scientifically invalid prior update.
3. Optimizing held-out sleep NLL alone does not establish that `q` covers the
   observed-data posterior.
4. More epochs of the identical objective are not justified by epochs 25 to 73:
   sleep NLL improved strongly while wake ESS remained at the floor.
5. The learned parent prior was not updated in this experiment.

## What is not yet established

Do not yet state that the raw amortized posterior is globally useless. The
available wake statistic measures the 32-particle defensive mixture on real
observations. An exact held-out comparison of the final best `q` against NUTS
has not yet been completed.

The current evidence does not distinguish among:

1. a too-narrow or geometrically misaligned `q`;
2. a mismatch between the selected synthetic sleep joint and real catalog
   observations;
3. a likelihood/feature/error-scale mismatch despite the canonical target;
4. localized failure near the magnitude cut or at low SNR;
5. a proposal that is directionally useful but impossible to assess with only
   32 particles in 15 dimensions.

## Required next diagnostic

Use the final `best.eqx` checkpoint without claiming prior-training success.
Run an exact posterior benchmark on a stratified 20--32 galaxy cohort:

- typical/high-SNR objects;
- low-SNR objects;
- objects near `r_obs = 25`;
- objects with previous catastrophic IS;
- a range of redshift and error patterns.

For each object compare:

1. raw `q` draws;
2. q-only IS;
3. the full defensive-mixture IS;
4. MAP;
5. converged NUTS, optionally MCLMC;
6. FENIKS truth for closure diagnostics only.

Report:

- ESS fraction, maximum weight and PSIS Pareto-k;
- marginal biases and width ratios relative to NUTS;
- generalized covariance eigenvalue ratios solving
  `Sigma_NUTS v = lambda Sigma_q v`;
- TARP/MIRA/coverage from dense draws;
- fraction outside the canonical fit bounds;
- results stratified by SNR and distance to the selection threshold.

## Questions for analysis

Please analyze the evidence without proposing another blind architecture sweep.

1. How can held-out inclusive sleep NLL improve from 7.72 to 2.51 while
   observed-data defensive ESS remains exactly one effective particle?
2. Which diagnostics most efficiently separate posterior contraction from
   simulation-to-observation mismatch?
3. Is 32-particle defensive IS intrinsically non-diagnostic in this 15D
   setting, and how should that be tested without an expensive broad sweep?
4. Should parent-prior learning use the already validated direct weighted-SMC
   E-step rather than amortized wake IS?
5. After an SMC prior update, should `q` be refreshed with sleep, SMC posterior
   distillation, or a controlled combination?
6. What minimum exact-benchmark evidence would justify calling the final
   encoder a fast Bayesian posterior estimator rather than only a simulator
   inverse model?

## Guardrails

- Do not relax support thresholds merely to obtain nonzero prior updates.
- Do not add selection probabilities to normalized per-object weights.
- Do not interpret lower sleep NLL as exact-posterior calibration.
- Do not use truth in the production loss; truth is closure evaluation only.
- Do not rerun an architecture matrix before diagnosing the final checkpoint
  against the exact posterior.
- Keep parent-population and selected-catalog distributions distinct.
