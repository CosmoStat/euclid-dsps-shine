# FENIKS self-supervised learned-prior Student-t2 array

This array trains only learned-prior models. Truth parameters are excluded from
every training proposal and loss; they are read after training for synthetic
closure diagnostics.

All candidates share:

- the immutable spline-15D `mixed_log_shifted_asinh` coordinates;
- an identity-initialized eight-layer population RealNVP `p(x)`;
- an independent conditional RealNVP posterior `q(x | flux, flux_err)`;
- three model-generated sleep epochs followed by one real-catalog wake epoch;
- a Student-t likelihood with two degrees of freedom, zero extra error floor,
  and Student-t2 noise in model-generated sleep;
- held-out inference on 5000 objects, posterior-predictive flux diagnostics,
  and 16384 learned-prior samples.

The two importance-RWS candidates run for 120 epochs. The SMC-Wake candidate is
bounded at 40 epochs because its four-particle tempered MALA wake step costs
about twice the complete wall-clock budget of either 120-epoch importance run.
It is therefore a compute-bounded diagnostic candidate, not an epoch-matched
comparison.

## Candidates

`selfsup_rws_k8_t2` is the controlled baseline. Its posterior has one diagonal
Gaussian base and wake uses the existing tempered mixture proposal with eight
importance particles.

`selfsup_rws_mix2_k8_t2` changes only the posterior base to an exact
two-component diagonal Gaussian mixture. A shared conditional RealNVP maps both
components into latent space, and `log q` uses the exact mixture density.

`selfsup_smcwake_mix2_k4_t2` retains the two-component posterior and replaces
the circular posterior proposal during wake with four prior particles,
likelihood tempering at `[0, 0.33, 0.67, 1]`, resampling, and one MALA move per
intermediate temperature. SMC particles and weights are stopped before the
posterior and population-prior updates. The production recovery of epoch 40
from the rolling epoch-39 checkpoint is a warm restart: model weights are
restored but AdamW state is reinitialized, and online validation is disabled to
avoid the observed host-pinned-memory allocator failure.

## Required diagnostics

The run is not selected from reconstruction chi-square alone. Each candidate
must provide:

- photometric fits, residual tails, full posterior/truth/prior corners, PIT and
  coverage plots;
- wake ESS, invalid-particle fraction and learned-prior M-step loss;
- mixture weight entropy and maximum component occupancy where applicable;
- SMC stage ESS and MALA acceptance for the SMC candidate;
- normalized marginal distances for all 15 native latent parameters and the
  absolute error of all 105 pairwise Spearman correlations;
- Physical Jacobian Lens spectra and loadings for 512 stratified held-out
  objects, including decoder, autoencoder and learned-prior score directions.

Training uses three concurrent four-H100 tasks. Jacobian diagnostics use four
one-H100 shards per candidate, also with a peak allocation of 12 H100. The
trainer is single-host JAX `pmap`; requesting GPUs on a second node for one task
would not accelerate it.
