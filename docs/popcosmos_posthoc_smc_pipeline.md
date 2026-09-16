# Pop-COSMOS post-hoc SMC and empirical-Bayes sequence

## Scientific question

The workflow separates two hypotheses without changing the 15D native FENIKS
forward model:

1. the amortized encoder does not cover the posterior target adequately;
2. after repairing that approximation, the learned population prior is still
   mismatched to the observed COSMOS population.

The first hypothesis must be tested before the second. A prior update based on
collapsed importance weights would fit proposal error rather than a population
distribution.

## Phase A: exact-target-density adaptive SMC pilot

For each galaxy, particles start from the frozen amortized joint proposal
`q_psi(x | y)`. Adaptive SMC bridges

```text
pi_beta(x | y) proportional to
    q_psi(x | y)^(1-beta)
    [p_phi(x) p(y | x)]^beta
```

from `beta=0` to `beta=1`. The target is evaluated in the same unconstrained
15D network coordinates as the exact learned prior. Adaptive temperature
increments target a fixed ESS, multinomial resampling prevents a single
particle from carrying an entire intermediate distribution, and MALA moves
rejuvenate the joint particles after resampling.

The pilot compares the existing Student-t2 likelihood with 0%, 2%, and 5%
relative photometric-error floors. It uses two independent SMC seeds per
variant and no spectroscopic redshift. Selection requires SMC support,
cross-seed evidence stability, and posterior-predictive photometric adequacy.
The six variant/seed experiments are sharded by objects so that more Jean-Zay
H100s reduce wall time without changing a galaxy's Markov chain.

All SMC, proposal-refresh, likelihood-selection, and ordinary-IS probe objects
are sampled from two disjoint subsets of the frozen 4000-object validation
pool. They were not used by the original wake updates and are disjoint from
the 5000-object spectroscopic evaluation cohort.

## Phase B: frozen-prior encoder refresh

If one likelihood passes, its two weighted joint SMC banks are combined as an
equally weighted mixture of replicates. Only the encoder is optimized using

```text
L(psi) = - mean_i sum_k w_ik log q_psi(x_ik | y_i).
```

The learned prior is checked leaf by leaf and must remain bitwise unchanged.
Early selection uses weighted NLL on held-out SMC objects. This is a proposal
repair, not an empirical-Bayes prior update.

The refreshed encoder is then tested with ordinary K=2048 importance sampling
on a second, disjoint set of validation rows. EM remains blocked unless both the
refresh validation and the ordinary-IS support gate pass.

## Phase C: empirical Bayes, intentionally deferred

After the preceding gates pass, a Monte Carlo EM iteration can update the
population prior:

```text
E step: exact-target-density SMC under the current prior
M step: maximize the stopped weighted expectation of log p_phi(x)
```

The M-step must use weighted joint particles, a trust penalty toward the
previous prior, and held-out evidence. After every accepted prior update, the
encoder must be refreshed and the E-step rerun under the new prior. The older
fixed-proposal generalized-EM workflow remains a diagnostic implementation;
it must not be launched on the previously collapsed Pop-COSMOS importance
banks and is not wired into this SMC pilot.

## Posterior contract

Every posterior artifact is a dense joint draw bank or a weighted empirical
joint distribution. Posterior medians are not used for SMC, encoder refresh,
likelihood selection, calibration, or a future prior M-step. A redshift median
is allowed only as the displayed point estimator in truth-versus-inferred
redshift figures.
