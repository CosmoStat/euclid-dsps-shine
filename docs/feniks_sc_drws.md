# FENIKS Selection-Corrected Defensive RWS

> We infer the parent distribution within the predefined FENIKS refinement
> and catalogue-support domain, while explicitly correcting the additional
> observed r<27.5 selection.

The upstream true-space filtering is conditioning information (`C0`), not a
selection that this workflow attempts to invert. Training, validation,
checkpoint selection, importance weighting, and prior learning read observed
flux, observed flux error, and mask only. Truth is permitted only in a
separate closure after the final checkpoints have been frozen.

## Object posterior and proposal

Phase A uses Student-t2 only to warm up a mass-covering proposal while the
identity-initialized population prior is frozen. Phase B uses the Gaussian
scientific target everywhere:

```text
log target_i(x) = log p_Gaussian(y_i | x, fluxerr_i) + log p_eta(x | C0)
```

The Phase-B first-pass deterministic mixture is

```text
r_128(x | y) = 0.50 q_T1 + 0.30 q_T1.5 + 0.15 q_T2.5 + 0.05 p_eta.
```

Every draw is divided by this complete mixture density. An object with
`ESS/K < 0.05` or `max weight > 0.90` receives 384 additional draws from
`0.45 q_T1.5 + 0.45 q_T2.5 + 0.10 p_eta`. The 512 draws are then rescored
under the deterministic multiple-importance mixture formed from the actual
integer component counts of both sampling rounds. Unresolved objects train q
with tempered stopped weights but never enter a prior update.

## Losses

For q, with `tau` annealed from 0.5 to exactly 1 over 80 wake updates,

```text
w_q,ik = normalize(w_exact,ik ** tau)
J_q,wake = -mean_i sum_k stop(w_q,ik) log q_psi(x_ik | y_i).
```

Sleep replay remains three epochs per wake epoch. The log-std floor anneals
from -1.5 to -4 over the first 100 total epochs and the flow scale clamp from
0.15 to the architecture value over 60 epochs. A one-sided entropy penalty is
active only during the first 75% of Phase B and is exactly zero thereafter.

For the single parent flow,

```text
J_prior = -mean_i sum_k stop(w_exact,ik) log p_eta(x_ik | C0)
          + log(alpha_eta)
          + lambda_trust KL(p_old || p_eta),

beta(x) = P(m_r,observed < 27.5 | x),
alpha_eta = E_p_eta[beta(x)].
```

`log(alpha_eta)` uses the existing score-function gradient. Neither `beta`
nor `alpha` occurs in an object's posterior weights. The selected population
is derived from, rather than fitted independently of, the parent:

```text
p_eta(x | A=1,C0) = beta(x) p_eta(x | C0) / alpha_eta.
```

## Promotion and artifacts

Promotion is fail closed: four 512-object pilot tasks (two architectures by
two seeds), two independent 2000-object confirmations for the selected
architecture, then two full-selected-catalogue seeds. Raw q and EMA q are
evaluated independently with ordinary Gaussian IW K=2048, PSIS diagnostics,
dense PPC, full entropy, finite selection gradients, and the learned prior.
The full run is not submitted by the pilot launcher.

The final report consumes joint draws, not marginal point summaries, and must
contain raw/EMA support, hard expansion rates, parent and beta-weighted prior
draws, population marginals/correlations, photometric PPC, runtime/DSPS cost,
the C0 statement, and the no-truth receipts. MIRA, TARP, PIT, coverage, truth
population recovery, and the optional 4-8 galaxy NUTS comparison are strictly
post-freeze closure products.
