# Geometry v1: what the data say

Source: `frozen_geometry_nuts_v1`, job 2002093, completed in 10m31s.
The downloaded 28 JSON/CSV/PNG artifacts match the completion inventory hashes.
The 24 binary artifacts were not transferred, so this analysis uses saved
audits and density slices, not a new analysis of complete sample banks.

## The network is missing good photometric regions

| Galaxy | ESS, bank 0 / bank 1 (4096 draws each) | Largest weight | Plain-language interpretation |
|---|---|---|---|
| observed_000 | 63.81 / 52.12 | 6.57% / 8.45% | More than one useful draw, but considerable waste |
| observed_005 | 1.10 / 1.82 | 95.11% / 66.81% | Almost all the information is carried by one or two draws |
| simulated_003 | 351.82 / 169.67 | 1.87% / 5.16% | Best overlap of these four cases, still not perfect |
| simulated_004 | 2.07 / 2.77 | 64.88% / 53.71% | Severe concentration in both independent banks |

For observed_005, bank 0's dominant and ordinary control draws have log
likelihoods 1228.11 and -73.69. Their proposal log densities are very similar:
-18.70 and -19.03. Thus this contrast is overwhelmingly about photometric fit,
not a tiny numerical error in the weight denominator. The second bank also
has a worse ordinary control draw, but a smaller likelihood gap (~151).

For simulated_004, the corresponding likelihood gaps are ~1882 and ~2375.
The network can produce good solutions, but it also produces extremely poor
ones. Two controls per galaxy do not quantify the fraction of poor draws.

## It is not a tiny isolated point in every direction

At the dominant observed_005 center, stepping by 0.01 network standard
deviations changes log target by between -0.307 and +0.225 over the 38 signed
directions. At 0.1 the range is -10.21 to +1.62; at 1 it is -799.08 to +2.54.
Some directions are tightly constrained, while others still allow movement.
The likelihood drives many steep drops; the prior contributes to the shape,
especially in star-formation-history coordinates.

The curves therefore support an anisotropic mismatch between proposal and
target. They do NOT measure posterior volume, prove that all modes were found,
or establish a globally narrow posterior. Oblique/coordinate slices can miss
curved correlated regions. The logarithmic horizontal scale also makes the
central region look visually broad; inspect the numerical distances.

## Largest weight does not mean highest target density

For simulated_003 bank 1, the dominant draw has log target 1227.53, whereas
the ordinary control has 1231.86 (higher). But their proposal log densities
are -25.53 and -15.66 respectively. The dominant draw wins on the target /
proposal ratio, not on target density alone. This is expected importance
weight behaviour, not evidence of an arithmetic error.

## Numerical checks and the prior

The four saved audits report forward/inverse proposal agreement and target
decomposition agreement. Directional derivatives approach the automatic
derivative at smaller steps in the inspected oblique direction. Some smallest
step initial-prior differences show a numerical floor. This is useful local
evidence, not a proof of every gradient everywhere or of historical MIS code.

Replacing the learned prior by the initial prior changes several SFH cuts.
Neither curve tells us which prior is correct. A/B NUTS keeps the target fixed
and changes initialization. A/C changes the prior with the same initial points.
Only compare convergence across chains for a common target; do not pool C
with A/B into one posterior.

## Next calculation

Run the corrected float64 NUTS comparison on these same inputs. This is now a
justified reference calculation, not another optimizer tuning experiment.
Require diagnostics and A/B agreement before interpreting posterior shapes.
Failure to mix is itself a result; do not discard inconvenient chains.

See [launch and restart instructions](feniks_geometry_nuts_runbook.md).

## Depth-4 NUTS result and importance-weight interpretation

The completed 12-task depth-4 run is computationally complete but not a
posterior reference: 384,337 of 393,216 retained transitions (97.74%) reached
the maximum 15 integration steps. Every group failed the complete convergence
gate. This directly motivates a dense-mass depth-5/6 comparison and a longer
depth-6 run; it does not justify discarding chains or trusting the smoothest
corner.

The weight formula and stable normalization passed the frozen geometry audits.
That is different from saying the finite importance estimate is usable. In
`simulated_003`, where the proposal overlaps the target reasonably well,
weighting moves the AVI marginals toward the provisional NUTS marginals. In
`observed_005` and `simulated_004`, one to three effective draws create narrow
orange spikes and destroy distributional information. Operationally, weighting
fails in those cases because the proposal supplies too little target support,
even though the arithmetic can be correct.

The initial prior used by group C is the configured identity-initialized
15-dimensional RealNVP. Its density is checked to be exactly a standard normal
in normalized latent `x`: `x ~ N(0, I)`. It is not uniform in physical galaxy
parameters because the bounded mixed coordinate transform and its Jacobian are
retained. The learned prior has the same flow architecture after population
training; neither prior is certified as the true population distribution here.
