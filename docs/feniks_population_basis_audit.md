# Identifiable hierarchical population basis

This audit follows the completed frozen 128-weight inversion. It does not run
DSPS, retrain a classifier, use individual posterior samples, or modify a
production prior.

## Why the hierarchy is needed

The 128 overlapping physical components can represent the known synthetic
parent, but their individual weights are more detailed than the photometry can
stably identify. Global KL shrinkage stabilizes those weights only by erasing
real population structure. The hierarchy instead fixes reference-conditional
mixtures of components and learns fewer group masses.

For a group `G`, calibrated classifier probabilities and selected-reference
frequencies are marginalized exactly:

`C_G(x) = sum_(j in G) C_j(x)`

`c_G = sum_(j in G) c_j`.

Consequently `C_G/c_G` is the density ratio for the grouped selected-reference
component. Its selection probability is the reference-parent-weighted mean of
the original `alpha_j`. Parent masses still follow the same explicit `v/alpha`
correction and sum to one.

## Truth-free construction and selection

One shared nested hierarchy is constructed from:

- normalized joint 5D component centers and widths;
- Hellinger-embedded confusion profiles from both frozen photometric
  classifiers on disjoint reference simulations.

The broad tail component remains a singleton. Known target-parent weights are
not inputs to clustering. Candidate resolutions are 32, 48, 64, 96 and 128.
For each resolution, a short KL path is evaluated. Candidate selection uses
only paired held-out target likelihood and bootstrap parent-density stability.
Known truth is read only for final closure metrics.

## Decision

The population-representation block passes only when both noiseless and noisy
arms satisfy held-out one-SE admissibility, median bootstrap physical
sliced-Wasserstein at most 0.015, and physical truth closure within 0.02 of the
precomputed component-family oracle. Passing authorizes decoder-contract repair,
not immediate scientific promotion. Failure routes to a smooth low-rank
population correction rather than another global-shrinkage scan.
