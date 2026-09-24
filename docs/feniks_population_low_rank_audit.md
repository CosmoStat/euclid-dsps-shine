# Smooth low-rank population correction audit

This audit follows the identifiable population-basis experiment. It reuses the
same immutable simulation banks, target split, frozen epoch-600 classifiers,
calibration and 128 physical components. It does not run DSPS, train a neural
network, use posterior samples, or modify a production prior.

## Parameterization

Let `c` be the selected-reference component frequencies. A truth-free graph is
built from the existing joint physical and frozen-classifier response
embedding. Its lowest-frequency Laplacian eigenvectors form the columns of
`Phi`; an explicit first contrast lets the broad tail component move. Every
mode is centered and standardized under `c`.

For each tested rank, selected-population weights are

`v = c * (1 + Phi gamma)`,

with explicit constraints `v_j >= epsilon`. Since `c^T Phi = 0`, the weights
sum to one for every feasible `gamma`. The selected-mixture log likelihood is
concave in `v` and `v` is affine in `gamma`, so minimizing its negative plus a
quadratic coefficient penalty remains a convex problem.

The parent is reconstructed only after the fit:

`u_j = (v_j / alpha_j) / sum_l (v_l / alpha_l)`.

No inverse-selection factor enters an individual posterior.

## Selection and gates

Ranks 8, 16, 24, 32, 48 and 64 and a short penalty path are tested in one
controlled experiment. Candidate selection uses paired held-out target
likelihood followed by minimum median bootstrap parent-density instability.
Only candidates inside the paired held-out one-standard-error set are
bootstrapped; candidates already rejected by held-out data cannot be selected
and are not recomputed 32 unnecessary times. Each bootstrap starts from its
full-catalogue optimum. Known synthetic truth is used only after selection for
closure evaluation.

Both noiseless and noisy arms must pass:

- paired held-out one-standard-error admissibility;
- certified convex-solver KKT tolerance and normalized selected/parent weights;
- nondegenerate parent weights;
- median bootstrap physical sliced-Wasserstein at most 0.015;
- physical truth closure within 0.02 of the component-family oracle;
- rank and penalty paths not selected at their upper boundary.

A complete pass authorizes decoder-contract repair followed by one end-to-end
parent fit. It does not by itself promote a scientific parent prior.
