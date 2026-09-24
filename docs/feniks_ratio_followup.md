# FENIKS ratio-estimator decision run

This diagnostic reuses the completed ratio-ladder reference and target banks.
It performs no new DSPS simulation and does not modify a production parent.

The four H100 cells are the Cartesian product of noiseless/noisy photometry and
the continued/or independent classifier seed. Every cell uses the same
reference train, validation, calibration and audit rows and the same target
fit/held-out rows. The calibration set is disjoint from training, validation,
audit and target data.

For logits `l_ij`, calibration fits only component intercepts `b_j` by the
convex objective

```text
mean_i logsumexp_j(l_ij + b_j) - sum_j c_j b_j + lambda ||b||^2 / 2,
```

where `c_j` is the selected-reference component frequency. Its first-order
condition enforces `E_ref[C_j/c_j] = 1` on calibration simulations. Both raw
and calibrated ratios are evaluated on an independent reference audit split
and on the same synthetic target split. No target truth enters calibration or
classifier training.

The dependent report recomputes exact-theta and physical-coordinate oracles on
the common target split. It reports classifier convergence, ratio moments,
held-out mixture likelihood, simplex geometry, zero-weight components and
joint physical sliced Wasserstein distance. A passing synthetic result is only
permission to repair the real-catalogue decoder contract next; it is not a
production parent-population claim.
