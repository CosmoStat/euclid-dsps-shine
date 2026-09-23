# FENIKS ratio ladder

This diagnostic localizes parent-recovery error without using posterior samples.
It generates one uniform-component reference bank and one independent target bank
from the same full 15D simulator and noisy observed-r selection.

The four arms differ only in the information used to estimate component ratios:

1. `exact_theta`: analytic selected-component densities
   `log g_j(theta) - log alpha_j`;
2. `physical_a`: a classifier receiving the five normalized physical coordinates;
3. `noiseless_photometry`: a classifier receiving noiseless flux features, while
   selection remains defined by the same noisy observed flux;
4. `noisy_photometry`: a classifier receiving the production noisy features.

Every arm fits selected weights `v`, reconstructs normalized parent weights
`u proportional to v / alpha`, and is evaluated on independent parent and selected
draws. The report compares physical density closure, selected predictive closure,
observable closure, held-out likelihood and bootstrap stability. Component-weight
L1 is diagnostic only because overlapping components can be non-identifiable.

The observation stage also replaces the invalid zero-mean check for selected
`lsst_r` noise with a truncated-normal conditional PIT. Decoder residuals are
stratified by S/N, redshift, selection state and physical coordinates.

No production prior or posterior is trained or modified.

## Interpretation

Read `report/ratio_ladder_summary.csv` from left to right. The first material
increase in physical sliced-Wasserstein identifies the first broken contract:

- failure of `exact_theta` points to the simplex objective, component support,
  selection-efficiency estimate, or `v -> u` reconstruction;
- degradation at `physical_a` points to learned classifier ratios or component
  identifiability even when all five informative coordinates are observed;
- degradation at `noiseless_photometry` points to information loss in the
  decoder/photometric projection;
- degradation only at `noisy_photometry` points to the noise and selection
  observation problem.

`report/FINAL.json` records this localization, but its fixed thresholds are
diagnostic gates rather than publication or production-promotion criteria.
Inspect the physical distributions and held-out predictive metrics directly.

Each arm writes `FINAL.json`, `weights.csv`, physical/SFH marginal and joint
metrics, per-component classifier diagnostics, bootstrap results, observable
closure and dense parent/selected draws. The shared observation stage writes
the conditional-noise table, decoder residual strata and the largest residuals.
The two primary figures are `report/ratio_ladder_summary.png` and
`report/ratio_ladder_diagnostics.png`; physical marginals for every rung are in
`report/ratio_ladder_physical.png`.
