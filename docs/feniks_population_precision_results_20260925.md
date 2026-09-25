# Precision audit: results and decisions, 2026-09-25

## Scope

Analyzed local mirror:
`outputs/forward_population_results/avi_population_precision_20260925_105102/`.
Cluster execution is complete. Scientific production readiness remains false.

Follow-up: [catalogue provenance audit](feniks_catalogue_provenance_20260925.md)
now isolates numerical drift with three distinct decoder paths and identifies
double-weighted truth references plus pre-r photometric cuts. Its concrete
dataset-repair actions supersede the generic next-work proposal below.

This is a known-parent, in-component-family population experiment, plus a
separate 128-object decoder check against the reference catalogue. It is not
a new real-catalogue population fit, flow-capacity test or posterior training.
The exact-ratio oracle sees the latent parameters; it is a controlled reference
with more information than photometry, not an achievable photometric optimum.

All 36 fit certificates and their required completion receipts were inspected.
Hashes of 84 present artifacts match the receipts. Intentionally excluded large
arrays and decoder batch files were not verified locally. The lightweight mirror
contains all geometry and fitted weights needed for the numerical re-evaluation.
Replayed 4096-draw metrics match the cluster to 4.6e-17. Raw cluster results were
not modified; new measurements and four figures are in `analysis/`.

## What improved

### Numerical optimization is now inexpensive and certified

- Four full fits take 0.94--1.71 seconds of recorded fitting time.
- The 32 bootstrap cells take 1.57--5.31 seconds per fit.
- Maximum full-fit KKT gap: 1.97045e-6; maximum bootstrap gap: 1.99583e-6.
- All pass the unchanged 2e-6 solver tolerance.
- Maximum error in either parent or selected weight normalization: 2.23e-16.

These timings exclude classifier caching, metric evaluation, queue wait and
other job overhead. They verify the optimization implementation on these
inputs; they do not certify the classifier or physical population model.

### The old metric obscured the interpretation

Mean physical joint sliced-Wasserstein distances to the known parent:

| Input and estimator | 4096 draws | 16384 draws | 65536 draws |
|---|---:|---:|---:|
| Noiseless / exact-latent oracle | 0.01636 | 0.00816 | 0.00701 |
| Noiseless / photometric | 0.02673 | 0.02274 | 0.02208 |
| Noisy / exact-latent oracle | 0.01697 | 0.00805 | 0.00703 |
| Noisy / photometric | 0.02991 | 0.02639 | 0.02941 |

The same fitted densities are evaluated throughout. There is no model learning
between these columns. Measurements use three numerical seeds, 64 fixed joint
projection directions and a fixed true-parent physical IQR scale. At 65536,
seed standard deviations are about 0.0004 for the oracle and 0.0028--0.0032 for
photometric fits. Values are dimensionless distances, not percentages of bias
or galaxies. Truth-to-itself distance is exactly zero with the common draws.

The maximum paired 4096-to-65536 change is 0.01265, much larger than the planned
0.002 resolution tolerance. This weakens previous threshold-only diagnoses of
fundamental non-identifiability. Nevertheless, a substantial photometric gap
remains at high precision. The noisy curve is not monotonic, and three seeds
do not prove that every 65536-draw metric is numerically converged.

### Stability is better than the original low-resolution verdict suggested

No fits were rerun locally. Each of the saved bootstrap weight vectors was
re-evaluated at 65536 draws with three numerical seeds, against its own full
catalogue fit. The per-repeat value is the mean over numerical seeds.

| Input / estimator | Catalogue repeats | Median SW | Maximum SW |
|---|---:|---:|---:|
| Noiseless / exact-latent oracle | 8 | 0.00467 | 0.00813 |
| Noiseless / photometric | 8 | 0.00959 | 0.01258 |
| Noisy / exact-latent oracle | 8 | 0.00465 | 0.00826 |
| Noisy / photometric | 8 | 0.01319 | 0.03908 |

At the cluster's 16384-draw evaluation, photometric medians were 0.00968 and
0.01694. The apparent noisy median failure weakens with more accurate
measurement, but noisy repeat 6 remains displaced at 0.03908 +/- 0.00087
(standard deviation over numerical seeds). Its oracle counterpart is 0.00711.
That difference is not just evaluation-seed noise.

These are eight paired catalogue resamples, not 32 independent resamples per
arm and not 24 after adding numerical seeds. Classifiers, selection efficiencies,
rank and regularization remain frozen. The experiment does not include classifier
training uncertainty, selection-efficiency uncertainty or model-choice uncertainty.
The old 0.015 line is a historical reference, not a newly certified accuracy
requirement. Do not declare a full PASS from the new medians alone.

## Which physical variables remain problematic?

Marginal W1 divided by fixed physical IQR, mean of three seeds at 65536 draws:

| Input / estimator | Redshift | log stellar mass | log metallicity | Dust Av | Dust slope |
|---|---:|---:|---:|---:|---:|
| Noiseless / exact | 0.0051 | 0.0042 | 0.0032 | 0.0128 | 0.0069 |
| Noiseless / photometric | 0.0057 | 0.0064 | 0.0115 | 0.0562 | 0.0180 |
| Noisy / exact | 0.0052 | 0.0040 | 0.0034 | 0.0129 | 0.0069 |
| Noisy / photometric | 0.0069 | 0.0051 | 0.0073 | 0.0729 | 0.0154 |

Dust Av dominates the marginal discrepancy. Noisy bootstrap repeat 6 also
differs from the full fit mostly in Av (0.0892) and dust slope (0.0330), with
smaller redshift (0.0161), metallicity (0.0081) and mass (0.0048) distances.
This makes dust-sensitive uncertainty a concrete next target rather than a
generic change in all 128 components. Marginal agreement alone is not joint
agreement; the joint SW gap remains.

It does NOT follow that stellar metallicity is accurately inferred for each
individual galaxy, or that the prior redshift bias on the reference catalogue
has disappeared. Those are different inference/validation questions. The
oracle-versus-photometry gap may contain both classifier approximation error
and genuine loss of physical information; this experiment does not separate them.

## Decoder screen: a real discrepancy and a failed comparison design

The inherited effective `decoder_model.json` already specifies
`photometry_integrator: merged_gauss4_v1`. The second branch also selects
`merged_gauss4_v1`. Their paired flux outputs are exactly equal, and all 16
pairs of batch hashes agree. The intended integration comparison was a no-op.
This is a setup mistake, not evidence that a numerical improvement worked.

The comparison of decoded fluxes with catalogue noiseless fluxes remains useful:

| Band | p95 absolute residual / reported sigma |
|---|---:|
| Roman F087 | 1.426 |
| Roman F062 | 0.853 |
| Roman F106 | 0.540 |
| Roman F146 | 0.390 |
| Euclid VIS | 0.329 |

These five bands exceed the configured 0.25 screen target. The other 13 pass
that per-band p95 screen, not necessarily every object-level residual.
Selection identity mismatches are zero. Conditional noise means/std are not
obviously catastrophic, but this 128-object screen does not re-certify the
earlier full noise audit.

Example: row 4746 in F087 has S/N 169.2. A -1.55 percent flux discrepancy is
-2.63 reported sigma. Bright objects expose small relative model differences;
this is not evidence that brighter data are intrinsically less informative.

The effective decoder also records float64 MDF/spline precision and a
float32 compressed SSP runtime. These are provenance fields to compare with
the catalogue generator, not identified causes of the residual. The copied
runtime configuration currently disables global/per-band calibration, but it
does not by itself establish the original generation contract. Recover the
archived generator, filters/assets, transformations and numerical settings
before deciding which bank, decoder or dataset needs to change.

## Next work, without a broad restart

1. **Correctness branch:** pin the original catalogue generation provenance and
   the effective inference configuration; replay the same small set of objects
   through both paths. Assert that a proposed numerical comparison really uses
   different relevant settings. Resolve the observed flux mismatch and confirm
   on a disjoint, S/N- and redshift-stratified set. Do not rerun identical variants.
2. **Uncertainty branch:** inspect the saved noisy repeat 6 and full-fit physical
   densities, then test their observable predictions on independent simulations.
   Reuse valid banks where possible; assess whether the classifier distinguishes
   populations that the independent forward predictions also distinguish. This
   can proceed alongside the provenance investigation, with conclusions scoped
   to the fixed simulator until compatibility is established.
3. **Confirmation:** after the observation contract is fixed, freeze one population
   configuration and run the parent-plus-supervised-15D-posterior chain. Require
   independent parent closure and posterior coverage/ranks/bias/widths, including
   parents outside the fitted component family. Allow broad SFH uncertainty;
   do not fix or remove the ten SFH coordinates.

No new HPC run was launched for this analysis. No classifier or flow was trained,
so this run gives no new neural-network convergence evidence. Adding filters,
another architecture sweep, or repeating an eight-hour bootstrap is not the
immediate next step justified by these results.

## Where to look

All plots are under the synchronized run's `analysis/` directory:

- `01_metric_precision.png`: why a low-draw metric changes the old verdict.
- `02_bootstrap_precision.png`: noiseless stability and the persistent noisy tail.
- `03_physical_errors.png`: dust Av versus redshift, mass and metallicity.
- `04_decoder_contract.png`: failing bands and the high-S/N residual pattern.

`CHECKS.json` records normalization/KKT checks and the no-op decoder comparison.
`bootstrap_precision.csv` contains all 96 numerical evaluations of the 32 saved
fits. `bootstrap_by_repeat.csv` keeps catalogue repeats distinct from numerical
seeds. Reproduction commands are in the precision runbook; the live scientific
checklist is [the validation roadmap](feniks_population_validation_roadmap.md).
