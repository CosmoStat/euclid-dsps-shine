# Coherent representation results, 2026-09-26

Run: `avi_coherent_representation_20260925_233915`, code from the coherent
representation implementation at `5aa3638`. This is a direct truth-trained
capacity oracle on the **unselected** coherent parent, not population recovery
from photometry and not individual posterior inference.

## Evidence boundary

Twenty available artifact hashes and all four stage manifest-contract hashes
match. The four stderr logs are empty. Four cache files, two best checkpoints
and `report/draws.npz` were intentionally not synchronized. Thus the stored
metrics/figures and their provenance are verified, but draws, numerical tails,
checkpoint readback and full source catalogues cannot be re-evaluated locally.

Reproduce the small-bundle analysis with:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python \
  -m scripts.analyze_feniks_coherent_representation \
  outputs/forward_population_results/avi_coherent_representation_20260925_233915
```

This writes only `analysis/`; the cluster's report and receipts are unchanged.

## What works

- Coordinate roundtrip maximum across all splits: 2.09e-14 in marginal-IQR
  units. The old support/clipping problem is absent on these recorded rows.
- Both trained transport audits pass. Largest reported expert discrepancy is
  2.45e-12, far below the 1e-5 tolerance. These are sampled numerical checks,
  not an independent global 15D integration. Normalization follows from the
  bijective coordinate map, normalized expert mixtures and conditional product.
- Physical distribution shapes are broadly reproduced. Stellar mass's main
  peak, the broad metallicity structure and the curved mass-metallicity and
  mass-dust-slope relationships are visible in the joint plot. The result does
  not support a categorical claim that this architecture cannot learn mass
  or metallicity.
- Maximum absolute Spearman error within physical coordinates is 0.03065;
  physical-to-SFH is 0.03825. Conditional dependence is present, not ten
  independent standard-normal nuisance coordinates. These correlations do not
  validate the entire nonlinear conditional distribution.

## Representation remains imperfect

| Space / group | Flow vs held-out | Validation vs test |
|---|---:|---:|
| Physical theta, 5 physical | 0.02974 | 0.01407 |
| Physical theta, 10 SFH | 0.08799 | 0.03688 |
| Normalized x, 5 physical | 0.02581 | 0.01290 |
| Normalized x, 10 SFH | 0.03325 | 0.02106 |

These are sliced-Wasserstein values normalized by the held-out marginal IQRs.
They are **not percentages**. Validation/test disagreement is context from two
finite, source-disjoint samples, not a universal noise floor or a significance
test. It cannot simply be subtracted to isolate model bias.

Physical marginal W1/IQR: redshift 0.02506, mass 0.03951, metallicity 0.04055,
Av 0.02767, dust slope 0.02808. The redshift plot shows a locally exaggerated
peak near z~1.4, but a signed mean/median redshift bias is not available in this
small bundle. Stronger tails are suggested by the very extended plot axes,
especially for SFH02/03; their frequency and contribution need saved draws.

SFH02 and SFH03 have physical W1/IQR of 0.20924 and 0.17599, compared with
0.05952 and 0.05827 in normalized coordinates. The asinh inverse expands tails;
this difference is consistent with tail sensitivity but does not by itself
prove that all error comes from rare outliers. Quantiles and tail W1 contributions
are required before imposing bounds or changing density capacity.

## Training has no demonstrated plateau

| Factor | Best epoch | Best validation NLL | Last NLL | Best gain, final 20 epochs |
|---|---:|---:|---:|---:|
| Physical | 114 | -2.78757 | -2.29915 | 0.09681 |
| SFH conditional | 116 | -19.35852 | -18.91589 | 0.21303 |

The final learning rate is 1.25e-5, with fixed validation rows. Physical
validation NLL ranges from -2.7876 to -1.2021 in the last 20 epochs; SFH ranges
from -19.3585 to -17.4383. Even the validation median loss changes materially,
so fluctuation is not demonstrably due only to one extreme object. Epoch
training losses average evolving parameter states, unlike end-of-epoch
validation; their gap is not an exact same-checkpoint generalization estimate.
There is no sustained, clean train/validation divergence establishing classical
overfitting. Optimization instability and learning narrow structures remain
plausible explanations, not proven causes.

The report uses epoch-114 and epoch-116 **best** weights. NLL continues improving
but distribution quality was measured only for these final selected checkpoints.
We cannot claim that the SW improved over training or that more epochs must fix it.

## SFH zero issue and an audit defect

For SFH10 the held-out exact-zero fraction is 5.145%; the flow has no exact zeros,
as expected for a continuous density. The more meaningful discrepancy is near
zero: within 0.01 truth-IQR, truth contains 5.96% versus only 0.8% in flow draws.
SFH09 similarly gives 3.72% versus 0.705%. Maximum within-SFH correlation error
is 0.08081. These are distribution discrepancies, not a demand for precise
individual-galaxy SFH recovery.

**Reporting bug found:** `projection_zero_table` tests log-SFR <= -29.99, following
the projection's 1e-30 logarithm safeguard. But
`model.build_diffsky_basic_sfh_table_jax` already clips raw SFR to 1e-14, with
logarithm approximately -14. The recorded "nonfloor" count can therefore include
upstream floor values. Its zero-origin attribution is not trustworthy.

Replay agreement is still valid (largest error 2.63e-6 below the 1e-5 tolerance),
and the bug does not affect training, transport or population-distance metrics.
It also does not prove the upstream floor caused these zeros. A separate local
32-object historical replay has late equal knots with log-SFR between about
-1.54 and +0.46, clearly away from either floor. Those objects are not the new
run's 1024 replay sample. Genuine plateaus and float32 resolution remain possible.

Do not add arbitrary jitter, silently discard zeros, or declare the continuous
SFH density exact because NLL decreases. Preserve the current target and first
measure which mechanism generates its repeated values.

## Next actions, in order

1. **No simulation or training:** read the existing `report/draws.npz` and
   `cache/test.npz` on Jean-Zay. Save small tables of truth/flow quantiles,
   extrema, tail probability and physical signed bias. Split W1 into central
   and extreme quantile contributions. Fix the zero-floor attribution, replay
   the same saved 1024 native rows and compare knot values with the effective
   upstream floor and local float32 resolution. This is a reporting/diagnostic
   correction, not a new catalogue or a large experiment.
2. **One bounded physical continuation:** start from the best physical weights,
   not epoch 120, at a lower learning rate (for example 3e-6), up to 60 additional
   epochs with fixed validation. Save density metrics and tail statistics at
   a few milestones on validation, choose checkpoints without test truth and
   reserve test for confirmation. A stable distribution matters alongside NLL.
   No larger architecture or sweep is yet justified. This is a proposed control,
   not a ready/validated continuation launcher.
3. **SFH conditional:** use the zero/tail audit to choose between ordinary
   lower-rate continuation and an explicit target/measure treatment. Do not
   continue solely to make the zero-spike NLL more negative. If a continuous
   approximation is retained, quantify its effect on photometry and selection.
   Underconstrained individual SFHs can be broad; a wrong SFH population can
   still bias physical population inference through the simulator.
4. **Then observed-only recovery:** freeze a 15D reference independent of the
   target truth, build compatible reference simulations, fit/calibrate observable
   ratios and selection-corrected parent weights. The truth-trained oracle above
   cannot silently be used as blind population recovery. Old fixed-independent-
   SFH banks are not validated for this target. After parent closure, train the
   supervised 15D posterior and evaluate coverage/ranks on a held-out catalogue.

The existing `--resume` skips completed factors; it will not extend this finished
120-epoch run. Do not edit its frozen YAML or resubmit it expecting continuation.
The completed coherent dataset remains reusable. No job was submitted and no
training or target code was changed during this analysis.

## Figures to read

- `analysis/convergence_zoom.png`: late instability and latest best checkpoints.
- `report/representation_physical.png`: physical marginals and normalized views.
- `report/physical_corner_theta.png`: the five-dimensional relationships.
- `analysis/dependence.png`: physical, SFH and cross-group rank correlations.
- `analysis/errors_and_zeros.png`: per-parameter error and missing SFH zero mass.
- `report/representation_sfh.png`: latent/physical tail and peak differences.

Roadmap: support/transport contracts pass on recorded checks; physical capacity
is promising but not converged/certified; SFH conditional is approximate with a
zero-origin audit defect; new-catalogue blind population and posterior stages
have not yet run. No production or publication-ready claim is made.
