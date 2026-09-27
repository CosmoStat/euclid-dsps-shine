# Coherent inference: results inspected on 2026-09-27

Run: `outputs/forward_population_results/avi_coherent_inference_20260926_094557`.
Reproduce the small-result analysis with:

```bash
python -m scripts.analyze_feniks_coherent_inference \
  outputs/forward_population_results/avi_coherent_inference_20260926_094557
```

The script verifies 30 available receipt artifacts, marks eight missing large
artifacts explicitly, and writes `analysis/SUMMARY.json`, calibration and
checkpoint CSVs, plus `analysis/diagnosis_summary.png`. Bank contents are not
present locally. Completed-attempt stderr contains only occasional hwloc CPU
binding warnings, not the old reference support traceback. Old logs are retained.

## What improved

The supervised control trained on the correct coherent selected population
achieves physical marginal coverage of 68.96% / 95.14% on 1024 test galaxies
(512 joint 15D draws each). Per-coordinate 68% coverage is 66.8--70.1%; PIT KS
is 0.026--0.037. Stellar mass and metallicity are learnable by this posterior
architecture in this controlled setting. This does not certify every joint
15D region or every S/N/redshift stratum, or a real survey.

Saved trained transport checks pass. Population u and selected v both sum to
one; recomputing normalized v/alpha reproduces u. KKT gap 7.87e-7 passes the
2e-6 numerical condition. No population update reads q-generated samples.

## What does not work

The learned parent reduces physical SW from the broad reference's 0.7229 to
0.1239, but the independent validation/test comparator is 0.01407. The remaining
discrepancy is about 8.8 times that comparator, not a claim of a calibrated
statistical significance. Mass W1/IQR is 0.284 and redshift 0.178. Learned
selected physical SW is 0.1182, also poor. SFH parent SW remains 0.651.

Three of 64 components carry 93.34% of parent probability. Effective component
count 1/sum(u^2) is 3.22. This describes concentration, not its cause: a sparse
optimum can reflect a restricted basis, inaccurate ratios or photometric
degeneracies. Here alpha_j lies in [0.168, 0.990], so unlimited inverse selection
of almost invisible components is not supported by the saved numbers.
Learned total selection probability is 0.5932 versus target-test 0.60635.

Final target-catalogue calibration is not acceptable:

| Coordinate | Oracle 68% | Final 68% | Final/oracle median 68% width |
|---|---:|---:|---:|
| Redshift | 69.3% | 75.1% | 1.10 |
| log stellar mass | 69.0% | 74.7% | 1.20 |
| log metallicity | 69.5% | 91.1% | 4.86 |
| Av | 70.1% | 72.8% | 1.14 |
| Dust slope | 66.8% | 88.9% | 2.82 |

The old statement "everything is too narrow" is wrong for this run. The final
metallicity posterior is broad AND biased (median bias -0.0622 dex, PIT KS
0.257). SFH10 still has undercoverage at 57.2%, hidden by group averages.
On the final model's own simulated population, physical average coverage is
67.34% / 94.79%, but individual PIT discrepancies persist (mass KS 0.096).
The 1024 in-model draws reuse only 871 distinct bank rows; this is not an
independent 1024-object calibration experiment. These comparisons implicate
population/reference mismatch while leaving a residual amortization error.

## Convergence is not just the watcher label

| Model | Best epoch / NLL | Last epoch / validation NLL | Interpretation |
|---|---|---|---|
| Classifier | 129 / 3.0990 | 160 / 3.1038 | Plateau reached under configured criterion |
| Oracle posterior | 199 / -34.2117 | 200 / -34.0666 | Budget exhausted; still improving |
| Final posterior | 29 / -0.1236 | 100 / +10.7655 | Strong validation deterioration, not a healthy plateau |

Final train NLL reaches -9.4058 while validation worsens. The source restores
`best.eqx` for evaluation: the reported calibration is intended to use epoch29,
not epoch100. The large checkpoint was excluded locally, so no independent
deserialization/hash check of its contents was performed here.

The weighted bank has 492953 selected training rows but importance effective
size 21435; validation 70445 rows, effective size 3017. These are not literal
independent-equivalent training counts, but show weight concentration and
justify investigating finite-bank overfitting. More epochs on this bank are
not the appropriate next action. Raw NLLs between oracle and final are not
comparable as a ranking: their target populations/training measures differ.

## Observable warning and unresolved causes

Raw flux W1 / target IQR reaches 211.8 in r and 221.2 in F184. These are not
magnitude errors. A small bright tail can dominate this linear-flux statistic;
the existing report uses 20000 importance-resampled bank rows. We cannot infer
the bulk error, or blame units, without weighted CDFs and tail diagnostics.

The current reference uses 32768 independent native 15D anchors, isotropic
kernel bandwidth 0.15 in normalized coordinates and 64 soft physical gates.
Only 64 weights adapt; conditional anchor laws and smoothing are fixed. The
gate definition mixes in a 2% uniform term BEFORE each component is normalized.
It is not automatically a 2% final-parent tail mass. Fixed smoothing or shared
tail support can limit achievable shapes; this is a hypothesis, not yet a
measured explanation of the redshift or flux bias.

The classifier learns (audit NLL 3.108 versus null 4.079; ratio-moment median
error 0.0125). Matching these moments does not establish all conditional ratios
or identifiability under a target outside the reference family.

## Reading the figures

1. `analysis/diagnosis_summary.png`: coverage, parent error, final overfitting,
   weight concentration in one figure.
2. `report/coverage_comparison.png`: blue is the correct-population control,
   orange is the learned-parent posterior on target objects, green is its own
   model check. Compare each coordinate with 68%/95%, not just an average.
3. `report/parent_selected_reference_15d.png`: compare true parent with learned
   parent. The selected distribution is a DIFFERENT target and is not supposed
   to equal the unselected parent. Displayed histogram tails are not renormalized.
4. `report/parent_physical_corner.png`: remaining physical correlations/shapes.
5. `report/training.png`: early best final checkpoint and later overfitting.
6. `report/pit_comparison.png`: final metallicity's nonuniform ranks.

## Immediate next step

Use the [read-only reference audit](feniks_coherent_reference_audit.md): one
short CPU job, no simulations/neural training. It tests the fixed component
family with analytic projected CDFs and TRAIN-only diagnostic truth weights,
then checks independent held-out distributions and exact weighted observable
tails. It must not export its truth-fitted weights as a production prior.

After that result, change the reference family only if capacity is limiting;
otherwise isolate ratio/inversion/identifiability errors. Generate diversified
final-parent NPE pairs only after parent predictive closure improves. Joint
and conditional posterior calibration and real-survey validation remain open.
