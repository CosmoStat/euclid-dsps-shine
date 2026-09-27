# Reference redesign: verified results and decision, 2026-09-27

Source: `avi_reference_redesign_20260927_163901`, contract
`7ffdc8e285389a1d87e74d6b6d3898dddde404d0803d327421ebca1a540b4d2b`.
The local download contains 42 receipt-verified artifacts; five excluded NPZs
are unavailable. This is reference-family qualification, not a new blind fit,
and contains no trained posterior. Completion does not mean capacity PASS.

## What improved

| Fixed reference | Validation CDF maximum | Physical SW | Qualified |
|---|---:|---:|---|
| Legacy | 0.1319 | 0.1739 | No |
| Affine log-mass only | 0.1340 | 0.1854 | No |
| Local 128 | 0.0371 | 0.0683 | No |
| Local 256 | 0.0320 | 0.0506 | No |

Local 256 reduces physical SW by about 71% relative to legacy. All five
physical marginal errors now pass the 0.1 W1/IQR tolerance:

| Parameter | W1 / truth IQR |
|---|---:|
| Redshift | 0.03595 |
| Log stellar mass | 0.05102 |
| Log stellar metallicity | 0.03310 |
| Dust Av | 0.06508 |
| Dust delta | 0.03445 |

This does NOT show that mass or metallicity cannot be learned. It shows a
substantial improvement in representability within the locally resolved family.
The remaining gates are validation CDF (0.0320 versus 0.0300, driven by mass)
and joint physical SW (0.0506 versus 0.0350). Empirical comparator SW is 0.0134;
the remaining discrepancy is not shown to be only finite evaluation noise.

## Why another larger family is not yet justified

The old LP globally minimizes maximum CDF discrepancy over a finite set of
projections in **transformed coordinates**. Local 256's TRAIN optimum is 0.02301.
That is a certificate for those TRAIN features, not the minimum attainable
physical-space SW. The fit and the downstream evaluation are different
objectives. It is premature to declare the entire local 256 family inadequate
for the physical objective, or to jump to 512/1024 components.

The next controlled change is only the TRAIN fitting objective: integrated
physical CDF error, with the old TRAIN CDF criterion kept as a constraint.
The fixed geometry, 15 dimensions, observation model and validation thresholds
are unchanged. This can falsify the objective-mismatch explanation cheaply.
It is not guaranteed to pass and must not relax acceptance criteria after seeing
the result. Even another failure would concern the tested fitting procedure,
not a mathematical impossibility theorem for all 15D density models.

## What the paired replay established

Saved bank reproduction passes at maximum noise-scaled error 4.66e-13.
The original 32 targeted extreme cases are already above validation r-band
q99.9 when evaluated at their native anchors, before kernel perturbation.
The reference-anchor population therefore matters, not only kernel widths.
These targeted cases do not estimate tail prevalence across the reference.

Median absolute log10 flux perturbation relative to the paired anchor:

| Cohort | Legacy | Affine mass | Local 128 |
|---|---:|---:|---:|
| Reserved controls | 0.2755 | 0.2672 | 0.03785 |
| Targeted extremes | 0.2179 | 0.2170 | 0.05303 |

Local physical kernels reduce perturbations strongly; affine mass alone has
little effect on most selected pairs. Its finite-linear-mass-moment property
remains mathematically useful, but it is not the complete empirical tail fix.
Replay evaluated local 128, not local 256: do not claim direct replay evidence
for a case that was not run. The next local 256 bank tests its observable tails.

## SFH caveat

Local 256 SFH marginal discrepancies range roughly from 0.22 to 0.89 W1/IQR.
Weakly constrained *individual* SFHs and legitimate zero star formation are
not themselves bugs. A mismatched *population conditional* of SFH given physical
parameters can nevertheless bias photometric population learning. Physical-only
mixture weights preserve joint native associations; they cannot guarantee an
arbitrary correct SFH conditional. This remains explicitly unvalidated.

## Look here

Under `outputs/forward_population_results/avi_reference_redesign_20260927_163901/`:

1. `local_256/capacity/physical.png`: truth versus the capacity-fitted parent,
   five physical marginals. Not a photometry-only recovery result.
2. `analysis/physical_sfh_errors.png`: newly generated per-parameter comparison;
   strong physical improvement, residual SFH population mismatch.
3. `replay/paired_bright_replay.png`: anchor flux versus perturbed flux; distinguish
   native bright rows from perturbation amplification.
4. `analysis/AUDIT.json`, `analysis/paired_amplification.csv`: source-checked numbers.

Reproduce the local review without the excluded large files:

```bash
.venv/bin/python -m scripts.analyze_feniks_reference_redesign \
  outputs/forward_population_results/avi_reference_redesign_20260927_163901
```

The implementation and next-run commands are in
[Reference to parent](feniks_reference_to_parent.md).
