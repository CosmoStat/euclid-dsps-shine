# Population validation roadmap, 2026-09-25

## Scope and evidence

Goal: learn a selection-corrected parent population and calibrated individual
15D posteriors. Keep the ten SFH coordinates active and marginalized through
simulation. Population learning has no input from approximate posterior draws.
Broad SFH posteriors are acceptable; a wrong assumed SFH distribution can still
bias inferred physical population coordinates through its photometric effects.

The current audit is a synthetic population experiment with a known parent in
the component family. Success here would not establish recovery for every
parent shape, for the reference FENIKS catalogue, or for real sky observations.

## Latest evidence: precision run completed

### 2026-09-26 coherent representation results

The [synchronized representation review](feniks_coherent_representation_results_20260926.md)
verifies 20 available hashes. Coordinate/transport checks pass. Physical SW is
0.02974 versus validation/test 0.01407; physical-SFH correlation maximum error
is 0.03825. These are promising representation results, not blind parent recovery.
Best checkpoints are epochs 114/116; both losses still improve and fluctuate.
SFH10 near-zero mass is 5.96% in truth versus 0.8% in draws. The zero audit used
the wrong floor (-30 versus upstream -14); origin attribution needs correction,
although replay and density metrics remain valid. Next: saved-draw tail/bias
tables plus correct zero replay, then one lower-rate physical continuation with
validation distribution metrics. Do not resimulate the coherent dataset or
silently promote its truth-trained oracle to an independent population reference.

### Coherent qualification and representation adapter

The user reports coherent dataset integrity PASS, but old-transform qualification
BLOCKED: 0.423% of masses and rare SFH contrasts lie outside the old support.
Physical/SFH maximum absolute Spearman is 0.360; the final SFH contrast has 5.788%
exact zeros. This invalidates reusing the old clipped coordinates or treating an
independent SFH reference as automatically adequate.

The [coherent representation job](feniks_coherent_representation.md) now provides
an unclipped mass/SFH coordinate adapter, a two-factor structured density oracle
and a small projection-zero replay. Reuse the finished dataset without new
photometry. Physical and conditional SFH factors train independently in parallel;
old checkpoints and banks are not reused. This is **truth-labelled representation
testing**, not observed-only parent recovery. Exact SFH atoms, the independent
reference conditional, new population closure and final posterior calibration
remain open scientific gates. No production promotion is implied by completion.

### Coherent dataset completed on Jean-Zay (user-reported)

The latest watcher reports `COHERENT_PARENT_DATASET_COMPLETE`: 140000 parent
rows and 85514 observed-r-selected rows, with all 14 photometry tasks complete.
This advances data preparation, not population or posterior inference. Artifacts
have not yet been independently read back locally. The
[next-jobs runbook](feniks_coherent_next_jobs.md) gives a read-only CPU check
and the dependencies for the next population/posterior jobs. The historical
forward launcher is not an adapter for the new dataset and must not be reused
unchanged. No further long bootstrap sweep is required at this stage.

### Provenance follow-up: target and observation contracts must be repaired

The [catalogue provenance audit](feniks_catalogue_provenance_20260925.md) now
identifies concrete problems behind previously ambiguous failures:

- The final catalogue is already proposal-weight resampled, but truth reports
  apply retained proposal weights again. Blind mean z changes from 1.39276 to
  1.28451. Correct report targets before interpreting prior biases.
- The same 128-object replay gives maximum per-band p95 residuals of 0.0194
  sigma (native/legacy), 0.1712 (spline/legacy), and 1.4260 (spline/current).
  Numerical drift is a substantial part of the decoder mismatch. This is a
  current-source replay, not exact recreation of the historical environment;
  legacy-spline individual outliers remain. No flow was used in this comparison.
- Historical root data already pass multi-band S/N/magnitude cuts. r-only
  correction cannot establish recovery of the photometrically unselected parent.
- Reassigned object-ID splits share source proposals. The original generator
  also reuses effective seeds across differently named split files. Future
  confirmation must separate effective source realizations, not ID prefixes.
  The 15D SFH conditional still needs scientific checks.

The user has now confirmed 256/56/59 raw proposal shard files on Jean-Zay.
The [coherent parent preparation workflow](feniks_coherent_parent_runbook.md)
implements once-only weights, disjoint source pools, reprojected 15D photometry
and explicit parent/selected outputs. The canonical 256-shard pool is partitioned
183/37/36 by effective seed; overlapping original split files are not counted twice.
Local tests and an actual-DSPS smoke pass;
remote completion is now user-reported; independent artifact readback and
scientific interpretation remain to be checked.
This is a new data-preparation launcher, not a production training launch.
Reuse compatible forward banks only after a full contract comparison. The old
generated catalogue remains an explicit mismatch test rather than being overwritten.

### Precision measurements

Run: `avi_population_precision_20260925_105102`. The synchronized receipts and
84 present artifact hashes were checked. Excluded large arrays remain unverified
locally. See [the detailed analysis](feniks_population_precision_results_20260925.md).
Original cluster reports were preserved; local higher-precision bootstrap
measurements and figures are in the run's `analysis/` directory.

- All four full fits and 32 bootstrap cells pass the configured KKT tolerance
  of 2e-6. Parent and selected weights normalize to floating-point precision.
  Recorded fit times are 0.94--1.71 s (full) and 1.57--5.31 s (bootstrap).
- The 4096-draw metric was not accurate enough for the old threshold decisions.
  At 65536 draws, mean parent SW is 0.0070 for the exact-latent oracle, 0.0221
  for noiseless photometry and 0.0294 for noisy photometry. The residual
  photometric gap is not explained away by increasing evaluation draws.
- Re-evaluating the existing eight catalogue resamples at 65536 draws and
  three numerical seeds gives median bootstrap SW 0.0096 (noiseless) and
  0.0132 (noisy), versus about 0.0047 for the matched exact-latent oracle.
  The noisy maximum remains 0.0391. These are conditional, eight-repeat
  diagnostics, not a full uncertainty certification or an automatic PASS.
- The largest physical marginal error is dust Av: W1/IQR 0.0562 without noise
  and 0.0729 with noise. Redshift and stellar-mass marginal distances are much
  smaller in this synthetic case; they are not the principal residual here.
- The decoder screen has a design defect: the inherited baseline already uses
  `merged_gauss4_v1`. Both variants therefore run the same integrator, with
  identical paired outputs. It has NOT tested an integration improvement.
  The actual catalogue mismatch still exists: F087 p95 is 1.426 reported sigma
  against the screen's 0.25 target. Selection identity has zero mismatches.

This changes the immediate priority: do not rebuild the population basis or add
bands solely because of the old 4096-draw failure. Resolve the observation
contract and characterize the surviving dust-sensitive variation first.

The evidence below describes the earlier low-rank run and is retained for
provenance; its raw metric values use a different finite-draw setup.

Verified source artifacts:

- `outputs/forward_population_results/avi_population_low_rank_audit_20260924_190751/`
  contains the complete noiseless result and incomplete noisy bootstrap.
- Its `noiseless_photometry/runtime/feature_stats.json` specifies six LSST bands,
  Euclid VIS plus three NISP bands, and eight Roman bands: 18 bands in total.
- Its manifest uses 4096 metric draws; `population_metrics` uses 64 projected
  directions after division by reference marginal IQRs.
- `_common_draws` reuses component-selection uniforms and Gaussian variates
  across mixture weights. Identical weights therefore produce identical samples
  and zero distance. An independent-sample null would not reproduce this metric.
- The noiseless bootstrap median is 0.01888 against a configured 0.015 gate;
  noisy heldout-admissible candidates have parent SW 0.03298--0.03542 against
  a 0.03056 closure gate. No noisy bootstrap table survived the timeout.
- The earlier ratio-ladder observation audit reports a decoder residual p95 of
  about 1.13 reported sigma in Roman F087. This is a discrepancy between numerical
  photometry and the reference catalogue, not proof of irreducible astrophysical
  model error. Its noise contract passes separately.

## Interpretation correction

The configured gates fail. This does NOT yet prove that the data fundamentally
cannot identify the parent or that more filters are required. The current
metric combines finite-catalogue variation, approximate classifier effects and
finite-draw distance estimation. Its bootstrap interval only measures resampling
of the saved replicates, conditional on the classifier and metric setup.

Likewise, stable or unstable component coefficients are not equivalent to stable
or unstable physical densities: overlapping components can exchange mass with
little observable or physical consequence. Assess the joint physical density,
redshift/mass/metallicity marginals, correlations and their uncertainty.

A parameter-space oracle observes more information than any photometric method.
If it succeeds while photometric inversion fails, either genuine information
loss or classifier error may explain the gap. Oracle success alone does not
identify classifier error. Oracle failure means the chosen estimator has not
demonstrated the desired precision, not a general impossibility theorem.

## Ordered decisions

### 1. Calibrate the evaluation and sampling reference

Reuse the saved component family, target catalogue, selection efficiencies and
exact latent ratios. Use the same selected-object identities and bootstrap
counts as the photometric fit. Freeze model choices before this comparison.

- Measure exact-ratio parent closure and bootstrap stability on matched samples.
- Separate variation from increasing the number of observed galaxies from
  variation due only to increasing metric draws (4096, 16384, 65536).
- Fix projection directions and physical scaling across comparisons; quantify
  seed variation with the existing common-random-number coupling. Include the
  equal-weight zero-distance invariant. Varying metric draws needs no DSPS.
- Report physical marginals and joint density uncertainty, not only one SW.
- Begin with eight paired replicates, saving every replicate. Extend only when
  the uncertainty on the resulting decision requires it. Eight is a diagnostic
  first tranche, not a publication-level bootstrap certification.
- Measure first-refit timing and cap wall time before submission. This proposal
  does not promise that 32 solves fit in a short allocation.

Deliverable: one comparison of numerical metric variation, oracle sampling
variation and photometric variation. Retain the original 0.015 gate in reports;
any revised scientific accuracy target must be justified independently and then
tested on an untouched confirmation sample.

### 2. Resolve the observation-model discrepancy

For identical latent objects, compare catalogue noiseless fluxes and decoded
fluxes with explicit filter curves, units, AB normalization, redshift, SED assets,
integration settings and latent transforms. Plot residuals in flux and reported
sigma by band, redshift and S/N. A small relative error can exceed sigma for a
bright object. Numerical and statistical model checks are different tests.

Existing decoder-quadrature work is documented separately in
`feniks_full_decoder_qualification_runbook.md`; its forward/gradient checks are
not evidence that catalogue compatibility has already passed. Preserve bank
provenance and regenerate only photometry invalidated by a verified correction.

### 3. Identify which physical ambiguities matter

Use the frozen photometric likelihood approximation to find poorly constrained
population directions, and express them as changes in p(z), p(M), p(Z) and their
joint distributions. Check these competing populations with independent forward
predictive data, because a learned likelihood Hessian alone cannot establish
intrinsic identifiability. All fifteen latent coordinates must remain variable.

If differences exceed the oracle sampling reference, distinguish classifier
approximation from indistinguishable photometric distributions before changing
population capacity. Good average classification NLL or normalization of ratio
moments alone is not enough. Broad uncertainty is appropriate where the data
leave genuinely different parent densities compatible.

### 4. Test added bands only against an explicit ambiguity

The current 18 measurements are not 18 independent physical constraints.
Overlapping bands can improve precision if noise is independent but may provide
little new spectral information. Candidate additions should sample spectral
features or wavelength ranges that distinguish the competing populations, with
realistic depth, covariance and missingness. Medium bands for redshift are a
motivated candidate, not a verified solution for this run.

Compare the original 18-band baseline with one justified augmented set, using
the same latent objects, original-band noise realizations, r-band selection,
splits and comparable training/convergence checks. Retrain the classifiers for
their respective inputs. Evaluate gain on independent test data and require
that it exceed metric/sampling variation. Keep r<29 selection fixed unless a
change in the survey selection is explicitly part of the experiment.

Additional bands must be measured for the target catalogue to help its actual
inference. Generating them only in simulations evaluates a future observing
scenario. Broad-band photometry does not guarantee sharp stellar metallicities;
do not interpret a narrow prior-driven result as new data information.

Scientific context: [Ilbert et al. (2009)](https://arxiv.org/abs/0809.2101)
demonstrate a 30-band photo-z pipeline with broad/intermediate/narrow bands;
its gains also depend on template and emission-line modeling. Those results
do not predict accuracy at r<29 for Feniks.
[Conroy (2013)](https://arxiv.org/abs/1301.7095) reviews SED constraints and their
degeneracies across stellar mass, metallicity, dust and SFH.

### 5. One end-to-end confirmation

After the preceding evidence supports a fixed observation model and population
estimator, fit the parent, freeze it, and train a supervised 15D posterior from
simulation truth. Validate parent marginals/joints, observable predictions,
68/95 percent coverage, ranks, bias and widths on independent simulations, with
redshift/magnitude/S/N strata. Test parents outside the exact fitted component
family and plausible SFH conditional changes before broad scientific claims.

Report uncertainty in the recovered parent. Individual intervals conditional on
a single fitted parent do not automatically include population uncertainty;
quantify this effect before claiming fully calibrated population-marginalized
posteriors. SFH recovery need not be sharp to pass, but physical inference must
remain calibrated after nuisance marginalization.

## Current checklist and next actions

| Block | Status after the precision run | Remaining evidence |
|---|---|---|
| Selection algebra and v/u normalization | Existing checks retained; actual new fits normalized | Efficiency uncertainty and support not newly qualified |
| Convex population solve | KKT passes all 36 fits; seconds per fit | This certifies this objective, not correct population recovery |
| Full 15D reference and no q feedback | Architecture retained, no q used in this audit | Correctness of the SFH conditional for a target is a separate question |
| Exact-latent population closure | Good in-family reference, SW about 0.007 | Out-of-family parents and other catalogue realizations |
| Evaluation precision | 4096-draw problem identified; 65536-draw checks completed | More seeds/draws only for decisions near a numerical boundary |
| Empirical target weighting | Once-only preparation COMPLETE in user watcher | Recheck completed receipts; historical reports remain distinct |
| Unselected-parent definition | 140k parent and 85514 selected rows prepared without photometric preselection | Verify hashes/views/split separation; SSP-metallicity support still explicit |
| Photometric parent stability | Noiseless encouraging; noisy tail remains | Dust-sensitive repeat 6; classifier error versus information loss |
| Noise and selection identity | Earlier noise pass retained; new selection check passes | 128-object screen is not a fresh full noise qualification |
| Decoder versus catalogue | New self-consistent 15D flux contract COMPLETE in user watcher | Independent numerical accuracy is not established by self-replay |
| New target versus old reference | Qualification implemented; not yet run on cluster artifacts | Bounds, transform clipping, SFH pileups and conditional assumptions |
| Reference-catalogue parent recovery | Not validated | Observation compatibility and independent population closure |
| Final individual 15D posterior | Not tested in this run | Independent 68/95 coverage, ranks, widths, bias and predictive checks |

The following original branches are superseded on the critical path by the
provenance follow-up above. They remain useful within a frozen simulator:

1. Observation compatibility: recover the archived catalogue-generator numerical
   contract and asset hashes; compare it with the saved effective decoder config.
   Replay the same objects through the generator-compatible and inference paths,
   requiring distinct effective contracts when testing a proposed change. Record
   filters, units, SED/SSP precision, transforms and any calibration. Fix a proven
   mismatch, then confirm on disjoint objects across S/N/redshift. A simple forced
   legacy-versus-merged comparison is insufficient unless it matches provenance.
2. Population uncertainty: reuse the saved full-fit and bootstrap weights to
   display physical marginals and observable predictions, especially noisy
   repeat 6. Compare likelihoods on independent forward data before attributing
   the variation to classifier error or intrinsic degeneracy. Do not select a
   new rank/penalty using the known truth in this diagnostic.

Only after these checks: one fixed end-to-end confirmation with a frozen learned
parent and supervised 15D posterior. No new classifier sweep, 8-hour bootstrap
restart or extra-band campaign is justified by these results alone. Steps 1 and
part of 3 advanced; step 2 remains the immediate correctness blocker. Scientific
production readiness is still false. No remote job was launched during analysis.

Each run exports execution-level `ROADMAP_STATUS` files. This document is the
scientific interpretation; do not replace historical raw receipts with revised
verdicts. The [launch runbook](feniks_population_precision_runbook.md) documents
the completed audit, and the [band assessment](feniks_popcosmos_bands_what_if.md)
remains a what-if rather than the next mandatory experiment.
