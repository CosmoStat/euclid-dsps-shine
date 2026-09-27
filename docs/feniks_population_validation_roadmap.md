# Population validation roadmap, updated 2026-09-27

## Scope and evidence

Goal: learn a selection-corrected parent population and calibrated individual
15D posteriors. Keep the ten SFH coordinates active and marginalized through
simulation. Population learning has no input from approximate posterior draws.
Broad SFH posteriors are acceptable; a wrong assumed SFH distribution can still
bias inferred physical population coordinates through its photometric effects.

The current coherent benchmark uses a known synthetic target parent and an
independent native reference. The target is NOT guaranteed to lie in that
reference's component family. Earlier in-family tests do not establish recovery
on this target, on the historical catalogue, or on real sky observations.

## Current execution decision: advance posterior in parallel

User explicitly authorizes advancing despite dust failure. The
[fresh-parent posterior continuation](feniks_parent_to_posterior.md) now adds
fresh direct-parent simulations -> unit-weight supervised 15D posterior ->
separate in-model/coherent-target calibration, in a new exploratory run.
No dust/SFH removal, q feedback or relabelling of source gates. The frozen CPU
recovery diagnostic (user-reported 253496-253498 pending) remains untouched and
independent. These two branches can run concurrently.

- Parent normalization/numerical checks: retained, not recomputed as new science.
- Parent recovery: dust, joint SW and ratio gates still unvalidated/failed.
- Fresh learned-parent posterior: implementation prepared; remote run pending.
- In-model vs target transfer: separate evaluation planned; no result claimed.
- Final production/paper validation: still requires full calibration and a
  genuinely untouched evaluation set. Tolerating dust is not passing all checks.

## Latest result: blind fit completes, physical recovery still fails

Exploratory run `avi_reference_to_parent_20260927_184817` is now synchronized.
User-reported recovery jobs 252749/252750 completed. Twenty-three small receipt
artifacts verify locally; basis/checkpoint and simulation banks are not local.
See the [completed blind-parent review](feniks_blind_parent_results_20260927.md).

| Block | Latest evidence / status |
|---|---|
| Numerical inversion and parent/selected weights | PASS on saved evidence: sums 1, u = normalized v/alpha, all KKT < 2e-6 |
| Selection fraction | PASS: learned 0.60235 vs validation 0.60995; component alpha >= 0.09399 |
| Classifier optimization | Best epoch 399, stopping at 460 on implemented validation plateau |
| Classifier ratio accuracy | Still unvalidated: independent moment 0.03170 > 0.030, sampling uncertainty not saved |
| Learned physical marginals | Four of five PASS; dust Av 0.18677 parent and 0.17356 selected exceeds 0.1 |
| Joint physical recovery | FAIL: parent SW 0.09751 and selected SW 0.07666 exceed 0.05 |
| Physical capacity control | Better parent SW 0.02798, but not a valid full-population oracle |
| Observable prediction | Marginal CDF/tail-frequency gates PASS; joint colors untested and raw tail-amplitude error remains |
| SFH conditional / joint reference | Unvalidated; SFH SW ~0.38 even for physical capacity control |
| Final posterior under new learned parent | Fresh simulation-trained exploratory run prepared; not yet evaluated |
| Production / paper claims | Not approved; strict capacity CDF FAIL retained |

**Next diagnostic implemented, awaiting Jean-Zay:** the
[frozen-classifier recovery check](feniks_parent_recovery.md), using two known
mixtures, disjoint reserved bank roles, fixed source penalty/alpha and label-aware
sampling/penalty controls. CPU cache -> two-case CPU array -> report, no new
DSPS, classifier training or posterior. Tail attribution, joint asinh-flux
diagnostics and conditional ratio-moment uncertainty reuse the same rows.
This separates an inversion/ratio problem from joint reference/target mismatch
before choosing a retraining intervention. In-family PASS would not validate
the coherent target or authorize the final posterior automatically.

Do not interpret four marginal passes or good one-band flux CDFs as full 15D
recovery. Conversely, the small ratio-moment gate excess alone does not establish
the cause of the substantially larger dust discrepancy. Normalization and KKT
success do not remove statistical or model-family error.

## Prior state before numerical recovery

Update from exploratory run `avi_reference_to_parent_20260927_184817`:
four reference banks completed and classifier stopped on validation plateau at
epoch 460, best NLL 1.59957. Parent fitting failed its numerical KKT certificate
(1.18467e-5 > 2e-6); no learned parent or scientific closure result yet.
The corrected solver and [same-run recovery](feniks_reference_to_parent.md) reuse
banks and classifier, with no relaxed certificate and no posterior training.
This is execution progress, not validation of population recovery or ratios.

Run `avi_reference_to_parent_20260927_173822` is synchronized and inspected.
Seven available receipt artifacts and the frozen configuration verify; two
excluded NPZs remain remote. Qualification completed numerically and exited 2
on the scientific gate. The bank and blind population stages did not run.

| Block | Latest evidence / status |
|---|---|
| Coherent target / simulator / selection infrastructure | Retained for this synthetic benchmark; not real-sky validation |
| 15D reference / no q feedback / parent-selection algebra | Retained; diagnostic u is nonnegative and sums to one |
| Five physical marginal capacity checks | PASS, all improve; worst W1/IQR 0.0470 |
| Joint physical SW criterion | PASS: 0.04588 -> 0.02770, threshold 0.035; empirical comparator 0.01430 |
| Validation CDF qualification | Only failed capacity gate: 0.033833 > effective 0.032400; excess 0.001433 |
| Numerical capacity optimization | HiGHS Optimal; TRAIN CDF constraint active; not a training crash |
| SFH population conditional / new-reference photometric prediction | Unvalidated; final SFH contrast W1/IQR 0.958 |
| New reference banks | DONE 4/4 in exploratory continuation 184817; none ran in strict qualification 173822 |
| Photometric classifier on these banks | Training stopped on validation plateau, epoch 460; ratio audit still pending |
| Parent learned from fluxes with this reference | INCOMPLETE: numerical KKT failure after classifier; same-run recovery prepared |
| Final learned-parent 15D posterior / independent calibration | NOT RUN; earlier truth-supervised control evidence remains available |
| Production / paper claims | Not approved |

The effective CDF threshold differs from the prior run because its empirical
sampling comparator differs. Do not interpret the old fit's changed PASS/FAIL
as an improvement in that fit. No evidence yet identifies the new worst CDF
direction or proves the excess is only sampling noise.

**Next action, explicitly requested by the user and now implemented:** continue
directly to the reference-bank -> photometry-only parent -> report benchmark.
Use `--explore-cdf` on the completed failed run in a new root. The small
CDF-only excess is admitted for exploratory execution, not relabelled PASS;
all other capacity gates must pass and original thresholds remain unchanged.
No capacity rerun, extra pilot or posterior training. The same reserved bank
also evaluates the truth-assisted capacity fit, to separate reference predictive
limitations from difficulties in blind inversion without additional DSPS work.
This supersedes the earlier recommendation for two additional preliminary checks.
Code preparation and local smoke tests are not completed remote parent recovery.
See the [launch runbook](feniks_reference_to_parent.md) and
[result review](feniks_reference_to_parent_results_20260927.md).

After this run, inspect learned-parent physical/selected closure, selection
efficiency, flux CDFs/tails and classifier ratio diagnostics together. Only if
these are satisfactory, freeze the learned parent and generate fresh supervised
15D posterior training simulations. Final independent calibration remains open.

## Prior qualification and implemented continuation, 2026-09-27

Reference redesign `avi_reference_redesign_20260927_163901` is synchronized and
verified (42 available receipt artifacts; five NPZs intentionally absent).
Local 256 now passes all five physical marginal tolerances. Physical SW improves
0.1739 -> 0.0506, but the unchanged 0.035 gate and CDF 0.030 gate still fail
(CDF 0.0320). This is not proof that physical SW cannot be improved with the same
family: the old LP minimized a different objective in transformed coordinates.

| Block | Latest change / next validation |
|---|---|
| Reference normalization / 15D / no q feedback | Retained and tested; not a new population result |
| Finite linear-mass moment | PASS for affine/local geometry; alone did not fix finite-sample flux tails |
| Five physical marginal capacity checks | PASS for local 256 |
| Joint physical capacity + CDF | Still FAIL; one TRAIN-only physical-CDF objective implemented, thresholds unchanged |
| Paired observation replay | PASS at 4.66e-13; local 128 reduces perturbations; 32 targeted native anchors are already bright |
| SFH population conditional | Still unvalidated, last contrast W1/IQR 0.889; not equivalent to weak individual SFH constraints |
| Blind parent with redesigned reference | NOT RUN; automatically follows the next CPU qualification only on PASS |
| Final supervised 15D posterior | Deferred until parent closure; old overfitted bank not reused for training |
| Publication / real-sky claims | Not validated; development truth has influenced design, fresh final evaluation required |

The [new runbook](feniks_reference_to_parent.md) connects qualification directly
to compatible reference banks, one classifier/selection-corrected parent fit,
and a predictive report. No oracle retraining, new target catalogue or posterior
training. It is implemented and locally tested, not remotely executed yet.
See [the source-backed review](feniks_reference_redesign_results_20260927.md)
for interpretation and figures. This section supersedes older "awaiting redesign"
actions below; those are retained as dated history.

## Prior checklist: synchronized coherent inference, 2026-09-27

Evidence from the prior inference; reference-specific progress is updated above.
Run: `avi_coherent_inference_20260926_094557`, repaired snapshot contract
`f45de11a563a45ab02251f9ec8718681c3e275e3b8343821c2d99561c491de4c`.
Thirty available receipt artifacts verified locally; eight referenced large
artifacts and all bank contents intentionally unavailable locally.

| Block | Current evidence / status |
|---|---|
| Coherent target data + observation model | Completed coherent benchmark, no reuse of historical mismatched photometry; not a real-sky validation |
| Coordinates / flow transport | PASS on saved checks, Av artificial upper bound removed |
| Parent/selected normalization | PASS: both sums 1; saved u agrees with normalized v/alpha |
| Convex population solver | PASS KKT 7.87e-7; certifies optimization, not the scientific family |
| Full 15D / no q feedback | Retained; native SFH correlations present, reference conditional still an assumption |
| Classifier | Validation plateau; audit NLL 3.108 vs null 4.079, ratio-moment error 0.0125; conditional ratios not certified |
| Truth-supervised posterior control | Physical marginal coverage 69.0% / 95.1% on 1024 test objects; strong capacity evidence, not joint/conditional certification |
| Oracle optimization | Maximum epoch 200, still improving; do not label converged |
| Blind parent recovery | FAIL: physical SW 0.124 vs empirical validation/test 0.0141 |
| Learned-parent NPE on its own model | PARTIAL: marginal averages near nominal, PIT discrepancies and only 871 unique resampled evaluation rows |
| Learned-parent NPE on target | FAIL: physical 68% coverage average 80.5%, metallicity 91.1%; not the previous universal narrowing failure |
| Final NPE convergence | Overfits weighted bank after best epoch 29; plateau stop at 100 is no-new-best, not healthy flat validation |
| Observable predictive | Raw r-band W1 dominated by rare bright predictions; bulk disagreement also remains; physical cause not yet isolated |
| Real catalogue / paper production | NOT VALIDATED |

### Completed reference audit, 2026-09-27

Run `avi_coherent_reference_audit_20260927_151331` is now inspected: thirteen
available receipt artifacts and source hashes verify; stderr is empty. The
optimal TRAIN projected-CDF residual is **0.1231**, test **0.1262**, versus
validation/test **0.02015**. This is a demonstrated restriction of the current
fixed family on the chosen features, even with truth-provided fitting targets.
It is not a lower bound on SW; the truth-CDF optimum's SW need not improve.

The raw r-band flux distance falls **236.59 -> 8.77** under diagnostic upper
capping; predicted mass beyond target q99.9 is **0.909% versus 0.107%**. Most
raw distance is upper-tail sensitive, but residual/CDF disagreement persists.
No clipping is authorized. Individual tail causes remain unidentified.

**Checklist update:** capacity diagnosis COMPLETE / current family INADEQUATE;
bright-tail diagnosis COMPLETE / reference-tail remediation PENDING. Parent
recovery and final posterior remain FAIL; the previous supervised marginal
control remains encouraging. See the [complete review and next-run design](feniks_coherent_reference_results_20260927.md).

### Next actions, in order

1. **Implemented, awaiting Jean-Zay:** one bounded reference-redesign qualification, not
   another unchanged run. Address mass-coordinate tail amplification, local
   physical kernel/gate resolution and forced shared broad tails. Retain 15D
   native associations and no q feedback. Analytic TRAIN/validation capacity
   plus a small paired replay of influential bright rows gate new simulations.
   Four CPU cells (legacy, affine log-mass, local 128, local 256) and a parallel
   paired H100 replay (64 objects x 4 cases) are prepared. The launcher, restart
   and watcher are in [the runbook](feniks_reference_redesign.md). Local tests
   validate implementation invariants, not scientific success of new bases.
2. **After capacity/tail qualification:** rebuild a compatible reference bank,
   classifier and selection efficiencies, then fit the parent from photometry
   only. Reuse the coherent target dataset and oracle control. Do not silently
   reuse obsolete component labels or truth-fitted diagnostic weights.
3. **After parent predictive closure improves:** use new or adequately diversified
   learned-parent training pairs for final supervised NPE, instead of continuing
   the overfitted finite bank. Keep best-validation checkpoint selection.
4. Add independent joint and conditional calibration (including S/N/redshift
   strata), population predictive checks and repeatability before a paper run.
   These gates are not replaced by the physical marginal averages above.

No target catalogue resimulation, extra bands, oracle continuation, large
bootstrap sweep or blind final training is currently justified. New qualification
ceilings: preparation 20 CPU minutes; four capacity cells 25 CPU minutes each;
one H100 replay 15 minutes; CPU report 5 minutes. Times are allocation ceilings,
not predictions; queue time is extra. A pass permits a new-reference-bank
benchmark, not automatic production. Changing components invalidates old labels,
classifier ratios, selection efficiencies and weighted posterior-bank reuse.

See [detailed result review](feniks_coherent_inference_results_20260927.md).

## Historical evidence and decisions

### 2026-09-26 independent-reference support repair

The first coherent-inference reference job (217287) failed before simulation
or training: its Av logit imposed an artificial upper limit of 6, while native
proposals can exceed 6 (local eligible native maximum 7.5906). This is a
preparation/support bug, not a population-recovery result. The correction uses
an opt-in positive log Av coordinate, retaining all 15 dimensions and all rows.
Old representation/refinement coordinate specifications are unchanged.

The [same-root repair](feniks_coherent_inference.md#recovery-of-reference-job-217287)
archives old provenance and upgrades the frozen snapshot/config only before
any reference completion or downstream computation. The observed dataset is
unchanged; no new coherent target catalogue is required. Population and both
posterior calibration gates remain pending remote completion and inspection.

### 2026-09-26 parallel coherent inference implementation

The user-reported refinement lowered NLL but did not improve held-out physical
SW (0.0282 -> 0.0308). Do not make another oracle continuation a prerequisite.
The [new coherent inference DAG](feniks_coherent_inference.md) implements one
supervised posterior control in parallel with independent native-reference
simulations, followed by photometry-only selected/parent weights and a 15D NPE
trained from the weighted simulator bank. No q feedback or truth-trained prior.
It keeps reference physical-SFH associations, but its finite joint kernel family
and smoothing scale remain scientific assumptions. The same coherent catalogue
is reused; only the new independent reference bank needs DSPS simulation.
Remote population recovery, in-model and target-catalogue calibration remain
UNVALIDATED until this experiment is executed and its report inspected. The
old in-family classifier checks cannot certify these new reference ratios.

### 2026-09-26 bounded refinement implementation

The [refinement runbook](feniks_coherent_refinement.md) provides one physical-only
best-checkpoint continuation (60 additional epochs at 3e-6) and a parallel CPU
tail/zero-precision audit. SFH remains frozen, data and source run immutable.
Validation distribution milestones supplement NLL; test metrics never choose
checkpoints. Local optimizer/recovery, metrics and provenance tests pass; this
is ready for remote submission, not a remotely completed stage. It does not open the blind-population or final
posterior gates. No new dataset or bootstrap sweep is needed for these checks.

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

## Historical checklist and next actions (before coherent inference)

Superseded by the 2026-09-27 current checklist above; retained for provenance.

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
