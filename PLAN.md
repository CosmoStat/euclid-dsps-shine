# Plan

## 2026-08-24 Exact-cohort SMC bootstrap

- Status: implemented and locally validated; Jean-Zay curriculum remains
  unexecuted. Frozen pilot `1311562` completed successfully and
  isolated the remaining scientific blockers: RW-MH acceptance is healthy
  (median 0.232) and particles move, but the standard fallback reaches only
  median beta 0.265, leaving 77.1% hard; the real pathwise Gaussian-m5
  selection gradient remains non-finite while its score-function identity is
  finite.
- Add one bounded exact-cohort curriculum that runs an extended bridge only
  until it has enough beta=1 eligible objects for one q macro-distillation,
  then immediately reruns the frozen standard-budget E-step on the same fixed
  cohort. It must never update the parent prior or submit production.
- Add an explicit selection-gradient estimator contract. Preserve the scalar
  `+log(alpha_eta)` objective, allow an exact score-function gradient surrogate
  only when configured, and retain pathwise and finite-difference diagnostics.
- Keep r0, the canonical target, the conditional RealNVP, DSPS, Student-t2,
  Gaussian-m5 completeness, no-truth inputs, and all fail-closed gates
  unchanged.
- Add focused tests, cost accounting, immutable Jean-Zay launch/validation
  scripts, and leave the big run unauthorized until the post-distillation
  standard-budget pilot passes.
- Implemented a diagnostic-only 4-H100 curriculum with a baseline standard
  frozen E-step, a bounded K=128 / 48-stage exact cohort that stops after at
  least 32 eligible beta=1 objects, four stopped inclusive-q distillation
  steps, and a common-random-number standard-budget E-step plus q-only IS
  comparison after the update. The parent prior is never updated.
- Promoted the exact score identity for the selection gradient behind
  `gradient_estimator: score_function`: the returned loss value remains the
  same Monte-Carlo `log(alpha_eta)`, while its gradient is the stopped
  beta-weighted prior score. DSPS/Gaussian-m5 evaluation remains chunked and
  pathwise/score diagnostics are both retained.
- The receipt records pessimistic latent-object DSPS evaluation ceilings,
  exact-object yield, q gradient/clipping, SMC cross-entropy change, q-only IS
  ESS/max-weight change, and standard-budget beta/hard-fraction change. It
  cannot authorize a training smoke unless all three kernel, q and selection
  gates pass.
- Local verification: 91 focused adaptive-SMC, selection, canonical-target and
  exact-posterior tests pass; the score gradient matches pathwise and finite
  differences on shift-normal and active RealNVP directions. Ruff, compileall,
  shell syntax and diff checks pass. No remote job was submitted.

## 2026-08-24 Adaptive-SMC E-step isolation

- Status: implementation complete and locally validated; the frozen H100 pilot
  remains unexecuted. The latest immutable H100 smoke failed scientifically:
  median beta remained near 0.27, 81-94% of training objects remained hard,
  q-only IS and held-out SMC cross-entropy worsened, and the real Gaussian-m5
  pathwise selection gradient was non-finite. No big run is authorized.
- Replace the movement mean used by the hard gate with per-particle diagnostics
  (median squared epsilon displacement, moved fraction, unchanged fraction),
  while retaining the legacy mean for compatibility.
- Accumulate only eligible final-SMC objects before q distillation and prior
  M-steps. Do not update either network from the 2-6 surviving objects of one
  micro-batch; log the skipped tail explicitly.
- Report q-SMC clipping separately from sleep clipping and compare the real
  pathwise selection gradient with the diagnostic score-function identity.
- Add an E-step-only H100 pilot that loads the immutable bootstrap checkpoint,
  freezes q and the prior, and tests corrected initial RW scales before another
  training smoke. The pilot must not submit the big run.
- Implemented per-particle movement diagnostics and changed the hard gate to use
  the median squared epsilon displacement rather than the legacy mean. q and
  prior updates now accumulate eligible objects and skip undersized tails;
  q-SMC clipping is reported separately from sleep clipping.
- The real selection preflight now reports both the production pathwise gradient
  and a disabled score-function diagnostic on the same DSPS/Gaussian-m5 graph.
  The diagnostic does not alter `+log(alpha_eta)` or enable a partial update.
- Added a 4-H100 frozen E-step pilot launcher using the immutable bootstrap
  checkpoint and initial RW scales 0.30/0.15. It writes a diagnostic receipt and
  never updates q/prior or submits production.
- Local verification: 15 bridge-SMC tests, 49 adaptive/target/exact tests,
  19 selection tests and focused receipt/movement regressions pass; Ruff,
  compileall, shell syntax and diff checks pass. The repository-wide suite was
  stopped after 11% because of its local runtime and is not claimed complete.
- Frozen pilot jobs `1310704` and `1311364` failed before SMC because the
  checkpoint is genuinely mixed dtype: trained encoder parameters are float64
  while fixed calibration state remains float32. The loader now reads the
  ordered serialized array headers and aligns every template leaf to its exact
  on-disk dtype and shape; the pilot also enables JAX x64. A subprocess
  regression serializes and reloads this exact mixed-dtype case.

## 2026-08-22 Adaptive-SMC measured remediation

- Status: implementation complete and locally validated; a fresh Jean-Zay
  scientific smoke remains required. The completed smoke proved two independent
  blockers: both prior macro losses were finite but their joint gradients were
  non-finite (`rejection_code=1`), while q-coordinate mutation accepted only
  6-10% and left 34-41% of training objects hard after fallback.
- The Gaussian-m5 selection probability is now evaluated in one detached common
  flux unit. This preserves exactly
  `beta=Phi((fhat_r-f_limit)/sigma_m5(fhat_r))` while avoiding cgs-scale
  reverse-mode products. Rejected M-steps now report separate data, selection
  and trust gradient norms/finite flags and still apply no partial update.
- Epsilon-space RW scale is adapted per object between exact MH kernels toward
  acceptance 0.30, with bounded scales. The fallback retains K=128 and the hard
  gate but uses four moves after resampling and two final moves. Final scale,
  unique ancestry and squared epsilon displacement are persisted into training
  and exact-posterior diagnostics.
- The 96-object smoke now performs at least 128 fresh sleep optimizer updates:
  43 epochs x 3 batches = 129 updates, instead of 36. q uses a measured clip
  norm 20 while the prior keeps 5. The production schedule remains 12 epochs
  over the full selected training manifest.
- The fail-closed receipt now requires q-only ordinary-IS support
  (`ESS/K>=0.05`, median maximum weight <=0.80), SMC acceptance in [0.15, 0.60],
  non-degenerate ancestry/movement, non-permanent q clipping, a finite nonzero
  selection gradient and at least one accepted finite prior M-step. Equal final
  weights after SMC resampling alone can no longer pass the workflow.
- Local verification passes 66 core adaptive-SMC/selection/canonical/exact
  posterior tests, 26 supplemental SMC/data-parallel tests, the added fused
  selection-gradient regression, Ruff, compileall, shell syntax and diff
  checks. The more active
  fallback raises the absolute configured primary-plus-fallback ceiling from
  4,480 to 7,680 latent-object DSPS evaluations; the measured smoke rates, not
  this pessimistic ceiling, decide whether the 20-hour big job is viable.
- Publish this patch and run exactly one new smoke root. Do not submit or resume
  the big run until the new receipt is `PASS`; the previous failed roots remain
  diagnostic artifacts only.

## 2026-08-21 Adaptive-SMC scientific smoke gate

- Status: remote smoke completed the intended training and checkpoint path,
  then the fail-closed validator exited with code 3 because the scientific
  receipt was `FAIL`; this is no longer a software crash.
- Held-out validation reached `beta=1` with finite SMC cross-entropy, but only
  5/8 objects were eligible after fallback (`hard_fraction=0.375`). Mutation
  acceptance was only 0.061 and q-only IS remained collapsed
  (`ESS/K=0.0156255`, maximum weight 0.999983). Final equal SMC weights after
  resampling do not establish mixing or q calibration.
- Both prior macro-updates were rejected. Inspect `prior_macro_log.csv` before
  changing the trust region or selection estimator; the receipt's downstream
  prior checks are necessarily false when no update was applied and do not by
  themselves identify the rejection cause.
- Do not submit the big run. The next patch must address the measured q/SMC
  bootstrap and mutation deficiencies without weakening the hard-object gate,
  then rerun one fresh smoke.

## 2026-08-21 Adaptive-SMC checkpoint-mode recovery

- Status: implemented and locally validated. The fresh 12-epoch Jean-Zay smoke
  completed bootstrap and all three observed SMC batches, then failed while
  writing `best.eqx`:
  the shared checkpoint architecture summary rejected the dedicated
  `adaptive_smc_wake` objective mode. No training receipt was written, so this
  root remains incomplete even though the sampler itself ran.
- Registered the dedicated mode for config/checkpoint/inference metadata,
  recorded its adaptive-SMC block and inclusive-distillation semantics in
  sidecars, and explicitly rejected it in the legacy generic trainer so it
  cannot silently fall through to an ELBO objective.
- Added a regression that exercises the production config through objective
  normalization and architecture-summary construction. The expanded focused
  suite passes (`86 passed`), together with Ruff, compileall and diff checks.
  Publish this isolated fix and launch a new smoke root rather than reusing the
  checkpoint-sidecar-incomplete run.
- Do not weaken the scientific gate: the observed post-fallback hard fractions
  were 0.344, 0.375 and 0.406, all above the strict `<0.30` threshold. The
  corrected workflow must reach the receipt and report `PASS` or `FAIL` from
  the complete held-out validation metrics before any big job is submitted.

## 2026-08-21 Adaptive-SMC smoke replica-sharding recovery

- Status: implemented and locally validated. Jean-Zay smoke job `1254124`
  reached two observed SMC batches, then failed immediately after the first
  prior macro-update because
  the rebuilt model retained a replicated `NamedSharding(mesh='devices',
  P())`; the next `filter_pmap` expected its leading internal pmap axis to be
  sharded. This is a runtime replica-layout bug, not an SMC target/support
  failure.
- Materialize the post-M-step model leaves through host arrays before rebuilding
  the leading device axis. Add a two-device regression that performs q-SMC,
  prior M-step, replica refresh, then a second q-SMC step.
- Local regression and the complete focused suite pass (`58 passed`), together
  with Ruff, compileall and diff checks. Publish the fix, then launch a fresh
  smoke root; the failed root is diagnostic only and must not be resumed.
- Keep the production 12-epoch sleep bootstrap in the smoke. The failed run's
  two-epoch shortcut left q cross-entropy near 22 and post-fallback hard
  fractions at 0.375 and 0.344; retaining the full bootstrap tests the intended
  production state without weakening the hard-object gate.

## 2026-08-21 Adaptive bridge SMC production training

- Status: implementation complete and locally validated. Static-SNIS wake and
  the older x-space MALA SMC remain diagnostic ablations; the production
  trainer uses only the new exact-target adaptive bridge SMC. Jean-Zay smoke,
  big training and final exact validation remain unexecuted.
- Keep the canonical learned-prior posterior target and selection contract:
  object weights contain only likelihood, prior and full proposal density;
  `log(alpha_eta)` enters the averaged prior M-step once and remains fully
  differentiable through prior, DSPS and Gaussian PhotoErr.
- Add exact single-component conditional-RealNVP transport helpers between
  standard-normal epsilon and latent x. Implement a JAX-native adaptive bridge
  from the exact defensive r0 mixture to the canonical target, conditional-ESS
  bisection, systematic resampling, exact epsilon-space random-walk MH, logZ,
  fixed-budget histories and explicit hard-object output/fallback.
- Add a dedicated no-truth production trainer with an initial sleep bootstrap,
  observed adaptive-SMC sweeps, stopped inclusive q distillation, macro-batch
  prior updates, `log(alpha_eta)`, prior trust region, distinct q/prior
  optimizers and fixed observed validation cohorts.
- Use bounds/fit-initial-only standardized-logit coordinates and a broad
  identity-initialized joint RealNVP population prior. Do not reuse the
  truth-trained spline15d prior weights or its truth-standardized coordinates.
- Deliver separate 4-H100 smoke and big-run submissions so the big job is not
  submitted until the smoke receipt is `PASS`. The exact 32-object workflow
  adds adaptive SMC to q/raw-IS/defensive-IS/MAP/NUTS, central coverage,
  full-15D TARP/MIRA, covariance geometry and population closure.
- Cost accounting counts batched latent-object DSPS evaluations. Per object
  and observed sweep the configured primary costs 128 evaluations with no
  resampling, 256 with one resampling, and 384 with two. The absolute primary
  plus fallback configured upper bound is 4,480; the smoke must measure actual
  stage/resampling/hard rates before accepting the theoretical big-job budget.
- Local verification: Ruff, `compileall`, shell syntax, `git diff --check`, and
  58 focused tests pass. These cover 1D, correlated-2D, 15D and multimodal
  targets, exact nominal r0 sampling, adaptive beta, epsilon-MH invariance,
  transport/Jacobians, hard fallback, stopped q/prior losses, selection,
  target consistency, no-truth config, two-device pmap, checkpoint restore and
  exact-workflow compatibility. The FENIKS parquet and DSPS SSP assets are
  absent locally, so no real FENIKS scientific smoke result is claimed.

## 2026-08-21 Exact encoder diagnostic after support-gated wake failure

- Status: implementation complete and locally validated. The first Jean-Zay
  smoke preparation failed before generating data because its H100 wrapper did
  not re-enable the JAX CUDA plugin autoload; the dependent array therefore
  never started. Align the preparation environment with the already-correct
  exact wrapper, then relaunch the smoke in a fresh root. Full diagnostic
  remains unexecuted. Evaluate the final sleep-NPE encoder
  without pretending that the parent prior was updated or validated.
- Build one immutable 32-object benchmark containing 16 stratified observed
  FENIKS rows and 16 synthetic pairs drawn from the exact configured sleep
  joint with observed error covariates and selection applied after noise.
- Reuse the canonical exact-posterior workflow to compare raw q, q-only IS,
  defensive-mixture IS, MAP and NUTS under the same target and physical bounds.
- Aggregate support, covariance geometry, TARP and MIRA separately for the
  observed and sleep-synthetic domains. The final receipt is diagnostic only:
  it cannot promote a prior or mark the failed training run production-ready.
- Deliver separate smoke and full launch modes. The smoke uses four galaxies
  and short NUTS chains; the full run uses 32 galaxies and clean chains.
- Local verification: Ruff, compileall, shell syntax, diff checks and 51
  focused posterior-target, selection, TARP/MIRA and workflow tests pass. The
  local checkout has no FENIKS parquet/checkpoint assets, so the four-object
  Jean-Zay smoke remains mandatory before the 32-object array.

## 2026-08-21 Sleep-NPE negative-result handoff

- Status: remote production completed all 80 epochs but every defensive wake
  batch through the last scheduled wake was support-gated. Preserve the best
  encoder checkpoint as a diagnostic artifact; do not promote the run as a
  learned selection-corrected parent prior.
- Export a self-contained scientific handoff with immutable commit/run paths,
  objective and proposal definitions, observed metrics, conclusions that are
  already justified, unresolved hypotheses, and the exact-posterior analyses
  required before another training change.

## 2026-08-21 Jean-Zay prior-wake NaN recovery

- Status: implementation complete and locally validated (`ruff`, `compileall`,
  shell syntax, and 100 posterior/selection/workflow tests pass, including the
  two-device pmap NaN regression). Jean-Zay recovery smoke and production
  remain unexecuted. The interrupted run must not be continued because every
  observed epoch 25 wake update had non-finite prior gradients and was skipped.
- Root cause: the support gate masked the prior loss with `jnp.where` only after
  differentiating the selection-normalization graph. JAX can propagate
  `0 * NaN` from that inactive branch. Move both the differentiable prior
  density and `log(alpha_eta)` computation inside the accepted branch of
  `jax.lax.cond`; rejected batches now have exactly finite zero prior gradients
  and do not evaluate alpha.
- Add a real prior-to-DSPS-to-PhotoErr selection-gradient preflight before the
  optimizer and make the production finalizer fail if any gradient is
  non-finite, no wake prior update is actually applied, or alpha Monte Carlo
  relative error exceeds 15 percent.
- The first remote recovery smoke exposed a compile-time branch mismatch only
  when JAX x64 is enabled: the wake loss was float64 while selection metrics
  were float32. Cast every `lax.cond` output to the wake dtype and cover the
  exact x64 plus pmap combination in the multi-device regression test.
- Reduce the fixed common-random-number alpha bank from 4096 draws in 64-draw
  chunks to 1024 draws in 256-draw chunks. This changes 64 decoder chunks to 4;
  the final population diagnostic retains 8192 independent draws.
- Resume parameters and fixed feature statistics from the exact epoch 24
  checkpoint in a new output tree. The optimizer state is intentionally
  reinitialized. Run an epoch 25 wake plus epoch 26 sleep smoke before the
  dependent epoch 25-to-80 recovery job.

## 2026-08-21 Mass-covering sleep NPE and selection-corrected parent prior

- Status: implementation complete and locally validated; Jean-Zay smoke,
  production training and exact-posterior confirmation remain unexecuted. The
  completed 20k architecture battle is diagnostic only: all three posterior
  families collapsed to roughly one to three effective samples out of 2048
  with catastrophic Pareto tails. Do not run another architecture sweep; keep
  the established conditional RealNVP and change the training objective and
  target contract.
- Introduce one canonical learned-prior posterior target, including the common
  physical-bounds mask, fixed DSPS decoder, fixed calibration, robust
  likelihood and learned-prior density. Reuse it in wake/SMC, stored inference
  targets, MAP and the exact NUTS/MCLMC benchmark. Persist per-chain bounds
  audits.
- Train the encoder only with model-generated inclusive-KL sleep using observed
  catalog errors as fixed covariates. Apply the observed `lsst_r < 25` cut only
  after drawing noisy photometry. Select encoder checkpoints with held-out
  sleep NLL and log full-flow entropy diagnostics.
- Freeze the prior for the first 24 epochs. Thereafter alternate seven sleep
  epochs with one defensive wake epoch. Wake freezes the encoder and updates
  only the learned parent-population prior with stopped normalized weights from
  the exact mixture `0.50 q_T1 + 0.25 q_T2 + 0.15 q_T4 + 0.10 p_eta`.
- Keep normalized wake/IS weights exactly
  `softmax(loglike + logprior - logproposal)`. Correct the mean prior M-step as
  `-E_w[log p_eta(x)] + log(alpha_eta)`, with differentiable Gaussian-PhotoErr
  survey completeness. Never add `beta(x)` or `log(alpha_eta)` to per-object
  normalized particle weights.
- Fail closed on poor wake support: record ESS, maximum weight and weight
  entropy, exclude unsupported objects, and skip the whole prior update when
  the batch median ESS fraction is below the configured floor. A falling
  photometric objective alone is not a success criterion.
- Deliver one production config and one four-H100 training job, preceded by a
  tiny smoke. Use the second compute stage only for a stratified exact-posterior
  array comparing raw q, q-only IS, defensive IS, MAP and NUTS, including
  generalized covariance-ratio diagnostics.
- The production launcher builds immutable manifests from observed
  `flux_lsst_r` and errors only, submits a two-epoch four-H100 smoke, then one
  80-epoch four-H100 run. The exact launcher submits 32 one-galaxy H100 tasks
  with at most eight concurrent tasks and one CPU finalizer. Task zero also
  evaluates parent prior, beta-weighted forward-selected prior, aggregate q,
  parent truth and selected truth without adding a third compute stage.
- The finalizer fails closed on target support, NUTS convergence, raw-q and
  defensive-IS ESS/Pareto gates, generalized covariance coverage, full-15D
  TARP/MIRA agreement with NUTS, prior physical support, parent-population
  closure and forward-selected population closure. Truth enters only this
  synthetic closure stage.
- Local verification: `compileall`, Ruff, `bash -n`, `git diff --check`, and
  144 focused target, posterior, selection, exact-inference, MAP, config and
  workflow tests pass; two environment-dependent tests are skipped. A real
  local training smoke was not possible because the FENIKS train/test parquet
  files are not present in this checkout; the dependency-gated Jean-Zay smoke
  is therefore mandatory before production.

## 2026-07-22 Self-Supervised Learned-Prior Production Candidate

- Status: smoke training, inference, Jacobian Lens, and finalization completed;
  production startup cache-path fix locally verified and ready for a full-only
  dependent-chain relaunch.
- Limit the scientific array to three learned-prior candidates, all using the
  immutable common 15D `mixed_log_shifted_asinh` normalization and the same
  identity-initialized population RealNVP prior.
- Match the synthetic sleep corruption to the Student-t likelihood with two
  degrees of freedom and record robust noise diagnostics instead of relying on
  a variance that does not exist for Student-t2.
- Compare a single-component posterior, an exact two-component Gaussian-mixture
  base passed through the same conditional flow, and a likelihood-tempered
  SMC-wake variant intended to reduce self-normalized importance-weight
  collapse.
- Use 12 H100 concurrently through a three-task training array (four local
  H100 per `pmap` task). Do not request multi-node GPUs for one task because the
  current trainer has no multi-host JAX initialization.
- Run the existing Jacobian-lens workflow as a dependent 12-task array, four
  one-GPU shards per trained candidate, then finalize each lens and the common
  comparison report in a CPU dependency job.
- Require photometric posterior-predictive plots, corner plots, full 15D prior
  marginal/correlation diagnostics, importance-weight ESS, mixture occupancy,
  SMC acceptance/ESS, Jacobian spectra, and latent/prior score sensitivities
  before selecting a production model.
- Implemented exact two-component posterior densities, matched Student-t2
  sleep noise, stopped likelihood-tempered SMC-Wake with MALA diagnostics,
  mixture occupancy metrics, the 15-marginal/105-correlation closure gate and
  correlation-error heatmap.
- Added a chained three-run/four-GPU training array, twelve one-GPU Jacobian
  shards, per-run finalizers, fail-fast input/GPU/Matplotlib checks, and a
  documented artifact contract. The smoke exercises both sleep and wake plus
  all Jacobian paths before the full array is released.
- Fixed an XLA-only numerical defect in the sleep `m5` error model: computing a
  tiny cgs flux before rescaling underflowed under JIT/pmap even though eager
  tests passed. The depth flux is now formed directly in scaled units and the
  four-device regression applies finite sleep and SMC updates.
- The first Jean-Zay smoke (`40583`) confirmed finite two-epoch training and
  checkpoint creation for all three candidates. Both RWS tasks completed, but
  SMC-Wake timed out only after training, during inference with the unchanged
  production budget of 16,384 prior samples. Smoke inference now uses 512
  prior samples while production retains 16,384; this still exercises two
  prior-predictive DSPS batches without spending the smoke wall time on final
  statistical precision.
- The second smoke training array (`41879`) completed all three candidates in
  2:57, 3:00, and 4:26. Its twelve Jacobian tasks exited immediately because
  the wrapper used `test -s` on the intentionally zero-byte `DONE` marker
  created by `touch`. The lens prerequisite now checks file existence with
  `test -f`, covered by a wrapper regression test; no model or Jacobian code
  executed in the failed tasks.
- The corrected smoke Lens array (`43316`) and finalizers (`43317`) completed
  all tasks. The first full array (`43318`) then failed before Conda/Python
  because its scheduler-provided `$JOBSCRATCH` paths were not writable. All
  three production wrappers now place their isolated Matplotlib cache directly
  under node-local `/tmp`, and a regression test rejects future `JOBSCRATCH`
  use in this workflow. The failed full root contains only empty candidate
  directories and can be reused safely.
- Verification: 46 focused posterior, likelihood, SMC, truth/gate, Jacobian and
  config tests pass; all three wake variants and sleep compile and update under
  simulated four-device `pmap`, and every architecture survives checkpoint
  serialization/reload. Compileall, Ruff, shell syntax, CLI help, a complete
  mocked six-job submission and diff checks pass. The production
  parquet catalog is absent from WSL, so its JAX-free contract check remains a
  mandatory first step in the Jean-Zay submission script.

## 2026-07-21 Self-Supervised RWS Prior and Posterior Matrix

- Status: completed; ready for the Jean-Zay smoke-plus-full submission.
- Preserve the production checkpoint-backed 15D
  `mixed_log_shifted_asinh` normalization for every prior and posterior.
- Match the synthetic closure likelihood to the generator with Gaussian
  `fluxerr`, zero added fractional floor, and zero jitter.
- Replace invalid-model masking with particle rejection and stabilize extreme
  spline-SFH exponentiation before DSPS evaluation.
- Add a one-loop reweighted wake-sleep objective: model-generated physical
  sleep updates train only the encoder; real-data wake updates train the
  encoder and learned prior from the same stopped importance weights.
- Add four H100 controls covering frozen prior, learned weighted-wake prior,
  sleep 3:1 with K=4, and sleep 3:1 with K=8, including a JAX-free fail-fast
  preflight, smoke dependency, inference plots, and locked comparison report.
- Keep production validation at an eight-epoch cadence with snapshots and JAX
  preallocation disabled; force both compiled paths through the ten-minute
  smoke before releasing the full array.
- Verification: 43 focused config, likelihood, posterior-flow, RWS-gradient,
  noise-contract, and spline tests pass on CPU; Compileall, Ruff, shell syntax,
  diff checks, and the new Sphinx page pass. The production catalog is absent
  from WSL, so the JAX-free validator runs on Jean-Zay before either array is
  submitted.

## 2026-07-20 Joint-Latent Prior Diagram Correction

- Status: completed.
- Correct the target and per-experiment architecture diagrams so the learned
  prior is shown as a population model fitted to latent samples inferred by
  the posterior from the photometric catalog, not as a downstream likelihood
  output.
- Distinguish simultaneous ELBO gradients from stopped-gradient VEM M-steps.
- Added separate object-reconstruction and latent-population lanes, the joint
  ELBO and VEM M-step objectives, and an explicit warning that the learned
  density is the selected-catalog population unless selection is modeled.

## 2026-07-20 Joint-Prior Array Explorer and Scientific Interpretation

- Status: completed.
- Add the completed six-run independent-posterior/learned-prior array to the
  standalone explorer with exact metrics, configs, corners, diagnostics, and
  artifact provenance.
- Give every experiment an explicit scientific role: sanity check, ablation,
  supervised upper-bound diagnostic, or viable photometry-only final-model
  candidate.
- Explain the identifiability problem when both population prior and posterior
  are learned without true latent distributions, and separate observable-space
  fit from latent-space validation available only in synthetic closure tests.
- Add a dedicated architecture diagram and training schedule for each run,
  including gradient paths, frozen/trainable components, truth usage, and the
  reason each comparison exists.
- Regenerate and validate the standalone HTML at desktop and mobile widths.
- Added all six completed joint-prior runs to the common A/B explorer, bringing
  the catalog to 15 runs. Each new entry embeds the exact corner, training,
  residual, photo-z, normalized config, prior-population metrics, and compact
  artifact inventory from the July 19 full array.
- Added a dedicated joint-prior science tab with the deployable photometry-only
  target architecture, an identifiability explanation, observable-only
  validation criteria, aggregate coverage/speed plots, and one gradient/schedule
  diagram for every experiment.
- Classified the frozen-prior run as a failed reference sanity check, the
  simultaneous/VEM runs as currently failed but methodologically deployable
  candidates, and the hybrid/oracle runs as synthetic-only capacity/plumbing
  diagnostics. Corrected the historical RQ-spline label to the serialized
  frozen RealNVP provenance.
- The report states the actual result rather than selecting a false winner:
  every photometry-only checkpoint fails posterior calibration; supervised NPE
  repairs q but not p; even the prior-truth oracle leaves a failed learned
  prior, so common normalization and prior-loss plumbing must be fixed before
  learned-prior conclusions are trusted.
- Regenerated the 108.2 MB standalone HTML and 110.1 MB payload. Compile, Ruff,
  diff checks, and a memory-limited Playwright pass succeed: 16 tabs, six
  experiment cards, 15 comparator options, nonblank aggregate figures, no body
  overflow at 390 px, and zero JavaScript errors.

## 2026-07-20 FENIKS Common-15D and Mode-Covering Posterior Control

- Status: completed; ready for the Jean-Zay smoke-plus-full submission.
- Extract one immutable spline-15D mixed marginal-normalization contract from
  the synthetic training split and use it unchanged for every posterior and
  prior variant. Keep this coordinate transform separate from the frozen
  reference RealNVP prior and from every learned RealNVP parameter.
- Fail fast before GPU allocation unless training, inference, checkpoint
  reload, prior sampling, and DSPS decoding resolve the same parameter order,
  bounds, transform families, locations, scales, and normalization hash.
- Use the common 15D coordinates in every new control so comparisons against
  the frozen-reference and learned-prior histories are no longer confounded by
  latent geometry.
- Limit the next full array to four unsupervised training controls, reusing the
  completed one-sample frozen-reference and identity-normalized VEM 4:1 runs
  as the zero-cost references:
  1. `common15d_vem4_elbo_k1`: learned RealNVP prior, common 15D transform,
     VEM 4:1, and the unchanged one-sample reverse-KL ELBO; this isolates the
     normalization correction from the old `ind_vem4` result.
  2. `frozen_ref_elbo_k2_antithetic`: known-good frozen JAX-COSMO reference
     prior and two antithetic posterior samples; this measures whether estimator
     variance, rather than KL direction, explains the narrow old frozen-prior
     posterior.
  3. `frozen_ref_periodic_wake_k4`: known-good frozen JAX-COSMO reference prior,
     ordinary
     one-sample ELBO for three epochs, then one four-particle tempered
     importance-weighted wake update; this isolates a mass-covering encoder
     update without simultaneously changing the prior.
  4. `common15d_vem4_periodic_wake_k4`: learned RealNVP prior with the common
     15D transform, VEM 4:1, and the same periodic four-particle wake update;
     this tests the complete unsupervised learned-prior candidate while the
     first three tasks retain enough controls to interpret its outcome.
- Record importance-weight ESS for both wake runs and reject conclusions from
  collapsed weights. Interpret the combined learned-prior plus wake run only
  after checking the common-normalization prior control and frozen-prior wake
  control independently.
- Correct the historical provenance in the new report: the checkpoint used by
  old `ind_frozen_rqspline` is actually a 12-layer RealNVP according to its
  serialized sidecar. Treat it as `frozen_ref_realnvp`; do not present it as an
  RQ-spline or silently switch to the separate dequantized RQ checkpoint, which
  would change both prior family and normalization.
- Keep the completed supervised NPE hybrid as an evaluation reference only;
  do not use truth parameters in any loss or proposal in this array. Truth may
  be read after training solely for held-out coverage, PIT, and recovery plots.
- Select a run only if both levels pass: the learned prior matches held-out
  population marginals/correlations without boundary saturation, and the
  conditional posterior passes per-object coverage, PIT, photo-z, and
  posterior-predictive diagnostics. Matching only the aggregate posterior is
  insufficient.
- Preserve the four-H100 data-parallel path and report decoder evaluations per
  object, seconds per encoder epoch, skipped updates, and peak memory so the
  mode-covering correction has an explicit speed cost.
- Implemented exact antithetic posterior sampling, tempered proposal densities,
  periodic self-normalized wake updates with stopped importance weights, and
  VEM-aware encoder-epoch scheduling. Wake steps update only the encoder;
  calibration and prior parameters are frozen, while prior M-steps continue to
  fit stopped posterior samples.
- Added checkpoint-time latent hashes and strict reload validation. All four
  configs resolve the same `spline15d_mixed` hash
  `48fe36f64913880149fde24603d75fb8219659cd8f21598aa7b76cd0a22c5a1b`.
- Added a fail-fast catalog/checkpoint/config validator, a ten-minute four-task
  smoke array followed by one complete `afterok` array, compact per-task and
  aggregate reports, ESS diagnostics, required corner/residual plots, and
  observed-versus-posterior photometry panels without writing the large raw
  posterior-predictive tables.
- Verification passes: 57 config/latent tests, four focused antithetic/wake
  tests, seven ELBO/prior tests, historical checkpoint reload, exact four-config
  hash resolution, compact photometry plot generation on a completed run,
  Compileall, Ruff, shell syntax, and `git diff --check`. A local production
  smoke was not run because the Jean-Zay catalog is not present in this WSL
  checkout; the submission preflight verifies it before `sbatch`.
- Jean-Zay smoke array `2127215` failed identically in all four tasks before
  training: the standalone validator imported `amortized.train`, whose model
  bootstrap disabled PJRT plugin discovery before the validator materialized a
  JAX array under `JAX_PLATFORMS=cuda`. Reimplemented the exact float32 latent
  hash in NumPy/JSON so the preflight is JAX-free; a fresh-process regression
  test and full temporary-catalog validation pass without importing `jax`.

## 2026-07-19 FENIKS Production Run Explorer

- Status: completed.
- Inventory the completed JAX-COSMO conditional-posterior production runs and
  preserve exact provenance across the recovered AVI and completed NPE roots.
- Extend the standalone forward-model report with a plain-language method and
  training-flow guide, exact per-run configuration and artifact inventories,
  embedded corner and diagnostic plots, and explicit scientific caveats.
- Add a two-run A/B comparator with numerical deltas, configuration
  differences, side-by-side plots, and metric interpretation.
- Regenerate the standalone HTML and payload, then validate navigation,
  rendering, responsive layout, and payload completeness.
- Added a plain-language French method guide that separates the population
  prior from the conditional posterior, explains AVI versus NPE, and documents
  the four posterior families and the role of every acceptance metric.
- Added an A/B run comparator covering the eight controlled production-matrix
  runs plus the separate learned-RQ-prior lineage. It embeds 37 exact local
  figures, all normalized config inputs, numerical deltas, provenance warnings,
  and compact artifact inventories with checkpoint/shard aggregation.
- Regenerated the 65.5 MB standalone report and 67.1 MB JSON payload. Compile,
  Ruff, diff checks, nine-run payload assertions, and Playwright navigation at
  desktop/mobile widths pass with zero JavaScript errors; all 15 tabs keep
  exactly one visible panel and embedded corners decode at full resolution.

## 2026-07-17 JAX-COSMO Prior-to-Inference Result Audit

- Status: in progress.
- Verify completeness and provenance across the learned RealNVP prior,
  amortized encoder training, and held-out 5,000-object inference run.
- Quantify prior fidelity, train/validation convergence, posterior recovery and
  calibration, photo-z performance, posterior-predictive residuals, parameter
  boundary behavior, and failure modes from the serialized tables.
- Produce a concise scientific assessment with explicit pass/fail conclusions
  and the highest-priority follow-up before comparing against the running
  dequantized RQ-spline pipeline.

## 2026-07-17 Continuous-Atom JAX-COSMO RQ-Spline Pipeline

- Status: completed.
- Build an isolated grouped spline-15D dataset whose exact-zero SFH contrasts
  are physically dequantized with uniform `+/-1e-3 dex` noise while retaining
  exact truth files for diagnostics.
- Generalize the production spline-15D prior trainer, checkpoint format, frozen
  amortized-prior loader, snapshots, and final diagnostics from RealNVP-only to
  both RealNVP and rational-quadratic spline coupling flows.
- Add a fully versioned Jean-Zay `afterok` chain from grouped dataset projection
  through RQ-spline prior training, four-H100 encoder/decoder training, and
  held-out inference, without overwriting the active RealNVP baseline.
- Validate with RQ checkpoint roundtrips, finite gradients, config resolution,
  shell checks, and a real-parquet end-to-end smoke including diagnostics.
- Verification completed on 64 rows per split: all 38 exact-zero atoms were
  removed only from the projected tables, the two-epoch RQ prior smoke wrote a
  reloadable best checkpoint and complete diagnostics, and the frozen
  amortized loader recovered the 15D RQ prior with the matching latent order.

## 2026-07-17 Amortized Warm-Restart Support

- Status: completed.
- Added explicit `--initial-checkpoint` and `--start-epoch` options to
  amortized training and exposed them through the H100 Slurm wrapper.
- The continuation uses the serialized model but intentionally initializes a
  fresh AdamW state; existing checkpoints do not contain optimizer state.
- Continuations write to a new run directory and retain absolute epoch numbers,
  so KL and objective schedules continue at the requested epoch without
  overwriting the interrupted run.

## 2026-07-16 JAX-COSMO 15D Normalization Regeneration

- Status: completed.
- Fitted the mixed marginal transforms directly on the new 40,000-row
  JAX-COSMO cubic-spline training projection and applied them unchanged to the
  5,000 held-out test rows.
- Regenerated the full 15-coordinate before/after distribution figure, with
  train/test overlays, standard-normal references, transform families, robust
  shifted-asinh parameters, and held-out Gaussian quantile errors.
- Serialized the exact parameters to `normalization_parameters.csv` and the
  complete contract to `normalization.json`; the held-out forward/inverse
  roundtrip maximum absolute error is `3.553e-15`.
- Rewired both the `Normalization` and `Results` tabs to these current
  JAX-COSMO artifacts. Historical PCHIP flow and encoder curves remain clearly
  marked as comparison results pending retraining.

## 2026-07-16 JAX-COSMO 15D Distribution Regeneration

- Status: completed.
- Confirmed that the report previously contained no raw 15D distribution plot
  from the new spline: only the generic JAX-COSMO K scan was current, while the
  embedded contrast/correlation figures still came from the PCHIP projection.
- Reran the complete 15D analysis with the JAX-COSMO cubic not-a-knot decoder
  under `outputs/analysis/feniks_jax_cosmo_spline_15d_prior_20260716/`.
- Reoptimized the shared 11-node placement on train/validation. The selected
  normalized nodes are `[0, 0.21580, 0.33724, 0.46995, 0.58117, 0.71003,
  0.80182, 0.87237, 0.92358, 0.96582, 1]`.
- Projected all 40,000 train and 5,000 test galaxies and regenerated the ten
  contrast distributions, full 15D correlation, node-placement examples,
  closure comparison, parquet projections, dequantization scan, metrics,
  contract, payload, and HTML/Markdown report.
- Visually inspected the new contrast-distribution and correlation figures;
  both are nonblank, legible, and consistent with the new latent ordering.
- Switched the explorer's default 15D payload to the JAX-COSMO directory. The
  `15D latent` tab and the first raw-distribution figure in `Results` now embed
  the new plots; downstream flow/normalization/encoder plots remain explicitly
  labeled as historical PCHIP artifacts pending retraining.
- Regenerated the standalone explorer. Ruff, compileall, `git diff --check`,
  and jsdom contract checks pass with zero JavaScript errors; the embedded
  contract reports JAX-COSMO `k=3` not-a-knot and the Results raw figure is
  byte-identical to the new latent-tab figure.

## 2026-07-16 FENIKS Explorer Spline and Normalization Detail

- Status: completed.
- Expanded the JAX-COSMO spline tab with the exact normalized-node to cosmic
  time map, the ten-contrast to eleven-height cumulative reconstruction, the
  interval cubic polynomial, C1/C2 knot continuity, not-a-knot endpoint
  conditions, log-time/log-SFR evaluation, and DSPS surviving-mass amplitude
  recovery.
- Kept the interactive native-versus-spline SFH, age-weight, photometry, node,
  and closure diagnostics after the method explanation.
- Added a dedicated Normalization tab showing the full physical-to-flow path,
  exact train-only shifted-asinh and positive-log formulas, analytic inverses,
  support semantics, numerical floors, non-goals, roundtrip requirements, and
  checkpoint contract.
- Added the complete 15-row serialized transform table, including family,
  location, train q16/q84, lambda, center, and scale.
- Preserved the prominent distinction between the historical PCHIP checkpoint
  parameters/results and the pending JAX-COSMO retraining.
- Regenerated the standalone report. Ruff, compileall, `git diff --check`, and
  jsdom checks pass: zero JavaScript errors, exactly one visible panel across
  all 13 tabs, 15 normalization rows, a non-empty embedded normalization image,
  and five spline formulas.

## 2026-07-16 FENIKS Explorer JAX-COSMO Presentation Audit

- Status: completed; JAX-COSMO method presentation is current, end-to-end
  JAX-COSMO training results remain pending.
- Verified that the active report forward uses
  `jax_cosmo.scipy.interpolate.InterpolatedUnivariateSpline` with `k=3` and
  the not-a-knot endpoint condition in log cosmic time and log SFR.
- Switched the embedded generic K scan from the July 10 PCHIP artifact to the
  July 16 JAX-COSMO held-out scan.
- Found that the optimized 11-node projection, epoch-645 RealNVP, and 80-epoch
  amortized encoder artifacts still derive from the serialized version-1
  `pchip_log_sfr_contrasts` dataset contract.
- Added prominent overview, latent-contract, and Results notices separating
  the current JAX-COSMO method from historical PCHIP training evidence. The
  report now explicitly forbids presenting those losses and distributions as
  final JAX-COSMO performance.
- Regenerated the standalone report. Ruff, compileall, `git diff --check`, and
  focused jsdom navigation/content checks pass with zero JavaScript errors.
- Required production sequence before a final JAX-COSMO results presentation:
  recompute the optimized 15D projection, retrain the flow prior, retrain the
  frozen-prior encoder, and rerun corrected held-out inference.

## 2026-07-16 FENIKS Explorer Navigation Regression Fix

- Status: completed.
- Reproduced navigation across all 12 tabs and found that a broad translation
  replacement had corrupted CSS/SVG keyword `none` into invalid `noe` values.
  Inactive stage panels therefore remained visible even though their active
  classes changed correctly.
- Restored every affected CSS, JavaScript, and SVG keyword, including
  `display`, `pointer-events`, list styling, path fills, and
  `non-scaling-stroke`.
- Made tab visibility redundant and robust: `setStage` now updates both the
  active class and the native `hidden` attribute, plus `aria-selected`; CSS
  enforces hidden panels with `display: none !important`.
- Raised and isolated the sticky navigation layer, constrained the original
  forward graph to its scroll container, and made its SVG responsive within a
  readable minimum width.
- Added a build-time navigation-contract check so report regeneration fails if
  the critical panel-hiding rules disappear.
- Regenerated the standalone HTML. Ruff, compileall, `git diff --check`, and a
  jsdom regression over all 12 tabs pass with zero JavaScript errors and
  exactly one visible stage after every click.

## 2026-07-16 Spline-15D JAX-COSMO Cubic Migration

- Status: completed locally; production amortized retraining and held-out
  inference rerun remain to be launched on Jean-Zay.
- Replace the production PCHIP SFH decoder with
  `jax_cosmo.scipy.interpolate.InterpolatedUnivariateSpline`, using cubic
  `not-a-knot` interpolation in log cosmic time and log SFR.
- Preserve the existing 11-node/10-contrast latent and stellar-mass
  normalization contracts while versioning the changed interpolation model.
- Add focused knot, JIT, gradient, SciPy-equivalence, and non-finite tests.
- Add a reproducible held-out benchmark comparing the legacy PCHIP and new
  JAX-COSMO decoder through SFH and full 18-band DSPS closure metrics.
- Replaced the production decoder and both spline-selection analysis paths with
  the JAX-COSMO degree-3 `not-a-knot` interpolator. The 11-node/10-contrast
  latent order and mass normalization are unchanged.
- Added `jax-cosmo` plus its required `setuptools<81` compatibility bound,
  versioned new projection contracts as v2, and documented the implementation,
  degree, and endpoint condition.
- Added SciPy-equivalence, eager/JIT agreement, knot interpolation, positivity,
  and finite-gradient tests. Ruff, compileall, `git diff --check`, and 24 focused
  spline/amortized-latent tests pass.
- Recomputed the full 388-object balanced held-out node scan under
  `outputs/analysis/feniks_jax_cosmo_spline_node_scan_20260716/`. No scanned
  generic grid passes every worst-group gate; `uniform_log_time, K=20` is the
  least-bad generic scan point. This does not supersede the active optimized
  11-node placement, which was not one of the three generic scan grids.
- Added an isolated production chain using
  `feniks_260617_spline15d_grouped_jaxcosmo_v1`: grouped projection, 800-epoch
  positive-support prior training with snapshots and final diagnostics,
  four-H100 frozen-prior amortized training, and held-out inference. The submit
  helper wires every stage with `afterok` and refuses existing outputs.
- Prior training now rejects legacy or mislabeled spline datasets unless the
  contract is v2 with the exact JAX-COSMO cubic type. A 16-row projection plus
  two-epoch RealNVP smoke completed with the new contract and diagnostics path.

## 2026-07-16 FENIKS Forward Explorer English Spline-15D Results Expansion

- Status: completed.
- Translate the complete standalone forward-model explorer to English and make
  the opening pipeline diagram easier to read.
- Add a before/after pipeline view showing how the spline-SFH representation
  bypasses the Diffstar parameterization and reduces inference to a 15D latent
  space feeding the DSPS-only forward path.
- Document the shape-preserving spline contract, serialized knot parameters,
  stellar-mass normalization, and the reduction from spline knots to ten SFH
  contrast coordinates.
- Add a results section grounded in local artifacts: raw 15D distributions,
  failed-flow example, shifted-asinh normalization and per-column parameters,
  full prior-flow convergence, atom/Dirac limitations, and normalized/physical
  prior recovery.
- Add frozen-prior encoder/decoder training convergence and KL diagnostics,
  clearly separating valid training evidence from invalidated pre-fix physical
  inference diagnostics.
- Regenerate the standalone HTML, run source checks, and inspect desktop and
  mobile screenshots before marking the phase complete.
- Translated all static and dynamically rendered report text to English,
  including parameter metadata, tooltips, tables, audit notices, and plot
  labels.
- Added a first-viewport before/after comparison: the original 18D
  Diffmah/Diffstar/DSPS chain versus the reduced 15D spline/DSPS chain.
- Added the exact 11-knot PCHIP and ten adjacent log-SFR contrast contract,
  mass normalization, inverse reconstruction, and serialized marginal
  normalization table.
- Added an artifact-backed Results tab containing raw target distributions, a
  failed-flow example, shifted-asinh before/after normalization, the complete
  epoch 0--645 prior trajectory, epoch-645 normalized/physical recovery,
  atom-aware next steps, and the 80-epoch frozen-prior encoder/KL history.
- Regenerated the 7.9 MB standalone report and its JSON payload. Ruff,
  compileall, `git diff --check`, and a jsdom execution check pass; jsdom
  reports zero JavaScript errors, 15 normalization rows, two rendered SVG
  charts, and five non-empty embedded Results images.
- Playwright screenshot inspection is blocked by missing host libraries
  (`libnspr4`, `libnss3`, and `libasound2t64`), not by the report. The report
  remains standalone and opens directly from disk.

## 2026-07-16 Spline-15D End-to-End Result Audit

- Status: completed from the locally retrieved supervised-prior, amortized
  training, and held-out inference artifacts; corrected inference rerun pending.
- Verified that the epoch-645 RealNVP and the frozen prior serialized inside the
  amortized checkpoint are bitwise identical across all 144 parameter leaves.
- Found a blocking inference-contract bug: training resolved the exact
  checkpoint-backed `spline15d_mixed` transform, while inference used the YAML
  placeholder `identity` transform. This invalidated every physical posterior,
  prior-predictive, derived-SFH, and DSPS diagnostic in the retrieved inference
  directory.
- Fixed inference to use `_latent_spec_for_amortized_config`, matching training.
  Focused inference/prior tests pass (`10 passed`), along with Ruff, compileall,
  and `git diff --check`.
- Recovered the original latent coordinates from the legacy physical outputs
  and regenerated latent-only diagnostics without rerunning Jean-Zay. The
  corrected prior remains faithful (median/max KS `0.067/0.116`), but the
  encoder posterior remains strongly under-dispersed (median 68/95 percent
  coverage `0.127/0.246`).
- The corrected encoder recovers stellar-mass and metallicity ordering
  (`r=0.867/0.940`) but only weakly recovers redshift (`r=0.413`) and does not
  recover individual SFH spline contrasts (most correlations near zero).
- Training itself was finite with 1,440/1,440 updates and exactly zero prior
  gradient, but every raw encoder gradient exceeded the clipping threshold;
  validation likelihood was still improving at epoch 80 and the posterior
  median log-standard-deviation had contracted to about `-2.16`.
- The old held-out photometric residuals and photo-z metrics must not be used:
  they were computed by DSPS from incorrectly decoded physical parameters.
  Rerun inference only after syncing the transform fix; retraining is not
  required to obtain a valid first evaluation of the existing encoder.
- Reproducible corrected latent audit and plots are under
  `outputs/reports/feniks_spline15d_end_to_end_audit_20260716/`.
- Consolidated inference diagnostics to one canonical 15D
  `truth / prior / aggregate posterior samples` corner. Removed the posterior
  median, prior-only, posterior-only, pairwise, and reduced MAP corner variants.
- The canonical corner now samples sharded posterior parquet outputs directly;
  it no longer silently falls back to the narrower distribution of per-object
  posterior medians when `combine_sample_shards=false`.
- Added `effective_latent_spec.json` to every inference run, recording the exact
  names, ordering, normalization, marginal transforms, scales, and prior
  checkpoint actually used by the encoder, flow, and DSPS decoder.

## 2026-07-15 Spline-15D Amortized DSPS Inference

- Status: implementation completed; Jean-Zay launch pending completion of the
  spline-15D prior continuation.
- Add a differentiable JAX path from the learned spline-15D flow coordinates
  through the mixed marginal inverse, spline SFH reconstruction, stellar-mass
  normalization, and the fixed DSPS photometric forward model.
- Load the completed spline-15D RealNVP as a frozen amortized prior, with strict
  checkpoint, parameter-order, normalization, and dataset-contract checks.
- Train the photometric encoder on the existing grouped Diffsky/FENIKS source
  catalog; join spline truths by `object_id` only for closure diagnostics.
- Add a dedicated 18-band configuration, H100 launcher, smoke tests,
  documentation, and a Slurm `afterok` launch command tied to the current prior
  job.
- Implemented the exact checkpoint transform contract, including positive log
  coordinates and shifted-asinh coordinates, without whitening or atom noise.
- Implemented the JAX spline-SFH DSPS decoder and a non-destructive catalog join
  that combines the existing 18-band photometry with exact spline truths.
- Added the frozen-prior configuration and H100 launcher. Local validation
  passed: Ruff, compileall, shell syntax, 18 focused tests, and an eight-object
  encoder-to-DSPS training smoke with zero prior gradient.
- The first single-H100 production epoch required about 652 seconds with an
  effective JAX batch of 128 and two Monte Carlo samples. Exposed the existing
  pmap path in the spline launcher so the retry can use four H100s, a global
  batch of 1024, one Monte Carlo sample, and less frequent validation.
- Made newline-based per-batch logging the launcher default on Slurm so `.out`
  files show loss, likelihood, KL, gradient norms, and update status without
  carriage-return progress-bar artifacts.
- The four-H100 retry completed 80 epochs with all 18 updates applied in the
  final epoch. Added an explicit frozen-flow checkpoint override to the
  held-out inference launcher so deserialization uses the same epoch-645 prior
  template as training.
- The first held-out inference completed all 5,000 galaxy shards but exhausted
  one H100 while deriving DSPS quantities for all 8,192 prior samples at once.
  Reused `prior_predictive_batch_size` to chunk those derived forwards, while
  retaining resumable shard discovery so galaxy inference is not repeated.

## 2026-07-15 Spline-15D V6 Positive-Support RealNVP

- Status: implementation completed; production launch pending on Jean-Zay.
- Build one minimal production run from exact atoms with no dequantization and
  no joint whitening.
- Preserve the V5-A RealNVP architecture and optimizer, but use fixed-support
  log transforms for strictly positive `z_obs` and `dust_av`; keep robust
  shifted-asinh marginals for the remaining 13 dimensions.
- Train for 400 epochs with validation-NLL selection and the existing physical,
  normalized, dependence, and inverse-tail snapshots every five epochs.
- Deliver a backward-compatible mixed normalization contract, focused tests,
  one H100 launcher, documentation, and exact Jean-Zay commands.
- Added normalization contract version 3: standardized log transforms for
  strictly positive `z_obs` and `dust_av`, with an explicit error for nonpositive
  inputs, plus robust shifted-asinh for the other 13 coordinates.
- Added
  `configs/prior_feniks_spline15d_realnvp_v6_positive_support.yaml` and the
  single-GPU launcher `scripts/feniks_spline15d_v6_positive_support_h100.slurm`.
- Local verification passed: Ruff, compileall, 11 focused tests, shell syntax,
  and a one-epoch real-parquet end-to-end smoke. Its serialized transform
  roundtrip error is exactly zero and all 256 sampled `z_obs`/`dust_av` values
  are strictly positive.
- The first Jean-Zay run reached epoch 160, then stopped while saving the
  auxiliary snapshot because its float32 forward/inverse maximum error was
  `0.0119`, just above the old absolute failure threshold `0.01`.
- Added continuation from a serialized checkpoint with strict architecture and
  normalization-contract checks. The original Adam state was not serialized,
  so continuation explicitly reinitializes it and uses a lower `1e-5` learning
  rate from the valid epoch-155 checkpoint.
- Kept the integrity warning at `1e-3` and finite/structural/amplitude checks,
  but set the spline-15D hard roundtrip threshold to `0.05`, serialized per
  checkpoint. Resume smoke from epoch 1 to 2 passed end to end.
- The epoch-155 continuation completed at epoch 400, but validation NLL still
  improved at the final epoch (`-12.8428`, best at 400). Added the launcher
  override `TARGET_EPOCHS` for a second continuation from 400 to 800 at a
  reduced `5e-6` learning rate.

## 2026-07-15 Spline-15D V5 Four-Run Result Audit

- Status: completed on all locally available artifacts; B and C require a
  second rsync for their final 45 and 35 epochs respectively.
- Compare convergence and numerical stability across the exact-atom versus
  dequantized and no-whitening versus Cholesky 2x2 design.
- Audit before/after normalization, normalized and physical marginal fidelity,
  joint dependence, exact-zero behavior, inverse-sinh tails, invalid physical
  samples, and comparison with the v4 80-epoch reference.
- Produce a compact local comparison artifact and recommend the simplest
  configuration supported by held-out physical diagnostics rather than NLL
  alone.
- Matched epoch-150 result: dequantization has no material effect. A/B and C/D
  curves are nearly identical across physical KS, rank/central correlations,
  inverse-sinh tails, invalid samples, and saturation. Reclip atom fractions do
  not match the exact truth atoms reliably.
- Bulk result: no-whitening A is best. At epoch 200 it reaches median/max
  physical KS `0.051/0.118` and physical Spearman error `0.625`, versus
  `0.058/0.139/0.755` for complete Cholesky run D.
- Tail result: Cholesky controls but does not eliminate rare invalid samples.
  D has inverse-sinh `|a|>5` fraction/max `0.00085/15.5`, negative z/dust
  `1.31%/5.75%`; A has `0.00261/26.2` and `1.74%/11.13%`. A's largest physical
  quantile-Wasserstein error is `229.8`, dominated by a few exponential SFH
  outliers, versus `0.296` for D.
- Optimization result: validation NLL still improves at epoch 200, but every
  late batch is gradient-clipped and scale saturation reaches `40.4%` for A
  and `25.2%` for D. NLL-only selection no longer tracks all physical quality
  metrics monotonically.
- Decision: exact atoms plus no whitening is the preferred minimal and
  transferable architecture for the distribution bulk; atom-aware
  dequantization is rejected by the ablation. This candidate is not yet a
  production prior because support/tail behavior remains unacceptable.
- Follow-up decision: do not rerun the interrupted B/C tasks; their matched
  trajectories already isolate both factors. A is not converged at epoch 200:
  over epochs 180--200 its validation NLL slope is `-0.029/epoch`, maximum KS
  improves `0.1205 -> 0.1182`, Spearman error `0.665 -> 0.625`, and inverse-sinh
  tail fraction/max `0.00320/33.6 -> 0.00261/26.2`. However negative dust mass
  rises to `11.1%` and scale saturation reaches `40.4%`, so a longer run alone
  cannot establish a physically valid prior.
- Recommended next single-run baseline: retain exact atoms, no joint whitening,
  and the current RealNVP; add only physically defined positive-support
  transforms for redshift and dust attenuation, then train long enough to
  observe a plateau. This remains transferable to unsupervised learning because
  the support is known from the parameter semantics, not estimated from latent
  population distributions.
- Artifacts: trajectory plot, matched and last-available CSV summaries, and
  report under `outputs/analysis/feniks_spline15d_v5_ablation_20260715/`.

## 2026-07-15 Spline-15D V5 Minimality Ablation Proposal

- Status: implemented and locally validated; ready for Jean-Zay.
- Goal: identify the simplest transferable RealNVP prior by isolating the two
  preprocessing choices that use latent-population structure: exact-zero atom
  dequantization and joint Cholesky whitening.
- Run one four-task H100 array with an exact 2x2 factorial design, identical
  grouped 50k splits, shifted-asinh marginals, RealNVP architecture, optimizer,
  seed, 200 epochs, and validation-NLL selection:
  - A: exact atoms retained, no whitening;
  - B: normalized zero-atom dequantization/reclipping, no whitening;
  - C: exact atoms retained, Cholesky whitening;
  - D: normalized zero-atom dequantization/reclipping plus Cholesky whitening.
- Do not add ZCA, learned penalties, temperature calibration, alternative flow
  architecture, or data regeneration: each would confound the two questions.
- Compare runs in physical sample space rather than raw normalized NLL across
  incompatible coordinate systems. Audit per-dimension physical KS, atom mass,
  Spearman and central Pearson correlation errors, invalid redshift/dust rates,
  inverse-sinh tails, and train/validation convergence.
- Selection rule: retain the least complex configuration whose physical sample
  diagnostics are statistically indistinguishable from the best configuration;
  whitening and atom handling must each demonstrate a material held-out benefit
  to remain in the production architecture.
- Implementation details:
  - added CLI overrides for the two array factors and an explicit
    `target_table: exact` contract, preventing the raw/no-whitening control from
    silently reading the projector's historically dequantized parquet;
  - added physical KS/Wasserstein aggregates, physical Spearman, central
    Pearson, per-dimension inverse-sinh tails, and exact-zero mass to snapshots;
  - added one shared 200-epoch configuration and a documented four-task Slurm
    array with one H100 per task and unambiguous output names;
  - reduced full snapshot cadence to every five epochs to preserve trajectory
    diagnostics without multiplying the v4 storage footprint by ten.
- Validation: all four one-epoch variants completed end to end on exact local
  spline tables, wrote the intended normalization contracts and extended
  diagnostics, and produced exact zero prior mass only in dequant/reclip tasks.
  Focused suite `9 passed`; Ruff, compileall, `bash -n`, and `git diff --check`
  pass.

## 2026-07-15 Spline-15D RealNVP V4 Production Result Audit

- Status: completed on retrieved run
  `outputs/runs/feniks_spline15d_realnvp_shifted_v4`.
- Determine whether epoch 80 is still improving or already trading sample
  quality for likelihood by auditing the complete epoch snapshot history.
- Compare epoch 0, intermediate checkpoints, best validation-NLL checkpoint,
  and final checkpoint in normalized and physical coordinates.
- Report marginal fidelity, joint correlations, inverse shifted-asinh tails,
  train/validation convergence, and the concrete next training decision.
- Result: the optimization is healthy and unfinished. Validation NLL decreases
  from `21.14` at epoch 0 to its run minimum `4.11` at epoch 80, with a
  still-negative last-10-epoch slope of about `-0.046` per epoch and a negligible
  epoch-80 train/validation gap (`4.107/4.110`).
- Normalized samples are substantially improved: median/max KS reach
  `0.069/0.145` on the fixed validation snapshot, versus `0.121/0.360` at epoch
  0. The maximum KS is still improving at the final epoch.
- The learned central dependence also improves late in training. Physical
  Spearman correlation Frobenius error decreases from `1.45` at epoch 20 to
  `0.92` at epoch 80; after excluding samples with any inverse-sinh argument
  above four, Pearson correlation error decreases from `1.76` to `1.04`.
- The remaining raw physical Pearson error (`3.22`) is dominated by rare joint
  tails, not by the bulk. At epoch 80, `0.186%` of marginal inverse-sinh
  arguments exceed five and the maximum is `20.64`, versus no truth argument
  above five and a truth maximum of `4.68`. The exponential inverse then creates
  SFH contrasts up to roughly `1e7` and invalid negative redshift/dust samples.
- The physical marginal shapes remain imperfect despite good whitened marginal
  KS because inverse whitening mixes all dimensions: worst physical KS values
  are `0.162` for `dust_delta`, `0.152` for metallicity, `0.140` for stellar
  mass, and `0.139` for dust attenuation.
- Decision: v4 is a real recovery and materially better than the failed
  checkpoint-selection runs, but is not yet a valid production prior. Continue
  beyond 80 epochs because both likelihood and central dependence are still
  improving; separately treat joint-tail support as an acceptance criterion,
  since maximum likelihood alone may not eliminate unobserved tail combinations.

## 2026-07-15 Spline-15D RealNVP V4 Shifted-Asinh Recovery

- Status: implemented and locally validated.
- Reuse the existing grouped 50k spline dataset; do not regenerate Diffsky or
  rerun the spline projection.
- Replace the Gaussian-QRMSE lambda scan with a train-only robust shifted-asinh:
  median location, half the q84-q16 interval as lambda, analytic inverse, then
  the existing atom dequantization and Cholesky whitening.
- Train the same identity-initialized 12x256 RealNVP with pure validation NLL
  checkpoint selection, learning rate `2e-5`, 80 complete epochs, and no
  truth-derived loss, checkpoint gate, temperature correction, or early stop.
- Save an epoch-0 baseline and every trained epoch with checkpoint, generated
  samples, physical/normalized truth overlays, correlation matrices, marginal
  metrics, and inverse-transform tail diagnostics.
- Deliver a dedicated v4 config, Jean-Zay H100 launcher, focused tests, and an
  updated runbook with exact launch/retrieval commands.
- Completed implementation:
  - added a versioned robust shifted-asinh transform while retaining exact
    compatibility with v1-v3 unshifted-asinh checkpoints;
  - changed the v4 selection contract to validation NLL only and disabled
    early stopping, weight decay, training jitter, temperature fitting, and
    all generated-truth checkpoint gates;
  - added fixed-seed epoch-0 and per-epoch snapshot bundles with checkpoints,
    physical/normalized samples and overlays, marginal tables, four correlation
    matrices, and pre-sinh tail diagnostics;
  - added the standalone v4 H100 job consuming the existing grouped-v3 spline
    dataset, without repartitioning, projection, or Diffsky generation.
- Validation: focused suite `9 passed`; Ruff, compileall, `bash -n`, and
  `git diff --check` pass. A one-epoch CPU smoke on 256 real local dataset rows
  completed, selected epoch 1 by validation NLL, reloaded the checkpoint, and
  wrote the complete epoch-0/epoch-1 snapshot contract.

## 2026-07-15 Spline-15D RealNVP Production Run Analysis

- Status: completed from the retrieved Jean-Zay run
  `outputs/runs/feniks_spline15d_realnvp_v1`.
- Numerical result: training converged without conventional overfitting; the
  best checkpoint is epoch 116 with train/validation/test NLL
  `9.549/9.620/9.629`, and strict forward/inverse integrity passes.
- Scientific result: the checkpoint is not ready as a prior. Its quality gate
  fails with median/max marginal KS `0.302/0.471`, correlation Frobenius error
  `3.76`, and large physical-space tails including `4.48%` negative redshifts
  and `11.89%` negative dust attenuation samples.
- Root diagnostic: held-out truth maps through the learned inverse flow to a
  base with mean coordinate standard deviation `0.209`, not the required unit
  Gaussian, while samples correctly map back to unit width. Forward sampling
  compounds coupling translations and grows aggregate normalized width from
  `1.0` to `2.79`; `6.70%` of generated normalized coordinates exceed `|x|=5`
  versus `0.10%` for held-out truth.
- Control: sampling the independent unit Gaussian before inverse `asinh`
  already gives median KS `0.059`, substantially better than the trained
  RealNVP, although it misses correlations. The RealNVP only reduces
  correlation Frobenius error from `4.77` to `3.76` while destroying marginals.
- Diagnostic-only temperature scan: using the learned flow with base
  temperature `T=0.15` gives median/max KS `0.102/0.125`, correlation error
  `1.61`, and combined negative-redshift plus negative-dust rate `0.62%`.
  `T=0.20` minimizes the inspected correlation error (`1.48`). These values
  were inspected on test and must not be adopted directly; temperature and
  checkpoint selection need to use validation data.
- Next production iteration: keep the train-fitted analytic `asinh`, add
  validation-time base typicality and generated-sample gates, calibrate a base
  temperature on validation, reduce `shift_clamp`, add fixed permutations, and
  select checkpoints by generated population quality rather than NLL alone.
- V2 implementation completed locally:
  - RealNVP now supports checkpoint-compatible fixed `roll`/`reverse`
    permutations, explicit base-temperature sampling/density, conservative
    clamps, and optional base mean/std/covariance regularization.
  - Checkpoints are selected from validation-only generated-sample metrics;
    the final test split is used only after model and temperature selection.
  - Added exact-truth train-overlap/novel-subset auditing, an independent-normal
    baseline, unit-temperature and validation-calibrated samples, and separate
    full/novel validation/test NLL reporting.
  - The truth-versus-prior diagnostic is now 15 rows by two columns, with
    physical space on the left and normalized space plus `N(0,1)` on the right.
  - Added a four-run Jean-Zay H100 ablation and a validation-only comparison
    command. Local two-epoch end-to-end smoke completed and reloaded the saved
    checkpoint, produced both diagnostic suites, and selected temperature from
    validation metadata.

### V2 four-run postmortem

- Status: all four Jean-Zay runs retrieved and audited; none is acceptable as
  a production prior.
- In every run, validation NLL improves while generated-sample quality degrades.
  The selected checkpoint is epoch 1 for all four models. By epoch 120 the
  truth-to-base mean standard deviation has collapsed to about `0.21--0.24`
  for A/B and `0.22--0.23` for C/D instead of one.
- The comparison script selected D using the pre-temperature checkpoint score.
  This is the wrong cross-run contract: temperature is calibrated only after
  checkpoint selection. B is the only run whose calibrated validation sample
  passes the configured gates and is the least-bad retrieved model, but it
  still visibly misses `dust_delta`, `sfh_dlog_sfr_02`, and
  `sfh_dlog_sfr_03`.
- A full-covariance Gaussian in train-fitted `asinh` space dominates every
  RealNVP: on the current test it obtains median/max marginal KS
  `0.059/0.153` and physical correlation Frobenius error `0.905`, versus
  `0.095/0.167/1.205` for calibrated B.
- The source dataset has a hard split-leakage bug. Split source seeds are
  consecutive while each shard key is `source_seed + shard_index`; therefore
  adjacent split/shard pairs reuse identical JAX keys. There are 1,367 unique
  effective proposals shared by train/validation and 1,386 by train/test.
  Exact 15D overlap affects 1,658/5,000 validation rows and 1,661/5,000 test
  rows. The current validation/test NLL values are not independent estimates.
- Weighted resampling with replacement additionally leaves only 32,543 unique
  train proposals among 40,000 rows. This is a valid abundance representation
  for histograms, but a poor continuous-density training table without a
  weighted-unique likelihood or broader dequantization.
- C and D show that tightening `scale_clamp` from `0.05` to `0.02` was the
  wrong direction: their final validation NLL is about `12.4` rather than
  `9.6--9.7`, and they do not improve samples. The D moment penalty contributes
  only about `0.1` to the objective and is empirically indistinguishable from C.
- Internal layer audit rules out a basic invertibility/NaN failure: checkpoint
  round trips pass at a few `1e-6` and all logged updates are finite. It instead
  shows architectural/optimization saturation. At epoch 120, `99.9%` or more
  of active RealNVP `log_scale` outputs sit above 90% of their clamp, while
  100% of minibatches in every run have raw gradient norm above the configured
  clipping threshold of one.
- Asinh remains a reasonable tail/skew preconditioner, not a Gaussianizer: its
  train marginal Gaussian-QRMSE ranges from `0.019` to `0.346`, with the worst
  dimensions being `dust_delta` and the early SFH contrasts. A monotonic asinh
  cannot remove their U shapes or multimodality. The success of the affine
  full-covariance Gaussian baseline shows that abandoning asinh is not the
  first action; adding joint whitening and then testing residual non-Gaussian
  structure is.
- Required next order: fix split key construction and regenerate independent
  data; train on unique weighted proposals or a straight-MC lightcone; add the
  full-covariance Gaussian as a hard acceptance baseline; use analytic asinh
  followed by train-only affine whitening; evaluate epoch 0 and calibrated
  generated quality during checkpoint selection; then retest RealNVP with a
  materially larger scale range and lower learning rate.
- Proposed attack ladder, to be implemented only after review:
  1. Add synthetic recovery tests in 15D: standard normal, correlated Gaussian,
     nonlinear banana, then two-mode/U-shaped marginals. This separates code,
     optimizer, affine-coupling capacity, and real-data problems.
  2. Add dataset preflight gates for unique effective RNG keys, group-disjoint
     splits, duplicate/atom fractions, covariance spectrum, and local intrinsic
     dimension. Abort training on leakage.
  3. Benchmark identity Gaussian, diagonal Gaussian, full-covariance Gaussian,
     and optionally a small Gaussian mixture using validation-only metrics.
  4. Define the target as `physical -> asinh -> affine whitening`; initialize
     RealNVP at identity and require epoch zero to reproduce the full-covariance
     Gaussian baseline exactly.
  5. Run one-factor ablations over scale range, learning rate, gradient clip,
     depth, and permutation. Log clamp saturation, per-coordinate base moments,
     log-det, and generated metrics every epoch.
  6. Select checkpoint and any temperature jointly on validation, with hard
     per-dimension rather than only aggregate gates. Test remains sealed until
     a candidate beats the affine Gaussian baseline.
- Refined implementation constraint: do not regenerate Diffsky. Pool the
  existing 50k source rows, derive a canonical effective-proposal group from
  `source_seed + shard_index` and proposal row, and repartition whole groups
  into new 40k/5k/5k splits. Preserve multiplicities, rebuild only the spline
  projection/normalization products, and require zero group or exact-truth
  overlap across splits.
- Single-run recovery design: retain the scientific 15D exact truths; use
  train-fitted analytic asinh followed by Cholesky whitening as the internal
  flow coordinates; broaden normalized zero-atom dequantization; initialize
  the RealNVP exactly at identity; use `roll`, `scale_clamp=0.5`,
  `shift_clamp=2`, learning rate `1e-4`, gradient clip `5`, and no base-moment
  penalty or production temperature correction. Evaluate epoch 0 and every
  epoch against the affine-Gaussian baseline, then launch one Jean-Zay job only.
- Recovery implementation completed:
  - grouped the existing 50k into exact 40k/5k/5k splits with zero effective
    proposal overlap while preserving within-split resampling multiplicities;
  - added normalized atom dequantization, train-only Cholesky whitening, exact
    scientific inverse/reclipping, and a zero exact-truth-overlap preflight;
  - made the identity epoch-0 Gaussian a strict fallback and require a minimum
    generative-score improvement before accepting a trained checkpoint;
  - added saturation and sliced-Wasserstein diagnostics, training jitter,
    generative early stopping, focused tests, documentation, and one sequential
    H100 job for regrouping, spline projection, and training;
  - full 50k regroup audit and two local end-to-end CPU smokes passed.

## 2026-07-15 Jean-Zay Spline-15D RealNVP Production Pipeline

- Status: completed locally; ready for Jean-Zay smoke then production launch.
- Goal: provide a documented, restartable Jean-Zay workflow that consumes an
  already-generated Diffsky/FENIKS dataset, projects native SFHs to the fixed
  spline-15D truth contract in a separate post-processing job, fits a simple
  invertible `asinh` normalization on train only, and trains/evaluates one
  supervised RealNVP prior.
- Active contract:
  - Never regenerate Diffsky inside spline projection or prior training.
  - Export exact and dequantized 15D train/validation/test parquets with shared
    fixed spline-node placement and stored metadata sufficient to reconstruct
    the SFH.
  - Fit all `asinh` lambdas/centers/scales from the projected train split only;
    freeze and serialize them for validation, test, sampling, and inference.
  - Train only RealNVP in this production path. Do not expose the alternative
    RQ-spline flow in its configs or launch scripts.
  - During the prior run, write two-column before/after marginal plots annotated
    with lambda, learned-sample versus truth overlays, checkpoint metadata,
    NLL/history, generated samples in normalized and physical 15D spaces, and
    reconstruction-ready normalization metadata.
  - Provide separate Slurm jobs and exact Jean-Zay commands for projection and
    prior training, with no dependency on local generated analysis artifacts.
- Deliverables: reusable production modules, CLIs/configs, Slurm scripts, tests,
  documentation, and smoke-tested local outputs.
- Completed:
  - Added the separate post-generation projector
    `scripts/build_feniks_spline15d_dataset.py`. It consumes existing Diffsky
    train/validation/test parquets and writes exact 15D truth, dequantized flow
    targets, absolute node audits, and a versioned `spline15d_contract.json`.
  - Versioned the eleven optimized normalized-log-time node positions and the
    minimum-distortion `+/-1e-4 dex` exact-zero dequantization in
    `configs/feniks_spline15d_postprocess.yaml`.
  - Added the dedicated `scripts/train_feniks_spline15d_realnvp.py` path. It
    fits per-coordinate analytic `asinh` lambdas on train only, serializes the
    exact inverse, hard-rejects non-RealNVP flows, trains directly in 15D, and
    reloads the best checkpoint before drawing final samples.
  - The training run writes the requested 15-by-2 before/after normalization
    figure with lambda annotations and Gaussian overlays, training history,
    train/validation/test NLL, learned-prior versus held-out-truth overlays,
    correlation/distribution metrics, normalized and physical samples, and
    strict best/last checkpoints with complete normalization metadata.
  - Added a reusable strict checkpoint loader, focused unit tests, two H100
    Slurm jobs, production/smoke YAML configs, and the standalone runbook
    `docs/source/spline15d_realnvp.rst`.
  - Local CPU smoke projected 64 rows per split, trained a 2-layer RealNVP for
    two epochs, reloaded both checkpoint topology and the embedded `asinh`
    contract, generated all plots/tables, and completed with an inverse
    round-trip maximum error of `1.78e-14`.
  - Validation: targeted suite `11 passed`; full suite `324 passed, 1 skipped`
    plus one pre-existing deterministic failure in the tiny Diffsky generator
    metallicity-trend gate. The failing test does not import or execute the new
    spline/RealNVP code. Ruff, compileall, `bash -n`, `git diff --check`, and a
    warning-as-error Sphinx build pass.

## 2026-07-15 FENIKS 15D Prior Normalization Audit

- Status: completed locally on 2026-07-15, including the base / `asinh` /
  shifted-`asinh` comparison.
- Goal: determine whether the new five-physical-plus-ten-SFH-contrast latent
  distribution is suitable for continuous normalizing-flow training and define
  a leakage-safe invertible normalization.
- Active contract:
  - Fit every normalization on a deterministic subset of the 40,000-row train
    table, select it on a disjoint train-validation subset, and evaluate it once
    on the untouched 5,000-row test table.
  - Compare the physical/dequantized distributions before normalization with
    affine scaling, nonlinear marginal Gaussianization, and optional joint
    whitening.
  - Audit atoms and near-atoms, tails, train/test drift, marginal Gaussianity,
    correlations, covariance conditioning, random projections, extrapolation,
    and numerical inverse accuracy.
  - Keep the exact 15D table as scientific truth and use the dequantized table
    only as the continuous flow target.
- Deliverables: normalized train/test parquets, an invertible machine-readable
  transform, metric tables, before/after figures, and an HTML/Markdown report
  under `outputs/analysis/feniks_spline_15d_normalization_20260715/`.
- Completed:
  - Added `scripts/analyze_feniks_spline_15d_normalization.py` with a grouped
    exact-truth train/validation split: 31,986 fit rows and 8,014 validation
    rows. No identical exact 15D truth crosses that normalization split.
  - Compared affine, optimized `asinh`, and 257-knot invertible quantile-spline
    marginals. Validation selects quantile splines for 14 coordinates and an
    `asinh` transform for `dust_av`.
  - On the full IID test sample, the maximum marginal Gaussian quantile RMSE
    drops from `0.6671` after affine standardization to `0.0354`; the mean drops
    from `0.3496` to `0.0220`. The normalized test tail fraction beyond
    `|x| > 5` is `2.67e-5`, with maximum `|x| = 5.21`.
  - Evaluated full covariance whitening as an ablation. It reduces the test
    covariance condition number from `321.7` to `1.51`, but worsens maximum
    marginal QRMSE to `0.530`, creates a `0.4%` tail beyond `|x| > 5`, and
    reaches `|x| = 21.3`. Fixed whitening is therefore not recommended; the NF
    should learn joint mixing after marginal normalization.
  - Found a provenance limitation independent of normalization: exact truth has
    7,457 duplicate excess rows in train and 1,661/5,000 exact test truths also
    occur in train. Dequantization reduces these to 7,222 and 1,605 but cannot
    remove resampling multiplicities. Full-test IID metrics and a conservative
    3,339-row novel-truth audit are both reported.
  - Exported marginal-normalized and whitening-ablation train/test parquets,
    transform/inverse JSON, split and overlap masks, metric CSVs, five figures,
    and standalone Markdown/HTML reports. Full test round-trip error is
    `2.53e-14`.
  - Ruff, compileall, `git diff --check`, parquet shape/finite checks, transform
    invariants, JSON contract checks, PNG decoding, and visual inspection of
    the marginal and correlation figures pass.
- Remaining NF benchmark: train matched NF architectures on affine versus
  marginal-normalized inputs and compare held-out NLL, generated marginals,
  correlations, and population diagnostics. This should follow a production
  data split by unique proposal identifiers before resampling.
- All-`asinh` comparison completed:
  - Exported optimized analytic `asinh` transforms and normalized 40,000-row
    train / 5,000-row test parquets alongside the selected hybrid tables.
  - Regenerated the physical and ten-SFH-coordinate plots with four columns:
    exact truth, dequantized flow target, all-`asinh`, and selected hybrid.
    Updated the score and correlation plots with the all-`asinh` branch.
  - On full IID test, all-`asinh` improves maximum marginal QRMSE from the affine
    `0.6671` to `0.3446`, but the selected hybrid reaches `0.0354`. Mean marginal
    QRMSE is `0.1572` versus `0.0220`.
  - All-`asinh` gives a lower covariance condition number (`151.2` versus
    `321.7`) but retains a `0.1%` tail beyond `|x| > 5` and reaches `|x| = 25.96`;
    the selected hybrid has a `2.67e-5` tail and maximum `|x| = 5.21`.
  - The report, metrics, JSON transform contract, and matched Parquet inputs now
    support a direct NF ablation between analytic `asinh` and quantile-based
    marginal preprocessing.
- Simple-transform comparison completed:
  - Fitted a deterministic shifted-`asinh` transform on each of the 15 fit
    coordinates and exported analytic transform/inverse specifications plus
    normalized train/test parquets.
  - Added dedicated physical and SFH marginal plots, a validation-QRMSE plot,
    and a correlation plot containing only affine base, optimized `asinh`, and
    optimized shifted-`asinh`.
  - On full IID test, mean/max marginal QRMSE is `0.3496/0.6671` for affine,
    `0.1572/0.3446` for `asinh`, and `0.08388/0.2861` for shifted-`asinh`.
    Shifted-`asinh` reduces the tail beyond `|x| > 5` to `2.93e-4` and maximum
    `|x|` to `8.73`, versus `1e-3` and `25.96` for ordinary `asinh`.
  - Shifted-`asinh` does not solve the U-shaped `dust_delta` distribution and
    retains the secondary modes of `q2/q3`. Its full-test covariance condition
    number is `453.2`, worse than affine `302.0` and ordinary `asinh` `151.2`.
  - The stellar-mass shifted-`asinh` optimum is flagged tail-fragile because it
    combines a small lambda with a shift close to the observed support. It must
    be constrained or replaced by ordinary `asinh` in a production simple
    hybrid.

## 2026-07-10 FENIKS 15D Spline Prior Contract

- Status: completed locally on 2026-07-15.
- Goal: evaluate the exact 15-dimensional prior target composed of
  `z_obs`, stellar mass, stellar metallicity, two dust parameters, and ten
  independent SFH shape coordinates.
- Active contract:
  - Represent ten independent SFH contrasts with eleven PCHIP ordinates on a
    shared redshift-aware time grid. The common log-SFR offset is removed and
    stellar mass supplies the absolute normalization.
  - Compare uniform-log-time, recent-lookback, hybrid, and train-optimized
    shared node placements at fixed latent dimension.
  - Optimize placement only on a balanced training subset; select and report
    closure on the untouched held-out test sample.
  - Recompute native/spline SFHs, 107 age weights, SEDs, and 18-band DSPS
    photometry, with p95/p99 diagnostics by state, redshift, mass, and high-sSFR
    tail.
  - Audit the ten contrast distributions for atoms, scale, correlations, and
    effective rank before recommending them for normalizing-flow training.
- Deliverables: a standalone analysis report, closure and latent-coordinate
  tables/figures, a machine-readable 15D contract, and projected train/test
  latent parquet files under `outputs/analysis/`.
- Completed and validated before pause:
  - Implemented the exact 15D contract with five physical coordinates and ten
    adjacent log-SFR contrasts defining eleven shared PCHIP nodes.
  - Optimized the nine independent shared node positions on a balanced train
    subset and selected `optimized_balanced` on a separate train-validation
    proxy. The normalized log-time nodes are `[0, 0.21079, 0.31848, 0.47244,
    0.58562, 0.71219, 0.79838, 0.86010, 0.91243, 0.95883, 1]`.
  - On the untouched balanced test diagnostic, prevalence-weighted population
    p95 closure is `0.00339 mag` and `0.0952 sigma`; main-sequence p95 is
    `0.00260 mag` and `0.0716 sigma`. Quenched tails remain above target at
    `0.0396 mag` and `1.14 sigma`.
  - Exported exact 40,000-row train and 5,000-row test 15D parquets, node-value
    audit tables, closure tables, four figures, and a standalone report under
    `outputs/analysis/feniks_spline_15d_prior_20260710/`.
  - Found exact plateau atoms in the contrast representation: 5,731 exact
    zeros in train, with the largest marginal atom `q10 = 0` at `7.8625%`.
    The exact table is therefore not ready for an unmodified continuous flow.
  - An exploratory test-side dequantization scan showed that uniform jitter up
    to `+/-0.002 dex` removes exact zeros while retaining the population p95
    targets. This is not yet a leakage-safe production choice.
  - Made the shared JAX PCHIP slope calculation safe for gradients through flat
    segments; the position gradient changed from NaN to finite without changing
    the closure interpolation path.
- Finalized on 2026-07-15:
  - Completed the leakage-safe dequantization-width scan on balanced train
    validation. No tested width passes the validation `p95 noise RMS < 0.1`
    gate; the minimum-distortion `+/-1e-4 dex` width is retained, with validation
    p95 `0.00326 mag` and `0.1013 sigma`.
  - Evaluated that preselected width once on the untouched test diagnostic. It
    gives population p95 `0.00339 mag` and `0.0885 sigma`; the exact spline gives
    `0.00339 mag` and `0.0952 sigma`. Quenched closure remains the limiting case
    at `0.0396 mag` and `1.14 sigma` p95.
  - Regenerated exact and dequantized 15D train/test parquets, node audits,
    validation/test width scans, contract JSON, payload, static report, figures,
    and closure tables. The dequantized 40,000-row train table contains no exact
    zero among its ten SFH contrasts; the exact truth table retains 5,731 for
    scientific audit.
  - Added a standalone `Prior 15D` view to
    `outputs/reports/feniks_forward_model_explorer.html`, including the placement,
    state closure, width selection, contrast distributions, and correlation
    diagnostics. The explorer now has eleven validated stages and embeds all
    four 15D figures.
  - Ruff, Ruff format, compileall, `git diff --check`, 15D parquet/contract
    invariants, exact node-to-contrast reconstruction, PNG decoding, embedded
    payload checks, JavaScript syntax, and DOM ID/stage checks pass. The PCHIP
    agrees with SciPy to `4.44e-16`, eager/JIT to `2.22e-16`, and has finite
    gradients with respect to both node values and positions.
  - No Playwright/browser engine is installed, so full-page screenshot rendering
    was not available; figures were previously inspected and the standalone
    HTML/JavaScript/DOM contracts were checked directly.

## 2026-07-10 FENIKS JAX Spline Node-Count Selection

- Status: completed locally.
- Goal: select the smallest spline node count in `6, 8, 10, 12, 16, 20`
  whose representation error is negligible for SFH summaries, DSPS age
  weights/SED/photometry, and the configured photometric uncertainties.
- Active contract:
  - Use the held-out 5,000-row test split, all available continuous/quenched
    rows, and an equal-size random main-sequence sample. Report both balanced
    state metrics and prevalence-weighted population metrics.
  - Use a JAX-native PCHIP in log cosmic time with knots placed at geometric
    normalized-lookback fractions, concentrating resolution near observation.
  - Scan SFH mass-fraction and mean-SFR errors in recent, intermediate, and old
    windows; log-SFH RMSE; DSPS age-weight L1; SED relative L1; per-band and
    maximum magnitude errors; and RMS flux residual in units of `fluxerr`.
  - Aggregate median, p95, p99, and maximum by quenching state, redshift
    quartile, stellar-mass quartile, high-sSFR tail, and population weighting.
  - Select the smallest K passing every predeclared p95/p99 gate in every group
    with adequate sample count. Do not force a selection when no K passes.
  - The current parquet has no `mc_sfh_type`, burst parameters, or bursty SFH
    realization. High-sSFR robustness can be measured, but a true Diffsky
    bursty-state result must be marked unavailable rather than inferred from a
    proxy label.
- Deliverables: machine-readable object/summary/band/gate tables, diagnostic
  PNGs and Markdown/HTML analysis under `outputs/analysis/`, plus an interactive
  node-scan tab in `outputs/reports/feniks_forward_model_explorer.html`.
- Completed:
  - Added `scripts/analyze_feniks_spline_node_scan.py`, which evaluates all
    `K = 6, 8, 10, 12, 16, 20` configurations on 388 held-out galaxies: every
    available continuous/quenched object and a matched main-sequence sample.
    The scan recomputes the JAX SFH, surviving-mass normalization, 107 DSPS age
    weights, SED, dust, IGM, and 18-band photometry for every configuration.
  - Compared uniform-log-time, recent-lookback, and hybrid node grids. A grid
    concentrated only near the observation loses too much early SFH structure;
    the hybrid grid gives the best overall tail fidelity.
  - No scanned configuration passes all 14 predeclared p95/p99 gates. The best
    candidate is hybrid `K=20`, but it still fails nine gates, mainly because
    quenched and low-redshift tails have large noise-normalized, age-weight,
    and magnitude residuals. Thus the scan deliberately returns no strict K.
  - For a less stringent `p95(max |delta mag|) < 0.01 mag` criterion, hybrid
    `K=16` is sufficient for the prevalence-weighted population and the main
    sequence. The high-sSFR tail already satisfies this photometric criterion
    and `p95(RMS residual / sigma) < 0.1` at `K=12`. Quenched galaxies satisfy
    neither criterion for any `K <= 20`, so they require more or adaptively
    placed nodes rather than a larger global latent vector for every galaxy.
  - Reported median/p95/p99/max by state, high-sSFR tail, redshift quartile,
    stellar-mass quartile, and weighted population. A true bursty-state result
    remains unavailable because the current parquet does not store its label
    or realized burst component.
  - Wrote the object, aggregate, per-band, gate, sample, and JSON payloads plus
    four diagnostic figures and Markdown/HTML reports under
    `outputs/analysis/feniks_spline_node_scan_20260710/`. Added a tenth
    interactive `Choix de K` tab to the standalone forward-model explorer.
  - Ruff, Ruff format, compileall, `git diff --check`, payload finiteness and
    shape checks, PNG decoding, embedded-image checks, JavaScript syntax, and
    ten-stage DOM contract checks pass. Figures were inspected directly; no
    browser engine was installed for a rendered full-page screenshot test.

## 2026-07-10 JAX Spline SFH Forward Explorer Study

- Status: completed locally.
- Goal: extend `outputs/reports/feniks_forward_model_explorer.html` with a
  per-galaxy comparison of the native Diffstar SFH and a differentiable spline
  surrogate, including DSPS age weights and 18-band photometry.
- Active contract:
  - Reuse the seven representative train galaxies and the current local DSPS
    runtime already embedded in the explorer.
  - Use fixed-count spline nodes, a deterministic redshift-aware time grid, and
    a JAX-native shape-preserving cubic Hermite interpolation in log-time and
    log-SFR; no SciPy interpolation in the forward path.
  - Renormalize the spline SFH with the same surviving-stellar-mass constraint
    as the native path before computing age weights, SEDs, dust, IGM, and
    magnitudes.
  - Display native versus spline SFH, 107 age weights, 18 magnitudes and
    residuals, node locations, and numerical reconstruction metrics for every
    selectable example.
- Completed:
  - Added a 20-node JAX-native Fritsch-Butland PCHIP in log cosmic time and
    log SFR. The spline passes through its knots exactly, agrees with SciPy to
    `1.22e-15`, has eager/JIT disagreement of only `2.22e-16`, and has finite
    JIT-compiled gradients with respect to every node value.
  - Renormalized every spline SFH with the same surviving-stellar-mass target
    as the native Diffstar path, then recomputed all 107 DSPS age weights and
    the full MDF, dust, IGM, and 18-band forward using JAX.
  - Added a dedicated interactive Spline tab with native/spline SFHs and knot
    markers, native/spline age weights, native/spline magnitudes, per-band
    residuals, all knot values, per-example metrics, and a seven-galaxy summary
    table.
  - The seven representative examples have maximum spline/native magnitude
    error `0.00815` mag. The quenched example has the largest age-weight L1
    (`0.0622`) and log-SFH RMSE (`0.0338` dex); the massive example has the
    largest magnitude error. All spline age weights sum to unity and surviving
    mass closes within `3e-8` dex.
  - Regenerated `outputs/reports/feniks_forward_model_explorer.html` and its
    JSON payload. Ruff, Ruff format, compileall, payload shape/finite checks,
    JavaScript syntax, nine-stage DOM ID checks, and `git diff --check` pass.
    No browser engine or jsdom installation was available for screenshot-based
    rendering validation; the standalone HTML and JavaScript/DOM contracts were
    validated directly instead.

## 2026-07-10 FENIKS Explorer Provenance and Graph Revision

- Status: completed locally.
- Goal: remove ambiguity between generation-time noiseless magnitudes, current
  runtime noiseless magnitudes, noisy observations, generative `true +/- sigma`
  intervals, and conventional `observed +/- sigma` measurement bars; replace
  the linear overview with a detailed, branched, Mermaid-like interactive graph.
- Completed:
  - Split the magnitude view into generation-time noiseless truth, current
    runtime noiseless forward, and noisy observed AB magnitude when the stored
    observed flux is positive; labels and tooltips now state each provenance.
  - Displayed both uncertainty conventions in the flux view: blue
    `flux_true +/- fluxerr`, black `flux_observed +/- fluxerr`, the orange
    realized-noise segment, and the green current-runtime forward point.
  - Added the Gaussian coverage explanation and measured the full training-set
    tail rate: `31.66%` of draws have `|noise / fluxerr| > 1`, consistent with
    the `31.73%` expected outside a one-sigma interval.
  - Replaced the eight-box overview with a four-region, nineteen-node branched
    SVG graph. It exposes all 18 raw names and selected values, halo assembly,
    quenching/rejuvenation, SFH normalization, SSP/MDF synthesis, dust, IGM,
    filter integration, generation forward, error construction, and noise.
  - Regenerated the standalone HTML and payload; Ruff, format, compileall,
    JavaScript syntax, `git diff --check`, and jsdom checks pass for seven
    examples, eighteen parameters, eighteen bands, and all page views.

## 2026-07-10 FENIKS Explorer Spectral, Filter, and Error Revision

- Status: completed locally.
- Goal: extend the interactive forward explorer with rest-frame filter
  overlays, physically converted `L_lambda` per Angstrom SEDs, explicit
  photometric error bars and noise draws, a dedicated error-model tab, and a
  single clickable graph spanning raw parameters through observed photometry.
- Contract: retain the same 18-band train parquet, generation manifest, current
  local forward, and stored-vs-current runtime distinction.
- Completed:
  - Converted the displayed spectra from the native `Lnu` in `Lsun/Hz` to
    `Llambda` in `Lsun/Angstrom` using `Llambda = Lnu * c / lambda^2`, and
    changed the spectral axis to rest-frame Angstroms over 300--59,624 A.
  - Embedded 3,240 downsampled throughput points for all 18 filters. The SED
    panel overlays each observed-frame throughput at
    `lambda_rest = lambda_observed / (1 + z)` on a separate transmission axis;
    all 18 filters overlap the plotted SED for every representative example.
  - Replaced the overview tiles with one clickable eight-node SVG graph from
    raw parameters through halo, SFH, mass normalization, SED, filters, error
    model, and observed photometry.
  - Added a dedicated Errors tab with the exact `m5_depth` variance formula,
    per-band `m5` and effective `gamma`, source/background/systematic sigma
    components, all-band and selected-band pull distributions, and the stored
    Gaussian noise realization.
  - Added a photometric flux graph with true and current-model points, observed
    points, explicit `+/- fluxerr` bars, and the true-to-observed noise segment.
    The table now includes noise in nJy and `(flux - flux_true) / fluxerr`.
  - Validation: the 720,000 stored noise pulls have mean `-0.00158` and standard
    deviation `0.9987`; decomposed sigma values reproduce stored `fluxerr` to
    `2.22e-16` relative error; pull identities agree to `4.44e-16`; Ruff,
    format, compileall, JavaScript syntax, `git diff --check`, payload coverage,
    and jsdom navigation checks pass across all eight page views.

## 2026-07-10 Interactive FENIKS Forward-Model Explorer

- Status: completed locally.
- Goal: generate a standalone interactive webpage that traces representative
  galaxies from the 18 raw FENIKS truth columns through the exact local
  Diffmah, Diffstar, DSPS, dust, IGM, filter, and noise stages, while showing
  the population distribution of every truth coordinate.
- Active contract:
  - Dataset: `Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/train.parquet`.
  - Config: `configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml`.
  - Output: `outputs/reports/feniks_forward_model_explorer.html` plus a compact
    machine-readable payload.
- Completed:
  - Added `scripts/build_feniks_forward_explorer.py` and the self-contained
    `scripts/templates/feniks_forward_explorer.html` template. The page uses
    embedded SVG renderers and has no CDN or server dependency.
  - Embedded train-population histograms for all 18 truth coordinates and all
    18 stored magnitudes, plus seven exact representative galaxies: typical,
    nearby, high-redshift, massive, dusty, actively quenched, and high-sSFR.
  - Exposed the actual intermediate arrays for each example: `alpha(t)`,
    `Mh(t)`, `dMh/dt`, `log y`, `Q(t)`, main-sequence/raw/renormalized SFH,
    cumulative formed mass, 107 SSP age weights, intrinsic/dusted/post-IGM SED,
    current model magnitudes, and stored true/noisy fluxes.
  - Made two non-obvious local behaviors explicit in the interface: the local
    wrapper passes `lgt0=log10(t_obs)` to Diffmah, and normalization to surviving
    stellar mass cancels the global `diffstar_lgy_at_mcrit` amplitude.
  - Kept generation-time and current-runtime predictions separate. The dataset
    manifest records commit `5a41c67`, Diffstar 1.0.3, DSPS 0.4.8, and JAX
    0.10.2; the current local runtime differs and produces example-level maximum
    magnitude drifts from `0.0277` to `0.740` mag.
  - Validation: the explicit Diffstar kernel decomposition matches the wrapper
    to `3.33e-06` relative error; SSP age weights sum to unity; surviving mass
    matches the requested truth to `2.03e-08` dex; Ruff, format, compileall,
    `git diff --check`, JavaScript syntax, JSON contract, and jsdom interaction
    checks pass across all six interactive stages.

## 2026-07-10 FENIKS Dirac / SFH Representation Decision Study

- Status: completed locally.
- Goal: determine whether the exact Diffstar atoms should be modeled as a
  discrete state, remapped, fixed, or replaced by an SED-native SFH
  representation, using quantitative SFH and DSPS-photometry reconstruction
  diagnostics rather than marginal-shape arguments alone.
- Completed:
  - Added `scripts/analyze_feniks_dirac_sfh_options.py` and generated the
    standalone Markdown/HTML study under
    `outputs/analysis/feniks_dirac_sfh_options_20260710/`, with seven inspected
    figures and CSV/JSON evidence tables.
  - Proved that the four 96.493% Diffstar atoms have identical row masks: they
    encode one shared main-sequence/no-quenching state. The train split has
    38,597 atom rows and only 1,403 continuous quenched rows; validation and
    test continuous counts are 152 and 194.
  - Traced the local generator and forward contracts. The current parquet
    stores the compact Diffstar/Diffmah coordinates but drops the generator's
    explicit `mc_sfh_type`, SFH table, burstiness realization, and SSP weights.
    The local DSPS closure can instead be reconstructed exactly from formed
    mass, 107 age weights, redshift, metallicity, and dust under the fixed
    global model settings.
  - Benchmarked direct age weights, seven PopCosmos bins, 16 mass-conserving
    lookback bins, and 12/20-knot log-time PCHIP SFHs on 192 held-out objects
    balanced by state. Direct age weights are exact; the 16-bin representation
    has the best worst-state p95 maximum magnitude error (`0.00943` mag) among
    the tested compact variants. The 20-knot spline has lower population p95
    (`0.00231` mag) but a `0.0521` mag quenched-state p95 tail.
  - Rejected atom clipping as a standalone normalization: the nearest observed
    joint continuous proxy gives a `1.78` mag p95 maximum 18-band change for
    atom rows. Forcing continuous rows to the atom gives `3.76` mag p95. Both
    require the same missing discrete state to restore exact values.
  - Recommended one explicit Bernoulli quenching state plus branch-specific
    continuous models for the native 18D prior, together with a separate
    SED-native age-weight product for exact closure.
  - Found a reproducibility blocker: the current local `shine` forward differs
    from the stored 18-band parquet by up to `0.792` mag in per-band p95 on the
    balanced sample, despite the generation-time report passing at commit
    `5a41c67` with JAX 0.10.2. Freeze the runtime and hash the compressed SSP
    asset before regenerating the next production dataset.
  - Validation: seven targeted normalization tests pass; Ruff, compileall,
    `git diff --check`, JSON/CSV loading, and all seven HTML image references
    pass.

## 2026-07-10 Jean-Zay Hybrid vs Dirac-Preserving Prior Benchmark

- Status: completed locally; ready for the Jean-Zay smoke array.
- Goal: make both normalization designs trainable with RealNVP and
  rational-quadratic spline priors, then provide reproducible Jean-Zay launch
  commands for the four-way comparison.
- Completed:
  - Added reusable, train-fitted heterogeneous marginal transforms with
    float64 forward/inverse checks and checked-in hybrid and Dirac-preserved
    specifications for the 18-band train split.
  - Added a benchmark trainer that fits one 18D continuous flow for the
    Dirac-preserved version, or the statistically explicit hybrid likelihood:
    one empirical Bernoulli atom state, a 14D atom-branch flow, and an 18D
    continuous-branch flow. Hybrid samples restore all four atom values exactly.
  - Added four H100 experiment configs covering hybrid/Dirac-preserved crossed
    with RealNVP/RQ-spline, plus a four-task Jean-Zay Slurm launcher and a
    physical-space comparison table generator. NLL is explicitly restricted to
    within-normalization comparisons because the two versions use different
    reference measures.
  - Kept smoke and production run directories separate, rejected accidental
    reuse of existing output directories, and documented the dependency-based
    train/compare commands in `docs/source/prior_learning.rst`.
  - Ran one local training epoch for all four combinations. Both hybrid flows
    generated exact shared-atom rows; both single-vector continuous flows
    generated zero exact atoms, as expected. Transform round trips were at most
    `1.34e-14` after retaining float64 normalization calculations.
  - Validation: `19 passed` across the new transform/config tests and existing
    supervised-prior suite; Ruff, compileall, `bash -n`, and `git diff --check`
    pass. The one-epoch smoke quality gates fail by construction and are not
    scientific results.

## 2026-07-10 Final Hybrid vs Dirac-Preserving Normalization Report

- Status: completed.
- Goal: deliver two explicit invertible normalization designs, final holdout
  diagnostics, before/after PNGs, and a standalone HTML report with formulas,
  fitted values, and parameter-by-parameter computation details.
- Completed:
  - Added `scripts/build_final_normalization_report.py` and generated two full
    18D before/after PNGs: the recommended shared-indicator hybrid model and a
    simpler single-vector design retaining each exact atom at normalized zero.
  - Generated a standalone HTML report with 18 parameter sections. Each
    version records the forward/inverse formula, fitted numerical values,
    train-only computation method, test metrics, and full spline knots.
  - Added an atom-centered asinh normalization whose scale is the conditional
    train RMS around the atom. This keeps both atom and continuous minority at
    order-unity magnitude while remaining analytically invertible.
  - Final test-split robustness review rejected the fragile log-like stellar-mass transform
    after a plausible test value produced `|x|=7.70`; the monotone spline now
    gives test RMSE `0.031` and max `|x|=4.29` without clipping.
  - Verified all 18 HTML sections and image references, maximum test round-trip
    error `1.69e-14`, and maximum transformed test amplitude `|x|=4.29` for
    both designs.
  - Recorded that the test split influenced the final stellar-mass family
    choice (but not fitted transform values), so reported test scores are now
    descriptive robustness metrics rather than an untouched final holdout.

## 2026-07-10 Invertible Marginal Normalization Selection

- Status: completed.
- Goal: select a scientifically defensible, invertible marginal transform for
  every FENIKS prior coordinate, favoring simple affine/asinh transforms and
  escalating only structurally non-Gaussian marginals to a monotone spline or
  mixed discrete/continuous treatment.
- Completed:
  - Added `scripts/analyze_invertible_prior_normalizations.py` to compare
    affine, widened-bound logit, unshifted asinh, shifted asinh, and monotone
    quantile-spline transforms on train and independent validation data.
  - Selected two unshifted asinh transforms (`z_obs`, `dust_av`), five shifted
    asinh transforms, one widened-bound logit (`diffmah_logtc`), six monotone
    quantile splines, and four atom-preserving mixed specifications whose
    continuous minority uses shifted asinh.
  - Wrote full/useful selected-transform figures, family-score comparisons,
    CSV summaries, a Markdown report, and complete JSON transform parameters
    including spline knots and analytic inverse metadata.
  - Verified maximum train and validation forward/inverse round-trip errors
    below `2e-14`. All selected validation coordinates stay below `|x|=5`, so
    no non-invertible hard clipping is needed.
  - Validation quantile RMSE is about `0.02-0.12` for ordinary continuous
    dimensions. Conditional atom-minority scores reach `0.19` because only 152
    non-atom validation rows are available, while the exact atom is preserved
    separately without modification.

## 2026-07-10 Per-Parameter Asinh Normalization Study

- Status: completed.
- Goal: distinguish narrow continuous distributions from true discrete atoms,
  then test a separate `lambda * asinh(theta / lambda)` compression for every
  FENIKS prior coordinate.
- Completed:
  - Added a `+/-0.5` residual zoom for all four quenching coordinates, showing
    the exact dominant value separately from the continuous minority. Each has
    the same 38,597/40,000 (`96.493%`) exact atom; only `qlglgdt` has minority
    values within 0.5 of that atom.
  - Scanned 161 lambda values per coordinate for
    `lambda * asinh(theta / lambda)`, independently standardized every result,
    and minimized empirical-to-standard-normal quantile RMSE.
  - Wrote full-18D and useful-only physical/current/asinh comparison figures,
    lambda scans, a per-parameter transform table, atom statistics, and a
    Markdown interpretation in the RealNVP run directory.
  - Finite asinh compression is strongly useful for `z_obs` (`lambda=0.292`),
    `diffstar_indx_hi` (`lambda=2.02`), and `dust_av` (`lambda=0.0216`). Five
    coordinates prefer the log-like limit and ten prefer the linear limit.
  - Recorded that monotone preprocessing cannot Gaussianize the four 96.493%
    atoms; these require a discrete/continuous mixture, explicit quenching-state
    conditioning, or removal from the continuous flow.
- Follow-up completed: revised the Dirac zoom so both panels retain every row
  in the same `+/-0.5` window around the dominant value. The left panel uses
  linear counts and the right uses log counts, directly distinguishing exact
  repetitions from nearby continuous values without removing the atom.
- Follow-up completed: draw the exact-value atom as an explicit black Dirac
  stem in both panels, with its physical value, count, and fraction in the
  legend. The histogram remains present behind the stem.
- Follow-up completed: replace the stem with a simpler global-distribution and
  atom-zoom pair, annotated directly with the exact value and row count.

## 2026-07-10 FENIKS RealNVP Normalization Diagnostics

- Status: completed.
- Goal: document the current `truth_standardized_logit` transform used by the
  supervised 18D FENIKS RealNVP, with readable physical parameter labels and
  one-dimensional truth distributions before and after normalization.
- Completed:
  - Added `scripts/plot_realnvp_normalization_diagnostics.py`, which labels the
    18 parameters physically and applies the exact stored checkpoint transform:
    bounded logistic logit followed by checkpoint train-set centering/scaling.
  - Wrote physical-vs-network and raw-logit 1D histograms, a readable
    parameter glossary, normalization metadata, and per-parameter statistics
    to `outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp_stdlogit_v2/`.
  - Confirmed that the supplied 18-band train parquet is not the parquet used
    by this checkpoint despite shared object IDs; the diagnostics explicitly
    apply the old checkpoint normalization to the supplied distribution.
  - Recorded the main numerical risks: the quenching-transition logit scale is
    floored at `0.1`, several dimensions contain sharp atoms, and rare bounded
    logit tails reach `|x|=48.78` (`diffmah_late_index`) and `45.40`
    (`diffmah_logm0`) while 99% remain below about 3.5.
- Completed follow-up:
  - Updated all plot labels and the glossary to English. Each parameter now
    has an explicit role and a one-line physical description; Diffstar is
    identified as SFH and Diffmah as halo assembly.
  - Added `realnvp_useful_1d_before_after_normalization.png` for the nine
    useful coordinates present in the 18D learned prior.
  - Added `configs/prior_diffsky_synthetic_feniks_18band_realnvp_widebounds.yaml`
    for a clean 18-band retraining. It widens the bounds consistently in the
    inference model and prior: `z_obs` to 6, `logm0` to 17, early/late halo
    indices to 12/6, `t_peak` to 20 Gyr, and `dust_av` to 7 mag.
  - The new bounds cover every finite train value, notably 4,372/40,000
    `diffmah_early_index` values above the prior upper bound of 6 in the old
    setup.

## 2026-07-09 RQ Spline Prior For FENIKS 18D

- Status: completed locally; ready for Jean-Zay launch.
- Goal: replace the failing supervised RealNVP truth-prior experiment with a
  more expressive rational-quadratic spline coupling flow that can be trained
  directly on the 18D FENIKS/Diffsky closure truth distribution, produce the
  same diagnostics/corner plots, and load as a supervised prior in downstream
  amortized training.
- Assumptions:
  - The main benchmark truth space remains the native 18D Diffsky/DSPS closure
    latent space; SFH-derived views remain diagnostics rather than the primary
    prior-learning target.
  - No new dependency should be required for Jean-Zay; the spline transform is
    implemented with JAX/Equinox already used by the project.
- Completed:
  - Add an RQ spline coupling prior with exact `sample`, `inverse`, and
    `log_prob` methods.
  - Generalize supervised prior checkpoints and integrity checks beyond
    RealNVP while preserving backward compatibility for existing checkpoints.
  - Add a Jean-Zay H100 experiment config and focused tests.
  - Added Slurm stages `supervised_spline_prior_train` and
    `supervised_spline_prior_report`.
  - Verified locally with `compileall`, flow/prior/amortized prior-source/data
    parallel/CLI tests, ruff, and config-load smoke.

## 2026-07-09 FENIKS HTML Report Polish

- Status: completed.
- Goal: improve readability of the FENIKS HTML report by restructuring the
  experimental-setup section, rendering the photometric-error equations in
  LaTeX, fixing the error-model table alignment, adding explicit result
  sentences under J-lens plots, and replacing hard-to-read useful corner plots
  with clearer compact diagnostics.
- Assumptions:
  - This is a generated-report update under `outputs/`, not a source-code
    behavior change.
  - The scientific interpretation remains unchanged: dataset closure is valid,
    all current amortized runs fail closure gates, and the joint annealed run is
    only the best current diagnostic candidate.
- Completed:
  - Rewrote `outputs/reports/feniks_prior_ladder_report.html` sections 1, 2,
    and 5 into structured subsections.
  - Added MathJax-rendered equations for the FENIKS/PhotErr-style error model,
    noise injection, likelihood effective scale, and J-lens derivatives.
  - Rebuilt the per-band error table from
    `configs/diffsky_synthetic_feniks_260617_50k.yaml` and added table-specific
    alignment CSS.
  - Replaced the hard-to-read useful corner as the primary per-experiment plot
    with the local useful distribution diagnostic, keeping compact 5D corners as
    secondary diagnostics.
  - Added explicit result sentences under J-lens and error-model diagnostic
    plots.
  - Verified the HTML references 83 images with zero missing targets.

## 2026-07-09 FENIKS HTML Report Revisions

- Status: completed.
- Goal: revise the FENIKS prior-ladder HTML report so the AE baseline does
  not imply a learned prior, the synthetic lightcone/data-generation section
  captures the weighted Diffsky/FENIKS workflow and caveats, the PhotErr-style
  error model is explicit, and the J-lens section is understandable with
  explanatory plots derived from local artifacts.
- Assumptions:
  - The AE corner plots currently labeled `truth/prior/posterior` contain an
    inactive standard-normal reference distribution decoded through the same
    latent transform, not a learned prior and not a loss term.
  - The active 14-band FENIKS dataset remains the one analyzed in this report;
    the email context also mentions an 18-band survey-like variant and should
    be framed as related context rather than silently changing this report's
    dataset contract.
- Completed:
  - Regenerated `outputs/reports/feniks_prior_ladder_report.html` with a much
    more detailed first section on the weighted Diffsky/FENIKS proposal
    lightcone, weighted resampling, local DSPS closure photometry, truth-space
    caveats, and completeness caveats.
  - Added an explicit PhotErr-style `m5_depth` error-model section with the
    flux-error/noise formula, configured per-band depth/gamma/eta table,
    injected-noise draw, likelihood `sigma_eff`, and existing error diagnostic
    plots.
  - Added a dedicated AE corner-semantics section explaining that AE
    `truth/prior/posterior` corners use an inactive standard-normal reference,
    not a learned prior and not a Bayesian posterior.
  - Generated report-specific useful corners for all five experiments under
    `outputs/reports/feniks_prior_ladder_report_assets/*_report_useful_truth_reference_posterior_corner.png`
    using truth columns directly from the FENIKS test parquet.
  - Added explanatory J-lens plots: relative singular spectra, rank/nullity
    boxplots, AE singular spectra, prior-score partition, posterior variance by
    direction kind, and physical loadings for visible vs exact-null directions.
  - Verified the updated HTML references 82 images with zero missing targets.

## 2026-07-09 FENIKS Prior-Ladder HTML Report

- Status: completed.
- Goal: build an English HTML report from the locally copied FENIKS training,
  inference, and J-lens artifacts, including dataset-generation context,
  caveats, per-experiment loss/residual/corner/J-lens plots, and cross-model
  interpretation.
- Assumptions:
  - The report is a generated artifact and belongs under `outputs/`.
  - Local `*_infer_lowmem` outputs are the currently usable inference products;
    original `*_infer` jobs may be incomplete because prior-predictive decoding
    previously ran out of GPU memory.
  - Existing RealNVP prior-ladder artifacts must be interpreted with the known
    pre-hardening caveat unless a post-mask/shift-clamp retrain is explicitly
    identified in the run metadata.
- Completed:
  - Wrote `outputs/reports/feniks_prior_ladder_report.html`, an English HTML
    report with FENIKS dataset-generation context, model-ladder explanation,
    caveats, per-experiment loss/residual/corner/J-lens plots, and
    cross-experiment interpretation.
  - Wrote comparison plots under
    `outputs/reports/feniks_prior_ladder_report_assets/` for validation
    negative log-likelihood, photo-z metrics, normalized residuals,
    posterior-vs-truth RMSE, prior-vs-truth overlap, and J-lens rank/noise
    summaries.
  - Wrote analysis tables:
    `feniks_report_experiment_summary.csv`,
    `feniks_report_posterior_vs_truth_metrics_long.csv`,
    `feniks_report_residual_tails_by_band_long.csv`,
    `feniks_report_jlens_summary.csv`, and
    `feniks_report_image_inventory.csv`.
  - Verified the HTML references 75 images with zero missing local targets.
  - Main conclusion recorded in the report: the FENIKS synthetic closure
    dataset is validated, but all five current amortized inference runs fail
    the posterior/prior closure gate; the joint annealed RealNVP run is the
    best current diagnostic candidate, while supervised-prior-dependent runs
    remain caveated by the known RealNVP prior issue.

## 2026-07-09 RealNVP Prior Fit Hardening

- Status: completed for code hardening, shift stabilization, standardized-logit
  prior training, and local validation.
- Goal: make the RealNVP prior path fail explicitly when the flow topology or
  generated prior samples are invalid, and report whether the learned prior is
  scientifically usable rather than relying on truth NLL alone.
- Current findings:
  - Locally copied `*_maskfix` prior artifacts still deserialize as old
    float-mask RealNVP checkpoints, so they were not produced with the local
    boolean-mask code despite the run name.
  - Training/inference scalar losses can look finite while prior samples are
    stuck on physical bounds. The next run must write integrity and quality
    gates that make this condition visible in JSON/Markdown summaries.
  - The first strict-integrity Jean-Zay relaunch failed at epoch 45 while
    writing an intermediate checkpoint: boolean masks were fixed, but the
    RealNVP affine shifts were still unbounded and made `forward/inverse`
    numerically inconsistent on generated samples.
  - The `shiftclamp_v1` Jean-Zay relaunch still failed strict final integrity:
    shifts were bounded, but fitting the 18D RealNVP directly on raw bounded
    logits forced the flow to learn artificial clipped-logit tails and boundary
    atoms up to about `|x|=13.8`, leaving the H100-scale model numerically
    fragile.
  - The first standardized-logit Jean-Zay relaunch (`stdlogit_v1`) failed only
    on a marginal float32 round-trip check (`0.00225` with threshold `0.001`).
    This is no longer the previous broken-flow regime; it should be recorded as
    a numerical warning while final distribution diagnostics decide scientific
    usability.
- Completed:
  - Added reusable RealNVP integrity diagnostics checking boolean/static
    coupling masks, forward/inverse round-trip consistency, finite self
    log-probability, and pathological generated-sample scale.
  - Added `shift_clamp` to RealNVP coupling layers, prior-learning defaults,
    amortized-prior defaults, checkpoint sidecars, and the FENIKS supervised
    prior config. Shifts are now `shift_clamp * tanh(raw_shift)`, preventing
    runaway samples while preserving enough range for the 18D truth latents.
  - Intermediate epoch checkpoints now record integrity diagnostics without
    aborting the whole run; final/best checkpoints remain strict.
  - Added `prior_learning.latent.normalization: truth_standardized_logit`.
    Supervised prior training now computes train-set raw-logit mean/std,
    trains the RealNVP in those standardized coordinates, stores the
    normalization in the checkpoint `LatentSpec`, and reuses it consistently for
    validation/test truth and prior-sample conversion back to physical theta.
  - RealNVP round-trip integrity now has separate WARN/FAIL thresholds:
    `>1e-3` is recorded as WARN, while `>1e-2` remains FAIL. This prevents a
    usable float32 model from aborting before writing samples, corners, and
    quality-gate diagnostics.
  - Supervised prior training now writes `prior_training_progress.json`
    incrementally and prints one progress line per epoch when progress logging is
    enabled, so Slurm logs no longer sit silent after the train/validation split.
  - The amortized supervised-checkpoint path now treats the checkpoint
    `LatentSpec` as the active latent coordinate system after validating names
    and physical bounds against the config. This avoids a later encoder/prior
    mismatch when the supervised prior uses standardized logits.
  - Checkpoint save/load for supervised and amortized RealNVP paths now refuses
    invalid flows instead of allowing downstream inference/J-lens to proceed.
  - Supervised-prior diagnostics now write `prior_quality_gate` and
    `prior_quality_gate_status` into `supervised_prior_summary.json`, plus a
    dedicated Markdown section explaining failed distribution checks.
  - Added regression tests for float-mask rejection, checkpoint sidecar
    integrity metadata, and quality-gate FAIL/PASS behavior.
  - Validation: local FENIKS 18D prior smokes with `shift_clamp=5.0` kept
    round-trip error near `1e-6` and generated latent samples at order-unity to
    low-teen scale instead of `1e5+`.
  - Validation: standardized-logit FENIKS 18D smokes pass strict RealNVP
    integrity. The full 12-layer/256-hidden architecture smoke has round-trip
    max error `1.43e-6` and generated standardized latent max `4.36`.
  - Validation commands: `python -m compileall euclid_dsps scripts`,
    `python -m pytest -q tests/test_amortized_flows.py
    tests/test_prior_learning_supervised.py`,
    `python -m pytest -q tests/test_amortized_data_parallel.py
    tests/test_cli.py`, and targeted `uvx ruff check` all pass.

## 2026-07-08 Supervised RealNVP Prior Failure Investigation

- Status: completed for root-cause isolation and code fix.
- Goal: identify why the supervised truth-prior RealNVP reports good NLL on
  truth latents but generates samples that saturate physical bounds and have
  very poor self log-probability.
- Findings:
  - Root cause: RealNVP coupling masks were stored as float arrays, so
    `eqx.is_inexact_array` included them in trainable Optax leaves. AdamW
    changed masks from binary `0/1` to values around `-2` and `3`, breaking the
    coupling-layer bijection and invalidating `forward`/`inverse` consistency.
  - The broken supervised truth-prior checkpoint maps truth latents into a tiny
    base-space ball with high apparent NLL, but `prior.sample()` produces
    latents around `1e5`-`1e7`; `inverse(forward(u))` then loses `u` by
    catastrophic float32 cancellation and `x_to_theta` saturates all physical
    bounds.
  - The same RealNVP mask bug contaminates the existing joint RealNVP amortized
    checkpoints and supervised-prior-dependent amortized checkpoints. Treat old
    RealNVP checkpoints from this ladder as non-scientific and retrain them.
  - Fixed `euclid_dsps/amortized/flows.py` so masks are boolean topology leaves,
    not inexact trainable leaves. Added clear load-time errors for old
    float-mask checkpoints and regression tests for mask immutability and
    round-trip sampling.

## 2026-07-08 FENIKS Infer/J-Lens Log Triage

- Status: completed for the copied-log triage.
- Goal: analyze the newly copied FENIKS infer and J-lens logs to determine
  which stages completed, which failed, and what should be launched or copied
  next.
- Assumptions:
  - The user has copied logs first; result artifacts may still need a separate
    rsync before numeric analysis.
  - J-lens shard/finalize status should be inferred from both `outputs/logs`
    and Slurm `out/err` files, because `outputs/logs` can miss individual array
    elements depending on how the launcher names log files.
- Findings:
  - Parsed the copied logs into
    `outputs/jeanzay_feniks_ladder/log_analysis/feniks_infer_jlens_log_status.csv`
    and JSON for follow-up inspection.
  - All five J-lens runs completed all four shards and finalization, writing
    1024 objects per run according to the logs. The next local step is to rsync
    the `outputs/runs/*_jlens/` artifact directories before numeric analysis.
  - All five amortized inference jobs failed with a JAX GPU OOM during the
    prior-predictive stage after the posterior batches had run. The failing
    allocation was 36.41 GiB in DSPS derived/prior-predictive decoding with
    `prior_samples=8192` and `prior_predictive_batch_size=256`.
  - No training or J-lens relaunch is needed from these logs. Relaunch only the
    inference stages with a smaller memory profile: smaller batch/JAX batch,
    fewer posterior/prior samples, and an explicit low
    `--prior-predictive-batch-size`.

## 2026-07-08 FENIKS Ladder Local Log Audit And Output Cleanup

- Status: completed for this cleanup slice.
- Goal: analyze the Jean-Zay FENIKS prior-ladder logs that were copied back
  locally, identify exactly which run artifacts still need to be rsynced for
  full analysis, and move non-FENIKS generated outputs under `outputs/legacy`
  so the active FENIKS workspace is easy to inspect.
- Assumptions:
  - The active copied logs live under `outputs/jeanzay_feniks_ladder/logs/`.
  - The active FENIKS production artifacts should remain directly visible under
    `outputs/`, while older HLTDS/Diffsky smoke/dev outputs can be archived.
  - No generated outputs should be deleted during cleanup; only moves are
    allowed.
- Completed:
  - Parsed the copied Slurm logs and wrote
    `outputs/jeanzay_feniks_ladder/log_analysis/feniks_prior_ladder_log_summary.csv`
    plus JSON.
  - Confirmed successful training/report jobs for the supervised truth prior,
    AE baseline, fixed-KL joint RealNVP, annealed-KL joint RealNVP, frozen
    supervised prior amortized run, and supervised-prior fine-tune run. The old
    `1558720` frozen-prior attempt is the known pre-fix snapshot histogram
    failure.
  - Recorded that the fine-tune run completed but skipped many non-finite
    updates, so it should be treated as an experimental/diagnostic comparison,
    not the main prior result.
  - Moved 79 non-FENIKS generated output entries into `outputs/legacy/` and
    wrote
    `outputs/jeanzay_feniks_ladder/log_analysis/outputs_legacy_move_manifest_20260708_175404.json`.
  - Added
    `outputs/jeanzay_feniks_ladder/log_analysis/rsync_feniks_ladder_artifacts.sh`
    to pull all current remote FENIKS runs, logs, configs, Slurm script, and
    runbook snapshot through the user’s CEA jump-host SSH route.
  - Re-analyzed the later copied `outputs/logs` payload: 131 logs total, 14
    FENIKS ladder logs relevant to the current experiment sequence, with the
    same successful final runs and the expected pre-fix failures.
  - Wrote
    `outputs/jeanzay_feniks_ladder/log_analysis/outputs_logs_summary.csv` and
    `outputs/jeanzay_feniks_ladder/log_analysis/feniks_ladder_outputs_logs_summary.csv`.
  - Moved 106 newly copied non-FENIKS logs into
    `outputs/legacy/logs/outputs_logs/`, keeping only FENIKS ladder/generation
    logs in the active `outputs/jeanzay_feniks_ladder/logs/outputs_logs/`
    folder.
  - Added
    `outputs/jeanzay_feniks_ladder/log_analysis/rsync_feniks_ladder_analysis_artifacts.sh`
    for the preferred artifact pull: plots, summaries, manifests, parquet/csv
    tables, configs, logs, and only `best`/`last` checkpoints, explicitly
    excluding epoch checkpoints and per-epoch training snapshot directories.

## 2026-07-08 FENIKS Results Readiness Audit

- Status: completed for the current local-readiness audit.
- Goal: determine which FENIKS artifacts are actually available locally after
  the log/artifact pulls, whether inference/J-lens/corner outputs have been
  produced, and whether the FENIKS synthetic closure is validated well enough
  to proceed with detailed analysis.
- Assumptions:
  - Training logs alone are not enough for posterior/corner analysis; dedicated
    inference and J-lens stages must exist or be launched.
  - Closure status should be inferred from FENIKS generation/validation logs and
    manifest/report artifacts, not from NN training loss alone.
- Findings:
  - The main 14-band FENIKS synthetic closure dataset is locally present and
    validated: `validation_report.json` passes gates for 40k/5k/5k
    train/validation/test rows, 18 truth columns, normalized noise residuals,
    metallicity checks, photometric S/N selection, and DSPS flux recomputation.
  - The five amortized ladder checkpoints are present with `best.eqx` and
    feature stats. AE, joint fixed-KL, joint annealed-KL, and frozen supervised
    prior runs completed with no skipped updates; supervised-prior fine-tune
    completed but skipped 908 non-finite updates and should remain a secondary
    diagnostic run.
  - No amortized inference outputs or J-lens outputs were found locally:
    `posterior_summary.parquet`, `posterior_samples.parquet`,
    `inference_summary.json`, `jacobian_lens_summary.json`, and lens object
    tables are absent. The next required stages are the `*_infer` and
    `*_jlens` ladder stages.
  - The supervised RealNVP truth-prior checkpoint is not scientifically usable
    as a generative prior in its current form. Its NLL improves, but
    truth-vs-prior diagnostics show very large KS/Wasserstein distances, and
    direct latent samples from the checkpoint have amplitudes around
    `1e5`-`1e6` instead of order-unity truth `x` values. Treat the frozen and
    fine-tune supervised-prior amortized runs as suspect until the supervised
    prior is fixed or replaced.

## 2026-07-07 Amortized Multi-GPU Training

- Status: completed for this implementation slice.
- Goal: add optional JAX `pmap` data-parallel training paths for amortized
  NN+DSPS runs and supervised RealNVP prior learning, while preserving the
  current single-GPU path as the default.
- Design:
  - Add `amortized.training.data_parallel` with values `single`, `auto`, and
    `pmap`.
  - Keep `single` as the default. `auto` uses pmap only when more than one
    local JAX device is visible. `pmap` requires at least two local devices.
  - Treat `batch_size` / `jax_batch_size` as global batch sizes. In pmap mode,
    require the JAX batch size to be divisible by the local device count.
  - Use replicated model/optimizer state, shard each batch over local devices,
    average gradients with `jax.lax.pmean`, and write the effective parallel
    mode/device count in logs and summaries.
  - Do not silently drop tail batches; pad epoch order by resampling existing
    rows with replacement and record the padded count.
- Completed:
  - Added `amortized.training.data_parallel` and CLI `--data-parallel` with
    `single`, `auto`, and `pmap`.
  - Added Equinox/JAX `filter_pmap` train steps for amortized NN+DSPS and
    supervised RealNVP prior learning. Both replicate model and optimizer
    state, shard the global batch across local devices, average
    loss/metrics/gradients with `jax.lax.pmean`, and conditionally skip
    non-finite updates without applying AdamW weight decay. The amortized path
    also applies the existing encoder/prior/calibration gradient masks.
  - Added epoch-order padding in pmap mode by resampling existing rows with
    replacement, with `data_parallel_epoch_padded_rows` recorded in the
    training log.
  - Added data-parallel metadata to training logs, training summaries, and
    checkpoint sidecars/config sidecars.
  - Set the new FENIKS amortized and supervised-prior experiment overlays to
    `data_parallel: auto`.
  - Updated `scripts/feniks_prior_ladder_h100.slurm` to pass
    `--data-parallel "$DATA_PARALLEL"` and documented multi-GPU launch commands
    using `sbatch --gres=gpu:4`.
  - Added CPU tests for mode resolution, single-device fallback, forced-pmap
    guardrails, epoch padding, batch sharding, and CLI parsing.
- Remaining caveat:
  - The pmap path is implemented and unit-tested for guardrails/sharding, but
    a real multi-H100 runtime smoke must be launched on a multi-GPU allocation.
    The local machine exposes one CPU JAX device only.
- Jean-Zay smoke fix:
  - Replaced removed `jax.device_put_replicated` usage in both pmap training
    paths with explicit leading-axis replication via `jnp.broadcast_to`, which
    is compatible with `eqx.filter_pmap` inputs on recent JAX.
  - Added a CPU regression test for the replication helper.
  - Validated with targeted ruff, data-parallel pytest, prior/CLI pytest, and
    `python -m compileall euclid_dsps scripts`.
- Jean-Zay pmap static-leaf fix:
  - Updated amortized and supervised-prior `eqx.filter_pmap` axes to use
    `eqx.if_array(0)` for replicated pytrees, so array leaves are mapped over
    local devices while Python/static leaves such as MLP activation functions
    remain broadcast/static.
  - Added a fake three-device CPU pmap smoke for `RealNVPPrior`, covering the
    `gelu` static leaf failure seen on Jean-Zay.
  - Validated with the fake three-device CPU smoke, targeted ruff, targeted
    pytest, `git diff --check`, and `python -m compileall euclid_dsps scripts`.
- Jean-Zay supervised-prior report plotting fix:
  - Made supervised prior report/snapshot diagnostics robust
    when prior/truth columns are constant or nearly constant after finite-row
    filtering, and avoid Matplotlib/corner histogram color crashes.
  - Replaced supervised prior `corner.corner` calls with an internal
    corner-like Matplotlib plotter that computes explicit ranges after finite
    filtering, handles one-sided constant distributions, caps plotted rows, and
    preserves corner metadata.
  - Added a regression test for a prior column that is constant while truth is
    dynamic, matching the report failure mode seen on Jean-Zay.
  - Validated with supervised-prior diagnostics tests, pmap/CLI tests, targeted
    ruff, `git diff --check`, and `python -m compileall euclid_dsps scripts`.
- Jean-Zay amortized snapshot plotting fix:
  - Fixed the amortized training snapshot corner-like diagonal
    histograms, which built duplicate `[x_name, x_name]` dataframes and caused
    Matplotlib to see two datasets with one color during `supervised_frozen_train`.
  - Added a regression test that writes the training snapshot corner-like plot
    with truth/prior/posterior overlays and verifies the diagonal histogram
    path no longer crashes.
  - Validated with data-parallel tests, supervised-prior/CLI tests, targeted
    ruff, `git diff --check`, and `python -m compileall euclid_dsps scripts`.
- Validation:
  - `python -m compileall euclid_dsps scripts` passed.
  - `uvx ruff check euclid_dsps/amortized/train.py euclid_dsps/amortized/config.py euclid_dsps/prior_learning/train.py euclid_dsps/cli.py tests/test_amortized_data_parallel.py tests/test_amortized_jacobian_lens.py tests/test_amortized_jacobian_lens_cli.py tests/test_amortized_flows.py tests/test_prior_learning_supervised.py` passed.
  - `python -m pytest -q tests/test_amortized_data_parallel.py tests/test_amortized_jacobian_lens.py tests/test_amortized_jacobian_lens_cli.py tests/test_amortized_flows.py tests/test_prior_learning_supervised.py tests/test_amortized_diagnostics.py tests/test_cli.py tests/test_amortized_likelihood.py` passed with 30 tests and the existing two NumPy constant-column warnings.
  - The five FENIKS amortized experiment overlays and the supervised-prior
    overlay load with `data_parallel: auto` and per-band zero-point calibration
    disabled.
  - `bash -n scripts/feniks_prior_ladder_h100.slurm` passed.
  - `python -m euclid_dsps.cli amortized-train-diffsky --help` exposes
    `--data-parallel {single,auto,pmap}`.
  - `uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -W --keep-going -b html docs/source docs/build/html` passed.
  - `git diff --check` passed.

## 2026-07-07 Physical Jacobian Lens + FENIKS Prior-Learning Ladder

- Status: completed for this implementation slice.
- Goal: add a focused, testable Physical Jacobian Lens for the active
  FENIKS NN+DSPS+NF pipeline, with reusable Jacobian diagnostics, a dedicated
  sharded CLI, lightweight training/prior snapshot hooks, 18D-aware diagnostic
  plots, experiment configs, and a Jean-Zay runbook.
- Initial assumptions:
  - Latent dimensionality must come from `latent_spec_from_config` and
    checkpoint metadata, not from hard-coded FENIKS defaults.
  - Band count must come from loaded config/data/filter definitions, not from
    hard-coded 14-band assumptions.
  - The main FENIKS production-like config currently uses 14 photometric bands
    and the 18D `diffsky_dsps_closure_full` latent.
  - Per-band zero-point calibration remains disabled for all new experiment
    configs in this phase.
  - The amortized ELBO prior term remains `E_q[logq - logp_beta]`
    (`q_to_p`); this phase adds explicit metadata/logging but does not change
    the KL convention.
  - Multi-GPU training is not a default requirement. The mandatory parallel
    path is sharded J-lens/inference diagnostics with single-GPU training as
    the safe fallback.
- Completed:
  - Added shared photometric effective-sigma helper so likelihood and J-lens
    use the same error-floor/jitter convention.
  - Added `euclid_dsps.amortized.jacobian_lens` with generic decoder/AE
    Jacobian core, full-SVD latent directions, exact nullspace accounting,
    physical loadings, posterior-variance summaries, prior-score projections,
    sharded DSPS wrapper outputs, compact top-k tables, JSON manifests, and
    summary plots.
  - Added `amortized-jacobian-lens-diffsky` and
    `amortized-finalize-jacobian-lens` CLI commands.
  - Added optional amortized training snapshots for encoder summaries,
    posterior/prior theta samples, lightweight corner-style plots, KL/prior
    metadata, and explicit deterministic/no-KL interpretation notes.
  - Added supervised truth-prior snapshots and epoch checkpoints through a
    guarded epoch callback.
  - Added 18D FENIKS full/useful diagnostic ordering, capped corner rows, and
    plot metadata for supervised prior diagnostics.
  - Added RealNVP identity initialization and checkpoint sidecar metadata, plus
    latent-spec checkpoint restoration checks for raw center/scale and
    normalization.
  - Added explicit amortized KL metadata/logging:
    `E_q[logq - logp_beta]`, `q_to_p`, prior source, training flag, freeze
    epochs, update schedule, and update phase.
  - Added six experiment overlays under `configs/experiments/`, all with
    per-band zero-point calibration disabled.
  - Added `scripts/feniks_prior_ladder_h100.slurm` and
    `docs/source/feniks_prior_ladder_jlens.rst` for the Jean-Zay ladder and
    sharded J-lens path.
- Remaining caveats:
  - Superseded by the later “Amortized Multi-GPU Training” slice above: this
    J-lens slice originally kept training single-GPU and only added sharded
    J-lens diagnostics.
  - In-training J-lens is intentionally not run inline; when requested in the
    snapshot config it writes a `SKIPPED.json` pointer to the dedicated sharded
    CLI path. This avoids recompiling DSPS Jacobians inside the training loop.
  - Full FENIKS/H100 runtime smoke commands were not run locally; validation
    below is CPU/unit/doc oriented.
- Validation:
  - `python -m compileall euclid_dsps scripts` passed.
  - `bash -n scripts/feniks_prior_ladder_h100.slurm` passed.
  - `uvx ruff check euclid_dsps/amortized/train.py euclid_dsps/amortized/jacobian_lens.py euclid_dsps/amortized/flows.py euclid_dsps/amortized/likelihood.py euclid_dsps/amortized/diagnostics.py euclid_dsps/prior_learning/train.py euclid_dsps/prior_learning/diagnostics.py euclid_dsps/cli.py tests/test_amortized_jacobian_lens.py tests/test_amortized_jacobian_lens_cli.py tests/test_amortized_flows.py tests/test_prior_learning_supervised.py` passed.
  - `python -m pytest -q tests/test_amortized_jacobian_lens.py tests/test_amortized_jacobian_lens_cli.py tests/test_amortized_flows.py tests/test_prior_learning_supervised.py tests/test_amortized_diagnostics.py tests/test_cli.py tests/test_amortized_likelihood.py` passed with 23 tests and two pre-existing NumPy constant-column warnings.
  - All six new experiment configs load through `euclid_dsps.config.load_config`
    and report 14 bands with per-band zero-point calibration disabled.
  - `python -m euclid_dsps.cli --help` passed and lists the new J-lens
    commands.
  - `uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -W --keep-going -b html docs/source docs/build/html` passed.
  - `git diff --check` passed.
- Verification follow-up:
  - Superseded by the later “Amortized Multi-GPU Training” slice above; pmap
    amortized training is now available as an optional path.
  - Rechecked that sharded J-lens uses `--num-shards` / `--shard-index` and
    `SLURM_ARRAY_TASK_COUNT` / `SLURM_ARRAY_TASK_ID`.
  - Re-ran config loading for all six experiment overlays; each resolves with
    14 bands and per-band zero-point calibration disabled.
  - Re-ran `python -m compileall euclid_dsps scripts`, targeted `uvx ruff`,
    targeted `pytest`, `bash -n scripts/feniks_prior_ladder_h100.slurm`,
    `python -m euclid_dsps.cli --help`, Sphinx, and `git diff --check`; all
    passed, with only the existing NumPy constant-column warnings in the
    supervised prior diagnostic test.

## 2026-07-07 Active FENIKS Source Tree Cleanup

- Status: completed.
- Goal: move non-essential source, configs, scripts, docs, and tests to
  `legacy/` so the active tree focuses on FENIKS synthetic data, supervised
  and inferred prior learning, NN+DSPS+NF training/inference, MAP under learned
  prior, direct MCLMC baselines, FS2 comparison, and HLTDS download/preparation.
- Scope for this phase:
  - Keep active package code needed by `diffsky-generate-dsps-closure`,
    `diffsky-validate-dsps-closure`, `diffsky-plan-prior-workflow`,
    `diffsky-train-supervised-prior`, `amortized-train-diffsky`,
    `amortized-infer-diffsky`, `diffsky-map-adam-prior`, `posterior`, FS2
    train/infer, and HLTDS `diffsky-*` data preparation commands.
  - Move OpenUniverse, COSMOS SED helpers, deprecated facades, old HLTDS
    experiment configs/scripts, reconstruction dashboards, and matching tests
    out of the active package/test/doc surface.
  - Preserve moved files under `legacy/` for reference rather than deleting
    them.
  - Keep `python -m pytest -q`, CLI help, Sphinx, bash syntax, and diff hygiene
    green after the move.
- Completed:
  - Moved OpenUniverse, COSMOS SED reconstruction helpers, deprecated
    pipeline/report/workflow facades, exact forward-closure runners, HLTDS
    experiment matrices, reconstruction/ablation diagnostics, broad benchmark
    scripts, and matching tests/docs/configs under `legacy/`.
  - Reduced the active config surface to FENIKS generation/validation, FENIKS
    supervised prior, FENIKS amortized NN+DSPS+NF, FS2 comparison, and HLTDS
    download/preparation/debug configs.
  - Replaced legacy module dependencies with small active helpers:
    `synthetic_diffsky.truth_theta` for the 18D truth vector and
    `amortized.redshift_metrics` for held-out photo-z/posterior diagnostics.
  - Removed the active `cosmos_sed` default contract and OpenUniverse band
    presets; PopCosmos/HLTDS semantics remain only where still required by the
    active model/debug paths.
  - Updated README, config docs, architecture docs, API docs, run setup,
    testing guidance, and the FENIKS workflow planner so validation uses the
    same FENIKS generation config instead of legacy `trueparam_closure` configs.
  - Relaxed and recorded the FENIKS DSPS recomputation tolerance to `5e-4`
    mag/relative flux so CPU validation remains reproducible while still being
    far below the photometric noise scale.
- Validation:
  - `python -m euclid_dsps.cli --help` passed.
  - `bash -n` passed for all active H100 launchers.
  - `uvx ruff check euclid_dsps tests scripts` passed.
  - `python -m compileall euclid_dsps scripts tests` passed.
  - `python -m pytest -q` passed with `279 passed, 9 skipped`.
  - `uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -W --keep-going -b html docs/source docs/build/html` passed.
  - `python -m euclid_dsps.cli --config configs/diffsky_synthetic_feniks_260617_50k.yaml diffsky-plan-prior-workflow --out /tmp/feniks_prior_workflow_review_clean` passed and reports the 40k/5k/5k dataset contract ready.
  - `env JAX_PLATFORMS=cpu PYTHONPATH=. conda run -n shine python -m euclid_dsps.cli --config configs/diffsky_synthetic_feniks_260617_50k.yaml diffsky-validate-dsps-closure --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure --sample-size 16 --batch-size 16 --runtime cpu` passed.
  - `git diff --check` passed.

## 2026-07-07 FENIKS Primary Surface Cleanup

- Status: completed for this cleanup slice.
- Goal: make FENIKS the default controlled dataset/experiment across the
  runnable prior-learning surface while keeping HLTDS and FS2 only as explicit
  debug/reference paths.
- Scope for this phase:
  - Remove HLTDS defaults and implicit HLTDS dataset rebuilds from the H100
    launchers used by the FENIKS prior-learning ladder.
  - Keep launchers fail-fast: if a FENIKS split/checkpoint is missing, the job
    should stop rather than synthesize a different experiment.
  - Make README and core docs show FENIKS commands as the copy-paste surface.
  - Remove small prior-learning duplication that obscures the train/validation
    contract.
  - Rename the planner readiness text so existing dataset-contract readiness
    is not confused with downstream checkpoint availability.
- Completed:
  - Updated `scripts/diffsky_amortized_train_h100.slurm`,
    `scripts/diffsky_amortized_infer_h100.slurm`,
    `scripts/diffsky_map_adam_prior_h100.slurm`,
    `scripts/diffsky_flat_mclmc_calibration_h100.slurm`, and
    `scripts/diffsky_inferred_prior_h100.slurm` to default to FENIKS configs,
    splits, output names, and checkpoints.
  - Removed the implicit HLTDS redshift-subset rebuild path from amortized
    inference.
  - Added shared prior-learning split helper in
    `euclid_dsps/prior_learning/splits.py`.
  - Rewrote the README around the FENIKS production ladder and relabeled
    HLTDS/FS2 as debug/reference.
  - Updated `docs/source/index.rst`, `docs/source/prior_learning.rst`, and
    `docs/source/amortized_inference.rst` so copy-paste examples point to
    FENIKS and remaining HLTDS material is marked debug/reference.
  - Updated the HLTDS config test to check direct/projected/missing truth
    semantics rather than pretending the debug config has only a basic direct
    truth set.
  - Regenerated the planner in `/tmp/feniks_prior_workflow_review`; it reports
    `Dataset contract ready: true`, 40k/5k/5k splits present, and downstream
    prior/NN/MAP stages waiting until checkpoints exist.
- Remaining gap:
  - Direct MCLMC still uses the configured physical priors as a calibration
    baseline. MAP under the learned RealNVP prior is available; true MCLMC
    under the learned NF prior still needs a posterior-target extension.
- Validation:
  - `python -m compileall euclid_dsps/prior_learning euclid_dsps/cli.py scripts tests/test_prior_learning_workflow.py tests/test_cli.py tests/test_config.py` passed.
  - `bash -n scripts/diffsky_amortized_train_h100.slurm scripts/diffsky_amortized_infer_h100.slurm scripts/diffsky_map_adam_prior_h100.slurm scripts/diffsky_flat_mclmc_calibration_h100.slurm scripts/diffsky_inferred_prior_h100.slurm` passed.
  - `uvx ruff check euclid_dsps/prior_learning tests/test_prior_learning_workflow.py tests/test_cli.py tests/test_config.py` passed.
  - `python -m pytest -q tests/test_prior_learning_workflow.py tests/test_cli.py tests/test_prior_learning_supervised.py tests/test_prior_learning_inferred.py` passed with `13 passed` and two NumPy constant-column warnings.
  - `python -m pytest -q tests/test_config.py::test_diffsky_hltds_simple_config_marks_debug_truth_contract` passed.
  - `python -m euclid_dsps.cli --config configs/diffsky_synthetic_feniks_260617_50k.yaml diffsky-plan-prior-workflow --out /tmp/feniks_prior_workflow_review` passed.
  - `uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -W --keep-going -b html docs/source docs/build/html` passed.
  - `python -m euclid_dsps.cli --help` passed.
  - `python -m pytest -q` passed with `387 passed, 10 skipped`.
  - `git diff --check` passed.

## 2026-07-07 FENIKS Prior Workflow Architecture Cleanup

- Status: completed for this cleanup slice; broader architecture goal remains
  active.
- Goal: make the new clean FENIKS/DSPS closure dataset easy to use for the
  full inference ladder: supervised NF prior on closure truths, amortized
  NN+DSPS+NF training, post-hoc NF priors from MAP/MCLMC, and MAP/MCLMC
  redshift inference under a learned prior.
- Scope for this phase:
  - Create a dedicated branch for the cleanup work.
  - Add a code-level workflow contract that resolves the configured FENIKS
    dataset splits, truth schema, prior checkpoints, NN checkpoints, and
    command recipes before expensive GPU jobs are launched.
  - Keep the existing prior-learning, amortized, MAP-Adam, and posterior/MCLMC
    implementations working; this phase is a discovery/orchestration cleanup,
    not a rewrite of the science kernels.
  - Add focused tests and docs so the entry point is understandable from the
    CLI and production runbook.
- Completed:
  - Created branch `feature/feniks-prior-workflow-cleanup`.
  - Added `diffsky-plan-prior-workflow`, backed by
    `euclid_dsps.prior_learning.workflow`, to audit the FENIKS dataset splits,
    18D closure truth schema, validation artifacts, expected prior/NN
    checkpoints, and launch order before expensive GPU jobs.
  - The planner writes `workflow_plan.md` and `workflow_plan.json` with
    copy-paste commands for generation, validation, supervised NF prior,
    NN+DSPS+NF training, held-out amortized inference, MAP under learned prior,
    direct MCLMC calibration, and post-hoc inferred-prior training.
  - Added explicit `--dataset` overrides for `amortized-train-diffsky`,
    `amortized-infer-diffsky`, `diffsky-map-adam-prior`, and `posterior`, so
    held-out validation/test parquets no longer require editing config files.
  - Updated H100 launchers to pass dataset/prior-checkpoint overrides through
    to the CLI; the MCLMC launcher can now create a small rowset from
    `DATASET` when no rowset file is supplied.
  - Documented the preflight in `README.md`,
    `docs/source/production.rst`, and `docs/source/prior_learning.rst`.
  - Ran the planner against the real local FENIKS config; it confirms the
    40k/5k/5k dataset and validation artifacts are present and the supervised
    prior stage is the next ready stage. The prior and amortized checkpoints
    are not present yet, so downstream NN/MAP stages are correctly marked
    waiting.
- Remaining gap:
  - Direct MCLMC still uses the configured physical priors as a calibration
    baseline. MAP under the learned RealNVP prior is available; true MCLMC
    under the learned NF prior still needs a posterior-target extension.
- Validation:
  - `python -m compileall euclid_dsps/prior_learning euclid_dsps/cli.py scripts tests/test_prior_learning_workflow.py tests/test_cli.py` passed.
  - `python -m pytest -q tests/test_prior_learning_workflow.py tests/test_cli.py` passed with `5 passed`.
  - `python -m pytest -q tests/test_prior_learning_supervised.py tests/test_prior_learning_inferred.py tests/test_prior_learning_workflow.py` passed with `10 passed` and two constant-column NumPy warnings.
  - `python -m pytest -q tests/test_cli.py tests/test_map_prior_sweep.py tests/test_amortized_prior_source.py` passed with `8 passed`.
  - `python -m pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py tests/test_prior_learning_workflow.py` passed with `22 passed, 2 skipped`.
  - `bash -n scripts/diffsky_amortized_train_h100.slurm scripts/diffsky_amortized_infer_h100.slurm scripts/diffsky_map_adam_prior_h100.slurm scripts/diffsky_flat_mclmc_calibration_h100.slurm` passed.
  - `uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -W --keep-going -b html docs/source docs/build/html` passed.
  - `git diff --check` passed.

## 2026-07-07 FENIKS Diffsky Feedback Bundle

- Status: completed.
- Goal: assemble a lightweight share folder for Andrew/Kumail with the
  FENIKS/DSPS closure generation contract, validation evidence, FS2 comparison
  diagnostics, key plots, and a compact parquet sample, without copying the
  full generated dataset into an email-sized package.
- Scope:
  - Include the exact generation/closure configs, schema, manifest, validation
    report, population summary, FS2 comparison reports, and selected plots.
  - Add a small sampled parquet preserving truths, weights, provenance, and
    photometry columns for quick external inspection.
  - Add a README explaining that Diffsky/FENIKS is used for the latent
    population while euclid_dsps regenerates closure photometry.
- Completed:
  - Created `outputs/share/feniks_diffsky_feedback_20260707/` with configs,
    metadata, validation report, population diagnostics, corrected FS2 phz1
    comparison reports, selected plots, and two 5k-row parquet samples.
  - Added `README.md` and `FILELIST.txt` inside the bundle so it is
    self-describing when shared externally.
  - Created `outputs/share/feniks_diffsky_feedback_20260707.tar.gz` as a
    15 MB archive suitable for transfer.
- Validation:
  - Verified the share folder is 17 MB and the archive is 15 MB.
  - Verified the archive file listing contains the expected configs,
    diagnostics, plots, metadata, and samples.

## 2026-07-07 FS2 Comparison Unit Fix

- Status: completed.
- Goal: correct the FS2 reference comparison without regenerating the FENIKS
  closure catalogues, so LSST/Euclid color-color and magnitude diagnostics use
  AB magnitudes derived from FS2 `fnu_cgs` flux columns.
- Scope:
  - Detect `reference_kind: fs2` photometry columns as `fnu_cgs` fluxes, not
    apparent magnitudes.
  - Materialize comparable `flux_<band>` and `mag_<band>` columns for FS2 while
    preserving the original columns.
  - Rename FS2 reports/output metadata so the comparison is clearly against
    Euclid FS2 phz1, not the older z<=0.35 HLTDS reference.
  - Add tests covering FS2 flux-to-AB conversion and color metrics.
  - Leave Diffstar/Diffmah atom-like nuisance dimensions available to inference
    while documenting that they should not be interpreted as smooth recovered
    science parameters.
- Completed:
  - `reference_kind: fs2` now materializes `flux_<band>` and `mag_<band>` for
    LSST and Euclid FS2 `fnu_cgs` columns before computing magnitude/color
    metrics and plots.
  - FS2 reports now label the reference as `Euclid FS2 phz1` and record which
    reference bands were converted from flux to AB magnitude.
  - The CLI help no longer describes the comparison command as HLTDS-only.
  - Added a regression test proving that FS2 flux columns are converted to AB
    magnitudes before magnitude and color metrics are computed.
- Validation:
  - `python -m compileall euclid_dsps/synthetic_diffsky/reference_comparison.py tests/test_synthetic_diffsky_closure.py`
    passed.
  - `python -m pytest -q tests/test_synthetic_diffsky_closure.py::test_reference_comparison_converts_fs2_flux_columns_to_ab_magnitudes tests/test_synthetic_diffsky_closure.py::test_reference_comparison_writes_population_and_photometry_tables`
    passed with `2 passed`.
  - `python -m compileall euclid_dsps` passed.
  - `python -m pytest -q tests/test_synthetic_diffsky_closure.py` passed with
    `17 passed, 2 skipped`.
  - Corrected local FS2 comparisons were regenerated for both `survey_like` and
    `inference_ready` outputs without rerunning Diffsky/DSPS generation.

## 2026-07-06 Survey-Like FENIKS Closure Dataset Plan

- Status: completed.
- Goal: evolve the FENIKS/DSPS closure generator from a single
  inference-ready 50k catalog into a layered dataset product with explicit
  raw weighted proposals, survey-like observable selection, and inference-ready
  subsets.
- Proposed scope:
  - Add configurable observable selections based on magnitude limits, S/N, and
    minimum detected-band counts, while avoiding a hard stellar-mass cut in the
    final survey-like catalog unless it is explicitly marked as a technical
    guardrail.
  - Preserve `raw_weighted`, `survey_like`, and `inference_ready` outputs, each
    with separate manifests, selection counters, ESS, duplication, and plots.
  - Add FS2/OpenUniverse-style LSST color-color comparisons, using only bands
    with a like-for-like filter definition unless a deliberate filter
    conversion experiment is configured.
  - Keep the HLTDS/FENIKS SSP and filter assets as the default closure assets;
    treat alternate SSP grids as domain-shift experiments unless the full
    generation/inference/validation stack is regenerated consistently.
- Completed:
  - Added magnitude-limit based photometric selection in addition to the
    existing S/N-count gates.
  - Added configurable output layers: raw weighted proposals remain under
    `proposals/`, `survey_like/` is written with looser observable cuts, and
    `inference_ready/` is mirrored to the dataset root for compatibility with
    validation and inference commands.
  - Added LSST+Euclid and LSST+Euclid+Roman band presets using existing
    repository filter assets.
  - Added 18-band production and validation configs, plus a Jean-Zay H100
    wrapper script.
  - Added FS2-aware reference comparison support, including FS2 column aliases
    and LSST/Euclid color-color diagnostics.
  - Updated the synthetic closure docs and config README with the layered
    dataset contract and FS2 comparison workflow.
- Validation:
  - `python -m compileall euclid_dsps scripts` passed.
  - `python -m pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    passed with `19 passed, 2 skipped`.
  - `git diff --check` passed.
  - `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm scripts/diffsky_synthetic_feniks_18band_h100.slurm`
    passed.
  - Full `python -m pytest -q` ran to completion with `383 passed, 10 skipped`
    and one pre-existing legacy config failure in
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`;
    the failing assertion is on `configs/diffsky_hltds_04_14_simple_gpu.yaml`
    truth columns, not the new FENIKS 18-band path.
  - A real local Diffsky smoke was not run because this local environment lacks
    `diffsky`, `diffstar`, `diffmah`, and `diffhalos`; run the provided smoke
    command on Jean-Zay where the FENIKS stack is installed.

## 2026-07-06 FENIKS 50k Result Analysis

- Status: completed.
- Goal: analyze the locally retrieved z<=5.5 FENIKS/DSPS closure dataset,
  validation report, population diagnostics, reference comparison, and plots
  before using it for prior learning or amortized inference.
- Scope:
  - Check manifest split sizes, ESS, duplication, proposal and photometric
    selection counters.
  - Check validation gates, closure recomputation, noise residuals, metallicity
    support, and S/N cuts.
  - Inspect distribution and color-color diagnostics for realism and remaining
    caveats.
- Findings:
  - Closure validation passes on the 50k catalog: exact split sizes, 18 truth
    columns, no metallicity clipping, exact DSPS flux recomputation on the
    validation sample, and normalized noise residuals close to N(0, 1).
  - The photometric selection makes the sample much more learnable than the
    first broad run, with at least five true S/N>5 bands and a median of eleven.
  - The population is not a low-redshift HLTDS match: the z<=0.35 overlap is
    much lower mass/fainter than the reference and has different correlations.
  - Main caveats before prior/inference work are weighted-resampling
    duplicates, low-mass/star-forming bias, central-dominated composition, and
    boundary spikes/atoms in several Diffstar/Diffmah truth dimensions.

## 2026-07-06 FENIKS Production Duplication Gate

- Status: completed.
- Goal: unblock the Jean-Zay z<=5.5 production generation where the selected
  train pool reached `2.4M` proposals and `ESS=1.04e5`, but the hard
  duplication gate still failed at `0.154` after all 256 shards.
- Scope:
  - Keep ESS and selected-pool-size gates mandatory.
  - Treat the configured duplication threshold as a warning after `max_shards`
    is exhausted, because weighted resampling from a high-dynamic-range FENIKS
    proposal has intrinsic repeated source proposals.
  - Preserve manifest reporting of the measured duplication fraction and
    provide a resume command that reuses the already written train shards.
- Completed:
  - Added `synthetic_diffsky.duplication_gate` with modes `fail`, `warn`, and
    `warn_after_max_shards`.
  - Set production to `duplication_gate: warn_after_max_shards` while keeping
    `max_duplication_fraction: 0.10` as an alert threshold.
  - Kept selected-pool-size and ESS gates mandatory; only the duplication gate
    can become a warning after `max_shards` is exhausted.
  - Added `pool_duplicate_fraction`, `max_duplication_fraction`,
    `duplication_gate`, and `resampling_duplicate_warning` manifest fields.
  - Updated the synthetic closure and production docs to describe the warning
    semantics.
- Validation completed:
  - `python -m compileall euclid_dsps scripts` passed.
  - `python -m compileall euclid_dsps/synthetic_diffsky` passed after the final
    gate fix.
  - `pytest -q tests/test_synthetic_diffsky_closure.py::test_duplication_gate_warns_only_after_max_shards`
    passed.
  - `pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    passed with `16 passed, 1 skipped`.
  - `git diff --check` passed.

## 2026-07-03 Production Documentation Cleanup

- Status: completed.
- Goal: remove outdated production guidance from the public documentation path,
  make the FENIKS/DSPS closure workflow the documented production route, and
  keep historical HLTDS/debug plans clearly separated from production
  acceptance gates.
- Current scope:
  - Add a production runbook with diagrams, canonical paths, commands,
    acceptance gates, and scientific limits.
  - Update the README, config README, Sphinx index, architecture, run setup,
    prior learning, amortized inference, forward model, science assessment,
    installation, and testing pages.
  - Mark historical plan/experiment pages as orphaned historical records so
    they are no longer presented as the production workflow.
  - Rebuild Sphinx with warnings as errors and run lightweight code/docs
    checks.
- Completed:
  - Added ``docs/source/production.rst`` as the production runbook for the
    FENIKS/DSPS closure workflow.
  - Updated the README, config README, Sphinx index, architecture, run setup,
    prior-learning, amortized-inference, forward-model, science-assessment,
    data/assets, HLTDS dataset, FS2 catalog-column, installation, and testing
    pages to put the FENIKS/DSPS closure workflow first and label HLTDS/MCLMC
    material as reference, debug, or historical where appropriate.
  - Marked ``diffsky_nn_experiment_matrix.rst``,
    ``diffsky_robust_prior_plan.rst``, and
    ``scientific_validation_plan.rst`` as orphaned historical records.
  - Cleaned the ``amortized-train-diffsky`` and ``amortized-infer-diffsky``
    CLI help labels so they are no longer HLTDS-only.
- Validation completed:
  - ``python -m compileall euclid_dsps scripts`` passed.
  - ``uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -W
    --keep-going -b html docs/source docs/build/html`` passed.
  - ``python -m euclid_dsps.cli --help`` passed and shows generic Diffsky
    amortized labels.
  - ``python -m pytest -q tests/test_cli.py tests/test_synthetic_diffsky_closure.py``
    passed with ``15 passed, 1 skipped``.
  - ``git diff --check`` passed.

## 2026-07-03 Documentation Rebuild

- Status: completed.
- Goal: rebuild the Sphinx documentation after the FENIKS closure creation
  process documentation update and record any warnings or failures.
- Completed:
  - `make -C docs html` failed because the active Python environment does not
    have `sphinx` installed.
  - `uv run --with sphinx --with sphinx-rtd-theme python -m sphinx -b html
    docs/source docs/build/html` succeeded and wrote the rebuilt HTML under
    `docs/build/html`.

## 2026-07-03 FENIKS Closure Creation Process Documentation

- Status: completed.
- Goal: document the concrete dataset-creation process after the z<=5.5,
  metallicity, and observability-selection fixes, so the production run can be
  interpreted without reading the generator code.
- Scope:
  - Explain the difference between raw weighted Diffsky proposals, selected
    proposal pools, and final unweighted learning catalogues.
  - Document the current production cuts, DSPS closure photometry generation,
    error model, manifest fields, diagnostics, and validation gates.
  - Keep the documentation aligned with
    `configs/diffsky_synthetic_feniks_260617_50k.yaml`.
- Completed:
  - Added a `Creation Process` section to
    `docs/source/diffsky_synthetic_closure.rst` describing the nine-step
    production path from Diffsky proposal shards to validated DSPS closure
    catalogues.
  - Added the current production selection block to the commands section:
    `z_max: 5.5`, `logsm_true >= 8`, no final metallicity clipping, and at
    least five true S/N>=5 bands.
  - Clarified that `lgmet_abs_used_true` is stored and that
    `log10_stellar_metallicity_true` is audited as
    `lgmet_abs_used_true - log10(model.z_sun)`.

## 2026-07-03 FENIKS Regeneration Fix Implementation

- Status: completed.
- Goal: make the next zero-regeneration scientifically usable by fixing the
  metallicity convention, extending the production redshift proposal beyond
  `z=5`, and adding explicit selection/validation gates for metallicity-grid
  coverage and photometric informativeness.
- Current implementation targets:
  - Clip FENIKS absolute `log10(Z)` medians against the absolute SSP grid before
    converting to `log10(Z/Zsun)`.
  - Keep weighted proposal shards raw on disk, then report configurable
    proposal-selection and photometric-selection counts in the manifest.
  - Default production to `z_max: 5.5`, `logsm_true >= 8`, no metallicity
    clipping in the final catalog, and at least five true-S/N detections above
    five sigma.
  - Validate SSP metallicity support, selected S/N gates, exact closure
    recomputation, and negative noisy-flux preservation.
- Completed:
  - Added `synthetic_diffsky.selection` with proposal-level mass/metallicity
    cuts and post-DSPS photometric S/N cuts; all cut sizes are written to the
    manifest.
  - Fixed the FENIKS metallicity conversion so absolute `log10(Z)` is clipped
    against the absolute SSP grid before conversion to `log10(Z/Zsun)`, and
    added `lgmet_abs_used_true` for convention auditing.
  - Updated production configuration to propose `0.001 <= z <= 5.5`, require
    `logsm_true >= 8`, require no clipped metallicities in the final catalog,
    and require at least five true bands with S/N >= 5.
  - Kept smoke runs lighter with `smoke_selection` so CPU preflight remains
    usable on small samples.
  - Updated validation to check the realized manifest selection, SSP
    metallicity support, absolute/relative metallicity consistency, S/N counts,
    and selected-proposal n(z) behavior.
  - Updated prior-learning redshift bounds to `[0.001, 5.5]` and refreshed the
    synthetic-closure documentation.
- Validation completed:
  - `python -m compileall euclid_dsps scripts` passed.
  - `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm` passed.
  - `git diff --check` passed.
  - `pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    passed with `15 passed, 1 skipped`.
  - Full `pytest -q` completed with `379 passed, 9 skipped` and the known
    existing failure
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`.

## 2026-07-03 FENIKS 50k Post-Run Audit and Regeneration Gate

- Status: completed for audit and regeneration planning; implementation fixes
  remain pending.
- Goal: audit the completed Jean-Zay 50k FENIKS DSPS-closure run before any
  zero-regeneration, identify scientific caveats, inspect available SSP grids,
  and define the fixes needed for a physically useful dataset.
- Current findings:
  - Closure recomputation passes because the generated truth vector and DSPS
    forward model are internally consistent, but this does not guarantee that
    every truth convention is physically correct.
  - The current metallicity transform appears to clip `log10(Z/Zsun)` values
    against an SSP grid stored in absolute `log10(Z)`, causing the final
    `log10_stellar_metallicity_true` convention to be wrong and the forward
    metallicity effectively too low.
  - The volume-complete/no-selection population contains many very low-mass and
    low-S/N galaxies, producing many non-detections and making the z<=0.35
    comparison to the current HLTDS reference strongly selection-mismatched.
  - Several Diffstar/Diffmah latent parameters have large point masses at
    calibration bounds or branch defaults, so the learned prior must handle
    atoms/mixtures rather than assuming a smooth 18D density.
- SSP audit:
  - All local closure-relevant stellar SSP grids share the same metallicity
    support, approximately `log10(Z)=[-4.3477,-1.3477]`, equivalent to
    `log10(Z/Zsun)=[-2.5,+0.5]` for `z_sun=0.0142`.
  - No local standard SSP grid currently expands the low-metallicity bound; a
    physically useful no-clipping FENIKS sample must therefore be selected away
    from the extremely low-metallicity/low-mass tail or use a newly generated
    SSP grid.
- Regeneration gate:
  - Fix metallicity conversion and add convention tests before any new
    production run.
  - Add configurable preselection/postselection gates for mass, metallicity
    grid coverage, magnitudes, S/N, and reference-calibrated diagnostics.
  - Treat the current 50k run as a debug artifact only; do not train the prior
    or inference model on it.

## 2026-07-03 FENIKS Full Run Criteria and OpenUniverse-Style Diagnostics

- Status: completed.
- Goal: make the 50k FENIKS generation pass with the observed weighted
  proposal behavior on Jean-Zay and improve diagnostics with clearer
  OpenUniverse-style color-color overlays.
- Completed:
  - Relax only the duplication gate needed by the observed full train run while
    preserving ESS reporting and exact manifest accounting.
  - Make resume safer when stale smoke final parquets are present.
  - Add black/reference and green/synthetic color-color scatter overlays for
    LSST/Roman color pairs.
  - Validate, commit, push, and provide clean Jean-Zay relaunch commands.
  - Validation completed: `python -m compileall euclid_dsps/synthetic_diffsky
    euclid_dsps/cli.py`, `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm`,
    `git diff --check`, and `pytest -q
    tests/test_synthetic_diffsky_closure.py tests/test_cli.py` (`13 passed, 1
    skipped`).
  - Full `pytest -q` completed with `377 passed, 9 skipped` and the known
    existing HLTDS config failure
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`.

## 2026-07-02 Jean-Zay Diffsky Backend Dependency Diagnostics

- Status: completed.
- Goal: expose the real missing dependency behind the generic Diffsky backend
  `ImportError` seen on Jean-Zay during proposal generation.
- Completed:
  - Preserve the original Python import error in the raised backend exception.
  - Add optional `DIFFHALOS_REPO` support to the H100 SLURM launcher because
    Diffsky's analytic lightcone generator imports `diffhalos`.
  - Provide relaunch commands that install or point to both external Diffsky
    and Diffhalos.
  - Verified with `python -m compileall euclid_dsps/synthetic_diffsky/backend.py`,
    `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm`, and
    `git diff --check`.

## 2026-07-02 Jean-Zay Diffsky Import Preflight

- Status: completed.
- Goal: handle the Jean-Zay failure where the H100 SLURM job starts from the
  correct `shine` environment but cannot import the external `diffsky` package.
- Completed:
  - Add a `DIFFSKY_REPO` hook to the SLURM launcher so a local Diffsky clone can
    be used through `PYTHONPATH` without reinstalling the conda environment.
  - Keep the script compatible with an editable pip install of Diffsky.
  - Provide exact commands to clone/update Diffsky and relaunch generation plus
    validation.
  - Verified with `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm`
    and `git diff --check`.

## 2026-07-02 Sync Commit and Relaunch Instructions

- Status: completed.
- Goal: fetch/pull the current Diffsky likelihood branch, commit the FENIKS
  synthetic-closure implementation without unrelated worktree noise, push it,
  and provide exact relaunch commands.
- Completed:
  - Ran `git fetch origin`; the local branch was aligned with
    `origin/feature/diffsky-likelihood-sanity-plan` (`0/0` ahead/behind), so no
    merge/rebase was required.
  - Staged only the FENIKS closure generation, validation, diagnostics, config,
    docs, tests, and SLURM files.
  - Left unrelated worktree noise unstaged: deleted `PLAN_*`/`RAPPORT.md` files
    and the untracked `notebooks/` directory.
  - Re-ran validation before commit: `python -m compileall euclid_dsps`,
    `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm`, and
    `pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    (`13 passed, 1 skipped`).

## 2026-07-02 H100 Verbose Generation and Error-Model Diagnostics

- Status: completed for implementation and smoke validation.
- Goal: make the 50k FENIKS DSPS-closure production run easier to monitor on
  H100, ensure the configured photometric error model is visibly applied during
  dataset generation, and provide a SLURM launcher for generation plus closure
  validation.
- Completed:
  - Add verbose progress output for split/shard/resampling/photometry/diagnostic
    stages. The generation CLI is verbose by default and has `--quiet` for
    batch logs that need less detail.
  - Kept production plots enabled in
    `configs/diffsky_synthetic_feniks_260617_50k.yaml`.
  - Added explicit generated-error diagnostics tied to the configured flux
    error model: `error_model_stats.csv`,
    `plots/error_model_band_summary.png`,
    `plots/normalized_noise_residual_histograms.png`, and
    `plots/fluxerr_vs_mag_true.png`.
  - Added `scripts/diffsky_synthetic_feniks_50k_h100.slurm`, supporting
    `STAGE=generate`, `STAGE=validate`, and `STAGE=both`, with a guard against
    using `OVERWRITE=1` and `RESUME=1` together.
- Smoke validation:
  - Ran `conda run -n shine` smoke generation with
    `--split all --smoke --max-galaxies 24 --overwrite`; it produced 18/3/3
    rows and wrote the error-model plots/statistics.
  - Ran closure validation on the smoke dataset with `--sample-size 24
    --batch-size 24 --runtime cpu`; gates passed, normalized noise residuals
    passed, and flux recomputation reported `max_abs_delta_mag =
    1.9073486328125e-06` and `max_relative_flux_error =
    1.7567345522703518e-06`.
- Validation completed:
  - `python -m compileall euclid_dsps/synthetic_diffsky euclid_dsps/cli.py`
    passed.
  - `conda run -n shine python -m compileall euclid_dsps/synthetic_diffsky
    euclid_dsps/cli.py` passed.
  - `bash -n scripts/diffsky_synthetic_feniks_50k_h100.slurm` passed.
  - `git diff --check` on touched files passed.
  - `pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    passed with `13 passed, 1 skipped`.
  - Full `pytest -q` completed with `377 passed, 9 skipped` and the known
    existing failure
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`.

## 2026-07-02 Automatic FENIKS 50k Population Diagnostics

- Status: completed for implementation and smoke validation.
- Goal: make every production FENIKS DSPS-closure generation run write
  automatic population diagnostics for the requested 50k dataset, including
  redshift/mass/SFR/dust/metallicity parameter summaries, colors, photometry,
  proposal-vs-final checks, plots, and corner plots.
- Redshift decision:
  - Configure the production synthetic closure run for ``0.001 <= z <= 3.0``
    as a broad LSST+Roman/OpenUniverse-like photometric range.
  - Keep comparisons to the current local z<=0.35 HLTDS parquet restricted to
    the overlapping low-redshift interval, because a whole z<3 catalog should
    not be directly compared to a z<0.35 reference sample.
- Completed:
  - Added `euclid_dsps.synthetic_diffsky.population_diagnostics` and wired it
    into `generate_dsps_closure_dataset` after `all_50k.parquet` is written.
  - Added automatic tables:
    `parameter_stats.csv`, `photometry_stats.csv`, `color_stats.csv`,
    `proposal_vs_final_metrics.csv`, and `correlation_matrices.json`.
  - Added automatic plots:
    truth-parameter histograms, physical-diagnostic histograms, magnitude and
    color histograms, per-band photometry summary, mass/redshift/SFR/dust
    scatter diagnostics, and core/full truth corner plots. Small smoke samples
    use scatter-matrix fallbacks to avoid invalid contour warnings.
  - Added optional overlap-only reference comparison against the local z<=0.35
    HLTDS parquet under `diagnostics/population/reference_comparison/`.
  - Updated validation weighted n(z) bins to use the generated manifest
    redshift range instead of hard-coding z<=0.35.
  - Updated the FENIKS 50k config to `z_min: 0.001`, `z_max: 3.0`, redshift
    fit bounds `[0.001, 3.0]`, and enabled automatic diagnostics.
  - Documented the redshift choice and diagnostics contract in
    `docs/source/diffsky_synthetic_closure.rst`.
- Smoke validation:
  - Ran real Diffsky/FENIKS smoke generation with
    `--split all --smoke --max-galaxies 24 --overwrite` in `conda activate
    shine`; output split sizes are 18/3/3 and realized redshift range is
    `0.343904` to `2.85961`.
  - Verified diagnostics under
    `Data/diffsky/synthetic/feniks_260617_dsps_closure/diagnostics/population/`.
  - Ran closure validation on the z<3 smoke dataset; it passed.
- Validation completed:
  - `python -m compileall euclid_dsps/synthetic_diffsky euclid_dsps/cli.py`
    passed.
  - `python -m compileall euclid_dsps` passed.
  - `git diff --check` passed.
  - `pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    passed with `13 passed, 1 skipped`.
  - Full `pytest -q` completed with `377 passed, 9 skipped` and the same
    existing HLTDS config expectation failure:
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`.

## 2026-07-02 FENIKS Synthetic vs Current z0.35 Dataset Investigation

- Status: completed for smoke-scale investigation; production-scale 50k
  comparison remains pending.
- Goal: generate or reuse a small FENIKS synthetic DSPS-closure sample and
  compare its latent, redshift, mass, SFR, dust, metallicity, and photometric
  distributions against the current z<=0.35 Diffsky HLTDS reference dataset.
- Scope completed:
  - Identify the canonical local z0.35 reference parquet and its comparable
    columns.
  - Add a reproducible comparison diagnostic if the existing tools do not cover
    this exact synthetic-vs-reference question.
  - Run the comparison on the available smoke sample first, then decide whether
    a larger local generation is safe.
  - Report mismatches clearly without claiming FENIKS calibration success from
    closure mechanics alone.
- Implementation:
  - Added `euclid_dsps.synthetic_diffsky.reference_comparison` and CLI command
    `diffsky-compare-dsps-closure-reference`.
  - The diagnostic writes `comparison_summary.json`, `distribution_metrics.csv`,
    `photometry_metrics.csv`, `correlation_metrics.json`, a Markdown report,
    and optional `proposal_weighted_metrics.csv` when a proposals directory is
    supplied.
  - Made `euclid_dsps.synthetic_diffsky` imports lazy so lightweight pandas
    diagnostics do not initialize JAX/GPU-only generation code.
- Local data generated:
  - Ran real Diffsky/FENIKS smoke generation in `conda activate shine` with
    `--max-galaxies 240 --overwrite`, producing 180/30/30 train/validation/test
    rows under `Data/diffsky/synthetic/feniks_260617_dsps_closure/`.
  - Ran closure validation on all 240 rows; validation passed with exact flux
    recomputation (`max_abs_delta_mag=0`, `max_relative_flux_error=0`), normal
    noise residuals (`mean=-1.23e-05`, `std=1.003`), and reported metallicity
    clipping (`clipped_fraction=0.617` on this smoke sample).
- Comparison outputs:
  - `outputs/audits/feniks_synthetic_vs_z035_reference_smoke240/report.md`
  - `outputs/audits/feniks_synthetic_vs_z035_reference_smoke240/comparison_summary.json`
  - `outputs/audits/feniks_synthetic_vs_z035_reference_smoke240/distribution_metrics.csv`
  - `outputs/audits/feniks_synthetic_vs_z035_reference_smoke240/photometry_metrics.csv`
  - `outputs/audits/feniks_synthetic_vs_z035_reference_smoke240/proposal_weighted_metrics.csv`
- Findings:
  - The canonical reference parquet is
    `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet`.
  - Closure mechanics are correct: the generated fluxes are exactly
    reproducible from the 18 truths with the configured euclid_dsps forward
    model.
  - The smoke FENIKS proposal/final population is not distributionally
    compatible with the current z0.35 reference catalog without additional
    selection. In the final smoke sample, only `18.3%` of objects have
    `logsm_true >= 9`, versus `96.6%` in the reference. The weighted proposal
    pool has a similar fraction (`19.0%`), so this mismatch is already in the
    weighted proposal population, not only in the resampling step.
  - Median synthetic closure magnitudes are much fainter than the reference
    Diffsky/HLTDS magnitudes: e.g. `lsst_g` median is `28.56` vs `21.75`,
    and Roman median offsets reach `~8.45` mag on this smoke sample.
  - Metallicities cannot be directly compared to the reference because the
    current z0.35 reference `log10_stellar_metallicity` column has no finite
    values.
- Validation completed:
  - `python -m compileall euclid_dsps` passed.
  - `git diff --check` passed.
  - `pytest -q tests/test_synthetic_diffsky_closure.py tests/test_cli.py`
    passed with `12 passed, 1 skipped`.
  - Full `pytest -q` completed with `376 passed, 9 skipped` and one existing
    config expectation failure:
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`.
    This failure is the same tracked HLTDS truth-column expectation issue and
    is not caused by the new comparison diagnostic.

## 2026-07-02 Port FENIKS Synthetic Closure Pipeline

- Status: code/config/test layer ported into this checkout; full real-Diffsky
  generation not run locally in this phase.
- Goal: port the `lightcone_gen` worktree implementation of the
  Diffsky/FENIKS DSPS closure dataset generator into this newer
  `feature/diffsky-likelihood-sanity-plan` checkout without overwriting
  unrelated local changes.
- Scope:
  - Reuse the ArgonneCPAC Diffsky main-branch integration work where possible.
  - Preserve the current branch's newer MAP/prior-learning/debug features.
  - Avoid local heavy Diffsky generation until the code path is compiled and
    unit-tested; run only small smoke commands when the environment remains
    stable.
- Completed:
  - Imported the structured `euclid_dsps.synthetic_diffsky` package, FENIKS
    generation/closure/prior/amortized configs, synthetic closure docs, and
    focused tests from the `lightcone_gen` worktree.
  - Added CLI commands `diffsky-generate-dsps-closure`,
    `diffsky-validate-dsps-closure`, and
    `diffsky-evaluate-dsps-closure-inference` while preserving the newer MAP
    and inferred-prior commands on this branch.
  - Added `lognormal_mdf_fixed_scatter` metallicity support for
    `sfh_model: diffsky_basic`, with DSPS MDF weights and compatibility for
    dense and compressed SSP grids.
  - Added strict `diffsky_dsps_closure_full` truth handling for forward closure
    and prior learning; this schema requires all 18 truths and does not use the
    historical fixed-metallicity fallback.
  - Added explicit train/validation/test dataset support for supervised prior
    learning and expanded prior diagnostics with correlation and multivariate
    distance metrics.
  - Linked the new synthetic closure documentation from the docs index and run
    setup page.
- Validation completed:
  - `python -m compileall euclid_dsps/synthetic_diffsky euclid_dsps/cli.py
    euclid_dsps/config.py euclid_dsps/model.py
    euclid_dsps/diffsky_forward_closure.py euclid_dsps/prior_learning`
    passed.
  - `python -m compileall euclid_dsps scripts` passed.
  - `git diff --check` passed.
  - CLI help for `diffsky-generate-dsps-closure` and
    `diffsky-validate-dsps-closure` passed.
  - The four new FENIKS configs load successfully with 18 free parameters and
    `stellar_metallicity_model: lognormal_mdf_fixed_scatter`.
  - `pytest -q tests/test_synthetic_diffsky_closure.py
    tests/test_diffsky_trueparam_closure.py tests/test_cli.py` passed with
    `13 passed, 1 skipped`.
  - Full `pytest -q` completed without runtime crash but reported one
    pre-existing/adjacent config expectation failure:
    `tests/test_config.py::test_diffsky_hltds_simple_config_is_recommended_basic_truth_fit`.
    The failing assertion concerns tracked HLTDS config `truth.parameter_columns`,
    not the new synthetic FENIKS files.
- Local execution note:
  - No real Diffsky lightcone generation was launched in this checkout after the
    earlier environment instability. Use a GPU/CUDA JAX environment for the
    real smoke and production generation.

## 2026-07-02 Diffsky Truth/DSPS Parameter Audit

- Status: completed.
- Goal: audit the exact DSPS parameter spaces used by the active Diffsky
  configs against the truth/generated-truth columns available in the current
  parquet files, so a learned-prior-vs-flat-MAP comparison can use an honest
  shared rowset.
- Scope: inspect configured free parameters, projected-truth columns,
  Diffsky-native closure columns, and any missing nuisance parameters without
  changing runtime behavior.
- Completed: generated
  `outputs/audits/diffsky_truth_dsps_parameter_audit/` with raw/projected
  inventories, a cross-space parameter matrix, a compact focus table, a JSON
  summary, and a human-readable audit report.
- Finding: the active 12D PopCosmos MAP space has finite truth/proxy columns
  for 10 dimensions, but `log10_stellar_metallicity` and `tau1_over_tau2` are
  all-NaN projected columns explicitly marked missing by the sidecar metadata.
- Finding: the Diffsky-native 18D closure has generated truth for Diffstar,
  Diffmah, `dust_av`, and `dust_delta`; object-level stellar metallicity is
  still unavailable and remains fixed to `-0.7` in the closure config.
- Finding: existing supervised-prior schemas are 5D (`diffsky_truth_basic`) or
  22D (`diffsky_truth_extended`) and are not directly the same space as the
  active 12D PopCosmos MAP.

## 2026-07-01 Robust Diffsky Population-Prior Learning

- Status: in progress.
- Goal: implement the five-phase debug and prior-learning plan for Diffsky
  population-level priors: unbiased latent geometry, projected-truth closure
  checks, redshift likelihood profiles, MAP sweeps under learned priors, and a
  post-hoc RealNVP prior learned from MAP/MCLMC rather than from a collapsing
  amortized encoder.
- Scope: amortized latent geometry, closure diagnostics, MAP/prior-learning
  workflow glue, Jean Zay SLURM launchers, and documentation/runbook updates.
- Phase 1 completed: decoupled encoder initialization from the latent coordinate
  center/scale, add safer low-z configs, and write latent-prior geometry
  diagnostics before large runs.
- Phase 2 completed: compare projected truth, NN posterior median, and
  flat-prior MAP in the same DSPS likelihood; truth-start MAP remains a
  heavier optional extension.
- Phase 3 completed: write fixed-nuisance redshift likelihood profiles; full
  profiled-redshift optimization remains a heavier follow-up mode.
- Phase 4 completed: run a minimal MAP sweep under the current learned prior on
  the same normal `balanced20k` rowset only.
- Phase 5 completed: train the first robust RealNVP population prior post-hoc
  from flat/weak-prior MAP or MCLMC samples, then use the NN only as a
  distillation/proposal layer.
- Completed: added H100 launchers for the phase 1-4 diagnostic array, MCLMC
  calibration shards, and post-hoc inferred-prior training.
- Completed: added a Sphinx runbook explaining the order of the
  phases, expected artifacts, Jean Zay array commands, and scientific decision
  criteria.
- Validation completed: targeted `ruff check`, `compileall`, `bash -n` on new
  SLURM files, CLI help, CPU latent-geometry smoke, CPU closure smoke without
  MAP on two galaxies, and CPU inferred-prior smoke in the local conda `shine`
  environment.
- 2026-07-01 follow-up: geometry artifacts from job `1182520` show the old
  `fit_initial` latent geometry puts large mass near physical bounds
  (`max_frac_within_either_5pct=0.383735`, driven by `tau2`, `z_obs`, and
  `tau1_over_tau2`), while `zscale005`, `zscale003`, and
  `zscale005_tau2safe` have no parameters above the 5% near-bound threshold.
- Completed: validated `zscale005_tau2safe` as the next low-z reference
  geometry: midpoint latent center, `z_obs` physical scale `0.05`, and
  `tau2` centered at `1.0` with physical scale `0.20`.
- Completed: refactored MAP-Adam to use a compiled `lax.scan` optimizer,
  optimize multiple starts per device chunk, write per-batch parquet shards,
  skip existing shards on resume, and record per-batch throughput.
- Completed: closure optimum diagnostics now write truth/NN residuals and
  redshift profiles before running expensive MAP; full MAP can be skipped by
  default and aggregated after completion.
- Completed: added explicit MAP `prior_density_space` handling (`x` by
  default, optional uncoupled physical `theta` Jacobian mode).
- Completed: added a sharded MAP prior sweep launcher over rowset shards and
  prior weights plus a finalizer command/script to combine per-shard results.
- Completed: updated the robust-prior runbook with the observed geometry table,
  the validated latent-weight change, smoke-vs-full MAP launch policy, and
  resume behavior.

## 2026-06-30 Canonical Projected Truth Dataset

- Status: completed.
- Goal: rely on the global low-z projected-truth parquet for NN/MCLMC
  diagnostics, including historical runs whose normalized config still points
  at the raw low-z parquet.
- Scope: projected-truth dataset builder, amortized inference diagnostics, and
  lightweight validation.
- Planned: teach inference diagnostics to prefer the sibling
  `_projected_truth.parquet` when available.
- Planned: add SFR/sSFR consistency metrics comparing catalog
  `logsfr_true`/`logssfr_true` against projected PopCosmos SFR bins.
- Completed: historical inference diagnostics now prefer the sibling
  projected-truth parquet even when `normalized_config.json` points at the raw
  low-z parquet.
- Completed: projected-truth generation now writes
  `projected_log10_sfr_bin_1..7` and a `.sfr_consistency.csv` sidecar.
- Completed: targeted tests cover projected-truth sibling lookup and SFR
  consistency metrics.

## 2026-06-30 Improve Full-Latent Corner Readability

- Status: completed.
- Goal: make the full latent NN corner visually comparable to the MCLMC corner,
  with readable truth/posterior/prior overlays and truth present for every
  defensible parameter.
- Scope: amortized inference plotting diagnostics only.
- Planned: switch the full-overlay corner to the config free-parameter order.
- Planned: use higher-contrast colors and filled posterior density behind
  stronger truth/prior contours.
- Planned: enrich truth lookup from the selected catalog rows when the
  `inference_truth.parquet` snapshot misses available columns.
- Completed: full overlay now uses config free-parameter order and a
  higher-contrast posterior/truth/prior style.
- Completed: truth lookup combines `inference_truth.parquet` with selected
  catalog rows by `row_index`.
- Completed: added `corner_full_latent_truth_prior_posterior_columns.csv` so
  missing truth dimensions are explicit.

## 2026-06-30 Add Full-Latent Truth/Prior/Posterior Corner

- Status: completed.
- Goal: make amortized inference write a MCLMC-comparable full latent corner
  overlay with posterior, learned prior, and truth/projected-truth on the same
  figure.
- Scope: inference diagnostics only; no training or inference rerun logic
  changes.
- Planned: use posterior samples when available, otherwise posterior medians.
- Planned: use `inference_truth.parquet` plus `truth.parameter_columns` for
  truth overlays, without inventing values for missing truth dimensions.
- Planned: keep the existing median-only and standard posterior/prior plots for
  backward compatibility.
- Completed: added `corner_full_latent_truth_prior_posterior.png` to inference
  diagnostics.
- Completed: the overlay uses posterior samples if combined, then falls back to
  posterior medians; learned prior can be read from either prior-sample parquet
  name used by the runs.
- Completed: smoke-tested the full overlay path in `/tmp` with 12 latent
  dimensions.

## 2026-06-29 Add Diffsky NN Run Matrix

- Status: completed.
- Goal: make the next Diffsky NN experiments launchable from Jean Zay with a
  documented order: build the balanced rowset, then run deterministic no-KL,
  stochastic no-KL, fixed-KL, and annealed-KL jobs on the same rowset.
- Scope: new experiment configs, new SLURM launchers, and a Sphinx runbook.
- Completed: added fixed-KL and annealed-KL RealNVP configs.
- Completed: added `scripts/diffsky_nn_build_rowsets_h100.slurm` for
  reproducible `balanced20k` and worst rowsets.
- Completed: added `scripts/diffsky_nn_experiment_matrix_h100.slurm` as a
  four-task SLURM array over `nokl_det`, `nokl_stoch`, `kl_fixed`, and
  `kl_annealed`.
- Completed: documented the launch order, outputs, diagnostics, and useful
  overrides in `docs/source/diffsky_nn_experiment_matrix.rst`.

## 2026-06-29 Fix Amortized Init And no-KL Diagnostics

- Status: completed.
- Goal: remove hidden physical-parameter constants from amortized encoder
  initialization, add a true deterministic no-KL reconstruction objective,
  add a documented balanced Diffsky rowset path, and expand diagnostics for
  initialization, parameter bounds, SNR, and flux residual interpretation.
- Scope: amortized latent/train/inference diagnostics, reconstruction rowset
  utilities, CLI wiring, focused tests/docs, and lightweight validation only.
- Planned: use `fit.free_parameters.<name>.initial` as the only configured
  physical initialization source, with midpoint fallback.
- Planned: keep learned-prior/KL training on the stochastic Gaussian encoder,
  while adding an explicit deterministic reconstruction mode for pure no-KL
  autoencoder experiments.
- Planned: build one documented `balanced20k` rowset option stratified by
  redshift and observable photometric quality/SNR proxies.
- Completed: moved physical initialization to the config-driven
  `initial_theta_from_config` helper and removed hidden encoder defaults such
  as `z_obs=0.8`.
- Completed: training now writes `initial_theta_diagnostics.json` with
  per-parameter bound distances and boundary warnings.
- Completed: added `amortized.objective.mode=deterministic_reconstruction`,
  inference support for deterministic checkpoints, and a dedicated H100
  deterministic no-KL config.
- Completed: stabilized posterior-predictive chi2 on tiny flux/error scales
  and split likelihood-space `residual_rms` from `flux_residual_rms`.
- Completed: added SNR/error-over-flux/absolute and fractional flux residual
  diagnostics to residual summaries and reconstruction comparisons.
- Completed: `diffsky-build-reconstruction-rowsets` now writes a documented
  `balanced20k` rowset by default, with size/seed CLI controls.
- Validation completed: ruff passed on touched files, compileall passed on
  touched modules/tests, targeted pytest passed
  (`tests/test_amortized_latent.py`, `tests/test_amortized_elbo.py`,
  `tests/test_amortized_infer.py`), config smoke confirmed `z_obs=0.25`, and
  rowset smoke wrote a 100-row balanced set under `/tmp`.

## 2026-06-26 Rename Deliverable And Add Install Cell

- Status: completed.
- Goal: remove supervisor-specific names from the zip-ready deliverable, rename
  the config/package paths cleanly, and add an optional notebook bootstrap cell
  for installing the public `euclid-dsps-shine` branch.
- Scope: final deliverable directory, package README/MANIFEST, and notebook
  path/config references; no heavy JAX/DSPS reruns.
- Completed: final deliverable now lives at
  `outputs/deliverables/diffsky_nokl_lowz_baseline/`; the old
  `diffsky_nokl_lowz_baseline_for_supervisor` directory was removed.
- Completed: renamed the packaged config to
  `configs/diffsky_nokl_trainval20k.yaml` and updated notebook, README, and
  MANIFEST references.
- Completed: added a notebook bootstrap cell that checks for `euclid_dsps` and,
  if missing, clones `https://github.com/CosmoStat/euclid-dsps-shine.git` on
  branch `feature/diffsky-likelihood-sanity-plan` into `_deps/` and installs it
  editable.
- Validation completed: no `supervisor`/`for_supervisor` names remain in the
  final deliverable, setup/install/config cells run from both package root and
  `notebooks/`, the final notebook parses with `ast`, all required/YAML paths
  exist, and `git diff --check` passed.

## 2026-06-26 Finalize Supervisor Zip Package

- Status: completed.
- Goal: make the no-KL Diffsky notebook and all loaded files zip-ready for a
  supervisor: portable package-root discovery, clear notebook explanations,
  complete local assets, and concise run instructions.
- Scope: active notebook, packaged notebook copy, package README/manifest, and
  a final copied deliverable directory; avoid heavy JAX/DSPS reruns.
- Completed: made package discovery portable by searching
  `DIFFSKY_PACKAGE_DIR`, the current directory, current parents, and only then
  the source-checkout fallback.
- Completed: cleared notebook outputs, updated README/MANIFEST, and copied a
  final lean package to
  `outputs/supervisor_package/diffsky_nokl_lowz_baseline_for_supervisor/`
  containing notebooks, configs, data, weights, and assets.
- Validation completed: final notebook has 12 cells, zero stored outputs, and
  parses with `ast`; setup/config cells run from both the package root and
  package `notebooks/`; all required files and YAML-referenced asset paths
  exist; `git diff --check` passed.

## 2026-06-26 Simplify Config Load Cell

- Status: completed.
- Goal: make the notebook config/data load cell easy to read by removing the
  verbose JSON dump and redundant tables, while preserving variables needed by
  later EDA, truth, training, and inference cells.
- Scope: active notebook and packaged notebook copy only.
- Completed: replaced the verbose config cell with a compact `important_config`
  table plus four direct prints for config path, dataset sizes, band list, and
  truth metadata row count.
- Completed: removed the unused `json` import and old `config_summary`,
  `readable_config`, `band_table`, and full `truth_metadata` display from the
  cell.
- Validation completed: both notebooks parse with `ast`, old verbose blocks no
  longer appear in notebook search, and `git diff --check` passed.

## 2026-06-26 Clarify Inference Residual Row Count

- Status: completed.
- Goal: make the notebook explicit that residual summaries are long-form
  object-band tables, so `N_INFER=4` with 14 bands produces 56 rows.
- Scope: active notebook and packaged notebook copy only.
- Completed: the inference cell now prints object count, band count, the
  object-band row formula, and asserts that the residual table length matches
  `n_objects * n_bands`.
- Validation completed: both notebooks parse with `ast`, and
  `git diff --check` passed.

## 2026-06-26 Remove MAP Section From Supervisor Notebook

- Status: completed.
- Goal: simplify the no-KL supervisor notebook by removing all MAP cells and
  leaving only loaded-weight inference followed by global and by-band residual
  plots.
- Scope: active notebook and packaged notebook copy; no JAX/DSPS execution.
- Completed: rebuilt both notebooks to 12 cells: setup, config/data load, EDA,
  truth inspection, neural-network definition, optional training, weight load,
  active-weight inference, global residual plot, and residuals by band.
- Completed: removed all notebook references to worst-object selection, MAP,
  cached reference plot display, and the old after-MAP diagnostics.
- Completed: cleared the generated package `runs/` directory; the optional
  training cell recreates `runs/retrain_nokl` only if it is executed.
- Validation completed: both notebooks parse with `ast`, targeted notebook
  search has no MAP/worst/reference-plot remnants, and `git diff --check`
  passed.

## 2026-06-26 Fix After-MAP Worst-Object Join

- Status: completed.
- Goal: make the notebook after-MAP residual cell read existing
  `map_estimates.parquet` files even when MAP stored the selected catalog
  position in `row_index` instead of the original dataset `row_index`.
- Scope: active notebook and packaged notebook copy only; no MAP rerun.
- Completed: the after-MAP cell now first tries the strict
  `row_index`/`object_id` join, then falls back to matching
  `catalog_position` against `map_estimates.row_index`, and finally to
  `object_id` only for debugging continuity.
- Validation completed: both notebooks parse with `ast`, the existing
  `classic_no_prior/map_estimates.parquet` matches all 8 worst objects via
  `catalog_position`, and `git diff --check` passed.

## 2026-06-26 Make MAP Notebook Cells Reload Safe

- Status: completed.
- Goal: prevent repeated `KeyError: 'z_obs'` in an already-running notebook
  kernel by making MAP closure diagnostics non-fatal and forcing notebook MAP
  cells to reload `euclid_dsps.amortized.map_adam` before calling it.
- Scope: `euclid_dsps/amortized/map_adam.py`, source notebook, packaged
  notebook; lightweight syntax checks only.
- Completed: wrapped `_write_map_closure_metrics` in `run_map_adam_under_prior`
  so closure diagnostics write `map_closure_warning` instead of aborting after
  `map_estimates.parquet` has already been produced.
- Completed: updated both MAP notebook cells to `importlib.reload(map_adam)`
  before direct function calls, so an already-running notebook kernel picks up
  local source fixes without restart.
- Validation completed: `py_compile` passed, both notebooks parse with `ast`,
  and `git diff --check` passed.

## 2026-06-26 Fix MAP Closure z_obs Merge Collision

- Status: completed.
- Goal: fix the `KeyError: 'z_obs'` raised at the end of
  `run_map_adam_under_prior` when the truth snapshot also contains a `z_obs`
  column and pandas suffixes MAP/truth columns during merge.
- Scope: `euclid_dsps/amortized/map_adam.py` plus lightweight syntax checks;
  do not rerun MAP locally.
- Completed: `_write_map_closure_metrics` now merges with explicit
  `("_map", "_truth")` suffixes and uses `z_obs_map` for MAP closure metrics
  when the truth snapshot also has `z_obs`.
- Validation completed: `py_compile` passed, a synthetic no-JAX closure test
  with colliding MAP/truth `z_obs` columns wrote
  `map_closure_photoz_metrics.csv`, and `git diff --check` passed.

## 2026-06-26 Reorder Residual Diagnostics And MAP Plots

- Status: completed.
- Goal: move the reference global/by-band residual diagnostics immediately
  after the neural-network definition, and add separate worst-object residual
  plots before and after MAP.
- Scope: source notebook plus packaged notebook copy; keep MAP execution
  opt-in, use direct Python functions only, and do not execute JAX/DSPS locally.
- Completed: moved `# Reference global residual plot` and
  `# Reference residuals by band` immediately after `# Neural-network
  definition`.
- Completed: split the MAP section into separate cells for worst-object
  Student-t selection, residual heatmaps before MAP, direct classic MAP run,
  residual heatmap/table after classic MAP, and optional learned-prior MAP.
- Completed: the before-MAP plot shows per-band `residual_sigma_median` and
  per-band Student-t NLL for the selected worst objects.
- Completed: the after-MAP plot reads `map_estimates.parquet` when present,
  decodes MAP parameters through DSPS, rebuilds a
  `posterior_predictive_residual_summary_frame`, and plots per-band residuals
  for the same worst objects.
- Validation completed: source and packaged notebook code cells parse with
  `ast`, `git diff --check` passes, and no JAX/DSPS MAP execution was run.

## 2026-06-26 Add Worst-Object MAP Checks To no-KL Notebook

- Status: completed.
- Goal: add two opt-in notebook cells after amortized inference: one selecting
  the worst NN posterior-predictive objects and running classic DSPS/JAX
  MAP-Adam with no learned prior, and one running the same MAP under a learned
  RealNVP prior when a KL-trained checkpoint is supplied.
- Scope: source notebook plus packaged notebook copy; keep both MAP cells off
  by default to avoid WSL instability and use direct Python function calls
  rather than CLI commands.
- Completed: inserted a classic no-prior MAP cell after inference. It ranks
  objects by summed Student-t negative log-likelihood from the reference 20k
  posterior-predictive residual summary, maps original `row_index`/`object_id`
  values back to configured parquet positions, writes
  `runs/notebook_worst_object_map/worst_nn_catalog_positions.txt`, and
  optionally calls `run_map_adam_under_prior` directly with `prior_weight=0.0`.
- Completed: inserted a learned-prior MAP cell that uses the same worst-object
  rowset, but requires a KL/RealNVP checkpoint supplied through
  `DIFFSKY_KL_CHECKPOINT` and optionally `DIFFSKY_KL_FEATURE_STATS`; it remains
  skipped for the packaged no-KL checkpoint because that prior is not learned.
- Completed: the MAP section now uses Student-t NLL as the NN failure score and
  direct Python calls only; CLI command printing was removed.
- Validation completed: source and packaged notebook code cells parse with
  `ast`, `git diff --check` passes, and no JAX/DSPS MAP execution was run.

## 2026-06-26 Make no-KL Notebook Explanatory

- Status: completed.
- Goal: make the no-KL supervisor notebook easier to read for a non-repo
  reader by adding plain comments, a compact config summary immediately after
  load, an explicit explanation of the MLP activation path, and a clear note on
  stellar metallicity being inferred without catalog truth supervision.
- Scope: source notebook plus packaged notebook copy; text/JSON edits only, no
  notebook execution or JAX/DSPS validation because WSL was unstable.
- Completed: added a readable config summary and compact JSON config print
  immediately after `load_config`.
- Completed: rewrote notebook cells with more spacing and English comments
  explaining setup, EDA, truth availability, model definition, optional
  training, weight loading, inference smoke check, and residual plots.
- Completed: clarified that the printed Equinox `GaussianEncoder` lists
  parameter-owning `Linear` modules only; the actual trunk applies `GELU`
  after each hidden linear layer.
- Completed: documented `log10_stellar_metallicity` as a DSPS-required latent
  with no reliable catalog truth in this dataset, learned only through the
  photometric reconstruction likelihood and therefore interpreted as an
  inferred nuisance parameter.
- Validation completed: source and packaged notebook code cells parse with
  `ast`, `git diff --check` passes, and no JAX/DSPS notebook execution was run
  after WSL instability.

## 2026-06-26 Clarify no-KL Notebook Residuals And Training Cell

- Status: completed.
- Goal: remove the confusing `train_like_jean_zay` wrapper from
  `notebooks/diffsky_baseline_nokl_minimal.ipynb`, replace it with a compact
  self-contained no-KL FS2 training loop, and make the notebook explicit about
  which residual plots are the reference 20k posterior-predictive diagnostics
  versus the small notebook smoke check.
- Scope: notebook source plus the packaged notebook copy; no dataset,
  checkpoint, or residual-summary artifact regeneration unless validation shows
  the existing package files are inconsistent.
- Completed: removed the `train_like_jean_zay` wrapper and any direct
  `train_amortized_fs2` import/call from the source and packaged notebooks.
- Completed: added a compact `simple_train_amortized_fs2` notebook cell that
  keeps the no-KL FS2 objective, feature normalization, Student-t likelihood,
  global SED scale, per-band calibration, AdamW updates, and validation best
  checkpoint save, while dropping the heavyweight training entrypoint's logs,
  run fingerprints, diagnostics, and progress machinery.
- Completed: added import comments for `read_feature_stats`,
  `latent_spec_from_config`, `x_to_theta`, `architecture_summary`, and
  `build_amortized_model`.
- Completed: made the residual distinction explicit: the notebook now treats
  `diagnostics/posterior_predictive_normalized_residual_hist.png` and
  `diagnostics/posterior_predictive_residuals_by_band.png` as the reference
  20k posterior-predictive plots, while the 256-row posterior-mean inference
  cell is labeled only as a smoke check.
- Completed: removed the misleading `notebook_small_inference_*.png` files
  from the package diagnostics directory.
- Validation completed: notebook JSON/code cells parse, `git diff --check`
  passes, no direct `train_amortized_fs2` import/call remains, and the packaged
  diagnostics directory contains only the two reference plots plus
  `posterior_predictive_residual_summary.parquet`. Full notebook execution was
  intentionally stopped after WSL instability.

## 2026-06-26 Supervisor Package Low-z Projected-Truth Dataset

- Status: completed.
- Goal: build the deliverable supervisor workflow around a default continuous
  low-z Diffsky dataset that already contains the m5-depth error model and
  projected DSPS truth columns, plus a 20k train/validation subset matching the
  reference no-KL run.
- Scope: generate the augmented dataset outside the notebook; document it and
  make configs default to it; copy dataset, subset, weights, feature stats,
  configs, and notebook into one zip-ready output directory; rewrite the
  notebook so it loads the packaged data, does EDA, exposes the NN/training
  entrypoint used on Jean Zay, loads provided weights, runs small inference,
  and reproduces the accepted residual plots.
- Completed: added `scripts/build_diffsky_lowz_projected_truth_dataset.py`,
  generated
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet`
  with 78651 rows and 106 columns, and generated the exact reference
  train/validation subset
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth_nokl_trainval20k.parquet`
  with 17999 train rows and 2001 validation rows.
- Completed: updated 04/14 configs and H100 script defaults to use the
  projected-truth parquet; updated truth-column metadata for the DSPS latent
  inputs where available and explicit missing columns where unavailable.
- Completed: rewrote `notebooks/diffsky_baseline_nokl_minimal.ipynb` into a
  package-oriented notebook with simple cells for data load, EDA, photometry
  error summaries, DSPS truth distributions, NN architecture, Jean-Zay training
  entrypoint, provided-weight loading, short inference, and the two residual
  plots.
- Completed: assembled the zip-ready package at
  `outputs/supervisor_package/diffsky_nokl_lowz_baseline/` with the full
  projected-truth parquet, 20k subset parquet, configs, train/validation index
  files, `best.eqx`, feature stats, training logs/summaries, residual
  diagnostics, HLTDS SSP/filter assets, README, manifest, and notebook.
- Validation completed: package notebook executed from the package directory in
  `conda shine` with `RUN_TRAINING=False`; source and package configs loaded
  and validated against their parquet schemas; Sphinx docs built successfully;
  `git diff --check` passed.

## 2026-06-26 Fix no-KL Baseline Notebook Consistency

- Status: completed.
- Goal: correct `notebooks/diffsky_baseline_nokl_minimal.ipynb` so it uses the
  active low-z `04_14` no-KL inference dataset/run, removes the misleading
  training/fit framing, reproduces the existing posterior-predictive residual
  plots, and materializes projected-truth diagnostics from the low-z rows.
- Scope: notebook plus the dataset-only `03_31_zmax335_m5depth` YAML cleanup;
  do not change the existing no-KL checkpoint, inference run outputs, or
  unrelated docs.
- Completed: rewrote the notebook as a read-only diagnostic notebook using
  `outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128_infer`
  and
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`.
- Completed: removed the training/save/load checkpoint cells and added cached
  low-z projected-truth outputs:
  `lowz_inference_projected_truth.parquet` and
  `lowz_inference_dataset_with_projected_truth.parquet` under
  `outputs/notebooks/diffsky_baseline_nokl_minimal/`.
- Completed: regenerated notebook copies of the two posterior-predictive
  diagnostics with the same plotting contract and image sizes as the reference
  run: `posterior_predictive_normalized_residual_hist.png` and
  `posterior_predictive_residuals_by_band.png`.
- Completed: made `configs/diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml`
  dataset-only by removing the inherited fit config and the explicit
  redshift/`fit.free_parameters.z_obs` block.
- Validation completed: parsed notebook code cells, executed the notebook in
  `conda shine`, verified the projected-truth and augmented low-z tables have
  20,000 rows, checked both regenerated PNG dimensions against the reference
  run, loaded the cleaned YAML with `load_config`, and ran `git diff --check`.

## 2026-06-26 Minimal Baseline no-KL Notebook

- Status: completed.
- Goal: replace the over-complex supervisor notebook path with a minimal
  notebook that takes only dataset/config/weights as inputs, exposes the
  JAX/Equinox architecture and train loop directly in notebook cells, and keeps
  the same checkpoint load/save contract as the repo.
- Scope: add a separate `notebooks/diffsky_baseline_nokl_minimal.ipynb`, keep
  the old debug playground untouched, and make the notebook clear about the
  difference between inspecting the existing low-z checkpoint outputs and
  training a new baseline on the canonical `zmax335_m5depth` dataset.
- Completed: rewrote `notebooks/diffsky_baseline_nokl_minimal.ipynb` as a
  linear 11-cell notebook: inputs, raw dataset EDA, explicit YAML config cell,
  model architecture plus direct train loop, train cell, save cell, load-best
  cell, NN+DSPS inference cell, residual histogram, and band-error summary.
- Completed: removed `RUN_*` style flags from the notebook. The with/without-KL
  behavior is controlled by the visible `KL_WEIGHT_MAX` config value and by the
  checkpoint selected in the load cell.
- Validation completed: parsed notebook JSON/code, executed the non-training
  path in `conda shine` through checkpoint load, NN+DSPS inference, residual
  histograms, and band-error tables; also ran a micro train smoke on 4 objects
  to confirm the notebook train loop compiles and applies updates.

## 2026-06-26 Supervisor-Ready Diffsky Prior Notebook

- Status: completed.
- Goal: make `notebooks/diffsky_prior_debug_playground.ipynb` self-contained
  for supervisor review: canonical dataset, train/validation rowsets, generated
  notebook configs, explicit no-KL versus KL training loop, and local weight
  registry.
- Scope: update notebook cells only plus this plan; keep heavy training
  opt-in, defaulting to printed commands and lightweight dataset/config
  inspection.
- Completed: rewrote the notebook setup so it starts from the canonical
  `03_31 zmax335 m5_depth` truth-rich dataset, extracts a deterministic
  notebook parquet under `outputs/notebooks/diffsky_prior_debug/datasets/`,
  writes local train/validation/debug rowsets, and materializes standard,
  no-KL, KL, and direct-DSPS configs under the notebook output directory.
- Completed: added a notebook training loop that prints and can optionally run
  no-KL and `kl_weight_max=0.05` training via `train_amortized_fs2`, plus a
  weight manifest listing the two existing local checkpoints and their training
  summaries.
- Validation completed: parsed the notebook JSON and code cells, executed the
  lightweight setup/training-manifest/load/debug cells without launching heavy
  jobs, verified the generated configs point to the notebook parquet, and ran
  `git diff --check`.

## 2026-06-26 Canonical Truth Dataset Documentation

- Status: completed.
- Goal: document the end-to-end projected-truth workflow completely and add the
  high-redshift truth-rich Diffsky subset with materialized errors as the
  canonical "real truth" dataset for truth/prior/projection work.
- Scope: create or document the `z <= 3.35` HLTDS truth subset with
  `m5_depth` `fluxerr_*`, add a dataset config, update Diffsky dataset/data
  download/run setup docs, and validate Sphinx.
- Completed: generated
  `Data/diffsky/processed/hltds_cosmos_260215_03_31_2026_zmax335_m5depth.parquet`
  from the 03/31 source with `--redshift-max 3.35 --error-model m5_depth`.
  The current local source keeps 493903 objects, spans
  `z=1.0552149..3.0319715`, and has 14 `flux_*` plus 14 `fluxerr_*` columns.
- Completed: added `configs/diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml`
  and documented the dataset as the canonical truth-rich reference in
  `docs/source/diffsky_dataset.rst`, `docs/source/data_download.rst`,
  `docs/source/run_setup.rst`, and `configs/README.md`.
- Completed: copied the new redshift, truth, and fractional-error diagnostic
  plots into `docs/source/_static/`.
- Validation completed: parsed the new YAML config, checked generated parquet
  summary/manifest/truth report, ran `git diff --check`, and built docs with
  `conda run -n shine python -m sphinx -b html docs/source docs/_build/html`.

## 2026-06-25 MCLMC Projection Method Documentation Detail

- Status: completed.
- Goal: document the exact generation method for each MCLMC projected-truth
  parameter, including the Diffstar/Diffmah SFH projection equations and the
  dust mapping, not only the final truth distributions.
- Scope: update `docs/source/diffsky_dataset.rst` and validate the Sphinx
  build; no plot regeneration is needed unless the documented artifact paths or
  generated images are stale.
- Completed: added the exact projected-truth generation chain to
  `docs/source/diffsky_dataset.rst`: direct catalog mappings, the
  Diffstar/Diffmah SFH source columns, `age_at_z`, SFH time grid,
  PopCosmos lookback-bin edges, trapezoidal SFH integration, adjacent
  log-SFR-ratio equations, dust mapping, and unavailable truth parameters.
- Validation completed: confirmed the aggregate MCLMC corners were regenerated
  at 2026-06-25 17:47, ran `git diff --check`, and rebuilt the docs with
  `conda run -n shine python -m sphinx -b html docs/source docs/_build/html`.

## 2026-06-25 MCLMC Truth Everywhere And Docs

- Status: completed.
- Goal: make the newly generated catalog/projected truth appear consistently in
  every MCLMC parameter diagnostic and document the truth distributions in the
  Sphinx `.rst` docs.
- Scope: update `scripts/build_diffsky_reconstruction_comparison.py` so the
  1D posterior-median distribution plot, aggregate corners, and individual
  corners all use the same finite projected-truth column selection; add a
  dedicated projected-truth distribution plot; add the generated truth
  distribution figure/explanation to `docs/source/diffsky_dataset.rst`.
- Completed: added `mclmc_projected_truth_distributions.png`, made
  `mclmc_posterior_median_distributions.png` use the same MCLMC projected-truth
  coordinate list as the corners, and regenerated good/average/bad plus legacy
  best/median/worst individual aliases.
- Completed: copied the projected-truth distribution figure into
  `docs/source/_static/` and documented direct truth, projected generated
  truth, and missing truth in `docs/source/diffsky_dataset.rst`.
- Validation completed: verified MCLMC corner columns and truth-distribution
  columns programmatically, visually inspected the new truth-only and
  posterior-median distribution plots, ran
  `python -m compileall scripts/build_diffsky_reconstruction_comparison.py`,
  `git diff --check`, and
  `conda run -n shine python -m sphinx -b html docs/source docs/_build/html`.

## 2026-06-25 MCLMC Projected Truth All-Axes Fix

- Status: completed.
- Goal: ensure every MCLMC corner plot that claims to show
  catalog/projected truth uses all posterior coordinates with finite projected
  truth, not the older reduced core axis list.
- Scope: update the MCLMC corner column selection in
  `scripts/build_diffsky_reconstruction_comparison.py`, regenerate the targeted
  MCLMC plot suite, and verify that `dlog10_sfr_1..6`, `tau2`, and
  `dust_index_n` are all included where truth exists.
- Completed: replaced the old `MCLMC_CORE_CORNER_PARAMETERS` selection for
  pooled, posterior-median, and individual corners with a helper that prefers
  every posterior coordinate with finite projected truth.
- Completed: regenerated the targeted MCLMC plot suite so the main corners and
  the good/average/bad individual aliases now include `z_obs`,
  `log10_stellar_mass`, `dlog10_sfr_1..6`, `tau2`, and `dust_index_n`.
- Validation completed: checked the selected column list programmatically,
  verified the projected-truth table has 100 finite values for each of those 10
  axes, visually inspected pooled/posterior-median/bad corners, ran
  `python -m compileall scripts/build_diffsky_reconstruction_comparison.py` and
  `git diff --check`.

## 2026-06-25 MCLMC Truth Overlay Visibility Fix

- Status: completed.
- Goal: make the catalog/projected truth overlays unmistakable on every MCLMC
  corner plot and keep that behavior in the reusable comparison pipeline.
- Scope: update `scripts/build_diffsky_reconstruction_comparison.py`, regenerate
  the MCLMC dashboard PNGs, and keep the README/plan language aligned with the
  fact that some truth is projected rather than direct catalog truth.
- Completed: aggregate MCLMC corners now show catalog/projected truth with
  orange density contours plus star markers, diagonal rugs, and dashed median
  lines; individual corners keep orange star/line truth markers and force finite
  truth values into the plotted range.
- Completed: regenerated the targeted MCLMC plot suite and README under
  `outputs/comparison/diffsky_reconstruction_debug/plots/mclmc/worst100_b32_w64_s256`.
- Validation completed: visually inspected pooled, posterior-median,
  projected-truth, good, average, and bad MCLMC corners; ran
  `python -m compileall scripts/build_diffsky_reconstruction_comparison.py` and
  `git diff --check`.

## 2026-06-25 MCLMC Projected Ground Truth Implementation

- Status: completed.
- Goal: generate the richest defensible ground-truth/proxy table for the active
  MCLMC PopCosmos parameters and overlay it on the MCLMC corner plots.
- Scope: update `scripts/build_diffsky_reconstruction_comparison.py`, using
  direct catalog truth where available, Diffstar/Diffmah SFH projection for
  `dlog10_sfr_*`, the existing dust mapping for `tau2`/`dust_index_n`, and
  explicit missing/nuisance metadata for unsupported parameters.
- Completed: added `mclmc_projected_truth_parameters.{csv,parquet}` and
  `mclmc_projected_truth_metadata.{csv,parquet}` under the MCLMC dashboard plot
  directory. The table includes direct truth for `z_obs`, stellar mass, SFR,
  and sSFR; projected generated truth for all six `dlog10_sfr_*` ratios from
  Diffstar/Diffmah SFHs; projected dust truth for `tau2` and `dust_index_n`;
  and explicit missing metadata for `log10_stellar_metallicity` and
  `tau1_over_tau2`.
- Completed: MCLMC pooled, median, projected-truth, and representative
  good/average/bad corner plots now overlay the catalog/projected truth where
  the plotted axis has a finite truth value, using explicit orange
  stars/rugs/median lines so the truth is visible in aggregate and individual
  corners.
- Validation completed: regenerated the MCLMC dashboard plot suite in
  `conda shine` so Diffstar/Diffmah projection was available, inspected the
  aggregate and individual corners, verified metadata finite fractions, ran
  `python -m compileall scripts/build_diffsky_reconstruction_comparison.py`,
  and ran `git diff --check`.

## 2026-06-25 Diffsky Ground-Truth Projection Investigation

- Status: completed.
- Goal: identify which active Diffsky/PopCosmos MCLMC parameters can receive
  direct catalog truth, which need deterministic projections from generated
  Diffsky latents, and which only support nuisance or pseudo-truth treatment.
- Completed: verified the active parquet contains `redshift_true`, `logsm_true`,
  `logsfr_true`, `logssfr_true`, `diffstar_*`, `diffmah_*`, `dust_av`,
  `dust_delta`, and `burst_*`, but no object-level stellar-metallicity or
  birth-cloud dust-ratio truth column.
- Completed: confirmed the cleanest SFH enrichment is to project the generated
  Diffstar/Diffmah SFH through `project_sfh_to_popcosmos_dlogsfr_jax`, rather
  than using the existing rough constant-slope `logssfr_true` proxy.
- Completed: confirmed dust can be mapped consistently with the existing
  `diffsky_basic_dust_params_jax` convention: `tau2=dust_av/1.086`,
  `dust_index_n=dust_delta`, and `tau1_over_tau2` remains a fixed/nuisance
  convention unless a birth-cloud latent is added.
- Validation completed: inspected the active parquet schema and relevant model
  adapters, checked that `diffstar`/`diffmah` are available in `conda shine`,
  and computed projected SFH/dust examples for rows `21788`, `78247`, and
  `77681`.

## 2026-06-25 MCLMC Corner Plot Upgrade

- Status: completed.
- Goal: replace the current rough MCLMC corner plots in
  `outputs/comparison/diffsky_reconstruction_debug` with clearer, more
  accurate diagnostics: aggregate posterior contours, truth overlays, and
  individual corners for representative good/typical/bad objects.
- Scope: update `scripts/build_diffsky_reconstruction_comparison.py` and
  regenerate only the comparison dashboard artifacts.
- Completed: added density-contour MCLMC corner plots for pooled posterior
  samples, posterior medians, and truth-comparable parameters, with catalog
  truth overlays limited to parameters that have direct truth columns.
- Completed: added stable individual corner aliases for representative
  `good`, `average`, and `bad` objects. For individual objects, catalog truth
  is drawn as explicit orange markers/lines and is forced into the displayed
  axis range so failures where MCLMC is far from truth remain visible.
- Validation completed: regenerated the MCLMC dashboard plot suite under
  `outputs/comparison/diffsky_reconstruction_debug/plots/mclmc/worst100_b32_w64_s256`,
  visually inspected the aggregate and individual corners, ran
  `python -m compileall scripts/build_diffsky_reconstruction_comparison.py`,
  and ran `git diff --check`.

## 2026-06-25 PhotErr Parameter Explanation

- Status: completed.
- Goal: expand the PhotErr explainer so the figure/PDF explain, in English,
  where each parameter comes from, which parquet columns are used, how `m5` is
  obtained, and what enters the Student-t likelihood.
- Scope: keep the existing compact equation/curve artifacts, but add a more
  verbose annotated artifact for supervision/reporting.
- Completed: added `photerr_error_model_annotated.{png,pdf,svg}` with explicit
  English panels for catalog columns, `m5` provenance, and Student-t inputs.
- Completed: expanded `photerr_error_model_equations.{tex,md,pdf,png}` and
  `photerr_error_model_summary.json` with column names, units, `m5` source,
  `gamma`/`eta` source, and the MAP versus amortized Student-t usage.
- Validation completed: regenerated the report artifacts, visually inspected
  the annotated and equation PNG previews, checked PDF headers/sizes, ran
  `python -m compileall scripts/generate_photerr_error_model_explainer.py`, and
  `git diff --check`.

## 2026-06-25 PhotErr Equation PDF

- Status: completed.
- Goal: produce a PDF version of the color-coded PhotErr equation explainer
  from the generated LaTeX equations.
- Scope: keep the output next to the existing report artifacts under
  `outputs/reports/photerr_error_model_explainer/`; if a system LaTeX engine
  is unavailable, generate an equivalent PDF rendering and keep a standalone
  `.tex` wrapper for later compilation.
- Completed: extended `scripts/generate_photerr_error_model_explainer.py` to
  write `photerr_error_model_equations.pdf` and
  `photerr_error_model_equations_standalone.tex`.
- Note: no local `pdflatex`, `latexmk`, `xelatex`, `lualatex`, `tectonic`, or
  `typst` executable was available, so the PDF was rendered with Matplotlib's
  math renderer while preserving the standalone TeX source for later native
  compilation.
- Validation completed: regenerated the report artifacts, checked the PDF
  header/size, ran
  `python -m compileall scripts/generate_photerr_error_model_explainer.py`, and
  `git diff --check`.

## 2026-06-25 PhotErr Error-Model Figure

- Status: completed.
- Goal: generate a compact, presentation-ready explanation of the active
  Diffsky flux-error model, with a colored plot showing the depth/random,
  PhotErr-style systematic, catalog, likelihood-floor, and total likelihood
  uncertainty terms, plus LaTeX-ready equations.
- Scope: keep this as a reproducible reporting artifact under
  `outputs/reports/photerr_error_model_explainer/` and use the implemented
  `m5_depth` formula instead of a hand-written approximation.
- Completed: added `scripts/generate_photerr_error_model_explainer.py` and
  generated PNG/PDF/SVG figure outputs plus LaTeX, Markdown, and JSON summary
  files under `outputs/reports/photerr_error_model_explainer/`.
- Validation completed: `python scripts/generate_photerr_error_model_explainer.py`,
  `python -m compileall scripts/generate_photerr_error_model_explainer.py`, and
  `git diff --check`.

## 2026-06-25 Diffsky Error-Model Documentation

- Status: completed.
- Goal: make the Diffsky photometric error contract understandable enough to
  interpret the worst100 MAP/MCLMC recovery plots, especially objects with huge
  apparent error bars and misleading normalized-residual gains.
- Scope: update the docs with the exact
  ``m5_depth``/``photo_err``/PhotErr-style formula, how ``fluxerr_*`` feeds
  ``sigma_eff`` in MAP, MCMC, and amortized diagnostics, and numerical examples
  from the generated huge-error-bar diagnostics.
- Completed: rewrote the Diffsky photometry contract documentation with the
  deterministic ``m5_depth`` formula, depth/gamma defaults, PhotErr-style
  ``sigma_sys_mag=0.005`` term, separate 2% likelihood floor, MAP/MCMC versus
  amortized floor-reference difference, and the ``row_index=10355`` numerical
  failure case.
- Completed: regenerated the dashboard report so
  `outputs/comparison/diffsky_reconstruction_debug/tables/worst100/worst100_huge_error_bar_explanation.md`
  now includes the same formula chain and worked example.
- Validation completed: `python scripts/build_diffsky_reconstruction_comparison.py`,
  `python -m compileall euclid_dsps scripts`, and
  `conda run -n shine python -m sphinx -b html docs/source docs/_build/html`.
  The system Python lacks Sphinx, so the docs build was run in the repo's
  `shine` environment.

## 2026-06-25 Diffsky Prior-Debug Notebook

- Status: implemented.
- Goal: create a self-contained exploratory notebook for supervisor-facing
  debugging of the learned-prior/amortized-encoder path against direct DSPS
  MAP and small MCLMC probes.
- Scope: the notebook should load the active Diffsky low-z parquet, inspect
  photometry/truth/error distributions, load existing amortized checkpoints and
  inference outputs when available, expose small local commands for rerunning
  amortized inference, MAP-under-prior, and MCLMC on tiny rowsets, and plot
  prior/posterior/input/output distributions from one place.
- Runtime policy: default cells must be safe to run interactively on CPU by
  reading existing outputs; DSPS decoding, MAP, and MCLMC reruns stay behind
  explicit boolean switches or printed commands.
- Completed: added
  `notebooks/diffsky_prior_debug_playground.ipynb`. It sets all paths and
  runtime toggles up front, loads the active Diffsky low-z parquet, builds a
  tiny debug rowset, plots input photometry/error/SNR distributions, loads
  existing amortized inference/MAP/dashboard outputs, overlays learned-prior,
  posterior, MAP, and truth distributions where available, and includes live
  encoder/prior inspection from a checkpoint.
- Completed: added opt-in notebook cells that print or run tiny commands for
  `amortized-infer-diffsky`, `diffsky-map-adam-prior`, direct DSPS `posterior
  --sampler mclmc`, and small config variants for likelihood/prior ablations.
- Validation completed: `python -m json.tool` on the notebook, compilation of
  all notebook code cells, `python -m compileall euclid_dsps scripts`, a
  minimal load smoke for config/parquet/rowset/photometry arrays, and
  `git diff --check`.

## 2026-06-22 Diffsky Reconstruction Experiment Matrix

- Status: implementation completed for the orchestration/tooling layer; science
  runs are ready to launch on Jean-Zay.
- Goal: make the no-KL Diffsky autoencoder debug loop reproducible and
  extensible across MAP-Adam, MCLMC, input-noise, NF-prior, KL-sweep, and
  prior-trajectory experiments on Jean-Zay H100 jobs capped at 20 hours.
- Reference run: use
  `outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128` and
  `outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128_infer`
  as the baseline. The catalog fingerprint points to
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  with 78,651 rows, schema hash `d6d00d03d52496a55f8613f17c007114`, random
  20k selection, seed 42, and best checkpoint epoch 27.
- Reference row contract: `train_indices.npy` plus `validation_indices.npy`
  exactly matches the inference `inference_indices.npy` set, so future runs
  should reuse these explicit row-index artifacts rather than relying only on
  random seed and limit.
- Current reconstruction status: the no-KL run is stable but not yet a
  likelihood-calibrated reconstruction pass. The all-band median residual is
  about `-0.38 sigma`, `24.3%` of band residuals exceed `|3 sigma|`, and
  `12.9%` exceed `|5 sigma|`; `lsst_u` and `lsst_g` dominate the failures.
- Baseline comparison target: build a direct reconstruction table on the same
  `worst_1000` rowset comparing (1) the existing amortized NN posterior
  predictive from the reference checkpoint, (2) deterministic DSPS
  JAX+Adam MAP fits, and (3) BlackJAX MCLMC posterior predictive summaries.
  The comparison must report identical per-object/per-band residual metrics,
  not only each method's native objective value. Start with `worst_500` if
  MCLMC runtime is too high, then promote to `worst_1000` once timing is
  measured.
- Required implementation phase 1: add reusable experiment rowsets and rowset
  manifests, including `reference_20k`, `reference_train`, `reference_val`, and
  ranked `worst_*` rowsets derived from
  `posterior_predictive_residual_summary.parquet`.
- Required implementation phase 2: add explicit row-index-file support to the
  amortized train, amortized infer, and `diffsky-map-adam-prior` paths, plus
  train/validation split-file support for exact reruns. The existing MAP
  `fit` and posterior `posterior` paths already support text/CSV row-index
  files.
- Required implementation phase 3: add a common reconstruction evaluator that
  normalizes outputs from amortized inference, MAP-Adam fits, and MCLMC
  posterior predictive runs into the same per-object/per-band residual tables
  and summary gates.
- Required implementation phase 4: add input-noise augmentation for the
  amortized encoder, document whether noise perturbs encoder inputs only
  (denoising) or both encoder inputs and likelihood targets, and record noise
  config/seed in run summaries.
- Required implementation phase 5: support NF-prior experiments with a clear
  truth/pseudo-truth separation. The recommended first science-debug track is
  the existing 5D `diffsky_truth_basic` parameterization, because those
  quantities have actual catalog truth columns (`z_obs`, stellar mass, sSFR or
  SFR, dust amplitude, dust slope) and the decoder already has an explicit
  adapter to PopCosmos-bin parameters. Do not invent 12D "truth" columns for
  `dlog10_sfr_1..6`, metallicity, `tau2`, `dust_index_n`, or
  `tau1_over_tau2` unless a parameterization-compatible truth projection is
  formally added and documented. If an apples-to-apples 12D prior is needed
  for the current no-KL architecture, train it as a labeled pseudo-prior from
  MAP or posterior estimates, not as a supervised truth prior.
- Required implementation phase 6: turn the existing KL annealing knobs into a
  reproducible KL sweep over fixed rowsets, and add prior-trajectory diagnostics
  from epoch checkpoints, including an epoch-0 initial-prior checkpoint.
- Required implementation phase 7: add Jean-Zay Slurm array wrappers so the
  experiment matrix can run in parallel with per-experiment output directories,
  logs, normalized configs, catalog fingerprints, rowset manifests, command
  manifests, and compact comparison reports.
- Initial runtime policy: run NN-based experiments on `reference_20k` because
  the 30-epoch reference took about 3.1 hours on H100; run MAP and MCLMC first
  on `worst_500` or `worst_1000`, then expand only if the timing and
  reconstruction gates justify it. MCLMC should remain a small selected-row
  diagnostic unless a vectorized/chunked posterior path is added.
- Implemented:
  - `diffsky-build-reconstruction-rowsets` writes `reference_20k`,
    `reference_train`, `reference_validation`, ranked `worst_*` rowsets, and a
    rowset manifest. The local reference rowsets were generated under
    `outputs/rowsets/diffsky_autoencoder_nokl_m5sys_z035_rand20k`.
  - Amortized train/infer and `diffsky-map-adam-prior` accept explicit
    row-index files. Amortized train also accepts exact train/validation index
    files for apples-to-apples reruns.
  - `diffsky-compare-reconstruction` normalizes amortized NN, standalone MAP
    `fit`, and MCLMC posterior predictive residual outputs into common
    per-band, per-object, and per-method summaries.
  - MCLMC batch posterior runs now write
    `batch_posterior_predictive_flux_residual_summary.{csv,parquet}`.
  - Amortized training supports encoder-input-only Gaussian noise via
    `amortized.input_noise` or `--input-noise-sigma-scale`; likelihood targets
    remain unchanged.
  - The supervised Diffsky NF-prior trainer accepts `--row-indices-file`, and
    amortized training accepts `--prior-checkpoint` so a freshly trained 5D
    `diffsky_truth_basic` prior can be used without editing YAML.
  - Training now saves `checkpoints/epoch_0000.eqx` to anchor prior/encoder
    trajectory diagnostics.
  - `scripts/diffsky_reconstruction_baselines_h100.slurm` covers rowset
    generation, NN inference, standalone MAP, MCLMC, comparison, input-noise
    training, KL-sweep training, supervised NF-prior training, and training
    with the supervised prior loaded.
  - Documentation was added to `docs/source/amortized_inference.rst` and
    `docs/source/run_setup.rst`.
- Validation completed:
  - `python -m compileall euclid_dsps scripts`
  - `bash -n scripts/diffsky_reconstruction_baselines_h100.slurm`
  - `pytest tests/test_amortized_catalog_identity.py tests/test_reconstruction_experiments.py tests/test_cli.py -q`
  - `pytest tests/test_mcmc.py -q`
  - Built real reference rowsets and ran an NN-only comparison smoke on
    `worst_500`.
- 2026-06-23 Jean-Zay first-submit audit:
  - Jobs `771654`-`771663` failed before science work because they started in
    parallel with `TASK=rowsets` and checked for rowset files before job
    `771653` finished writing them. `771653` completed successfully and wrote
    the rowset manifest.
  - Fixed `scripts/diffsky_reconstruction_baselines_h100.slurm` so dependent
    tasks call `ensure_rowsets` and build rowsets under a filesystem lock if
    needed, instead of failing immediately on missing rowset files.
  - Added explicit output checks for `TASK=compare`, and updated the Jean-Zay
    documentation to launch long dependencies with Slurm `afterok`.
  - Follow-up live audit showed `MAP` was using the default `fit --limit 25`
    and `MCLMC` was using the default `posterior --limit 5` despite the
    `worst_500` row-index file. Fixed the wrapper to pass `--all` for MAP and
    MCLMC rowset jobs, and added `posterior --all` support in the CLI.
- 2026-06-23 Jean-Zay runtime audit:
  - Synced logs from jobs `804614`-`805144` show the NN inference completed,
    the amortized training variants are doing real work, and the supervised
    prior path produced outputs.
  - The standalone MAP relaunch `805143` is still unusably slow because sparse
    `worst_500` row indices are yielded one selected row per parquet batch:
    each iteration then pads `n_rows=1` to `128` and spends about 8 minutes in
    one JAX optimization. The fix is to coalesce row-index-filtered parquet
    chunks until a full selected batch is available.
  - MCLMC jobs `804618` and `805144` failed because BlackJAX is missing in the
    Jean-Zay `shine` environment. Add an early wrapper preflight and relaunch
    MCLMC first on `worst_50`/`worst_100`, not `worst_500`.
  - Implemented rowset coalescing in `iter_catalog_batches`: a sparse
    `worst_500` rowset with `batch_size=256` now yields selected batch sizes
    `[256, 244]` locally instead of 500 one-row batches. This should turn the
    MAP job from about 68 hours to a small number of JAX optimization chunks.
  - Updated the H100 wrapper default `MAP_BATCH_SIZE` to `256`, added
    `worst_50` and `worst_100` rowsets for MCLMC probes, and added an early
    BlackJAX preflight for `TASK=mclmc`.
  - Validation completed: `pytest tests/test_io.py -q`,
    `python -m compileall euclid_dsps scripts`, `bash -n
    scripts/diffsky_reconstruction_baselines_h100.slurm`, and a real local
    `worst_500` batching check.
- 2026-06-23 MCLMC batched-galaxy follow-up:
  - Jean-Zay job `809612` confirms the MAP fix: `worst_500` completed in
    about 17 minutes with two compact selected batches (`256` and `244` rows).
  - Jean-Zay job `809613` shows the remaining bottleneck: MCLMC still runs one
    galaxy at a time in `sample_batch`, reaching only `2/100` objects after
    about 38 minutes. Implement a true batched-galaxy MCLMC path for
    `sample.sampler=mclmc` and `--batch-size > 1`.
  - The first implementation should preserve the existing output tables and
    compare compatibility, use vmap/JAX batching within each MCLMC transition,
    keep the sequential path available for `--batch-size 1`, and leave room to
    aggregate multiple H100 runs over disjoint rowsets.
  - Implemented `sample_galaxy_batch_mclmc`: for `sample.sampler=mclmc` and
    `--batch-size > 1`, the workflow now builds a joint factorized MCLMC state
    over a compact galaxy batch, evaluates per-galaxy log densities with JAX
    batching, and emits the same per-galaxy posterior output tables as the
    sequential path. The joint state is flattened before entering BlackJAX for
    better API compatibility.
  - `sample_batch` now runs MAP initialization with the existing vmap Adam
    batch path before batched MCLMC, avoiding one MAP optimization per galaxy.
    `scripts/diffsky_reconstruction_baselines_h100.slurm` defaults
    `MCLMC_BATCH_SIZE=8`, still overrideable to `1` for the old sequential
    behavior.
  - Added `scripts/merge_mclmc_runs.py` so disjoint MCLMC runs from several
    H100 jobs can be concatenated into a compare-compatible output directory.
  - Local validation completed without BlackJAX installed:
    `pytest tests/test_mcmc.py tests/test_io.py -q`,
    `python -m compileall euclid_dsps scripts`, `bash -n
    scripts/diffsky_reconstruction_baselines_h100.slurm`, and
    `python scripts/merge_mclmc_runs.py --help`. A real BlackJAX smoke must be
    run on Jean-Zay.
- 2026-06-24 comparison-dashboard phase:
  - Build `outputs/comparison/diffsky_reconstruction_debug` as the human-facing
    comparison directory. The canonical reference is the full 20k no-KL
    inference run
    `outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128_infer`;
    the `worst_500`/`worst_100` rowsets are diagnostic slices derived from this
    reference, not independent references.
  - Keep original run directories intact and expose them through symlinks under
    the comparison directory. Generate normalized residual tables, MAP/MCLMC
    residual plots matching the NN diagnostics, corner-style parameter plots,
    NN training/inference galleries, and a README/index documenting exactly
    which source run each plot uses.
  - Implemented `scripts/build_diffsky_reconstruction_comparison.py`, which
    builds the dashboard idempotently from the local canonical run plus the
    synced Jean-Zay outputs. It writes `manifest.json`, `README.md`,
    `index.html`, normalized `reference_full`, `worst500`, and `worst100`
    residual tables, symlinks to all compared runs, MAP/MCLMC residual and
    corner plots, and NN training/inference galleries.
  - Generated the dashboard locally at
    `outputs/comparison/diffsky_reconstruction_debug`. The full-reference
    summary is median `|residual|=1.338 sigma`; on the diagnostic `worst_100`,
    the canonical NN has median `|residual|=9.434 sigma` while MAP reaches
    `1.726 sigma` and MCLMC reaches `1.349 sigma`, supporting the conclusion
    that DSPS can recover much of the photometry on NN failure cases.
  - Corrected the diagnostic plots after inspection: all residual histograms
    and boxplots now use `(flux_in - flux_out) / sigma_eff` with `-3`/`+3`
    guides, MAP observed-vs-modeled plots have explicit flux axis labels,
    MAP/MCLMC parameter corners and distributions overlay catalog truth where
    available plus flat-prior bounds, NN inference PNGs are copied into the
    dashboard so browser links render, and MAP/MCLMC include best/median/worst
    photometric SED-point plots. The README/index now explain input-noise
    training, full-reference versus worst-slice semantics, and pooled MCLMC
    samples versus posterior medians.
  - Refocused the comparison dashboard on the photometric recoverability
    question: for the exact same worst-slice row indices, generate paired
    baseline-vs-method tables and plots at object level and `(object, band)`
    level. Positive paired-improvement values mean the tested method has lower
    absolute photometric error than the canonical NN baseline. The `worst_100`
    comparison now also includes the larger `map_1000_iter400` run filtered to
    those same 100 objects.
- 2026-06-25 visual diagnostic follow-up:
  - Goal: make the worst-slice recoverability argument more explicit by showing
    where the selected `worst_100` objects sit inside the full 20k NN baseline
    error distribution, where those same objects land after DSPS MAP fitting,
    and several per-object photometric SED examples for baseline NN, MAP, and
    MCLMC rather than only best/median/worst examples.
  - Implemented in `scripts/build_diffsky_reconstruction_comparison.py`.
    Regenerated `outputs/comparison/diffsky_reconstruction_debug` with
    `plots/worst100_dsps_recovery/worst100_location_in_full_nn_and_map.png`,
    which overlays the selected `worst_100` and the same objects after MAP on
    the full 20k NN baseline object/band error distributions.
  - Added
    `plots/worst100_dsps_recovery/sed_examples_baseline_map_mclmc_grid.png`,
    a multi-example SED grid with NN baseline, DSPS MAP, and DSPS MCLMC columns.
    Rows are chosen from the same `worst_100` by MAP recovery outcome: large
    MAP gains, typical MAP gains, and cases still hard after MAP. The catalog
    flux is plotted as the reconstruction target/truth because no separate
    noiseless truth-flux column exists in the processed Diffsky parquet.
  - Validation completed: `python scripts/build_diffsky_reconstruction_comparison.py`
    and `python -m compileall euclid_dsps scripts`.
- 2026-06-25 error-bar clarification follow-up:
  - Goal: simplify
    `plots/worst100_dsps_recovery/worst100_location_in_full_nn_and_map.png`
    because the overlaid histogram view is hard to parse, and investigate the
    apparent "huge MAP gain" SED examples whose catalog error bars are so large
    that the object is not scientifically recovered despite a small normalized
    residual.
  - Implemented a simpler location plot: the left panel is now a strip plot of
    full-reference NN objects, the selected `worst_100` in NN space, and the
    same objects after MAP; the right panel is a direct same-object NN-vs-MAP
    scatter with huge `obs_err/abs(obs_flux)` cases circled and labeled.
  - Added `plots/worst100_dsps_recovery/huge_error_bar_diagnostics.png`,
    `tables/worst100/worst100_huge_error_bar_band_diagnostics.{csv,parquet}`,
    `tables/worst100/worst100_huge_error_bar_object_diagnostics.{csv,parquet}`,
    and `tables/worst100/worst100_huge_error_bar_explanation.md`.
  - Main finding: the worst apparent gains are near-zero-flux catalog rows with
    finite synthetic depth errors from the `m5_depth` `fluxerr_*` model. Example:
    row `10355`, `lsst_u`, has `F_obs=2.106595e-41`, `fluxerr=2.535871e-31`,
    `fluxerr/abs(F_obs)=1.203777e10`; MAP has normalized residual
    `0.000146` but `abs(F_obs-F_map)/abs(F_obs)=1.762682e6`.
  - Updated the SED grid labels so the first group is explicitly
    `huge-error gain`, followed by `credible MAP gain` and `still hard after
    MAP`.
  - Validation completed: `python scripts/build_diffsky_reconstruction_comparison.py`
    and `python -m compileall euclid_dsps scripts`.

## 2026-06-22 PhotErr Error-Model Slides

- Status: implemented.
- Goal: create a Reveal.js slide deck explaining the PhotErr error model,
  the flux-space formula implemented in this repository, the regenerated error
  diagnostics, and how `fluxerr_*` plugs into the likelihood.
- Scope: use local diagnostic PNGs and MathJax equations; do not add a
  JavaScript build pipeline or package dependency.
- Completed: wrote `outputs/reports/photerr_error_model_slides.html` with
  32 Reveal.js slides covering the PhotErr formula, the flux-space DSPS
  implementation, regenerated error plots, and the effective likelihood sigma.
- Validation completed: checked local image references in the HTML and ran
  `git diff --check` on the slide deck and `PLAN.md`.

## 2026-06-21 No-KL z<0.35 Rerun With Updated Error Model

- Status: implemented.
- Goal: rerun the pure autoencoder/DSPS reconstruction sanity check on the
  active continuous low-z Diffsky subset with the updated `m5_depth` +
  PhotErr-style systematic flux-error model.
- Dataset status: the local active subset
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  has been regenerated from the `m5_depth` source parquet with
  `sigma_sys_mag: 0.005`; it contains 78,651 objects with
  `redshift_true <= 0.35` and no high-redshift 0.4 feature.
- Slurm update: `scripts/diffsky_autoencoder_nokl_h100.slurm` now defaults to
  the full active subset (`LIMIT=`), 20 epochs, no KL
  (`--kl-weight-max 0.0`), and a 20-hour wall time. A smaller smoke run can
  still be requested explicitly with `LIMIT=10000`.
- Runtime guard: the script keeps the memory-safe posterior predictive
  inference knobs (`DECODER_SAMPLE_CHUNK_SIZE` and
  `PRIOR_PREDICTIVE_BATCH_SIZE`) so the post-training diagnostics can be run on
  the full subset without repeating the previous H100 OOM pattern.
- Follow-up adjustment: full-subset 30-epoch training is too slow for the
  current diagnostic loop. The Slurm wrapper now exposes `SIGMA_SYS_MAG` and
  `SELECTION_MODE` explicitly. The recommended short rerun is `LIMIT=20000`,
  `SELECTION_MODE=random`, `FORCE_REBUILD_DATASET=1`, `ERROR_MODEL=m5_depth`,
  and `SIGMA_SYS_MAG=0.005` so the parquet is rebuilt with the intended
  PhotErr-inspired systematic floor before training.
- Runtime fix: the H100 wrappers no longer require `$WORK` to be exported.
  They infer `REPO_DIR` from `SLURM_SUBMIT_DIR` when submitted from the repo,
  and infer `MINICONDA_PATH` from the parent directory of `REPO_DIR` when
  needed.
- Analysis completed for Jean-Zay job 730441
  `diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128`: the run used
  `m5_depth`, `sigma_sys_mag=0.005`, `FORCE_REBUILD_DATASET=1`, random
  20k-row selection, and `kl_weight_max=0.0`. Training is stable and improves
  through the run, with the best validation checkpoint at epoch 27.
- Reconstruction verdict: the no-KL encoder+DSPS path recovers the central
  photometry qualitatively better than the previous run, but it does not yet
  pass a likelihood-calibrated flux-reconstruction gate. On the 20k selected
  objects, `residual_sigma_median=(flux_in-flux_out)/sigma_eff` has global
  median `-0.38`, robust width `~1.90 sigma`, standard deviation `4.70`,
  `24.3%` of band residuals outside `|3 sigma|`, and `12.9%` outside
  `|5 sigma|`. Validation residuals are essentially the same as train
  residuals, so this is not simply a train/validation overfit artifact.
- Dominant residual failures remain band-dependent: `lsst_u` and `lsst_g`
  have median absolute residuals of roughly `6.8 sigma` and `5.4 sigma`, while
  `roman_F129`, `roman_F146`, and `roman_F158` are close to acceptable.
- Follow-up diagnostics update: future amortized inference outputs now add
  exact DSPS-derived `log10_sfr_at_obs` and `log10_ssfr_at_obs` quantities to
  posterior summaries and diagnostic prior samples. Extended truth comparisons
  and the truth/posterior/prior population corner include SFR and sSFR when
  `logsfr_true` and `logssfr_true` are available.

## 2026-06-21 PhotErr-Inspired Systematic Floor

- Status: implemented.
- Goal: align the synthetic `m5_depth` flux-error model with the core PhotErr
  point-source formulation without adding a runtime dependency on `photerr`.
- Planned model: keep the existing Rubin/PhotErr random term
  `sigma_rand^2 = (0.04 - gamma) * abs(flux) * f5 + gamma * f5^2`, then add
  the PhotErr-style irreducible systematic floor in quadrature,
  `sigma_sys^2 = (sys_frac * abs(flux))^2`, with
  `sys_frac = 10 ** (sigma_sys_mag / 2.5) - 1` and default
  `sigma_sys_mag = 0.005`.
- Scope: implement the formula in `photometric_uncertainty.py`, expose the
  default in manifests/config payloads, update tests and docs, and run targeted
  validation. Do not add `photerr` as a package dependency.
- Completed: added `DEFAULT_PHOTERR_SIGMA_SYS_MAG = 0.005` and the quadrature
  systematic term to the `m5_depth` model, exposed `--sigma-sys-mag` on
  Diffsky dataset/subset generation, and recorded `sigma_sys_mag` in manifests.
- Completed: regenerated
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet`
  and
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`.
  The current manifests now declare `sigma_sys_mag: 0.005`.
- Completed: regenerated subset flux-error diagnostics and analytical
  error-vs-flux curves, including the Sphinx static PNG copies.
- Validation completed: `pytest tests/test_photometry.py
  tests/test_diffsky_prepare_dataset.py -q`, `python -m compileall euclid_dsps
  scripts`, `diffsky-validate-dataset`, `pytest tests/test_config.py
  tests/test_cli.py tests/test_diffsky_validation.py -q`, strict Sphinx build,
  and `git diff --check`.

## 2026-06-19 Full-Dataset No-KL m5-Depth Rerun Setup

- Status: implemented.
- Goal: rerun the pure autoencoder/DSPS sanity check on the full active
  Diffsky dataset after regenerating or copying the new `m5_depth` `fluxerr_*`
  model.
- Added `FORCE_REBUILD_DATASET=1` support to the H100 autoencoder, inference,
  and full-validation Slurm scripts. When set, the scripts rebuild the active
  subset parquet from `SOURCE_DATASET` even if `DATASET` already exists; when
  unset, an existing copied parquet is used as-is.
- Fixed full-dataset overrides for the autoencoder and inference scripts:
  `LIMIT=` and `INFER_LIMIT=` now mean no row limit instead of falling back to
  the old 10k/5k defaults.
- Current default generated dataset contract in the H100 scripts:
  `DATASET=Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  rebuilt from
  `SOURCE_DATASET=Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet`
  with `ERROR_MODEL=m5_depth`, `REDSHIFT_MIN=0.0`, and
  `REDSHIFT_MAX=0.35`.

## 2026-06-19 No-KL z<0.35 Run Analysis

- Status: completed.
- Scope: analyze the completed Jean-Zay run
  `outputs/runs/diffsky_autoencoder_nokl_z035_retry1` and its inference output
  `outputs/runs/diffsky_autoencoder_nokl_z035_retry1_infer`.
- Checks to perform: confirm KL was disabled in the objective, inspect
  train/validation likelihood convergence, summarize posterior-predictive
  normalized residual distributions globally and per band, inspect redshift and
  truth diagnostics, and identify whether the no-KL autoencoder passes the
  flux-reconstruction sanity gate.
- Confirmed the run is genuinely no-KL in the objective: all logged rows have
  `kl_weight=0.0`, the phase is `joint_no_prior`, the prior source is
  `standard_normal`, `train_prior=false`, and `prior_grad_norm=0.0`. The logged
  `kl_mc_mean` is only a diagnostic and is not multiplied into the loss.
- Training/validation NLL improves through epoch 12, and the best checkpoint is
  epoch 12. The run is numerically stable, but the inference closure gate fails.
- Flux-reconstruction sanity verdict: fail under the current likelihood errors.
  The global normalized residual distribution has mean `1.47`, standard
  deviation `7.58`, median `-0.10`, `37.8%` outside `|3 sigma|`, and `24.5%`
  outside `|5 sigma|`.
- Dominant failing bands: `lsst_u`, `lsst_g`, `lsst_r`, `roman_F062`,
  `lsst_i`, and `roman_F213`. Median absolute residuals are especially large
  for `lsst_u` (`13.4 sigma`), `lsst_g` (`8.1 sigma`), `lsst_r`
  (`4.1 sigma`), and `roman_F062` (`4.1 sigma`).
- Inferred redshift collapses to the upper bound: median `z_obs_median` is
  `0.34994`, `97.4%` of objects have `z_obs_median > 0.345`, and `90.9%` have
  `z_obs_median > 0.349`. The median 68% posterior width in redshift is
  `7.3e-5`, so the posterior is extremely overconfident.
- Closure metrics fail on redshift: photo-z median bias `0.0804`, `z_obs`
  median bias `0.1004`, and `coverage_68=0.0`. Stellar mass median bias is
  moderate (`0.182 dex`), but this is not enough to accept the run.
- The scalar `posterior_predictive_chi2_median` is `inf` for all 10k inference
  objects while the per-band residual summary is finite. Treat that scalar as a
  diagnostics bug/gap for this run and use
  `posterior_predictive_residual_summary.parquet` plus
  `posterior_predictive_normalized_residual_tails.csv` instead.
- Recommended next gate: do not proceed to full KL/NF science training from
  this checkpoint. First isolate whether the issue is the DSPS/data/error model
  or the amortized encoder by running a redshift-fixed/truth-redshift no-KL
  closure and a small likelihood-only MAP/z-grid check on the same z<0.35
  subset.

## 2026-06-19 Synthetic Flux Error Implementation Plan

- Status: implemented.
- Goal: replace the main Diffsky `fractional_snr` synthetic errors with a
  simple band-depth error model that depends on flux, remains deterministic, and
  is usable for all LSST+Roman bands.
- Model: use `sigma_cat_b^2 = 0.04 * ((1 - eta_b) * abs(flux_b) * f5_b +
  eta_b * f5_b^2)`, where `f5_b = fnu(m5_b)` and `eta_b` is the background
  fraction of the 5-sigma variance. This is equivalent to the Rubin/LSST
  `m5,gamma` form with `gamma_b = 0.04 * eta_b`.
- LSST defaults: use the Rubin/LSST `m5,gamma` model for LSST bands. `m5_b`
  must be chosen from the intended scenario, e.g. single-visit, 10-year coadd,
  or HLTDS-like synthetic depth. `gamma_b` can be configured per band, with
  Science Book/rubin_sim values as defaults.
- Roman defaults: do not reuse LSST `gamma` as an official Roman model. Use
  Roman WFI 5-sigma AB sensitivities as `m5_b`; set `eta_b=1.0` for a pure
  depth floor or `eta_b~0.95` for a simple source-plus-background approximation
  until an ETC/Pandeia-derived SNR table is available.
- Integration: add a `depth_flux` or `m5_depth` error-model type in
  `photometric_uncertainty.py`, expose per-band `m5` and `eta/gamma` through
  `diffsky-redshift-subset`, update `configs/diffsky_dataset_hltds_04_14.yaml`
  and H100 Slurm defaults, regenerate the `continuous_lowz_fluxerr` parquet,
  and update docs/manifests to distinguish catalog/statistical `fluxerr_*` from
  the likelihood model floor.
- Likelihood cleanup: keep `fluxerr_*` as catalog noise and use a separate
  `flux_error_floor_frac` only for model/calibration tolerance. Apply the same
  floor reference convention in MAP/MCMC and amortized likelihoods before
  comparing likelihood values across methods.
- Completed: added the `m5_depth` model to `photometric_uncertainty.py` and
  threaded `band_name` through dataset generation and all observation-loading
  fallbacks (`io.py`, `observation_arrays.py`, and workflow batch arrays).
- Completed: changed Diffsky dataset generation defaults and H100 preflight
  scripts from `fractional_snr` to `m5_depth`; configs now point to
  `hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet` as the
  full source artifact.
- Completed: regenerated the full source parquet
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet`
  and the active low-z subset
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  with `m5_depth` `fluxerr_*` columns.
- Completed: generated error diagnostics under
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr/`:
  `flux_error_summary.csv`, `flux_fractional_error_by_band.png`,
  `flux_snr_by_band.png`, and `flux_vs_fluxerr_by_band.png`.
- Completed: generated explanatory per-band error-vs-flux diagnostics:
  `flux_error_model_curves_by_band.png`,
  `flux_fractional_error_model_curves_by_band.png`, individual
  `flux_error_model_curves/flux_error_model_curve_<band>.png` files, and
  `flux_error_model_curve_summary.csv`.
- Completed: updated Roman synthetic noise from a pure depth approximation
  (`eta=1.0`, `gamma=0.04`) to `eta=0.95` (`gamma=0.038`) so Roman bands retain
  a non-zero source/Poisson-like term while remaining depth dominated.
  Regenerated the full source parquet, active low-z subset, manifests, standard
  error diagnostics, and model-curve plots. Rebuilt `docs/_build/html` with the
  explanation of absolute versus relative error.
- Validation completed: targeted tests
  `pytest tests/test_photometry.py tests/test_diffsky_prepare_dataset.py
  tests/test_diffsky_validation.py -q`, config/CLI tests
  `pytest tests/test_config.py tests/test_cli.py -q`, `python -m compileall
  euclid_dsps scripts`, `bash -n` on the H100 Slurm scripts,
  `diffsky-validate-dataset` on the regenerated low-z subset, strict Sphinx
  build, and `git diff --check`.

## 2026-06-19 Web Review of Synthetic Photometric Errors

- Status: completed.
- Scope: check external survey/photometry references for whether the current
  `fluxerr_* = abs(flux) / 50` model is scientifically appropriate.
- Finding: the formula is internally coherent only as a constant-SNR tolerance
  model. It follows directly from `SNR = flux / sigma`, but it assumes every
  band and every object is measured at the same fractional precision.
- Finding: it is not a realistic Rubin/Roman/HLS-style photometric error model.
  Survey references compute SNR from source counts plus sky/background,
  read/dark/instrumental noise, aperture/PSF footprint, exposure time, object
  size, and band-dependent limiting depth. Roman WFI documentation also gives
  band/source-size/integration-dependent 5-sigma AB limits.
- Recommended fix: replace the main science synthetic-errors path with a
  band-dependent depth model, e.g. `sigma_depth_b = fnu(m5_b) / 5`, then combine
  it in quadrature with an explicit calibration/model floor. Keep
  `abs(flux)/50` only for closure/smoke tests where the desired assumption is a
  fixed fractional tolerance.
- Recommended fix: avoid double-counting the same fractional uncertainty in
  both materialized `fluxerr_*` and `fit.flux_error_floor_frac`; decide whether
  `fluxerr_*` means catalog/statistical noise and `flux_error_floor_frac` means
  model/calibration tolerance, then apply the same convention in MAP/MCMC and
  amortized likelihoods.

## 2026-06-19 Documentation Contract Cleanup

- Status: completed.
- Scope: update repository guidance and public docs after the documentation,
  dataset, and flux-error audit. Keep the public path on the
  `continuous_lowz_fluxerr` Diffsky subset, document the deterministic
  `fluxerr_*` generation, and remove stale references to removed PopCosmos and
  FS2-only amortized paths.
- Completed: refreshed `AGENTS.md`, the top-level README, `configs/README.md`,
  and Sphinx pages for architecture, dataset, forward-model, and run setup.
  Public commands now build the no-error 04/14 source parquet, materialize the
  `z < 0.35` continuous subset with `fractional_snr` SNR 50 errors, and use the
  `continuous_lowz_fluxerr` dataset for Diffsky MAP, closure, supervised-prior,
  amortized, and diagnostics examples.
- Completed: documented that `flux_*` columns are AB-magnitude-derived
  `fnu_cgs` fluxes and that current `fluxerr_*` columns are deterministic
  likelihood inputs, `max(abs(flux) / 50, 1e-40)`, not native survey errors or
  per-fit random draws.
- Follow-up: code still uses observed-flux fractional floors for MAP/MCMC and
  model-flux fractional floors for amortized likelihood/residual diagnostics.
  The docs now state this explicitly; unify the implementation before using
  absolute MAP-vs-amortized likelihood values as a quantitative comparison.
- Validation completed: `git diff --check`, `python -m compileall euclid_dsps
  scripts`, and `uv run python -m sphinx -W --keep-going -b html docs/source
  outputs/doc_audit_html`.

## 2026-06-19 Documentation, Dataset, and Flux-Error Audit

- Status: completed.
- Scope: compare README/Sphinx/config documentation against the current
  checkout, local processed datasets, public configs, and implemented
  flux-error/likelihood paths.
- Questions to answer: whether dataset choices and artifacts are documented
  coherently, whether stale config names remain in the docs, and whether the
  per-band flux-error model is random or an explicit deterministic likelihood
  assumption.
- Findings: the main Diffsky dataset contract is coherent in code/configs and
  current Sphinx pages: active public configs point to
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  which exists locally with `78651` objects, `z=0.006876--0.334662`, 14
  `flux_*` bands, and 14 `fluxerr_*` bands documented as synthetic
  `fractional_snr` SNR 50 errors in both manifest and schema.
- Findings: the top-level README is partially stale because several example
  Diffsky commands still pass the source
  `*_photometry_truth_noerr.parquet` to closure/prior/overlap/redshift
  diagnostics, while `docs/source/run_setup.rst` and active configs use the
  `continuous_lowz_fluxerr` subset.
- Findings: `configs/README.md` and `docs/source/diffsky_dataset.rst` still
  describe `amortized_diffsky_hltds_04_14_realnvp_gpu.yaml` as the main
  Diffsky amortized config, but the public command docs promote
  `amortized_diffsky_hltds_joint_realnvp_gpu.yaml`, which extends the base and
  explicitly sets `prior.source: joint_realnvp`.
- Findings: `docs/source/architecture.rst` is stale for amortized inference:
  it still says `amortized/` is FS2-only and lists only the old narrow config
  set, while the current code/configs support Diffsky HLTDS amortized training,
  inference, prior overlap, sharded finalization, and H100 validation.
- Findings: the synthetic flux-error model is deterministic after dataset
  materialization, not random during fitting:
  `fluxerr_* = max(abs(flux) / 50, 1e-40)` for the main subset. The
  likelihood then adds a 2% fractional floor plus jitter in quadrature.
- Blocker/gap: the documentation overstates the effective uncertainty formula
  as using `max(abs(obs_flux), abs(model_flux))`; current code uses observed
  flux for MAP/MCMC/posterior-target paths and model flux for amortized
  likelihood/posterior-predictive diagnostics unless `error_floor_reference`
  is overridden.
- Blocker/gap: older local processed datasets with synthetic `fluxerr_*`
  columns (`*_photometry_truth.parquet` and `03_31`) have manifests that note
  synthetic SNR errors, but their schema JSON files do not list
  `flux_error_columns`; they should be treated as historical artifacts or
  regenerated if promoted again.
- Validation completed: public config-load smoke for Diffsky/FS2 configs,
  local parquet/manifest/schema inventory, CLI help checks, and
  `uv run python -m sphinx -W --keep-going -b html docs/source outputs/doc_audit_html`.

## 2026-06-19 Jean-Zay Dataset Preflight For z<0.35 Runs

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Root cause from Jean-Zay logs: the no-KL job used the intended main dataset
  path,
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  but that derived parquet was not present in the remote checkout, so training
  failed during catalog fingerprinting before any model step.
- Updated H100 Slurm scripts to treat the z<0.35 continuous subset as a
  first-class runtime dependency. Before training/inference/validation they now
  check `DATASET`, print its size when present, or rebuild it with
  `diffsky-redshift-subset` from `SOURCE_DATASET` using
  `REDSHIFT_MIN=0.0`, `REDSHIFT_MAX=0.35`, `ERROR_MODEL=fractional_snr`, and
  `ERROR_SNR=50`.
- The same preflight was added to:
  `scripts/diffsky_autoencoder_nokl_h100.slurm`,
  `scripts/diffsky_amortized_infer_h100.slurm`, and
  `scripts/diffsky_full_validation_h100.slurm`.
- Documentation updated to state that the main subset has `78651` objects and
  that H100 entry points can rebuild the derived subset on Jean-Zay when the
  source `*_photometry_truth_noerr.parquet` exists.
- Validation completed: `bash -n` on the three edited Slurm scripts, rebuilt
  Sphinx HTML in both documented output locations, and `git diff --check`.

## 2026-06-18 Remove 0.4 Redshift Island From Main Subset

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Recompute the main continuous-low-z Diffsky subset with
  `redshift_true <= 0.35` instead of `<= 0.5` to exclude the separated
  `z~0.4` island seen in the redshift histogram.
- Keep the main parquet path stable:
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`,
  but update manifests, configs, Slurm defaults, docs, and rebuilt HTML docs to
  state the `0.0--0.35` subset contract.
- Update Jean-Zay run names to include `z035` so new no-KL and
  posterior-predictive runs are not confused with older `lowz` outputs.
- Recomputed the stable main parquet with `78651` objects,
  `redshift_true` range `0.006876--0.334662`, median `0.249744`, and the same
  `fractional_snr` SNR 50 `fluxerr_*` model.
- Copied the rebuilt redshift/truth plots into `docs/source/_static/` and
  rebuilt both `docs/build/html/diffsky_dataset.html` and
  `docs/_build/html/diffsky_dataset.html`; both HTML pages now show the z<0.35
  subset plots and no `z~0.4` island.
- Validation completed: `python -m compileall euclid_dsps scripts`, `bash -n`
  on H100 Slurm scripts, config-load smoke for active Diffsky configs,
  `uv run ruff check euclid_dsps tests`, targeted pytest (`63 passed`), and
  full `uv run pytest` (`349 passed, 5 skipped, 2 warnings`).

## 2026-06-18 Continuous Low-z Dataset As Main Diffsky Dataset

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Promote
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  to the default 04_14 Diffsky training/evaluation dataset. The old
  `*_photometry_truth_noerr.parquet` catalog remains a source/input artifact
  for rebuilding subsets, not the default science training catalog.
- Recompute the continuous redshift subset with materialized `fluxerr_*`
  columns using the shared flux-dependent error model and write companion
  manifest, schema, truth summary/report, and distribution plots.
- Clean active 04_14 configs so PopCosmos-like runs expose all SFH ratio bins
  in the latent/free-parameter vector. Low-dimensional legacy experiment
  filenames should no longer freeze `dlog10_sfr_2..6`.
- Update Jean-Zay Slurm defaults so no-KL autoencoder and posterior predictive
  residual diagnostics operate on the continuous low-z subset by default.
- Document in `.rst` the main dataset path, the uncertainty model,
  likelihood residual units, encoder flux/error feature standardization,
  standardized-logit latent coordinates, DSPS physical parameter decoding, and
  SSP/compressed SSP usage.
- Recomputed the main subset:
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`
  with `78651` objects, `redshift_true` range `0.006876--0.334662`,
  median `0.249744`, and 14 materialized `fluxerr_*` bands from the
  `fractional_snr` SNR 50 model. Companion manifest, schema, summary,
  truth summary/report, and redshift/truth distribution plots were written.
- Generated the SSP/dust audit under
  `outputs/reports/diffsky_dust_ssp_audit/`, including
  `dust_ssp_audit.md`, `dust_ssp_audit.json`, `dust_transmission_grid.csv`,
  and `dust_transmission_grid.png`.
- Validation completed after the documentation/config cleanup:
  `python -m compileall euclid_dsps scripts`, `bash -n` on the updated Slurm
  scripts, config-load smoke for active Diffsky configs,
  `uv run ruff check euclid_dsps tests`, targeted pytest (`63 passed`),
  `git diff --check`, and full `uv run pytest`
  (`349 passed, 5 skipped, 2 warnings`).

## 2026-06-18 Proposed Diffsky Model Sanity Plan

- Status: implemented on branch `feature/diffsky-likelihood-sanity-plan`.
- Add likelihood-consistent normalized residual diagnostics for post-training
  inference: signed `(flux_in - flux_out) / sigma_eff` histograms globally and
  per band, box/violin summaries by band, Gaussian reference curves, and
  vertical markers at -3 and +3. Use the same `sigma_eff` definition as the
  training likelihood, including flux error model, floor, and jitter, and
  write both plots and machine-readable summary tables with tail fractions.
- Generalize the existing posterior-predictive residual plots so every
  residual labeled "sigma" is dimensionless and computed from flux units
  consistently. Keep the sign convention explicit: input/catalog flux minus
  predicted/model flux.
- Reactivate all PopCosmos-like SFH bins by making `dlog10_sfr_4`,
  `dlog10_sfr_5`, and `dlog10_sfr_6` free parameters in the relevant
  Diffsky HLTDS configs, with matching bounds, latent names, encoder latent
  dimension, RealNVP dimension, inference summaries, truth/proxy diagnostics,
  and MAP start generation updated together.
- Audit the current Prospector/FSPS dust parameters (`tau2`, `dust_index_n`,
  `tau1_over_tau2`) against the SSP/data assets and DSPS implementation:
  confirm physical meaning, allowed ranges, wavelength behavior, young/old
  stellar population split, and whether the current bounds are coherent for
  HLTDS closure tests.
- Define and implement an explicit per-galaxy/per-band uncertainty model as a
  function of flux for `*_noerr` catalogs. Candidate models to compare:
  fractional-SNR Gaussian, magnitude-tolerance propagation, flux floor plus
  fractional floor, and, only if survey exposure/depth metadata exists, a
  Poisson/depth-inspired model. The chosen model must materialize
  `fluxerr_*` columns or an equivalent runtime array and be used identically by
  training, inference diagnostics, and MAP diagnostics.
- Add normalized latent coordinates for the autoencoder path: train the
  encoder/NF in a standardized latent coordinate system, then denormalize or
  invert-transform before passing physical parameters to DSPS. Preserve the
  existing bounded-parameter safety checks and document the exact transform in
  run summaries.
- Add a no-KL autoencoder sanity experiment: train with `kl_weight=0` and the
  NF prior disabled or ignored, then check whether the encoder+DSPS decoder can
  reconstruct catalog fluxes on train/validation using the normalized residual
  diagnostics. This is the pass/fail gate before interpreting learned priors.
- Build a continuous-redshift 04_14 subset dataset for easier first training:
  identify the continuous-redshift slice, recompute a processed catalog for
  that subset, report object counts, redshift distribution, and ground-truth
  parameter distributions, then point dedicated smoke/science configs at this
  subset.
- Validation order after implementation: compileall and targeted tests,
  regenerate the subset manifest, run a no-KL autoencoder smoke, run normal
  KL/RealNVP training only if the no-KL flux reconstruction passes, and inspect
  normalized residual tails before scaling.
- Implemented `euclid_dsps.photometric_uncertainty` as the shared source for
  flux-dependent synthetic errors and likelihood `sigma_eff`; Diffsky no-error
  AB-mag presets now synthesize `fractional_snr` errors by default, while
  prepared/subset datasets can materialize `fluxerr_*` columns.
- Implemented likelihood-consistent posterior-predictive residual diagnostics:
  signed `(flux_in - flux_out) / sigma_eff`, global and per-band histograms
  with Gaussian reference and +/-3 markers, by-band box summaries, and
  machine-readable tail tables.
- Implemented full 12D PopCosmos-like HLTDS latent by freeing
  `dlog10_sfr_4..6`, updating encoder latent dimension, adding standardized
  logit latent coordinates, and preserving identity normalization for
  supervised-prior checkpoint configs.
- Implemented `diffsky-redshift-subset` to build continuous-redshift prepared
  parquet subsets with optional `fluxerr_*` materialization, object counts,
  redshift/truth summaries, manifests, and plots.
- Implemented `diffsky-dust-ssp-audit` for SSP wavelength/age summaries and
  Prospector/FSPS dust transmission curves over configured dust parameter
  bounds.
- Added configs for the continuous low-z subset and no-KL autoencoder sanity
  check, plus `scripts/diffsky_autoencoder_nokl_h100.slurm` for train+infer
  reconstruction validation.
- Validation completed locally: `python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, `bash -n` on the new Slurm script,
  `git diff --check`, `diffsky-redshift-subset` smoke on 100 objects, targeted
  amortized/config tests, and full `pytest` (`346 passed, 8 skipped`).

## 2026-06-18 Model/Likelihood Audit

- Current analysis slice: audit the current DSPS/amortized model rather than
  changing science behavior. Answer the outstanding questions on photometric
  error modeling, normalized flux residuals, latent/flux normalization,
  RealNVP prior initialization/evolution, and the exact DSPS parameterization.
- Inspect the code paths that load catalog errors, build likelihood weights,
  normalize observed fluxes/features, initialize/train the NF prior, and map
  latent vectors into DSPS inputs. Record gaps and follow-up implementation
  work in this plan after the audit.
- Audit completed: the active Diffsky HLTDS H100 configs use the `*_noerr`
  processed catalog with no native or synthetic flux-error columns, so the fit
  currently relies on the per-band `sigma_mag=0.10` model-tolerance fallback
  plus likelihood floor/jitter rather than a Poisson or survey-depth noise
  model. Older/non-noerr prepared catalogs and OpenUniverse helpers use a
  simple Gaussian fractional-SNR recipe (`sigma = abs(flux) / snr`, usually
  SNR 50), not a Poisson derivation.
- Audit completed: MAP comparison outputs already contain
  likelihood-space normalized residuals (`chi_likelihood`) including
  floor/jitter, while amortized posterior-predictive residual summaries use
  raw photometric errors, the opposite sign convention, and absolute-value
  histograms. Follow-up: add signed `(flux_in - flux_out) / sigma_eff`
  residual plots with Gaussian overlays and +/-3 lines using exactly the
  likelihood sigma.
- Audit completed: encoder photometry features are robustly normalized per
  band, but DSPS parameters are physical bounded parameters mapped through an
  unconstrained logit latent; they are not z-scored. Flux amplitude is set by
  the surviving stellar mass parameter plus trainable global/per-band
  calibration, not by a separate DSPS normalization parameter.
- Audit completed: the joint RealNVP prior is initialized from random
  RealNVP coupling networks over the unconstrained latent, with a standard
  normal base distribution and small scale clamp; current stable configs
  freeze the prior for early epochs and then alternate encoder/prior updates.
  Existing training logs expose `logprior_mean`, `kl_mc_mean`,
  `prior_grad_norm`, `update_phase`, entropy, and likelihood-temperature
  evolution, but a dedicated prior-evolution report/plot should be added for
  science runs.
- Audit completed: the current Diffsky HLTDS decoder is a PopCosmos-binned
  DSPS proxy with free `z_obs`, stellar mass, three SFH ratios, stellar
  metallicity, and dust parameters. Later SFH ratios, gas/AGN, several dust
  details, SSP metadata policy, and calibration priors are fixed by config.
  The runtime uses the configured SSP HDF5 and compressed SSP basis rather
  than plain DSPS package defaults.

## 2026-06-17 Diffsky Collapse-Fix Validation Ladder

- Current implementation slice: keep redshift free and keep training
  unsupervised. Truth columns are used only for closure diagnostics and plots.
- Added `PLAN_FIX.md` as the implementation checklist for phases A-E:
  extended truth diagnostics, actual `popcosmos_bins` decoder proxy closure,
  MAP-Adam multistart likelihood-landscape diagnostics, low-dimensional
  controlled configs, and automatic collapse gates.
- Implemented extended truth snapshots/diagnostics for inference and MAP runs:
  posterior/MAP/prior-vs-truth tables, object-level parquet outputs, truth
  vs estimate plots, bias plots, population overlays, and compact population
  corner plots for comparable/proxy parameters.
- Implemented `diffsky-popcosmos-proxy-closure` plus
  `scripts/diffsky_popcosmos_proxy_closure_h100.slurm` to validate the exact
  amortized `popcosmos_bins` decoder against Diffsky truth proxies before
  interpreting encoder/NF results.
- Implemented MAP-Adam start modes `encoder`, `prior`, `z_grid`, `lowz_grid`,
  `latin_hypercube`, and `mixed`, with per-start-family parquet/CSV outputs
  and plots. This supports likelihood-only debug runs via `PRIOR_WEIGHT=0.0`.
- Added H100 configs for full widened redshift bounds and low-dimensional
  `z + mass + SFH proxy + dust` diagnostics:
  `diffsky_hltds_popcosmos_full_bounds_h100.yaml`,
  `diffsky_hltds_popcosmos_lowdim_z_mass_dust_h100.yaml`, and
  `diffsky_hltds_popcosmos_lowdim_z_mass_dust_lowz_h100.yaml`.
- Added automatic training/inference/MAP collapse gates that write JSON
  artifacts and surface pass/warn/fail status in run summaries.

## 2026-06-17 Diffsky Unsupervised RealNVP Stabilization

- Current implementation slice: keep the science objective fully
  photometry-driven and unsupervised. Do not train on `redshift_true` and do
  not fix redshift; truth columns are closure diagnostics only.
- Stabilize the encoder/DSPS/RealNVP path so the learned NF prior captures
  population degeneracies instead of amplifying a collapsed amortized posterior.
  The target changes are likelihood-temperature scheduling, posterior entropy
  floors, delayed/alternating prior updates, prior-predictive distribution
  diagnostics, and checkpoint gates based on non-supervised collapse signals.
- Add dense inference batching so selected rows are packed into full
  `jax_batch_size` batches rather than hundreds of sparse catalog-window
  micro-shards.
- Add an inference finalizer that can combine shard outputs, report incomplete
  runs, compute diagnostics, and generate plots after a Slurm interruption.
- Add a MAP-Adam under learned RealNVP prior workflow so final science
  estimates can optimize the DSPS photometric likelihood plus learned prior
  log-density, with redshift remaining free.
- Implemented in the working tree:
  `configs/experiments/diffsky_hltds_joint_realnvp_unsup_stable_h100.yaml`,
  anti-collapse objective controls in the amortized ELBO/training loop,
  prior update schedules, prior-predictive color diagnostics, dense selected
  inference batches, an `amortized-finalize-inference` CLI, and
  `diffsky-map-adam-prior` plus `scripts/diffsky_map_adam_prior_h100.slurm`.
- Validation completed: `uv run python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, targeted amortized/redshift pytest
  (`18 passed`), CLI help smokes for train/infer/finalize/MAP, `bash -n` on
  the three H100 Slurm scripts, and `git diff --check`.
- Next Jean-Zay order: run a 5k/5-epoch RealNVP smoke, infer 1k with dense
  shards, run MAP-Adam 1k under the learned prior, then scale to 30k/20 epochs
  only if prior-predictive colors, entropy/log-std diagnostics, photo-z closure
  plots, and MAP optimizer traces look non-collapsed.

## 2026-06-16 Diffsky Photo-z RealNVP Calibration

- Current implementation slice: refocus the next Diffsky H100 run on
  recovering redshift from photometry and comparing against true redshift,
  with `joint_realnvp` as the main prior path rather than treating
  `standard_normal` as a science target.
- Drop fixed-z closure from the main decision gate. The gate is now
  free-redshift inference on a diagnostic subset: photo-z bias, RMSE,
  outlier fraction, PIT/coverage, residuals by band, chi2 outliers, and
  redshift bias by truth-redshift bin. Stellar mass remains a degeneracy
  symptom, not the primary pass/fail metric.
- Add trainable per-band flux calibration in the amortized path so the model
  can absorb small zero-point/color mismatches without forcing redshift or mass
  to compensate. The calibration is saved to JSON/CSV/PNG and included in
  training/inference summaries.
- Add a train-only H100 Slurm entry point for amortized Diffsky runs, and add
  a dedicated `joint_realnvp` photo-z H100 experiment config with per-band
  calibration enabled, post-annealing best-checkpoint selection, sharded
  inference defaults, and resumable outputs.
- Add `photoz_metrics_by_redshift_bin.csv` to the standard inference photo-z
  metrics so the diagnostic gate has an explicit redshift-bin bias/coverage
  table, not only per-object samples.
- Follow-up after the failed 10k/5k smoke: inference and metric outputs now
  preserve `row_index`, write `inference_truth.parquet`, prefer row-index
  truth joins, fail on duplicate legacy object-id joins, save catalog
  fingerprints, and support balanced/random/stratified inference selection
  through CLI and H100 Slurm exports. The photo-z H100 config now defaults to
  balanced redshift selection and a conservative first RealNVP
  `kl_weight_max=0.05`.
- Validation completed with `python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, targeted amortized pytest
  (`19 passed`), CLI help smokes for `amortized-train-diffsky` and
  `amortized-infer-diffsky`, and `bash -n`
  on the H100 train/infer/full-validation Slurm scripts.

## 2026-06-15 Diffsky H100 Pipeline Cleanup

- Current implementation slice: clean up the H100 Diffsky science pipeline after
  the partial Jean-Zay run showed supervised-prior collapse, bad
  `best.eqx` selection under KL annealing, and non-resumable monolithic
  inference writes.
- Remove supervised-prior stages from the default full-validation science path.
  Keep the supervised-prior code available as an explicit diagnostic/dev tool,
  but do not feed collapsed supervised priors into amortized science stages.
- Make the default science sequence: true-parameter forward closure,
  `standard_normal` amortized baseline, `joint_realnvp` amortized target,
  sharded inference, redshift diagnostics, population realism, and final report.
- Select amortized `best.eqx` on a stable post-annealing metric
  (`validation_negative_loglike` by default) instead of the annealed
  `validation_loss`.
- Add sharded, resumable amortized inference outputs so long runs write useful
  partial products batch by batch and avoid concatenating large posterior
  samples/flux tables in memory.
- Add Slurm entry points for cleaned full validation and checkpoint-only
  sharded inference from an existing `last.eqx`/epoch checkpoint.
- Completed in the working tree: H100 full validation defaults now skip
  supervised-prior stages and run `standard_normal` plus `joint_realnvp`;
  amortized `best.eqx` selection uses post-annealing
  `validation_negative_loglike`; inference can write resumable shards with
  compact combined summaries and optional large predictive/residual outputs;
  redshift and population reports can consume posterior sample shards; and
  `scripts/diffsky_amortized_infer_h100.slurm` launches checkpoint-only
  sharded inference from an existing training run.
- Validation completed with `uv run python -m compileall euclid_dsps scripts`,
  `uv run ruff check euclid_dsps tests`, targeted `pytest` for
  redshift/diagnostics/full-validation
  reporting, CLI help smokes for `diffsky-run-full-validation` and
  `amortized-infer-diffsky`, H100 config-load smoke, and `bash -n` on the H100
  Slurm scripts.
- Review follow-up completed in the working tree: standalone
  `amortized-infer-diffsky --jax-batch-size` now overrides
  `amortized.inference.jax_batch_size`, and shard resume metadata now includes
  run/batch partition details plus an object-id digest. Redshift and population
  diagnostics prefer `posterior_shards_manifest.json` so stale shards in the
  same output directory are ignored after a completed run.
- Review follow-up validation completed with `uv run python -m compileall
  euclid_dsps scripts`, `uv run ruff check euclid_dsps tests`, targeted
  `pytest` (`11 passed`), CLI help smokes, `bash -n` on the H100 Slurm scripts,
  and `git diff --check`.

## 2026-06-11 Documentation Style Refresh

- Current implementation slice: improve the public-facing documentation style
  without changing runtime behavior or scientific assumptions.
- Focus on the first surfaces a reader sees: top-level `README.md`, the
  Sphinx landing page, and lightweight HTML styling/configuration.
- Keep the existing Sphinx/RTD dependency set; avoid introducing new
  documentation tooling for a cosmetic pass.
- Completed in the working tree: rewrote the README entry point with workflow
  tables and clearer sectioning; rebuilt the Sphinx landing page around
  reader goals, workflow stages, boundaries, and guardrails; grouped the
  toctree into Getting Started, Science Workflows, and Reference; added a
  local RTD-theme CSS override; and normalized visible heading style in the
  installation/run setup pages.
- Validation completed with `uv run sphinx-build -W --keep-going docs/source
  docs/_build/html`.

## 2026-06-11 Diffsky Student-t and Global SED Scale Calibration

- Follow-up implementation slice: close the remaining review gaps by adding a
  real `diffsky-run-full-validation` entrypoint/report, prior-predictive
  photometry with raw/scaled fluxes, alpha/likelihood metadata in redshift
  metric outputs, conditional architecture metadata, and a stronger frozen
  alpha update test.
- Follow-up completed in the working tree: added
  `euclid_dsps.diffsky_full_validation`, CLI command
  `diffsky-run-full-validation`, report-only aggregation for existing stage
  outputs, prior-predictive DSPS photometry output
  `prior_predictive_flux.parquet`, likelihood/alpha metadata in
  `photoz_metrics.csv`, conditional architecture component reporting, and
  stronger tests for frozen alpha and full-validation report contents.
- Follow-up validation completed with `uv run ruff check euclid_dsps tests`,
  `uv run python -m compileall euclid_dsps tests`, `uv run sphinx-build -W
  --keep-going docs/source docs/_build/html`, CLI help smoke for
  `diffsky-run-full-validation`, and `uv run pytest` (`332 passed, 5 skipped,
  2 warnings`).
- Current implementation slice: make the Diffsky science path use a robust
  Student-t photometric likelihood by default and add a single global
  `alpha_sed` nuisance parameter for SED/flux normalization mismatch.
- Keep `alpha_sed` outside supervised truth-prior learning. It is a global
  decoder/likelihood calibration parameter, not a per-galaxy physical
  component of `theta`.
- Apply the global scale exactly once on model photometry paths, expose
  raw/scaled flux diagnostics, and report the stellar-mass degeneracy through
  raw and alpha-corrected mass summaries.
- Update forward closure with disabled/fixed/fit-global alpha modes and
  before/after residual outputs. Update amortized training/inference so
  `log_alpha_sed` can be trainable or frozen independently of the prior.
- Update public configs, tests, and docs so the new default is explicit and
  Gaussian remains available only as an explicit ablation choice.
- Implementation completed in the working tree: added
  `euclid_dsps.calibration`, Student-t defaults for Diffsky science configs,
  amortized trainable/frozen global `log_alpha_sed`, raw/scaled flux outputs,
  alpha-corrected mass summaries, forward-closure alpha modes and reports,
  H100 experiment config metadata, and documentation of the calibration
  nuisance parameter.
- Validation completed with `uv run ruff check euclid_dsps tests`,
  `uv run pytest` (`328 passed, 5 skipped, 2 warnings`),
  `uv run python -m compileall euclid_dsps tests`, config-load smoke for all
  public YAML files, and `uv run sphinx-build -W --keep-going docs/source
  docs/_build/html`.

## 2026-06-11 Diffsky Physical Validation PR 3-7

- Implemented the remaining Diffsky physical-validation scaffold from
  `PR3_PR7_TODO.md` while keeping the three scientific objectives separate:
  supervised truth-prior learning, same-parameter forward closure, and
  photometric amortized inference.
- PR 3 completed in the working tree: added
  `euclid_dsps.diffsky_forward_closure`, CLI command
  `diffsky-forward-closure`, config
  `configs/diffsky_hltds_04_14_trueparam_closure_gpu.yaml`, and tests for
  truth-column mapping, missing Diffstar/Diffmah failures, fixed nuisance
  metallicity recording, and mock zero-residual closure.
- PR 4 completed in the working tree: amortized inference now supports
  `standard_normal`, `supervised_checkpoint`, and `joint_realnvp` prior
  sources; frozen supervised priors have their gradients zeroed before the
  optimizer update; loaded supervised checkpoints validate latent names and
  bounds against the active amortized schema; new public configs cover the
  three Diffsky prior modes.
- PR 5 completed in the working tree: added
  `euclid_dsps.diffsky_redshift_ablation`, CLI command
  `diffsky-redshift-ablation`, per-run `photoz_metrics.csv` and
  `posterior_vs_truth_metrics.csv` outputs during inference, and ablation
  reports with bias, sigma MAD, RMSE, outlier fraction, PIT, coverage, and
  posterior-width metrics.
- PR 6 completed in the working tree: extended Diffsky population-realism
  diagnostics to include logSFR/logSSFR derived quantities when exported, dust
  terms when fitted, Diffstar/Diffmah/burst generated-truth marginals for
  supervised-prior diagnostics, q_agg/prior/truth plots, and
  `population_realism_report.md`. Raw `dlog10_sfr_i` ratios are intentionally
  not compared directly to `logsfr_true`.
- PR 7 completed in the working tree: added public configs and docs for the
  dataset, supervised priors, true-param closure, amortized prior-source modes,
  redshift ablation, and scientific validation plan. Older simple/fixed-z
  Diffsky configs remain documented as legacy/debug paths rather than the main
  physical-validation path.
- Validation completed in `conda run -n shine`: targeted Ruff on modified
  files, targeted PR3-PR7 pytest, `compileall euclid_dsps scripts`, CLI help,
  config-load smoke for seven public configs, Sphinx `-W --keep-going`, and
  full pytest (`318 passed, 4 skipped, 1 warning`). The skipped MCMC tests are
  due to an installed NumPyro/JAX incompatibility and now use the module's
  existing compatibility check.

## 2026-06-11 Diffsky Physical Validation PR 1/2

- Current implementation slice: split the new Diffsky/HLDTS validation path
  into PR 1 dataset-integrity/truth-semantics work and PR 2 supervised
  truth-prior learning. Keep both separate from photometric amortized
  inference and from true-parameter forward closure.
- PR 1 targets in progress: make prepared object ids globally unique while
  preserving `core_tag`, classify columns into truth/generated/derived/
  diagnostic/proxy/unavailable semantics, record the exact error model, and
  write a first-class dataset integrity report.
- PR 2 targets in progress: add an independent `prior_learning` package and
  CLI commands for supervised RealNVP density learning on truth parameters,
  with its own schema reduction/missing-column policy, logs, samples, summary,
  and truth-vs-prior diagnostics.
- PR 1 dataset-integrity slice completed in the working tree: prepared HLTDS
  parquet files now preserve `core_tag`, guarantee unique `object_id` values,
  add `global_object_id/source_file/source_row` when source ids are duplicated,
  classify columns by truth semantics in manifest/schema/reports, write
  explicit `error_model` provenance, and generate
  `diffsky_dataset_integrity_report.md`.
- PR 2 supervised-prior slice completed in the working tree: added
  `euclid_dsps.prior_learning`, configs for basic/extended supervised RealNVP
  priors, CLI commands `diffsky-train-supervised-prior`,
  `diffsky-sample-supervised-prior`, and `diffsky-supervised-prior-report`,
  plus toy/schema/output tests and documentation. This path uses truth columns
  directly and remains separate from photometric amortized inference.
- Validation completed in `conda run -n shine`: targeted Ruff, targeted
  pytest, full `compileall euclid_dsps scripts`, CLI help, config-load smoke
  for the two new prior configs, and Sphinx `-W --keep-going`.
- Added `PR3_PR7_TODO.md` as the remaining implementation checklist. Next
  priority is PR 3 same-parameter Diffsky forward closure; PR 4 should only
  start after PR 3 establishes whether truth parameters reproduce HLTDS
  photometry well enough for physical recovery claims.
- Updated `PR3_PR7_TODO.md` to align explicitly with the full prompt:
  A/B/C objective separation, PR dependency gates, command/config examples,
  expected test files, and global acceptance criteria.

## 2026-06-10 Prior-Learning Physical Audit

- Current audit: answer which parameters are actually fitted by the learned
  prior workflows, whether the compact Diffsky/PopCosmos setup is coherent
  with the DSPS forward model, which dataset columns are treated as truth, and
  which physical caveats remain before interpreting redshift inference under
  the learned degenerate prior.
- Audit outcome: Diffsky amortized prior learning fits a compact 9D
  PopCosmos-bin DSPS latent (`z_obs`, stellar mass, three SFR-ratio
  parameters, stellar metallicity, and three dust parameters) with
  `dlog10_sfr_4..6` fixed. FS2 keeps the full 16D PopCosmos-like latent. The
  learned RealNVP is a variational population prior in unconstrained latent
  space and inference currently samples the trained encoder approximation,
  not an exact reweighted/sampled posterior under a frozen learned prior. HLTDS
  truth columns are redshift, stellar mass, sSFR/SFR, halo/central/size fields;
  Diffstar/Diffmah/dust/burst columns are generated-truth diagnostics. Main
  caveats are decoder/data-model mismatch, no native HLTDS photometric errors,
  small truth tails outside configured bounds, and duplicated object IDs at
  shard boundaries.

## 2026-06-10 Diffsky HLTDS Amortized Prior Learning

- Objective validated: learn a joint degenerate population prior over redshift
  and compact DSPS physical parameters from HLTDS photometry, compare learned
  prior and aggregate posterior to direct/generative dataset distributions,
  then evaluate redshift recovery under that learned prior.
- Implemented public config
  `configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`.
  It uses HLTDS 04/14 LSST+Roman 14-band AB magnitudes, feature dimension 28,
  a 9D latent (`z_obs`, stellar mass, three SFH-ratio parameters, metallicity,
  and dust parameters), Gaussian encoder, RealNVP prior, fixed DSPS decoder,
  and the HLTDS SSP compressed basis asset.
- Compressed HLTDS SSP support validated. The builder now preserves physical
  axes at source precision while compressing the large basis/coefficient
  payload, avoiding wavelength-axis mismatch for the long HLTDS wavelength
  tail.
- The amortized data path is now generic in `B` and preserves large Diffsky
  `object_id` values as `int64` outside JAX, avoiding silent int32 truncation
  in posterior and validation outputs.
- The amortized decoder now merges compact latent free parameters with
  `model.fixed_parameters` before calling DSPS, so compact schemas can use the
  existing PopCosmos-bin forward model without exposing every nuisance
  parameter in the encoder.
- Added CLI paths `amortized-train-diffsky`,
  `amortized-infer-diffsky`, and `amortized-prior-overlap-diffsky`.
  The overlap report writes CSV/JSON/Markdown plus plots comparing direct truth
  to `q_agg` and the learned RealNVP prior for `z_obs` and
  `log10_stellar_mass`.
- Smoke validation completed on CPU with 8-object training, 4-object
  inference, and prior-overlap report. This validates code execution only; it
  is not a scientific-quality run.
- CUDA stability fix after first user GPU smoke: the Diffsky amortized config
  now upcasts compressed SSP arrays to resident `float32` and caps actual
  compiled DSPS/JAX batches with `amortized.training.jax_batch_size: 4` and
  `amortized.inference.jax_batch_size: 4`. User-facing `--batch-size` can stay
  larger; the command logs the internal cap.
- Scientific limit kept explicit: `logsfr_true` is not yet directly comparable
  to fitted `dlog10_sfr_*` ratios. A derived DSPS SFR diagnostic is still
  required before claiming SFR recovery.

## 2026-06-10 Public Pipeline Cleanup

- Public config surface reduced to:
  `configs/fs2_gpu.yaml`,
  `configs/diffsky_hltds_04_14_simple_gpu.yaml`,
  `configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml`, and
  `configs/amortized_fs2_realnvp.yaml`. The public surface now also includes
  `configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml` for HLTDS learned
  prior experiments.
- Deprecated public configs removed from `configs/`: old Diffstar variants,
  old OpenUniverse fit-ready experiments, old non-GPU Diffsky variants, and
  broad ablation configs.
- Main documented dataset is now
  `hltds_cosmos_260215_04_14_2026`, prepared as
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet`
  with `--no-synthetic-errors`.
- Main Diffsky fit model is deliberately simple: PopCosmos-bin DSPS, no AGN,
  fixed nebular SSP treatment, no Diffstar/Diffmah latent recovery, and direct
  comparison only to available basic truth columns (`redshift_true`,
  `logsm_true`, `logsfr_true`).
- Documentation reset completed for the public path: downloader, local data
  locations, Diffsky dataset contract, FS2 commands, GPU runtime, and
  amortized FS2 prior learning.

## 2026-06-08 Experimental SSP INR Compression

- Objective: investigate NeRF/implicit-neural-representation style compression
  for SSP HDF5 cubes as an experimental path, separate from the production
  dense and low-rank DSPS model paths.
- Scope for first implementation: add isolated training/evaluation/report CLI
  commands for direct coordinate MLPs and latent spectral-basis MLPs; compare
  against dense HDF5 payloads and existing low-rank compressed assets; write
  loss curves, reconstruction plots, metrics JSON/CSV, and Markdown reports.
- Integration policy: do not add a new `model.ssp_model` runtime mode until the
  asset-level benchmarks show a useful error/runtime/memory tradeoff.
- First implementation completed in the working tree:
  `euclid_dsps.experimental.ssp_inr` now provides HDF5 subset loading, direct
  Fourier/SIREN-style coordinate MLPs, latent spectral-basis MLPs, checkpoint
  serialization, evaluation against existing compressed assets, plots, CSV/JSON
  metrics, and Markdown reports.
- CLI commands added: `experimental-ssp-inr-train`,
  `experimental-ssp-inr-eval`, and `experimental-ssp-inr-report`.
- Validation completed with targeted smoke runs on the real Chabrier stellar
  SSP, `uv run pytest tests/test_experimental_ssp_inr.py -q`,
  `uv run ruff check euclid_dsps/experimental/ssp_inr euclid_dsps/cli.py
  tests/test_experimental_ssp_inr.py`, `uv run python -m compileall
  euclid_dsps scripts`, and `uv run python -m euclid_dsps.cli --help`.
- First benchmark review completed after running the quick/full SSP, gas, and
  AGN experiments. Direct coordinate MLPs are slower and less accurate than the
  latent path. The latent path is promising for stellar SSP in all-wavelength
  log metrics, but the existing low-rank assets remain much better on
  physically relevant wavelength/flux masks and are especially dominant for
  AGN. Next improvement should be metric design and a stronger latent residual
  architecture before any runtime integration.
- Second implementation pass in progress: add physically masked metrics,
  log-SVD baselines, end-to-end latent log-flux training, residual latent
  models over existing compressed assets, AGN `fagn` factorization, and zoomed
  reconstruction plots.
- Second implementation pass completed: `experimental-ssp-inr-eval` now writes
  `useful_wave`, `peak1em04`, `peak1em06`, and combined useful/significant
  metrics; supports repeatable `--log-svd-k`; and writes full, useful-wave, and
  UV/optical reconstruction plots plus masked error histograms. Latent training
  defaults to end-to-end Huber loss on weighted log-flux reconstruction, with
  optional coefficient loss, residual training over a compressed baseline, and
  AGN `fagn` factorization. Validation passed with targeted Ruff, targeted
  pytest, full `compileall`, CLI help checks, and a real-asset smoke run.
- Runtime follow-up: experimental SSP INR modules now apply the repository JAX
  runtime configuration before importing JAX and report a concise GPU backend
  installation error when `--runtime gpu` is requested but CUDA JAX is missing
  or version-mismatched.
- GPU run blocker diagnosed after the first full residual latent command:
  WSL sees an RTX 4060 Laptop GPU with driver 581.80 / CUDA 13.0, but the
  active `shine` environment reported an incompatible `jax_cuda13_plugin`
  version relative to `jaxlib`. The next full benchmark should only be rerun
  after reinstalling matching CUDA 13 JAX wheels in `shine`; CPU/auto commands
  remain usable for smoke/debug runs.
- Current phase: review the newly generated full SSP INR benchmark outputs,
  compare direct INR, latent residual, explicit log-SVD, and existing compressed
  baselines, then identify the next implementation improvements.
- Benchmark review outcome: for stellar SSP, `v2_stellar_residual_k32_full`
  is much better than the direct Fourier coordinate MLP and gives useful-wave
  p95 errors around a few percent, but it is still slower and less accurate
  than the existing low-rank compressed asset on useful/significant wavelengths.
  `v2_stellar_latent_k128_full` has lower training loss but worse useful-mask
  reconstruction, showing that the current loss is not aligned enough with the
  science metric. The direct coordinate model is currently disadvantaged by
  linear wavelength normalization over 100--9.95e7 Angstrom and should be
  retested with log-wavelength encoding plus masked/weighted sampling.
- Next priorities from the review: make evaluation plots/metrics mask-aware by
  default, add log-wavelength and useful-wave weighting for direct INR, add
  coefficient-table replacement experiments over existing low-rank bases,
  rerun high-dimensional gas/AGN with v2 objectives and factorization, and use
  actual photometry/filter wavelength coverage to define the reconstruction
  domain.
- Current implementation phase: add an experimental coefficient-MLP model that
  keeps an existing compressed spectral basis but replaces the coefficient table
  with a neural map from curve coordinates to coefficients; make comparison
  plots science-mask-first; add ECDF, per-wavelength, age/metal heatmap, and
  worst-spectrum plots for stronger benchmark diagnosis; prepare serious
  stellar/gas/AGN benchmark commands.
- Implementation completed: `compressed_coeff_mlp` is available through
  `experimental-ssp-inr-train`, using `--coeff-baseline`,
  `--coeff-loss={coeff,log_flux,mixed}`, `--coeff-log-weight`, and optional
  `--factor-agn-fagn`. Checkpoints store the kept basis plus coefficient
  standardization statistics and reconstruct spectra by predicting coefficients
  from curve coordinates. Evaluation now supports these checkpoints, uses
  useful/significant-wave error in the runtime-size plot, and writes ECDF,
  wavelength-profile, age/metal heatmap, and worst-case reconstruction plots.
  Validation passed with compileall, Ruff, targeted pytest, real stellar
  coefficient-MLP smoke train/eval, and real AGN factorized smoke train.

## 2026-06-05 OpenUniverse / Diffsky Validation Pivot

- Objective: change the validation priority from FS2-first to
  OpenUniverse/Diffsky-first, using LSST+Roman 14-band photometry as the first
  OpenUniverse target while keeping FS2 only as a domain-shift comparison and
  diagnostic dataset.
- Truth policy: label directly available OpenUniverse quantities as `truth`,
  future Diffsky/Diffstar exports as `generated_truth`, derived stand-ins as
  `proxy`, and absent quantities as `unavailable`. Do not describe FS2 catalog
  proxies as physical ground truth.
- First implementation slice (PR 1) in progress: add
  `euclid_dsps.openuniverse` schema, parquet IO/join, local/S3 path resolution,
  photon-rate unit contract, toy noise models, LSST+Roman subset preparation,
  `openuniverse-prepare` CLI, configs, docs, and synthetic unit tests. No bulk
  OpenUniverse download should be triggered by default.
- PR 1 completed in the working tree: mini OpenUniverse parquets can be joined
  and normalized into 14 truth fluxes, 14 noisy fluxes, 14 errors, and 14 masks;
  the CLI refuses implicit full-dataset processing; OpenUniverse units remain
  native photon-rate with explicit conversion TODOs; Diffsky extended truth
  extraction is a clear optional-dependency placeholder; toy photo-z and
  prior-overlap metrics are available for later reports; the amortized encoder
  now reads `input_dim` from config so B=14 feature tensors have shape `[N, 28]`.
- PR 1 validation completed with `uv run python -m compileall euclid_dsps scripts`,
  targeted Ruff, Sphinx `-W --keep-going`, targeted pytest (`75 passed`), and
  full pytest (`255 passed, 5 skipped`). Remaining warning: extended Diffsky
  truths are unavailable unless an optional Diffsky export path is provided.
- 2026-06-08 EDA completed for downloaded OpenUniverse preview hpix 10307:
  raw main/flux files each have 3,561,877 unique galaxy ids and join exactly;
  prepared subset has 10,000 rows, 65 columns, all 14 LSST+Roman masks valid,
  median subset redshift 0.983, and median log10 stellar mass 7.999. Report
  written under `outputs/reports/openuniverse_eda_10307/` with schema catalog,
  physical/flux stats, QA tables, and first diagnostic plots. Caveat: the
  10,000-row subset is a head/limit preview sample and not representative.
- 2026-06-08 PR 2/3 slice implemented on
  `feature/openuniverse-truth-sed-validation`: standalone
  `python -m euclid_dsps.openuniverse.cli` commands for truth inventory, basic
  truth export, photo-z metrics, prior-overlap metrics, and OpenUniverse
  B=14 feature stats; bounded SED HDF5 inventory/reader for
  `/meta/wave_list` plus `/galaxy/<prefix>/<galaxy_id>` datasets; prepared
  parquet loader into generic `PhotometryArrays`; and synthetic tests. Real
  hpix 10307 inventory found 312 wavelength bins, 3,561,877 SED datasets shaped
  `(3, 312)`, direct truth for redshift/redshiftHubble/stellar_mass/fluxes,
  generated-truth low-resolution SED, proxy MW/lensing columns, and unavailable
  SFH/Diffstar/dust-internal/metallicity/halo latents.
- 2026-06-08 data-first photon closure slice implemented without touching the
  DSPS decoder: OpenUniverse filter loading, SED Fnu to integrated photon-rate
  helpers, `sed-flux-closure` CLI, per-band robust calibration factors, and
  `merge-external-truth` for explicit future Diffsky latent tables. Real
  hpix 10307 LSST-only closure over 200 galaxies used exact repository LSST
  filters and found calibrated median residuals near zero, sigma_MAD about
  2.4--3.0%, and p95 absolute calibrated residuals about 6--8%. Roman closure
  remains blocked on exact Roman filter curves unless top-hat smoke filters are
  explicitly enabled.
- 2026-06-09 fit-ready OpenUniverse slice implemented data-side, without
  changing the DSPS decoder: `make-fit-ready` computes `mu_lensing` from
  convergence/shear, preserves public photon-rate fluxes in explicit lensed
  audit columns, writes unlensed photon columns, and converts the standard
  `flux_*`/`fluxerr_*` columns to DSPS-compatible `fnu_cgs`. The conversion
  uses per-band AB0 photon rates and defaults to `filter_response_mode:
  dsps_clipped` so Roman effective-area-like filters match
  `euclid_dsps.filters.load_ascii_filter`. Local hpix 10307 validation wrote
  `Data/openuniverse/processed/ou_lsst_roman_14_subset_fit_ready.parquet`,
  confirmed B=14 feature stats with feature_dim=28, and ran a one-object MAP
  smoke fit. Caveat: Roman zeropoints/response conventions still need deeper
  validation before MAP parameters should be interpreted scientifically.
- Follow-up slices: optional full Diffsky latent export, physical
  OpenUniverse train/infer commands after photon-rate decoder/conversion,
  redshift ablation on real posteriors, prior-overlap plots from model outputs,
  and FS2-vs-OpenUniverse comparison plots.

## 2026-06-02 Andrew Hearin Discussion Preparation

- Objective: prepare a didactic discussion package for a meeting with Andrew
  Hearin about the current DSPS/PopCosmos-like Euclid+LSST forward-modeling
  work.
- Deliverables completed under
  `outputs/report/andrew_hearin_discussion_2026-06-02/`: one explanatory
  Markdown document with Mermaid architecture diagrams, a RevealJS slide deck,
  a concise question list, and a metrics summary grounded in existing
  run/benchmark artifacts.
- Slide refresh completed: replaced the first pipeline diagram and SED
  construction diagram with static HTML diagrams, added detailed slides on
  model ingredients, `model.py`'s DSPS adapter role, active free parameters, and
  all current free-parameter bounds.
- Follow-up slide refresh completed after review: slide 4 now uses a vertical
  detail stack for physical ingredients and code changes relative to base DSPS;
  the dust slide now explains the old-star versus young-star attenuation model
  with equations and code mapping; the `model.py` slide now has vertical
  details for context loading, forward pass, mass normalization, and diagnostic
  versus likelihood paths.
- Added generated component figures from the real JAX model components and
  slides explaining emission-line construction, AGN component construction,
  AGN/IGM ordering, flux-space Student-t errors, and surviving-mass
  normalization.
- Second visual refresh completed: replaced the main component plot with
  `sed_component_build_row471_detail.png` because row 471 has visible AGN/IGM
  effects; added `emission_lines_before_after_row471.png`,
  `agn_igm_detail_row471.png`, and `igm_before_after_transmission.png` to show
  emission-line before/after, AGN ratios, and IGM transmission directly.
- Updated the AGN optical-depth bound explanation: `ln_tauagn` bounds map to
  tau 5.0 to 148.0, matching the PopCosmos/Prospector hard-limit convention and
  the FSPS tabulated `agn_tau` range before extrapolation.
- Rewrote the final Andrew questions to focus on decisions: minimal per-galaxy
  model, gas/line corrections, AGN/dust/IGM conventions, likelihood/model
  mismatch, and population-prior strategy.
- Amortized learned-prior slide update completed: added a vertical RevealJS
  stack for the FS2-only Gaussian encoder + RealNVP prior + fixed DSPS decoder
  prototype, with Mermaid architecture/output diagrams, ELBO details, caveats,
  and dedicated Andrew questions.
- Source policy: use existing code, configs, `PLAN.md`, documentation, and
  outputs as evidence; do not rerun expensive MAP/posterior jobs for this
  communication package unless explicitly needed.
- Main talk caveats to keep explicit: the current model is FSPS/Prospector-like
  rather than an official PopCosmos reproduction; PopCosmos learned emission-line
  corrections are not included; existing MAP SED diagnostics did not load the
  COSMOS proxy columns even though the FS2 parquet contains them; full-AGN MAP
  is under-constrained from 10 bands and shows redshift, gas, and AGN
  degeneracies.

## 2026-06-02 PopCosmos Learned Prior Preparation

- Branch: `feature/popcosmos-prior-learning`, created from the current
  `feature/mclmc-posterior` state.
- Feature objective: prepare for learning a POP-COSMOS-style prior from the
  existing PopCosmos/FSPS compressed workflow outputs and catalog-derived
  parameter distributions.
- Initial cleanup policy: keep scientific inputs in `Data/` and run products
  in `outputs/` untouched. Only remove local Python/test/tool caches during
  branch preparation.
- Preparation completed: removed local `__pycache__`, `.pytest_cache`, and
  `.ruff_cache` directories from source/test areas; deliberately did not run
  broad `git clean` because it would remove local `Data/` assets and
  `outputs/` products.
- Detailed implementation planning moved into local running plan
  `posterior_plan.md` after the updated prompt. This file is intentionally not
  for commit; it tracks PR 0 through PR 5, API contracts, tests, docs, and open
  implementation decisions for the FS2-only amortized posterior feature.
- Implementation breakdown:
  - PR 0: public `parameter_vectors.py`, array photometry bridge, and
    architecture docs.
  - PR 1: `amortized/latent.py`, `features.py`, and `likelihood.py`.
  - PR 2: Equinox Gaussian encoder, RealNVP prior, and optional dependency
    extra.
  - PR 3: DSPS decoder wrapper, Monte Carlo ELBO, and asset-free synthetic
    smoke.
  - PR 4: FS2 dataloader, joint training loop, config, and
    `amortized-train-fs2` CLI.
  - PR 5: inference, catalog export, diagnostics, and
    `amortized-infer-fs2` CLI.
- Implementation completed in the working tree:
  - Added public theta-vector helpers and array photometry extraction.
  - Added `euclid_dsps.amortized` with latent transforms, features,
    Student-t likelihood, Equinox encoder, RealNVP prior, decoder, ELBO,
    synthetic smoke, FS2 training, inference, catalog export, and diagnostics.
  - Added `configs/amortized_fs2_realnvp.yaml`, optional dependency extra
    `amortized`, CLI commands, tests, README/Sphinx docs, and lockfile update.
  - Runtime follow-up: a real DSPS inference run with
    `posterior_samples=32` and `batch_size=8` exhausted GPU memory because all
    posterior samples were decoded in one DSPS vmap. Inference now chunks the
    posterior predictive decode over the sample axis, defaults
    `decoder_sample_chunk_size` to 1, exposes
    `--decoder-sample-chunk-size`, and records the chunk size in
    `inference_summary.json`.
  - Follow-up from the first chunked inference run: posterior predictive files
    were finite, but bright low-redshift FS2 objects produced extreme linear
    encoder features and poor flux predictions. New feature stats now default
    to `flux_transform: asinh` while legacy stats without a transform remain
    linear for checkpoint compatibility. Inference now writes normalized
    posterior predictive residual tables, top chi-square objects, feature-scale
    diagnostics, redshift proxy comparisons when available, and diagnostic
    plots including a compact posterior corner plot.
  - Next diagnostic addition requested: treat the RealNVP as a learned prior
    product, not only a KL term. Inference should sample `p_beta(x)`, export
    learned-prior samples in theta space, compare learned prior versus aggregate
    amortized posterior with true corner contours, write redshift distribution
    comparisons, and add a redshift PIT diagnostic when FS2 redshift proxy
    columns are present.
  - Implemented learned-prior inference diagnostics: `learned_prior_samples`
    now stores both latent `x_00` ... `x_15` and physical `theta` samples with
    exact `logprior`; diagnostics write learned-prior quantiles, a log-density
    histogram, contour-style learned-prior and posterior-vs-prior corners,
    redshift distribution comparison, and redshift PIT tables/plots.
  - Added explicit FS2 catalog proxy diagnostics for the amortized outputs:
    `catalog_proxy_comparison.parquet/csv`, stellar-mass posterior/prior/proxy
    histograms, stellar-mass proxy residual histogram, catalog SFR proxy
    distribution, and proxy mass-SFR plane. These are deliberately labeled as
    catalog proxies; SFR is not overlaid against posterior samples until a
    model-derived `log10_sfr_at_obs` export is added.
  - Added anti-collapse training controls for Jean Zay/H100 runs: reproducible
    FS2 row selection with `sequential`, `random`, or `stratified_redshift`
    modes; balanced/proportional redshift stratification; epoch-level training
    shuffle; train/validation split artifacts; validation ELBO rows in
    `training_log.csv`; `validation_redshift_bin_metrics.csv`; validation loss
    plots by redshift bin; validation-based `best.eqx` checkpointing when a
    validation split exists; and explicit `kl_weight_max` plus longer default
    KL annealing.
  - Cleaned training history diagnostics after the first H100 run: raw
    `training_log.csv` remains batch-level, but plots are now recomputed from
    `training_epoch_summary.csv` using `epoch` on the x-axis, train/validation
    means with p16/p84 bands, a compact overview figure, and redshift-bin
    validation heatmaps for loss, negative loglike, chi-square, and KL.
  - Validation passed with `uv sync`, `uv sync --extra dev`,
    `uv sync --extra dev --extra amortized`,
    `uv run python -m compileall euclid_dsps scripts`, Black on the feature
    Python files, targeted Ruff on the feature diff,
    `uv run pytest tests -q` with amortized extras installed
    (`226 passed, 5 skipped`), Sphinx `-W --keep-going`,
    `uv run python -m euclid_dsps.cli --help`, and the mock-decoder
    `amortized-synthetic-smoke` CLI.
  - Equinox/Optax remain optional dependencies; CLI commands report the missing
    dependency clearly if the `amortized` extra is not installed.
- 2026-06-02 architecture and `conda shine` recheck:
  - Static audit confirms the amortized implementation stays FS2-only, does not
    import private `fit.py` helpers, does not call `predict_batch_mags`, does
    not construct `GalaxyObservation` in the training path, and keeps Pandas
    use to IO/reporting surfaces outside the hot JAX decoder/ELBO path.
  - `conda run -n shine python -c "import jax, equinox, optax"` passed with
    JAX `0.10.1`, Equinox `0.13.7`, and Optax `0.2.8`. In this shell, JAX
    falls back to CPU because the installed CUDA plugin is incompatible with
    the installed `jaxlib`; real FS2 DSPS training should be run after fixing
    the `shine` CUDA/JAX stack if GPU is required.
  - `conda run -n shine python -m euclid_dsps.cli --help` passed and shows the
    three amortized commands without breaking the existing `check`, `fit`, and
    `posterior` command registration.
  - `conda run -n shine python -m compileall euclid_dsps scripts` passed.
  - `conda run -n shine python -m pytest tests/test_amortized_encoder.py
    tests/test_amortized_flows.py tests/test_amortized_elbo.py
    tests/test_amortized_synthetic.py -q` passed: 6 passed.
  - `conda run -n shine python -m pytest tests/test_amortized_synthetic.py -q`
    passed after tightening the synthetic summary criterion.
  - `conda run -n shine python -m euclid_dsps.cli --config
    configs/amortized_fs2_realnvp.yaml amortized-synthetic-smoke
    --mock-decoder --n-objects 64 --epochs 2 --batch-size 8 --out
    outputs/runs/dev_amortized_synthetic_prompt_check_shine` passed and wrote
    `training_log.csv`, `training_summary.json`, `feature_stats.json`, best/last
    checkpoints, and diagnostics. The summary now reports
    `loss_decreased: true` using the best observed loss while retaining
    `last_loss_decreased` for the final-mini-batch comparison.
- 2026-06-02 DSPS e2e and progressive training outputs:
  - Added `tests/test_amortized_dsps_e2e.py`, which builds a tiny synthetic
    ten-filter PopCosmos-like `DspsContext`, runs the non-mock
    `model_flux_from_x` decoder through `model_mags_jax_dynamic`, evaluates the
    ELBO, and verifies nonzero encoder and RealNVP prior gradients from the
    same `eqx.filter_value_and_grad` call.
  - Training and synthetic smoke now write `training_progress.json`, update
    `checkpoints/last.eqx` during training, write epoch checkpoints such as
    `checkpoints/epoch_0001.eqx`, and regenerate diagnostics every
    `amortized.output.diagnostics_every` epochs.
  - `training_log.csv` now includes `encoder_grad_norm`, `prior_grad_norm`, and
    `joint_grad_norm`; the diagnostics writer plots those columns to make joint
    encoder/RealNVP optimization visible.
  - Checkpoint JSON sidecars now include an architecture summary covering the
    Gaussian MLP encoder, RealNVP prior, fixed DSPS decoder, and Monte Carlo
    `logq - logp` KL objective.
  - Validation passed:
    `uv run pytest tests/test_parameter_vectors.py tests/test_amortized_latent.py
    tests/test_amortized_features.py tests/test_amortized_likelihood.py
    tests/test_amortized_encoder.py tests/test_amortized_flows.py
    tests/test_amortized_elbo.py tests/test_amortized_synthetic.py
    tests/test_amortized_dsps_e2e.py -q` (`20 passed`),
    `uv run python -m sphinx -W --keep-going -b html docs/source
    /tmp/dsps_docs_amortized_arch_check`, and
    `conda run -n shine python -m pytest tests/test_amortized_dsps_e2e.py -q`
    (`1 passed`, with the known JAX CUDA plugin warning).
  - `conda run -n shine python -m euclid_dsps.cli --config
    configs/amortized_fs2_realnvp.yaml amortized-synthetic-smoke
    --mock-decoder --n-objects 32 --epochs 2 --batch-size 8 --out
    outputs/runs/dev_amortized_synthetic_progress_check_shine` passed and wrote
    epoch checkpoints plus `encoder_grad_norm.png`, `prior_grad_norm.png`, and
    `joint_grad_norm.png`.
  - Full test suite passed after the follow-up:
    `uv run pytest tests -q` (`227 passed, 5 skipped, 1 warning`).
    `git diff --check` passed. Full-repo Ruff currently reports an unrelated
    pre-existing `B007` unused loop variable in `tests/test_fsps_grid_scripts.py`;
    targeted Ruff on the amortized files passed.
  - Added default verbose console logging and per-epoch progress bars for
    `amortized-train-fs2`. The command now reports setup stages, JAX
    backend/devices, DSPS loading, architecture summary, epoch starts, and
    epoch summaries. The progress bar displays live loss, NLL, MC KL, encoder
    gradient norm, and RealNVP prior gradient norm. CLI flags `--quiet` and
    `--no-progress` disable these outputs when needed.
  - Validation for the verbose/progress follow-up passed:
    `uv run ruff check euclid_dsps/amortized/train.py euclid_dsps/cli.py`,
    `uv run python -m compileall euclid_dsps scripts`,
    `uv run python -m sphinx -W --keep-going -b html docs/source
    /tmp/dsps_docs_verbose_check`, `git diff --check`, and
    `conda run -n shine python -m euclid_dsps.cli --config
    configs/amortized_fs2_realnvp.yaml amortized-train-fs2 --help`.
  - Fixed the first real FS2 amortized-training NaN failure:
    flux-space likelihood now evaluates in stop-gradient normalized flux units
    to avoid float32 underflow of cgs variances near `1e-30`; encoder features
    now use scale-relative floors instead of an absolute `1e-8`; the FS2
    encoder initializes near a stable PopCosmos theta point (`z_obs` around
    0.8 rather than the midpoint of the broad redshift bound); and the training
    loop skips non-finite updates instead of applying corrupt gradients.
  - Added `loss_finite`, `grads_finite`, and `update_applied` columns to
    `training_log.csv`, plus `updates_applied`/`updates_skipped` in progress
    and summary JSON.
  - NaN-fix validation passed:
    first-batch inspection reports finite loglike and finite encoder/RealNVP
    gradients; `UV_CACHE_DIR=/tmp/uv-cache uv run pytest
    tests/test_amortized_features.py tests/test_amortized_likelihood.py
    tests/test_amortized_dsps_e2e.py tests/test_amortized_synthetic.py -q`
    passed (`8 passed`); `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall
    euclid_dsps scripts` passed; `git diff --check` passed; and a direct real
    FS2 mini-run with `limit=16`, `batch_size=8`, `epochs=2` applied 4/4
    finite updates with no NaNs.
- Remaining before production use: run the real `amortized-train-fs2` smoke in
  `shine` once the FS2/DSPS assets and desired CUDA-capable JAX stack are
  active; this was not run during the architecture audit.

## Current State

2026-06-01 MCLMC implementation phase:

- Branch `feature/mclmc-posterior` starts from committed MAP recovery fixes.
- Target implementation: single-galaxy 16D posterior sampling for compressed
  PopCosmos using experimental BlackJAX MCLMC while leaving NumPyro HMC/NUTS
  untouched.
- Implemented contract: `sample.sampler: mclmc` routes through a pure-JAX
  bounded posterior target over an unconstrained vector, with log-Jacobian,
  prior terms, existing Gaussian/Student-t distribution log-probs, and the
  PopCosmos gas metallicity constraint.
- Current limitation: first MCLMC backend is unadjusted MCLMC with configured
  or automatic `L` and `step_size`; it is an engineering diagnostic sampler
  until compared against HMC/NUTS on selected rows.
- Debug/progress update: the MCLMC path now reports JAX backend/devices,
  compile/warmup/sample phase timings, optional per-chunk diagnostics, and
  chunked progress bars controlled by `sample.mclmc_progress_chunk_size`.
- Stability update: the BlackJAX target now reuses the existing chi-based
  Student-t objective up to an additive constant, uses a triangular transform
  for the free `log10_stellar_metallicity`/`log10_gas_metallicity` pair, and
  avoids nesting a separately jitted MCLMC step inside `lax.scan`.
- Multi-chain update: MCLMC now supports `sample.num_chains` through sequential
  chains to keep VRAM bounded. Posterior predictive magnitudes are written in
  configurable chunks, chain metadata is exported, trace plots mark chain
  boundaries, and posterior summaries/corner truth plots include catalog
  redshift truth as `truth_z_obs`.
- Initialization update: MCLMC supports `sample.init_strategy` values `map`,
  `map_jitter`, `config`, and `random_uniform`. `map_jitter` is the preferred
  first diagnostic for multi-chain runs because it keeps chains near a good MAP
  mode but gives each chain a distinct unconstrained starting point.
- Guardrail update: MCLMC now raises a clear error if a warmup or sampling phase
  returns zero valid transitions (`nonans=0` everywhere), preventing frozen MAP
  repeats from being written as apparent posterior samples.

2026-06-01 large MAP finalization fix:

- Observed failure: a `configs/popcosmos_binned_compressed.yaml` MAP run with
  `--limit 10000 --batch-size 128` completed 78 full chunks, then crashed at
  9984/10000 when the final 16-row chunk triggered a new JAX/XLA GPU graph
  capture path (`cuda_blas_lt.cc` workspace failure).
- Fix: independent batch MAP now pads partial chunks internally to the requested
  static `batch_size`, then filters synthetic rows before checkpoint/final
  outputs. This keeps one stable JAX batch shape for long production runs. The
  padding is not applied to population/hierarchical fits because duplicated
  rows would change the statistical objective.
- Recovery tool: `scripts/finalize_fit_from_chunks.py` concatenates existing
  `_chunks` checkpoints and regenerates aggregate MAP tables, QC plots, and
  completion metadata without rerunning the optimizer. For the interrupted
  n=10000 run, it can produce a scientifically usable report over the 9984
  completed galaxies and explicitly records the missing row indices.

2026-05-30 speed optimization phase:

- Goal: improve PopCosmos compressed MAP throughput and GPU batch capacity for
  learned-prior/posterior production.
- Current measured bottleneck: `jax_optimize` dominates runtime on
  `configs/popcosmos_binned_compressed.yaml`; warm chunks take about 75 s for
  64 galaxies at 200 Adam iterations.
- Implementation target: remove per-iteration diagnostic forward passes from
  the MAP loop when configured, and use a likelihood-only PopCosmos
  `model_mags` path that avoids constructing diagnostic SED components during
  optimization.
- Validation target: `model_mags` fast path must match the full forward model
  magnitudes, and the default dense/science configs must remain compatible.
- Implemented slice: `fit.trace_mode` and `fit.trace_interval` now control
  expensive Adam trace diagnostics. `optimizer` mode records optimizer loss and
  gradient summaries without per-iteration photometric diagnostic forward
  passes; `none` disables trace rows. The compressed PopCosmos config defaults
  to `optimizer` every 20 iterations.
- Implemented slice: PopCosmos binned and Diffstar reduced6 now have
  likelihood-only JAX magnitude paths. MAP, population MAP, MCMC likelihoods,
  and benchmark code that ask only for model magnitudes no longer build the
  diagnostic SED result object.
- Implemented slice: population MAP uses the analytic bounded-transform chain
  rule instead of evaluating a second full loss gradient each Adam step.
- Smoke result on the local RTX 4060 Laptop GPU with
  `configs/popcosmos_binned_compressed.yaml`, `batch-size=64`, and 200 Adam
  iterations: the optimized path completed 128 galaxies in 188.9 s with a
  2.44 GiB GPU peak. The previous 512-galaxy compressed run used 4.66 GiB peak
  and 1.32 s/gal overall; warm-chunk runtime is now lower and peak GPU memory is
  roughly halved for the same compressed model family.

2026-05-31 forward/runtime optimization phase:

- Goal: continue reducing GPU memory pressure and wall time for production MAP
  batches used before learned posterior/prior training.
- New target: remove avoidable `age x wavelength` intermediates from
  likelihood-only forward passes. The current dust models apply one attenuation
  curve to old populations and one to young populations, so the fast path can
  sum young/old SSP contributions before applying dust while preserving the
  full diagnostic forward model unchanged.
- Benchmarking policy: save every smoke/runtime run under `outputs/runs/` and
  write a progress report under `outputs/report/` so the speed progression can
  be shown externally.
- Result: the young/old dust-sum rewrite and a hand-decomposed photometry
  rewrite were both rejected after GPU benchmarks because they increased XLA
  memory and runtime. They are documented as negative results, not kept in the
  model code.
- Kept optimization: the GPU runtime preset now sets
  `TF_GPU_ALLOCATOR=cuda_malloc_async`, which removed the large allocator/autotune
  memory spike on the RTX 4060 Laptop GPU.
- Kept optimization: `--reporting-level light` now actually skips plot-heavy
  batch reports while keeping tables, JSON summaries, and performance
  benchmarks.
- Current best production setting on the local GPU: compressed full-AGN binned
  model with `batch-size=128`, `maxiter=200`, `sed-samples=0`, and
  `reporting-level=light`. A two-chunk smoke run processed 256 galaxies in
  175.35 s with a 4.49 GiB GPU peak; the warm optimization chunk took 63.83 s
  for 128 galaxies, roughly twice the warm throughput of the previous
  `batch-size=64` run.

2026-05-31 JAX compilation/options investigation:

- Goal: test whether JAX writing/compilation knobs can improve the compressed
  PopCosmos MAP path beyond simply running on GPU.
- Implemented reproducible fit options for controlled benchmarking:
  `fit.scan_unroll`, `fit.donate_optimizer_inputs`, and
  `fit.remat_model_mags`, exposed through the CLI as `--fit-scan-unroll`,
  `--fit-donate-inputs`, and `--fit-remat-model-mags`.
- Result: all three options stay disabled by default. On the local RTX 4060
  Laptop GPU, `scan_unroll=2` gave no warm-speed improvement and much higher
  compile/memory pressure; input donation produced JAX donation warnings and
  was slower; rematerializing the magnitude forward was slower and did not
  reduce the real graph memory.
- Kept recommendation: use the compressed config defaults with the GPU runtime
  preset, which now sets `TF_GPU_ALLOCATOR=cuda_malloc_async`.
- Added `scripts/check_photometry_equivalence.py` to compare the full
  diagnostic forward path against the optimized likelihood-only magnitude path.
  The current compressed PopCosmos smoke check reports exactly zero magnitude
  difference for the sampled points.
- Report and plots:
  `outputs/report/jax_compilation_investigation_2026-05-31/`.

2026-05-31 advanced optimizer/MCMC investigation:

- Tested a deeper JAX autodiff rewrite for independent MAP batches:
  `vmap(value_and_grad(single_objective))` versus
  `value_and_grad(sum(vmap(single_objective)))`. The latter is exposed as
  `fit.batch_grad_mode: sum` and CLI `--fit-batch-grad-mode sum` for future
  hardware benchmarks.
- Result: rejected as production default on the local RTX 4060 Laptop GPU.
  Warm optimization time was essentially unchanged, while compile time and peak
  GPU memory roughly doubled. Keep `fit.batch_grad_mode: per_galaxy`.
- MCMC path now calls `model_mags_jax_dynamic` so large compressed DSPS arrays
  are passed as model arguments before future sampler experiments.
- Local `shine` has NumPyro but not `blackjax`, `jaxopt`, `optax`, or `flowMC`.
  MCLMC therefore requires a separate dependency/backend slice rather than a
  config-only switch.
- A compressed full-model HMC smoke with one warmup, one sample, and one
  leapfrog was terminated after more than two minutes of CPU-heavy compilation.
  Posterior acceleration should be benchmarked separately with a sampler
  backend designed for this use case.
- Report:
  `outputs/report/jax_advanced_optimization_2026-05-31/`.

2026-05-31 wavelet compression benchmark:

- Added `scripts/benchmark_wavelet_compression.py` to compare the current
  low-rank compressed assets against a sparse Haar-wavelet oracle on sampled
  dense stellar SSP, gas, and AGN spectra.
- The benchmark intentionally measures resident payload size and spectral
  reconstruction loss, not disk compression. It does not add a runtime wavelet
  mode yet.
- Result: keep the current SVD/factored-SVD compression as the production
  direction for gas and AGN. Sparse Haar is much worse for AGN and larger or
  worse for gas at the tested loss levels. It is only competitive for the base
  stellar SSP on the sampled median spectral metric, but its worst-tail error
  and sparse-index runtime complexity need a separate photometric/JAX benchmark
  before it can be considered.
- Report and plots:
  `outputs/report/wavelet_compression_benchmark_2026-05-31/`.

2026-05-31 compressed production documentation/config update:

- Promoted `configs/popcosmos_binned_compressed.yaml` as the documented
  production MAP config and added `configs/popcosmos_diffstar_compressed.yaml`
  for the compressed Diffstar comparison path.
- Added `docs/source/ssp_compression.rst` and linked it from the Sphinx index.
  The page documents the SVD `basis/coeff/scale` format, gas and AGN compressed
  parameter axes, the wavelet audit result, asset build commands, and the
  production compressed fit command.
- Updated README and Sphinx run/setup/forward-model/science/testing docs so
  dense configs are described as reference/closure paths while compressed
  configs are the recommended high-throughput runtime path.
- Validation run before commit: `compileall`, `pytest tests/test_config.py`,
  `pytest tests/test_model.py`, `pytest tests/test_fit_memory.py
  tests/test_mcmc.py`, Sphinx `-W --keep-going`, and compressed photometry
  equivalence smoke all pass locally in `conda shine`.

2026-05-31 MCLMC posterior planning:

- Added `PLAN_MCLMC.md` as the implementation plan for a BlackJAX MCLMC
  posterior backend.
- Current decision: do not put posterior sampling on the critical path for the
  next MAP sprint. First run compressed MAP `n=10k`, generate QC, test SSP
  `k128`, rerun dense-vs-compressed `n=500`, then scale to `n=100k+`.
- MCLMC should be developed in parallel as an experimental posterior backend
  over the compressed model. It needs a pure JAX log-density over an
  unconstrained parameter vector, a bounded-parameter transform with Jacobian,
  lazy BlackJAX dependency handling, and a benchmark against current NumPyro
  HMC/NUTS on selected MAP-QC rows.
- Use `mclmc` as the canonical spelling in config/code/docs. Treat unadjusted
  MCLMC as an engineering/diagnostic sampler until compared against NUTS/HMC;
  add adjusted MCLMC as the science candidate if the installed BlackJAX API
  supports it.

Branch objective: integrate the Diffstar SFH implementation from
`feature/diffstar` into the current PopCosmos FSPS gas/AGN workflow without
losing the generated-grid scripts, GPU/JIT memory fixes, or documentation
cleanup.

The repository currently has six active PopCosmos-like configurations:

```text
configs/popcosmos_binned.yaml
configs/popcosmos_binned_compressed.yaml
configs/popcosmos_diffstar.yaml
configs/popcosmos_diffstar_compressed.yaml
configs/popcosmos_binned_noagn.yaml
configs/popcosmos_diffstar_noagn.yaml
```

The compressed full AGN configs are now the recommended production path:
`popcosmos_binned_compressed` first for high-throughput GPU MAP batches, and
`popcosmos_diffstar_compressed` as the comparison. The dense full AGN configs
remain available for dense-vs-compressed and FSPS/Prospector closure checks.
The no-AGN configs remain available for controlled ablations and fallback
debugging.

They are standalone and represent the current PopCosmos-like DSPS/JAX paths:

- LSST `ugrizy` plus Euclid VIS/Y/J/H photometry.
- Flux-space likelihood with catalog per-band flux errors.
- Configurable photometric objective, with POP-COSMOS-style Student-t as the
  current default for both full AGN and no-AGN PopCosmos-like configs.
- Seven-bin PopCosmos-like SFH ratios in `popcosmos_binned.yaml`.
- Six-free-parameter Diffstar SFH in `popcosmos_diffstar.yaml`.
- Single stellar metallicity.
- Explicit age-dependent dust modes:
  `charlot_fall_powerlaw` preserves the previous behavior, while
  `prospector_fsps` is the current approximate Prospector/FSPS-like target mode.
- Madau95-style approximate IGM.
- PopCosmos-like assets must be explicit Chabrier assets with HDF5 metadata:
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5`,
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5`,
  `Data/popcosmos_chabrier_gas_ssp_grid.h5`, and
  `Data/popcosmos_chabrier_agn_component_ssp_grid.h5`.
- The first one-row `popcosmos_binned_noagn` smoke fit passes after generating
  those assets, with `EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0` and
  `JAX_PLATFORMS=cuda` in the `shine` environment.
- PopCosmos-like `z_sun` is `0.0142`; metallicity conversion is
  `log10(Zstar_abs) = log10(0.0142) + log10_stellar_metallicity`.
- Full configs keep the 16-parameter fit including SFH, gas, and AGN.
  No-AGN configs remove `ln_fagn` and `ln_tauagn`.
- `configs/popcosmos_binned_compressed.yaml` and
  `configs/popcosmos_diffstar_compressed.yaml` inherit their dense full AGN
  configs and swap only the resident spectral assets to compressed SSP, gas,
  and AGN component bases.

Older config variants, presets, examples, and partial smoke configs have been
removed from source. Tests now target the active configs plus low-level
synthetic fixtures.

## Scientific Caveats

- This is PopCosmos-like, not yet an audited reproduction of the full
  POP-COSMOS population model.
- Independent gas and stellar metallicity variation in FSPS is useful for
  fitting but not fully self-consistent for all nebular line ratios.
- A hard PopCosmos-like constraint now enforces
  `log10_gas_metallicity >= log10_stellar_metallicity`, with both quantities in
  `log10(Z/Zsun)`.
- `model.emission_line_corrections: none` means the gas grid is
  `uncalibrated_cloudy`. This remains usable for development but is non-final
  for a science prior. `popcosmos_table` support exists for enriched grids with
  `nebular_continuum_flux`, `emline_wavelengths`, and `line_flux_grid`.
- `prospector_fsps` dust is a JAX approximation to the FSPS/Prospector
  `dust_type=4` plus birth-cloud behavior. It uses `dust2=tau2`,
  `dust1=tau1_over_tau2*tau2`, `dust_tesc_logyr=7.0`, and
  `dust1_index=-1.0`, but still needs direct FSPS/Prospector benchmarking before
  it can be called an exact reproduction.
- Legacy `agn_model: template_grid` still follows the repository convention
  `fagn * integrated stellar Lbol` and remains an audit-only approximation.
  The active `agn_model: fsps_component_grid` path instead loads an FSPS-native
  AGN component SSP grid and convolves it with the same SFH weights as the
  stellar SSP.
- `agn_host_attenuation: fsps_diffuse_unit_tau` is the current FSPS-aligned
  AGN host attenuation convention: it applies the PopCosmos/Prospector-like
  diffuse `dust_type=4` attenuation curve at unit V-band optical depth to the
  AGN component, matching the local FSPS `agn_dust.f90` convention instead of
  multiplying by `dust2`. The older `diffuse` and `prospector_fsps` scaled AGN
  modes remain diagnostics only.
- The IGM model is a stable Madau95-style approximation.
- The fit and post-fit batch prediction paths pass large SSP/gas/AGN arrays as
  dynamic JAX arguments so JIT can stay enabled without compiling the gas grid
  as a multi-GiB XLA constant.
- Gas-grid interpolation uses direct four-corner bilinear interpolation in
  `(gas_lgmet, gas_lgu)`, avoiding a batch-scaled intermediate over the full
  gas-ionization axis.

## Standard Commands

Generate/validate the FSPS Chabrier SSP, gas, and AGN assets as documented in
`docs/source/data_download.rst`. The Chabrier SSP is generated locally with
`scripts/generate_fsps_ssp_grid.py`; it is not a legacy `manage_ssp.py`
download.

Short one-row fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_compressed_fullagn_one_short \
  --sed-samples 1
```

Batched fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_binned_compressed.yaml \
  fit --limit 1000 \
  --batch-size 128 \
  --fit-maxiter 200 \
  --out outputs/runs/popcosmos_binned_compressed_map_n1000_bs128 \
  --sed-samples 0 \
  --reporting-level light
```

Diffstar short one-row fit:

```bash
python -m euclid_dsps.cli \
  --config configs/popcosmos_diffstar_compressed.yaml \
  fit --index 0 \
  --fit-maxiter 20 \
  --out outputs/runs/dev_popcosmos_diffstar_compressed_fullagn_one_short \
  --sed-samples 1
```

Verification:

```bash
uv run python -m compileall euclid_dsps scripts
uv run pytest tests
uv run python -m euclid_dsps.cli --config configs/popcosmos_binned_compressed.yaml fit --help
uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar_compressed.yaml fit --help
```

## Remaining Work

### SSP Shape And Compression Investigation

2026-05-28 SSP shape investigation phase started:

- Scope: inspect the Chabrier SSP HDF5 assets, generate plots by wavelength,
  age, and metallicity dimensions, and compare simple compression candidates
  for reducing SSP storage without erasing spectral information.
- Primary assets: `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5` and
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5`.
- Candidate diagnostics: sample spectra, age and metallicity tracks, spectral
  heatmaps, SVD energy/error, piecewise quadratic fits in log wavelength, and a
  simple Haar-wavelet proxy on a log-wavelength grid.
- Outputs should be written under `outputs/ssp_shape_investigation/` and kept
  out of source.

2026-05-28 SSP shape investigation phase completed:

- Added `scripts/investigate_ssp_shapes.py`, a reproducible HDF5 diagnostic
  script for SSP shape plots and simple compression-error metrics.
- Generated plots and metrics under `outputs/ssp_shape_investigation/`,
  including wavelength, age, and metallicity slices; heatmaps; noNE-vs-wNE
  ratios; SVD energy; reconstruction examples; and compression tradeoffs.
- The active Chabrier SSP grids are both `12 x 107 x 11149`; the default
  compression diagnostics focus on 900-30000 Angstrom, which contains 9239 of
  the 11149 wavelength samples.
- Initial result: a shared low-rank basis is the strongest first candidate for
  stellar-continuum compression. On a 2048-point log-wavelength grid, 32 SVD
  components give p95 log-flux errors around 0.013-0.015 dex at roughly 24x
  nominal compression, and 64 components give p95 errors around 0.004-0.005 dex
  at roughly 12x nominal compression.
- Piecewise quadratics are not competitive for narrow spectral structure:
  they only reach p95 below 0.05 dex at low compression, especially for the
  fixed-nebular SSP.
- Haar-style sparsity is useful as a baseline but not clearly better than SVD
  at the tested tolerances. If wavelets are pursued, use a real codec with
  quantization and explicit coefficient/index storage accounting.
- New priority if compression becomes implementation work: split continuum and
  emission-line information before compressing gas or fixed-nebular grids,
  rather than fitting narrow lines with smooth polynomial segments.

2026-05-28 SSP compression documentation extension started:

- Scope: make cleaner summary plots, add explicit interpolation/knot-storage
  tests, document candidate compressed representations, and clarify the VRAM
  tradeoff for larger galaxy batches.
- Additional diagnostics should distinguish file size, resident VRAM size,
  interpolation reconstruction error, and extra compute from on-the-fly
  decoding.

2026-05-28 SSP compression documentation extension completed:

- Extended `scripts/investigate_ssp_shapes.py` with log-wavelength knot
  interpolation tests and cleaner compression-frontier plots.
- Added `outputs/ssp_shape_investigation/popcosmos_asset_size_summary.png` to
  show that the gas SSP grid dominates the resident tensor size.
- Added `docs/source/ssp_compression.rst` and linked it from the docs index.
  The doc recommends a low-rank continuum representation plus separate sparse
  line storage, and records the key caveat that VRAM only drops if JAX consumes
  the compressed representation directly.
- Verification: `uv run python -m compileall scripts/investigate_ssp_shapes.py`,
  `uv run ruff check scripts/investigate_ssp_shapes.py`, and
  `uv run sphinx-build -b html docs/source /tmp/dsps_docs_ssp_check` pass.

2026-05-28 dedicated implementation planning update:

- Added `PLAN_SSP_COMPRESSION.md` as the complete execution plan for testing
  linear-flux compression, continuum/line separation, compressed HDF5 assets,
  JAX integration, age-dependent dust handling, and dense-vs-compressed
  benchmarks.
- Next step remains gated on explicit user approval before implementing the
  compressed-grid experiments or changing model code.

2026-05-29 VRAM-compression planning update:

- Reframed `PLAN_SSP_COMPRESSION.md` around resident GPU memory reduction, not
  disk-level HDF5 compression. The compressed representation must be consumed
  directly by JAX; decoding back to dense gas/AGN tensors is explicitly
  non-goal.
- First implementation step is now to create a clean branch from `dev`, without
  creating a separate worktree. Existing dense `Data/` assets are reference
  inputs and must not be modified in place.
- Priority order is AGN component compression first, then gas compression with
  continuum-plus-sparse-lines, then optional base SSP compression.
- Required gates are dense-vs-compressed benchmarks for AGN-only, gas-only,
  `full_noagn`, and `full_agn`, followed by FSPS/Prospector runs at `n=50` and
  `n=500`.

2026-05-29 VRAM-compression implementation slice:

- Added metadata-first inventory and VRAM-estimation scripts under
  `scripts/inventory_spectral_assets.py` and `scripts/profile_vram_batch.py`.
  They do not instantiate `load_context` by default.
- Baseline metadata outputs were generated under `outputs/ssp_compression/`.
  The current dense full-AGN PopCosmos config has about `6.66 GiB` of estimated
  float32 resident spectral payload before JAX overhead and batch intermediates.
- Added `scripts/build_compressed_agn_component_grid.py` and
  `agn_model: compressed_fsps_component_grid`, which loads `agn_basis`,
  `agn_coeff`, and `agn_scale` instead of the dense AGN component tensor.
- Added `scripts/build_compressed_gas_grid.py` and
  `nebular_model: compressed_gas_grid`, which loads `gas_basis`, `gas_coeff`,
  and `gas_scale` instead of the dense gas `ssp_flux` tensor.
- The first compressed gas mode is explicitly a full-spectrum low-rank
  prototype for VRAM and dense-vs-compressed testing. It is not yet the final
  continuum-plus-sparse-lines science representation.
- Added `scripts/validate_compressed_spectral_asset.py` for compressed AGN and
  gas assets. It validates compressed files without reading dense source grids.
- Added `scripts/benchmark_dense_vs_compressed_spectral_assets.py` for
  identical-parameter dense-vs-compressed photometry checks with progress
  reporting. It now loads one benchmark level at a time and uses lazy dense
  AGN HDF5 slice reads by default, so the default command does not preload the
  dense AGN grid twice.
- Added `scripts/benchmark_photometry_engines.py` for subprocess-isolated
  `dsps_dense_lazy`, `dsps_dense_resident`, `dsps_compressed`, and
  `fsps_prospector` comparison, including per-engine wall time and peak RSS.
  Dense-lazy keeps the dense FSPS AGN component physics but reads only the HDF5
  slices required for each point. Dense-resident has a memory guard and is
  skipped when its estimated static payload plus overhead is too close to
  `MemAvailable`.
- Updated the FSPS/Prospector benchmark harness so compressed gas and
  compressed AGN configs can be passed after the compressed assets exist.
- Targeted verification passed with `python -m compileall euclid_dsps scripts
  tests/test_config.py tests/test_model.py tests/test_fsps_grid_scripts.py` and
  `pytest -q tests/test_config.py tests/test_model.py
  tests/test_fsps_grid_scripts.py`.

2026-05-29 deeper VRAM-compression analysis:

- The current compressed gas asset is coefficient dominated:
  `gas_coeff` is about `15.36 MiB` raw, while `gas_basis` is about `2.72 MiB`.
  Further wins should target coefficient dtype/rank or a different gas
  representation, not HDF5 repacking.
- Sampled rank truncation indicates AGN can likely shrink from `k32` toward
  `k12-k16` with little information loss, pending broad-band benchmarks.
- Sampled dense AGN slices show `agn_lnu_per_mformed / fagn` is effectively
  invariant across `fagn_grid` at the `~1e-7` relative level. The next AGN
  compressed format should remove the explicit `fagn` axis and apply `fagn` as
  a runtime multiplier, after a dedicated benchmark/test gate.
- Gas still benefits from `k48-k64`; dropping full-spectrum gas to low ranks
  damages line-sensitive structure. The preferred final representation remains
  low-rank continuum plus sparse emission-line luminosities.
- After gas/AGN compression, the base SSP is now the largest resident tensor
  left, about `55 MiB`. Revisit compressed base SSP modes at `k64/k128`; this
  can save more memory than small AGN basis tweaks once the multi-GiB grids are
  gone.
- Naively storing all compressed arrays as `float16` is unsafe because the
  scale arrays are around `1e-22..1e-11`. Mixed precision should keep scale as
  `float32` or store `log10(scale)`, and reconstruct in `float32`.
- Next experiments should benchmark mixed-precision assets:
  AGN `basis=float32, coeff=float16, scale=float32` at `k12/k16`; gas
  `basis=float16, coeff=float16, scale=float32` at `k48/k64`; and optional
  coefficient-only int8 quantization with explicit scales.

2026-05-29 compression tradeoff audit and implementation update:

- Added `scripts/audit_compression_tradeoffs.py`. It samples dense gas/AGN
  spectra without materializing full dense tensors and writes
  `outputs/ssp_compression/tradeoffs/compression_tradeoff.csv`,
  `compression_tradeoff_summary.json`, `compression_factor_vs_loss.png`, and
  `candidate_payload_mib.png`.
- Audit highlights from `n=256` sampled dense spectra:
  - AGN `fagn_factored_svd_k12_coeff16_basis32`: estimated payload
    `0.82 MiB`, compression factor `~4800x`, median sampled p95 relative loss
    `2.4e-4`.
  - AGN `fagn_factored_svd_k8_coeff16_basis32`: estimated payload
    `0.56 MiB`, compression factor `~7000x`, median sampled p95 relative loss
    `5.8e-4`.
  - Gas `k64 mixed_f16`: estimated payload `9.28 MiB`, compression factor
    `~288x`, median sampled p95 relative loss `3.7e-3`.
  - Gas `k48 mixed_f16`: estimated payload `7.02 MiB`, compression factor
    `~381x`, median sampled p95 relative loss `6.2e-3`.
  - Stellar SSP `k64` remains the recommended first compact SSP point from
    existing log-flux diagnostics: estimated payload `4.47 MiB`, p95
    `0.00437 dex`; `k128` is safer at `8.90 MiB`, p95 `0.00135 dex`.
- Implemented explicit runtime support for:
  - `model.ssp_model: compressed_basis` with `model.compressed_ssp_path`;
  - compressed SSP payloads `ssp_basis`, `ssp_coeff`, `ssp_scale`;
  - compressed AGN assets with `fagn_handling: linear_runtime_multiplier`,
    meaning `fagn` is multiplied at runtime and the coefficient tensor drops
    the `fagn` axis;
  - mixed-precision compressed payloads where basis/coefficients can stay
    `float16` resident on device while selected slices are reconstructed in
    `float32`.
- Added `scripts/build_compressed_ssp_grid.py`; extended
  `scripts/build_compressed_agn_component_grid.py` with `--factor-fagn`,
  `--basis-dtype`, and `--coeff-dtype`; extended
  `scripts/build_compressed_gas_grid.py` with `--basis-dtype` and
  `--coeff-dtype`.
- Benchmark harnesses now accept `--compressed-ssp` in addition to compressed
  gas and AGN paths.

### Implementation Plan: PopCosmos Benchmark Closure

2026-05-28 forward-model closure phase started:

- Scope: implement the next correction slice after the noNE benchmark report:
  diagnose and fix verified DSPS/FSPS mismatches in the pure-stellar
  normalization, redshift/luminosity-distance/filter path, UV/IGM behavior,
  Prospector/FSPS-like dust, and then gas/line-continuum handling.
- Current gate remains unchanged: do not train the first learned prior until
  `stellar_only`, `stellar_plus_dust`, `stellar_plus_gas`, and `full_noagn`
  pass the benchmark criteria or any remaining approximation is explicitly
  named and accepted.

Goal: turn the current PopCosmos-like no-AGN path into a benchmarked prior-v0
forward model. The `n=500` FSPS/Prospector benchmark in
`outputs/benchmarks/popcosmos_binned_noagn_fsps_n500/` runs end to end, but its
residuals are not yet within the recommended prior-readiness thresholds.

Phase 1 - Pure-stellar baseline:

- Add a pure-stellar Chabrier SSP generation mode, with
  `add_neb_emission=0` and `add_neb_continuum=0`.
- Write it to an unambiguous path such as
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5`.
- Add HDF5 attrs making the no-nebular contract explicit:
  `imf_type=1`, `imf_name=chabrier`, `z_sun=0.0142`,
  `add_neb_emission=0`, `add_neb_continuum=0`, units, FSPS version, isochrones,
  and spectral library.
- Extend the benchmark so `stellar_only` and `stellar_plus_dust` use this
  pure-stellar SSP on the DSPS side. The current `fixed_ssp` path uses the base
  SSP and therefore includes fixed nebular emission if the base SSP HDF5 was
  generated that way.
- Tests: pure-stellar SSP validation passes; benchmark level selection uses the
  pure-stellar context for gas-free levels; the old fixed-nebular SSP remains
  usable only where explicitly requested.

Phase 2 - Strict asset metadata:

- Tighten PopCosmos-like IMF validation so contradictory metadata fails.
  If `imf_type` exists it must be `1`; if `imf_name` exists it must be
  `chabrier`; if both exist they must agree; if neither exists, fail.
- Add tests for inconsistent metadata, especially `imf_type=2` with
  `imf_name=chabrier` and `imf_type=1` with `imf_name=kroupa`.
- Keep legacy lognormal assets compatible outside PopCosmos-like configs.

Phase 3 - Benchmark diagnostics:

- Extend `benchmark_summary.json` to report residual statistics per
  `(level, band)` using finite-only counts and explicit non-finite counts.
- Add residual correlations per `(level, band)` instead of only pooled across
  all levels and bands.
- Add recent-SFR diagnostics derived from the PopCosmos SFH bins, at minimum
  the youngest-bin SFR and a recent-to-old SFR contrast.
- Add plots for residuals against `z_obs`, `tau2`, `dust_index_n`,
  `log10_stellar_metallicity`, `log10_gas_metallicity`,
  `log10_gas_ionization`, and recent SFR.

Phase 4 - Rerun staged benchmarks:

- Rerun a tiny benchmark after each code/data change:
  `--n 5`, then `--n 50`, then `--n 500`.
- Compare each stage against
  `outputs/benchmarks/popcosmos_binned_noagn_fsps_n500/`.
- Do not move to prior training until broad bands are close to the target:
  median `|Delta mag| < 0.02`, p95 `|Delta mag| < 0.05`, and no clear monotone
  residual trend with redshift, dust, metallicity, gas, or recent SFR.

Phase 5 - Dust/gas correction decisions:

- If pure-stellar residuals remain large, audit SSP normalization, mass units,
  age-bin mapping, luminosity distance, filter integration, and surviving-mass
  normalization before touching gas or dust.
- If pure-stellar passes but `stellar_plus_dust` fails, benchmark the
  `prospector_fsps` dust curve directly against FSPS/Prospector as a function
  of wavelength, `tau2`, `dust_index_n`, `dust1`, and age.
- If `stellar_plus_gas` or `full_noagn` fails after the baseline is clean,
  generate an enriched gas grid with separated continuum and line fluxes, then
  apply the PopCosmos emission-line correction table.
- Keep `emission_line_corrections: none` labeled as `uncalibrated_cloudy`.

Phase 6 - Later AGN and production scaling:

- Audit AGN normalization against FSPS internals or external CLUMPY convention
  before recommending full AGN configs.
- Scale CUDA batch fitting only after the no-AGN forward model passes the
  benchmark gates.
- Add synthetic recovery tests with known DSPS-generated parameters.

2026-05-28 forward-model closure implementation update:

- Fixed the dominant pure-stellar DSPS/Prospector mismatch by replacing the
  generic log-cosmic-time DSPS age-weight interpolation with a direct overlap
  integral from PopCosmos lookback bins onto the SSP age grid when
  `model.sfh_time_grid: prospector_step` is active.
- Ported FSPS Madau95 IGM attenuation into `igm_model: fsps_madau95`, replacing
  the earlier coarse UV approximation for PopCosmos-like configs.
- Ported the FSPS/Prospector `dust_type=4` diffuse attenuation shape into the
  JAX `prospector_fsps` dust mode while preserving the existing age-dependent
  architecture: diffuse dust applies to all stellar ages; birth-cloud dust
  applies only at ages `<= dust_tesc_logyr`.
- Added optional loading and use of `ssp_surviving_mstar` from FSPS-generated
  SSP HDF5 files. This removes the remaining few-percent normalization offset
  from using the DSPS analytic surviving-mass approximation for Chabrier FSPS
  SSPs.
- Updated `scripts/generate_fsps_ssp_grid.py` to write
  `ssp_surviving_mstar[stellar_lgmet, age]` plus units metadata, and
  regenerated:
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5` and
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5`.
- Latest staged benchmark:
  `outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n20/`.
  Finite overall residuals now show:
  `stellar_only` median `|Delta mag| = 0.0060`, p95 `0.0107`;
  `stellar_plus_gas` median `0.0060`, p95 `0.0183`.
  Excluding effectively-zero flux cases with either DSPS or reference magnitude
  `>80`, `stellar_plus_dust` median is `0.0058`, p95 `0.0154`, and
  `full_noagn` median is `0.0058`, p95 `0.0193`.
- Remaining blocker is the extreme-dust/tiny-flux tail where the reference
  remains finite at magnitudes `~180` while the float32 DSPS path either
  underflows or floors around magnitudes `~100-115`. Benchmark summaries must
  keep reporting these non-finite/effectively-faint counts separately; prior
  training should use flux-space likelihoods and avoid interpreting these
  magnitude deltas as normal broadband residuals.
- The larger `n=500` benchmark completed at
  `outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n500/`.
  It confirms the `n=20` result: `stellar_only` median `|Delta mag| = 0.0058`,
  p95 `0.0115`; `stellar_plus_gas` median `0.0059`, p95 `0.0178`.
  For bright finite rows with DSPS and reference magnitudes `<80`,
  `stellar_plus_dust` median is `0.0057`, p95 `0.0125`, and `full_noagn`
  median is `0.0057`, p95 `0.0135`. The remaining raw dust/full p99 outliers
  are still the expected extreme-faint magnitude tail.
- Wrote the benchmark analysis report:
  `outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n500/analysis_report.md`.

## Latest Verification

2026-05-28 forward-model closure verification:

- Regenerated and validated the Chabrier noNE and fixed-nebular SSP HDF5 assets
  with `ssp_surviving_mstar`.
- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py
  scripts/generate_fsps_ssp_grid.py scripts/fsps_grid_common.py
  tests/test_model.py tests/test_config.py tests/test_benchmark.py
  tests/test_fsps_grid_scripts.py` passed. The repo-wide ruff check is blocked
  by an unrelated pre-existing import-order issue in
  `scripts/investigate_ssp_shapes.py`.
- `uv run pytest -q tests/test_config.py tests/test_model.py tests/test_benchmark.py`
  passed: 72 passed, 3 skipped.
- `uv run pytest -q` passed: 140 passed, 3 skipped.
- `conda run -n shine bash -lc 'JAX_PLATFORMS=cpu python
  scripts/benchmark_against_fsps_prospector.py --config
  configs/popcosmos_binned_noagn.yaml --n 20 --seed 0 --out
  outputs/benchmarks/popcosmos_binned_noagn_forwardfix_mstar_n20'` passed.

2026-05-28 AGN benchmark audit mode:

- Extended `scripts/benchmark_against_fsps_prospector.py` so configs with
  `model.agn_model: template_grid` automatically run two additional audit
  levels: `stellar_plus_agn` and `full_agn`.
- The AGN reference side now passes `fagn=exp(ln_fagn)` and
  `agn_tau=exp(ln_tauagn)` into FSPS/Prospector and enables
  `add_dust_emission` for AGN levels.
- The DSPS side uses the repository AGN template grid. The benchmark summary
  explicitly labels this as an AGN audit rather than a final PopCosmos AGN
  validation because the DSPS template-grid bolometric normalization convention
  is still approximate.
- Smoke run passed:
  `outputs/benchmarks/smoke_popcosmos_binned_agn_audit_n1/`.
  It confirms the no-AGN levels remain at `~0.005` mag while AGN levels expose
  larger UV/blue differences, especially `lsst_u`, which is the intended audit
  signal for the next AGN-normalization phase.
- Verification passed:
  `uv run python -m compileall scripts/benchmark_against_fsps_prospector.py`,
  `uv run ruff check scripts/benchmark_against_fsps_prospector.py
  tests/test_benchmark.py`, and `uv run pytest -q tests/test_benchmark.py`.
- User-run AGN audit benchmark analysed:
  `outputs/benchmarks/popcosmos_binned_agn_audit_n500/`.
  Report written to `outputs/report/popcosmos_binned_agn_audit_n500/report.md`
  with figures and CSV tables. The no-AGN levels remain benchmark-ready in
  bright finite bands, but `stellar_plus_agn` has a severe UV/blue tail and
  `full_agn` fails the readiness gates by a large margin.
- Current AGN verdict: keep `popcosmos_binned_noagn` as the prior-v0 candidate.
  Do not use the full AGN path for a scientific learned prior until a direct
  SED-level AGN normalization audit resolves the template-grid versus
  FSPS/Prospector `fagn`/`agn_tau` convention.

2026-05-28 proposed AGN SED-audit implementation plan:

- Do not tune `full_agn` photometry directly. First isolate the pure AGN
  component before dust, gas, IGM, and filters can hide the source of the
  mismatch.
- Add DSPS component outputs for AGN debugging: intrinsic stellar SED, dusted
  stellar SED, optional gas SED, AGN SED, pre-IGM SED, and post-IGM SED.
- Extend the AGN template generator to support signed finite-difference
  templates and grids over `agn_tau`, `fagn_normalization`, normalization age,
  and normalization metallicity. Keep the current single-age/single-metallicity
  template as the legacy approximate convention.
- Generate a new explicit audit asset whose filename and metadata state that it
  is an FSPS finite-difference AGN calibration grid, not yet a production
  PopCosmos AGN asset.
- Add an `agn_component_only` benchmark level comparing
  `FSPS(fagn, agn_tau) - FSPS(fagn=0)` against the DSPS AGN component before
  IGM and photometric integration.
- Add configurable AGN host attenuation experiments:
  `none` for the current behavior, plus at least one host-dust variant that
  applies the diffuse Prospector/FSPS-like dust curve to the AGN component.
- Rerun staged benchmarks in this order: tiny `agn_component_only`, `n=50`
  AGN SED audit, then `n=500` photometric AGN audit only if the component-level
  SED comparison is acceptable.

2026-05-28 AGN SED-audit implementation started:

- Scope: implement the proposed AGN component audit path, including DSPS
  component outputs, signed/multi-axis AGN finite-difference template support,
  an `agn_component_only` benchmark level, and configurable AGN host attenuation
  experiments.

2026-05-28 AGN SED-audit implementation completed:

- `JaxModelResult` now carries debug component SEDs:
  `stellar_intrinsic_sed`, `stellar_dusted_sed`, `gas_sed`, `agn_sed`,
  `pre_igm_sed`, and `post_igm_sed`.
- The PopCosmos binned and Diffstar forward paths now build the AGN component
  separately before IGM attenuation, so benchmark code can inspect it without
  going through filters.
- Added `model.agn_host_attenuation` with supported values `none`, `diffuse`,
  and `prospector_fsps`. The full AGN configs explicitly keep `none` for
  backward-compatible behavior until the audit chooses a physical convention.
- `scripts/generate_fsps_agn_grid.py` now supports audit grids over
  `fagn_grid`, `agn_tau_grid`, `tage_gyr_grid`, and `stellar_logzsol_grid`,
  with optional `--signed-delta`. The legacy 2D `agn_tau x wave` format still
  loads and validates.
- `scripts/benchmark_against_fsps_prospector.py` now accepts
  `--levels agn_component_only` and writes SED-ratio audit rows with
  `dsps_agn_lnu`, `reference_agn_lnu`, `delta_log10_lnu`, and equivalent
  `delta_mag = -2.5 log10(DSPS_AGN/FSPS_AGN)`.
- Added `--agn-template` to the benchmark CLI so AGN audit assets can be tested
  without editing the science config.
- Added `--agn-host-attenuation` to the benchmark CLI so the `none`, `diffuse`,
  and `prospector_fsps` AGN host-attenuation experiments can be compared without
  creating temporary YAML configs.
- Added component-only diagnostic plots under the benchmark `diagnostics/`
  directory, including SED ratio versus rest wavelength and point-level
  residual trends versus `ln_fagn`, `ln_tauagn`, `z_obs`, and `tau2`.
- Generated the first signed AGN audit asset:
  `Data/popcosmos_chabrier_agn_fspsdiff_audit_grid.h5`, shape
  `(5, 9, 4, 3, 11149)` for `fagn`, `agn_tau`, `tage_gyr`,
  `stellar_logzsol`, and wavelength. This is an audit asset, not yet the
  default science AGN template.
- Smoke `agn_component_only` run with the audit grid passed at
  `outputs/benchmarks/smoke_popcosmos_binned_agn_component_only_audit_grid/`.
  The `n=1` smoke has median absolute equivalent AGN-component residual
  `~0.33` mag and p95 `~5.26` mag over 64 sampled rest wavelengths, indicating
  that the component-level mismatch is now directly measurable and still needs
  science interpretation before enabling AGN prior training.
- User-run AGN audit results analysed:
  `outputs/benchmarks/popcosmos_binned_agn_component_only_audit_grid_n50`,
  `outputs/benchmarks/popcosmos_binned_agn_audit_grid_nohost_n500`, and
  `outputs/benchmarks/popcosmos_binned_agn_audit_grid_hostdust_n500`.
  Report written to
  `outputs/report/popcosmos_binned_agn_audit_grid_comparison_2026-05-28/report.md`.
- The signed 5D audit grid does not materially change AGN photometry in the
  no-host-dust convention: row-wise p95 absolute difference from the old 2D
  template is only `~0.017` mag. `full_agn` remains at bright p95 `~20` mag.
- `agn_host_attenuation: prospector_fsps` reduces the aggregate `full_agn`
  bright p95 from `~20.0` mag to `~10.4` mag, but shifts the failure mode from
  DSPS being too bright to DSPS often being too faint. This is not a final AGN
  convention.
- The `agn_component_only` `n=50` run shows that the component-level mismatch is
  wavelength dependent: median absolute equivalent residual is `~0.18` mag in
  rest `3000-10000 A`, but the p95 is `~7.1` mag globally and is dominated by
  far-UV/Lyman and IR wavelength regions.
- Updated verdict: the no-AGN path remains the only prior-v0 candidate. The AGN
  path now has the right audit instrumentation, but still needs a component
  convention/normalization fix before any scientific AGN prior training.
- Verification passed:
  `uv run python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py scripts/generate_fsps_agn_grid.py
  scripts/fsps_grid_common.py tests/test_model.py tests/test_benchmark.py
  tests/test_fsps_grid_scripts.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py tests/test_fsps_grid_scripts.py` with 88 passed and
  3 skipped; latest `uv run pytest -q` with 148 passed and 3 skipped.

2026-05-28 FSPS-native AGN component implementation completed:

- Added `scripts/generate_fsps_agn_component_grid.py`, which writes an
  SSP-shaped AGN component grid:
  `agn_lnu_per_mformed[fagn, agn_tau, Zstar, age, wave] =
  FSPS(fagn, agn_tau) - FSPS(fagn=0)`.
  Its default `fagn` and `agn_tau` axes cover the current PopCosmos AGN prior
  bounds, including `ln_fagn [-14, 1]` and `ln_tauagn [1.609438, 5.010635]`.
- Added `model.agn_model: fsps_component_grid` and
  `model.agn_component_grid_path`. This path does not use the legacy
  `fagn * Lbol * template` normalization; the AGN component is interpolated in
  `fagn`, `agn_tau`, and stellar metallicity, then summed over SSP age with the
  same PopCosmos/Diffstar SFH weights and formed mass as the stellar component.
- The AGN component grid loader validates the SSP wavelength, age, and
  metallicity axes against the active base SSP and keeps the strict PopCosmos
  Chabrier/z_sun metadata checks.
- `scripts/benchmark_against_fsps_prospector.py` now accepts
  `--agn-component-grid`, mutually exclusive with `--agn-template`, so the
  legacy template-grid audit and the FSPS-native component-grid audit can be
  compared without editing science YAML files.
- Added tests covering config validation, synthetic forward scaling with
  formed mass, benchmark level selection, and the component-grid generator
  using the fake FSPS test backend.
- Remaining AGN work is empirical: generate a production-size component grid,
  rerun `agn_component_only`, then rerun `stellar_plus_agn` and `full_agn`.
  If the component-level SED benchmark passes but photometric AGN levels still
  fail, the next suspect is host/torus attenuation order rather than AGN
  normalization.

2026-05-28 AGN component benchmark runtime fix:

- Fixed `scripts/benchmark_against_fsps_prospector.py` so
  `--levels agn_component_only` only loads the stellar-only DSPS context and no
  longer loads the gas-grid context.
- Added `--runtime config|auto|cpu|gpu`; use `--runtime cpu` for large
  component-grid audits that do not fit on the GPU.
- Updated runtime setup so a non-auto runtime platform forces
  `JAX_PLATFORMS`, while `auto` still clears stale platform requests.
- Smoke benchmark passed with the small AGN component grid:
  `outputs/benchmarks/smoke_popcosmos_binned_agn_component_grid_cpu_lazy_n1/`.

2026-05-28 full-AGN lazy component-grid loading implemented:

- `full_agn` and `stellar_plus_agn` benchmark levels with
  `model.agn_model: fsps_component_grid` no longer load the full 3.9 GB AGN
  component grid into JAX.
- The benchmark now loads the DSPS stellar/gas context with AGN disabled, then
  reads only the HDF5 slices needed for the sampled
  `(fagn, agn_tau, stellar metallicity)` point, interpolates those slices, and
  adds the AGN component before IGM and photometric integration.
- Axis interpolation is strict: sampled AGN values outside the component-grid
  axes now raise an explicit error instead of clipping silently.
- This is a benchmark-only memory fix. Production fitting with
  `fsps_component_grid` still needs a dedicated compressed/lazy model path
  before it is safe for large runs.

2026-05-29 AGN dust/gas isolation implementation started:

- Scope: add a tunable AGN host-attenuation scale and two benchmark levels that
  isolate dust+AGN and gas+AGN before attempting any `full_agn n=500` run.
- Motivation from `full_agn_hostdust_n50`: `agn_host_attenuation: none`
  under-attenuates the faint dusty tail, while `prospector_fsps` with full
  strength over-attenuates some high-redshift dusty points.

2026-05-29 AGN dust/gas isolation implementation completed:

- Added `model.agn_host_attenuation_scale`, defaulting to `1.0`, with
  non-negative config validation. The scale multiplies the host-dust optical
  depth applied to the AGN component, so `0.0` is equivalent to no host
  attenuation and `1.0` is the previous full-strength behavior.
- Added benchmark CLI option `--agn-host-attenuation-scale`.
- Added benchmark levels:
  `stellar_plus_dust_plus_agn` for pure-stellar+dust+AGN, and
  `stellar_plus_gas_plus_agn` for gas+AGN with dust disabled.
- Verification passed:
  `uv run python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py tests/test_model.py
  tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 87 passed and 3 skipped.

2026-05-29 FSPS AGN diffuse attenuation alignment started:

- Scope: replace the diagnostic interpretation of
  `agn_host_attenuation_scale` with a fixed FSPS-native AGN host attenuation
  mode. Local FSPS source inspection shows `agn_dust.f90` applies
  `exp(-attn_curve(...))` to the AGN dust template, without multiplying by
  `dust2`. The existing scaled modes remain audit diagnostics, but should not
  be used as learned science parameters.

2026-05-29 FSPS AGN diffuse attenuation alignment completed:

- Added `model.agn_host_attenuation: fsps_diffuse_unit_tau`. This mode applies
  the JAX Prospector/FSPS diffuse attenuation shape directly at unit optical
  depth and is intentionally independent of fitted `tau2`.
- Config validation now rejects `agn_host_attenuation_scale != 1.0` with
  `fsps_diffuse_unit_tau`, so the exact FSPS convention cannot be silently
  turned into a learned or tuned scale.
- The full AGN configs now use `agn_host_attenuation: fsps_diffuse_unit_tau`.
  They remain non-recommended for prior v0 until the component-grid AGN
  benchmark passes.
- Benchmark isolation keeps `stellar_plus_agn` and `stellar_plus_gas_plus_agn`
  explicitly host-dust-free; the unit-tau FSPS AGN attenuation is only used for
  dust+AGN and full-AGN levels.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py tests/test_model.py
  tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 89 passed and 3 skipped.

2026-05-29 FSPS AGN/IGM ordering alignment started:

- Scope: add an explicit `model.agn_igm_order` convention. The existing DSPS
  behavior applies IGM after adding AGN (`pre_igm`). Local FSPS source shows
  `compsp.f90` applies IGM before `agn_dust`, so the benchmark needs an
  `fsps_after_igm` mode where stellar/gas light is IGM-attenuated before the
  AGN component is added.

2026-05-29 FSPS AGN/IGM ordering alignment completed:

- Added `model.agn_igm_order` with supported values `pre_igm` and
  `fsps_after_igm`. The default remains `pre_igm` for backward compatibility.
- Full AGN PopCosmos-like configs now set `agn_igm_order: fsps_after_igm`.
- `combine_agn_and_igm_jax` centralizes the ordering in the production forward
  model. In `fsps_after_igm`, stellar/gas light is attenuated by IGM first and
  the already host-attenuated AGN component is added afterward.
- The lazy AGN component-grid benchmark path now uses the same helper, so
  `fsps_component_grid` audits test the same ordering as production.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py tests/test_model.py
  tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 92 passed and 3 skipped.

2026-05-29 FSPS AGN baked-attenuation replacement completed:

- Analysis of the after-IGM benchmark showed `stellar_plus_agn` is clean in the
  bright regime, while `stellar_plus_dust_plus_agn` and `full_agn` remain too
  faint in dusty blue bands. The active AGN component/template assets were
  generated through FSPS `agn_dust` with `dust_type=0`, so they already include
  a unit-tau power-law AGN attenuation before runtime applies the
  `fsps_diffuse_unit_tau` target curve.
- Added `model.agn_baked_attenuation` with supported values `none` and
  `fsps_powerlaw_unit_tau`, plus `model.agn_baked_dust_index` defaulting to
  `-0.7`, the FSPS default from `sps_vars.f90`.
- In `agn_host_attenuation: fsps_diffuse_unit_tau`, DSPS now replaces baked
  power-law attenuation by applying
  `exp(-(tau_fsps_dust_type4 - tau_baked_dust_type0))`. This avoids stacking
  the baked FSPS `dust_type=0` attenuation and the target `dust_type=4`
  attenuation.
- The full AGN PopCosmos-like configs now declare
  `agn_baked_attenuation: fsps_powerlaw_unit_tau` and
  `agn_baked_dust_index: -0.7`.
- AGN grid generators now write these baked-attenuation metadata keys for newly
  generated template/component assets.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run ruff check euclid_dsps/model.py euclid_dsps/config.py
  scripts/benchmark_against_fsps_prospector.py
  scripts/generate_fsps_agn_component_grid.py scripts/generate_fsps_agn_grid.py
  tests/test_model.py tests/test_config.py tests/test_benchmark.py`;
  `uv run pytest -q tests/test_config.py tests/test_model.py
  tests/test_benchmark.py` with 94 passed and 3 skipped.

2026-05-29 FSPS AGN baked-attenuation benchmark analysed:

- User-run benchmarks completed:
  `outputs/benchmarks/popcosmos_binned_agn_afterigm_replace_baked_n50` and
  `outputs/benchmarks/popcosmos_binned_agn_afterigm_replace_baked_n500`.
- The replacement fix closes the previous dusty blue-band failure. At `n=500`,
  `full_agn` has median p95 `0.0123` mag and max p95 `0.0371` mag over the
  finite/effectively-bright summary, while `stellar_plus_dust_plus_agn` has
  median p95 `0.0114` mag and max p95 `0.0316` mag. Both satisfy the broad-band
  prior-readiness target for finite observable rows.
- Compared to the previous `fsps_unit_tau_afterigm_n500`, the `full_agn`
  `lsst_u` p95 drops from `~5.0` mag to below the global max of `0.0371` mag,
  confirming the failure was stacked baked `dust_type=0` plus target
  `dust_type=4` AGN attenuation.
- `stellar_plus_agn` still has a very faint `lsst_u` tail when all finite rows
  are included, but it is clean in the observable regime (`mag<35` max p95
  `0.0255` mag). This is a diagnostic faint-UV tail rather than the blocker for
  the full dusty AGN model.

2026-05-29 full forward FSPS/Prospector closure benchmark analysed:

- User-run benchmark completed:
  `outputs/benchmarks/popcosmos_binned_full_forward_fsps_closure_n500`.
  It covers 500 sampled points, 8 benchmark levels, and 10 bands, including
  `stellar_only`, `stellar_plus_dust`, `stellar_plus_gas`, `full_noagn`,
  `stellar_plus_agn`, `stellar_plus_dust_plus_agn`,
  `stellar_plus_gas_plus_agn`, and `full_agn`.
- Report written to
  `outputs/report/popcosmos_binned_full_forward_fsps_closure_n500/report.md`,
  with summary tables, worst residuals, threshold sensitivity, correlations,
  heatmaps, CDFs, and trend plots.
- Production broad-band levels now pass the configured bright finite
  FSPS/Prospector closure target. `full_noagn` has median band p95
  `0.0129` mag and max band p95 `0.0166` mag. `full_agn` has median band p95
  `0.0123` mag and max band p95 `0.0371` mag.
- `stellar_only`, `stellar_plus_dust`, `stellar_plus_gas`, and
  `stellar_plus_dust_plus_agn` also pass the bright finite target. Remaining
  non-finite rows in dusty/full levels are explicit magnitude-space
  zero-flux/effectively-faint cases and should be handled in flux space for
  inference.
- The only large residual tails are diagnostic component levels with dust
  disabled: `stellar_plus_agn` and `stellar_plus_gas_plus_agn`, mainly in
  `lsst_u/g` at very faint or high-redshift UV points. These remain documented
  component-isolation limitations, not a production `full_agn` blocker.
- Current verdict: the DSPS forward model is now FSPS/Prospector-like for
  broad-band prior training. It is still not an official PopCosmos reproduction
  because PopCosmos learned emission-line correction tables are not included;
  the gas path remains raw FSPS/CLOUDY with
  `emission_line_corrections: none`.

2026-05-29 documentation and default-config cleanup started:

- Promote the validated full AGN path to the default documented setup:
  `configs/popcosmos_binned.yaml` first, `configs/popcosmos_diffstar.yaml` as
  the comparison, with no-AGN configs retained only for fallback/ablation.
- Update the full AGN configs to use `agn_model: fsps_component_grid` and
  `Data/popcosmos_chabrier_agn_component_ssp_grid.h5`, matching the benchmark
  that closed against FSPS/Prospector.
- Rewrite README and Sphinx docs so the SED pipeline is understandable from
  SSP generation to photometry, including active assets, assumptions,
  implementation files, benchmark commands, and remaining scientific caveats.
- Reduce `docs/source/science_assessment.rst` to the current status and caveats
  rather than retaining stale historical AGN/no-AGN conclusions.
- Remove unused legacy/audit HDF5 files from `Data/`, while keeping
  `ssp_data_fsps_v3.2_lgmet_age.h5` because a legacy smoke test still
  references it.

2026-05-29 documentation and default-config cleanup completed:

- `README.md` now presents the validated full AGN path as the default, documents
  active assets, generation commands, fit commands, and the FSPS/Prospector
  closure benchmark command.
- Added `docs/source/forward_model.rst` as the step-by-step SED pipeline:
  HDF5 SSP axes, SFH weights, surviving mass, dust, gas, AGN component grid,
  IGM/order, photometry, implementation files, assumptions, and benchmark
  status.
- Rewrote `docs/source/data_download.rst`, `docs/source/run_setup.rst`, and
  `docs/source/science_assessment.rst` around the current full AGN model.
  `science_assessment.rst` is now intentionally short and only records current
  status, validated scope, and remaining caveats.
- Updated `docs/source/architecture.rst`, `catalog_columns.rst`,
  `installation.rst`, `testing.rst`, and `index.rst` to match the active assets
  and commands.
- Updated `configs/popcosmos_binned.yaml` and `configs/popcosmos_diffstar.yaml`
  to use `agn_model: fsps_component_grid`,
  `agn_component_grid_path: Data/popcosmos_chabrier_agn_component_ssp_grid.h5`,
  `stellar_only_ssp_path`, and Student-t photometric likelihood by default.
- Removed unused legacy/audit assets from `Data/`: old Kroupa SSPs, old gas
  grids, legacy AGN template grids, the AGN smoke/audit grids, and Windows zone
  identifier sidecar files. Retained the active Chabrier assets and
  `ssp_data_fsps_v3.2_lgmet_age.h5`.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `pytest tests/test_config.py tests/test_model.py tests/test_benchmark.py
  tests/test_fsps_grid_scripts.py` with 104 passed and 3 skipped;
  `uv run python -m sphinx -W --keep-going -b html docs/source
  docs/build/html`;
  `git diff --check`.

2026-05-29 didactic documentation pass started:

- Add short explanations of the main acronyms and model ingredients:
  SSP, SED, SFH, IMF, isochrones, C3K, IGM, CLOUDY, emission lines, AGN,
  FSPS, Prospector, and DSPS.
- Make the distinction between full AGN and no-AGN explicit in README,
  `forward_model.rst`, and run setup.
- Remove the Sphinx `SSP Compression Notes` page for now because compressed SSP
  execution is not implemented in the production model.

2026-05-29 didactic documentation pass completed:

- `docs/source/forward_model.rst` now includes a concise glossary for SED, SSP,
  SFH, IMF, isochrones, C3K, IGM, CLOUDY, emission lines, AGN, FSPS,
  Prospector, and DSPS. README links to this page rather than duplicating the
  glossary.
- `docs/source/forward_model.rst` now explains the same concepts before the
  step-by-step SED pipeline and records why the current FSPS/MIST/C3K/Chabrier
  choices are used.
- `docs/source/run_setup.rst` now defines full AGN versus no-AGN before listing
  commands.
- Removed `docs/source/ssp_compression.rst` from source and from the Sphinx
  toctree. Compression remains only a historical plan item, not user-facing
  production documentation.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `uv run python -m sphinx -W --keep-going -b html docs/source
  docs/build/html`.

2026-05-28 fit-regression diagnosis started:

- Scope: compare the user-provided 512-galaxy runs
  `popcosmos_binned_chi2_512_b10`,
  `popcosmos_binned_student_t_512_b10`, and
  `popcosmos_binned_noagn_chabrier_student_t_512_b10` to explain why redshift
  recovery degrades and why some inferred parameters are flat across galaxies.
- Initial hypothesis to test from the generated reports: the fit is not simply
  optimizer noise; likely contributors are changed likelihood/physics
  contracts, forward-model mismatch in blue bands, redshift attractors, and
  poorly constrained or inactive parameters that sit at initialization/bounds.

2026-05-28 Reveal.js recap deck started:

- Scope: create a didactic French slide deck summarizing the code delta from
  `master`/`dev`/the current working tree, PopCosmos-like inference pre-work,
  new SSP generation, DSPS photometry flow, FSPS/Prospector benchmark results,
  and the next benchmark tests to implement.
- Target output:
  `outputs/reports/ssp_generation_assessment_2026-05-28/slides_popcosmos_ssp_benchmark_reveal.html`.
- The deck should reuse the generated report figures and cite the papers behind
  the main scientific choices.
- Completed the deck at the target path. It uses local figures from the SSP,
  Student-t, `n=500`, and noNE smoke benchmark reports, plus linked paper
  references for FSPS, Chabrier, MIST, DSPS, Prospector, dust, IGM, Diffstar,
  CLUMPY, and PopCosmos.
- Verified the HTML parses and that all seven local image paths resolve.

2026-05-28 PopCosmos benchmark-closure implementation started:

- Scope: implement the first next-step slice from the `n=500` benchmark review.
- Added `scripts/generate_fsps_ssp_grid.py --stellar-only`, which writes the
  pure-stellar Chabrier no-nebular-emission asset
  `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5`.
- Generated and validated `Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5` in the
  `shine` conda environment. Metadata confirms `imf_type=1`,
  `imf_name=chabrier`, `z_sun=0.0142`, `add_neb_emission=0`, and
  `add_neb_continuum=0`.
- Added `model.stellar_only_ssp_path` to the no-AGN configs. The benchmark also
  accepts `--stellar-ssp` to override this path.
- Updated `scripts/benchmark_against_fsps_prospector.py` so
  `stellar_only` and `stellar_plus_dust` use the pure-stellar DSPS context,
  while `stellar_plus_gas` and `full_noagn` use the Chabrier gas-grid context.
- Tightened PopCosmos-like IMF validation so contradictory HDF5 metadata now
  fails instead of accepting `imf_type=1` or `imf_name=chabrier` independently.
- Extended benchmark summaries with `n_total`, `n_finite_both`,
  `n_nonfinite_dsps`, `n_nonfinite_reference`, and `n_nonfinite_delta` per
  `(level, band)`, plus per-level/per-band residual correlations including a
  recent-SFR proxy.
- Added per-level diagnostic plots under `diagnostics/` for residuals against
  redshift, dust, metallicity, gas, and recent SFR.
- Ran a smoke benchmark:
  `outputs/benchmarks/smoke_popcosmos_binned_noagn_noNE/`. It completes and
  writes `benchmark_points.csv`, `benchmark_summary.json`,
  `delta_mag_by_band.png`, and per-level diagnostics. Residuals remain too
  large for science use, so the next step is baseline debugging, not prior
  training.
- Fixed the standalone benchmark runtime setup so it applies the config
  `runtime` block before importing `euclid_dsps.model`. This lets
  `runtime: auto` clear a stale shell-level `JAX_PLATFORMS=cuda` request before
  JAX is imported. Verified that `--help` works with `JAX_PLATFORMS=cuda`, that
  a CPU-forced `shine` smoke benchmark with `--n 1` completes at
  `outputs/benchmarks/smoke_popcosmos_binned_noagn_noNE_cpu_check/`, and that
  `runtime: auto` clears a shell-level `JAX_PLATFORMS=cuda` for the `--n 1`
  smoke run at
  `outputs/benchmarks/smoke_popcosmos_binned_noagn_noNE_auto_runtime_check/`.
- Generated the comparison report for the old fixed-nebular benchmark versus
  the new pure-stellar noNE benchmark at
  `outputs/report/popcosmos_binned_noagn_fsps_noNE_n500/report.md`. The
  comparison confirms that `stellar_only` median residuals improve from
  `0.0718` to `0.0531` mag and `stellar_plus_dust` non-finite rows drop from
  199 to 192, but all benchmark levels still fail the prior-readiness gates.
  Next priority is a pure-stellar normalization/UV/IGM audit before changing
  gas or training the first learned prior.

2026-05-28 FSPS/Prospector `n=500` benchmark available:

- Completed benchmark directory:
  `outputs/benchmarks/popcosmos_binned_noagn_fsps_n500/`.
- Outputs present: `benchmark_points.csv`, `benchmark_summary.json`,
  `delta_mag_by_band.png`, and `run.log`.
- The benchmark has 20,000 rows: 500 sampled points x 4 levels x 10 bands.
- It runs end to end, but residuals do not meet the recommended prior-v0
  criteria. `stellar_only` already has broad residuals larger than target,
  especially in LSST `u/g/r` and Euclid VIS, so the next priority is a clean
  pure-stellar baseline before tuning dust or gas.
- For dust/gas levels, some DSPS magnitudes are non-finite in the current
  benchmark output while the FSPS/Prospector reference remains finite. The next
  benchmark summary should report finite and non-finite counts explicitly per
  `(level, band)`.
- Updated remaining-work plan to prioritize pure-stellar SSP generation,
  stricter IMF metadata validation, richer benchmark diagnostics, staged reruns,
  and then dust/gas decisions.

2026-05-28 SSP generation assessment report completed:

- Wrote `outputs/reports/ssp_generation_assessment_2026-05-28/report.md`.
- Generated companion diagnostics in the same directory:
  `ssp_axis_contract.png`, `ssp_sed_flux_ratios.png`,
  `ssp_broadband_offsets.png`, `gas_grid_reference_match.png`,
  `fsps_prospector_benchmark_residuals.png`, plus CSV/JSON summary tables.
- Findings: the new Chabrier base SSP is DSPS-layout compatible with the legacy
  Kroupa `logGasU=-2` asset; the older DSPS v3.2 SSP keeps the same age and
  stellar-metallicity axes but uses a different wavelength grid; the Chabrier
  and Kroupa spectra are not flux-identical, as expected from the IMF change.
- The Chabrier gas grid exactly reproduces the base SSP at
  `gas_logu=-2`, `gas_logz=0` in the local HDF5 assets.
- The existing `n=500` FSPS/Prospector benchmark still misses the desired
  photometry residual targets, so the report recommends a pure-stellar
  Chabrier SSP and a cleaner layered benchmark before calling the forward model
  science-ready.

2026-05-28 SSP generation assessment report started:

- Scope: write a didactic Markdown report under `outputs/reports/` explaining
  how the new locally generated SSP assets are produced, whether they match the
  previous SSP assets closely enough for the current PopCosmos-like workflow,
  and how DSPS turns physical parameters plus SSPs into broadband photometry.
- Planned checks: inspect `scripts/generate_fsps_ssp_grid.py`,
  `scripts/fsps_grid_common.py`, the active config asset paths, and the HDF5
  schemas/statistics for the Chabrier, legacy Kroupa, and old DSPS SSP grids;
  generate compact comparison plots for the report.

2026-05-27 PopCosmos forward-model gap correction completed:

- Added recommended no-AGN configs:
  `configs/popcosmos_binned_noagn.yaml` and
  `configs/popcosmos_diffstar_noagn.yaml`. They use
  `model.agn_model: none`, Student-t photometric likelihood, and do not expose
  `ln_fagn` or `ln_tauagn` as free parameters.
- Updated PopCosmos-like configs and FSPS generation scripts to explicit
  Chabrier asset names and metadata. PopCosmos-like `load_context` now rejects
  Kroupa-named assets, missing/ambiguous IMF metadata, and `z_sun` mismatches.
- PopCosmos-like `z_sun` is now `0.0142`; legacy lognormal defaults remain
  compatible with `0.0134`.
- Added `dust_model: charlot_fall_powerlaw` as the explicit name for the
  previous behavior and `dust_model: prospector_fsps` as the current approximate
  JAX target mode. Direct FSPS/Prospector benchmarking is still required before
  calling this exact.
- Added the hard `Zgas >= Zstar` constraint, strict gas-grid axis/unit
  validation, optional enriched-grid emission-line correction support, and tests
  for the correction affecting only the line-covered filter.
- Added `scripts/benchmark_against_fsps_prospector.py`. It defines the CLI,
  sampling, summary/output contract, and uses an independent
  `prospect.sources.FastStepBasis` + python-FSPS + sedpy reference. It refuses
  unsupported mappings rather than falling back to DSPS.
- Smoke benchmark with `--n 5` now runs and writes
  `outputs/benchmarks/smoke_popcosmos_binned_noagn/benchmark_points.csv`,
  `benchmark_summary.json`, and `delta_mag_by_band.png`. Current residuals do
  not meet the recommended prior-readiness criteria; the largest UV/dust/gas
  discrepancies should be audited before any final learned prior.
- Verification passed:
  `python -m compileall euclid_dsps scripts`;
  `pytest tests/test_config.py` (28 passed);
  `pytest tests/test_model.py` (33 passed, 3 skipped);
  `pytest tests/test_benchmark.py tests/test_fsps_grid_scripts.py` (6 passed);
  full `pytest` (123 passed, 4 skipped);
  `git diff --check`.

2026-05-27 PopCosmos forward-model gap correction phase started:

- Scope: remove avoidable PopCosmos-like mismatches before learning a first
  prior, without touching the already implemented manual Student-t likelihood.
- `configs/fs2_phz1_science.yaml` is not present in this checkout; active
  inputs are the binned and Diffstar PopCosmos-like configs.
- Current PopCosmos-like assets/configs still reference Kroupa-named SSP data
  and gas metadata with `imf_type=2`; this phase will switch PopCosmos-like
  generation/config contracts to explicit Chabrier assets and make ambiguous
  HDF5 metadata fail loudly.
- Planned outputs: no-AGN configs for prior v0, PopCosmos-like `z_sun=0.0142`,
  explicit dust modes, gas metallicity and grid-axis validation, optional line
  correction support, and an FSPS/Prospector benchmark harness.

2026-05-27 group meeting report drafting:

- Preparing a presentation-ready scientific report summarizing the `main` to
  `dev` changes, the two SFH fitting paths, the PopCosmos-like configuration,
  the cleaner SSP/gas/AGN generation, and the two comparison runs in
  `outputs/runs/`.
- Wrote `outputs/reports/group_meeting_popcosmos_dev_2026-05-28.md` with:
  codebase changes from `master`/`main` to `dev`, scientific justification,
  gas/AGN asset details, chi-square vs Student-t run comparison, caveats,
  suggested figures, next steps, and a short meeting talk track.
- Report uses the local `master` branch as the base because no local `main`
  branch exists in this checkout.

2026-05-27 likelihood comparison report in progress:

- Comparing `outputs/runs/popcosmos_binned_chi2_512_b10` against
  `outputs/runs/popcosmos_binned_student_t_512_b10`.
- Goal: supervisor-ready report with paired Gaussian vs Student-t diagnostics,
  fit quality, redshift truth/proxy residuals, band residuals, and runtime
  comparison.
- Added `scripts/compare_likelihood_runs.py` to generate paired comparison
  reports.
- Generated `outputs/reports/popcosmos_binned_chi2_vs_student_t_512/`.
- Paired 512 shared galaxies. Student-t improves median reduced Gaussian chi2
  from 15.76 to 1.56, median |dz| from 0.291 to 0.120, median |dz|/(1+z) from
  0.150 to 0.060, and median mean absolute magnitude residual from 0.136 to
  0.049.
- Galaxy-by-galaxy tables are written, including top redshift improvements and
  degradations for Student-t relative to Gaussian chi2.
- `uv run python -m compileall scripts/compare_likelihood_runs.py` passed.
- `uv run --extra dev ruff check scripts/compare_likelihood_runs.py` passed.

2026-06-10 Diffsky simple recovery reset:

- Decision: stop using the Diffsky/Diffstar generated latents as first-order
  MAP recovery targets from broad-band photometry. They remain useful
  generated truths for later population diagnostics, but not for the first
  differentiable DSPS closure.
- Added a recommended simple HLTDS 04/14 path based on `popcosmos_bins`, HLTDS
  SSP/filter assets, native AB magnitudes, and explicit model-tolerance
  magnitudes rather than synthetic flux errors.
- New configs:
  `configs/diffsky_hltds_04_14_simple.yaml`,
  `configs/diffsky_hltds_04_14_simple_gpu.yaml`,
  `configs/diffsky_hltds_04_14_fixedz_closure.yaml`, and
  `configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml`.
- `diffsky-prepare-dataset` now supports `--no-synthetic-errors`; the simple
  fit configs do not consume `fluxerr_*` and instead use explicit
  `sigma_mag` model tolerance.
- Fit targets are restricted to basic direct truths that DSPS can plausibly
  recover: redshift, stellar mass, and recent SFR proxy. Halo mass, centrality,
  and sizes are retained as context columns but are not DSPS photometric fit
  targets.
- Remote/data investigation: HLTDS 04/14 is still the best current science
  target; HLTDS 03/31 is a high-z slice with the same schema; `sparse_cosmos`
  is useful for fast debugging but has a very coarse SSP grid; `smdpl` is too
  small locally; `lsstdesc_diffsky_data` is testdata, not a population dataset.
- Smoke diagnostics: fixed-z simple fit on 8 HLTDS objects gives median
  residual 0.073 mag and reduced chi2 ~1.25 with 0.10 mag model tolerance.
  The same setup with free redshift still has poor z recovery, so the next
  blocker is redshift initialization/multi-start/photo-z strategy, not
  Diffstar latent fitting.

2026-05-27 Student-t likelihood switch:

- Confirmed from Thorp et al., *Scaleable inference of galaxy properties and
  redshifts with a data-driven population model*, section 2.2.1, that the
  POP-COSMOS photometric likelihood is a per-band flux Student-t with 2 degrees
  of freedom.
- Added `fit.photometric_likelihood` with `gaussian` and `student_t` modes,
  defaulting to Gaussian chi-square for backward compatibility.
- Added `fit.student_t_dof`, default `2.0`, and CLI override
  `--fit-likelihood student_t`.
- Added mode-aware `fit_quality`/`reduced_fit_quality` diagnostics that follow
  the configured photometric likelihood. `chi2`/`reduced_chi2` remain Gaussian
  comparison metrics at the final parameters.
- Student-t is wired through independent MAP, population MAP, and NumPyro
  posterior sampling.
- Batch dashboards, objective components, redshift attractor summaries, workflow
  MAP-vs-population plots, and SED sample ranking now prefer mode-aware fit
  quality over Gaussian chi-square when available.

2026-05-27 verification:

- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run --extra dev ruff check euclid_dsps tests` passed.
- `uv run pytest` passed: 106 passed, 2 skipped.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --help` passed and exposes `--fit-likelihood {gaussian,student_t}`.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml fit
  --help` passed and exposes `--fit-likelihood {gaussian,student_t}`.
- Student-t one-row smoke passed:
  `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --index 0 --fit-maxiter 1 --fit-likelihood student_t --out
  outputs/runs/dev_popcosmos_student_t_one_iter --sed-samples 0
  --reporting-level light`.
- Student-t batch smoke passed:
  `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --limit 1 --batch-size 1 --fit-maxiter 1 --fit-likelihood student_t --out
  outputs/runs/dev_popcosmos_student_t_batch1_quality2 --sed-samples 0
  --reporting-level light`.
- Added the `gpu` optional dependency extra for the `uv` environment using the
  official JAX CUDA wheel path (`jax[cuda12]`).
- `uv sync --extra gpu` installed the CUDA JAX stack in `.venv`.
- `uv run --extra gpu python -c "import jax; print(jax.devices());
  print(jax.default_backend())"` reports `[CudaDevice(id=0)]` and `gpu`.

2026-05-26 combined Diffstar branch:

- Created `feature/pop-cosmos-diffstar` from `feature/pop-cosmos`.
- Source Diffstar commit: `497895a Add Diffstar PopCosmos model path` on
  `feature/diffstar`.
- Ported the Diffstar SFH path while keeping the current production FSPS
  gas/AGN grid scripts, `gas_grid_path`/`agn_template_path` config contract,
  dynamic JAX context arguments, and four-corner gas-grid interpolation.
- Added `configs/popcosmos_diffstar.yaml` as the combined gas + AGN + Diffstar
  configuration. `configs/` now contains only:
  `configs/popcosmos_binned.yaml` and `configs/popcosmos_diffstar.yaml`.
- Added the optional packaging extra `diffstar` for `diffstar` and `diffmah`.
- Updated tests and docs to describe both production configs and the Diffstar
  caveat that halo assembly currently uses Diffmah defaults until catalog MAH
  inputs are wired.

2026-05-26 verification:

- `git diff --check` passed.
- `uv run python -m compileall euclid_dsps scripts` passed.
- `uv run pytest` passed after installing the optional Diffstar extra: 103
  passed.
- `uv run --extra diffstar pytest tests/test_model.py -k diffstar` passed: 2
  passed, 17 deselected.
- `uv run --extra dev ruff check euclid_dsps scripts tests` passed.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_binned.yaml fit
  --help` passed.
- `uv run python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml fit
  --help` passed.
- `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra
  diffstar python -m euclid_dsps.cli --config configs/popcosmos_diffstar.yaml
  fit --index 0 --fit-maxiter 1 --out
  outputs/runs/dev_popcosmos_diffstar_one_short --sed-samples 0` passed and
  wrote the expected diagnostic outputs.
2026-07-17 conditional posterior experiment matrix (completed):
- Implemented one H100 array comparing stochastic-ELBO and supervised NPE
  objectives with the current Gaussian encoder, a prior-transported Gaussian,
  a conditional RealNVP posterior, and a conditional RQ-spline posterior.
- Added exact conditional-posterior sampling and density evaluation, supervised
  truth loading, NPE loss/checkpoint selection, and shared inference support for
  all four posterior families while keeping the frozen population prior fixed.
- Each full array task trains, runs 5,000-object held-out inference, verifies the
  photometric-fit and truth/prior/posterior corner plots, and writes coverage,
  accuracy, timing, and posterior-predictive metrics. A ten-minute smoke array
  gates the single full array through an `afterok` dependency.
- Added an Agg Matplotlib setup with a private per-job cache, strict four-device
  checks, non-overwrite guards, per-task completion markers, locked aggregation,
  comparison plots, a selection report, experiment configs, and a runbook.
- Verification: `compileall`, shell syntax checks, Ruff, config/model construction,
  exact RealNVP/RQ-spline roundtrips, finite NPE gradients, and 39 focused tests
  passed. The complete suite reached 350 passed and 1 skipped; its sole failure
  is the unrelated pre-existing tiny-sample metallicity-trend gate in
  `test_toy_smoke_generation_validation_and_parquet_roundtrip`. Sphinx renders
  the new page; strict docs still reports eight pre-existing RST errors in
  `docs/source/spline15d_realnvp.rst`.

2026-07-18 conditional posterior result audit and AVI recovery (completed):
- Audited the completed frozen-RQSpline/Gaussian-encoder run, the eight smoke
  tasks, the four completed NPE tasks, and the four interrupted AVI tasks.
- Identified a common validation-time JAX pinned-host-memory allocation failure
  near epoch 91 in every AVI task; the training updates themselves remained
  finite and family-specific numerical failure was not the cause.
- Added a guarded four-task AVI warm-restart array using the last safe model
  checkpoints, validation every four epochs, disabled JAX preallocation, and a
  fresh comparison root that reuses the completed NPE results.
- Corrected resumed-run timing summaries to divide elapsed time by epochs
  actually executed and suppressed expected all-NaN redshift-reference warnings.
- Verification: shell syntax, submit-contract mock, `compileall`, two focused
  timing tests, Ruff, and `git diff --check` passed. Tests importing the full
  spline15d stack could not collect locally because `jax_cosmo` is absent from
  both available local Python environments; it remains present in the Jean-Zay
  `shine` environment used by the production runs.

2026-07-18 independent posterior and learned-prior matrix (completed):
- Added a direct-latent conditional RealNVP posterior so `q_phi(x|y)` and the
  unconditional population flow `p_psi(x)` have independent transformations
  and exact, separately evaluated densities.
- Added simultaneous joint AVI and prior-only variational-EM schedules. VEM
  samples are stopped before the prior NLL, skip the DSPS decoder, and strict
  component restoration prevents AdamW momentum or weight decay from moving
  parameters during their frozen phase.
- Added six H100 experiments: frozen pretrained RQ-spline control, simultaneous
  RealNVP prior, VEM 1:1, VEM 4:1, VEM 4:1 plus NPE, and a synthetic-only prior
  truth oracle. The NPE weight is 50 based on the measured AVI/NPE encoder
  gradient-norm ratio, and every VEM configuration retains 120 encoder epochs.
- Added a ten-minute smoke array followed by one full six-task array. Each task
  trains, infers 5,000 held-out objects, requires the photometry-fit and corner
  plots, and participates in a locked aggregate comparison.
- Hardened the old epoch-91 failure path with spaced validation, disabled JAX
  preallocation, Agg Matplotlib, a private cache, and exact four-GPU checks.

2026-07-19 learned-prior smoke failure recovery (completed):
- Audited all six tasks from smoke array 2067913. The frozen-prior control
  completed end to end; tasks 1-5 failed at their first pmapped batch with the
  same lazy-import `UnexpectedTracerError`.
- Removed the module-level JAX spline-node constant that became a leaked tracer
  when `prior_learning.spline15d` was first imported inside `pmap`. Default
  nodes are now converted from the NumPy constant inside the traced function.
- Added a JIT regression test for default spline-node reconstruction. The fix
  applies to the model path itself rather than relying on eager imports in the
  Slurm wrapper.
# 2026-08-22 Adaptive-SMC mixing and prior-gradient completion

## Current scientific gate

- Audited HEAD: `2a5eeba13d15c8ae286ce1de836f48f72c22a955`.
- The latest Jean-Zay smoke correctly failed closed: training batches reached
  only median `beta_final=0.235-0.347`, hard fractions were `0.656-0.812`, and
  prior macro-updates were rejected because the selection gradient was NaN
  while the data and trust gradients were finite.
- Keep the canonical target, defensive `r0`, conditional RealNVP, broad
  identity RealNVP parent prior, standardized-logit no-truth coordinates,
  Student-t2 likelihood, Gaussian-m5 selection correction, and K64/K128
  primary/fallback budgets unchanged.
- The big run remains blocked until a new immutable smoke passes every
  scientific and numerical gate.

## Implementation phase

- Add genealogical ancestor ESS and a combined mixing failure contract.
- Adapt the per-object RW scale only between bridge stages.
- Propagate mixing diagnostics through fallback, training, validation, exact
  benchmarking, receipts, and the hard-object queue.
- Record a common-random-number q baseline immediately after sleep bootstrap
  and compare it to the post-SMC-distillation validation.
- Keep prior updates fail-closed, stabilize invalid selection draws, and add a
  score-function gradient diagnostic without enabling it in production.
- Rename the mixed-likelihood selection score so it is not presented as an
  exact marginal evidence.
- Run targeted and full local tests, update this plan, then commit and push.

## Completed local gate

- Implemented ancestor ESS, the combined per-object mixing rule, and fixed-scale
  within-stage RW-MH adaptation without changing the bridge target or budgets.
- Added post-bootstrap common-random-number validation and fail-closed q/prior
  receipt checks. Final ESS remains diagnostic only.
- Stabilized the Gaussian-m5 selection gradient at physical CGS scales and
  added a score-function comparison helper that is diagnostic-only.
- Targeted Adaptive-SMC/selection tests: `39 passed`.
- Full repository suite: `631 passed, 8 skipped` (`8` non-failing warnings).
- `ruff check` and `python -m compileall euclid_dsps scripts tests` pass.
- No Jean-Zay job and no big run were submitted. The next permissible action
  is one new immutable scientific smoke; the big run remains fail-closed on
  its receipt.
