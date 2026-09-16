# FENIKS: diagnostic reset before another VI run

Date: 2026-09-08. Audited source HEAD:
`1f5462fbae6a90f4db7bbf7b606997f4227fd6d1`.
Status: analysis and proposed protocol, NOT an implemented or submitted runner.

## Decision and scope

Do not launch another sleep/ELBO-weight/architecture sweep. First distinguish
software inconsistency, optimization failure, amortization error, approximation
error, and population-model mismatch. These need different interventions.
Success of a diagnostic phase is an actionable distinction, not necessarily a
promotable posterior. Do not promise a production posterior on a fixed date.

Keep DSPS, filters, noise, selection, latent conventions and the parent fixed
while testing inference. No catalogue-truth columns, truth-derived bounds,
point-estimate posterior substitutes, MCMC, HMC, NUTS, Langevin, SMC, AIS or
nested sampling. Generated simulation parameters and exact analytical examples
are allowed. Optimizer iterates are not posterior samples: retain only direct
draws from normalized variational densities.

## Evidence and limitations

Balanced-v2 K256 results were supplied in the conversation, not independently
read from the cluster. All four arms use 256 objects and 256 draws:

| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Median raw ESS | 4.550 | 3.913 | 4.016 | 4.068 |
| Fraction k>0.7 or nonfinite | .887 | .945 | .938 | .918 |
| P90 maximum raw weight | .994 | 1.000 | .999 | .995 |
| Median absolute photometric residual | 1.857 | 2.152 | 2.027 | 1.797 |
| RMS photometric residual | 17.500 | 20.732 | 13.295 | 8.643 |

D improves primarily photometric tails, not importance support. B does not
demonstrate improvement from more pure sleep. A is an imperfect control, not
ground truth. Topology repair was necessary but not sufficient. None of this
identifies which mechanism causes failure. In particular low ESS alone does
not identify narrowness, missing modes, noise mismatch or a wrong parent.

Local artifacts inspected include the earlier topology-pilot resolved B
configuration; balanced-v2 resolved cluster configs/checkpoints/banks have not
been read locally in this analysis. Source inspection is not certification of
all serialized runtime arrays. Obtain those artifacts before the SED audit.

## Concrete audit findings

1. `run_feniks_sc_drws_balanced_npe.py:checks/choose` requires tracking K256
   PASS as well as K1024 PASS. The supplied B/C/D results already disqualify
   every candidate under the unchanged contract. More K1024 draws cannot
   change these stored K256 results. Completing diagnostics can be useful;
   waiting for them cannot turn this experiment into a successful selection.
2. `evaluate_feniks_sc_drws_topology_npe_internal.py:evaluate` computes
   marginal and random-projection ranks on generated parameters, but does not
   compute direct importance support on those same simulated objects or SBC
   ranks of a data-dependent log likelihood. A simulation PASS is incomplete.
3. Sleep uses `_sleep_encoder_features`, observed inference uses
   `make_encoder_features`; simulator noise and likelihood scales also use
   separate helper paths. These warrant numerical parity tests, not an
   assumption that common configuration implies identical behavior.
4. The downloaded B config has Gaussian amortized likelihood, zero error
   floor/jitter, and `sleep.error_model=observed_catalog`. Do not diagnose it
   from unrelated top-level Student-t settings. Re-pairing observed errors
   with parent draws is consistent with a model conditioning on fixed errors
   independent of latent parameters; whether that conditioning matches the
   synthetic catalogue's generation process remains to be audited.
5. The canonical individual target is log L(y|x,c)+log p_parent(x).
   `posterior_target.py` intentionally has no particle-dependent selection
   factor. Selection stays in the population normalizer; preserving the
   observed selection band matters for masked-band inference.

### Executed negative and positive controls

Used the repository's marginal and projected rank functions on this exact
two-dimensional Gaussian experiment (CPU, no DSPS, no optimization):
theta~N(0,I), y=theta+N(0,0.01 I), 1024 objects, 128 draws, NumPy seed 260908,
rank/projection seed 1, KS/ECE cutoffs .06, 32 projections, unit scales.

| Direct proposal | Marginal ranks | Projection ranks | Loglike ranks |
|---|---|---|---|
| Prior, ignores y | PASS (.0325 KS) | PASS (.0531 KS) | FAIL (.9491 KS) |
| Exact N(y/1.01, .01/1.01 I) | PASS (.0386 KS) | PASS (.0452 KS) | PASS (.0214 KS) |

The loglike test quantity was -0.5*sum(((theta-y)/.1)^2), evaluated both at the
generated theta and at each proposal draw. These are analytical software
controls, not a diagnosis that the SED encoder actually ignores observations.
For selected catalogues the negative control must respect the selected joint
law; do not transplant unconditional SBC calibration assumptions unchanged.

## Phase 0: freeze the question and reuse evidence

Deliverable: one immutable experiment inventory and one paired per-object
diagnostic table. Record code/config/checkpoint hashes, seeds, row identities,
likelihood parameters, band ordering, parent identity, context model and costs.

Reuse existing K256 draws and residuals before decoding anything. Report
per-object changes, not only pooled metrics; bootstrap objects, not individual
draws or bands as if they were independent galaxies. Split observed difficulty
by predeclared SNR/mask groups, including Roman bands. This localizes symptoms,
not causal blame. Do not cherry-pick only high-ESS objects.

Operational recommendation: stop treating remaining K1024 as a selection
prerequisite for a winner that is already impossible. Preserve completed shards
and run missing internal diagnostics independently if useful. Implement an
explicit diagnostic-stop receipt before changing orchestration; do not forge
completion or alter the frozen gates. No cancellation was executed here.

## Phase 1: certify the mathematical and data contracts

CPU analytical controls, followed by at most two generated SED objects and a
small set of latent perturbations. No catalogue truth is needed.

- Recompute sample logq using inverse log_prob after saving/reloading the actual
  checkpoint; test inverse/forward, topology, density spaces and Jacobians.
- On exactly the same x and observation, compare sleep features to inference
  features and the ELBO target to the exported-inference target. Include masks,
  negative flux, all 18 bands, boundary points and physically valid SFHs.
- Compare cache and live DSPS fluxes, calibration ordering, parameter ordering,
  theta/x round trips and filter hashes. Check effective sigma, not just the
  nominal Gaussian/Student-t name. Audit catalogue error generation provenance.
- Check reparameterized gradients by finite differences on a few coordinates
  away from clips. Log per-term gradients BEFORE and AFTER clipping, parameter
  update norms and change in independent loss estimates. `update=1` alone is
  not proof of useful optimization. Check prior input gradients remain while
  prior parameter gradients are blocked.
- Use an exact Gaussian posterior and a low-dimensional bimodal target with
  quadrature to check densities, mode coverage, selection normalization and
  sensitivity of diagnostics. A control must fail when deliberately broken.

Exit: all deterministic identities pass precision-aware tolerances fixed before
the SED run. Statistical controls pass appropriate finite-sample checks. Any
mismatch blocks further scientific training until isolated and regression-tested.

## Phase 2: measure the amortization problem directly

Fixed parent, likelihood and initial A posterior. Primary diagnostic has only
16 observed validation objects, selected by flux/SNR/mask strata, and 16 fresh
simulated objects under the same model/contexts. Reuse existing A/D banks for
paired context; do not start another four-arm training benchmark.

For each case compare the amortized distribution with local VI initialized
from it. Optimize object-specific base AND coupling parameters with the same
flow family, keeping photometric context fixed. Changing only mean/scale is a
limited repair test, not a test of the full amortization gap. A second
deterministically perturbed initialization checks optimization dependence.

Local objective: E_q[logq-logprior-loglike], with all entropy/Jacobian terms.
This is ordinary VI at fixed parent, not supervised fitting or a point estimate.
Use fresh direct draws for evaluation. Reverse KL may miss modes: agreement of
two starts is not proof of correctness. Keep their distributions separate; if
testing a mixture, use its full known density, not best-of-K or pooled weights
computed against individual components.

Measure on BOTH generated and observed cases: raw ESS, finite and nonfinite k
separately, max weights, independent log-evidence replication, loglike/prior/q
decomposition, distribution widths/correlations and robust residuals. On
generated cases also test marginal/joint/data-dependent likelihood ranks.
Sixteen generated objects provide a failure-localization probe, not a reliable
SBC qualification; expand to a separate 128-256 set only after a useful result.

Initial hard ceiling: 32 cases, 2 local starts, 64 optimizer steps, 4 draws/step.
This is 16,384 differentiable decoder evaluations. Baseline and each fitted
start get two fresh K128 evaluations: 8,192 + 16,384 forward evaluations,
40,960 total before the small simulation/audit overhead. Reuse baseline banks
where valid. Backpropagation and compilation cost more than forward calls:
measure a two-object microbenchmark first and cap the whole diagnostic at
one H100 on one node for three hours. If it cannot fit, reduce cases/steps
before submission; report an inconclusive optimization budget, not family failure.

## Phase 3: act on the outcome, not on a new hyperparameter guess

| Observation after contract audit | Interpretation | Next intervention |
|---|---|---|
| Local VI improves same-family support/target agreement on simulated and observed cases | Evidence of amortization/optimization error | Semi-amortized VI: encoder initialization plus bounded local refinement; amortize it only after verification |
| Local VI remains unstable even on generated cases | Optimization, conditioning or family remains unresolved | Test Gaussian/low-D SED controls and Jacobian conditioning; compare one justified local full-covariance/flow alternative, not a sweep |
| Generated cases work, observed cases do not | Domain/context mismatch or observed-target geometry suspected | Audit flux-error-mask support and selected prior predictive photometry; do not assume noise inflation is the remedy |
| Target approximation is reliable but population photometry is wrong | Model/parent inadequacy, not an inference-only problem | Start small, selection-corrected population updates with reliable integration |
| Local runs disagree by mode or evidence | Unresolved approximation/support | No teacher distillation or population update; diagnose modes using known-density direct proposals and low-D controls |

Failure of 64 steps cannot prove that a family cannot represent the target.
Conversely a larger ELBO is not by itself sufficient evidence for reliable
moments, evidence or mode mass. Preserve uncertainty in these decisions.

## Phase 4: separate inference readiness from population adequacy

The parent need not already reproduce catalogue truth to learn a better parent.
A wrong parent still defines a mathematical posterior. Require evidence that
q/integration represents that posterior, not perfect truth calibration under a
misspecified parent. Do not create a circular gate requiring the population
problem to be solved before allowing population learning.

Keep two explicit records in a NEW protocol, without changing existing receipts:

- Inference readiness: density/gradient identities, simulated target checks,
  representative observed importance stability and no unresolved catastrophic
  integration cases. Existing ESS/k checks remain useful screening, not universal
  certificates. K-dependent uncertainty and paired replicated integrals matter.
- Population adequacy: selected prior predictive photometry, held-out predictive
  performance and population-evidence improvement. Bad values can motivate a
  prior update once inference is reliable; they are not automatically a q bug.

Then test one small prior step on a representative training subset with recorded
inclusion probabilities, using frozen direct proposals and full denominators.
Optimize sum_i log integral L(y_i|x,c_i)p_phi(x)dx - sum_i log alpha_phi(c_i).
Derive alpha from the real context/noise/selection contract; a single shared
alpha requires justification. Check integration and gradients across independent
banks before and under the candidate prior, then refresh q for the new parent.
Do not average only finite/easy objects, or use better ESS alone to select a prior.
Do not require accurate full 15D truth recovery where photometry is uninformative.

## SFH and scientific endpoint

Audit the error-whitened DSPS Jacobian in documented latent coordinates on
generated and direct-q draws, including singular directions and clipping.
Separate constraints on observable flux combinations from unconstrained SFH
directions. Low sensitivity at a few points is not global non-identifiability.
Any continuity regularization must be physically specified and sensitivity-tested,
not set from catalogue-truth extrema. Retain joint distributions throughout.

Endpoint: report conditional posterior approximation quality, selected-photometry
predictive quality and prior-sensitive directions separately. An accurate 15D
parent cannot be guaranteed from broadband photometry alone.

## Engineering and spending rules

- One branch of the decision tree at a time, with an explicit next decision.
- Each compute request states its hypothesis, falsifier, cases, decoder calls,
  time/memory ceiling and checkpoint/receipt reuse. Missing these means no launch.
- Run internal diagnostics first, K128/K256 screening next; K1024 is targeted to
  a bounded ambiguity/stability question, not automatic for every failed arm.
- Profile decoder chunk sizes on two objects; certify equal outputs and memory
  headroom before changing the conservative chunk=1 setting. No untested OOM fix.
- Test the actual CLI and actual device count through at least one optimizer
  update and one diagnostic. The previous inference-only smoke missed pmap.
- Decouple train, observed support, internal validation and report jobs so an
  internal failure cannot force repeated hours of already-completed inference.
- Keep one compact dashboard of decisions, costs and failures, not just Slurm
  completion and unrelated training losses. No automatic production follow-up.

## Sources and execution record

Approximation versus amortization error: [Cremer et al., 2018](https://proceedings.mlr.press/v80/cremer18a.html).
Data-dependent SBC quantities: [Modrak et al.](https://arxiv.org/abs/2211.02383).
Importance-weight reliability: [Vehtari et al., 2024](https://jmlr.org/papers/v25/19-556.html).

Executed here: source/config inspection and two CPU Gaussian rank controls.
Not executed: balanced-v2 artifact readback, SED local VI, hardware profiling,
new training, gate modification, population updates or cluster job cancellation.
Only this plan and PLAN.md were edited for the diagnostic-reset request.
