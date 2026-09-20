# Decoupled forward population inference

## Scope and provenance

This is a new pipeline, not a modification of the SBEB/RWS training objective.
The active SBEB source files were checked against
`feniks_sbeb_gpt_pro_bundle_20260919.tar.gz`; the current checkout supplies the
implementation. The older `rws_k8_t2_seed2.7z` contains historical configurations
and results/checkpoints, not the source tree to modify.

The only reused learned objects are the frozen decoder calibration and observed
feature scaling/latent coordinate specification. The old encoder/prior can be
deserialized by the existing runtime loader, but neither generates population
samples nor supplies classifier labels or supervised posterior targets.
There is no E-step, wake loss, posterior aggregation update, NUTS, or SMC.

## Statistical contract

Let theta=(a,b), with the five physical coordinates
`z_obs, log10_stellar_mass, log10_stellar_metallicity, dust_av, dust_delta`.
All ten SFH coordinates remain random and are passed to DSPS normally.
Writing z=T^{-1}(theta) for the existing invertible latent coordinates:

```
g_j(z) = Normal(z_a; mu_j, sigma_j^2 I) Normal(z_b; 0, I)
p0(z) = (1/J) sum_j g_j(z)
p_u(z) = sum_j u_j g_j(z)
```

The correction p_u/p0 depends only on physical coordinates. Each g_j is
analytically normalized in 15D, without a fitted partition constant or rejection
normalizer. Sampling is a categorical component draw plus a full 15D Gaussian
draw, then the repository's transform to physical parameters. `PopulationPrior`
provides latent log density, sampling and Jacobian-correct physical log density.

The main basis has 128 overlapping components: 127 scrambled Sobol centers
mapped through a clipped normal quantile, multiplied by 1.5, with physical
standard deviation 0.7; one central component has standard deviation 2.5.
The geometry is joint 5D, not five independent histogram fits.
The reference nuisance conditional is the existing identity-flow N(0,I)
reference, **not** the previously fitted population flow. Nuisance coordinates
are not fixed, removed, or individually optimized.

This is an explicit modeling restriction: an arbitrary true parent, especially
its SFH conditional, need not belong to this family. Decoupling eliminates the
q-feedback failure mechanism, not model misspecification or nonidentifiability.
The recovered parent is within the configured physical/C0 support domain;
objects outside that initial domain cannot be reconstructed by this run.

## Observation and selection

The bank uses the frozen DSPS forward model/calibration, the existing m5 error
law and feature preprocessing (flux, errors, masks). The synthetic catalogue
generator uses Gaussian m5 noise, so this experiment explicitly uses Gaussian
noise, not the old robust Student-t fitting likelihood. The cut is applied to
**noisy observed** `lsst_r` flux, strictly greater than flux(AB=29), with a valid
r-band mask. All rejected simulations remain in the efficiency denominator.

The first implementation requires all bands observed and checks both training
and validation masks. It stops for missing bands instead of recycling selected
masks as an unjustified parent mask distribution. Adapting to a real variable
depth/missingness survey requires its generative metadata law, not a silent
approximation. No 1/beta(theta) enters individual posterior weights.

The synthetic catalogue computes reported errors from noiseless flux. This can
make the error columns informative about theta; matching that joint law is
intentional here. Calibration on such a catalogue must not be extrapolated to
real surveys with estimated uncertainties without validating their error model.

## Classifier and population fit

A selected-simulation multiclass MLP (3 hidden layers, width 256, GELU) estimates
C_j(x). Its training frequency c_j is measured **after selection and splitting**.
The split is 70% classifier training, 15% checkpoint validation, 15% independent
ratio/known-mixture testing. Cross entropy uses ordinary class frequencies, not
class-balanced sampling with an uncorrected c.

```
d_ij = C_j(x_i) / c_j
maximize_v sum_i log(sum_j v_j d_ij),   v >= 0, sum(v)=1
alpha_j = selected_component_draws / all_component_draws
u_j = (v_j / alpha_j) / sum_l (v_l / alpha_l)
alpha_parent = sum_j u_j alpha_j
```

The objective is evaluated with row-centered log ratios (exponent floor -700)
and analytic gradients. SLSQP solves the simplex problem; a separate linear
oracle checks its Frank-Wolfe/KKT gap (tolerance 2e-6 per object) and feasibility.
Displacement or the solver's success flag alone is not a convergence test.
The selected likelihood is concave; classifier errors and ill-conditioned
components nevertheless remain statistical errors, not optimization guarantees.

Selection support uses binomial counts and Wilson lower 95% bounds. Components
with fewer than 128 accepted draws or lower bound <0.001 are weakly identified.
Their combined **parent** mass is bounded by 0.05. This becomes a linear
constraint on v:

```
sum_j v_j (1_weak(j) - 0.05) / alpha_j <= 0.
```

Thus the problem remains convex. The constraint is reported, including whether
it binds; it is a declared regularization assumption, not proof of recovery.
Any zero-alpha component stops the fit rather than applying an arbitrary floor
or pretending an unseen part of the parent is identified. Both u and v are
always saved with alpha, counts, class frequencies and support flags.

## Frozen-parent supervised posterior

After population fitting, `parent.json` is frozen by checksum. A new independent
bank is drawn from that full parent, with the same observation law and selection.
The posterior is initialized **from scratch**, using four independent conditional
15D rational-quadratic-spline flow experts, 12 coupling layers, width 256,
16 bins, tail bound 12, and latent-x output. The inherited feature-context
architecture is retained; the configured MLP hidden sizes are 256/256/256.

```
L(phi) = -E_{theta,x ~ final parent simulator, S(x)=1} log q_phi(theta | x)
```

Training uses latent-x log density; the physical Jacobian is independent of phi.
Because selection is determined by x, selected-pair training has the correct
individual conditional target for selected galaxies. All targets are simulator
draws, never q samples. AdamW, gradient clipping, validation-NLL best checkpoint
and resumable epoch checkpoints are used. There are 100 configured epochs.
No posterior temperature or artificial width inflation is applied.
This is empirical Bayes: uncertainty in fitted population weights is not
integrated into individual posteriors in this first implementation.

## One main Jean-Zay run

Configuration: `configs/experiments/feniks_forward_population_r29.yaml`.

| Stage | Forward draws | Resources |
|---|---:|---|
| Contract preflight | 256 | 1 H100, 1h limit |
| Reference | 4,194,304 | 8 shards, 1 H100 each, 20h each |
| Classifier/population | reuse reference | 1 H100, 20h |
| Posterior training bank | 2,097,152 | 4 shards, 1 H100 each, 20h each |
| Independent evaluation bank | 131,072 | 1 shard, 1 H100, 20h |
| Supervised posterior | reuse bank | 1 H100, 20h |
| Blind report | reuse bank + q draws | 1 H100, 10h |

Total: **6,422,528 bank simulations + 256 preflight simulations**. Reference
components each receive 32,768 attempts. Posterior-bank concurrency is four;
reference concurrency is eight. Peak is **8 H100**, not four GPUs per task.
Allocated-time ceiling is **311 H100-hours**, excluding resubmissions. This is
the sum of walltime limits, not a measured runtime prediction. CPU-only local
tests cannot certify H100 throughput or finish time. The launcher prints the
implementation summary and resources before submitting anything.

Run from the updated existing Jean-Zay checkout, with its `Data/` and filters:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
OLD="$BASE/avi_sbeb_benchmark_20260917_230455"
ROOT="$BASE/avi_forward_population_r29_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_feniks_forward_population.sh \
  "$OLD/runtime/r29" "$ROOT" \
  configs/experiments/feniks_forward_population_r29.yaml "$OLD"
bash scripts/watch_feniks_forward_population.sh "$ROOT"
```

This prepares **one** dependency chain, not a hyperparameter array. The code is
archived and extracted into an immutable SCRATCH snapshot; datasets are linked,
not cloned. Input configs, catalogues, decoder/filter assets and basis are hashed.
The launcher does not submit from this local machine or assume SSH works.
The r27 case is not submitted automatically: use its own runtime and change
the configuration cut to 27 only after examining the r29 result.

## Recovery

Banks resume at saved batch blocks; optimizers resume at saved epochs. Resume
with the same immutable code, root, config and input hashes. Do not prepare a
new root to resume a partially computed bank. For a timed-out job, inspect its
stderr and receipt before requeueing that **original job ID** on Jean-Zay:

```bash
source "$ROOT/JOBS.env"
# Example only after verifying a TIMEOUT, not a scientific support failure:
scontrol requeue "$TRAIN_JOB"
```

Requeue availability and dependent-job behavior are site-controlled. If Slurm
refuses it, resubmit only the failed stage using the saved CODE_DIR and rewire
the remaining afterok dependencies; do not duplicate running arrays. Scientific
contract failures (zero selection support, mask mismatch, failed KKT) are not
fixed by repeated requeueing.

## Outputs and interpretation

### Recovery after the population KKT stop

SLSQP can report successful objective convergence while the independently
computed KKT gap still exceeds `2e-6`. The solver now restarts with the same
objective scaled for numerical precision, then uses feasible Frank-Wolfe
steps with a scalar line search if necessary. The original KKT tolerance,
likelihood and weak-selection parent-mass constraint are unchanged. Receipts
record the initial/final gap and refinement iterations.

For a failed population stage with all reference banks and classifier epochs
complete, update the checkout and run from its root:

```bash
bash scripts/resume_feniks_forward_population.sh "$ROOT"
bash scripts/watch_feniks_forward_population.sh "$ROOT"
```

This validates input/bank hashes and classifier resume state, refuses active
jobs or an already finalized population, preserves previous code/job pointers
under `recovery/<timestamp>/`, freezes a new code snapshot, cancels only pending
jobs in the old population-to-report chain, and submits replacement dependencies.
Reference simulations and completed classifier training are not repeated.
The existing classifier resume loader returns its best checkpoint after
skipping already completed epochs. Ratio evaluation and population fitting
are repeated. The new `observed_ratios.npz` and `component_selection.csv` are
saved before fitting to support numerical failure diagnosis.

If submission itself fails, `RECOVERY_PENDING` points to the recovery directory
and its submitted job IDs. Inspect these before any retry; do not blindly remove
the marker and create duplicate work. No downstream simulation bank may already
be finalized when using this specific recovery path.

- `MANIFEST.json`, `experiment.yaml`, `CODE_SHA256`: immutable contracts.
- `reference/shard_*/bank.npz`: selected and rejected full forward simulations.
- `population/component_weights.csv`: u, v, c, alpha, Wilson bounds and support.
- `population/parent.json`, `best.eqx`, `FINAL.json`: parent, classifier, fit gates.
- `population/simulation_closure.csv`: independent known-family ratio closure.
- `posterior_bank/shard_*/bank.npz`: new simulator-supervised pairs.
- `posterior/best.eqx`, `training.jsonl`, `RESUME.json`: 15D posterior/checkpoints.
- `report/REPORT.md`: figure guide and scientific limitations.
- `report/population_physical*.png`, `population_15d.png`: explicit true/learned
  parent and selected distributions; W1/IQR and physical joint sliced W1 tables.
- `report/selected_aggregate_physical.png`: equal-object mixture of entire
  individual posteriors compared with selected prior and true selected population.
- `report/*calibration.csv`, `*pit_truth.png`: 68/95% coverage, PIT KS, median
  bias and widths, separated into physical and nuisance groups.
- `report/individual_*`, `simulation_individual_*`: representative physical corners with truth and
  full 15D marginal distributions, plus raw posterior NPZ files.
- `report/observable_predictive*`: frozen-parent forward observable PPC.
- `report/baseline_*`, `old_sbeb_*.csv`: historical calibration/parent comparison,
  explicitly descriptive if cohorts differ, not a paired significance test.

Half of the independent evaluation shard selects posterior checkpoints; the
other half tests calibration. Catalogue truths are opened only by the final
report, not population fitting or either network optimizer. Known-family closure
tests estimator implementation; fitted-parent SBC tests q under its assumed
model. **Neither alone proves real-parent recovery.** Blind catalogue closure
and observable PPC are essential. Check 68/95% coverage and widths together,
not ESS alone. No automatic scientific promotion occurs.

The existing factorial report's `parent_physical_w1_over_iqr` bug is fixed by
filtering `group == physical` before aggregation, with a regression test.

## Local verification

```bash
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu .venv/bin/pytest -q \
  tests/test_forward_population.py tests/test_forward_population_pipeline.py \
  tests/test_avi_em_factorial.py
python -m compileall -q euclid_dsps scripts
bash -n scripts/submit_feniks_forward_population.sh
bash -n scripts/feniks_forward_population.slurm
bash -n scripts/watch_feniks_forward_population.sh
```

The CPU stage-integration test executes bank creation, classifier fitting,
selected/parent optimization, frozen-parent supervised spline training and
report generation with lightweight synthetic photometry in place of DSPS.
The actual DSPS/noise/mask/GPU interface is additionally checked by the single
256-simulation preflight before the large reference bank is allocated.
