# Frozen-parent NPE: next bounded experiment

## Evidence and interpretation

The Sept 8 completed pilot reports no winner. At K=1024 the median ESS
fractions are A=0.0116, B=0.0056, C=0.0057, with bad Pareto-tail fractions
0.734, 0.891, 0.891. This is approximately 12, 6, 6 effective draws per object,
not 1024 useful independent posterior draws. B passes the existing marginal
simulation check but fails held-out prediction; C reverses this tradeoff.
The completed training and corrected topology are technical progress, not
evidence of validated joint posteriors. The parent did not change.

The downloaded earlier training logs show B's sleep validation NLL still
decreasing through epoch 8. C's mixed objective is not comparable to B's sleep
NLL. Its single 8-object gradient audit showed a much larger weighted observed
gradient than sleep gradient. That is a warning, not an estimator of an optimal
coefficient: norms, directions, clipping, training time and the sampled cohort
all matter. The new coefficients are prespecified exploratory hypotheses.

The pilot's `simSBC=PASS` tested marginal coordinates, not the complete joint
distribution. Its held-out group included the selection-defining r band, while
the encoder was not deliberately trained on that missing-band pattern. The new
experiment keeps r observed: the selection event then remains determined by
the conditioning photometry, and the individual target is still L(y|x)p(x).

PSIS diagnostics assess importance approximation reliability, not whether q
is simply too narrow. See [Vehtari et al., 2024](https://www.jmlr.org/papers/v25/19-556.html).
No change to the likelihood, decoder, physical bounds, or noise inflation is used.

## Implemented experiment

1. S: continue the certified corrected B checkpoint for 24 epochs, pure sleep.
   First run a two-object, four-draw SED inference smoke on training photometry.
   Keep the parent, calibration, feature statistics and serialized topology
   fixed. Reuse a new cached bank of 131072 directly generated parent parameters
   and noiseless fluxes; renew simulation noise every optimization step.
2. B/C/D: all start from S's best checkpoint, with newly initialized optimizers,
   the same random seed and 8 epochs at learning rate 1e-5. B is pure sleep;
   C and D retain sleep plus observed reverse KL with weights 1e-4 and 1e-3.
   Observed reverse KL uses 4 direct reparameterized draws and the canonical
   posterior target, including logq and all existing Jacobians. Invalid target
   draws reject the update instead of silently conditioning the ELBO on validity.
3. Teach missing-band conditioning during sleep: after noisy-r selection,
   remove the u/i/H group for 25% of simulated objects. Selection masks are
   evaluated before that augmentation; r remains observed. Observed ELBO uses
   the full observed training photometry, not a simulated substitute.
4. Compare historical A and B/C/D on the same observed development cohort:
   256 objects at K256, 128 at K1024, raw-weight diagnostics and dense residuals.
   Internal simulations share their seed, parent and context; their input hash
   must match across all four arms. Test marginal and 32 fixed joint-projection
   ranks with full and masked conditioning. Projection scales come only from
   the generated training bank. Persist held-out residuals versus simulated
   references and the bounded flux-Jacobian diagnostics.
   Export `matched_validation.csv` and `matched_validation.png` in the run root.
5. Freeze a candidate only if every technical gate passes, including marginal
   and joint-projection simulation checks and a common-simulation NLL no more
   than 0.5 nats worse than B. Among eligible candidates choose that common NLL.
   A is a historical control, not an eligible replacement. Confirm the single
   frozen candidate on 256 reserved observed-validation rows with independent
   seeds. No second candidate is tried if confirmation fails.

The reserve is excluded from this continuation's epoch/checkpoint validation
and from development support. It is not a newly unseen catalogue: the older
models previously used the full validation split. This limitation is recorded
in the manifest. No observed test truth or historical truth metrics are read.
The resolved configs explicitly disable inherited truth-redshift stratification
and catalogue fingerprint histograms, as well as truth snapshots/diagnostics;
empty `truth.parameter_columns` alone is not this guarantee.

No population job is submitted, even after technical confirmation. The current
parent's fit to q remains an initialization, not proof that its 15D distribution
is correct. `posterior_technical_ready=true` is a prerequisite for a separately
reviewed population experiment, not a scientific promotion or production flag.

## Budget

The schedule uses one H100 per GPU task, one node allocation per task and 16 CPU
cores per GPU. S runs alone; B/C/D train concurrently (3 GPUs); A/B/C/D validate
concurrently (4 GPUs); confirmation uses 1 GPU. Peak: 4 H100 across 1-4 physical
nodes depending on scheduler placement. No multi-GPU distributed training.

Allocation ceilings: 4 hours for S, 4 for each candidate, 10 for each of four
validations, and 10 for confirmation: at most 66 GPU-hours. This is a ceiling,
not a runtime prediction. Training reports decoder counts and wall time; the
ELBO arms necessarily cost more than pure sleep. Each validation makes 196608
observed posterior decoder evaluations plus its bounded internal diagnostics.
The earlier K256+K1024 validation took roughly 7.5 hours with decoder chunk 1;
we retain that memory-safe chunk and batch size 8 and allow 10 hours.

## Launch on Jean-Zay

From the login node, with the updated committed branch:

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
unset BALANCED_ROOT RESUME_SUBMISSION
bash scripts/submit_feniks_sc_drws_balanced_npe.sh \
  outputs/logs/feniks_sc_drws_topology_npe_pilot_latest.env
bash scripts/monitor_feniks_sc_drws_balanced_npe.sh
```

Preparation verifies source hashes and split contracts before submitting. It
creates `frozen_parent_balanced_npe_v1` next to the old pilot. It never erases the
pilot or overwrites an existing experiment. Every submitted job is immediately
recorded in `SUBMISSION.json` and the latest environment. To resume only an
interrupted *submission* use `RESUME_SUBMISSION=1` with the same command/commit;
this does not retry failed jobs. An ambiguous interrupted sbatch fails closed
and needs the queue checked before proceeding.

GPU validation actions are shard-resumable and do not delete existing inference
outputs. A failed internal diagnostic need not recompute completed K256/K1024
summaries. Partial training is preserved and explicitly refused for overwrite.
Recovery of failed jobs must restore their dependencies rather than submit a
second full experiment blindly.

```bash
source outputs/logs/feniks_sc_drws_balanced_npe_latest.env
sacct -X -j "$ALL_JOBS" --format=JobID,State,Elapsed,ExitCode
tail -F "$BALANCED_LOG_ROOT/candidates-${CANDIDATES_JOB}_0.out" \
  "$BALANCED_LOG_ROOT/candidates-${CANDIDATES_JOB}_1.out" \
  "$BALANCED_LOG_ROOT/candidates-${CANDIDATES_JOB}_2.out"
python -m json.tool "$BALANCED_ROOT/BALANCED_NPE_COMPLETE.json"
```

## Limits and next decision

This is a controlled attempt, not a promise that more sleep or smaller ELBO
weights suffice. Projection SBC does not prove all conditional modes are
present. A generated reference can share model error; held-out prediction and
importance support remain independent requirements. The first-batch gradient
audit is local, not an epoch-wide gradient estimate. Direct IS has finite-K bias.

If no arm passes, preserve all results and examine whether the failure is already
present on model-generated full/masked photometry, or only on observed rows.
That distinction determines whether to investigate posterior approximation and
optimization, or the frozen parent's predictive distribution/context model.
Do not respond by loosening gates, clipping truth-derived SFH bounds, filtering
out low-ESS galaxies, or launching population updates with unstable integrals.

Execution record: prepared code and local tests only. Actual SED checkpoints
and Jean-Zay H100 performance must still be exercised by the submitted run;
no new cluster training was launched by this implementation turn.

Local checks: 83 targeted tests passed (posterior topology/densities/ELBO
gradients, finite-rank diagnostics, population-VI primitives, frozen-parent
contracts, and the new submission/reservation/gating tests). `compileall`,
CLI help and shell syntax checks passed. The Gaussian projection test is an
analytical software smoke, not a demonstration of SED posterior calibration.
