# Post-hoc posterior calibration experiments

This workflow diagnoses two possible sources of posterior miscalibration while
keeping the DSPS forward model and likelihood fixed:

1. amortized inference error, tested with raw importance weighting and PSIS;
2. population-prior error, tested with proposal-refresh generalized EM.

It does not test or change the forward model.

## Distribution contract

Every calculation consumes joint posterior draws. The workflow writes the raw
and PSIS weights, a seeded joint resample, ESS and Pareto-k diagnostics, PIT,
coverage, MIRA/TARP artifacts, exact checkpoint hashes, and completion markers.
A posterior median is only used as the conventional point estimate in
`z_true` versus `z_inferred` metrics. It is never treated as a posterior or
used to decode a joint latent vector.

## Importance experiment

For every joint proposal draw, the exact stored densities define

```text
log w = loglike + logprior - logq.
```

The same objects are evaluated at progressively larger proposal budgets.
PSIS can stabilize a sufficiently overlapping proposal but cannot recover a
mode that was never sampled. ESS and Pareto-k are therefore scientific outputs,
not optional logging.

The Jean-Zay array contains FENIKS synthetic and Pop-COSMOS tasks for every
requested budget. Each task uses one H100. The default `128,512,2048` grid is a
pilot; add 8192 only after inspecting the first three budgets.

## Empirical-Bayes experiment

Each outer iteration performs:

1. proposal generation with the current checkpoint;
2. an E-step with stopped per-object weights;
3. a weighted maximum-likelihood M-step for the exact-density prior flow;
4. proposal regeneration under the updated prior before the next iteration.

The M-step includes a cross-entropy trust penalty using samples from the old
prior. Training and evaluation cohorts remain disjoint. Public spectroscopy is
used only after inference for Pop-COSMOS redshift evaluation.

The support gate stops EM when the proposal is too degenerate. Setting
`ALLOW_LOW_ESS=1` is allowed only for a diagnostic run and is recorded as an
override; it does not make the resulting prior scientifically valid.

## Main artifacts

- `weighted_samples/`: original joint draws with raw and PSIS weights;
- `resampled_samples/`: seeded PSIS-resampled joint draws for MIRA/TARP;
- `importance_diagnostics.parquet`: ESS, max weight, Pareto-k per object;
- `redshift_weighted_objects.parquet`: weighted PIT and intervals;
- `iteration_*/prior_update/checkpoints/best.eqx`: prior after each M-step;
- `em_history.csv`: held-out importance-evidence and support diagnostics;
- `calibration_comparison/`: shared-region MIRA/TARP before versus after EM;
- `DONE` and JSON manifests: terminal status and input hashes.

## Jean-Zay launch

From the existing `$WORK/dsps-popcosmos` checkout and `shine` environment,
first run the controlled importance-budget matrix:

```bash
export REPO_DIR="$WORK/dsps-popcosmos"
export MINICONDA_PATH="$WORK/miniconda3"
export CONDA_ENV=shine
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export RUN_TAG=$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT="outputs/runs/posthoc_calibration_${RUN_TAG}/importance_probes"
export LIMIT=256
export BUDGETS_CSV=128,512,2048
export ARRAY_CONCURRENCY=2
bash scripts/submit_posthoc_importance_probes.sh
```

After every array task is `COMPLETED`, collect the diagnostics and inspect the
largest budget separately for FENIKS and Pop-COSMOS:

```bash
source outputs/logs/posthoc_importance_latest.env
sacct -X -j "$IMPORTANCE_JOB" \
  --format=JobID,JobName%20,State,Elapsed,Timelimit,AllocTRES%40,ExitCode,NodeList

CALIB_ROOT="${IMPORTANCE_OUTPUT_ROOT%/importance_probes}"
python scripts/summarize_posthoc_calibration.py \
  --root "$CALIB_ROOT" \
  --out "$CALIB_ROOT/decision_tables"
column -s, -t < "$CALIB_ROOT/decision_tables/importance_decision_table.csv" | less -S
```

Only if ESS/Pareto-k support is adequate, launch the alternating prior pilot:

```bash
export OUTPUT_ROOT="$CALIB_ROOT/empirical_bayes"
export TRAIN_LIMIT=5000
export EVAL_LIMIT=500
export PROPOSAL_SAMPLES=512
export EVAL_SAMPLES=512
export EM_ITERATIONS=3
export MSTEP_EPOCHS=5
export ARRAY_CONCURRENCY=2
export ALLOW_LOW_ESS=0
bash scripts/submit_posthoc_empirical_bayes.sh
```

Monitor and summarize it with:

```bash
source outputs/logs/posthoc_empirical_bayes_latest.env
sacct -X -j "$EMPIRICAL_BAYES_JOB" \
  --format=JobID,JobName%20,State,Elapsed,Timelimit,AllocTRES%40,ExitCode,NodeList

for task in 0 1; do
  tail -n 100 "outputs/logs/posthoc_em-${EMPIRICAL_BAYES_JOB}_${task}.out"
  test ! -s "outputs/logs/posthoc_em-${EMPIRICAL_BAYES_JOB}_${task}.err"
done

python scripts/summarize_posthoc_calibration.py \
  --root "$CALIB_ROOT" \
  --out "$CALIB_ROOT/decision_tables_final"
```

The default EM task fails closed on inadequate proposal support. Do not set
`ALLOW_LOW_ESS=1` for a result intended for the paper. If the 512-draw bank
fails, submit a new output root with 2,048 draws; if that also fails, the next
experiment is a target-exact reference or a richer proposal, not a forced EM
update.
