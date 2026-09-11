# Full-catalogue AVI experiments

## What We Learn

We want the encoder to produce plausible **joint distributions**, directly from
photometry. We compare seven ways to improve that encoder. Every experiment
sees the full frozen training selection: **37,641 observed catalogue rows**.
No catalogue truth is used for training, validation or checkpoint selection.

The learned population prior, physical decoder, photometric noise model and
feature scales are held fixed. This is intentional: changing the prior at the
same time would change the answer the encoder is trying to learn. **This wave
does not learn a new population prior.** It chooses a credible encoder method
before a separate, selection-corrected population-learning experiment.

| Array index | Arm | Simple description | Question |
| --- | --- | --- | --- |
| 0 | A | Repair the current encoder. | Is retraining enough? |
| 1 | B | Repair four full flow experts with a learned mixing network. | Does a more flexible distribution help? |
| 2 | C | Restart the single encoder, keep the learned prior. | Is the old encoder a bad starting point? |
| 3 | D | Restart four independent full flow experts. | Do flexibility and a fresh start work together? |
| 4 | E | B plus a small direct photometric ELBO term. | Can decoder gradients teach the encoder what its own draws get wrong? |
| 5 | F | B plus existing joint NUTS draws for eight training galaxies. | Does a small empirical distribution teacher help? |
| 6 | G | F plus small changes to those galaxies' photometry. | Can the teacher help nearby observations too? |

There are **no new NUTS jobs**. F/G are auxiliary tests, not the main inference
method. Their bank retains complete joint coordinates and all retained chains.
It is not replaced with parameter medians or independent one-dimensional fits.
The existing NUTS `diagnostics_pass=false` values are preserved in `teachers.json`.
Equal-length chains define an empirical teacher, not certified relative mode
probabilities. Improvement on these eight galaxies alone is not generalization.

## Fixed Settings

- Seven tasks, one seed. One node and **four H100 GPUs per task**, 48 CPU cores.
  Concurrency seven means at most **28 H100s across seven nodes**. Preflight and
  training are successive waves, not 56 simultaneous GPUs.
- Real data parallelism over galaxies. Global batch 256 = 4 devices x 32
  galaxies x 2 accumulated microbatches. Gradients are averaged before one AdamW
  update. No four independent copies training unrelated models.
- Sixty epochs: two sleep epochs, one wake epoch, repeated. Scratch encoders
  begin with six sleep epochs. Each epoch covers every selected training row;
  the final batch wraps the permutation rather than dropping rows.
- Sleep means learning from photometry simulated with the frozen physical model
  and observed noise/mask patterns. Wake means learning from weighted draws for
  the observed catalogue. Gaussian zero-floor target, inherited selection-aware
  sleep generation, qualified float64 conditional transport.
- AdamW: learning rate 2e-5, 5% linear warmup from 2e-6, then cosine decay to
  1e-6, global gradient clipping at 5. No moving loss-dependent scheduler or
  automatic prior update. Nonfinite updates fail the task instead of silently
  training through them.
- Wake uses **128 total draws per galaxy**, not 128 per expert. Sixteen come
  from the frozen prior; the other 112 are divided equally among experts.
  Exact proposal-mixture density is used in the denominator. This launch does
  not enable a second K512 hard-case training pass; K512 is for evaluation.
  The decoder handles blocks of eight draws across local galaxies, instead of
  serial single-draw calls or one unbounded tensor of all physical activations.
- E adds 0.1 times the negative ELBO, with two reparameterized draws per expert
  and exact summation over the learned expert probabilities. Both continuous
  expert parameters and mixing probabilities receive gradients.
- F/G add a teacher loss with coefficient 0.1. Each optimizer step uses sixteen
  teacher observations and 64 joint bank draws per teacher observation.
- G perturbs flux by 0.2 times its reported error, keeping errors and masks.
  Its weights are proportional to `p(new photometry | x) / p(old photometry | x)`.
  The prior cancels because it is unchanged. Empirical neighbour batches with
  ESS below 16 do not contribute that auxiliary loss. This does not drop any
  observed galaxies from the main wake loss and does not update the prior.

The arms have equal catalogue exposure and optimizer-step budgets, **not equal
decoder cost**. E/G do extra physical evaluations. Compare timing as well as
posterior metrics; do not assume seven identical runtimes.

## Launch

Update the existing checkout through Git; no manual file transfer is needed.
Run from the updated repository on Jean-Zay, not from an old frozen worktree.
The launcher snapshots the actual Python/configuration/shell source, including
uncommitted runtime changes. It records hashes and never modifies old runs.

```bash
cd "$WORK/dsps-popcosmos"
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
ROOT="$BASE/avi_encoder_experiments_v1"
bash scripts/submit_feniks_avi_experiments.sh "$BASE" "$ROOT" 7
```

Required upstream inputs: `$BASE/manifests/{manifest.json,full_train_indices.npy,
confirmation_indices.npy}`, `frozen_parent_wake_holdout_v1`, the arm-C checkpoint
and feature statistics it references, and the completed
`frozen_geometry_nuts_observed8_dense_depth6_v1` banks. Missing or changed inputs
stop preparation. The default validation catalogue is `test.parquet` next to
the training catalogue. At most 512 upstream confirmation rows are used for
validation. They become development validation, **not an untouched final test**.
Other test rows remain unused. Simulated NUTS cases are not teachers.

Each preflight uses four GPUs and runs two real sleep and two real wake updates,
including its arm's auxiliary objective. It checks finite work, parameter
changes and frozen prior/calibration. The long array is held behind an `afterok`
dependency on the entire preflight array. Preflight has a two-hour limit;
training allocations have a twenty-hour limit. These are ceilings, not runtime
predictions. We have not benchmarked this implementation on H100 yet.

If a preflight fails, inspect its error log; do not release the dependent array
by hand. A passed preflight establishes executability, not posterior quality.

## Monitor And Resume

After v3 preflight 2047358, use `avi_encoder_experiments_v4`. The snapshot now
includes the contents of `filters/`, including site-local symlink targets.
Preparation parses the filters and checks SSP/configured model asset files
before GPU submission. Their hashes join the immutable input manifest.
Existing physical assets stay on Jean-Zay; no transfer or substitute filters.

For recovery after v2 preflight 2047211, use a new
`avi_encoder_experiments_v3` root. The fix preserves YAML free-parameter order
and checks the latent coordinate hash against the source checkpoint during
preparation. Do not disable checkpoint hash checks or edit old run snapshots.

The GPU launcher explicitly enables JAX plugin discovery with
`EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0`, matching the NUTS launcher.
Without it, the repository's conservative default hides installed CUDA plugins.
The first AVI preflights (2046734) failed before training for this reason.
After pulling this fix, submit to a new root (for example
`avi_encoder_experiments_v2`); do not resume the old frozen source snapshot.

```bash
bash scripts/watch_feniks_avi_experiments.sh "$ROOT"
```

After reconnecting, set `ROOT` again and run the same command. Ctrl-C only stops
the watcher. Percentage is completed optimizer steps / planned steps. A
`step ... start` message is not a completed step. The first occurrence of each
objective includes compilation; compare subsequent `seconds` measurements.

```bash
tail -n 40 -F "$ROOT"/logs/*.out "$ROOT"/logs/*.err
```

The optimizer, encoder, next-step index and input contract are checkpointed
together. Random keys and epoch permutations derive deterministically from the
seed and absolute step, including teacher-bank draws. Resume therefore preserves
the learning-rate schedule and Adam moments. The last complete checkpoint can
replay work after an abrupt kill; logs after that checkpoint are removed on resume.

```bash
# Only after the old tasks have stopped. Completed tasks exit without retraining.
bash scripts/submit_feniks_avi_experiments.sh resume "$ROOT" '0-6%7'
# Or resume just an unfinished task, for example G:
bash scripts/submit_feniks_avi_experiments.sh resume "$ROOT" '6%1'
```

At eighteen hours, or upon the Slurm pre-timeout signal, training saves a
resumable `PAUSED.json` after the current step. A paused allocation can exit
successfully without having completed training: require the arm's `FINAL.json`,
not just Slurm `COMPLETED`. Simultaneous writers to one arm are blocked by a lock.

## Results And Interpretation

```bash
CODE=$(cat "$ROOT/CODE_DIR")
(cd "$CODE" && python -m scripts.feniks_avi_experiments summarize --root "$ROOT")
```

Outputs: `avi_comparison.csv`, `avi_comparison.png`, each arm's `training.csv`,
`validation_source.csv`, `validation_epoch_*.csv`, `validation_final.csv`, and
`encoder.eqx`. Validation uses all four GPUs and two fresh K512 banks per galaxy,
with fixed evaluation randomness shared across arms and epochs. Evaluation
never selects a training update or silently restores a checkpoint.
`avi_training.png` separates sleep and wake objectives. The comparison plot
includes the unchanged source as a dashed line. Preflight receipts also contain
a rough runtime estimate from the second execution of each objective, excluding
validation and I/O; it is not a completion guarantee.

For normalized importance weights `w_i`, `ESS = 1 / sum(w_i**2)` and
`ess_fraction = ESS / K`. A high maximum weight means a few draws dominate.
These diagnose the draw distribution, not a proof of complete mode coverage.
Raw predictive RMS is computed before importance weighting: a weighted fit
improving while raw draws remain poor is not successful amortization.
Negative ELBO is `mean(log q(x|y) - log p(y,x))`; lower is better for the same
galaxy and frozen target. Monte Carlo noise and missed modes still matter.

The saved encoder may be a mixture of full flows. It is **not** a drop-in
single-encoder `best.eqx` for legacy inference. Reconstruct its architecture from
the arm manifest and source model, then load `encoder.eqx`; use
`avi_experiments.log_prob` for its exact joint density. Never flatten experts
into marginal fits. All outputs remain development artifacts, without automatic
scientific promotion.

## Next Population Stage

Choose the encoder method using held-out photometry, raw predictive fit, weight
stability and independent seeds. Then learn the population prior with the
catalogue selection correction still present. This next stage must not filter
the population to only easy/high-ESS galaxies. The empirical teacher target
also changes when the prior changes, so F/G banks cannot be reused unmodified.
The current launcher intentionally does not submit that second stage.

## Local Verification

Targeted CPU tests cover MIS densities, mixture gate gradients against finite
differences, invalid weights, teacher provenance, real four-device gradient
accumulation, rollback, all seven preflights on mock physics, resume and plots.
Transport/RWS/runtime/feature regression suites pass. Real H100 and DSPS-asset
execution remain the responsibility of the submitted preflights. Legacy
AGENTS.md fit/posterior smoke configurations are absent in this checkout.
