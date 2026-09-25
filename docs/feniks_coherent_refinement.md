# Bounded coherent representation follow-up

This is a truth-trained density-capacity test, not a blind parent reconstruction.
It does not train an individual posterior, resimulate photometry, or modify the
completed dataset/reference run. All 15 coordinates remain stochastic.

## Questions and resources

| Job | What it checks | Resource ceiling |
| --- | --- | --- |
| `fit` | Does lower-rate physical training improve both validation likelihood and distribution agreement? | 1 H100, 60 minutes |
| `audit` | Are saved distribution errors central or in tails? Are exact SFH zeros sensitive to projection precision? | 4 CPU threads, 30 minutes |
| `report` | Compare before/after and retain the frozen SFH conditional in joint 15D draws | 4 CPU threads, 30 minutes |

`fit` and `audit` run in parallel. These are allocation limits, not timing promises.
No SLURM memory flags, new simulation banks, classifier training or bootstrap array.
The report uses `afterany`: failures produce an explicit blocked report, not an
indefinitely unsatisfied dependency.

## Training contract

- Load `physical/best.eqx`, never the last optimizer state. For source run
  `avi_coherent_representation_20260925_233915`, this is epoch 114.
- Same two-expert 5D spline mixture, 12 layers, 16 bins, residual context trunk;
  architecture, transform, training targets and batch size inherited unchanged.
- Reset Adam once at the selected best checkpoint; constant LR `3e-6`, at most
  60 additional epochs. Existing checkpoints resume both model and optimizer.
- Keep `sfh_conditional/best.eqx` frozen (source best epoch 116). It is only
  loaded for final joint sampling. Changing the physical marginal can still
  change the *unconditional* SFH distribution through the frozen conditional.
- Reuse the original fixed validation subset and seed. Best checkpoint selection
  uses validation NLL, not test metrics. Evaluate the initial checkpoint too, so
  a deteriorating continuation cannot replace it as the best-NLL model.
- At epochs 0, 20, 40, 60 retain best-so-far checkpoints and 20k physical draws.
  Common random seeds and 64 fixed projection directions support paired comparisons.
  Report validation SW in theta and normalized x, marginal W1 and signed biases.
- Evaluate the test split only in the final report. That split has already been
  inspected in earlier experiments; it is not a pristine paper-confirmation set.
- Budget exhaustion is not a convergence certificate. No automated production
  promotion or claim that 60 more epochs must solve the residual mismatch.

## Tail and zero audit

The CPU job reads the existing full joint draws and test truth. W1 is the integral
of the absolute quantile difference. Its contributions over probability intervals
`[0,.01]`, `[.01,.99]`, `[.99,1]` sum to the **full empirical W1**, even for unequal
sample counts. There is no trimming/reweighting. Signed mean/median bias, tail
probabilities, extrema and quantiles are saved. Central plot panels use counts
divided by the full sample size and bin width, not a renormalized cropped histogram.

Replay the saved 1024 training native objects with the original projection and
with opt-in float64 inputs/parameters/time/knots. The production/default float32
path remains unchanged. SFH floor attribution now uses the upstream `1e-14` SFR
floor (log10=-14), not the later protective log floor `1e-30`. Count floor zeros,
zeros that disappear in float64, and zeros that persist. Persistent zeros do not
prove physical atoms; spline sampling and the native model can also make plateaus.
Original targets are never replaced, jittered, clipped or removed. This does not
yet quantify the photometric impact of SFH-density errors.

## Launch on Jean-Zay

Run from a new/reconnected shell. Strict mode is inside a subshell so an error
does not close the SSH session. Source paths are explicit, not guessed from `find`.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine

BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
REP="$BASE/avi_coherent_representation_20260925_233915"
REFINE="$BASE/avi_coherent_refinement_$(date +%Y%m%d_%H%M%S)"
printf 'export REFINE=%q\n' "$REFINE" > "$BASE/avi_coherent_refinement_latest.env"
bash scripts/submit_feniks_coherent_refinement.sh "$REP" "$REFINE"
)
```

Watcher, including after a disconnect:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_coherent_refinement_latest.env"
bash scripts/watch_feniks_coherent_refinement.sh "$REFINE"
```

Add `--once` for one display. A terminal disconnect does not cancel SLURM jobs.
On timeout/failure, inspect `logs/*.err` and `sacct` first. Once all previous jobs
have stopped, reactivate `shine` and run:

```bash
bash scripts/submit_feniks_coherent_refinement.sh --resume "$REFINE"
```

This preserves completed stages/milestones and resumes the physical optimizer
from the last completed epoch. Source/checkpoint hashes are rechecked. Never
delete the run to retry. Code is snapshotted; resume does not silently adopt
changed checkout code or settings.

## Artifacts and decisions

Read in this order:

1. `report/physical_before_after.png`: true, source-best, continued-best physical
   marginals, central and full ranges.
2. `report/continuation.png`, `report/validation_trajectory.csv`: likelihood and
   distribution progression; an improving NLL with worsening SW remains a warning.
3. `report/tails_and_zeros.png`, `audit/saved_draw_tail_bias.csv`,
   `audit/sfh_precision.csv`: central/tail discrepancy and corrected zero origins.
4. `report/physical_corner_theta.png`, `physical_corner_x.png`,
   `joint15_metrics.csv`: joint structure with unchanged conditional SFH.
5. `report/physical_joint.csv`, `physical_tail_bias.csv`: paired before/after test
   comparison, signed errors, and validation/test empirical disagreement.

For a small rsync, retrieve only `report/*.png`, `report/*.csv`, `report/*.md`,
`report/FINAL.json`, `audit/*.csv`, `audit/FINAL.json`, `physical/training.jsonl`,
and manifests/configuration/logs. Exclude `*.eqx`, `*.npz`, `*.npy` and code archives.

Roadmap: this addresses **representation and SFH-target diagnosis only**. A good
result does not validate the independent parent reference, blind selection-aware
population recovery, or individual posterior calibration. Next decisions depend
on the measured residual tails/zeros and their observable impact, not only NLL.

## Local verification environment

Use `.venv/bin/python` with `JAX_PLATFORMS=cpu`, `EUCLID_DSPS_JAX_PLATFORMS=cpu`,
`EUCLID_DSPS_REQUIRE_GPU=0`, `EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1`,
`JAX_ENABLE_X64=true`, `OPENBLAS_NUM_THREADS=4`, `OMP_NUM_THREADS=4` for tests.
The native 32-object smoke required the pure-Python Diffstar/Diffmah packages
from the local shine installation on `sys.path` after importing the .venv's
JAX/DSPS/jax_cosmo; no environment or dependency files were modified. Jean-Zay
uses the existing complete `shine` environment. Local fixture metrics are
implementation checks, not scientific results for the remote run.

Verification: 66 distinct focused tests pass; two optional native-DSPS tests
skip without their optional local dependencies. The separate actual native
32-object replay confirms bitwise parity of the default path. Historical CLI
one-row/batch fit commands cannot start because
`configs/diffsky_hltds_04_14_simple_gpu.yaml` is absent from this checkout.
No remote training was executed during implementation.
