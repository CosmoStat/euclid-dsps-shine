# Fast parallel population/decoder audit

Implements the first diagnostic tranche of the population validation roadmap.
The known-parent oracle is evaluation-only. No classifier, prior or individual
posterior is trained. No extra photometric bands are added.

## Dependency graph

```text
                       cached ratios + four full fits (1 H100, <=25 min)
                       /                                            \
two metric tasks (CPU, <=15 min each)      eight paired-bootstrap tasks (CPU)
                                         four simultaneous, <=15 min each

decoder replay: baseline / merged quadrature (1 H100, <=25 min), independent

                 report after all jobs terminate, including failures (CPU)
```

Peak GPU usage is two H100s, with a summed GPU allocation ceiling of 50 minutes.
CPU metric/bootstrap/report allocation ceiling is 155 job-minutes at four CPU
threads each. Queue wait is not included; these are caps, not runtime promises.
The direct bootstrap critical path is capped at 25 + 2*15 + 5 = 60 minutes of
allocated execution if both waves obtain resources immediately. A fit has an
additional 120-second optimizer budget checked during objective evaluations.

Speed comes from computing the affine low-rank likelihood projection once per
fit. Its objective, positivity constraints and KKT tolerance are unchanged.
Only one rank/penalty pair per photometric arm is frozen from the previous
heldout maximum. The exact-ratio oracle uses the same family and penalty as
each photometric arm; it is not a penalty-free optimum or an information bound.
Alpha is fixed to the saved simulation estimate, not resampled.

A local four-thread timing check (50,000 objects, 128 components, 64 modes)
compared the previous implementation with the projected objective: 38.016 s
versus 0.474 s, both 15 optimizer iterations, both passing KKT at 2e-6.
Maximum selected-weight difference was 2.46e-10. This is a synthetic solver
benchmark, not a guarantee for the cluster or total pipeline runtime.

Numerical metrics compare 4096, 16384 and 65536 common-random-number draws with
three seeds, fixed IQR scaling and 64 fixed projection directions. Bootstrap
uses the same full catalogue size and the same counts across all four fits.
Eight repeats diagnose sampling variation; they do not certify final uncertainty.
Increasing catalogue size is a later conditional decision, not part of this run.
The historical 0.015 line remains visible as a reference, not an automatic
pass/fail after changing metric precision. The frozen true-parent IQR scale is
shared across this diagnostic; older plots scaled each comparison by its own
reference sample. Evaluate their comparability before revising any accuracy gate.

The decoder screen uses 128 deterministic positions in the previous audit
cohort and batches of eight. It compares the inherited integration rule to
`merged_gauss4_v1` on the same latent truths and reports catalogue residuals,
selection identities and conditional-noise summaries. It does not certify
gradients, all catalogue strata, or the complete observation model. Do not
promote the merged rule merely because its residuals decrease in this screen.

## Launch

Use a subshell so an error does not close the interactive SSH session:

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$BASE/avi_population_low_rank_audit_20260924_190751"
PRECISION="$BASE/avi_population_precision_$(date +%Y%m%d_%H%M%S)"
printf 'export PRECISION=%q\n' "$PRECISION" > "$BASE/avi_population_precision_latest.env"
bash scripts/submit_feniks_population_precision.sh "$SOURCE" "$PRECISION"
)
```

The submitter uses the repository's existing jrx@cpu/cpu_p1 and jrx@h100/gpu_p6
accounts. `FENIKS_CPU_ACCOUNT`, `FENIKS_CPU_PARTITION`, `FENIKS_GPU_ACCOUNT` and
`FENIKS_GPU_PARTITION` allow explicit site overrides. Resources are configurable
in `configs/experiments/feniks_population_precision.yaml` before preparation.
Jean-Zay rejects explicit `--mem`, `--mem-per-cpu` and `--mem-per-gpu` options.
The launcher leaves memory assignment to the site allocation policy; it does
not promise a fixed RAM capacity. `FENIKS_CPU_MEM` and `FENIKS_GPU_MEM` are unused.

## Monitor and resume

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_population_precision_latest.env"
bash scripts/watch_feniks_population_precision.sh "$PRECISION"
```

The watcher takes `--once`. Ctrl-C affects only the display. Each completed
bootstrap fit has its own weights, solver certificate and artifact receipt.
Decoder output is saved after every batch. A failed fit stays failed; the final
report explicitly lists partial completion and never substitutes missing values.

After all previous jobs have terminated, resume only incomplete work:

```bash
bash scripts/submit_feniks_population_precision.sh --resume "$PRECISION"
```

Resume uses the same immutable code snapshot. It verifies completed artifact
hashes, skips finished cells and reuses saved fits if only metric work was
interrupted. Each submission attempt has a separate `JOBS_<timestamp>.env`.
Reports depend on `afterany`; invalid downstream dependencies are cancelled by
SLURM instead of remaining pending indefinitely.

If preparation succeeded but the first `sbatch` was rejected for explicit
memory flags, update the checkout and use `--resume` on the same root. The
current checkout's launcher supplies the corrected submission options, while
the scientific code still comes from the original `CODE_DIR`. No snapshot,
manifest, bank or checkpoint needs to be recreated. A missing `JOBS.env` is
expected when no job was accepted.

## Results and roadmap tracking

- `ROADMAP_STATUS.md` and `ROADMAP_STATUS.json`: execution completion and deferred
  scientific stages, regenerated at report time (live progress is in the watcher).
- `report/precision_decoder_summary.png`: numerical precision, paired bootstrap,
  and decoder residuals in one figure.
- `report/metric_precision.csv`, `report/bootstrap_summary.csv`: numerical values.
- `bootstrap/repeat_NNN/<arm>_<ratio>/`: independently reusable fit artifacts.
- `decoder/bands.csv`, `decoder/residuals.csv`, `decoder/noise.csv`: flux audit.
- `report/FINAL.json`: completion and metric-resolution decision; production is
  always false for this diagnostic campaign.

The durable project roadmap is `docs/feniks_population_validation_roadmap.md`;
this run addresses steps 1 and 2 only. Joint physical ambiguity and posterior
calibration are later stages. Scientific results are pending the Jean-Zay run.

## Local verification

The precision workflow, low-rank solver, selection/population regression tests,
ratio ladder and decoder qualification tests pass: 52 tests in the repository
`.venv` with CPU JAX and x64 enabled. They include simulated SLURM dependencies,
selective recovery, end-to-end cached fitting/reporting and the full synthetic
SED/gradient smoke. A generated report figure was visually checked. Compileall,
Ruff and Bash syntax checks pass. The separate local conda `shine` environment
lacks `jax_cosmo` for the broad decoder tests; no cluster or real-catalogue
qualification is inferred from these local checks.
