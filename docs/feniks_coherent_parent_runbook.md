# Coherent 15D parent catalogue

## Purpose

This run fixes the dataset contract identified by the
[provenance audit](feniks_catalogue_provenance_20260925.md). It does not train
another classifier, fit a population or train an individual posterior.

Input is the saved raw weighted Diffsky proposal pool, **not** root
`all_50k.parquet`, `survey_like` or `inference_ready`. Jean-Zay file counts
256/56/59 were confirmed by the user. The jobs still verify names, schema,
identities, weights and hashes before trusting the contents.

### Scientific target

- 100000/20000/20000 parent draws from newly disjoint source pools.
- Parent means **no photometric selection**, within the declared domain of
  proposals whose metallicities were not clipped to the SSP grid. It does not
  mean the entire astrophysical population outside this physical support.
- Saved `galaxy_weight` is used exactly once when resampling raw proposals.
  Its historical astrophysical construction is unchanged, not revalidated here.
- The original value is retained as `source_proposal_weight`. Final catalogue
  `population_weight` and the compatibility `galaxy_weight` column are both one.
- The old generator uses `source_seed + shard_index`; its original split seeds
  26061701/26061702/26061703 overlap after adding shard indices. The 256 original
  train shards cover every saved effective seed. We use that canonical pool
  once, ignoring the overlapping 56/59 files, and randomly partition whole
  realizations into 183/37/36 shards before any weighted resampling. Both the
  original identity and canonical `effective_proposal_key` are retained.
  The report explicitly checks effective-proposal separation, not just prefixes.
  Within-split repeated draws are legitimate weighted sampling; unique counts
  and proposal ESS are reported. They are not extra independent latent objects.
  Each split is a finite independent estimate of the same generator population,
  not exactly the same finite empirical mixture.
- All 15 parameters remain active. Native SFHs are projected using the existing
  spline helper; physical and SFH correlations remain those of the source draws.
  SFH zero atoms are preserved and counted, not silently dequantized.
- Photometry is regenerated from those **projected** 15D parameters under the
  frozen current merged/MDF64/spline64 decoder. This intentionally defines a
  self-consistent projected-Feniks benchmark, not exact native Diffsky photometry.
- Noise uses the saved m5 law and independent Gaussian draws, including negative
  observed fluxes. All bands are observed in this synthetic mask contract.
- Only the noisy observed `lsst_r < 29` cut creates the selected catalogue.
  Rejected objects remain in the saved parent and the efficiency denominator.

Sampling is exact across shards: choose a shard in proportion to its total
eligible weight, then a row in proportion to its weight within that shard.
This produces probability proportional to the original weight over the whole
eligible pool. No giant concatenation of the raw pool is necessary.

## Execution and outputs

Three CPU sampling tasks feed 14 H100 photometry tasks, at most four concurrent.
Each photometry task handles 10000 parent objects and saves blocks of 1024 rows.
Only unfinished blocks are computed on resume. There is no expensive per-object
inference. All outputs are in a new root; source catalogues remain untouched.

Default ceilings: 30 minutes per CPU sampling task, 45 minutes per H100 task,
15 minutes for the CPU report. The full H100 allocation ceiling is 10.5 hours
summed across GPUs, not expected runtime or elapsed time. Queue wait is separate.
No explicit memory options are sent to Jean-Zay. Reports use `afterany` so a
timeout produces a partial status, not an indefinitely blocked report.

Important files:

- `MANIFEST.json`, `decoder.json`, `noise.json`, `CODE_SHA256`: frozen contracts.
- `sampling/<split>/sampling.json`: source hashes, eligible mass, ESS and counts.
- `sampling/<split>/native_parent.parquet`: sampled native parameters, unit weights.
- `photometry/task_*/block_*/parent.parquet`: resumable forward results.
- `dataset/parent/{train,validation,test}.parquet`: full unselected projected parent.
- `dataset/selected_r29/{train,validation,test}.parquet`: exact selected views.
- `dataset/CONTRACT.json`: dataset readiness and hashes of all final catalogues.
- `report/checks.json`, `report/parent_vs_selected.png`, `ROADMAP_STATUS.md`.

The gate checks finite fluxes, weights, exact selection identities, proposal
split separation, Gaussian residual moments on the **parent**, selected counts
against conditional Gaussian selection probabilities, and saved-theta decoder
replay with a different batch shape. A corrupt completed artifact fails rather
than being silently overwritten. Existing raw and failed-run outputs are kept.

`ready_for_population_benchmark` is strictly a dataset-contract verdict.
`ready_for_production` stays false. Self-consistent regeneration cannot prove
independent numerical accuracy, reference-prior support/SFH compatibility,
recoverable population shapes or posterior calibration. Those remain explicit
checks for the next population-plus-posterior run. Existing banks/classifiers
are reusable only after checking their full observation and reference contracts;
do not automatically reuse old selected identities, efficiencies or truth weights.

## Launch on Jean-Zay

Use a subshell so an error with `set -e` cannot close the interactive SSH shell.
Do not compare a full commit hash with a short hash using string equality.

```bash
(
set -euo pipefail
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine

BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
SOURCE="$PWD/Data/diffsky/synthetic/feniks_260617_dsps_closure_18band"
DECODER="$BASE/avi_sbeb_benchmark_20260917_230455/runtime/r29/source_config.yaml"
TAG=$(date +%Y%m%d_%H%M%S)
COHERENT="$BASE/avi_coherent_parent_$TAG"
printf 'export COHERENT=%q\n' "$COHERENT" > "$BASE/avi_coherent_parent_latest.env"

bash scripts/submit_feniks_coherent_parent.sh "$SOURCE" "$DECODER" "$COHERENT" \
  configs/experiments/feniks_coherent_parent.yaml |
  tee "$BASE/avi_coherent_parent_submission_$TAG.txt"
)
```

Monitor after any reconnection:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
BASE=/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111
source "$BASE/avi_coherent_parent_latest.env"
bash scripts/watch_feniks_coherent_parent.sh "$COHERENT"
```

The watcher accepts `--once`; Ctrl-C stops the display only. Once all jobs have
finished (including a partial report), resume missing work in the same root:

```bash
bash scripts/submit_feniks_coherent_parent.sh --resume "$COHERENT"
```

Resume uses the original code snapshot, not a silently changed checkout. It
does not cancel active jobs or overwrite corrupted completed data. If a genuine
code change is needed, stop and arrange an explicit provenance-preserving
recovery rather than deleting receipts or mixing implementations.

## Local validation

Eleven focused tests cover exact weighted sampling, deterministic replay, identity
and schema guards, support restriction, changing source hashes, Gaussian noise,
negative fluxes, observed selection, doubled-weight rejection, partial reports,
block-level resume, corruption detection and selective mocked SLURM resubmission.
The tests also reject Jean-Zay memory flags and check shell syntax.
Two regressions explicitly cover overlapping effective seeds and differently
named original identities referring to the same generated proposal.

A separate actual-DSPS CPU smoke projected eight existing native galaxies and
generated all 18 bands with the current decoder. Fluxes were finite. Replaying
the stored theta at batch size one versus four differed by at most 9.9e-12
reported sigma. Local .venv used existing shine's optional diffstar/diffmah
packages; no new physics or optional-package installation was performed.
This is not a benchmark of H100 throughput or a claim of scientific convergence.
