# FENIKS encoder-posterior MIRA evaluation

This workflow evaluates held-out encoder posterior samples without rerunning
the encoder or decoding posterior samples through DSPS.

## Input contract

The truth table must contain one row per held-out object, with `object_id`,
optional `row_index`, and the canonical 15 FENIKS spline parameters. Each
posterior source must contain:

- `object_id` and optional `row_index`;
- `sample_id`;
- the same 15 physical parameter columns;
- the same number of unique sample IDs for every truth object.

The source may be an inference directory containing
`posterior_samples.parquet`, an inference directory containing
`posterior_samples/batch_*.parquet`, the shard directory itself, or a
monolithic parquet file.

## Jean-Zay

The H100 wrapper compares the completed `rws_k8_t2_seed2` and
`rws_k8_t2_seed3` runs in one process. Both models therefore use exactly the
same MIRA region bank, fiducial resamples, and reference draws. One H100 is
sufficient; splitting the two seeds across devices would lose that paired
comparison.

Update the checkout and verify the input contract before submitting:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git fetch origin feature/feniks-exact-posterior-benchmark
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
mkdir -p outputs/logs

for run in rws_k8_t2_seed2 rws_k8_t2_seed3; do
  test -e "outputs/runs/feniks_selfsup_paper_v1/$run/DONE"
  test -s "outputs/runs/feniks_selfsup_paper_v1/$run/inference/inference_truth.parquet"
  test -s "outputs/runs/feniks_selfsup_paper_v1/$run/inference/posterior_shards_manifest.json"
  ls "outputs/runs/feniks_selfsup_paper_v1/$run/inference/posterior_samples"/*.parquet | wc -l
done
```

Run a bounded smoke first:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine

SMOKE_JOB=$(sbatch --parsable \
  --export=ALL,LIMIT=256,NUM_REGIONS=20,NUM_BOOTSTRAP=100,OUT_DIR=outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_smoke_256_v2 \
  scripts/feniks_mira_h100.slurm)
echo "$SMOKE_JOB"
```

Verify completion and artifacts:

```bash
sacct -X -j "$SMOKE_JOB" \
  --format=JobID%20,JobName%18,State%20,ExitCode,Elapsed,Reason%28
tail -n 80 "outputs/logs/feniks_mira-${SMOKE_JOB}.out"
test ! -s "outputs/logs/feniks_mira-${SMOKE_JOB}.err"
test -e outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_smoke_256_v2/DONE
cat outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_smoke_256_v2/mira_summary.json
```

Only after `sacct` reports `COMPLETED` with `ExitCode=0:0`, run the complete
5,000-object evaluation:

```bash
FULL_JOB=$(sbatch --parsable \
  --export=ALL,OUT_DIR=outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_v2 \
  scripts/feniks_mira_h100.slurm)
echo "$FULL_JOB"
```

Verify the full run independently of `squeue`:

```bash
sacct -X -j "$FULL_JOB" \
  --format=JobID%20,JobName%18,State%20,ExitCode,Elapsed,MaxRSS,AllocTRES%30,Reason%28
tail -n 100 "outputs/logs/feniks_mira-${FULL_JOB}.out"
test ! -s "outputs/logs/feniks_mira-${FULL_JOB}.err"

MIRA_OUT=outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_v2
test -e "$MIRA_OUT/DONE"
cat "$MIRA_OUT/mira_summary.json"
column -s, -t < "$MIRA_OUT/mira_pairwise_differences.csv" | less -S
sha256sum "$MIRA_OUT"/mira_{manifest.json,scores.parquet,object_contributions.parquet}
```

The wrapper defaults are:

```text
truth:
  outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2/inference/inference_truth.parquet
posteriors:
  rws_k8_t2_seed2=outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2/inference
  rws_k8_t2_seed3=outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed3/inference
samples per object: 128
regions per object: 100
fiducial bootstrap replicates: 1000
```

The equivalent direct evaluator command, once inside an allocated H100 job, is:

```bash
python scripts/evaluate_feniks_mira.py \
  --truth outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2/inference/inference_truth.parquet \
  --posterior rws_k8_t2_seed2=outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2/inference \
  --posterior rws_k8_t2_seed3=outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed3/inference \
  --out outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_v2 \
  --samples-per-object 128 \
  --num-regions 100 \
  --num-bootstrap 1000 \
  --seed 260730
```

## Outputs

- `mira_scores.csv` and `mira_scores.parquet`: full 15D, physical 5D, SFH
  10D, and marginal scores with direct MIRA mean/std and bootstrap mean/std;
- `mira_object_contributions.parquet`: region-averaged per-fiducial values
  retained for auditability;
- `mira_bootstrap_scores.parquet`: one score per fiducial-bootstrap replicate;
- `mira_pairwise_differences.csv`: paired model differences when multiple
  posterior sources are supplied;
- `mira_normalization*.csv`: the fixed truth min-max transform and posterior
  support diagnostics;
- `mira_manifest.json`: input hashes, seeds, JAX device, method contract, and
  provenance;
- `mira_scores.png`: joint and marginal score overview;
- `DONE`: terminal success marker.

The calibrated reference is `2/3`. For `L` truth fiducials, the paper's
theoretical per-region reference has variance `1 / (18 L)`, and the gray band
in the plot is the corresponding one-standard-deviation band
`+/-sqrt(1 / (18 L))`. The final score averages the `L_r` region contributions,
and the direct `mira` standard deviation is the scatter across those region
runs. The bootstrap follows `mira_bootstrap`: each replicate resamples `L`
fiducials and draws one region for each resampled fiducial. The plot therefore
shows bootstrap mean `+/-` bootstrap standard deviation, matching the two
values returned by `mira_bootstrap`; quantiles remain in the CSV as diagnostics.
The figure labels both `L` and `N`, where `N` is the number of posterior samples
per fiducial; `N=128` is not the number of truth galaxies. The run must use the
actual fiducial count: a `50,000`-galaxy band cannot be attached to a
`5,000`-galaxy evaluation without rerunning MIRA on 50,000 truth rows.

To regenerate only the plot from an existing score table after changing its
annotation or uncertainty display:

```bash
python scripts/plot_feniks_mira.py \
  --scores outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_v2/mira_scores.csv \
  --out outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_v2/mira_scores.png
```

Scientific uncertainty should be read from the bootstrap mean/std (with the
quantiles available for diagnostics). MIRA is a global closure diagnostic; it
does not establish
observational calibration on real galaxies without latent truth and does not
replace posterior-predictive or coverage diagnostics.
