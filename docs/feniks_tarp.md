# FENIKS encoder-posterior TARP evaluation

This workflow runs TARP on exactly the same held-out FENIKS truth rows and
encoder posterior parquet files as the completed MIRA v2 evaluation. It uses
the same deterministic loader and selects the first 128 sorted `sample_id`
values for every object and both RWS seeds.

The implementation follows the distance-rank and expected-coverage curve from
[`tarp==0.1.1`](https://github.com/Ciela-Institute/tarp/tree/0.1.1), evaluated
with JAX on one H100. It uses one uniform reference point per held-out object,
shared between the two models, and truth min-max normalization. The central
curve is the upstream TARP statistic. Its confidence band and the
seed2-minus-seed3 ATC interval use a paired bootstrap over held-out objects,
keeping each truth/posterior/reference tuple together. The method is described
in the [TARP paper](https://arxiv.org/abs/2302.03026).

MIRA does not need to be rerun. This is a separate evaluation of its existing
inputs.

## Jean-Zay

Update the checkout and verify the completed inference inputs:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine
git fetch origin feature/feniks-exact-posterior-benchmark
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
mkdir -p outputs/logs

PAPER_ROOT=outputs/runs/feniks_selfsup_paper_v1
for run in rws_k8_t2_seed2 rws_k8_t2_seed3; do
  test -e "$PAPER_ROOT/$run/DONE"
  test -s "$PAPER_ROOT/$run/inference/inference_truth.parquet"
  test -s "$PAPER_ROOT/$run/inference/posterior_shards_manifest.json"
  find "$PAPER_ROOT/$run/inference/posterior_samples" \
    -maxdepth 1 -type f -name '*.parquet' | sort
done
```

Run a 256-object smoke test first. With `NUM_ALPHA_BINS=0`, TARP uses its
default `L // 10`, hence 25 bins for this smoke test:

```bash
TARP_SMOKE_OUT="$PAPER_ROOT/tarp_rws_k8_t2_seed2_seed3_smoke_256_v1"
TARP_SMOKE_JOB=$(sbatch --parsable \
  --export=ALL,LIMIT=256,NUM_BOOTSTRAP=100,NUM_ALPHA_BINS=0,SAMPLES_PER_OBJECT=128,SEED=260730,OUT_DIR="$TARP_SMOKE_OUT" \
  scripts/feniks_tarp_h100.slurm)
echo "$TARP_SMOKE_JOB"
```

Do not infer completion from `squeue`. Check Slurm, logs, and artifacts:

```bash
sacct -X -j "$TARP_SMOKE_JOB" \
  --format=JobID%20,JobName%18,State%20,ExitCode,Elapsed,Reason%28
tail -n 100 "outputs/logs/feniks_tarp-${TARP_SMOKE_JOB}.out"
test ! -s "outputs/logs/feniks_tarp-${TARP_SMOKE_JOB}.err"
test -e "$TARP_SMOKE_OUT/DONE"
cat "$TARP_SMOKE_OUT/tarp_summary.json"
```

Only after the smoke reports `COMPLETED` and `ExitCode=0:0`, submit the full
5,000-object evaluation. Its upstream-default alpha grid has 500 bins:

```bash
TARP_OUT="$PAPER_ROOT/tarp_rws_k8_t2_seed2_seed3_v1"
TARP_JOB=$(sbatch --parsable \
  --export=ALL,LIMIT=,NUM_BOOTSTRAP=1000,NUM_ALPHA_BINS=0,SAMPLES_PER_OBJECT=128,SEED=260730,OUT_DIR="$TARP_OUT" \
  scripts/feniks_tarp_h100.slurm)
echo "$TARP_JOB"
```

Verify the full run:

```bash
sacct -X -j "$TARP_JOB" \
  --format=JobID%20,JobName%18,State%20,ExitCode,Elapsed,MaxRSS,AllocTRES%30,Reason%28
tail -n 120 "outputs/logs/feniks_tarp-${TARP_JOB}.out"
test ! -s "outputs/logs/feniks_tarp-${TARP_JOB}.err"
test -e "$TARP_OUT/DONE"

python - "$TARP_OUT" "$PAPER_ROOT/mira_rws_k8_t2_seed2_seed3_v2" <<'PY'
import json
import sys
from pathlib import Path

tarp_out = Path(sys.argv[1])
mira_out = Path(sys.argv[2])
tarp = json.loads((tarp_out / "tarp_summary.json").read_text())
tarp_manifest = json.loads((tarp_out / "tarp_manifest.json").read_text())
mira_manifest = json.loads((mira_out / "mira_manifest.json").read_text())

assert tarp["status"] == "complete"
assert tarp["models"] == ["rws_k8_t2_seed2", "rws_k8_t2_seed3"]
assert tarp["num_objects"] == 5000
assert tarp["num_posterior_samples"] == 128
assert tarp["num_alpha_bins"] == 500
assert tarp["num_bootstrap"] == 1000
assert tarp["jax_backend"] == "gpu"
assert tarp["companion_truths_checked"] == 2
assert tarp["selected_sample_ids"] == list(range(128))

# Exact same truth and posterior parquet bytes as MIRA v2.
assert tarp_manifest["truth_file"]["sha256"] == mira_manifest["truth_file"]["sha256"]
assert tarp_manifest["truth_file"]["sha256"] == (
    "3e8cc54bb4bd6f5de85fb92ed8184028aa1fa9b45528574327b9143bf82d7cec"
)
for model in tarp["models"]:
    tarp_files = [item["sha256"] for item in tarp_manifest["posterior_files"][model]]
    mira_files = [item["sha256"] for item in mira_manifest["posterior_files"][model]]
    assert tarp_files == mira_files

print("PASS: TARP complete and byte-identical inputs to MIRA v2")
for row in tarp["full_15d"]:
    print(
        row["model"],
        "ATC=", row["atc"],
        "CI95=", (row["bootstrap_atc_q025"], row["bootstrap_atc_q975"]),
        "KS-p=", row["ks_pvalue"],
    )
PY

column -s, -t < "$TARP_OUT/tarp_pairwise_differences.csv" | less -S
sha256sum "$TARP_OUT"/tarp_{manifest.json,coverage.parquet,coverage_values.parquet}
```

The wrapper defaults are:

```text
truth objects L: 5000
posterior samples per object N: 128
models: rws_k8_t2_seed2 and rws_k8_t2_seed3
alpha bins: L // 10 = 500
paired object bootstrap replicates: 1000
random seed: 260730
device: one H100
```

## Outputs and interpretation

- `tarp_coverage.csv/parquet`: ECP curves and pointwise bootstrap bands;
- `tarp_summary.csv/json`: ATC, KS p-value, curve errors, and ATC intervals;
- `tarp_coverage_values.parquet`: per-object TARP rank fractions;
- `tarp_pairwise_differences.csv`: paired seed2-minus-seed3 ATC intervals;
- `tarp_normalization*.csv`: normalization and posterior-support diagnostics;
- `tarp_manifest.json`: hashes, selected sample IDs, seeds, method, and git SHA;
- `tarp_coverage.png`: joint, block, and marginal expected-coverage curves;
- `DONE`: terminal success marker.

The calibrated target is the diagonal `ECP = alpha`. An ATC close to zero is
the scalar summary used here. Positive ATC is consistent with a posterior that
is too wide, while negative ATC is consistent with one that is too narrow;
bias can produce similar curve deviations, so the joint, block, and marginal
curves must be inspected rather than reducing the conclusion to one number.
The colored regions are paired object-bootstrap uncertainty bands for each
measured curve. They are not a theoretical uncertainty band around the ideal
diagonal.
