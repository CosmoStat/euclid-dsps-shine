# Pop-COSMOS RWS redshift calibration

This workflow compares the `bands26/full_cont120` and
`bands24_no_irac/full_cont120` RWS redshift posteriors with paired MIRA and
TARP diagnostics. Both methods use the same 1,395 COSMOS evaluation objects
with finite public spectroscopy, the same object order, and all 128 posterior
draws per object.

The remaining 3,605 evaluation objects have no finite redshift truth and are
excluded before any smoke-test limit is applied. No calibration claim is made
for those objects or for the complete photometric catalog.

The public Pop-COSMOS `summaries.txt` table contains posterior quantiles, not
dense draws. It remains in the same-cohort photo-z metric comparison but is not
assigned a MIRA or TARP value. The public `chains.h5` product can support such
an evaluation later, but its compressed download is approximately 44 GB.

## Jean-Zay

Run from the checkout that owns the Pop-COSMOS outputs:

```bash
cd /lustre/fswork/projects/rech/jrx/urx63nr/dsps-popcosmos
git fetch origin feature/feniks-exact-posterior-benchmark
git switch feature/feniks-exact-posterior-benchmark
git pull --ff-only origin feature/feniks-exact-posterior-benchmark
mkdir -p outputs/logs

RUN_ROOT=outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536
for variant in bands26 bands24_no_irac; do
  INFERENCE="$RUN_ROOT/$variant/full_cont120/inference"
  test -e "$RUN_ROOT/$variant/full_cont120/DONE"
  test -s "$INFERENCE/inference_truth.parquet"
  test -s "$INFERENCE/posterior_shards_manifest.json"
  find "$INFERENCE/posterior_samples" -maxdepth 1 -name '*.parquet' | wc -l
done
```

Run a 256-spec-z smoke first:

```bash
SMOKE_OUT="$RUN_ROOT/redshift_calibration_full_cont120_smoke256_v1"
SMOKE_JOB=$(sbatch --parsable \
  --export=ALL,LIMIT=256,NUM_REGIONS=20,NUM_BOOTSTRAP=100,NUM_ALPHA_BINS=0,SAMPLES_PER_OBJECT=128,SEED=260805,OUT_DIR="$SMOKE_OUT" \
  scripts/popcosmos_redshift_calibration_h100.slurm)
echo "$SMOKE_JOB"
```

Verify the smoke independently of `squeue`:

```bash
sacct -X -j "$SMOKE_JOB" \
  --format=JobID%20,JobName%20,State%20,ExitCode,Elapsed,Reason%28
tail -n 120 "outputs/logs/popcosmos_zcal-${SMOKE_JOB}.out"
test ! -s "outputs/logs/popcosmos_zcal-${SMOKE_JOB}.err"
test -e "$SMOKE_OUT/DONE"
cat "$SMOKE_OUT/mira/mira_summary.json"
cat "$SMOKE_OUT/tarp/tarp_summary.json"
```

Only after `COMPLETED` and `ExitCode=0:0`, run the complete 1,395-object
spectroscopic cohort:

```bash
CAL_OUT="$RUN_ROOT/redshift_calibration_full_cont120_v1"
CAL_JOB=$(sbatch --parsable \
  --export=ALL,LIMIT=,NUM_REGIONS=100,NUM_BOOTSTRAP=1000,NUM_ALPHA_BINS=0,SAMPLES_PER_OBJECT=128,SEED=260805,OUT_DIR="$CAL_OUT" \
  scripts/popcosmos_redshift_calibration_h100.slurm)
echo "$CAL_JOB"
```

The full job also reads the completed FENIKS redshift diagnostics from:

```text
/lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine/outputs/runs/feniks_selfsup_paper_v1/mira_rws_k8_t2_seed2_seed3_v2
/lustre/fswork/projects/rech/jrx/urx63nr/euclid-dsps-shine/outputs/runs/feniks_selfsup_paper_v1/tarp_rws_k8_t2_seed2_seed3_v1
```

Override `FENIKS_ROOT` at submission if those artifacts were moved.
When `redshift_only_comparison_v1/redshift_method_metrics.csv` is present, its
same-cohort RWS/Pop-COSMOS photo-z metrics are also joined into the comparison
table; otherwise the job records a warning and still completes MIRA/TARP.

Verify the full run and its contract:

```bash
sacct -X -j "$CAL_JOB" \
  --format=JobID%20,JobName%20,State%20,ExitCode,Elapsed,MaxRSS,AllocTRES%30,Reason%28
tail -n 160 "outputs/logs/popcosmos_zcal-${CAL_JOB}.out"
test ! -s "outputs/logs/popcosmos_zcal-${CAL_JOB}.err"
test -e "$CAL_OUT/DONE"

python - "$CAL_OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
mira = json.loads((out / "mira/mira_summary.json").read_text())
tarp = json.loads((out / "tarp/tarp_summary.json").read_text())
for result in (mira, tarp):
    assert result["status"] == "complete"
    assert result["models"] == ["rws26", "rws24"]
    assert result["num_objects"] == 1395
    assert result["num_posterior_samples"] == 128
    assert result["selected_sample_ids"] == list(range(128))
    assert result["score_groups"] == ["marginal_z_obs"]
    assert result["primary_group"] == "marginal_z_obs"
    assert result["jax_backend"] == "gpu"
assert mira["num_regions"] == 100
assert mira["num_bootstrap"] == 1000
assert tarp["num_alpha_bins"] == 139
assert tarp["num_bootstrap"] == 1000
assert (out / "comparison_with_current/DONE").exists()
print("PASS")
print("MIRA:", mira["primary"])
print("TARP:", tarp["primary"])
PY

column -s, -t < "$CAL_OUT/mira/mira_pairwise_differences.csv" | less -S
column -s, -t < "$CAL_OUT/tarp/tarp_pairwise_differences.csv" | less -S
column -s, -t < "$CAL_OUT/comparison_with_current/redshift_calibration_comparison.csv" | less -S
```

## Interpretation

- MIRA target: `2/3`; use the bootstrap interval and the paired
  `rws26-rws24` difference.
- TARP target: `ECP = alpha` and `ATC = 0`; inspect the curve as well as ATC.
- The two RWS methods are paired on exactly the same objects and draws, so
  their difference is directly interpretable.
- FENIKS versus COSMOS is not paired: it contrasts a synthetic held-out
  closure cohort with a spectroscopy-selected real cohort and must be reported
  as descriptive context, not as a direct ranking.
