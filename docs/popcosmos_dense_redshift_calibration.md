# Pop-COSMOS dense redshift calibration

This workflow compares RWS-26, RWS-24, and the public Pop-COSMOS posterior
chains on the exact same 1,395 public spectroscopic redshifts already frozen in
`feniks_popcosmos_recap_v1`.

## Scientific contract

- Truth is restricted to public COSMOS DR1.1 spectroscopy in the existing
  paired benchmark.
- The join key is the exact Farmer `object_id` / Pop-COSMOS `index_cosmos`.
- The primary comparison uses 128 posterior draws per object for every method.
  Pop-COSMOS draws are selected deterministically across the complete 2,560-draw
  flattened chain rather than taking one contiguous segment.
- MIRA regions, TARP references, and object bootstrap resamples are shared
  across methods.
- A Pop-COSMOS-only 2,560-draw sensitivity run is produced separately.
- The result establishes redshift calibration conditional on the public
  spectroscopic selection. It is not a 16-dimensional real-galaxy closure
  test.

## Jean-Zay submission

The 43.9 GB Zenodo archive is downloaded and extracted only on `prepost`.
The dependent H100 job performs no network access.

```bash
export REPO_DIR="$WORK/dsps-popcosmos"
export MINICONDA_PATH="$WORK/miniconda3"
export CONDA_ENV=shine
export RUN_ROOT="outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536"
export ASSET_ROOT="$SCRATCH/popcosmos_v1p1_chains"
export OUT_DIR="$RUN_ROOT/popcosmos_chain_redshift_calibration_v1"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"

df -h "$SCRATCH"
bash scripts/submit_popcosmos_chains_redshift_calibration.sh
```

The submission writes `outputs/logs/popcosmos_chains_latest.env`. The download
uses HTTP range resume and verifies the official Zenodo MD5 before extraction.
Re-running the submission reuses the validated archive, extracted HDF5, and
redshift Parquet when their completion markers exist.

## Monitoring

```bash
cd "$WORK/dsps-popcosmos"
source outputs/logs/popcosmos_chains_latest.env

echo "PREPOST_JOB=$PREPOST_JOB"
echo "EVALUATION_JOB=$EVALUATION_JOB"
squeue -j "$PREPOST_JOB,$EVALUATION_JOB"

tail -F "outputs/logs/popcosmos_chains-${PREPOST_JOB}.out"
```

After the prepost job completes:

```bash
sacct -X -j "$PREPOST_JOB,$EVALUATION_JOB" \
  --format=JobID,JobName%24,Partition,State,Elapsed,Timelimit,ExitCode,NodeList

tail -n 200 "outputs/logs/popcosmos_chaincal-${EVALUATION_JOB}.out"
test ! -s "outputs/logs/popcosmos_chaincal-${EVALUATION_JOB}.err"
```

## Completion gates

```bash
source outputs/logs/popcosmos_chains_latest.env

for path in \
  "$ASSET_ROOT/PREPOST_COMPLETE.json" \
  "$POSTERIOR_OUT/DONE" \
  "$POSTERIOR_OUT/extraction_manifest.json" \
  "$CALIBRATION_OUT/DONE" \
  "$CALIBRATION_OUT/mira_128/mira_summary.json" \
  "$CALIBRATION_OUT/tarp_128/tarp_summary.json" \
  "$CALIBRATION_OUT/pit_coverage_128/redshift_calibration_summary.json" \
  "$CALIBRATION_OUT/mira_popcosmos_all2560/mira_summary.json" \
  "$CALIBRATION_OUT/tarp_popcosmos_all2560/tarp_summary.json"; do
  [[ -s "$path" || -e "$path" ]] \
    && echo "OK      $path" \
    || echo "MISSING $path"
done
```

## Recovery after a completed download

If the prepost job completed but the dependent H100 job failed, preserve the
download and submit only the offline evaluation:

```bash
source outputs/logs/popcosmos_chains_latest.env
export SKIP_PREPOST=1
export OUT_DIR="${CALIBRATION_OUT}_retry_$(date +%Y%m%d_%H%M%S)"
bash scripts/submit_popcosmos_chains_redshift_calibration.sh
```

## Retrieval without the large chain files

Run locally from WSL:

```bash
REMOTE_HOST="urx63nr@jean-zay.idris.fr"
REMOTE_RUN="/lustre/fswork/projects/rech/jrx/urx63nr/dsps-popcosmos/outputs/runs/popcosmos_native15d_rws_k8_array_20260803_162536/popcosmos_chain_redshift_calibration_v1"
LOCAL_RECAP="/home/maxime/src/DSPS/outputs/comparisons/feniks_popcosmos_recap_v1/popcosmos_dense_chain_calibration"

mkdir -p "$LOCAL_RECAP"
rsync -avh --progress --partial \
  --exclude='*.h5' \
  --exclude='*.h5.zip' \
  -e "ssh -J mr287471@hubble.extra.cea.fr" \
  "$REMOTE_HOST:$REMOTE_RUN/" \
  "$LOCAL_RECAP/"
```
