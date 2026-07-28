# Pop-COSMOS A24 comparison with joint RWS

This workflow applies the repository's joint RWS method to the public
COSMOS2020 Farmer v2.1 photometry used by Alsing et al. (2024). It does not
retrain the Pop-COSMOS neural ODE. The comparison is:

- same public catalog version and 26-band order;
- same conservative mask, galaxy flag, and corrected `r < 25` selection;
- DSPS forward model and per-object Farmer errors;
- conditional-flow posterior plus jointly learned RealNVP population prior;
- optional adjusted MCLMC under the learned prior;
- public A24 checkpoint and posterior summaries retained as reference artifacts.

The first result is a model of the **selected catalog distribution**. Do not
claim an intrinsic, selection-corrected galaxy population until the likelihood
contains the explicit selection normalization.

## Local end-to-end smoke

Install the local method and the optional sampler:

```bash
conda activate shine
python -m pip install -e '.[amortized,samplers]'
```

Download a small live ESO sample and all 26 SVO curves:

```bash
python scripts/download_cosmos2020_assets.py \
  --out Data/cosmos2020/assets \
  --max-rows 512 \
  --skip-external-repo \
  --skip-zenodo

python scripts/prepare_cosmos2020_farmer.py \
  --input Data/cosmos2020/assets/cosmos2020_farmer_v21_top512.fits \
  --out Data/cosmos2020/prepared \
  --sizes 16,32,64

python scripts/validate_cosmos2020_reproduction.py \
  --data-dir Data/cosmos2020/prepared \
  --asset-dir Data/cosmos2020/assets
```

The selected row count in a `TOP 512` TAP response is intentionally small and
is not expected to equal 512. Use its generated `farmer_a24_full.parquet` for
the local smoke:

```bash
EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1 \
python -m euclid_dsps.cli \
  --config configs/experiments/popcosmos_a24_rws_joint.yaml \
  amortized-train-cosmos \
  --runtime cpu \
  --dataset Data/cosmos2020/prepared/farmer_a24_full.parquet \
  --out outputs/runs/dev_popcosmos_local \
  --batch-size 2 --jax-batch-size 2 \
  --epochs 1 --n-samples 1 \
  --validation-fraction 0 --no-progress

EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1 \
python -m euclid_dsps.cli \
  --config configs/experiments/popcosmos_a24_rws_joint.yaml \
  amortized-infer-cosmos \
  --runtime cpu \
  --dataset Data/cosmos2020/prepared/farmer_a24_full.parquet \
  --checkpoint outputs/runs/dev_popcosmos_local/checkpoints/best.eqx \
  --feature-stats outputs/runs/dev_popcosmos_local/feature_stats.json \
  --out outputs/runs/dev_popcosmos_local/inference \
  --limit 2 --batch-size 2 --jax-batch-size 2 \
  --posterior-samples 2 --prior-samples 4 \
  --decoder-sample-chunk-size 1 --prior-predictive-batch-size 2 \
  --selection-mode sequential --no-residual-samples
```

## Jean-Zay data and installation job

The FSPS/DSPS spectral grids are generated assets and are intentionally not
stored in Git. Before submitting the download job, transfer the five exact
runtime files from this checkout:

```bash
REMOTE=your_login@jean-zay.idris.fr
REMOTE_WORK=$(ssh "$REMOTE" 'printf %s "$WORK"')
REMOTE_REPO="$REMOTE_WORK/euclid-dsps-shine"
rsync -avP \
  Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5 \
  Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 \
  Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5 \
  Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5 \
  Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5 \
  "$REMOTE:$REMOTE_REPO/Data/"
```

`validate_cosmos2020_reproduction.py` checks their pinned SHA-256 digests, so
an incomplete or different grid fails before training. Run Internet downloads
only on `prepost`. From the repository checkout on Jean-Zay:

```bash
conda activate shine
mkdir -p outputs/logs
prepost_job=$(sbatch --parsable scripts/cosmos2020_prepost.slurm)
echo "$prepost_job"
squeue -j "${prepost_job%%;*}"
```

The job installs this checkout, clones Pop-COSMOS at commit
`28690aab5ae1aeca01db1ceaf7bc7fe2a58378a7`, downloads the ESO TAP table, the
26 SVO curves, and Zenodo record `13820043`, then builds deterministic nested
subsets. It fails if the A24 selection does not yield exactly 140,938 rows.

After it leaves `squeue`, verify terminal state and artifacts:

```bash
job="${prepost_job%%;*}"
sacct -j "$job" --format=JobID,State,Elapsed,AllocTRES,ExitCode
test -s Data/cosmos2020/assets/DOWNLOAD_COMPLETE.json
test -s Data/cosmos2020/prepared/PREPOST_COMPLETE.json
python scripts/validate_cosmos2020_reproduction.py \
  --data-dir Data/cosmos2020/prepared \
  --asset-dir Data/cosmos2020/assets \
  --expected-full 140938
```

Do not submit H100 jobs unless `sacct` reports `COMPLETED` and all three
commands above pass.

## Staged H100 training

Submit only the 512-object smoke:

```bash
bash scripts/submit_cosmos2020_reproduction.sh smoke
```

After each stage, inspect the log, `training_collapse_gate.json`, inference
diagnostics, `a24_comparison/comparison_summary.json`, and Slurm accounting.
The comparison script sky-matches the shared Farmer objects, reports normalized
photo-z bias/NMAD/outliers/68% coverage on the public spectroscopic subset, and
summarizes RWS-minus-A24 posterior-median differences. Promote one stage at a
time:

```bash
bash scripts/submit_cosmos2020_reproduction.sh n5k
bash scripts/submit_cosmos2020_reproduction.sh n20k
bash scripts/submit_cosmos2020_reproduction.sh n40k
bash scripts/submit_cosmos2020_reproduction.sh full
```

Each resume command requires the preceding `DONE` marker and submits only one
new stage. To submit a dependency chain in one call from a fresh output root,
pass the last desired stage, for example:

```bash
ROOT_DIR=outputs/runs/popcosmos_a24_rws_v2 \
bash scripts/submit_cosmos2020_reproduction.sh n20k
```

The configured ladder is:

| stage | rows | epochs | planning cap |
|---|---:|---:|---:|
| smoke | 512 | 2 | 0.5 H100 h |
| n5k | 5,000 | 4 | 1 H100 h |
| n20k | 20,000 | 8 | 4 H100 h |
| n40k | 40,000 | 20 | 12 H100 h |
| full | 140,938 | 40 | 35-70 H100 h |

These are allocation caps, not measured runtimes. Use `sacct` from each
completed stage to revise the next request. Every stage warm-starts from the
previous checkpoint but recomputes feature statistics and the data split on its
larger nested sample. The first three stages use one H100. `n40k` and `full`
request four H100s and switch training to the existing local `pmap` path; the
20-hour full wall time therefore represents at most 80 allocated H100-hours.

## Adjusted MCLMC cohort

Start with the `n40k` learned prior, six observed-property cases, and two
concurrent H100 tasks:

```bash
MODEL_STAGE=n40k MODE=pilot bash scripts/submit_cosmos2020_mclmc.sh
```

For the two-galaxy wiring smoke, use `MODE=smoke`; the submitter reduces the
array to tasks `0-1` automatically.

This produces encoder/importance/MAP initialization and four adjusted-MCLMC
chains per galaxy. `lp_zBEST` is used only to choose diverse examples and is
never stored or scored as physical truth. Inspect:

```bash
sacct -j <array_job_id> --format=JobID,State,Elapsed,AllocTRES,ExitCode
find outputs/runs/popcosmos_a24_mclmc_v1/galaxies \
  -name MCLMC_DONE -o -name diagnostics.json
```

## Minimum publication comparisons

Report wall time and H100 hours separately for training, amortized inference,
and MCLMC. On spectroscopic cross-matches, report photo-z bias, NMAD, outlier
fraction, PIT/coverage, and performance versus magnitude and redshift. On the
full photometric sample, compare posterior predictive residuals, color and
magnitude distributions, learned redshift distribution, prior predictive
checks, and A24 public summaries. Use at least three RWS seeds for the final
table.

The method comparison must distinguish:

1. population-prior quality;
2. per-object posterior calibration;
3. physical identifiability of DSPS parameters.

The public A24 summaries are a reference comparison, not paired ground truth,
and the SVO/DSPS forward model is not identical to the A24 Photulator emulator.
