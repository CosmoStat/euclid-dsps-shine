# Euclid DSPS SHINE

Standalone DSPS/JAX workflow for photometric inference experiments.

The public end-to-end paths are intentionally narrow:

- `configs/fs2_gpu.yaml`: Euclid FS2 MAP/posterior baseline.
- `configs/diffsky_hltds_04_14_simple_gpu.yaml`: main Diffsky HLTDS simple MAP fit.
- `configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml`: Diffsky fixed-redshift closure diagnostic.
- `configs/amortized_fs2_realnvp.yaml`: FS2 amortized encoder plus learned RealNVP prior.
- `configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`: main Diffsky HLTDS amortized prior-learning run.

Old Diffstar, OpenUniverse fit-ready, non-GPU, and broad ablation configs are not part of the public config surface.

## Install

```bash
conda activate shine
python -m pip install -e .
```

For GPU runs:

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## Diffsky HLTDS Dataset

Main source:

```text
https://portal.nersc.gov/cfs/hacc/aphearin/diffsky_data/hltds_cosmos_260215_04_14_2026/
```

List, rank, download, inventory, prepare, and validate:

```bash
python -m euclid_dsps.cli diffsky-list-remote \
  --url https://portal.nersc.gov/cfs/hacc/aphearin/diffsky_data/hltds_cosmos_260215_04_14_2026/ \
  --max-depth 1 \
  --out outputs/diffsky_hltds_04_14_listing.json

python -m euclid_dsps.cli diffsky-inventory-remote \
  --listing outputs/diffsky_hltds_04_14_listing.json \
  --out outputs/diffsky_hltds_04_14_candidates.csv

python -m euclid_dsps.cli diffsky-download-subset \
  --listing outputs/diffsky_hltds_04_14_listing.json \
  --out-dir Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
  --max-files 12 \
  --max-total-gb 2 \
  --include diffsky_gals \
  --include param \
  --include ssp \
  --include transmission \
  --include t_table \
  --include yaml \
  --yes

python -m euclid_dsps.cli diffsky-inventory-local \
  --root Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
  --out outputs/diffsky_hltds_04_14_local_inventory.json

python -m euclid_dsps.cli diffsky-prepare-dataset \
  --raw-root Data/diffsky/raw/hltds_cosmos_260215_04_14_2026 \
  --inventory outputs/diffsky_hltds_04_14_local_inventory.json \
  --out Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
  --no-synthetic-errors

python -m euclid_dsps.cli diffsky-validate-dataset \
  --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
  --manifest Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_manifest.yaml
```

## Run Fits

Diffsky simple free-redshift batch:

```bash
python -m euclid_dsps.cli \
  --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
  fit --limit 1000 --batch-size 128 --fit-maxiter 220 \
  --sed-samples 0 --reporting-level light \
  --out outputs/runs/diffsky_hltds_simple_n1000
```

Diffsky fixed-redshift closure:

```bash
python -m euclid_dsps.cli \
  --config configs/diffsky_hltds_04_14_fixedz_closure_gpu.yaml \
  fit --limit 128 --batch-size 128 --fit-maxiter 180 \
  --sed-samples 0 --reporting-level light \
  --out outputs/runs/diffsky_hltds_fixedz_closure_n128
```

Euclid FS2 smoke:

```bash
python -m euclid_dsps.cli \
  --config configs/fs2_gpu.yaml \
  fit --index 0 --fit-maxiter 20 --sed-samples 1 \
  --out outputs/runs/fs2_gpu_one_short
```

Regenerate a Diffsky fit report:

```bash
python -m euclid_dsps.cli diffsky-fit-report \
  --run outputs/runs/diffsky_hltds_simple_n1000 \
  --config configs/diffsky_hltds_04_14_simple_gpu.yaml \
  --label batch_fit \
  --reporting-level light
```

## Prior Learning

Diffsky HLTDS joint prior-learning smoke/debug run:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml \
  amortized-train-diffsky \
  --limit 10000 --batch-size 64 --epochs 10 --n-samples 2 \
  --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000

python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml \
  amortized-infer-diffsky \
  --checkpoint outputs/runs/amortized_diffsky_hltds_realnvp_n10000/checkpoints/best.eqx \
  --limit 10000 --batch-size 64 --posterior-samples 64 --prior-samples 8192 \
  --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer

python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml \
  amortized-prior-overlap-diffsky \
  --run outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer \
  --dataset Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet \
  --out outputs/runs/amortized_diffsky_hltds_realnvp_n10000_infer/prior_overlap \
  --max-objects 10000
```

The Diffsky amortized path learns a joint degenerate prior over redshift and a
compact DSPS physical parameter set, then compares learned-prior and aggregate
posterior distributions to direct HLTDS truth columns where available.
The config caps compiled DSPS/JAX micro-batches with
`amortized.training.jax_batch_size: 4` and uses
`model.compressed_ssp_runtime_dtype: float32` to avoid CUDA half-precision SSP
segfaults; `--batch-size` can still be larger.

FS2 prior-learning remains available as the Euclid comparison path:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_fs2_realnvp.yaml \
  amortized-train-fs2 \
  --limit 10000 --batch-size 64 --epochs 5 --n-samples 2 \
  --out outputs/runs/amortized_fs2_realnvp_debug
```

See `docs/source/data_download.rst`, `docs/source/diffsky_dataset.rst`, and `docs/source/run_setup.rst` for the full documented workflow.
