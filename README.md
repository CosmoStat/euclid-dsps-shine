# Euclid DSPS SHINE

Standalone DSPS/JAX workflows for the controlled Diffsky/FENIKS closure
catalogue and the NN+DSPS+NF prior-learning ladder.

FENIKS is the primary science dataset in this checkout. HLTDS and Euclid FS2
remain available as debug/reference paths, but they are not the default
experiment surface.

## Start Here

| Need | Use |
| --- | --- |
| Install the package | `conda activate shine && python -m pip install -e .` |
| Read the production runbook | `docs/source/production.rst` |
| Generate the controlled dataset | `configs/diffsky_synthetic_feniks_260617_50k.yaml` |
| Project Diffsky truth to spline 15D | `configs/feniks_spline15d_postprocess.yaml` |
| Train the spline-15D RealNVP prior | `configs/prior_feniks_spline15d_realnvp.yaml` |
| Validate same-parameter closure | `diffsky-validate-dsps-closure` on the generated FENIKS splits |
| Train the supervised FENIKS prior | `configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml` |
| Train NN+DSPS+NF inference | `configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml` |
| Preflight the full ladder | `diffsky-plan-prior-workflow` |

Full docs live under `docs/source/`; the most useful entry points are
`production.rst`, `spline15d_realnvp.rst`, and `amortized_inference.rst`.

## Production Configs

| Config | Purpose |
| --- | --- |
| `configs/diffsky_synthetic_feniks_260617_50k.yaml` | Generate the 40k/5k/5k Diffsky/FENIKS DSPS-closure splits. |
| `configs/diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml` | Generate the LSST+Euclid+Roman 18-band FENIKS comparison sample. |
| `configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml` | Train a supervised RealNVP prior on the full 18D closure truth vector. |
| `configs/feniks_spline15d_postprocess.yaml` | Create exact/dequantized spline-15D splits from an existing Diffsky dataset. |
| `configs/prior_feniks_spline15d_realnvp.yaml` | Fit train-only `asinh` normalization and train the 15D RealNVP. |
| `configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml` | Train and infer with the 18D NN+DSPS model using the supervised FENIKS prior checkpoint. |

Reference/debug configs:

| Config | Role |
| --- | --- |
| `configs/diffsky_dataset_hltds_04_14.yaml` | Rebuild and validate the low-z HLTDS reference parquet. |
| `configs/diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml` | Rebuild and validate the higher-redshift HLTDS truth-rich reference parquet. |
| `configs/fs2_gpu.yaml` | Euclid FS2 MAP/posterior comparison path. |
| `configs/amortized_fs2_realnvp.yaml` | FS2 amortized comparison path. |

Historical HLTDS MAP/amortized experiments, OpenUniverse helpers, COSMOS SED
tools, reconstruction dashboards, and old docs/tests live under `legacy/`.

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

## FENIKS Workflow

Preflight the current dataset, checkpoints, and launch order:

```bash
python -m euclid_dsps.cli \
  --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
  diffsky-plan-prior-workflow \
  --out outputs/reports/feniks_prior_workflow
```

Generate and validate the controlled closure dataset on Jean-Zay:

```bash
GEN_JOB=$(sbatch --parsable --export=ALL,STAGE=generate,OVERWRITE=1,RESUME=0 \
  scripts/diffsky_synthetic_feniks_50k_h100.slurm)

sbatch --dependency=afterok:${GEN_JOB} --export=ALL,STAGE=validate \
  scripts/diffsky_synthetic_feniks_50k_h100.slurm
```

Validate directly when the splits already exist:

```bash
python -m euclid_dsps.cli \
  --config configs/diffsky_synthetic_feniks_260617_50k.yaml \
  diffsky-validate-dsps-closure \
  --dataset-dir Data/diffsky/synthetic/feniks_260617_dsps_closure \
  --sample-size 256 \
  --batch-size 256 \
  --runtime gpu
```

Train the supervised truth prior:

```bash
python -m euclid_dsps.cli \
  --config configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml \
  diffsky-train-supervised-prior \
  --out outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp
```

Train NN+DSPS+NF inference on the train split:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
  amortized-train-diffsky \
  --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/train.parquet \
  --prior-checkpoint outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp/checkpoints/best.eqx \
  --out outputs/runs/amortized_diffsky_synthetic_feniks_full
```

Infer on the held-out test split:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
  amortized-infer-diffsky \
  --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
  --checkpoint outputs/runs/amortized_diffsky_synthetic_feniks_full/checkpoints/best.eqx \
  --feature-stats outputs/runs/amortized_diffsky_synthetic_feniks_full/feature_stats.json \
  --out outputs/runs/amortized_diffsky_synthetic_feniks_full_test_infer \
  --shard-outputs
```

Run MAP under the learned NF prior:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
  diffsky-map-adam-prior \
  --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
  --checkpoint outputs/runs/amortized_diffsky_synthetic_feniks_full/checkpoints/best.eqx \
  --feature-stats outputs/runs/amortized_diffsky_synthetic_feniks_full/feature_stats.json \
  --out outputs/runs/map_diffsky_synthetic_feniks_under_prior \
  --prior-weight 0.05 \
  --prior-density-space x
```

Run direct MCLMC as a flat-prior posterior baseline:

```bash
python -m euclid_dsps.cli \
  --config configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml \
  posterior \
  --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
  --sampler mclmc \
  --limit 16 \
  --batch-size 4 \
  --out outputs/runs/mclmc_diffsky_synthetic_feniks_flat
```

Direct MCLMC currently uses the configured physical priors as a calibration
baseline. MAP under the learned RealNVP prior is implemented; MCLMC under the
learned NF prior needs the posterior target to load and evaluate the NF density.

## Scientific Guardrails

The repository separates:

- direct closure truth from generated/projected/reference truth;
- supervised truth priors from post-hoc priors trained on inferred samples;
- FENIKS production runs from HLTDS and FS2 debug/reference runs;
- physical latent recovery from photometric reconstruction quality.

A good photometric fit is not evidence of physical recovery. Physical claims
require same-parameter forward closure, supervised prior-vs-truth diagnostics,
posterior calibration, and derived-quantity comparisons.
