# Active Configs

The public config surface is intentionally small. FENIKS is the controlled
science dataset; HLTDS and FS2 are reference/debug paths.

## FENIKS closure and prior learning

The conditional-posterior matrix is defined by
`configs/experiments/feniks_{avi,npe}_{gaussian_x,gaussian_u,realnvp,rqspline}.yaml`.
Use `scripts/submit_feniks_conditional_posterior_matrix.sh`; each full array task
trains and runs the same held-out photometry/corner diagnostics.

The four-task unsupervised mode-covering control is defined by
`configs/experiments/feniks_mode_*.yaml` and launched with
`scripts/submit_feniks_mode_covering.sh`. It fixes one checkpoint-backed 15D
coordinate transform across all tasks, then isolates normalization, antithetic
Monte Carlo, and periodic importance-weighted wake updates.

- `diffsky_synthetic_feniks_260617_50k.yaml`: generate and validate the
  14-band train/validation/test synthetic FENIKS DSPS-closure catalogues.
- `diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml`: generate the
  LSST+Euclid+Roman 18-band FENIKS comparison sample and validate the mirrored
  inference-ready splits.
- `prior_diffsky_synthetic_feniks_full_realnvp.yaml`: train the supervised
  RealNVP prior on the full 18D `diffsky_dsps_closure_full` truth vector.
- `feniks_spline15d_postprocess.yaml`: project existing Diffsky splits into the
  fixed 11-node/10-contrast spline-SFH 15D contract in a separate run.
- `feniks_spline15d_grouped_jaxcosmo_v1.yaml`: build the isolated grouped
  JAX-COSMO v2 spline contract from the leakage-audited 18-band source splits.
- `prior_feniks_spline15d_jaxcosmo_v1.yaml`: train the production positive-
  support RealNVP for 800 epochs with five-epoch snapshots and final prior
  diagnostics on the grouped JAX-COSMO dataset.
- `amortized_feniks_spline15d_jaxcosmo_v1_gpu.yaml`: train and infer with the
  frozen JAX-COSMO spline prior and the matching joined 18-band catalog.
- `feniks_spline15d_grouped_jaxcosmo_dequant1e3_v1.yaml`: build the isolated
  JAX-COSMO dataset with uniform physical dequantization of exact-zero SFH
  contrasts in `+/-0.001 dex`, while retaining exact truth sidecars.
- `prior_feniks_spline15d_rqspline_dequant1e3_v1.yaml`: train the continuous
  15D rational-quadratic spline coupling prior and write epoch/final
  diagnostics for comparison with the exact-atom RealNVP baseline.
- `amortized_feniks_spline15d_rqspline_dequant1e3_v1_gpu.yaml`: train and infer
  with the frozen RQ-spline prior and matching JAX-COSMO catalog.
- `prior_feniks_spline15d_realnvp.yaml`: fit train-only analytic `asinh`
  transforms and train the production supervised RealNVP directly on 15D.
- `amortized_diffsky_synthetic_feniks_full_gpu.yaml`: train and run the 18D
  NN+DSPS+NF inference model with the supervised FENIKS prior checkpoint.

## HLTDS reference/download

- `diffsky_dataset_hltds_04_14.yaml`: download, prepare, validate, and diagnose
  the low-z HLTDS debug/reference parquet.
- `diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml`: download, prepare,
  validate, and diagnose the higher-redshift HLTDS truth-rich debug/reference
  parquet.

## FS2 comparison

- `fs2_gpu.yaml`: Euclid FS2 MAP/posterior comparison path.
- `amortized_fs2_realnvp.yaml`: FS2 amortized encoder plus learned RealNVP
  comparison path.

Historical HLTDS MAP/amortized experiments, exact forward-closure configs,
prior experiments, and broad experiment matrices were moved to `legacy/configs/`.
