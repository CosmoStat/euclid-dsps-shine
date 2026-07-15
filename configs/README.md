# Active Configs

The public config surface is intentionally small. FENIKS is the controlled
science dataset; HLTDS and FS2 are reference/debug paths.

## FENIKS closure and prior learning

- `diffsky_synthetic_feniks_260617_50k.yaml`: generate and validate the
  14-band train/validation/test synthetic FENIKS DSPS-closure catalogues.
- `diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml`: generate the
  LSST+Euclid+Roman 18-band FENIKS comparison sample and validate the mirrored
  inference-ready splits.
- `prior_diffsky_synthetic_feniks_full_realnvp.yaml`: train the supervised
  RealNVP prior on the full 18D `diffsky_dsps_closure_full` truth vector.
- `feniks_spline15d_postprocess.yaml`: project existing Diffsky splits into the
  fixed 11-node/10-contrast spline-SFH 15D contract in a separate run.
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
