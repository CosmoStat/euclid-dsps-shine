# Public Configs

The public config surface is intentionally small. The main science path is
Diffsky HLTDS 04/14 on the continuous `z < 0.35` subset with materialized
`m5_depth` synthetic `fluxerr_*` columns. Euclid FS2 remains the comparison
path.

- `diffsky_dataset_hltds_04_14.yaml`: rebuild and validate
  `Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet`.
- `diffsky_hltds_04_14_simple_gpu.yaml`: main Diffsky HLTDS MAP setup. It uses
  14 LSST+Roman `fnu_cgs` flux bands, `fluxerr_*` errors, the HLTDS SSP/filter
  assets, and a 12-parameter PopCosmos-bin DSPS proxy.
- `diffsky_hltds_04_14_fixedz_closure_gpu.yaml`: diagnostic MAP setup with
  redshift fixed to `redshift_true`.
- `diffsky_hltds_04_14_trueparam_closure_gpu.yaml`: same-parameter
  Diffsky-truth forward closure.
- `prior_diffsky_hltds_supervised_basic_realnvp.yaml` and
  `prior_diffsky_hltds_supervised_extended_realnvp.yaml`: supervised RealNVP
  truth-prior experiments.
- `amortized_diffsky_hltds_standard_normal_gpu.yaml`: Diffsky amortized
  standard-normal prior baseline.
- `amortized_diffsky_hltds_supervised_prior_gpu.yaml`: Diffsky amortized
  encoder with a frozen supervised-prior checkpoint.
- `amortized_diffsky_hltds_joint_realnvp_gpu.yaml`: main Diffsky joint encoder
  plus RealNVP prior run. It extends
  `amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`, uses 14 flux + 14 error
  encoder features, a 12D standardized-logit latent, and upcasts compressed
  HLTDS SSP payloads to `float32` at runtime.
- `fs2_gpu.yaml`: Euclid/FS2 MAP/posterior comparison path.
- `amortized_fs2_realnvp.yaml`: FS2 amortized encoder plus learned RealNVP
  prior comparison path.

Deprecated Diffstar, OpenUniverse fit-ready, non-GPU, and broad ablation
configs are not part of the documented end-to-end pipeline. Historical code
paths may still exist for tests or development.
