# Public Configs

The production config surface is centered on the synthetic Diffsky/FENIKS
DSPS-closure workflow. HLTDS low-z configs remain useful for reference
comparisons, regression checks, and legacy PopCosmos-style debugging. Euclid
FS2 remains the external comparison path.

## Production FENIKS/DSPS closure path

- `diffsky_synthetic_feniks_260617_50k.yaml`: generate the 40k/5k/5k
  train/validation/test closure catalogues under
  `Data/diffsky/synthetic/feniks_260617_dsps_closure/`. It uses independent
  Diffsky/FENIKS proposal pools, weighted resampling, local DSPS true-flux
  generation, and the configured `m5_depth` error model.
- `diffsky_synthetic_feniks_260617_trueparam_closure.yaml`: validate the
  synthetic closure dataset with strict 18D truth, split, noise, metallicity,
  S/N, and DSPS flux-recompute gates.
- `diffsky_synthetic_feniks_260617_50k_survey_like_18band.yaml`: generate the
  LSST+Euclid+Roman closure dataset under
  `Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/`. It keeps raw
  weighted proposals, writes a looser `survey_like/` sample, mirrors the
  stricter `inference_ready/` sample to the dataset root, and compares
  LSST+Euclid colors to FS2 when the FS2 parquet is available.
- `diffsky_synthetic_feniks_260617_trueparam_closure_18band.yaml`: validate
  the 18-band inference-ready closure sample.
- `prior_diffsky_synthetic_feniks_full_realnvp.yaml`: train the supervised
  RealNVP prior over the full 18D `diffsky_dsps_closure_full` truth schema.
- `amortized_diffsky_synthetic_feniks_full_gpu.yaml`: train the 18D amortized
  closure inference model with the supervised FENIKS prior checkpoint frozen.

## HLTDS reference and debug path

- `diffsky_dataset_hltds_04_14.yaml`: rebuild and validate the 04/14 low-z
  HLTDS reference parquet with materialized `m5_depth` `fluxerr_*` columns and
  projected DSPS truth columns.
- `diffsky_dataset_hltds_03_31_zmax335_m5depth.yaml`: rebuild and validate the
  03/31 `z <= 3.35` derivative used for generated-truth population and
  projection diagnostics.
- `diffsky_hltds_04_14_simple_gpu.yaml`: legacy HLTDS MAP setup using a
  12-parameter PopCosmos-bin DSPS proxy. This is a reconstruction/debug path,
  not the strict 18D production closure contract.
- `diffsky_hltds_04_14_fixedz_closure_gpu.yaml`: diagnostic MAP setup with
  redshift fixed to `redshift_true`.
- `diffsky_hltds_04_14_trueparam_closure_gpu.yaml`: same-parameter
  Diffsky-truth forward closure on prepared HLTDS photometry.
- `prior_diffsky_hltds_supervised_basic_realnvp.yaml` and
  `prior_diffsky_hltds_supervised_extended_realnvp.yaml`: supervised RealNVP
  truth-prior experiments on available HLTDS direct/generated truths.
- `amortized_diffsky_hltds_standard_normal_gpu.yaml`: HLTDS amortized
  standard-normal prior baseline.
- `amortized_diffsky_hltds_supervised_prior_gpu.yaml`: HLTDS amortized encoder
  with a frozen supervised-prior checkpoint.
- `amortized_diffsky_hltds_joint_realnvp_gpu.yaml`: HLTDS joint encoder plus
  RealNVP prior run. It extends `amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`
  and uses 14 flux plus 14 error encoder features.

## Euclid comparison path

- `fs2_gpu.yaml`: Euclid/FS2 MAP/posterior comparison path.
- `amortized_fs2_realnvp.yaml`: FS2 amortized encoder plus learned RealNVP
  prior comparison path.

Deprecated Diffstar, OpenUniverse fit-ready, non-GPU, and broad ablation
configs are not part of the documented end-to-end pipeline. Historical code
paths may still exist for tests or development.
