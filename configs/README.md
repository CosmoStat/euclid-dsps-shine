# Public Configs

The public config surface is intentionally small. All runtime configs are GPU
configs.

- `fs2_gpu.yaml`: legacy Euclid/FS2 MAP fitting with the compressed
  PopCosmos-style DSPS model. Use this only as the FS2 comparison path.
- `diffsky_hltds_04_14_simple_gpu.yaml`: main Diffsky HLTDS 04/14 MAP setup.
  It uses HLTDS SSP/filter assets, native AB magnitudes, no native photometric
  error columns, and a simple PopCosmos-bin DSPS parameterization.
- `diffsky_hltds_04_14_fixedz_closure_gpu.yaml`: diagnostic only. It uses
  `redshift_true` to test photometric closure at the correct redshift.
- `amortized_fs2_realnvp.yaml`: current RealNVP prior-learning/amortized
  inference config for FS2. It inherits `fs2_gpu.yaml`.
- `amortized_diffsky_hltds_04_14_realnvp_gpu.yaml`: main Diffsky HLTDS
  amortized RealNVP prior-learning config. It uses the 14 HLTDS LSST+Roman
  AB-magnitude bands, a compact 9-parameter DSPS latent, and the compressed
  HLTDS SSP asset. It intentionally upcasts compressed SSP payloads to
  `float32` at runtime and caps compiled JAX batches to 4 objects for CUDA
  stability.

Deprecated configs for Diffstar, OpenUniverse fit-ready experiments, and
non-GPU variants have been removed from this directory. The corresponding code
paths may still exist for historical tests or development, but they are not the
documented end-to-end pipeline.
