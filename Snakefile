"""Reproducible entry points for PopCosmos-like DSPS runs."""

configfile: "configs/popcosmos_binned.yaml"

RUN_DIR = config.get("run_dir", "outputs/runs/popcosmos_fit_snakemake")
LIMIT = int(config.get("limit", 100))
BATCH_SIZE = int(config.get("batch_size", 64))
SED_SAMPLES = int(config.get("sed_samples", 8))
SCIENCE_CONFIG = config.get("science_config", "configs/popcosmos_binned.yaml")


rule all:
    input:
        f"{RUN_DIR}/batch_fit_results.parquet",
        f"{RUN_DIR}/batch_fit_photometry_comparison.parquet",
        f"{RUN_DIR}/normalized_config.json",


rule science_fit:
    output:
        fits=f"{RUN_DIR}/batch_fit_results.parquet",
        phot=f"{RUN_DIR}/batch_fit_photometry_comparison.parquet",
        config=f"{RUN_DIR}/normalized_config.json",
    shell:
        """
        euclid-dsps \
          --config {SCIENCE_CONFIG} \
          fit \
          --limit {LIMIT} \
          --batch-size {BATCH_SIZE} \
          --sed-samples {SED_SAMPLES} \
          --out {RUN_DIR}
        """
