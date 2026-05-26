"""Reproducible entry points for Euclid+LSST DSPS science runs."""

configfile: "configs/fs2_phz1_science.yaml"

RUN_DIR = config.get("run_dir", "outputs/runs/science_fit_snakemake")
LIMIT = int(config.get("limit", 1000))
BATCH_SIZE = int(config.get("batch_size", 2048))
SED_SAMPLES = int(config.get("sed_samples", 24))
SCIENCE_CONFIG = config.get("science_config", "configs/fs2_phz1_science.yaml")


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
        euclid-dsps fit \
          --config {SCIENCE_CONFIG} \
          --limit {LIMIT} \
          --batch-size {BATCH_SIZE} \
          --sed-samples {SED_SAMPLES} \
          --out {RUN_DIR}
        """

