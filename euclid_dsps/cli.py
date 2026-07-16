"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euclid-dsps",
        description=(
            "DSPS-like SED inference from Euclid/LSST photometry. "
            "Use fit, posterior, and check for the normal workflow."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/fs2_gpu.yaml",
        help="YAML configuration file.",
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar=(
            "{download-assets,check,fit,posterior,"
            "amortized-synthetic-smoke,amortized-train-fs2,amortized-infer-fs2,"
            "amortized-train-diffsky,amortized-infer-diffsky,"
            "amortized-finalize-inference,amortized-jacobian-lens-diffsky,"
            "amortized-finalize-jacobian-lens,diffsky-map-adam-prior,"
            "diffsky-train-supervised-prior,diffsky-sample-supervised-prior,"
            "diffsky-train-inferred-prior,"
            "diffsky-plan-prior-workflow,"
            "diffsky-supervised-prior-report,"
            "diffsky-generate-dsps-closure,diffsky-validate-dsps-closure,"
            "diffsky-evaluate-dsps-closure-inference,"
            "diffsky-compare-dsps-closure-reference,"
            "diffsky-list-remote,diffsky-inventory-remote,diffsky-download-subset,"
            "diffsky-inventory-local,diffsky-prepare-dataset,"
            "diffsky-dataset-diagnostics,diffsky-redshift-subset,"
            "diffsky-validate-dataset,"
            "diffsky-fit-report}"
        ),
    )

    assets = sub.add_parser(
        "download-assets", help="Download native DSPS smoke-test assets."
    )
    assets.add_argument("--out", default="Data", help="Output directory.")
    assets.add_argument(
        "--overwrite", action="store_true", help="Replace existing files."
    )

    check = sub.add_parser(
        "check",
        help="Run EDA or forward sanity checks.",
    )
    check.add_argument(
        "--kind",
        choices=("forward", "eda"),
        default="forward",
        help="Check type.",
    )
    check.add_argument("--out", default="outputs/check", help="Output directory.")
    check.add_argument("--index", type=int, help="Catalog row index to select.")
    check.add_argument(
        "--limit", type=int, default=100, help="Maximum catalog rows to process."
    )
    check.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First catalog row_index for contiguous batch processing.",
    )
    check.add_argument(
        "--batch-size", type=int, default=1000, help="Parquet batch size."
    )
    check.add_argument("--all", action="store_true", help="Process the full catalog.")
    check.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    add_output_overrides(check, show_advanced=False)
    add_sed_diagnostic_overrides(check)

    fit = sub.add_parser(
        "fit",
        help="Run MAP fit for one row or a batch.",
    )
    fit.add_argument("--out", default="outputs/runs/fit", help="Output directory.")
    fit.add_argument("--index", type=int, help="Catalog row index to fit.")
    fit.add_argument(
        "--limit", type=int, default=25, help="Maximum catalog rows to fit."
    )
    fit.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First catalog row_index for contiguous batch processing.",
    )
    fit.add_argument("--batch-size", type=int, default=64, help="Parquet batch size.")
    fit.add_argument("--all", action="store_true", help="Process the full catalog.")
    fit.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    fit.add_argument(
        "--no-optimize",
        action="store_true",
        help="Run forward model only, without MAP optimization.",
    )
    add_fit_overrides(fit, show_advanced=False)
    add_output_overrides(fit, show_advanced=False)
    add_sed_diagnostic_overrides(fit)

    posterior = sub.add_parser(
        "posterior",
        help="Sample one-row or small-subset posterior with HMC/NUTS/MCLMC.",
    )
    posterior.add_argument(
        "--out", default="outputs/runs/posterior", help="Output directory."
    )
    posterior.add_argument("--dataset", help="Override config catalog_path.")
    posterior.add_argument("--index", type=int, help="Catalog row index to sample.")
    posterior.add_argument(
        "--limit", type=int, default=5, help="Maximum catalog rows to sample."
    )
    posterior.add_argument(
        "--all",
        action="store_true",
        help="Process all rows in --row-indices-file; full-catalog posterior is refused.",
    )
    posterior.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First catalog row_index for contiguous batch processing.",
    )
    posterior.add_argument(
        "--batch-size", type=int, default=1, help="Parquet batch size."
    )
    posterior.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    add_fit_overrides(posterior)
    add_sample_overrides(posterior)

    synthetic = sub.add_parser(
        "amortized-synthetic-smoke",
        help="Run asset-free amortized inference smoke with a mock decoder.",
    )
    synthetic.add_argument("--out", default="outputs/runs/dev_amortized_synthetic")
    synthetic.add_argument("--mock-decoder", action="store_true", default=True)
    synthetic.add_argument("--n-objects", type=int, default=128)
    synthetic.add_argument("--epochs", type=int)
    synthetic.add_argument("--batch-size", type=int, default=32)
    synthetic.add_argument("--seed", type=int)

    train = sub.add_parser(
        "amortized-train-fs2",
        help="Train the FS2 amortized encoder and RealNVP prior.",
    )
    train.add_argument("--out", default="outputs/runs/dev_amortized_fs2")
    train.add_argument("--limit", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--n-samples", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--row-indices-file")
    train.add_argument("--train-indices-file")
    train.add_argument("--validation-indices-file")
    train.add_argument(
        "--selection-mode",
        choices=["sequential", "random", "stratified_redshift"],
        help="Override amortized.data.selection_mode.",
    )
    train.add_argument(
        "--stratified-strategy",
        choices=["balanced", "proportional"],
        help="Override amortized.data.stratified_strategy.",
    )
    train.add_argument(
        "--validation-fraction",
        type=float,
        help="Override amortized.data.validation_fraction.",
    )
    train.add_argument(
        "--kl-annealing-epochs",
        type=int,
        help="Override amortized.training.kl_annealing_epochs.",
    )
    train.add_argument(
        "--kl-weight-max",
        type=float,
        help="Override amortized.training.kl_weight_max.",
    )
    train.add_argument(
        "--validation-every",
        type=int,
        help="Override amortized.training.validation_every.",
    )
    train.add_argument(
        "--prior-checkpoint",
        help="Override amortized.prior.checkpoint.",
    )
    train.add_argument(
        "--input-noise-sigma-scale",
        type=float,
        help="Enable encoder input flux noise with this multiple of flux_err.",
    )
    train.add_argument(
        "--input-noise-mode",
        choices=["encoder_flux"],
        help="Input-noise injection mode.",
    )
    train.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce amortized training console output.",
    )
    train.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable amortized training progress bars.",
    )

    train_diffsky = sub.add_parser(
        "amortized-train-diffsky",
        help="Train a Diffsky amortized encoder and RealNVP prior.",
    )
    _add_amortized_train_arguments(
        train_diffsky,
        default_out="outputs/runs/dev_amortized_diffsky",
    )

    infer = sub.add_parser(
        "amortized-infer-fs2",
        help="Run FS2 amortized posterior inference from a checkpoint.",
    )
    infer.add_argument("--out", default="outputs/runs/dev_amortized_fs2_infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--batch-size", type=int)
    infer.add_argument("--jax-batch-size", type=int)
    infer.add_argument("--posterior-samples", type=int)
    infer.add_argument("--row-indices-file")
    infer.add_argument(
        "--prior-samples",
        type=int,
        help="Number of samples to draw from the learned RealNVP prior diagnostics.",
    )
    infer.add_argument(
        "--decoder-sample-chunk-size",
        type=int,
        help=(
            "Number of posterior samples decoded by DSPS at once. "
            "Use 1 to minimize GPU memory."
        ),
    )
    infer.add_argument(
        "--prior-predictive-batch-size",
        type=int,
        help="Number of learned-prior samples decoded by DSPS at once.",
    )
    _add_amortized_infer_shard_arguments(infer)
    infer.add_argument("--seed", type=int)
    infer.add_argument("--feature-stats")

    infer_diffsky = sub.add_parser(
        "amortized-infer-diffsky",
        help="Run Diffsky amortized posterior inference from a checkpoint.",
    )
    _add_amortized_infer_arguments(
        infer_diffsky,
        default_out="outputs/runs/dev_amortized_diffsky_infer",
    )

    finalize = sub.add_parser(
        "amortized-finalize-inference",
        help="Combine sharded amortized inference outputs and write diagnostics.",
    )
    finalize.add_argument("--out", required=True, help="Inference output directory.")
    finalize.add_argument("--limit", type=int)
    finalize.add_argument(
        "--combine-sample-shards",
        action="store_true",
        help="Write monolithic posterior_samples.parquet as well as summary tables.",
    )
    finalize.add_argument("--quiet", action="store_true")

    jlens = sub.add_parser(
        "amortized-jacobian-lens-diffsky",
        help="Run Physical Jacobian Lens diagnostics for a trained Diffsky model.",
    )
    _add_amortized_jlens_arguments(jlens)

    finalize_jlens = sub.add_parser(
        "amortized-finalize-jacobian-lens",
        help="Combine sharded Physical Jacobian Lens outputs and write plots.",
    )
    finalize_jlens.add_argument("--out", required=True, help="J-lens output directory.")
    finalize_jlens.add_argument("--quiet", action="store_true")

    map_prior = sub.add_parser(
        "diffsky-map-adam-prior",
        help="Fit free-redshift MAP DSPS estimates under a learned RealNVP prior.",
    )
    map_prior.add_argument("--out", default="outputs/runs/dev_diffsky_map_prior")
    map_prior.add_argument("--dataset", help="Override config catalog_path.")
    map_prior.add_argument("--checkpoint", required=True)
    map_prior.add_argument("--feature-stats")
    map_prior.add_argument("--limit", type=int)
    map_prior.add_argument("--row-indices-file")
    map_prior.add_argument("--batch-size", type=int)
    map_prior.add_argument("--n-starts", type=int)
    map_prior.add_argument("--maxiter", type=int)
    map_prior.add_argument("--learning-rate", type=float)
    map_prior.add_argument("--prior-weight", type=float)
    map_prior.add_argument(
        "--prior-density-space",
        choices=["x", "theta"],
        help=(
            "Density used for the MAP prior term. 'x' keeps the learned latent "
            "density; 'theta' applies the sigmoid-transform Jacobian."
        ),
    )
    map_prior.add_argument(
        "--start-mode",
        choices=["encoder", "prior", "z_grid", "lowz_grid", "latin_hypercube", "mixed"],
    )
    map_prior.add_argument(
        "--start-chunk-size",
        type=int,
        help="Number of MAP starts optimized together on device.",
    )
    map_prior.add_argument("--seed", type=int)
    map_prior.add_argument(
        "--selection-mode",
        choices=["sequential", "random", "stratified_redshift"],
    )
    map_prior.add_argument(
        "--stratified-strategy",
        choices=["balanced", "proportional"],
    )
    map_prior.add_argument("--selection-seed", type=int)
    map_prior.add_argument(
        "--no-shard-outputs",
        action="store_true",
        help="Disable per-batch MAP parquet shards.",
    )
    map_prior.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute batches even when per-batch MAP shards already exist.",
    )
    map_prior.add_argument("--quiet", action="store_true")

    train_prior = sub.add_parser(
        "diffsky-train-supervised-prior",
        help="Train a supervised RealNVP prior directly on Diffsky truth parameters.",
    )
    train_prior.add_argument("--dataset", help="Prepared Diffsky parquet override.")
    train_prior.add_argument("--schema", help="Truth schema override.")
    train_prior.add_argument("--out", default="outputs/runs/diffsky_supervised_prior")
    train_prior.add_argument("--limit", type=int)
    train_prior.add_argument("--row-indices-file")
    train_prior.add_argument("--batch-size", type=int)
    train_prior.add_argument("--epochs", type=int)
    train_prior.add_argument("--seed", type=int)
    train_prior.add_argument(
        "--data-parallel",
        choices=["single", "auto", "pmap"],
        help="Override prior_learning.training.data_parallel.",
    )
    train_prior.add_argument("--validation-fraction", type=float)
    train_prior.add_argument(
        "--missing-policy",
        choices=("reduce", "fail"),
        help="Reduce schema when optional truth columns are missing, or fail explicitly.",
    )
    train_prior.add_argument("--quiet", action="store_true")
    train_prior.add_argument("--no-progress", action="store_true")

    workflow_plan = sub.add_parser(
        "diffsky-plan-prior-workflow",
        help="Audit and write the FENIKS prior-learning workflow plan.",
    )
    workflow_plan.add_argument(
        "--validation-config",
        default="configs/diffsky_synthetic_feniks_260617_50k.yaml",
        help="Validation config used for closure gates.",
    )
    workflow_plan.add_argument(
        "--prior-config",
        default="configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml",
        help="Supervised NF-prior config.",
    )
    workflow_plan.add_argument(
        "--amortized-config",
        default="configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml",
        help="Amortized NN+DSPS+NF config.",
    )
    workflow_plan.add_argument(
        "--out",
        default="outputs/reports/feniks_prior_workflow",
        help="Directory for workflow_plan.md and workflow_plan.json.",
    )
    workflow_plan.add_argument(
        "--prior-out",
        default="outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp",
    )
    workflow_plan.add_argument(
        "--amortized-out",
        default="outputs/runs/amortized_diffsky_synthetic_feniks_full",
    )
    workflow_plan.add_argument(
        "--inference-out",
        default="outputs/runs/amortized_diffsky_synthetic_feniks_full_test_infer",
    )
    workflow_plan.add_argument(
        "--map-out",
        default="outputs/runs/map_diffsky_synthetic_feniks_under_prior",
    )
    workflow_plan.add_argument(
        "--mclmc-out",
        default="outputs/runs/mclmc_diffsky_synthetic_feniks_flat",
    )
    workflow_plan.add_argument(
        "--inferred-prior-out",
        default="outputs/runs/prior_diffsky_synthetic_feniks_from_inferred",
    )
    workflow_plan.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the preflight finds blocking missing inputs.",
    )

    train_inferred_prior = sub.add_parser(
        "diffsky-train-inferred-prior",
        help="Train a RealNVP prior from MAP or MCLMC inferred theta samples.",
    )
    train_inferred_prior.add_argument(
        "--input",
        action="append",
        required=True,
        help="MAP/MCLMC parquet or CSV containing configured parameter columns.",
    )
    train_inferred_prior.add_argument(
        "--out",
        default="outputs/runs/diffsky_inferred_prior",
    )
    train_inferred_prior.add_argument("--limit", type=int)
    train_inferred_prior.add_argument("--batch-size", type=int)
    train_inferred_prior.add_argument("--epochs", type=int)
    train_inferred_prior.add_argument("--seed", type=int)
    train_inferred_prior.add_argument("--validation-fraction", type=float)
    train_inferred_prior.add_argument("--quiet", action="store_true")

    sample_prior = sub.add_parser(
        "diffsky-sample-supervised-prior",
        help="Sample theta values from a supervised prior checkpoint.",
    )
    sample_prior.add_argument("--checkpoint", required=True)
    sample_prior.add_argument(
        "--out", default="outputs/runs/diffsky_supervised_prior_samples"
    )
    sample_prior.add_argument("--n-samples", type=int)
    sample_prior.add_argument("--seed", type=int)

    report_prior = sub.add_parser(
        "diffsky-supervised-prior-report",
        help="Regenerate supervised prior truth-vs-prior diagnostics.",
    )
    report_prior.add_argument("--run", required=True)
    report_prior.add_argument("--dataset")
    report_prior.add_argument("--schema")
    report_prior.add_argument("--out")
    report_prior.add_argument("--max-truth", type=int)

    generate_closure = sub.add_parser(
        "diffsky-generate-dsps-closure",
        help="Generate a synthetic Diffsky/FENIKS DSPS closure dataset.",
    )
    generate_closure.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="all",
    )
    generate_closure.add_argument("--max-galaxies", type=int)
    generate_closure.add_argument("--smoke", action="store_true")
    generate_closure.add_argument("--overwrite", action="store_true")
    generate_closure.add_argument("--resume", action="store_true")
    generate_closure.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress generation progress logs.",
    )

    validate_closure = sub.add_parser(
        "diffsky-validate-dsps-closure",
        help="Validate a synthetic Diffsky/FENIKS DSPS closure dataset.",
    )
    validate_closure.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing train/validation/test closure parquets.",
    )
    validate_closure.add_argument("--sample-size", type=int, default=256)
    validate_closure.add_argument("--batch-size", type=int, default=256)
    validate_closure.add_argument(
        "--runtime",
        choices=("config", "cpu", "auto", "gpu"),
        default="config",
        help="Override the JAX runtime for validation.",
    )

    eval_closure = sub.add_parser(
        "diffsky-evaluate-dsps-closure-inference",
        help="Evaluate closure posterior outputs against synthetic DSPS truths.",
    )
    eval_closure.add_argument("--run", required=True, help="Inference run directory.")
    eval_closure.add_argument(
        "--dataset",
        required=True,
        help="Held-out closure parquet, usually test.parquet.",
    )
    eval_closure.add_argument("--out", help="Output evaluation directory.")

    compare_closure_reference = sub.add_parser(
        "diffsky-compare-dsps-closure-reference",
        help="Compare a synthetic DSPS closure catalog to HLTDS or Euclid FS2 phz1.",
    )
    compare_closure_reference.add_argument(
        "--synthetic",
        default="Data/diffsky/synthetic/feniks_260617_dsps_closure/all_50k.parquet",
        help="Synthetic closure parquet to compare.",
    )
    compare_closure_reference.add_argument(
        "--reference",
        default=(
            "Data/diffsky/processed/"
            "hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr_projected_truth.parquet"
        ),
        help="Reference parquet; supports HLTDS or Euclid FS2 phz1 with --reference-kind.",
    )
    compare_closure_reference.add_argument(
        "--out",
        default="outputs/audits/feniks_synthetic_vs_z035_reference",
        help="Output diagnostic directory.",
    )
    compare_closure_reference.add_argument(
        "--proposal-dir",
        help=(
            "Optional proposals directory; when provided, weighted proposal "
            "diagnostics are added to the report."
        ),
    )
    compare_closure_reference.add_argument("--max-reference", type=int)
    compare_closure_reference.add_argument("--seed", type=int, default=260617)
    compare_closure_reference.add_argument(
        "--reference-kind",
        choices=("auto", "hltds", "fs2"),
        default="auto",
        help="Reference column convention used for magnitude/truth aliases.",
    )
    compare_closure_reference.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional matplotlib plots and write only tables/JSON/report.",
    )

    from .diffsky_data.cli import add_diffsky_subcommands

    add_diffsky_subcommands(sub)
    return parser


def _add_amortized_train_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out: str,
) -> None:
    parser.add_argument(
        "--runtime",
        choices=("config", "cpu", "auto", "gpu"),
        default="config",
        help="Override the JAX runtime for training or local smoke tests.",
    )
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--dataset", help="Override config catalog_path.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jax-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--row-indices-file")
    parser.add_argument("--train-indices-file")
    parser.add_argument("--validation-indices-file")
    parser.add_argument(
        "--selection-mode",
        choices=["sequential", "random", "stratified_redshift"],
        help="Override amortized.data.selection_mode.",
    )
    parser.add_argument(
        "--stratified-strategy",
        choices=["balanced", "proportional"],
        help="Override amortized.data.stratified_strategy.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Override amortized.data.validation_fraction.",
    )
    parser.add_argument(
        "--kl-annealing-epochs",
        type=int,
        help="Override amortized.training.kl_annealing_epochs.",
    )
    parser.add_argument(
        "--kl-weight-max",
        type=float,
        help="Override amortized.training.kl_weight_max.",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        help="Override amortized.training.validation_every.",
    )
    parser.add_argument(
        "--best-checkpoint-min-epoch",
        type=int,
        help="Override amortized.training.best_checkpoint_min_epoch.",
    )
    parser.add_argument(
        "--data-parallel",
        choices=["single", "auto", "pmap"],
        help="Override amortized.training.data_parallel.",
    )
    parser.add_argument(
        "--prior-freeze-epochs",
        type=int,
        help="Freeze RealNVP prior gradients for this many initial epochs.",
    )
    parser.add_argument(
        "--prior-update-schedule",
        choices=["joint", "delayed_joint", "alternating", "encoder_then_prior"],
        help="Override amortized.prior.update_schedule.",
    )
    parser.add_argument(
        "--prior-checkpoint",
        help="Override amortized.prior.checkpoint.",
    )
    parser.add_argument(
        "--likelihood-temperature-initial",
        type=float,
        help="Initial temperature dividing the photometric NLL.",
    )
    parser.add_argument(
        "--likelihood-temperature-final",
        type=float,
        help="Final temperature dividing the photometric NLL.",
    )
    parser.add_argument(
        "--likelihood-temperature-annealing-epochs",
        type=int,
        help="Epochs used to anneal likelihood temperature.",
    )
    parser.add_argument(
        "--entropy-floor-weight",
        type=float,
        help="Override posterior entropy floor regularization weight.",
    )
    parser.add_argument(
        "--entropy-floor-min-log-std",
        type=float,
        help="Minimum encoder log_std encouraged by entropy floor.",
    )
    parser.add_argument(
        "--input-noise-sigma-scale",
        type=float,
        help="Enable encoder input flux noise with this multiple of flux_err.",
    )
    parser.add_argument(
        "--input-noise-mode",
        choices=["encoder_flux"],
        help="Input-noise injection mode.",
    )
    parser.add_argument(
        "--input-noise-apply-to",
        choices=["train", "training", "all", "none"],
        help="Scope for input-noise augmentation.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")


def _add_amortized_infer_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out: str,
) -> None:
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--dataset", help="Override config catalog_path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prior-checkpoint",
        help="Override the frozen flow checkpoint used to rebuild the model template.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jax-batch-size", type=int)
    parser.add_argument("--posterior-samples", type=int)
    parser.add_argument("--prior-samples", type=int)
    parser.add_argument("--decoder-sample-chunk-size", type=int)
    parser.add_argument("--prior-predictive-batch-size", type=int)
    parser.add_argument("--row-indices-file")
    parser.add_argument(
        "--selection-mode",
        choices=["sequential", "random", "stratified_redshift"],
        help="Select inference rows sequentially, randomly, or by redshift strata.",
    )
    parser.add_argument(
        "--stratified-strategy",
        choices=["balanced", "proportional"],
        help="Inference strategy for stratified_redshift selection.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        help="Seed for random or stratified inference row selection.",
    )
    _add_amortized_infer_shard_arguments(parser)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--feature-stats")


def _add_amortized_infer_shard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shard-outputs",
        action="store_true",
        default=None,
        help="Write posterior inference outputs as resumable per-batch shards.",
    )
    parser.add_argument(
        "--no-resume-shards",
        dest="resume_shards",
        action="store_false",
        default=None,
        help="Recompute existing inference shards instead of skipping complete ones.",
    )
    parser.add_argument(
        "--no-posterior-predictive",
        dest="write_posterior_predictive",
        action="store_false",
        default=None,
        help="Skip large posterior_predictive_flux outputs.",
    )
    parser.add_argument(
        "--no-residual-samples",
        dest="write_residual_samples",
        action="store_false",
        default=None,
        help="Skip large per-sample residual outputs; compact residual summaries remain.",
    )
    parser.add_argument(
        "--combine-sample-shards",
        action="store_true",
        default=None,
        help="Also write monolithic posterior_samples.parquet after sharded inference.",
    )
    parser.add_argument(
        "--no-combine-summary-shards",
        dest="combine_summary_shards",
        action="store_false",
        default=None,
        help="Do not write monolithic summary/feature/residual-summary tables.",
    )


def _add_amortized_jlens_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default="outputs/runs/dev_amortized_diffsky_jlens")
    parser.add_argument("--dataset", help="Override config catalog_path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-stats")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--row-indices-file")
    parser.add_argument(
        "--selection-mode",
        choices=["sequential", "random", "stratified_redshift"],
        default="sequential",
    )
    parser.add_argument(
        "--stratified-strategy",
        choices=["balanced", "proportional"],
        default="balanced",
    )
    parser.add_argument("--selection-seed", type=int, default=260617)
    parser.add_argument(
        "--mode",
        choices=["decoder", "autoencoder", "both"],
        default="decoder",
    )
    parser.add_argument("--posterior-point", choices=["mean", "median"], default="mean")
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--direction-top-k", type=int, default=5)
    parser.add_argument(
        "--include-prior-score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-ae-lens",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip this shard when complete J-lens outputs already exist.",
    )
    parser.add_argument("--quiet", action="store_true")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_backed_diffsky_commands = {
        "diffsky-train-supervised-prior",
        "diffsky-sample-supervised-prior",
        "diffsky-supervised-prior-report",
        "diffsky-generate-dsps-closure",
        "diffsky-validate-dsps-closure",
        "diffsky-evaluate-dsps-closure-inference",
        "diffsky-compare-dsps-closure-reference",
        "diffsky-map-adam-prior",
        "diffsky-train-inferred-prior",
        "diffsky-plan-prior-workflow",
    }
    if args.command == "download-assets":
        from .assets import download_assets

        download_assets(Path(args.out), overwrite=bool(args.overwrite))
        return
    if (
        args.command.startswith("diffsky-")
        and args.command not in config_backed_diffsky_commands
    ):
        from .diffsky_data.cli import run_diffsky_command

        run_diffsky_command(args)
        return

    from .config import RUNTIME_PRESETS, load_config
    from .jax_runtime import apply_jax_runtime_env

    config = load_config(args.config)
    runtime_config = config.get("runtime", {})
    if args.command == "amortized-synthetic-smoke":
        runtime_config = {
            **runtime_config,
            "jax_platforms": "auto",
            "require_gpu": False,
        }
    if args.command == "diffsky-generate-dsps-closure" and getattr(
        args, "smoke", False
    ):
        runtime_config = {
            **runtime_config,
            "jax_platforms": "cpu",
            "disable_jax_plugin_autoload": True,
            "require_gpu": False,
        }
    if args.command == "diffsky-validate-dsps-closure" and getattr(
        args, "runtime", "config"
    ) != "config":
        runtime_config = {
            **runtime_config,
            **RUNTIME_PRESETS[str(args.runtime)],
        }
    if args.command.startswith("amortized-train-") and getattr(
        args, "runtime", "config"
    ) != "config":
        runtime_config = {
            **runtime_config,
            **RUNTIME_PRESETS[str(args.runtime)],
        }
    apply_jax_runtime_env(runtime_config)

    if args.command == "amortized-synthetic-smoke":
        _run_amortized_synthetic(config, args)
        return
    if args.command == "amortized-train-fs2":
        _run_amortized_train(config, args)
        return
    if args.command == "amortized-infer-fs2":
        _run_amortized_infer(config, args)
        return
    if args.command == "amortized-train-diffsky":
        _run_amortized_train(config, args, dataset_label="Diffsky")
        return
    if args.command == "amortized-infer-diffsky":
        _run_amortized_infer(config, args, dataset_label="Diffsky")
        return
    if args.command == "amortized-finalize-inference":
        _run_amortized_finalize_inference(config, args)
        return
    if args.command == "amortized-jacobian-lens-diffsky":
        _run_amortized_jacobian_lens(config, args)
        return
    if args.command == "amortized-finalize-jacobian-lens":
        _run_amortized_finalize_jacobian_lens(config, args)
        return
    if args.command == "diffsky-map-adam-prior":
        _run_diffsky_map_adam_prior(config, args)
        return
    if args.command == "diffsky-train-supervised-prior":
        _run_diffsky_train_supervised_prior(config, args)
        return
    if args.command == "diffsky-train-inferred-prior":
        _run_diffsky_train_inferred_prior(config, args)
        return
    if args.command == "diffsky-plan-prior-workflow":
        _run_diffsky_prior_workflow_plan(config, args)
        return
    if args.command == "diffsky-sample-supervised-prior":
        _run_diffsky_sample_supervised_prior(config, args)
        return
    if args.command == "diffsky-supervised-prior-report":
        _run_diffsky_supervised_prior_report(config, args)
        return
    if args.command == "diffsky-generate-dsps-closure":
        _run_diffsky_generate_dsps_closure(config, args)
        return
    if args.command == "diffsky-validate-dsps-closure":
        _run_diffsky_validate_dsps_closure(config, args)
        return
    if args.command == "diffsky-evaluate-dsps-closure-inference":
        _run_diffsky_evaluate_dsps_closure_inference(config, args)
        return
    if args.command == "diffsky-compare-dsps-closure-reference":
        _run_diffsky_compare_dsps_closure_reference(config, args)
        return
    from .workflows import (
        fit_batch,
        fit_one,
        run_batch,
        run_eda,
        run_one,
        sample_batch,
        sample_one,
    )

    if args.command == "check":
        _apply_selection_overrides(config, args)
        _apply_output_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        if args.kind == "eda":
            run_eda(config, Path(args.out))
        elif getattr(args, "index", None) is not None:
            run_one(config, Path(args.out))
        else:
            run_batch(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                row_indices_file=getattr(args, "row_indices_file", None),
                start_index=getattr(args, "start_index", 0),
            )
    elif args.command == "fit":
        _apply_selection_overrides(config, args)
        _apply_fit_overrides(config, args)
        _apply_output_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        if getattr(args, "index", None) is not None:
            if args.no_optimize:
                run_one(config, Path(args.out))
            else:
                fit_one(config, Path(args.out))
        elif args.no_optimize:
            run_batch(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                row_indices_file=getattr(args, "row_indices_file", None),
                start_index=getattr(args, "start_index", 0),
            )
        else:
            fit_batch(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                row_indices_file=getattr(args, "row_indices_file", None),
                start_index=getattr(args, "start_index", 0),
            )
    elif args.command == "posterior":
        if getattr(args, "dataset", None):
            config = dict(config)
            config["catalog_path"] = str(args.dataset)
        _apply_selection_overrides(config, args)
        _apply_fit_overrides(config, args)
        _apply_sample_overrides(config, args)
        if getattr(args, "index", None) is not None:
            sample_one(config, Path(args.out))
        else:
            sample_batch(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                row_indices_file=getattr(args, "row_indices_file", None),
                start_index=getattr(args, "start_index", 0),
            )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def _limit_arg(args) -> int | None:
    return None if getattr(args, "all", False) else args.limit


def _run_amortized_synthetic(config: dict, args) -> None:
    try:
        from .amortized.config import amortized_config
        from .amortized.synthetic import run_synthetic_smoke
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    if getattr(args, "jax_batch_size", None) is not None:
        config = dict(config)
        amortized = dict(config.get("amortized", {}) or {})
        inference_override = dict(amortized.get("inference", {}) or {})
        inference_override["jax_batch_size"] = int(args.jax_batch_size)
        amortized["inference"] = inference_override
        config["amortized"] = amortized
    cfg = amortized_config(config)
    training = cfg["training"]
    run_synthetic_smoke(
        config,
        Path(args.out),
        n_objects=int(args.n_objects),
        epochs=int(args.epochs or training.get("epochs", 2)),
        batch_size=int(args.batch_size or training.get("batch_size", 32)),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        mock_decoder=bool(args.mock_decoder),
    )


def _run_amortized_train(
    config: dict,
    args,
    *,
    dataset_label: str = "FS2",
) -> None:
    try:
        from .amortized.config import amortized_config
        from .amortized.train import train_amortized_fs2
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    config = _apply_amortized_train_overrides(config, args)
    cfg = amortized_config(config)
    training = cfg["training"]
    train_amortized_fs2(
        config,
        Path(args.out),
        limit=args.limit,
        batch_size=int(args.batch_size or training.get("batch_size", 32)),
        epochs=int(args.epochs or training.get("epochs", 10)),
        n_samples=int(args.n_samples or training.get("n_samples", 1)),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        verbose=not bool(getattr(args, "quiet", False)),
        progress=not bool(getattr(args, "no_progress", False)),
        dataset_label=dataset_label,
        row_indices_file=getattr(args, "row_indices_file", None),
        train_indices_file=getattr(args, "train_indices_file", None),
        validation_indices_file=getattr(args, "validation_indices_file", None),
    )


def _run_diffsky_train_supervised_prior(config: dict, args) -> None:
    try:
        from .prior_learning.train import prior_learning_config, train_supervised_prior
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = prior_learning_config(config)
    training = cfg["training"]
    if getattr(args, "data_parallel", None) is not None:
        config = dict(config)
        prior_learning = dict(config.get("prior_learning", {}) or {})
        prior_training = dict(prior_learning.get("training", {}) or {})
        prior_training["data_parallel"] = str(args.data_parallel)
        prior_learning["training"] = prior_training
        config["prior_learning"] = prior_learning
        cfg = prior_learning_config(config)
        training = cfg["training"]
    train_supervised_prior(
        config,
        Path(args.out),
        dataset_path=args.dataset,
        schema_name=args.schema,
        limit=args.limit,
        row_indices_file=getattr(args, "row_indices_file", None),
        batch_size=int(args.batch_size or training.get("batch_size", 256)),
        epochs=int(args.epochs or training.get("epochs", 20)),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        validation_fraction=args.validation_fraction,
        missing_policy=args.missing_policy,
        verbose=not bool(getattr(args, "quiet", False)),
        progress=not bool(getattr(args, "no_progress", False)),
    )


def _run_diffsky_train_inferred_prior(config: dict, args) -> None:
    try:
        from .prior_learning.inferred import train_inferred_prior
        from .prior_learning.train import prior_learning_config
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = prior_learning_config(config)
    training = cfg["training"]
    train_inferred_prior(
        config,
        Path(args.out),
        input_paths=tuple(Path(path) for path in args.input),
        limit=args.limit,
        batch_size=int(args.batch_size or training.get("batch_size", 256)),
        epochs=int(args.epochs or training.get("epochs", 20)),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        validation_fraction=args.validation_fraction,
        verbose=not bool(getattr(args, "quiet", False)),
    )


def _run_diffsky_prior_workflow_plan(config: dict, args) -> None:
    from .config import load_config
    from .prior_learning.workflow import (
        build_feniks_prior_workflow_plan,
        write_feniks_prior_workflow_plan,
    )

    validation_config = load_config(args.validation_config)
    prior_config = load_config(args.prior_config)
    amortized_config = load_config(args.amortized_config)
    plan = build_feniks_prior_workflow_plan(
        config,
        validation_config=validation_config,
        prior_config=prior_config,
        amortized_config=amortized_config,
        generation_config_path=str(args.config),
        validation_config_path=str(args.validation_config),
        prior_config_path=str(args.prior_config),
        amortized_config_path=str(args.amortized_config),
        prior_out=str(args.prior_out),
        amortized_out=str(args.amortized_out),
        inference_out=str(args.inference_out),
        map_out=str(args.map_out),
        mclmc_out=str(args.mclmc_out),
        inferred_prior_out=str(args.inferred_prior_out),
    )
    outputs = write_feniks_prior_workflow_plan(plan, Path(args.out))
    print(f"[workflow] markdown -> {outputs['markdown']}")
    print(f"[workflow] json -> {outputs['json']}")
    if plan.blockers:
        print("[workflow] blockers:")
        for blocker in plan.blockers:
            print(f"[workflow] - {blocker}")
        if bool(getattr(args, "strict", False)):
            raise SystemExit(2)


def _run_diffsky_sample_supervised_prior(config: dict, args) -> None:
    try:
        from .prior_learning.infer import sample_supervised_prior
        from .prior_learning.train import prior_learning_config
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = prior_learning_config(config)
    output = cfg["output"]
    training = cfg["training"]
    sample_supervised_prior(
        config,
        Path(args.out),
        checkpoint=Path(args.checkpoint),
        n_samples=int(args.n_samples or output.get("prior_samples", 8192)),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
    )


def _run_diffsky_supervised_prior_report(config: dict, args) -> None:
    try:
        from .prior_learning.infer import write_supervised_prior_run_report
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    outputs = write_supervised_prior_run_report(
        config,
        run_dir=Path(args.run),
        out_dir=Path(args.out) if args.out else None,
        dataset_path=Path(args.dataset) if args.dataset else None,
        schema_name=args.schema,
        max_truth=args.max_truth,
    )
    print(f"[prior] report -> {outputs['report']}")


def _run_diffsky_generate_dsps_closure(config: dict, args) -> None:
    from .synthetic_diffsky import generate_dsps_closure_dataset

    out = generate_dsps_closure_dataset(
        config,
        split=str(args.split),
        max_galaxies=args.max_galaxies,
        smoke=bool(args.smoke),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        verbose=not bool(args.quiet),
    )
    print(f"[diffsky] synthetic DSPS closure dataset -> {out}")


def _run_diffsky_validate_dsps_closure(config: dict, args) -> None:
    from .synthetic_diffsky import validate_dsps_closure_dataset

    report = validate_dsps_closure_dataset(
        config,
        dataset_dir=Path(args.dataset_dir),
        sample_size=int(args.sample_size),
        batch_size=int(args.batch_size),
    )
    print(f"[diffsky] synthetic DSPS closure validation -> {report}")


def _run_diffsky_evaluate_dsps_closure_inference(config: dict, args) -> None:
    del config
    from .synthetic_diffsky.inference_evaluation import evaluate_closure_inference

    report = evaluate_closure_inference(
        run_dir=Path(args.run),
        dataset_path=Path(args.dataset),
        out_dir=Path(args.out) if args.out else None,
    )
    print(f"[diffsky] synthetic DSPS closure inference evaluation -> {report}")


def _run_diffsky_compare_dsps_closure_reference(config: dict, args) -> None:
    from .synthetic_diffsky.reference_comparison import (
        compare_synthetic_closure_to_reference,
    )

    bands = [str(band["name"]) for band in config.get("bands", [])]
    outputs = compare_synthetic_closure_to_reference(
        synthetic_path=Path(args.synthetic),
        reference_path=Path(args.reference),
        out_dir=Path(args.out),
        proposal_dir=Path(args.proposal_dir) if args.proposal_dir else None,
        bands=bands or None,
        max_reference=args.max_reference,
        seed=int(args.seed),
        plots=not bool(args.no_plots),
        reference_kind=str(args.reference_kind),
    )
    print(f"[diffsky] synthetic-vs-reference report -> {outputs['report']}")


def _apply_amortized_train_overrides(config: dict, args) -> dict:
    config = dict(config)
    if getattr(args, "dataset", None):
        config["catalog_path"] = str(args.dataset)
    amortized = dict(config.get("amortized", {}) or {})
    data = dict(amortized.get("data", {}) or {})
    training = dict(amortized.get("training", {}) or {})
    prior = dict(amortized.get("prior", {}) or {})
    posterior_regularization = dict(amortized.get("posterior_regularization", {}) or {})
    input_noise = dict(amortized.get("input_noise", {}) or {})
    if args.selection_mode is not None:
        data["selection_mode"] = args.selection_mode
    if args.stratified_strategy is not None:
        data["stratified_strategy"] = args.stratified_strategy
    if args.validation_fraction is not None:
        data["validation_fraction"] = float(args.validation_fraction)
    if getattr(args, "jax_batch_size", None) is not None:
        training["jax_batch_size"] = int(args.jax_batch_size)
    if args.kl_annealing_epochs is not None:
        training["kl_annealing_epochs"] = int(args.kl_annealing_epochs)
    if args.kl_weight_max is not None:
        training["kl_weight_max"] = float(args.kl_weight_max)
    if args.validation_every is not None:
        training["validation_every"] = int(args.validation_every)
    if getattr(args, "best_checkpoint_min_epoch", None) is not None:
        training["best_checkpoint_min_epoch"] = int(args.best_checkpoint_min_epoch)
    if getattr(args, "data_parallel", None) is not None:
        training["data_parallel"] = str(args.data_parallel)
    if getattr(args, "prior_freeze_epochs", None) is not None:
        prior["freeze_epochs"] = int(args.prior_freeze_epochs)
    if getattr(args, "prior_update_schedule", None) is not None:
        prior["update_schedule"] = str(args.prior_update_schedule)
    if getattr(args, "prior_checkpoint", None) is not None:
        prior["checkpoint"] = str(args.prior_checkpoint)
    if getattr(args, "likelihood_temperature_initial", None) is not None:
        training["likelihood_temperature_initial"] = float(
            args.likelihood_temperature_initial
        )
    if getattr(args, "likelihood_temperature_final", None) is not None:
        training["likelihood_temperature_final"] = float(
            args.likelihood_temperature_final
        )
    if getattr(args, "likelihood_temperature_annealing_epochs", None) is not None:
        training["likelihood_temperature_annealing_epochs"] = int(
            args.likelihood_temperature_annealing_epochs
        )
    if getattr(args, "entropy_floor_weight", None) is not None:
        posterior_regularization["entropy_floor_enabled"] = True
        posterior_regularization["weight"] = float(args.entropy_floor_weight)
    if getattr(args, "entropy_floor_min_log_std", None) is not None:
        posterior_regularization["entropy_floor_enabled"] = True
        posterior_regularization["min_log_std"] = float(args.entropy_floor_min_log_std)
    if getattr(args, "input_noise_sigma_scale", None) is not None:
        input_noise["enabled"] = True
        input_noise["sigma_scale"] = float(args.input_noise_sigma_scale)
    if getattr(args, "input_noise_mode", None) is not None:
        input_noise["mode"] = str(args.input_noise_mode)
    if getattr(args, "input_noise_apply_to", None) is not None:
        apply_to = str(args.input_noise_apply_to)
        input_noise["apply_to"] = apply_to
        if apply_to == "none":
            input_noise["enabled"] = False
    amortized["data"] = data
    amortized["training"] = training
    amortized["prior"] = prior
    if posterior_regularization:
        amortized["posterior_regularization"] = posterior_regularization
    if input_noise:
        amortized["input_noise"] = input_noise
    config["amortized"] = amortized
    return config


def _run_amortized_infer(
    config: dict,
    args,
    *,
    dataset_label: str = "FS2",
) -> None:
    try:
        from .amortized.config import amortized_config
        from .amortized.infer import infer_amortized_fs2
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    if getattr(args, "dataset", None):
        config = dict(config)
        config["catalog_path"] = str(args.dataset)
    if getattr(args, "prior_checkpoint", None):
        config = dict(config)
        amortized = dict(config.get("amortized", {}) or {})
        prior = dict(amortized.get("prior", {}) or {})
        prior["checkpoint"] = str(args.prior_checkpoint)
        amortized["prior"] = prior
        config["amortized"] = amortized
    if getattr(args, "jax_batch_size", None) is not None:
        config = dict(config)
        amortized = dict(config.get("amortized", {}) or {})
        inference = dict(amortized.get("inference", {}) or {})
        inference["jax_batch_size"] = int(args.jax_batch_size)
        amortized["inference"] = inference
        config["amortized"] = amortized
    cfg = amortized_config(config)
    training = cfg["training"]
    inference = cfg["inference"]
    infer_amortized_fs2(
        config,
        Path(args.out),
        checkpoint=Path(args.checkpoint),
        limit=args.limit,
        batch_size=int(args.batch_size or training.get("batch_size", 32)),
        posterior_samples=int(
            args.posterior_samples
            if args.posterior_samples is not None
            else inference.get("posterior_samples", 32)
        ),
        prior_samples=int(
            args.prior_samples
            if args.prior_samples is not None
            else inference.get("prior_samples", 8192)
        ),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        feature_stats_path=Path(args.feature_stats) if args.feature_stats else None,
        decoder_sample_chunk_size=int(
            args.decoder_sample_chunk_size
            if args.decoder_sample_chunk_size is not None
            else inference.get("decoder_sample_chunk_size", 1)
        ),
        prior_predictive_batch_size=int(
            args.prior_predictive_batch_size
            if args.prior_predictive_batch_size is not None
            else inference.get("prior_predictive_batch_size", 256)
        ),
        shard_outputs=getattr(args, "shard_outputs", None),
        resume_shards=getattr(args, "resume_shards", None),
        write_posterior_predictive=getattr(args, "write_posterior_predictive", None),
        write_residual_samples=getattr(args, "write_residual_samples", None),
        combine_sample_shards=getattr(args, "combine_sample_shards", None),
        combine_summary_shards=getattr(args, "combine_summary_shards", None),
        selection_mode=getattr(args, "selection_mode", None),
        stratified_strategy=getattr(args, "stratified_strategy", None),
        selection_seed=getattr(args, "selection_seed", None),
        row_indices_file=getattr(args, "row_indices_file", None),
        dataset_label=dataset_label,
    )


def _run_amortized_finalize_inference(config: dict, args) -> None:
    try:
        from .amortized.infer import finalize_amortized_inference
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    payload = finalize_amortized_inference(
        config,
        Path(args.out),
        limit=args.limit,
        combine_sample_shards=bool(args.combine_sample_shards),
        dataset_label="Diffsky HLTDS",
        verbose=not bool(getattr(args, "quiet", False)),
    )
    print(
        "[amortized] finalized "
        f"{payload['n_processed']}/{payload['expected_selected_rows']} objects "
        f"complete={payload['complete']}"
    )


def _run_amortized_jacobian_lens(config: dict, args) -> None:
    try:
        from .amortized.jacobian_lens import run_jacobian_lens_diffsky
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    if getattr(args, "dataset", None):
        config = dict(config)
        config["catalog_path"] = str(args.dataset)
    payload = run_jacobian_lens_diffsky(
        config,
        Path(args.out),
        checkpoint=Path(args.checkpoint),
        feature_stats_path=Path(args.feature_stats) if args.feature_stats else None,
        limit=args.limit,
        batch_size=int(args.batch_size),
        row_indices_file=getattr(args, "row_indices_file", None),
        selection_mode=str(args.selection_mode),
        stratified_strategy=str(args.stratified_strategy),
        selection_seed=int(args.selection_seed),
        mode=str(args.mode),
        posterior_point=str(args.posterior_point),
        max_objects=args.max_objects,
        direction_top_k=int(args.direction_top_k),
        include_prior_score=bool(args.include_prior_score),
        include_ae_lens=bool(args.include_ae_lens),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        resume=bool(args.resume),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    print(
        "[jlens] wrote "
        f"{payload.get('n_objects', 0)} objects -> {payload.get('shard_dir')}"
    )


def _run_amortized_finalize_jacobian_lens(config: dict, args) -> None:
    del config
    try:
        from .amortized.jacobian_lens import finalize_jacobian_lens
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    payload = finalize_jacobian_lens(
        Path(args.out),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    print(
        "[jlens] finalized "
        f"{payload.get('n_objects', 0)} objects -> {args.out}"
    )


def _run_diffsky_map_adam_prior(config: dict, args) -> None:
    try:
        from .amortized.config import amortized_config
        from .amortized.map_adam import run_map_adam_under_prior
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    if getattr(args, "dataset", None):
        config = dict(config)
        config["catalog_path"] = str(args.dataset)
    cfg = amortized_config(config)
    training = cfg["training"]
    map_cfg = dict(cfg.get("map_adam", {}) or {})
    summary = run_map_adam_under_prior(
        config,
        Path(args.out),
        checkpoint=Path(args.checkpoint),
        feature_stats_path=Path(args.feature_stats) if args.feature_stats else None,
        limit=args.limit,
        batch_size=int(args.batch_size or map_cfg.get("batch_size", 128)),
        n_starts=int(args.n_starts or map_cfg.get("n_starts", 4)),
        maxiter=int(args.maxiter or map_cfg.get("maxiter", 120)),
        learning_rate=float(
            args.learning_rate
            if args.learning_rate is not None
            else map_cfg.get("learning_rate", 0.02)
        ),
        prior_weight=float(
            args.prior_weight
            if args.prior_weight is not None
            else map_cfg.get("prior_weight", 0.05)
        ),
        prior_density_space=str(
            getattr(args, "prior_density_space", None)
            or map_cfg.get("prior_density_space", "x")
        ),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        start_mode=str(
            getattr(args, "start_mode", None) or map_cfg.get("start_mode", "encoder")
        ),
        start_chunk_size=(
            int(args.start_chunk_size)
            if getattr(args, "start_chunk_size", None) is not None
            else int(map_cfg.get("start_chunk_size", 1))
        ),
        selection_mode=getattr(args, "selection_mode", None),
        stratified_strategy=getattr(args, "stratified_strategy", None),
        selection_seed=getattr(args, "selection_seed", None),
        row_indices_file=getattr(args, "row_indices_file", None),
        shard_outputs=not bool(getattr(args, "no_shard_outputs", False)),
        resume=not bool(getattr(args, "no_resume", False)),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    print(f"[map-prior] summary -> {Path(args.out) / 'map_summary.json'}")
    print(f"[map-prior] objects -> {summary['n_objects']}")


def add_sample_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sampler",
        choices=("nuts", "hmc", "mclmc"),
        help="Override sample.sampler for Bayesian mode.",
    )
    parser.add_argument(
        "--num-warmup", type=int, help="Override sample.num_warmup for Bayesian mode."
    )
    parser.add_argument(
        "--num-samples", type=int, help="Override sample.num_samples for Bayesian mode."
    )
    parser.add_argument(
        "--num-chains", type=int, help="Override sample.num_chains for Bayesian mode."
    )
    parser.add_argument(
        "--chain-method",
        choices=("parallel", "sequential", "vectorized"),
        help="Override NumPyro chain_method.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        help="Override sample.max_tree_depth for Bayesian mode.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override fixed HMC leapfrog steps when --sampler hmc is used.",
    )
    parser.add_argument(
        "--step-size", type=float, help="Override initial HMC/NUTS step size."
    )
    parser.add_argument(
        "--mclmc-l",
        type=float,
        help="Override sample.mclmc_l for BlackJAX MCLMC.",
    )
    parser.add_argument(
        "--mclmc-step-size",
        type=float,
        help="Override sample.mclmc_step_size for BlackJAX MCLMC.",
    )
    parser.add_argument(
        "--mclmc-progress-chunk-size",
        type=int,
        help="Number of MCLMC steps between progress/debug syncs.",
    )
    parser.add_argument(
        "--mclmc-debug",
        action="store_true",
        help="Print MCLMC backend, phase, and per-chunk diagnostic messages.",
    )
    parser.add_argument(
        "--mcmc-init-strategy",
        choices=("map", "config", "map_jitter", "random_uniform"),
        help=(
            "Posterior initialization strategy. map/map_jitter run MAP first; "
            "config/random_uniform skip MAP."
        ),
    )
    parser.add_argument(
        "--mcmc-init-jitter-scale",
        type=float,
        help="Unconstrained-space jitter scale for --mcmc-init-strategy map_jitter.",
    )
    parser.add_argument(
        "--posterior-predictive-batch-size",
        type=int,
        help="Chunk size for posterior predictive model magnitudes.",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        help="Override sample.target_accept_prob for Bayesian mode.",
    )
    parser.add_argument(
        "--seed", type=int, help="Override sample.seed for Bayesian mode."
    )
    parser.add_argument(
        "--dense-mass",
        action="store_true",
        help="Use a dense adapted mass matrix for HMC/NUTS.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable NumPyro progress bar in Bayesian mode.",
    )
    parser.add_argument(
        "--no-map-init",
        action="store_true",
        help="Start HMC from prior/default initialization instead of Adam/MAP.",
    )


def add_fit_overrides(
    parser: argparse.ArgumentParser, *, show_advanced: bool = True
) -> None:
    advanced_help = None if show_advanced else argparse.SUPPRESS
    parser.add_argument(
        "--fit-maxiter",
        type=int,
        help=advanced_help or "Override fit.maxiter for MAP and population steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=advanced_help
        or "Override fit.learning_rate for MAP and population steps.",
    )
    parser.add_argument(
        "--fit-likelihood",
        choices=("gaussian", "student_t"),
        help="Photometric likelihood objective for MAP/posterior fits.",
    )
    parser.add_argument(
        "--student-t-dof",
        type=float,
        help=advanced_help or "Override fit.student_t_dof when using Student-t.",
    )
    parser.add_argument(
        "--fit-trace-mode",
        choices=("full", "optimizer", "none"),
        help=advanced_help
        or (
            "Override fit.trace_mode. 'optimizer' avoids extra diagnostic "
            "forward passes inside Adam iterations."
        ),
    )
    parser.add_argument(
        "--fit-trace-interval",
        type=int,
        help=advanced_help or "Record MAP trace diagnostics every N iterations.",
    )
    parser.add_argument(
        "--fit-scan-unroll",
        type=int,
        help=advanced_help or "Override fit.scan_unroll for the Adam lax.scan loop.",
    )
    parser.add_argument(
        "--fit-donate-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=advanced_help
        or "Donate temporary optimizer input buffers to XLA when possible.",
    )
    parser.add_argument(
        "--fit-remat-model-mags",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=advanced_help
        or "Checkpoint model-magnitude forward calls to trade compute for memory.",
    )
    parser.add_argument(
        "--fit-batch-grad-mode",
        choices=("per_galaxy", "sum"),
        help=advanced_help
        or (
            "Override fit.batch_grad_mode. 'sum' differentiates the summed "
            "batch objective instead of vmapping per-galaxy gradients."
        ),
    )
    parser.add_argument(
        "--n-sfh-bins",
        type=int,
        help=advanced_help
        or "Override model.n_sfh_bins. Lower values compile/run faster.",
    )


def add_output_overrides(
    parser: argparse.ArgumentParser, *, show_advanced: bool = True
) -> None:
    advanced_help = None if show_advanced else argparse.SUPPRESS
    parser.add_argument(
        "--reporting-level",
        choices=("full", "light", "none"),
        help=advanced_help
        or "full writes plots and tables; light/none skip plot-heavy reports.",
    )
    parser.add_argument(
        "--output-format",
        choices=("both", "parquet", "csv"),
        help=advanced_help or "Tabular format for large workflow outputs.",
    )
    parser.add_argument(
        "--verbose-benchmark",
        action="store_true",
        help=advanced_help or "Print benchmark timings for each workflow stage.",
    )


def add_sed_diagnostic_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sed-samples",
        "--save-sed-samples",
        dest="save_sed_samples",
        type=int,
        help="Write rich SED diagnostic plots/tables; batch fits select worst rows.",
    )
    parser.add_argument(
        "--plot-filters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay configured filters on SED diagnostic plots.",
    )
    parser.add_argument(
        "--plot-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reserved legacy ground-truth SED overlay; disabled in active workflows.",
    )


def _apply_sample_overrides(config: dict, args) -> None:
    sample = config.setdefault("sample", {})
    for attr in (
        "sampler",
        "num_warmup",
        "num_samples",
        "num_chains",
        "chain_method",
        "max_tree_depth",
        "num_steps",
        "step_size",
        "mclmc_l",
        "mclmc_step_size",
        "mclmc_progress_chunk_size",
        "posterior_predictive_batch_size",
        "target_accept_prob",
        "seed",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            sample[attr] = value
    if getattr(args, "dense_mass", False):
        sample["dense_mass"] = True
    if getattr(args, "no_progress", False):
        sample["progress_bar"] = False
    init_strategy = getattr(args, "mcmc_init_strategy", None)
    if init_strategy is not None:
        sample["init_strategy"] = init_strategy
        sample["init_from_map"] = init_strategy in {"map", "map_jitter"}
    init_jitter_scale = getattr(args, "mcmc_init_jitter_scale", None)
    if init_jitter_scale is not None:
        sample["init_jitter_scale"] = init_jitter_scale
    if getattr(args, "no_map_init", False) and init_strategy is None:
        sample["init_from_map"] = False
        sample["init_strategy"] = "config"
    if getattr(args, "mclmc_debug", False):
        sample["mclmc_debug"] = True


def _apply_selection_overrides(config: dict, args) -> None:
    index = getattr(args, "index", None)
    if index is not None:
        config.setdefault("selection", {})["index"] = int(index)


def _apply_fit_overrides(config: dict, args) -> None:
    fit = config.setdefault("fit", {})
    if getattr(args, "fit_maxiter", None) is not None:
        fit["maxiter"] = args.fit_maxiter
    if getattr(args, "learning_rate", None) is not None:
        fit["learning_rate"] = args.learning_rate
    if getattr(args, "fit_likelihood", None) is not None:
        fit["photometric_likelihood"] = args.fit_likelihood
    if getattr(args, "student_t_dof", None) is not None:
        fit["student_t_dof"] = args.student_t_dof
    if getattr(args, "fit_trace_mode", None) is not None:
        fit["trace_mode"] = args.fit_trace_mode
    if getattr(args, "fit_trace_interval", None) is not None:
        fit["trace_interval"] = args.fit_trace_interval
    if getattr(args, "fit_scan_unroll", None) is not None:
        fit["scan_unroll"] = args.fit_scan_unroll
    if getattr(args, "fit_donate_inputs", None) is not None:
        fit["donate_optimizer_inputs"] = bool(args.fit_donate_inputs)
    if getattr(args, "fit_remat_model_mags", None) is not None:
        fit["remat_model_mags"] = bool(args.fit_remat_model_mags)
    if getattr(args, "fit_batch_grad_mode", None) is not None:
        fit["batch_grad_mode"] = args.fit_batch_grad_mode
    if getattr(args, "n_sfh_bins", None) is not None:
        config.setdefault("model", {})["n_sfh_bins"] = int(args.n_sfh_bins)


def _apply_output_overrides(config: dict, args) -> None:
    if getattr(args, "reporting_level", None) is not None:
        config.setdefault("reporting", {})["level"] = args.reporting_level
    if getattr(args, "output_format", None) is not None:
        config.setdefault("output", {})["format"] = args.output_format
    if getattr(args, "verbose_benchmark", False):
        config.setdefault("output", {})["verbose_benchmark"] = True


def _apply_sed_diagnostic_overrides(config: dict, args) -> None:
    reporting = config.setdefault("reporting", {})
    if getattr(args, "save_sed_samples", None) is not None:
        reporting["save_sed_samples"] = int(args.save_sed_samples)
    if getattr(args, "plot_filters", None) is not None:
        reporting["plot_filters"] = bool(args.plot_filters)
    if getattr(args, "plot_ground_truth", None) is not None:
        reporting["plot_ground_truth"] = bool(args.plot_ground_truth)


if __name__ == "__main__":
    main()
