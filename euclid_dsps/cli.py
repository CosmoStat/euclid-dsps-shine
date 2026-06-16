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
            "amortized-prior-overlap-diffsky,"
            "diffsky-train-supervised-prior,diffsky-sample-supervised-prior,"
            "diffsky-supervised-prior-report,"
            "diffsky-forward-closure,diffsky-redshift-ablation,"
            "diffsky-run-full-validation,"
            "diffsky-list-remote,diffsky-inventory-remote,diffsky-download-subset,"
            "diffsky-inventory-local,diffsky-prepare-dataset,"
            "diffsky-dataset-diagnostics,diffsky-validate-dataset,"
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
        help="Run EDA, forward sanity checks, or standalone COSMOS SED checks.",
    )
    check.add_argument(
        "--kind",
        choices=("forward", "eda", "cosmos"),
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
    check.add_argument(
        "--plot-samples",
        type=int,
        help="Number of COSMOS SEDs to overlay for check --kind cosmos.",
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
    posterior.add_argument("--index", type=int, help="Catalog row index to sample.")
    posterior.add_argument(
        "--limit", type=int, default=5, help="Maximum catalog rows to sample."
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

    ou_prepare = sub.add_parser(
        "openuniverse-prepare",
        help=argparse.SUPPRESS,
    )
    _hide_subparser_from_help(sub, "openuniverse-prepare")
    ou_prepare.add_argument(
        "--input-root",
        help="Directory or URI containing galaxy_<hpix> and galaxy_flux_<hpix> files.",
    )
    ou_prepare.add_argument(
        "--hpix",
        nargs="+",
        type=int,
        help="One or more nside=32 HEALPix ids to process.",
    )
    ou_prepare.add_argument(
        "--limit",
        type=int,
        help="Maximum joined rows to write across the selected HEALPix ids.",
    )
    ou_prepare.add_argument(
        "--min-flux-valid-bands",
        type=int,
        help="Minimum finite positive truth-flux bands required per object.",
    )
    ou_prepare.add_argument(
        "--noise-snr",
        type=float,
        help="Override fractional SNR for the default noise model.",
    )
    ou_prepare.add_argument("--seed", type=int, help="Noise RNG seed.")
    ou_prepare.add_argument(
        "--out",
        help="Output normalized parquet path.",
    )

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
        help="Train the Diffsky HLTDS amortized encoder and RealNVP prior.",
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
        help="Run Diffsky HLTDS amortized posterior inference from a checkpoint.",
    )
    _add_amortized_infer_arguments(
        infer_diffsky,
        default_out="outputs/runs/dev_amortized_diffsky_infer",
    )

    overlap = sub.add_parser(
        "amortized-prior-overlap-diffsky",
        help="Compare Diffsky truth, posterior aggregate, and learned RealNVP prior.",
    )
    overlap.add_argument("--run", required=True, help="Inference output directory.")
    overlap.add_argument("--dataset", help="Prepared Diffsky parquet override.")
    overlap.add_argument("--out", help="Output report directory.")
    overlap.add_argument("--max-objects", type=int)

    train_prior = sub.add_parser(
        "diffsky-train-supervised-prior",
        help="Train a supervised RealNVP prior directly on Diffsky truth parameters.",
    )
    train_prior.add_argument("--dataset", help="Prepared Diffsky parquet override.")
    train_prior.add_argument("--schema", help="Truth schema override.")
    train_prior.add_argument("--out", default="outputs/runs/diffsky_supervised_prior")
    train_prior.add_argument("--limit", type=int)
    train_prior.add_argument("--batch-size", type=int)
    train_prior.add_argument("--epochs", type=int)
    train_prior.add_argument("--seed", type=int)
    train_prior.add_argument("--validation-fraction", type=float)
    train_prior.add_argument(
        "--missing-policy",
        choices=("reduce", "fail"),
        help="Reduce schema when optional truth columns are missing, or fail explicitly.",
    )
    train_prior.add_argument("--quiet", action="store_true")
    train_prior.add_argument("--no-progress", action="store_true")

    sample_prior = sub.add_parser(
        "diffsky-sample-supervised-prior",
        help="Sample theta values from a supervised prior checkpoint.",
    )
    sample_prior.add_argument("--checkpoint", required=True)
    sample_prior.add_argument("--out", default="outputs/runs/diffsky_supervised_prior_samples")
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

    forward_closure = sub.add_parser(
        "diffsky-forward-closure",
        help="Run true-parameter Diffsky forward closure against prepared photometry.",
    )
    forward_closure.add_argument("--dataset", required=True)
    forward_closure.add_argument("--limit", type=int)
    forward_closure.add_argument("--batch-size", type=int, default=64)
    forward_closure.add_argument(
        "--out",
        default="outputs/runs/diffsky_trueparam_forward_closure",
    )

    redshift_ablation = sub.add_parser(
        "diffsky-redshift-ablation",
        help="Compare redshift posterior metrics across Diffsky inference runs.",
    )
    redshift_ablation.add_argument("--dataset", required=True)
    redshift_ablation.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run directory, optionally as label=path. Repeat for each method.",
    )
    redshift_ablation.add_argument(
        "--out",
        default="outputs/reports/diffsky_redshift_ablation",
    )

    full_validation = sub.add_parser(
        "diffsky-run-full-validation",
        help="Run or aggregate the full Diffsky physical-validation workflow.",
    )
    full_validation.add_argument("--dataset")
    full_validation.add_argument(
        "--out",
        default="outputs/runs/diffsky_full_validation",
    )
    full_validation.add_argument("--limit", type=int)
    full_validation.add_argument("--batch-size", type=int)
    full_validation.add_argument("--epochs", type=int)
    full_validation.add_argument("--n-samples", type=int)
    full_validation.add_argument(
        "--jax-batch-size",
        type=int,
        help="Override amortized JAX/DSPS compiled batch cap for full validation stages.",
    )
    full_validation.add_argument(
        "--decoder-sample-chunk-size",
        type=int,
        help="Override posterior/prior decoder sample chunk size for amortized inference.",
    )
    full_validation.add_argument(
        "--prior-predictive-batch-size",
        type=int,
        help="Override learned-prior predictive DSPS batch size.",
    )
    full_validation.add_argument("--posterior-samples", type=int)
    full_validation.add_argument("--prior-samples", type=int)
    full_validation.add_argument("--seed", type=int)
    full_validation.add_argument(
        "--report-only",
        action="store_true",
        help="Only aggregate existing stage outputs and write the final report.",
    )
    full_validation.add_argument(
        "--run",
        action="append",
        default=[],
        help="Existing inference run, optionally as label=path. Repeatable.",
    )
    full_validation.add_argument("--closure-run")
    full_validation.add_argument("--quiet", action="store_true")
    full_validation.add_argument("--no-progress", action="store_true")

    inr_train = sub.add_parser(
        "experimental-ssp-inr-train",
        help=argparse.SUPPRESS,
    )
    _hide_subparser_from_help(sub, "experimental-ssp-inr-train")
    add_experimental_ssp_inr_common(inr_train)
    inr_train.add_argument(
        "--model",
        choices=(
            "direct_fourier_mlp",
            "direct_siren",
            "latent_basis_mlp",
            "compressed_coeff_mlp",
        ),
        required=True,
        help="Experimental compression model to train.",
    )
    inr_train.add_argument("--steps", type=int)
    inr_train.add_argument("--batch-size", type=int, default=4096)
    inr_train.add_argument("--learning-rate", type=float, default=1.0e-3)
    inr_train.add_argument("--hidden-width", type=int, default=64)
    inr_train.add_argument("--hidden-layers", type=int, default=3)
    inr_train.add_argument("--fourier-features", type=int, default=6)
    inr_train.add_argument("--basis-k", type=int, default=32)
    inr_train.add_argument(
        "--latent-loss",
        choices=("log_flux", "coeff"),
        default="log_flux",
        help="Latent model objective. log_flux trains reconstruction end-to-end.",
    )
    inr_train.add_argument(
        "--loss-kind",
        choices=("huber", "mse"),
        default="huber",
        help="Pointwise loss for --latent-loss log_flux.",
    )
    inr_train.add_argument("--huber-delta", type=float, default=0.05)
    inr_train.add_argument(
        "--flux-weight-floor-frac",
        type=float,
        default=1.0e-4,
        help="Train/evaluate emphasis mask: flux must exceed this fraction of each curve peak.",
    )
    inr_train.add_argument(
        "--residual-baseline",
        help="Existing compressed asset; latent model learns log residual over it.",
    )
    inr_train.add_argument(
        "--coeff-baseline",
        help=(
            "Existing compressed asset whose basis is kept and whose coefficient "
            "table is replaced by compressed_coeff_mlp."
        ),
    )
    inr_train.add_argument(
        "--coeff-loss",
        choices=("coeff", "log_flux", "mixed"),
        default="mixed",
        help="Objective for compressed_coeff_mlp.",
    )
    inr_train.add_argument(
        "--coeff-log-weight",
        type=float,
        default=0.1,
        help="Log-flux loss weight when --coeff-loss=mixed.",
    )
    inr_train.add_argument(
        "--factor-agn-fagn",
        action="store_true",
        help="For agn_lnu_per_mformed, learn the fagn-factored component.",
    )
    inr_train.add_argument("--val-size", type=int, default=8192)
    inr_train.add_argument("--no-progress", action="store_true")

    inr_eval = sub.add_parser(
        "experimental-ssp-inr-eval",
        help=argparse.SUPPRESS,
    )
    _hide_subparser_from_help(sub, "experimental-ssp-inr-eval")
    add_experimental_ssp_inr_common(inr_eval)
    inr_eval.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Path to an experimental SSP INR model.npz. Repeat for multiple models.",
    )
    inr_eval.add_argument(
        "--compressed-baseline",
        action="append",
        default=[],
        help="Existing compressed SSP/gas/AGN HDF5 asset to compare. Repeatable.",
    )
    inr_eval.add_argument("--chunk-size", type=int, default=65536)
    inr_eval.add_argument("--timing-repeats", type=int, default=10)
    inr_eval.add_argument("--relative-flux-floor", type=float, default=1.0e-30)
    inr_eval.add_argument(
        "--peak-floor-frac",
        action="append",
        type=float,
        default=[],
        help="Add per-curve significant-flux mask threshold. Repeatable.",
    )
    inr_eval.add_argument(
        "--log-svd-k",
        action="append",
        type=int,
        default=[],
        help="Add an oracle explicit log-SVD baseline rank. Repeatable.",
    )

    inr_report = sub.add_parser(
        "experimental-ssp-inr-report",
        help=argparse.SUPPRESS,
    )
    _hide_subparser_from_help(sub, "experimental-ssp-inr-report")
    inr_report.add_argument(
        "--metrics",
        required=True,
        help="metrics.json or metrics_summary.csv from experimental-ssp-inr-eval.",
    )
    inr_report.add_argument(
        "--out",
        default="outputs/ssp_inr/report.md",
        help="Markdown report path.",
    )

    from .diffsky_data.cli import add_diffsky_subcommands

    add_diffsky_subcommands(sub)
    return parser


def _hide_subparser_from_help(sub: argparse._SubParsersAction, name: str) -> None:
    sub._choices_actions = [  # type: ignore[attr-defined]
        action
        for action in sub._choices_actions  # type: ignore[attr-defined]
        if getattr(action, "dest", None) != name
    ]


def _add_amortized_train_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out: str,
) -> None:
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jax-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")


def _add_amortized_infer_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_out: str,
) -> None:
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jax-batch-size", type=int)
    parser.add_argument("--posterior-samples", type=int)
    parser.add_argument("--prior-samples", type=int)
    parser.add_argument("--decoder-sample-chunk-size", type=int)
    parser.add_argument("--prior-predictive-batch-size", type=int)
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    supervised_prior_commands = {
        "diffsky-train-supervised-prior",
        "diffsky-sample-supervised-prior",
        "diffsky-supervised-prior-report",
        "diffsky-forward-closure",
        "diffsky-redshift-ablation",
        "diffsky-run-full-validation",
    }
    if args.command == "download-assets":
        from .assets import download_assets

        download_assets(Path(args.out), overwrite=bool(args.overwrite))
        return
    if args.command.startswith("diffsky-") and args.command not in supervised_prior_commands:
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
    if args.command.startswith("experimental-ssp-inr") and getattr(args, "runtime", "config") != "config":
        runtime_config = {
            **runtime_config,
            **RUNTIME_PRESETS[str(args.runtime)],
        }
    apply_jax_runtime_env(runtime_config)

    if args.command == "amortized-synthetic-smoke":
        _run_amortized_synthetic(config, args)
        return
    if args.command == "openuniverse-prepare":
        _run_openuniverse_prepare(config, args)
        return
    if args.command == "amortized-train-fs2":
        _run_amortized_train(config, args)
        return
    if args.command == "amortized-infer-fs2":
        _run_amortized_infer(config, args)
        return
    if args.command == "amortized-train-diffsky":
        _run_amortized_train(config, args, dataset_label="Diffsky HLTDS")
        return
    if args.command == "amortized-infer-diffsky":
        _run_amortized_infer(config, args, dataset_label="Diffsky HLTDS")
        return
    if args.command == "amortized-prior-overlap-diffsky":
        _run_amortized_prior_overlap_diffsky(config, args)
        return
    if args.command == "diffsky-train-supervised-prior":
        _run_diffsky_train_supervised_prior(config, args)
        return
    if args.command == "diffsky-sample-supervised-prior":
        _run_diffsky_sample_supervised_prior(config, args)
        return
    if args.command == "diffsky-supervised-prior-report":
        _run_diffsky_supervised_prior_report(config, args)
        return
    if args.command == "diffsky-forward-closure":
        _run_diffsky_forward_closure(config, args)
        return
    if args.command == "diffsky-redshift-ablation":
        _run_diffsky_redshift_ablation(config, args)
        return
    if args.command == "diffsky-run-full-validation":
        _run_diffsky_full_validation(config, args)
        return
    if args.command == "experimental-ssp-inr-train":
        _run_experimental_ssp_inr_train(args)
        return
    if args.command == "experimental-ssp-inr-eval":
        _run_experimental_ssp_inr_eval(args)
        return
    if args.command == "experimental-ssp-inr-report":
        _run_experimental_ssp_inr_report(args)
        return

    from .workflows import (
        fit_batch,
        fit_one,
        reconstruct_cosmos_seds,
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
        elif args.kind == "cosmos":
            reconstruct_cosmos_seds(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                index=getattr(args, "index", None),
                compare_dsps=False,
                fit_dsps=False,
                population_dsps=False,
                sample_plot_count=getattr(args, "plot_samples", None),
            )
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
        _apply_selection_overrides(config, args)
        _apply_fit_overrides(config, args)
        _apply_sample_overrides(config, args)
        if getattr(args, "index", None) is not None:
            sample_one(config, Path(args.out))
        else:
            sample_batch(
                config,
                Path(args.out),
                limit=getattr(args, "limit", 5),
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


def _run_openuniverse_prepare(config: dict, args) -> None:
    from .openuniverse.prepare import prepare_openuniverse_lsst_roman_subset

    ou_cfg = dict(config.get("openuniverse", {}) or {})
    hpix_ids = args.hpix if args.hpix is not None else ou_cfg.get("hpix_ids")
    if not hpix_ids:
        raise SystemExit(
            "openuniverse-prepare requires --hpix or openuniverse.hpix_ids. "
            "Refusing to process an implicit large dataset."
        )
    input_root = args.input_root or ou_cfg.get("input_root")
    if not input_root:
        raise SystemExit(
            "openuniverse-prepare requires --input-root or config input_root"
        )
    output_path = args.out or ou_cfg.get(
        "output_path",
        "Data/openuniverse/processed/ou_lsst_roman_14.parquet",
    )
    noise_model = dict(ou_cfg.get("noise_model", {}) or {})
    if args.noise_snr is not None:
        noise_model = {"type": "fractional_snr", "snr": float(args.noise_snr)}
    if not noise_model:
        noise_model = None
    manifest = prepare_openuniverse_lsst_roman_subset(
        hpix_ids=hpix_ids,
        input_root=input_root,
        output_path=output_path,
        limit=args.limit if args.limit is not None else ou_cfg.get("limit"),
        min_flux_valid_bands=int(
            args.min_flux_valid_bands
            if args.min_flux_valid_bands is not None
            else ou_cfg.get("min_flux_valid_bands", 8)
        ),
        noise_model=noise_model,
        seed=int(args.seed if args.seed is not None else ou_cfg.get("seed", 42)),
    )
    print(
        "[openuniverse] prepared "
        f"{manifest['number_of_rows']} rows -> {manifest['output_path']}"
    )
    print(
        "[openuniverse] manifest -> "
        f"{Path(manifest['output_path']).with_suffix('.manifest.yaml')}"
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
    )


def _run_diffsky_train_supervised_prior(config: dict, args) -> None:
    try:
        from .prior_learning.train import prior_learning_config, train_supervised_prior
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = prior_learning_config(config)
    training = cfg["training"]
    train_supervised_prior(
        config,
        Path(args.out),
        dataset_path=args.dataset,
        schema_name=args.schema,
        limit=args.limit,
        batch_size=int(args.batch_size or training.get("batch_size", 256)),
        epochs=int(args.epochs or training.get("epochs", 20)),
        seed=int(args.seed if args.seed is not None else training.get("seed", 42)),
        validation_fraction=args.validation_fraction,
        missing_policy=args.missing_policy,
        verbose=not bool(getattr(args, "quiet", False)),
        progress=not bool(getattr(args, "no_progress", False)),
    )


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


def _run_diffsky_forward_closure(config: dict, args) -> None:
    from .diffsky_forward_closure import run_diffsky_forward_closure

    report = run_diffsky_forward_closure(
        config,
        dataset_path=Path(args.dataset),
        out_dir=Path(args.out),
        limit=args.limit,
        batch_size=int(args.batch_size),
    )
    print(f"[diffsky] forward closure report -> {report}")


def _run_diffsky_redshift_ablation(config: dict, args) -> None:
    del config
    from .diffsky_redshift_ablation import parse_run_specs, run_redshift_ablation

    if not args.run:
        raise SystemExit("diffsky-redshift-ablation requires at least one --run")
    report = run_redshift_ablation(
        dataset_path=Path(args.dataset),
        runs=parse_run_specs(args.run),
        out_dir=Path(args.out),
    )
    print(f"[diffsky] redshift ablation report -> {report}")


def _run_diffsky_full_validation(config: dict, args) -> None:
    from .diffsky_full_validation import run_diffsky_full_validation
    from .diffsky_redshift_ablation import parse_run_specs

    report = run_diffsky_full_validation(
        config,
        out_dir=Path(args.out),
        dataset_path=Path(args.dataset) if args.dataset else None,
        limit=args.limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_samples=args.n_samples,
        jax_batch_size=args.jax_batch_size,
        decoder_sample_chunk_size=args.decoder_sample_chunk_size,
        prior_predictive_batch_size=args.prior_predictive_batch_size,
        posterior_samples=args.posterior_samples,
        prior_samples=args.prior_samples,
        seed=args.seed,
        report_only=bool(args.report_only),
        runs=parse_run_specs(args.run),
        closure_run=Path(args.closure_run) if args.closure_run else None,
        verbose=not bool(args.quiet),
        progress=not bool(args.no_progress),
    )
    print(f"[diffsky] full validation report -> {report}")


def _apply_amortized_train_overrides(config: dict, args) -> dict:
    config = dict(config)
    amortized = dict(config.get("amortized", {}) or {})
    data = dict(amortized.get("data", {}) or {})
    training = dict(amortized.get("training", {}) or {})
    if args.selection_mode is not None:
        data["selection_mode"] = args.selection_mode
    if args.stratified_strategy is not None:
        data["stratified_strategy"] = args.stratified_strategy
    if args.validation_fraction is not None:
        data["validation_fraction"] = float(args.validation_fraction)
    if args.kl_annealing_epochs is not None:
        training["kl_annealing_epochs"] = int(args.kl_annealing_epochs)
    if args.kl_weight_max is not None:
        training["kl_weight_max"] = float(args.kl_weight_max)
    if args.validation_every is not None:
        training["validation_every"] = int(args.validation_every)
    if getattr(args, "best_checkpoint_min_epoch", None) is not None:
        training["best_checkpoint_min_epoch"] = int(args.best_checkpoint_min_epoch)
    amortized["data"] = data
    amortized["training"] = training
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
        dataset_label=dataset_label,
    )


def _run_amortized_prior_overlap_diffsky(config: dict, args) -> None:
    from .amortized.prior_overlap import write_diffsky_prior_overlap_report

    run_dir = Path(args.run)
    dataset_path = Path(args.dataset) if args.dataset else Path(config["catalog_path"])
    out_dir = Path(args.out) if args.out else run_dir / "prior_overlap"
    report = write_diffsky_prior_overlap_report(
        dataset_path=dataset_path,
        run_dir=run_dir,
        out_dir=out_dir,
        config=config,
        max_objects=args.max_objects,
    )
    print(f"[amortized] prior overlap -> {report}")


def _run_experimental_ssp_inr_train(args) -> None:
    try:
        from .experimental.ssp_inr.train import train_experiment

        result = train_experiment(
            asset=args.asset,
            dataset=args.dataset,
            model=args.model,
            out=args.out,
            quick=bool(args.quick),
            max_curves=args.max_curves,
            max_wave=args.max_wave,
            max_elements=int(args.max_elements),
            allow_large_load=bool(args.allow_large_load),
            seed=int(args.seed),
            steps=args.steps,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            hidden_width=int(args.hidden_width),
            hidden_layers=int(args.hidden_layers),
            fourier_features=int(args.fourier_features),
            basis_k=int(args.basis_k),
            eps=float(args.eps),
            val_size=int(args.val_size),
            plot_examples=int(args.plot_examples),
            progress=not bool(args.no_progress),
            latent_loss=str(args.latent_loss),
            loss_kind=str(args.loss_kind),
            huber_delta=float(args.huber_delta),
            wave_min=float(args.wave_min),
            wave_max=float(args.wave_max),
            flux_weight_floor_frac=float(args.flux_weight_floor_frac),
            residual_baseline=args.residual_baseline,
            coeff_baseline=args.coeff_baseline,
            coeff_loss=str(args.coeff_loss),
            coeff_log_weight=float(args.coeff_log_weight),
            factor_agn_fagn=bool(args.factor_agn_fagn),
        )
    except RuntimeError as exc:
        if "JAX could not initialize the requested GPU backend" in str(exc):
            raise SystemExit(str(exc)) from exc
        raise
    print(f"[ssp-inr] wrote checkpoint -> {result['checkpoint']}")
    for plot in result.get("plots", []):
        print(f"[ssp-inr] wrote plot -> {plot}")


def _run_experimental_ssp_inr_eval(args) -> None:
    try:
        from .experimental.ssp_inr.evaluate import evaluate_experiment

        result = evaluate_experiment(
            asset=args.asset,
            dataset=args.dataset,
            out=args.out,
            checkpoints=list(args.checkpoint or []),
            compressed_baselines=list(args.compressed_baseline or []),
            quick=bool(args.quick),
            max_curves=args.max_curves,
            max_wave=args.max_wave,
            max_elements=int(args.max_elements),
            allow_large_load=bool(args.allow_large_load),
            seed=int(args.seed),
            eps=float(args.eps),
            relative_flux_floor=float(args.relative_flux_floor),
            chunk_size=int(args.chunk_size),
            timing_repeats=int(args.timing_repeats),
            plot_examples=int(args.plot_examples),
            wave_min=float(args.wave_min),
            wave_max=float(args.wave_max),
            peak_floor_fracs=tuple(args.peak_floor_frac or [1.0e-4, 1.0e-6]),
            log_svd_k=list(args.log_svd_k or []),
        )
    except RuntimeError as exc:
        if "JAX could not initialize the requested GPU backend" in str(exc):
            raise SystemExit(str(exc)) from exc
        raise
    outputs = result["outputs"]
    print(f"[ssp-inr] wrote metrics -> {outputs['metrics_json']}")
    print(f"[ssp-inr] wrote summary -> {outputs['metrics_csv']}")
    print(f"[ssp-inr] wrote report -> {outputs['report']}")


def _run_experimental_ssp_inr_report(args) -> None:
    from .experimental.ssp_inr.report import write_report

    path = write_report(metrics_path=args.metrics, out=args.out)
    print(f"[ssp-inr] wrote report -> {path}")


def add_experimental_ssp_inr_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asset",
        default="Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5",
        help="Dense HDF5 spectral asset.",
    )
    parser.add_argument(
        "--dataset",
        default="ssp_flux",
        help="Dense spectral dataset inside --asset.",
    )
    parser.add_argument(
        "--out",
        default="outputs/ssp_inr/run",
        help="Output directory.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small curve/wavelength subset and shorter default training.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        help="Maximum number of curve-axis samples to load.",
    )
    parser.add_argument(
        "--max-wave",
        type=int,
        help="Maximum number of wavelength samples to load.",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=20_000_000,
        help="Safety cap for loaded curve*wavelength elements.",
    )
    parser.add_argument(
        "--allow-large-load",
        action="store_true",
        help="Allow loading more than --max-elements values.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1.0e-30)
    parser.add_argument("--plot-examples", type=int, default=4)
    parser.add_argument(
        "--wave-min",
        type=float,
        default=900.0,
        help="Lower Angstrom bound for useful-wave metrics and zoom plots.",
    )
    parser.add_argument(
        "--wave-max",
        type=float,
        default=50000.0,
        help="Upper Angstrom bound for useful-wave metrics and zoom plots.",
    )
    parser.add_argument(
        "--runtime",
        choices=("config", "auto", "cpu", "gpu"),
        default="cpu",
        help="Runtime override for this experimental JAX command.",
    )


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
        help="Overlay COSMOS proxy SED when local columns/resources are available.",
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
