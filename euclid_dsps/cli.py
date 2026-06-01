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
        default="configs/popcosmos_binned.yaml",
        help="YAML configuration file.",
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{download-assets,check,fit,posterior}",
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
        "--batch-size", type=int, default=64, help="Parquet batch size."
    )
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
        "--batch-size", type=int, default=1, help="Parquet batch size."
    )
    posterior.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    add_fit_overrides(posterior)
    add_sample_overrides(posterior)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download-assets":
        from .assets import download_assets

        download_assets(Path(args.out), overwrite=bool(args.overwrite))
        return

    from .config import load_config
    from .jax_runtime import apply_jax_runtime_env

    config = load_config(args.config)
    apply_jax_runtime_env(config.get("runtime", {}))
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
            )
        else:
            fit_batch(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                row_indices_file=getattr(args, "row_indices_file", None),
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
            )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def _limit_arg(args) -> int | None:
    return None if getattr(args, "all", False) else args.limit


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
        help=advanced_help or "Override fit.learning_rate for MAP and population steps.",
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
    if getattr(args, "no_map_init", False):
        sample["init_from_map"] = False
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
