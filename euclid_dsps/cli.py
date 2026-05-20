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
        default="configs/fs2_phz1_science.yaml",
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
        help="Sample one-row or small-subset posterior with HMC/NUTS.",
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

    eda = sub.add_parser(
        "eda",
        help=argparse.SUPPRESS,
        description="Legacy alias for: check --kind eda.",
    )
    eda.add_argument("--out", default="outputs/eda", help="Output directory.")

    cosmos = sub.add_parser(
        "cosmos-sed",
        help=argparse.SUPPRESS,
        description="Legacy COSMOS proxy SED command.",
    )
    cosmos.add_argument(
        "--out", default="outputs/runs/cosmos_sed", help="Output directory."
    )
    cosmos.add_argument(
        "--limit", type=int, default=10, help="Maximum catalog rows to process."
    )
    cosmos.add_argument(
        "--batch-size", type=int, default=1000, help="Parquet batch size."
    )
    cosmos.add_argument("--index", type=int, help="Catalog row index to process.")
    cosmos.add_argument("--all", action="store_true", help="Process the full catalog.")
    cosmos.add_argument(
        "--compare-dsps",
        action="store_true",
        help="Also compare COSMOS proxy rest SEDs against DSPS forward SEDs.",
    )
    cosmos.add_argument(
        "--fit-dsps",
        action="store_true",
        help="Fit DSPS per row before COSMOS-vs-DSPS comparison.",
    )
    cosmos.add_argument(
        "--population-dsps",
        action="store_true",
        help="Use chunked population MAP DSPS fits before COSMOS-vs-DSPS comparison.",
    )
    cosmos.add_argument(
        "--plot-samples",
        type=int,
        help="Number of reconstructed COSMOS SEDs to overlay for visual inspection.",
    )
    add_fit_overrides(cosmos)
    add_output_overrides(cosmos)

    run = sub.add_parser(
        "run-one",
        help=argparse.SUPPRESS,
        description="Legacy alias for: fit --no-optimize --index.",
    )
    run.add_argument(
        "--out", default="outputs/runs/smoke_one", help="Output directory."
    )
    run.add_argument("--index", type=int, help="Catalog row index to select.")
    add_sed_diagnostic_overrides(run)

    forward = sub.add_parser(
        "forward",
        help=argparse.SUPPRESS,
        description="Legacy alias for: fit --no-optimize.",
    )
    forward.add_argument("--out", default="outputs/runs/forward", help="Output directory.")
    forward.add_argument("--index", type=int, help="Catalog row index to select.")
    forward.add_argument(
        "--limit", type=int, default=100, help="Maximum catalog rows to process."
    )
    forward.add_argument(
        "--batch-size", type=int, default=1000, help="Parquet batch size."
    )
    forward.add_argument("--all", action="store_true", help="Process the full catalog.")
    forward.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    add_output_overrides(forward)
    add_sed_diagnostic_overrides(forward)

    fit_one_parser = sub.add_parser(
        "fit-one",
        help=argparse.SUPPRESS,
        description="Legacy alias for: fit --index.",
    )
    fit_one_parser.add_argument("--out", default="outputs/runs/fit_one", help="Output directory.")
    fit_one_parser.add_argument("--index", type=int, help="Catalog row index to select.")
    fit_one_parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Run NumPyro HMC/NUTS posterior sampling instead of Adam/MAP.",
    )
    add_fit_overrides(fit_one_parser)
    add_sample_overrides(fit_one_parser)
    add_sed_diagnostic_overrides(fit_one_parser)

    batch = sub.add_parser(
        "run-batch",
        help=argparse.SUPPRESS,
        description="Legacy alias for: fit --no-optimize.",
    )
    batch.add_argument("--out", default="outputs/runs/batch", help="Output directory.")
    batch.add_argument(
        "--limit", type=int, default=100, help="Maximum catalog rows to process."
    )
    batch.add_argument(
        "--batch-size", type=int, default=1000, help="Parquet batch size."
    )
    batch.add_argument("--all", action="store_true", help="Process the full catalog.")
    batch.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    add_output_overrides(batch)
    add_sed_diagnostic_overrides(batch)

    fit_batch = sub.add_parser(
        "fit-batch",
        help=argparse.SUPPRESS,
        description="Legacy alias for: fit.",
    )
    fit_batch.add_argument(
        "--out", default="outputs/runs/batch_fit", help="Output directory."
    )
    fit_batch.add_argument(
        "--limit", type=int, default=25, help="Maximum catalog rows to fit."
    )
    fit_batch.add_argument(
        "--batch-size", type=int, default=64, help="Parquet batch size."
    )
    fit_batch.add_argument(
        "--all", action="store_true", help="Process the full catalog."
    )
    fit_batch.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    fit_batch.add_argument(
        "--bayesian",
        action="store_true",
        help="Run NumPyro HMC/NUTS per galaxy instead of Adam/MAP.",
    )
    add_fit_overrides(fit_batch)
    add_output_overrides(fit_batch)
    add_sample_overrides(fit_batch)
    add_sed_diagnostic_overrides(fit_batch)

    population = sub.add_parser(
        "fit-population",
        help=argparse.SUPPRESS,
        description="Advanced/research command.",
    )
    population.add_argument(
        "--out", default="outputs/runs/population_fit", help="Output directory."
    )
    population.add_argument(
        "--limit", type=int, default=25, help="Maximum catalog rows to fit."
    )
    population.add_argument(
        "--batch-size", type=int, default=64, help="Parquet batch size."
    )
    population.add_argument(
        "--all", action="store_true", help="Process the full catalog."
    )
    population.add_argument(
        "--row-indices-file",
        help="CSV/TXT file containing one catalog row_index per line.",
    )
    population.add_argument(
        "--map-init-file",
        help="batch_fit_results.csv used to initialize population MAP parameters.",
    )
    add_fit_overrides(population)
    add_output_overrides(population)
    add_sed_diagnostic_overrides(population)

    workflow = sub.add_parser(
        "fit-workflow",
        help=argparse.SUPPRESS,
        description="Advanced/research command.",
    )
    workflow.add_argument(
        "--out", default="outputs/runs/fit_workflow", help="Output directory."
    )
    workflow.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum catalog rows for MAP and population fits.",
    )
    workflow.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for independent MAP fit."
    )
    workflow.add_argument(
        "--population-batch-size",
        type=int,
        help="Batch size for population MAP fit. Defaults to --batch-size.",
    )
    workflow.add_argument(
        "--hmc-n",
        type=int,
        default=20,
        help="Number of galaxies selected for Bayesian HMC.",
    )
    workflow.add_argument(
        "--hmc-batch-size",
        type=int,
        default=1,
        help="Batch size for Bayesian HMC subset.",
    )
    add_fit_overrides(workflow)
    workflow.add_argument(
        "--hmc-select",
        choices=("stratified", "random", "best", "worst"),
        default="stratified",
        help="How to select HMC galaxies from MAP reduced chi2.",
    )
    add_sample_overrides(workflow)
    add_output_overrides(workflow)
    add_sed_diagnostic_overrides(workflow)

    report = sub.add_parser(
        "report-workflow",
        help=argparse.SUPPRESS,
        description="Advanced/research command.",
    )
    report.add_argument(
        "--run-dir", required=True, help="Existing fit-workflow output directory."
    )

    _hide_legacy_subcommands(
        sub,
        {
            "eda",
            "cosmos-sed",
            "run-one",
            "forward",
            "fit-one",
            "run-batch",
            "fit-batch",
            "fit-population",
            "fit-workflow",
            "report-workflow",
        },
    )
    return parser


def _hide_legacy_subcommands(
    subparsers: argparse._SubParsersAction, names: set[str]
) -> None:
    subparsers._choices_actions = [
        action
        for action in subparsers._choices_actions
        if getattr(action, "dest", None) not in names
    ]


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
        fit_population,
        fit_workflow,
        reconstruct_cosmos_seds,
        report_workflow,
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
    elif args.command == "eda":
        run_eda(config, Path(args.out))
    elif args.command == "cosmos-sed":
        _apply_fit_overrides(config, args)
        _apply_output_overrides(config, args)
        reconstruct_cosmos_seds(
            config,
            Path(args.out),
            limit=_limit_arg(args),
            batch_size=args.batch_size,
            index=getattr(args, "index", None),
            compare_dsps=bool(getattr(args, "compare_dsps", False))
            or bool(getattr(args, "fit_dsps", False))
            or bool(getattr(args, "population_dsps", False)),
            fit_dsps=bool(getattr(args, "fit_dsps", False)),
            population_dsps=bool(getattr(args, "population_dsps", False)),
            sample_plot_count=getattr(args, "plot_samples", None),
        )
    elif args.command == "run-one":
        _apply_selection_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        run_one(config, Path(args.out))
    elif args.command == "forward":
        _apply_selection_overrides(config, args)
        _apply_output_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        if getattr(args, "index", None) is not None:
            run_one(config, Path(args.out))
        else:
            run_batch(
                config,
                Path(args.out),
                limit=_limit_arg(args),
                batch_size=args.batch_size,
                row_indices_file=getattr(args, "row_indices_file", None),
            )
    elif args.command == "fit-one":
        _apply_selection_overrides(config, args)
        _apply_fit_overrides(config, args)
        _apply_sample_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        if args.bayesian:
            sample_one(config, Path(args.out))
        else:
            fit_one(config, Path(args.out))
    elif args.command == "run-batch":
        _apply_output_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        run_batch(
            config,
            Path(args.out),
            limit=_limit_arg(args),
            batch_size=args.batch_size,
            row_indices_file=getattr(args, "row_indices_file", None),
        )
    elif args.command == "fit-batch":
        _apply_fit_overrides(config, args)
        _apply_output_overrides(config, args)
        _apply_sample_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        if args.bayesian:
            sample_batch(
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
    elif args.command == "fit-population":
        _apply_fit_overrides(config, args)
        _apply_output_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        fit_population(
            config,
            Path(args.out),
            limit=_limit_arg(args),
            batch_size=args.batch_size,
            row_indices_file=getattr(args, "row_indices_file", None),
            map_init_file=getattr(args, "map_init_file", None),
        )
    elif args.command == "fit-workflow":
        _apply_output_overrides(config, args)
        _apply_sample_overrides(config, args)
        _apply_fit_overrides(config, args)
        _apply_sed_diagnostic_overrides(config, args)
        fit_workflow(
            config,
            Path(args.out),
            limit=args.limit,
            batch_size=args.batch_size,
            hmc_n=args.hmc_n,
            hmc_batch_size=args.hmc_batch_size,
            population_batch_size=args.population_batch_size,
            hmc_select=args.hmc_select,
            seed=int(config.get("sample", {}).get("seed", 42)),
        )
    elif args.command == "report-workflow":
        report_workflow(config, Path(args.run_dir))
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def _limit_arg(args) -> int | None:
    return None if getattr(args, "all", False) else args.limit


def add_sample_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sampler",
        choices=("nuts", "hmc"),
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
        "--n-sfh-bins",
        type=int,
        help=advanced_help
        or "Override model.n_sfh_bins. Lower values compile/run faster.",
    )
    parser.add_argument(
        "--fast-warmstart",
        action="store_true",
        help=advanced_help
        or "Skip Adam loop and run one JAX warm-start prediction pass.",
    )
    parser.add_argument(
        "--fast-grid",
        action="store_true",
        help=advanced_help
        or "Fit redshift on a small row-level grid plus analytic mass warm-start.",
    )
    parser.add_argument(
        "--full-adam",
        action="store_true",
        help=advanced_help or "Disable fast fit shortcuts and run full Adam.",
    )
    parser.add_argument(
        "--redshift-grid-size",
        type=int,
        help=advanced_help or "Override fit.redshift_grid_size for --fast-grid.",
    )
    parser.add_argument(
        "--fast-grid-parameters",
        help=advanced_help or "Comma-separated parameters scanned by --fast-grid.",
    )
    parser.add_argument(
        "--fast-grid-prior-width",
        type=float,
        help=advanced_help
        or "Prior sigma half-width for non-redshift fast-grid axes.",
    )


def add_output_overrides(
    parser: argparse.ArgumentParser, *, show_advanced: bool = True
) -> None:
    advanced_help = None if show_advanced else argparse.SUPPRESS
    parser.add_argument(
        "--reporting-level",
        choices=("full", "light"),
        help=advanced_help
        or "full writes plots and tables; light writes only tables and benchmarks.",
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
        help="Write rich SED diagnostic plots/tables for the first N processed rows.",
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
    if getattr(args, "fast_warmstart", False):
        fit["fast_warmstart_only"] = True
    if getattr(args, "fast_grid", False):
        fit["fast_grid_search"] = True
        fit["fast_warmstart_only"] = False
    if getattr(args, "full_adam", False):
        fit["fast_grid_search"] = False
        fit["fast_warmstart_only"] = False
    if getattr(args, "redshift_grid_size", None) is not None:
        fit["redshift_grid_size"] = int(args.redshift_grid_size)
    if getattr(args, "fast_grid_parameters", None):
        fit["fast_grid_parameters"] = [
            item.strip()
            for item in str(args.fast_grid_parameters).split(",")
            if item.strip()
        ]
    if getattr(args, "fast_grid_prior_width", None) is not None:
        fit["fast_grid_prior_width"] = float(args.fast_grid_prior_width)
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
