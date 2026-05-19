"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="euclid-dsps")
    parser.add_argument(
        "--config",
        default="configs/fs2_phz1.yaml",
        help="YAML configuration file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    assets = sub.add_parser(
        "download-assets", help="Download native DSPS smoke-test assets."
    )
    assets.add_argument("--out", default="Data", help="Output directory.")
    assets.add_argument(
        "--overwrite", action="store_true", help="Replace existing files."
    )

    eda = sub.add_parser(
        "eda", help="Write schema, stats, missing values, and flux plots."
    )
    eda.add_argument("--out", default="outputs/eda", help="Output directory.")

    cosmos = sub.add_parser(
        "cosmos-sed",
        help="Reconstruct COSMOS-template proxy SEDs from catalog latent columns.",
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

    run = sub.add_parser("run-one", help="Run DSPS for one selected galaxy.")
    run.add_argument(
        "--out", default="outputs/runs/smoke_one", help="Output directory."
    )
    run.add_argument("--index", type=int, help="Catalog row index to select.")

    fit = sub.add_parser(
        "fit-one", help="Fit configured parameters for one selected galaxy."
    )
    fit.add_argument("--out", default="outputs/runs/fit_one", help="Output directory.")
    fit.add_argument("--index", type=int, help="Catalog row index to select.")
    fit.add_argument(
        "--bayesian",
        action="store_true",
        help="Run NumPyro HMC/NUTS posterior sampling instead of Adam/MAP.",
    )
    add_fit_overrides(fit)
    add_sample_overrides(fit)

    batch = sub.add_parser(
        "run-batch", help="Run configured model over many catalog rows."
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

    fit_batch = sub.add_parser(
        "fit-batch", help="Fit configured parameters over many catalog rows."
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

    population = sub.add_parser(
        "fit-population", help="Fit chunked hierarchical population MAP models."
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

    workflow = sub.add_parser(
        "fit-workflow",
        help="Run MAP batch, HMC subset, population MAP, and comparison reports.",
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

    report = sub.add_parser(
        "report-workflow",
        help="Regenerate workflow comparison plots from an existing fit-workflow output.",
    )
    report.add_argument(
        "--run-dir", required=True, help="Existing fit-workflow output directory."
    )

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

    if args.command == "eda":
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
        run_one(config, Path(args.out))
    elif args.command == "fit-one":
        _apply_selection_overrides(config, args)
        _apply_fit_overrides(config, args)
        _apply_sample_overrides(config, args)
        if args.bayesian:
            sample_one(config, Path(args.out))
        else:
            fit_one(config, Path(args.out))
    elif args.command == "run-batch":
        _apply_output_overrides(config, args)
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


def add_fit_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fit-maxiter",
        type=int,
        help="Override fit.maxiter for MAP and population steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override fit.learning_rate for MAP and population steps.",
    )
    parser.add_argument(
        "--n-sfh-bins",
        type=int,
        help="Override model.n_sfh_bins. Lower values compile/run faster.",
    )
    parser.add_argument(
        "--fast-warmstart",
        action="store_true",
        help="Skip Adam loop and run one JAX warm-start prediction pass.",
    )
    parser.add_argument(
        "--fast-grid",
        action="store_true",
        help="Fit redshift on a small PHZ grid plus analytic mass warm-start.",
    )
    parser.add_argument(
        "--full-adam",
        action="store_true",
        help="Disable fast fit shortcuts and run full Adam.",
    )
    parser.add_argument(
        "--redshift-grid-size",
        type=int,
        help="Override fit.redshift_grid_size for --fast-grid.",
    )
    parser.add_argument(
        "--fast-grid-parameters",
        help="Comma-separated parameters scanned by --fast-grid.",
    )
    parser.add_argument(
        "--fast-grid-prior-width",
        type=float,
        help="Prior sigma half-width for non-redshift fast-grid axes.",
    )


def add_output_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reporting-level",
        choices=("full", "light"),
        help="full writes plots and tables; light writes only tables and benchmarks.",
    )
    parser.add_argument(
        "--output-format",
        choices=("both", "parquet", "csv"),
        help="Tabular format for large workflow outputs.",
    )
    parser.add_argument(
        "--verbose-benchmark",
        action="store_true",
        help="Print benchmark timings for each workflow stage.",
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


if __name__ == "__main__":
    main()
