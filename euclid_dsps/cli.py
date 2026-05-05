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

    assets = sub.add_parser("download-assets", help="Download native DSPS smoke-test assets.")
    assets.add_argument("--out", default="Data", help="Output directory.")
    assets.add_argument("--overwrite", action="store_true", help="Replace existing files.")

    eda = sub.add_parser("eda", help="Write schema, stats, missing values, and flux plots.")
    eda.add_argument("--out", default="outputs/eda", help="Output directory.")

    run = sub.add_parser("run-one", help="Run DSPS for one selected galaxy.")
    run.add_argument("--out", default="outputs/runs/smoke_one", help="Output directory.")
    run.add_argument("--index", type=int, help="Catalog row index to select.")

    fit = sub.add_parser("fit-one", help="Fit configured parameters for one selected galaxy.")
    fit.add_argument("--out", default="outputs/runs/fit_one", help="Output directory.")
    fit.add_argument("--index", type=int, help="Catalog row index to select.")
    fit.add_argument("--bayesian", action="store_true", help="Run NumPyro HMC/NUTS posterior sampling instead of Adam/MAP.")
    add_sample_overrides(fit)

    batch = sub.add_parser("run-batch", help="Run configured model over many catalog rows.")
    batch.add_argument("--out", default="outputs/runs/batch", help="Output directory.")
    batch.add_argument("--limit", type=int, default=100, help="Maximum catalog rows to process.")
    batch.add_argument("--batch-size", type=int, default=1000, help="Parquet batch size.")
    batch.add_argument("--all", action="store_true", help="Process the full catalog.")
    batch.add_argument("--row-indices-file", help="CSV/TXT file containing one catalog row_index per line.")

    fit_batch = sub.add_parser("fit-batch", help="Fit configured parameters over many catalog rows.")
    fit_batch.add_argument("--out", default="outputs/runs/batch_fit", help="Output directory.")
    fit_batch.add_argument("--limit", type=int, default=25, help="Maximum catalog rows to fit.")
    fit_batch.add_argument("--batch-size", type=int, default=64, help="Parquet batch size.")
    fit_batch.add_argument("--all", action="store_true", help="Process the full catalog.")
    fit_batch.add_argument("--row-indices-file", help="CSV/TXT file containing one catalog row_index per line.")
    fit_batch.add_argument("--bayesian", action="store_true", help="Run NumPyro HMC/NUTS per galaxy instead of Adam/MAP.")
    add_sample_overrides(fit_batch)

    population = sub.add_parser("fit-population", help="Fit chunked hierarchical population MAP models.")
    population.add_argument("--out", default="outputs/runs/population_fit", help="Output directory.")
    population.add_argument("--limit", type=int, default=25, help="Maximum catalog rows to fit.")
    population.add_argument("--batch-size", type=int, default=64, help="Parquet batch size.")
    population.add_argument("--all", action="store_true", help="Process the full catalog.")
    population.add_argument("--row-indices-file", help="CSV/TXT file containing one catalog row_index per line.")
    population.add_argument("--map-init-file", help="batch_fit_results.csv used to initialize population MAP parameters.")

    workflow = sub.add_parser("fit-workflow", help="Run MAP batch, HMC subset, population MAP, and comparison reports.")
    workflow.add_argument("--out", default="outputs/runs/fit_workflow", help="Output directory.")
    workflow.add_argument("--limit", type=int, default=1000, help="Maximum catalog rows for MAP and population fits.")
    workflow.add_argument("--batch-size", type=int, default=64, help="Batch size for independent MAP fit.")
    workflow.add_argument("--population-batch-size", type=int, help="Batch size for population MAP fit. Defaults to --batch-size.")
    workflow.add_argument("--hmc-n", type=int, default=20, help="Number of galaxies selected for Bayesian HMC.")
    workflow.add_argument("--hmc-batch-size", type=int, default=1, help="Batch size for Bayesian HMC subset.")
    workflow.add_argument("--fit-maxiter", type=int, help="Override fit.maxiter for MAP and population steps.")
    workflow.add_argument("--learning-rate", type=float, help="Override fit.learning_rate for MAP and population steps.")
    workflow.add_argument(
        "--hmc-select",
        choices=("stratified", "random", "best", "worst"),
        default="stratified",
        help="How to select HMC galaxies from MAP reduced chi2.",
    )
    add_sample_overrides(workflow)

    report = sub.add_parser("report-workflow", help="Regenerate workflow comparison plots from an existing fit-workflow output.")
    report.add_argument("--run-dir", required=True, help="Existing fit-workflow output directory.")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download-assets":
        from .assets import download_assets

        download_assets(Path(args.out), overwrite=bool(args.overwrite))
        return

    from .config import load_config
    from .pipeline import fit_batch, fit_one, fit_population, fit_workflow, report_workflow, run_batch, run_eda, run_one, sample_batch, sample_one

    config = load_config(args.config)
    if args.command == "eda":
        run_eda(config, Path(args.out))
    elif args.command == "run-one":
        _apply_selection_overrides(config, args)
        run_one(config, Path(args.out))
    elif args.command == "fit-one":
        _apply_selection_overrides(config, args)
        _apply_sample_overrides(config, args)
        if args.bayesian:
            sample_one(config, Path(args.out))
        else:
            fit_one(config, Path(args.out))
    elif args.command == "run-batch":
        run_batch(config, Path(args.out), limit=_limit_arg(args), batch_size=args.batch_size, row_indices_file=getattr(args, "row_indices_file", None))
    elif args.command == "fit-batch":
        _apply_sample_overrides(config, args)
        if args.bayesian:
            sample_batch(config, Path(args.out), limit=_limit_arg(args), batch_size=args.batch_size, row_indices_file=getattr(args, "row_indices_file", None))
        else:
            fit_batch(config, Path(args.out), limit=_limit_arg(args), batch_size=args.batch_size, row_indices_file=getattr(args, "row_indices_file", None))
    elif args.command == "fit-population":
        fit_population(
            config,
            Path(args.out),
            limit=_limit_arg(args),
            batch_size=args.batch_size,
            row_indices_file=getattr(args, "row_indices_file", None),
            map_init_file=getattr(args, "map_init_file", None),
        )
    elif args.command == "fit-workflow":
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
    parser.add_argument("--num-warmup", type=int, help="Override sample.num_warmup for Bayesian mode.")
    parser.add_argument("--num-samples", type=int, help="Override sample.num_samples for Bayesian mode.")
    parser.add_argument("--num-chains", type=int, help="Override sample.num_chains for Bayesian mode.")
    parser.add_argument(
        "--chain-method",
        choices=("parallel", "sequential", "vectorized"),
        help="Override NumPyro chain_method.",
    )
    parser.add_argument("--max-tree-depth", type=int, help="Override sample.max_tree_depth for Bayesian mode.")
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override fixed HMC leapfrog steps when --sampler hmc is used.",
    )
    parser.add_argument("--step-size", type=float, help="Override initial HMC/NUTS step size.")
    parser.add_argument("--target-accept-prob", type=float, help="Override sample.target_accept_prob for Bayesian mode.")
    parser.add_argument("--seed", type=int, help="Override sample.seed for Bayesian mode.")
    parser.add_argument("--dense-mass", action="store_true", help="Use a dense adapted mass matrix for HMC/NUTS.")
    parser.add_argument("--no-progress", action="store_true", help="Disable NumPyro progress bar in Bayesian mode.")
    parser.add_argument("--no-map-init", action="store_true", help="Start HMC from prior/default initialization instead of Adam/MAP.")


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


if __name__ == "__main__":
    main()
