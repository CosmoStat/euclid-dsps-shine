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

    fit = sub.add_parser("fit-one", help="Fit configured parameters for one selected galaxy.")
    fit.add_argument("--out", default="outputs/runs/fit_one", help="Output directory.")

    batch = sub.add_parser("run-batch", help="Run configured model over many catalog rows.")
    batch.add_argument("--out", default="outputs/runs/batch", help="Output directory.")
    batch.add_argument("--limit", type=int, default=100, help="Maximum catalog rows to process.")
    batch.add_argument("--batch-size", type=int, default=1000, help="Parquet batch size.")
    batch.add_argument("--all", action="store_true", help="Process the full catalog.")

    fit_batch = sub.add_parser("fit-batch", help="Fit configured parameters over many catalog rows.")
    fit_batch.add_argument("--out", default="outputs/runs/batch_fit", help="Output directory.")
    fit_batch.add_argument("--limit", type=int, default=25, help="Maximum catalog rows to fit.")
    fit_batch.add_argument("--batch-size", type=int, default=64, help="Parquet batch size.")
    fit_batch.add_argument("--all", action="store_true", help="Process the full catalog.")

    population = sub.add_parser("fit-population", help="Fit chunked hierarchical population MAP models.")
    population.add_argument("--out", default="outputs/runs/population_fit", help="Output directory.")
    population.add_argument("--limit", type=int, default=25, help="Maximum catalog rows to fit.")
    population.add_argument("--batch-size", type=int, default=64, help="Parquet batch size.")
    population.add_argument("--all", action="store_true", help="Process the full catalog.")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download-assets":
        from .assets import download_assets

        download_assets(Path(args.out), overwrite=bool(args.overwrite))
        return

    from .config import load_config
    from .pipeline import fit_batch, fit_one, fit_population, run_batch, run_eda, run_one

    config = load_config(args.config)
    if args.command == "eda":
        run_eda(config, Path(args.out))
    elif args.command == "run-one":
        run_one(config, Path(args.out))
    elif args.command == "fit-one":
        fit_one(config, Path(args.out))
    elif args.command == "run-batch":
        run_batch(config, Path(args.out), limit=_limit_arg(args), batch_size=args.batch_size)
    elif args.command == "fit-batch":
        fit_batch(config, Path(args.out), limit=_limit_arg(args), batch_size=args.batch_size)
    elif args.command == "fit-population":
        fit_population(config, Path(args.out), limit=_limit_arg(args), batch_size=args.batch_size)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def _limit_arg(args) -> int | None:
    return None if getattr(args, "all", False) else args.limit


if __name__ == "__main__":
    main()
