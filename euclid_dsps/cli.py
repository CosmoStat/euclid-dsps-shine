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
        metavar=(
            "{download-assets,check,fit,posterior,"
            "openuniverse-prepare,"
            "amortized-synthetic-smoke,amortized-train-fs2,amortized-infer-fs2,"
            "diffsky-list-remote,diffsky-inventory-remote,diffsky-download-subset,"
            "diffsky-inventory-local,diffsky-prepare-dataset,"
            "diffsky-dataset-diagnostics,diffsky-validate-dataset}"
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
        help="Prepare a small OpenUniverse LSST+Roman 14-band parquet subset.",
    )
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

    infer = sub.add_parser(
        "amortized-infer-fs2",
        help="Run FS2 amortized posterior inference from a checkpoint.",
    )
    infer.add_argument("--out", default="outputs/runs/dev_amortized_fs2_infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--batch-size", type=int)
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
    infer.add_argument("--seed", type=int)
    infer.add_argument("--feature-stats")

    from .diffsky_data.cli import add_diffsky_subcommands

    add_diffsky_subcommands(sub)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download-assets":
        from .assets import download_assets

        download_assets(Path(args.out), overwrite=bool(args.overwrite))
        return
    if args.command.startswith("diffsky-"):
        from .diffsky_data.cli import run_diffsky_command

        run_diffsky_command(args)
        return

    from .config import load_config
    from .jax_runtime import apply_jax_runtime_env

    config = load_config(args.config)
    runtime_config = config.get("runtime", {})
    if args.command == "amortized-synthetic-smoke":
        runtime_config = {
            **runtime_config,
            "jax_platforms": "auto",
            "require_gpu": False,
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


def _run_amortized_train(config: dict, args) -> None:
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
    )


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
    amortized["data"] = data
    amortized["training"] = training
    config["amortized"] = amortized
    return config


def _run_amortized_infer(config: dict, args) -> None:
    try:
        from .amortized.config import amortized_config
        from .amortized.infer import infer_amortized_fs2
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

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
