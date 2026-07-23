from __future__ import annotations

from euclid_dsps.cli import build_parser


def test_jacobian_lens_parser_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "amortized-jacobian-lens-diffsky",
            "--checkpoint",
            "checkpoint.eqx",
        ]
    )

    assert args.command == "amortized-jacobian-lens-diffsky"
    assert args.mode == "decoder"
    assert args.include_prior_score is True
    assert args.include_ae_lens is False
    assert args.posterior_samples == 128
    assert args.posterior_seed == 260722
    assert args.shard_index == 0
    assert args.num_shards == 1


def test_finalize_jacobian_lens_parser() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "amortized-finalize-jacobian-lens",
            "--out",
            "outputs/runs/jlens",
        ]
    )

    assert args.command == "amortized-finalize-jacobian-lens"
    assert args.out == "outputs/runs/jlens"
