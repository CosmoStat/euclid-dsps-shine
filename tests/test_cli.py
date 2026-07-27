from __future__ import annotations

from euclid_dsps import cli


def test_diffsky_map_adam_prior_dispatches_to_config_backed_handler(
    monkeypatch,
) -> None:
    called: dict[str, bool] = {}

    def fail_diffsky_data_dispatch(args) -> None:
        raise AssertionError(f"unexpected diffsky_data dispatch for {args.command}")

    def fake_load_config(path: str) -> dict:
        called["load_config"] = True
        return {"runtime": {}, "amortized": {"training": {}, "map_adam": {}}}

    def fake_apply_runtime(runtime: dict) -> None:
        called["runtime"] = True

    def fake_map_handler(config: dict, args) -> None:
        called["map_handler"] = True
        assert args.command == "diffsky-map-adam-prior"
        assert args.checkpoint == "checkpoint.eqx"

    monkeypatch.setattr(
        "euclid_dsps.diffsky_data.cli.run_diffsky_command",
        fail_diffsky_data_dispatch,
    )
    monkeypatch.setattr("euclid_dsps.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "euclid_dsps.jax_runtime.apply_jax_runtime_env",
        fake_apply_runtime,
    )
    monkeypatch.setattr(cli, "_run_diffsky_map_adam_prior", fake_map_handler)

    cli.main(
        [
            "--config",
            "config.yaml",
            "diffsky-map-adam-prior",
            "--checkpoint",
            "checkpoint.eqx",
        ]
    )

    assert called == {
        "load_config": True,
        "runtime": True,
        "map_handler": True,
    }


def test_diffsky_compare_closure_reference_dispatches_to_config_backed_handler(
    monkeypatch,
) -> None:
    called: dict[str, bool] = {}

    def fail_diffsky_data_dispatch(args) -> None:
        raise AssertionError(f"unexpected diffsky_data dispatch for {args.command}")

    def fake_load_config(path: str) -> dict:
        called["load_config"] = True
        return {
            "runtime": {},
            "bands": [
                {"name": "lsst_g"},
                {"name": "lsst_r"},
            ],
        }

    def fake_apply_runtime(runtime: dict) -> None:
        called["runtime"] = True

    def fake_compare_handler(config: dict, args) -> None:
        called["compare_handler"] = True
        assert args.command == "diffsky-compare-dsps-closure-reference"
        assert args.synthetic == "synthetic.parquet"
        assert args.reference == "reference.parquet"
        assert args.out == "out"

    monkeypatch.setattr(
        "euclid_dsps.diffsky_data.cli.run_diffsky_command",
        fail_diffsky_data_dispatch,
    )
    monkeypatch.setattr("euclid_dsps.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "euclid_dsps.jax_runtime.apply_jax_runtime_env",
        fake_apply_runtime,
    )
    monkeypatch.setattr(
        cli,
        "_run_diffsky_compare_dsps_closure_reference",
        fake_compare_handler,
    )

    cli.main(
        [
            "--config",
            "config.yaml",
            "diffsky-compare-dsps-closure-reference",
            "--synthetic",
            "synthetic.parquet",
            "--reference",
            "reference.parquet",
            "--out",
            "out",
        ]
    )

    assert called == {
        "load_config": True,
        "runtime": True,
        "compare_handler": True,
    }


def test_diffsky_plan_prior_workflow_dispatches_to_config_backed_handler(
    monkeypatch,
) -> None:
    called: dict[str, bool] = {}

    def fail_diffsky_data_dispatch(args) -> None:
        raise AssertionError(f"unexpected diffsky_data dispatch for {args.command}")

    def fake_load_config(path: str) -> dict:
        called["load_config"] = True
        assert path == "config.yaml"
        return {"runtime": {}, "synthetic_diffsky": {"output_dir": "Data/feniks"}}

    def fake_apply_runtime(runtime: dict) -> None:
        called["runtime"] = True

    def fake_plan_handler(config: dict, args) -> None:
        called["plan_handler"] = True
        assert args.command == "diffsky-plan-prior-workflow"
        assert args.prior_config == "prior.yaml"
        assert args.amortized_config == "amortized.yaml"
        assert args.out == "out"

    monkeypatch.setattr(
        "euclid_dsps.diffsky_data.cli.run_diffsky_command",
        fail_diffsky_data_dispatch,
    )
    monkeypatch.setattr("euclid_dsps.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "euclid_dsps.jax_runtime.apply_jax_runtime_env",
        fake_apply_runtime,
    )
    monkeypatch.setattr(cli, "_run_diffsky_prior_workflow_plan", fake_plan_handler)

    cli.main(
        [
            "--config",
            "config.yaml",
            "diffsky-plan-prior-workflow",
            "--prior-config",
            "prior.yaml",
            "--amortized-config",
            "amortized.yaml",
            "--out",
            "out",
        ]
    )

    assert called == {
        "load_config": True,
        "runtime": True,
        "plan_handler": True,
    }
