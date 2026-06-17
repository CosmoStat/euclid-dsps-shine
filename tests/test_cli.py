from __future__ import annotations

from euclid_dsps import cli


def test_diffsky_map_adam_prior_dispatches_to_config_backed_handler(monkeypatch) -> None:
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
