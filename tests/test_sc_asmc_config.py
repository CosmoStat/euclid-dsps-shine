from __future__ import annotations

import copy

import pytest

from euclid_dsps.amortized.sc_asmc_config import (
    sc_asmc_em_config_hash,
    sc_asmc_em_hierarchy,
    validate_sc_asmc_em_config,
)
from euclid_dsps.config import load_config

CONFIG = "configs/experiments/feniks_sc_asmc_em_r25.yaml"


def test_final_sc_asmc_config_is_exact_and_no_truth() -> None:
    config = load_config(CONFIG)
    original = copy.deepcopy(config)
    receipt = validate_sc_asmc_em_config(config)
    hierarchy = sc_asmc_em_hierarchy(config)

    assert receipt["status"] == "valid"
    assert receipt["truth_used"] is False
    assert receipt["input_dim"] == 54
    assert receipt["schedule"].outer_iterations == 2
    assert hierarchy.primary.n_particles == 64
    assert hierarchy.fallback.n_particles == 128
    assert hierarchy.extended.max_stages == 48
    assert config == original


def test_final_model_trainable_parameter_count() -> None:
    import equinox as eqx
    import jax

    from euclid_dsps.amortized.latent import latent_spec_from_config
    from euclid_dsps.amortized.train import build_amortized_model

    config = load_config(CONFIG)
    model = build_amortized_model(
        config,
        jax.random.PRNGKey(0),
        latent_spec=latent_spec_from_config(config),
    )

    def count(tree) -> int:
        return sum(
            int(leaf.size)
            for leaf in jax.tree_util.tree_leaves(tree)
            if eqx.is_inexact_array(leaf)
        )

    assert count(model.encoder) == 2_441_298
    assert count(model.prior) == 1_179_888
    assert count(model.encoder) + count(model.prior) == 3_621_186


def test_config_hash_matches_truth_stripped_runtime() -> None:
    config = load_config(CONFIG)
    runtime = copy.deepcopy(config)
    runtime["truth"] = {"parameter_columns": {}}

    assert sc_asmc_em_config_hash(runtime) == sc_asmc_em_config_hash(config)
    runtime["amortized"]["sc_asmc_em"]["q_distillation"]["epochs"] = 4
    assert sc_asmc_em_config_hash(runtime) != sc_asmc_em_config_hash(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["truth"]["parameter_columns"].update(
                {"z_obs": {"column": "z_obs"}}
            ),
            "truth.parameter_columns",
        ),
        (
            lambda cfg: cfg["amortized"]["sc_asmc_em"].update({"outer_iterations": 3}),
            "outer_iterations",
        ),
        (
            lambda cfg: cfg["amortized"]["likelihood"].update({"type": "student_t"}),
            "Gaussian",
        ),
        (
            lambda cfg: cfg["amortized"]["prior"].update({"checkpoint": "old.eqx"}),
            "warm starts",
        ),
    ],
)
def test_final_sc_asmc_config_fails_closed(mutation, message: str) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_sc_asmc_em_config(config)
