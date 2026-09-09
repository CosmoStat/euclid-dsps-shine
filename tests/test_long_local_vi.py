import pytest

from scripts.feniks_qualified_local_vi import (
    long_optimization_recipe,
    optimization_seed,
    prepare,
)


def test_fixed_recipe_and_nonoverlapping_streams():
    recipe = long_optimization_recipe()
    assert recipe["steps"] == 512
    assert recipe["trajectory_steps"] == [64, 128, 256]
    assert recipe["evaluation_draws"] == 512
    assert recipe["gradient_draws"] == 32
    assert recipe["learning_rate"] == 0.0001
    training, evaluation = set(), set()
    for case in range(16):
        seed = 260910 + 3000000 + case * 100000
        evaluation.update([seed, seed + 1])
        for start in range(2):
            evaluation.update([seed + 500 + start * 10 + r for r in range(2)])
            for step in range(512):
                key = optimization_seed(recipe, seed, start, step)
                assert key not in training
                training.add(key)
    assert training.isdisjoint(evaluation)
    # Gradient work plus four evaluations and the baseline, with audit margin.
    assert 16 * (2 * 512 * 32 + 9 * 1024) < recipe["maximum_decoder_evaluations"]


@pytest.mark.parametrize(
    "extra",
    [{"controlled": True}, {"support_probe_root": "old"}, {"objects": 2}, {"arm": "B"}],
)
def test_long_recipe_rejects_changed_protocol(tmp_path, extra):
    with pytest.raises(ValueError):
        prepare(
            tmp_path / "out",
            tmp_path / "source",
            tmp_path / "night",
            long_optimization=True,
            **extra,
        )
    assert not (tmp_path / "out").exists()
