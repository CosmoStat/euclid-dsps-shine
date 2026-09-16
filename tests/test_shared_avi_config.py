from scripts.feniks_shared_avi import avi_config


def test_truth_free_prior_frozen_and_source_unchanged():
    source = {
        "truth": {"parameter_columns": {"x": "truth_x"}},
        "extra_columns": ["truth_x"],
        "amortized": {
            "objective": {"mode": "reweighted_wake_sleep"},
            "prior": {"train_jointly": True},
        },
    }
    cfg = avi_config(source)
    assert cfg["truth"]["parameter_columns"] == {}
    assert cfg["extra_columns"] == []
    assert cfg["amortized"]["objective"]["mode"] == "stochastic_elbo"
    assert cfg["amortized"]["prior"]["train_jointly"] is False
    assert source["amortized"]["prior"]["train_jointly"] is True
