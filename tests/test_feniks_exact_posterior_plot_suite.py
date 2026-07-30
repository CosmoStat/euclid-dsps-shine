from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.regenerate_feniks_exact_posterior_plots import (
    KEY_PARAMETERS,
    PLOT_VARIANTS,
    TRACE_VARIANTS,
    _comparison_legend,
    _corner_figure,
    _density_curve,
    _expected_prior_label,
    _parameter_ranges,
    _validate_prior_identity,
)


def test_pairwise_variants_keep_prior_and_encoder_as_the_baseline() -> None:
    assert PLOT_VARIANTS == (
        ("prior_encoder", None),
        ("prior_encoder_nuts", "nuts"),
        ("prior_encoder_map", "map"),
        ("prior_encoder_is", "is"),
    )
    assert TRACE_VARIANTS == (
        ("prior_encoder_nuts", None),
        ("prior_encoder_nuts_map", "map"),
        ("prior_encoder_nuts_is", "is"),
    )
    assert len(KEY_PARAMETERS) == 5


def test_plot_legends_name_the_requested_scientific_methods() -> None:
    baseline = [
        handle.get_label()
        for handle in _comparison_legend(None, include_nuts_chains=False)
    ]
    nuts = [
        handle.get_label()
        for handle in _comparison_legend("nuts", include_nuts_chains=False)
    ]
    map_labels = [
        handle.get_label()
        for handle in _comparison_legend("map", include_nuts_chains=False)
    ]
    importance = [
        handle.get_label()
        for handle in _comparison_legend("is", include_nuts_chains=False)
    ]
    assert baseline == ["Learned prior", "Encoder posterior", "Truth"]
    assert "NUTS (not converged)" in nuts
    assert "MAP" in map_labels
    assert "Encoder + IS" in importance


def test_prior_identity_must_match_the_checkpoint_run_label() -> None:
    contract = {
        "checkpoint": (
            "outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2/"
            "train/checkpoints/best.eqx"
        )
    }
    label = _expected_prior_label(contract)
    assert label == "rws_k8_t2_seed2"
    _validate_prior_identity(
        Path(
            "outputs/jean_zay_20260724/feniks_selfsup_paper_v1/"
            "rws_k8_t2_seed2/inference/learned_prior_samples.parquet"
        ),
        label,
    )
    try:
        _validate_prior_identity(
            Path("outputs/another_run/learned_prior_samples.parquet"),
            label,
        )
    except ValueError as error:
        assert "do not belong" in str(error)
    else:
        raise AssertionError("A learned-prior provenance mismatch must fail")


def test_shared_ranges_include_posteriors_points_and_physical_bounds() -> None:
    names = ("z_obs", "dust_av")
    prior = pd.DataFrame(
        {
            "z_obs": np.linspace(0.2, 2.0, 1000),
            "dust_av": np.linspace(0.0, 1.0, 1000),
        }
    )
    encoder = pd.DataFrame(
        {
            "z_obs": np.linspace(0.8, 1.2, 500),
            "dust_av": np.linspace(0.2, 0.4, 500),
        }
    )
    importance = encoder.copy()
    nuts = encoder.copy()
    truth = pd.Series({"z_obs": 2.4, "dust_av": 1.4})
    map_best = pd.Series({"z_obs": 0.6, "dust_av": 0.1})
    ranges = _parameter_ranges(
        names,
        prior=prior,
        posterior_frames=(encoder, importance, nuts),
        points=(truth, map_best),
        lower=[0.001, 0.0],
        upper=[5.5, 6.0],
    )
    assert ranges["z_obs"][0] >= 0.001
    assert ranges["z_obs"][1] > 2.4
    assert ranges["dust_av"][0] >= 0.0
    assert ranges["dust_av"][1] > 1.4


def test_smoothed_density_curve_is_finite_and_normalized() -> None:
    grid, density = _density_curve(np.linspace(0.0, 1.0, 1000), 0.0, 1.0)
    assert grid.shape == density.shape
    assert np.isfinite(density).all()
    assert np.all(density >= 0.0)
    assert np.trapezoid(density, grid) > 0.95


def test_smoothed_corner_layers_render_with_current_corner_version() -> None:
    parameters = ("z_obs", "dust_av")
    frames = {
        "prior": pd.DataFrame(
            {
                "z_obs": np.linspace(0.1, 2.2, 300),
                "dust_av": np.linspace(0.0, 1.1, 300),
            }
        ),
        "encoder": pd.DataFrame(
            {
                "z_obs": np.linspace(0.7, 1.3, 300),
                "dust_av": np.linspace(0.2, 0.5, 300),
            }
        ),
        "nuts": pd.DataFrame(
            {
                "z_obs": np.linspace(0.8, 1.4, 300),
                "dust_av": np.linspace(0.15, 0.55, 300),
            }
        ),
    }
    figure = _corner_figure(
        frames,
        parameters,
        ranges={"z_obs": [0.0, 2.5], "dust_av": [0.0, 1.5]},
        truth=pd.Series({"z_obs": 1.0, "dust_av": 0.3}),
        map_best=pd.Series({"z_obs": 1.1, "dust_av": 0.35}),
        comparison="nuts",
        title="test galaxy",
        nuts_warning="NUTS not converged",
        max_samples=250,
    )
    assert len(figure.axes) == 4
    plt.close(figure)
