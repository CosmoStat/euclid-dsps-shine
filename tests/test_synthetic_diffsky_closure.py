from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from euclid_dsps.config import normalize_config
from euclid_dsps.diffsky_forward_closure import build_trueparam_theta
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.photometry import abmag_to_fnu_cgs
from euclid_dsps.prior_learning.schema import build_truth_schema
from euclid_dsps.synthetic_diffsky import (
    generate_dsps_closure_dataset,
    validate_dsps_closure_dataset,
)
from euclid_dsps.synthetic_diffsky.generation import _pool_is_sufficient
from euclid_dsps.synthetic_diffsky.inference_evaluation import (
    evaluate_closure_inference,
)
from euclid_dsps.synthetic_diffsky.metallicity import (
    absolute_lgmet_to_logzsol,
    lognormal_mdf_weights,
)
from euclid_dsps.synthetic_diffsky.photometry import GROUND_TRUTH_COLUMNS
from euclid_dsps.synthetic_diffsky.population_diagnostics import (
    run_generation_population_diagnostics,
)
from euclid_dsps.synthetic_diffsky.reference_comparison import (
    compare_synthetic_closure_to_reference,
)
from euclid_dsps.synthetic_diffsky.resampling import (
    effective_sample_size,
    resample_weighted_proposals,
)
from euclid_dsps.synthetic_diffsky.selection import (
    apply_photometric_selection,
    apply_proposal_selection,
    photometric_selection_enabled,
)


def test_diffsky_basic_parameter_order_is_canonical_18() -> None:
    assert DIFFSKY_BASIC_PARAMETER_NAMES == (
        "z_obs",
        "log10_stellar_mass",
        "diffstar_lgmcrit",
        "diffstar_lgy_at_mcrit",
        "diffstar_indx_lo",
        "diffstar_indx_hi",
        "diffstar_lg_qt",
        "diffstar_qlglgdt",
        "diffstar_lg_drop",
        "diffstar_lg_rejuv",
        "diffmah_logm0",
        "diffmah_logtc",
        "diffmah_early_index",
        "diffmah_late_index",
        "diffmah_t_peak",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
    )
    assert len(DIFFSKY_BASIC_PARAMETER_NAMES) == 18


def test_ground_truth_mapping_has_one_column_per_free_parameter() -> None:
    assert tuple(GROUND_TRUTH_COLUMNS) == DIFFSKY_BASIC_PARAMETER_NAMES
    assert len(set(GROUND_TRUTH_COLUMNS.values())) == len(DIFFSKY_BASIC_PARAMETER_NAMES)
    assert GROUND_TRUTH_COLUMNS["log10_stellar_metallicity"] == (
        "log10_stellar_metallicity_true"
    )
    assert GROUND_TRUTH_COLUMNS["diffmah_logm0"] == "diffmah_logm0_true"


def test_full_truth_schema_requires_metallicity_truth() -> None:
    frame = _truth_frame().drop(columns=["log10_stellar_metallicity_true"])
    with pytest.raises(ValueError, match="log10_stellar_metallicity"):
        build_trueparam_theta(
            frame,
            {"truth": {"schema": "diffsky_dsps_closure_full"}},
        )


def test_prior_full_truth_schema_uses_canonical_order() -> None:
    schema = build_truth_schema(
        _truth_frame().columns,
        schema_name="diffsky_dsps_closure_full",
        missing_policy="fail",
    )
    assert tuple(param.name for param in schema.parameters) == DIFFSKY_BASIC_PARAMETER_NAMES
    assert schema.reduced is False


def test_absolute_metallicity_conversion_and_clipping() -> None:
    logzsun = np.log10(0.0142)
    grid = logzsun + np.asarray([-2.5, -1.0, 0.0, 0.5])
    converted = absolute_lgmet_to_logzsol(
        np.asarray([logzsun]),
        z_sun=0.0142,
        ssp_lgmet=grid,
        policy="fail",
    )
    assert converted.log10_z_over_zsun[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="exceed the SSP grid"):
        absolute_lgmet_to_logzsol(
            np.asarray([grid.min() - 0.1]),
            z_sun=0.0142,
            ssp_lgmet=grid,
            policy="fail",
        )
    clipped = absolute_lgmet_to_logzsol(
        np.asarray([grid.min() - 0.1, grid.max() + 0.1]),
        z_sun=0.0142,
        ssp_lgmet=grid,
        policy="clip_with_warning",
    )
    assert clipped.clip_low_count == 1
    assert clipped.clip_high_count == 1
    assert clipped.log10_z_over_zsun.tolist() == pytest.approx([-2.5, 0.5])
    assert clipped.lgmet_abs_used.tolist() == pytest.approx(
        [float(grid.min()), float(grid.max())]
    )


def test_lognormal_mdf_weights_are_finite_positive_normalized_continuous() -> None:
    grid = np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0])
    w0 = lognormal_mdf_weights(grid, -1.0, 0.2)
    w1 = lognormal_mdf_weights(grid, -0.99, 0.2)
    assert np.isfinite(w0).all()
    assert np.all(w0 >= 0.0)
    assert w0.sum() == pytest.approx(1.0)
    assert np.max(np.abs(w1 - w0)) < 0.05


def test_diffsky_basic_accepts_single_and_lognormal_mdf_modes() -> None:
    for metallicity_model in ("single", "lognormal_mdf_fixed_scatter"):
        cfg = _base_config_dict("ssp.h5", "out")
        cfg["model"]["stellar_metallicity_model"] = metallicity_model
        normalized = normalize_config(cfg)
        assert normalized["model"]["stellar_metallicity_model"] == metallicity_model


def test_weighted_resampling_and_ess_are_reproducible() -> None:
    proposals = pd.DataFrame(
        {
            "source_proposal_id": [f"p{i}" for i in range(4)],
            "galaxy_weight": [1.0, 2.0, 3.0, 4.0],
        }
    )
    split = type(
        "Split",
        (),
        {
            "name": "train",
            "n_final": 20,
            "resample_seed": 123,
            "object_id_start": 10,
        },
    )()
    result_a = resample_weighted_proposals(proposals, split)
    result_b = resample_weighted_proposals(proposals, split)
    assert effective_sample_size(np.asarray([1.0, 1.0, 1.0])) == pytest.approx(3.0)
    assert result_a.frame["source_proposal_id"].tolist() == result_b.frame[
        "source_proposal_id"
    ].tolist()
    assert result_a.frame["object_id"].iloc[0] == 10


def test_duplication_gate_warns_only_after_max_shards() -> None:
    result = SimpleNamespace(pool_size=500, ess=250.0, duplicate_fraction=0.154)
    split = SimpleNamespace(n_final=100)
    cfg = SimpleNamespace(
        pool_size_factor=4.0,
        min_ess_fraction=2.0,
        max_duplication_fraction=0.10,
        duplication_gate="fail",
    )
    assert not _pool_is_sufficient(result, split, cfg, final_attempt=True)
    cfg.duplication_gate = "warn_after_max_shards"
    assert not _pool_is_sufficient(result, split, cfg, final_attempt=False)
    assert _pool_is_sufficient(result, split, cfg, final_attempt=True)
    cfg.duplication_gate = "warn"
    assert _pool_is_sufficient(result, split, cfg, final_attempt=False)


def test_proposal_selection_filters_mass_and_metallicity_clipping() -> None:
    proposals = pd.DataFrame(
        {
            "logsm_true": [7.9, 8.1, 9.0, 10.0],
            "metallicity_clipped": [False, False, True, False],
            "galaxy_weight": [1.0, 2.0, 3.0, np.nan],
        }
    )
    selected, summary = apply_proposal_selection(
        proposals,
        {"min_logsm": 8.0, "require_metallicity_unclipped": True},
    )
    assert selected["logsm_true"].tolist() == [8.1]
    assert summary["selected_size"] == 1
    assert summary["cuts"]["min_logsm"]["rejected"] == 1


def test_photometric_selection_keeps_negative_noisy_fluxes() -> None:
    frame = pd.DataFrame(
        {
            "flux_true_lsst_u": [10.0, 1.0],
            "flux_lsst_u": [-5.0, 1.0],
            "fluxerr_lsst_u": [1.0, 1.0],
            "flux_true_lsst_g": [8.0, 1.0],
            "flux_lsst_g": [-4.0, 1.0],
            "fluxerr_lsst_g": [1.0, 1.0],
        }
    )
    selected, summary = apply_photometric_selection(
        frame,
        ["lsst_u", "lsst_g"],
        {"snr_threshold": 5.0, "min_true_snr_bands": 2},
    )
    assert len(selected) == 1
    assert selected["flux_lsst_u"].iloc[0] < 0.0
    assert summary["selected_size"] == 1


def test_photometric_selection_supports_magnitude_limits() -> None:
    frame = pd.DataFrame(
        {
            "mag_true_lsst_g": [24.0, 29.0, 25.0],
            "mag_true_lsst_r": [23.5, 28.0, 30.0],
            "flux_true_lsst_g": [10.0, 10.0, 10.0],
            "flux_lsst_g": [10.0, 10.0, 10.0],
            "fluxerr_lsst_g": [1.0, 1.0, 1.0],
            "flux_true_lsst_r": [10.0, 10.0, 10.0],
            "flux_lsst_r": [10.0, 10.0, 10.0],
            "fluxerr_lsst_r": [1.0, 1.0, 1.0],
        }
    )
    selection = {
        "magnitude_limits": {"lsst_g": 26.0, "lsst_r": 26.0},
        "min_magnitude_limit_bands": 2,
    }
    assert photometric_selection_enabled(selection)
    selected, summary = apply_photometric_selection(
        frame,
        ["lsst_g", "lsst_r"],
        selection,
    )
    assert selected.index.tolist() == [0]
    assert selected["n_bands_mag_true_le_limit"].tolist() == [2]
    assert summary["cuts"]["min_magnitude_limit_bands"]["kept"] == 1


def test_lsst_euclid_roman_18_band_preset_resolves() -> None:
    cfg = normalize_config(
        {
            "catalog_path": "dummy.parquet",
            "ssp_path": "dummy.h5",
            "bands": "diffsky_hltds_lsst_euclid_roman_18_fnu_cgs",
        }
    )
    names = [band["name"] for band in cfg["bands"]]
    assert len(names) == 18
    assert names[:6] == ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]
    assert {"euclid_vis", "euclid_nisp_y", "euclid_nisp_j", "euclid_nisp_h"} <= set(names)
    assert {"roman_F062", "roman_F213"} <= set(names)


def test_toy_smoke_generation_validation_and_parquet_roundtrip(tmp_path) -> None:
    pytest.importorskip("diffmah")
    pytest.importorskip("diffstar")
    ssp_path = tmp_path / "ssp.h5"
    _write_synthetic_ssp(ssp_path)
    out = tmp_path / "closure"
    cfg = normalize_config(_base_config_dict(str(ssp_path), str(out)))

    dataset_dir = generate_dsps_closure_dataset(
        cfg,
        split="all",
        smoke=True,
        overwrite=True,
    )
    report = validate_dsps_closure_dataset(
        cfg,
        dataset_dir=dataset_dir,
        sample_size=8,
        batch_size=8,
    )

    assert (dataset_dir / "train.parquet").exists()
    assert (dataset_dir / "schema.json").exists()
    assert report.exists()
    train = pd.read_parquet(dataset_dir / "train.parquet")
    validation = pd.read_parquet(dataset_dir / "validation.parquet")
    assert set(train["object_id"]).isdisjoint(set(validation["object_id"]))
    assert len(train) == 24
    assert (train[[f"flux_lsst_u", f"flux_lsst_g"]] < 0.0).any().any()


def test_layered_toy_generation_writes_survey_and_inference_layers(tmp_path) -> None:
    pytest.importorskip("diffmah")
    pytest.importorskip("diffstar")
    ssp_path = tmp_path / "ssp.h5"
    _write_synthetic_ssp(ssp_path)
    out = tmp_path / "closure_layers"
    cfg = _base_config_dict(str(ssp_path), str(out))
    cfg["synthetic_diffsky"]["smoke_split_sizes"] = {
        "train": 8,
        "validation": 0,
        "test": 0,
    }
    cfg["synthetic_diffsky"]["output_layers"] = {
        "enabled": True,
        "survey_like": {
            "enabled": True,
            "size_factor": 1.0,
            "selection": {"min_true_snr_bands": 0},
        },
        "inference_ready": {
            "enabled": True,
            "size_factor": 1.0,
            "mirror_to_root": True,
            "selection": {"min_true_snr_bands": 0},
        },
    }
    dataset_dir = generate_dsps_closure_dataset(
        normalize_config(cfg),
        split="train",
        smoke=True,
        overwrite=True,
    )
    root = pd.read_parquet(dataset_dir / "train.parquet")
    survey = pd.read_parquet(dataset_dir / "survey_like" / "train.parquet")
    inference = pd.read_parquet(dataset_dir / "inference_ready" / "train.parquet")
    assert len(root) == len(inference) == 8
    assert len(survey) == 8
    assert set(root["sample_layer"]) == {"inference_ready"}
    assert set(survey["sample_layer"]) == {"survey_like"}


def test_closure_inference_evaluation_outputs_metrics(tmp_path) -> None:
    truth = _truth_frame().copy()
    truth.insert(0, "object_id", [10, 11, 12])
    dataset = tmp_path / "test.parquet"
    truth.to_parquet(dataset, index=False)
    run = tmp_path / "run"
    run.mkdir()
    summary = pd.DataFrame({"object_id": truth["object_id"]})
    samples = []
    for _, row in truth.iterrows():
        for sample_id, offset in enumerate([-0.01, 0.0, 0.01]):
            sample = {"object_id": row["object_id"], "sample_id": sample_id}
            for name in DIFFSKY_BASIC_PARAMETER_NAMES:
                sample[name] = row[GROUND_TRUTH_COLUMNS[name]] + offset
            samples.append(sample)
        for name in DIFFSKY_BASIC_PARAMETER_NAMES:
            summary.loc[summary["object_id"] == row["object_id"], f"{name}_q16"] = (
                row[GROUND_TRUTH_COLUMNS[name]] - 0.02
            )
            summary.loc[summary["object_id"] == row["object_id"], f"{name}_median"] = (
                row[GROUND_TRUTH_COLUMNS[name]]
            )
            summary.loc[summary["object_id"] == row["object_id"], f"{name}_q84"] = (
                row[GROUND_TRUTH_COLUMNS[name]] + 0.02
            )
    summary.to_parquet(run / "posterior_summary.parquet", index=False)
    pd.DataFrame(samples).to_parquet(run / "posterior_samples.parquet", index=False)

    report = evaluate_closure_inference(run_dir=run, dataset_path=dataset)

    assert report.exists()
    metrics = pd.read_csv(run / "closure_evaluation" / "posterior_truth_parameter_metrics.csv")
    coverage = pd.read_csv(run / "closure_evaluation" / "posterior_truth_coverage.csv")
    assert set(metrics["parameter"]) == set(DIFFSKY_BASIC_PARAMETER_NAMES)
    assert {"50", "68", "90", "95"} <= set(coverage["interval"].astype(str))


def test_reference_comparison_writes_population_and_photometry_tables(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    synthetic = pd.DataFrame(
        {
            "redshift_true": [0.05, 0.1, 0.2, 0.3],
            "logsm_true": [9.0, 9.5, 10.0, 10.5],
            "logsfr_true": [-1.0, -0.5, 0.0, 0.5],
            "logssfr_true": [-10.0, -10.0, -10.0, -10.0],
            "central_true": [True, False, True, False],
            "dust_av_true": [0.1, 0.2, 0.3, 0.4],
            "dust_delta_true": [-0.4, -0.3, -0.2, -0.1],
            "mag_true_lsst_g": [24.0, 23.5, 23.0, 22.5],
            "mag_true_lsst_r": [23.8, 23.2, 22.9, 22.4],
            "mag_true_lsst_i": [23.5, 23.0, 22.7, 22.1],
            "flux_true_lsst_g": [1.0, 2.0, 3.0, 4.0],
            "flux_lsst_g": [0.9, 2.1, 2.8, 4.2],
            "fluxerr_lsst_g": [0.1, 0.1, 0.2, 0.2],
            "mask_lsst_g": [True, True, True, True],
            "flux_true_lsst_r": [1.1, 2.1, 3.1, 4.1],
            "flux_lsst_r": [1.0, 2.2, 3.0, 4.3],
            "fluxerr_lsst_r": [0.1, 0.1, 0.2, 0.2],
            "mask_lsst_r": [True, True, False, True],
            "flux_true_lsst_i": [1.2, 2.2, 3.2, 4.2],
            "flux_lsst_i": [1.1, 2.3, 3.1, 4.4],
            "fluxerr_lsst_i": [0.1, 0.1, 0.2, 0.2],
            "mask_lsst_i": [True, True, True, True],
        }
    )
    reference = pd.DataFrame(
        {
            "redshift_true": [0.04, 0.11, 0.19, 0.31, 0.33],
            "logsm_true": [8.9, 9.4, 9.9, 10.4, 10.6],
            "logsfr_true": [-1.1, -0.6, 0.1, 0.6, 0.7],
            "logssfr_true": [-10.0, -10.0, -9.8, -9.8, -9.9],
            "central_true": [True, False, True, False, True],
            "dust_av": [0.1, 0.2, 0.25, 0.4, 0.5],
            "dust_delta": [-0.4, -0.3, -0.25, -0.1, -0.05],
            "mag_lsst_g": [24.1, 23.4, 23.1, 22.6, 22.3],
            "mag_lsst_r": [23.9, 23.1, 22.8, 22.5, 22.2],
            "mag_lsst_i": [23.6, 22.9, 22.5, 22.1, 21.9],
            "flux_lsst_g": [1.0, 2.0, 2.9, 4.1, 4.4],
            "fluxerr_lsst_g": [0.1, 0.1, 0.2, 0.2, 0.3],
            "mask_lsst_g": [True, True, True, True, False],
            "flux_lsst_r": [1.1, 2.1, 3.0, 4.2, 4.5],
            "fluxerr_lsst_r": [0.1, 0.1, 0.2, 0.2, 0.3],
            "mask_lsst_r": [True, True, True, False, False],
            "flux_lsst_i": [1.2, 2.2, 3.1, 4.3, 4.6],
            "fluxerr_lsst_i": [0.1, 0.1, 0.2, 0.2, 0.3],
            "mask_lsst_i": [True, True, True, True, False],
        }
    )
    synthetic_path = tmp_path / "synthetic.parquet"
    reference_path = tmp_path / "reference.parquet"
    synthetic.to_parquet(synthetic_path, index=False)
    reference.to_parquet(reference_path, index=False)

    outputs = compare_synthetic_closure_to_reference(
        synthetic_path=synthetic_path,
        reference_path=reference_path,
        out_dir=tmp_path / "comparison",
        bands=("lsst_g", "lsst_r", "lsst_i"),
        plots=True,
    )

    assert outputs["report"].exists()
    distribution = pd.read_csv(outputs["distribution_metrics"])
    photometry = pd.read_csv(outputs["photometry_metrics"])
    assert "redshift" in set(distribution["quantity"])
    assert "lsst_g-lsst_r" in set(photometry["quantity"])
    assert "mask_fraction" in set(photometry["group"])
    summary = json.loads(outputs["summary"].read_text())
    assert any(
        "color_color_reference_black_synthetic_green" in path
        for path in summary["plot_paths"]
    )


def test_reference_comparison_converts_fs2_flux_columns_to_ab_magnitudes(tmp_path) -> None:
    synthetic = pd.DataFrame(
        {
            "redshift_true": [0.5, 0.7, 1.0],
            "logsm_true": [9.0, 9.5, 10.0],
            "mag_true_lsst_g": [24.0, 23.5, 23.0],
            "mag_true_lsst_r": [23.7, 23.2, 22.9],
            "mag_true_euclid_vis": [23.4, 22.9, 22.5],
            "mag_true_euclid_nisp_y": [23.1, 22.6, 22.3],
            "flux_true_lsst_g": [1.0, 2.0, 3.0],
            "flux_lsst_g": [1.0, 2.0, 3.0],
            "fluxerr_lsst_g": [0.1, 0.1, 0.1],
            "mask_lsst_g": [True, True, True],
        }
    )
    reference = pd.DataFrame(
        {
            "z_true_gal": [0.45, 0.75, 1.05],
            "log_stellar_mass": [9.2, 9.8, 10.2],
            "log_sfr_true": [-0.5, 0.0, 0.5],
            "dust_ebv_true": [0.05, 0.1, 0.2],
            "metallicity_true": [10.0, 10.1, 10.2],
            "lsst_g": abmag_to_fnu_cgs(np.asarray([24.1, 23.4, 23.1])),
            "lsst_r": abmag_to_fnu_cgs(np.asarray([23.8, 23.1, 22.8])),
            "euclid_vis": abmag_to_fnu_cgs(np.asarray([23.5, 22.8, 22.4])),
            "euclid_nisp_y": abmag_to_fnu_cgs(np.asarray([23.2, 22.5, 22.1])),
        }
    )
    synthetic_path = tmp_path / "synthetic.parquet"
    reference_path = tmp_path / "fs2.parquet"
    synthetic.to_parquet(synthetic_path, index=False)
    reference.to_parquet(reference_path, index=False)

    outputs = compare_synthetic_closure_to_reference(
        synthetic_path=synthetic_path,
        reference_path=reference_path,
        out_dir=tmp_path / "fs2_comparison",
        bands=("lsst_g", "lsst_r", "euclid_vis", "euclid_nisp_y"),
        plots=False,
        reference_kind="fs2",
    )

    summary = json.loads(outputs["summary"].read_text())
    photometry = pd.read_csv(outputs["photometry_metrics"])
    assert summary["reference_kind"] == "fs2"
    assert summary["reference_label"] == "Euclid FS2 phz1"
    assert summary["reference_photometry_units"]["lsst_g"]["mag_column"] == "mag_lsst_g"
    assert summary["reference_photometry_units"]["lsst_g"]["flux_column"] == "flux_lsst_g"
    assert summary["reference_photometry_units"]["lsst_g"]["mag_from_flux"] is True
    mag_g = photometry[
        (photometry["group"] == "mag_true_vs_reference_mag")
        & (photometry["quantity"] == "lsst_g")
    ].iloc[0]
    assert mag_g["status"] == "ok"
    assert mag_g["reference_column"] == "mag_lsst_g"
    assert mag_g["reference_median"] == pytest.approx(23.4)
    flux_g = photometry[
        (photometry["group"] == "flux_true_vs_reference_flux")
        & (photometry["quantity"] == "lsst_g")
    ].iloc[0]
    assert flux_g["reference_column"] == "flux_lsst_g"
    color = photometry[
        (photometry["group"] == "color_true_vs_reference_color")
        & (photometry["quantity"] == "euclid_vis-euclid_nisp_y")
    ].iloc[0]
    assert color["status"] == "ok"
    assert color["reference_column"] == "mag_euclid_vis-mag_euclid_nisp_y"
    assert color["reference_median"] == pytest.approx(0.3)


def test_population_diagnostics_write_stats_and_proposal_metrics(tmp_path) -> None:
    dataset_dir = tmp_path / "closure"
    proposal_root = dataset_dir / "proposals"
    dataset_dir.mkdir()
    for split, offset in (("train", 0), ("validation", 10), ("test", 20)):
        frame = _truth_frame().copy()
        frame["object_id"] = np.arange(offset, offset + len(frame))
        frame["split"] = split
        frame["logsfr_true"] = frame["logsm_true"] - 10.0
        frame["logssfr_true"] = -10.0
        frame["logmp_true"] = frame["logsm_true"] + 1.0
        frame["logmp0_true"] = frame["logsm_true"] + 1.2
        frame["central_true"] = [True, False, True]
        for band, base in (("lsst_u", 24.0), ("lsst_g", 23.5)):
            frame[f"mag_true_{band}"] = base - 0.1 * np.arange(len(frame))
            frame[f"flux_true_{band}"] = np.linspace(1.0, 3.0, len(frame))
            frame[f"flux_{band}"] = frame[f"flux_true_{band}"] + 0.01
            frame[f"fluxerr_{band}"] = 0.1
            frame[f"mask_{band}"] = True
        frame.to_parquet(dataset_dir / f"{split}.parquet", index=False)
        proposals = frame.copy()
        proposals["galaxy_weight"] = np.linspace(1.0, 3.0, len(proposals))
        split_proposals = proposal_root / split
        split_proposals.mkdir(parents=True)
        proposals.to_parquet(split_proposals / "shard_00000.parquet", index=False)

    cfg = _base_config_dict("ssp.h5", str(dataset_dir))
    cfg["synthetic_diffsky"]["diagnostics"] = {
        "enabled": True,
        "make_plots": False,
        "make_corner": False,
        "reference_dataset": None,
        "max_rows": 100,
    }

    summary = run_generation_population_diagnostics(
        cfg,
        dataset_dir=dataset_dir,
        smoke=True,
    )

    assert summary is not None and summary.exists()
    out = summary.parent
    assert (out / "parameter_stats.csv").exists()
    assert (out / "photometry_stats.csv").exists()
    assert (out / "error_model_stats.csv").exists()
    assert (out / "color_stats.csv").exists()
    assert (out / "proposal_vs_final_metrics.csv").exists()
    parameter_stats = pd.read_csv(out / "parameter_stats.csv")
    error_model_stats = pd.read_csv(out / "error_model_stats.csv")
    proposal_metrics = pd.read_csv(out / "proposal_vs_final_metrics.csv")
    assert "z_obs" in set(parameter_stats["quantity"])
    assert set(error_model_stats["band"]) == {"lsst_u", "lsst_g"}
    assert "residual_std" in error_model_stats
    assert "logsm" in set(proposal_metrics["quantity"])


def _truth_frame() -> pd.DataFrame:
    n = 3
    data = {
        "redshift_true": np.linspace(0.1, 0.3, n),
        "logsm_true": np.linspace(9.0, 10.0, n),
        "log10_stellar_metallicity_true": np.linspace(-0.8, -0.3, n),
    }
    for name in DIFFSKY_BASIC_PARAMETER_NAMES:
        if name in {"z_obs", "log10_stellar_mass", "log10_stellar_metallicity"}:
            continue
        data[f"{name}_true"] = np.linspace(0.1, 0.3, n)
    return pd.DataFrame(data)


def _write_synthetic_ssp(path) -> None:
    wave = np.linspace(1000.0, 12000.0, 48).astype(np.float32)
    lg_age = np.linspace(-3.0, 0.9, 16).astype(np.float32)
    lgmet = np.asarray([-2.0, -1.4, -0.8, -0.2, 0.2], dtype=np.float32)
    met_factor = np.linspace(0.8, 1.2, len(lgmet))[:, None, None]
    age_factor = np.linspace(1.5, 0.5, len(lg_age))[None, :, None]
    wave_factor = (1.0 + 0.2 * wave / wave.max())[None, None, :]
    flux = (1.0e-3 * met_factor * age_factor * wave_factor).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle["ssp_wave"] = wave
        handle["ssp_lg_age_gyr"] = lg_age
        handle["ssp_lgmet"] = lgmet
        handle["ssp_flux"] = flux
        handle.attrs["imf_type"] = 1
        handle.attrs["imf_name"] = "chabrier"
        handle.attrs["z_sun"] = 0.0142


def _base_config_dict(ssp_path: str, out: str) -> dict:
    return {
        "catalog_path": f"{out}/train.parquet",
        "ssp_path": ssp_path,
        "bands": [
            {
                "name": "lsst_u",
                "column": "flux_lsst_u",
                "units": "fnu_cgs",
                "error_column": "fluxerr_lsst_u",
                "error_units": "fnu_cgs",
                "filter": {"kind": "tophat", "wave_min": 3200.0, "wave_max": 4000.0},
            },
            {
                "name": "lsst_g",
                "column": "flux_lsst_g",
                "units": "fnu_cgs",
                "error_column": "fluxerr_lsst_g",
                "error_units": "fnu_cgs",
                "filter": {"kind": "tophat", "wave_min": 4000.0, "wave_max": 5500.0},
            },
        ],
        "synthetic_diffsky": {
            "output_dir": out,
            "proposal_backend": "toy",
            "split_sizes": {"train": 24, "validation": 8, "test": 8},
            "smoke_split_sizes": {"train": 24, "validation": 8, "test": 8},
            "n_host_halos_per_shard": 64,
            "max_shards": 3,
            "jax_batch_size": 8,
            "z_min": 0.001,
            "z_max": 0.35,
            "pool_size_factor": 1.0,
            "min_ess_fraction": 0.2,
            "max_duplication_fraction": 0.95,
            "metallicity_grid_policy": "clip_with_warning",
            "flux_error_model": {
                "type": "fractional_snr",
                "snr": 0.1,
                "min_sigma_fnu_cgs": 1.0e-40,
            },
        },
        "truth": {"schema": "diffsky_dsps_closure_full"},
        "model": {
            "n_sfh_bins": 24,
            "sfh_model": "diffsky_basic",
            "ssp_model": "dense",
            "stellar_metallicity_model": "lognormal_mdf_fixed_scatter",
            "stellar_metallicity_scatter_dex": 0.2,
            "dust_model": "prospector_fsps",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "none",
            "z_sun": 0.0142,
            "fixed_parameters": {},
        },
        "fit": {
            "free_parameters": {
                name: {"initial": 0.0, "bounds": [-20.0, 20.0]}
                for name in DIFFSKY_BASIC_PARAMETER_NAMES
            }
        },
    }
