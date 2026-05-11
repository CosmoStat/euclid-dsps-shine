from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.config import normalize_config
from euclid_dsps.cosmos import (
    CosmosSedError,
    MissingCosmosColumnsError,
    apply_cosmos_extinction,
    component_fractions,
    cosmos_catalog_columns,
    fnu_to_flambda,
    load_cosmos_sed_resources,
    photometry_target_sets,
    reconstruct_cosmos_proxy_sed,
    synthetic_fnu_from_flambda,
    validate_cosmos_catalog,
)
from euclid_dsps.filters import FilterCurve


def test_synthetic_fnu_from_flambda_recovers_flat_fnu() -> None:
    wave = np.linspace(1000.0, 10_000.0, 4000)
    expected_fnu = 2.5e-17
    flambda = fnu_to_flambda(wave, expected_fnu)
    filt = FilterCurve(
        name="euclid_vis",
        wave=np.linspace(4300.0, 9000.0, 300),
        transmission=np.ones(300),
        source="toy",
    )

    assert synthetic_fnu_from_flambda(wave, flambda, filt, "photon") == pytest.approx(
        expected_fnu, rel=1.0e-4
    )
    assert synthetic_fnu_from_flambda(wave, flambda, filt, "energy") == pytest.approx(
        expected_fnu, rel=1.0e-4
    )


def test_load_cosmos_templates_uses_list_order(tmp_path) -> None:
    config = _cosmos_config(tmp_path)
    resources = load_cosmos_sed_resources(config["cosmos_sed"])

    assert [template.name for template in resources.templates] == [
        "template_0.sed",
        "template_1.sed",
    ]
    assert resources.templates[0].template_id == 0
    assert resources.templates[1].template_id == 1
    assert resources.extinction_mapping[1] == "SMC_prevot"


def test_apply_cosmos_extinction_uses_configured_curve(tmp_path) -> None:
    config = _cosmos_config(tmp_path)
    resources = load_cosmos_sed_resources(config["cosmos_sed"])
    wave = np.asarray([1000.0, 2000.0])
    flux = np.ones(2)

    attenuated = apply_cosmos_extinction(
        wave, flux, ebv=0.5, curve_code=1, resources=resources
    )

    assert attenuated.tolist() == pytest.approx([10 ** (-0.4 * 0.5 * 2.0)] * 2)


def test_reconstruct_cosmos_proxy_sed_normalizes_to_abs_flux(tmp_path) -> None:
    config = _cosmos_config(tmp_path)
    resources = load_cosmos_sed_resources(config["cosmos_sed"])
    filt = FilterCurve(
        name="euclid_vis",
        wave=np.linspace(1200.0, 3900.0, 200),
        transmission=np.ones(200),
        source="toy",
    )
    row = {
        "sed_cosmos_1": 0,
        "sed_cosmos_2": 1,
        "ebv_cosmos_1": 0.0,
        "ebv_cosmos_2": 0.0,
        "ext_curve_cosmos_1": 0,
        "ext_curve_cosmos_2": 0,
        "frac_cosmos_1": 0.7,
        "frac_cosmos_2": 0.3,
        "euclid_vis_abs": 3.0e-15,
    }

    result = reconstruct_cosmos_proxy_sed(
        row,
        7,
        resources,
        {"euclid_vis": filt},
        config["bands"],
        config["cosmos_sed"],
    )

    assert result.alpha == pytest.approx(3.0, rel=1.0e-4)
    assert result.synthetic_abs_fluxes_after["euclid_vis"] == pytest.approx(3.0e-15)
    assert result.diagnostics["component_fraction_policy_used"] == "as_catalog"


def test_component_fractions_equal_if_missing_is_explicit_fallback() -> None:
    f1, f2, diagnostics = component_fractions(
        {}, {"component_fraction_policy": "equal_if_missing"}
    )

    assert f1 == pytest.approx(0.5)
    assert f2 == pytest.approx(0.5)
    assert diagnostics["component_fraction_policy_used"] == "equal_if_missing"


def test_component_fractions_strict_missing_fails() -> None:
    with pytest.raises(MissingCosmosColumnsError, match="frac_cosmos_1"):
        component_fractions({}, {"component_fraction_policy": "strict"})


def test_component_fractions_normalize_nonunit_sum() -> None:
    f1, f2, diagnostics = component_fractions(
        {"frac_cosmos_1": 2.0, "frac_cosmos_2": 1.0},
        {"component_fraction_policy": "strict"},
    )

    assert f1 == pytest.approx(2.0 / 3.0)
    assert f2 == pytest.approx(1.0 / 3.0)
    assert diagnostics["component_fraction_policy_used"] == "normalized"


def test_validate_cosmos_catalog_reports_invalid_ids(tmp_path) -> None:
    config = _cosmos_config(tmp_path)
    df = pd.DataFrame(
        {
            "sed_cosmos_1": [0, 31],
            "sed_cosmos_2": [1, 1],
            "ebv_cosmos_1": [0.0, 0.1],
            "ebv_cosmos_2": [0.0, 0.1],
            "ext_curve_cosmos_1": [0, 1],
            "ext_curve_cosmos_2": [0, 5],
            "frac_cosmos_1": [0.6, 0.4],
            "frac_cosmos_2": [0.4, 0.6],
            "euclid_vis_abs": [1.0e-15, 1.0e-15],
        }
    )

    report = validate_cosmos_catalog(df, config, available_columns=set(df.columns))

    assert report["sed_cosmos_1_invalid_count"] == 1
    assert report["ext_curve_cosmos_2_invalid_count"] == 1
    assert report["missing_optional_fraction_columns"] == []
    assert report["frac_sum_not_close_to_one_count"] == 0


def test_cosmos_catalog_columns_strict_includes_fraction_columns(tmp_path) -> None:
    config = _cosmos_config(tmp_path)

    columns = cosmos_catalog_columns(config, include_optional=False)

    assert "frac_cosmos_1" in columns
    assert "frac_cosmos_2" in columns


def test_invalid_fraction_values_fail() -> None:
    with pytest.raises(CosmosSedError, match="Invalid COSMOS component fractions"):
        component_fractions(
            {"frac_cosmos_1": -1.0, "frac_cosmos_2": 1.0},
            {"component_fraction_policy": "strict"},
        )


def test_photometry_target_sets_continuum_uses_all_configured_bands() -> None:
    bands = [
        {"name": "lsst_u", "column": "lsst_u"},
        {"name": "euclid_vis", "column": "euclid_vis"},
    ]

    target_sets = photometry_target_sets(bands, ["continuum_internal_dust"])

    assert [item["name"] for item in target_sets] == ["continuum_internal_dust"]
    assert [item["band_name"] for item in target_sets[0]["bands"]] == [
        "lsst_u",
        "euclid_vis",
    ]
    assert target_sets[0]["bands"][0]["target_column"] == "lsst_u"


def test_photometry_target_sets_noisy_uses_euclid_error_columns_only() -> None:
    bands = [
        {"name": "lsst_u", "column": "lsst_u"},
        {"name": "euclid_vis", "column": "euclid_vis"},
    ]

    target_sets = photometry_target_sets(bands, ["noisy_observation"])

    assert [item["band_name"] for item in target_sets[0]["bands"]] == ["euclid_vis"]
    assert (
        target_sets[0]["bands"][0]["error_column"]
        == "euclid_vis_el_model3_ext_odonnell_ext_error"
    )


def _cosmos_config(tmp_path) -> dict:
    lephare = tmp_path / "lephare"
    sed_dir = lephare / "sed" / "GAL" / "COSMOS_SED"
    ext_dir = lephare / "ext"
    sed_dir.mkdir(parents=True)
    ext_dir.mkdir(parents=True)
    wave = np.linspace(1000.0, 4000.0, 200)
    flambda = fnu_to_flambda(wave, 1.0e-15)
    np.savetxt(sed_dir / "template_0.sed", np.column_stack([wave, flambda]))
    np.savetxt(sed_dir / "template_1.sed", np.column_stack([wave, flambda]))
    (sed_dir / "COSMOS_MOD.list").write_text(
        "COSMOS_SED/template_0.sed\nCOSMOS_SED/template_1.sed\n",
        encoding="utf-8",
    )
    np.savetxt(
        ext_dir / "SMC_prevot.dat",
        np.column_stack([wave, np.full_like(wave, 2.0)]),
    )
    base = {
        "catalog_path": "catalog.parquet",
        "ssp_path": "ssp.h5",
        "bands": [
            {
                "name": "euclid_vis",
                "column": "euclid_vis",
                "units": "fnu_cgs",
                "sigma_mag": 0.05,
                "filter": {"kind": "tophat"},
            }
        ],
        "cosmos_sed": {
            "lephare_data_dir": str(lephare),
            "expected_template_count": 2,
            "extinction": {
                "curves": {
                    0: "none",
                    1: "SMC_prevot",
                }
            },
            "normalization_bands": [
                {"band_name": "euclid_vis", "target_column": "euclid_vis_abs"}
            ],
        },
    }
    return normalize_config(base)
