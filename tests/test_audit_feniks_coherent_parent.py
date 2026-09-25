import numpy as np
import pandas as pd
import pytest

from scripts import audit_feniks_coherent_parent as audit
from scripts.feniks_avi_experiments import sha, write
from scripts.feniks_coherent_parent import finish


def spec_fixture(path):
    from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

    write(
        path,
        dict(
            names=SPLINE15D_PARAMETER_NAMES,
            lower=[-1.0] * 15,
            upper=[1.0] * 15,
            raw_center=[0.0] * 15,
            raw_scale=[1.0] * 15,
            normalization="identity",
            arithmetic_precision="float64_v1",
            coordinate_information_source="test only",
        ),
    )
    return audit.load_spec(path)


def test_bounds_are_checked_before_clipped_roundtrip(tmp_path):
    spec = spec_fixture(tmp_path / "spec.json")
    frame = pd.DataFrame(np.zeros((20, 15)), columns=spec.names)
    frame.loc[0, "z_obs"] = 2.0
    frame.loc[1, "z_obs"] = -1.0
    table, _, detail = audit.geometry(frame, spec)
    assert detail["outside_rows"] == 1
    assert detail["boundary_rows"] == 1
    assert detail["max_roundtrip_over_iqr"] > 1e-4
    assert len(detail["sfh_coordinates_with_zero_fraction_above_1pct"]) == 10
    assert table.loc[table.parameter == "z_obs", "outside_fraction"].item() == 0.05
    assert detail["maximum_physical_sfh_abs_spearman"] is None


def test_parent_dependence_is_descriptive_and_transform_is_invertible(tmp_path):
    spec = spec_fixture(tmp_path / "spec.json")
    frame = pd.DataFrame(
        np.random.default_rng(3).uniform(-0.7, 0.7, (100, 15)), columns=spec.names
    )
    frame[spec.names[-1]] = frame.z_obs
    table, _, detail = audit.geometry(frame, spec)
    assert detail["outside_rows"] == 0
    assert detail["max_roundtrip_over_iqr"] < 1e-10
    assert detail["maximum_physical_sfh_abs_spearman"] == pytest.approx(1.0)
    assert (
        table.loc[table.group == "physical", "sfh_w1_to_standard_normal"].isna().all()
    )
    assert table.loc[table.group == "sfh", "sfh_w1_to_standard_normal"].notna().all()


def dataset_fixture(tmp_path):
    from euclid_dsps.synthetic_diffsky.coherent_parent import catalogue_checks, observe

    spec_path = tmp_path / "spec.json"
    spec = spec_fixture(spec_path)
    root = tmp_path / "dataset_root"
    root.mkdir()
    (root / "report").mkdir()
    for kind in ("parent", "selected_r29"):
        (root / "dataset" / kind).mkdir(parents=True)
    bands = [dict(name="lsst_r")]
    selection = dict(band="lsst_r", max_mag_ab=29.0)
    cfg = dict(parent_rows={s: 300 for s in audit.SPLITS}, selection=selection)
    write(root / "decoder.json", dict(bands=bands))
    write(
        root / "MANIFEST.json",
        dict(
            frozen_files={"decoder.json": sha(root / "decoder.json")},
            settings=cfg,
            parent_scope="synthetic test",
        ),
    )
    files, checks = [], {}
    for i, split in enumerate(audit.SPLITS):
        frame = pd.DataFrame(
            np.random.default_rng(i).uniform(-0.7, 0.7, (300, 15)), columns=spec.names
        )
        frame["object_id"] = np.arange(300) + i * 1000
        frame["population_weight"] = frame["galaxy_weight"] = 1.0
        frame["split"] = split
        frame["effective_source_seed"] = i
        frame["effective_proposal_key"] = [f"{i}:{j}" for j in range(300)]
        frame = observe(
            frame,
            np.full((300, 1), 9e-32),
            bands,
            dict(
                type="m5_depth",
                m5={"lsst_r": 27.5},
                gamma={"lsst_r": 0.039},
                sigma_sys_mag=0.005,
            ),
            seed=i,
            selection=selection,
        )
        checks[split] = catalogue_checks(frame, bands, selection)
        for kind, data in (
            ("parent", frame),
            ("selected_r29", frame[frame.selected_r29]),
        ):
            p = root / "dataset" / kind / f"{split}.parquet"
            data.to_parquet(p, index=False)
            files.append(p)
    write(
        root / "dataset/CONTRACT.json",
        dict(
            status="COHERENT_PARENT_DATASET_COMPLETE",
            ready_for_population_benchmark=True,
        ),
    )
    write(root / "report/checks.json", dict(splits=checks))
    finish(
        root / "report",
        [*files, root / "dataset/CONTRACT.json", root / "report/checks.json"],
        sha(root / "MANIFEST.json"),
    )
    return root, spec_path


def test_complete_read_only_audit_never_promotes_training(tmp_path):
    root, spec_path = dataset_fixture(tmp_path)
    before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    result = audit.run(root, spec_path, tmp_path / "audit")
    assert result["dataset_integrity_pass"]
    assert result["old_transform_compatible"]
    assert result["geometry"]["rows"] == 300  # parent train, not pooled or selected
    assert not result["ready_for_population_training"]
    assert before == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="new output"):
        audit.run(root, spec_path, root / "unsafe")
    p = root / "dataset/parent/test.parquet"
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt completed"):
        audit.run(root, spec_path, tmp_path / "audit2")


def test_selected_view_must_be_exact_even_with_updated_hashes(tmp_path):
    root, _ = dataset_fixture(tmp_path)
    p = root / "dataset/selected_r29/train.parquet"
    data = pd.read_parquet(p)
    data.loc[0, "flux_lsst_r"] += 1.0
    data.to_parquet(p, index=False)
    receipt = audit.read(root / "report/FINAL.json")
    receipt["artifacts"]["../dataset/selected_r29/train.parquet"] = sha(p)
    write(root / "report/FINAL.json", receipt)
    with pytest.raises(AssertionError):
        audit.verify_catalogues(root)
