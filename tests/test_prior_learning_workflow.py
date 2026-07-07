from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.prior_learning.workflow import (
    build_feniks_prior_workflow_plan,
    expected_closure_truth_columns,
    write_feniks_prior_workflow_plan,
)


def test_feniks_prior_workflow_plan_audits_splits_and_commands(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "Data" / "diffsky" / "synthetic" / "feniks"
    dataset_dir.mkdir(parents=True)
    train = dataset_dir / "train.parquet"
    validation = dataset_dir / "validation.parquet"
    test = dataset_dir / "test.parquet"
    _write_closure_truth_parquet(train, rows=2)
    _write_closure_truth_parquet(validation, rows=1)
    _write_closure_truth_parquet(test, rows=1)
    (dataset_dir / "manifest.yaml").write_text("dataset: toy\n", encoding="utf-8")
    (dataset_dir / "schema.json").write_text("{}", encoding="utf-8")
    (dataset_dir / "validation_report.json").write_text("{}", encoding="utf-8")
    prior_checkpoint = tmp_path / "outputs" / "prior" / "checkpoints" / "best.eqx"
    prior_checkpoint.parent.mkdir(parents=True)
    prior_checkpoint.write_bytes(b"checkpoint")

    generation_config = {
        "synthetic_diffsky": {
            "output_dir": str(dataset_dir),
            "split_sizes": {"train": 2, "validation": 1, "test": 1},
        },
    }
    prior_config = {
        "catalog_path": str(train),
        "prior_learning": {
            "train_dataset": str(train),
            "validation_dataset": str(validation),
            "test_dataset": str(test),
            "schema": "diffsky_dsps_closure_full",
            "missing_policy": "fail",
        },
    }
    amortized_config = {
        "catalog_path": str(train),
        "amortized": {"prior": {"checkpoint": str(prior_checkpoint)}},
    }

    plan = build_feniks_prior_workflow_plan(
        generation_config,
        prior_config=prior_config,
        amortized_config=amortized_config,
        generation_config_path="configs/generate.yaml",
        validation_config_path="configs/validate.yaml",
        prior_config_path="configs/prior.yaml",
        amortized_config_path="configs/amortized.yaml",
        prior_out=str(tmp_path / "outputs" / "prior"),
        amortized_out=str(tmp_path / "outputs" / "nn"),
        inference_out=str(tmp_path / "outputs" / "infer"),
        map_out=str(tmp_path / "outputs" / "map"),
        mclmc_out=str(tmp_path / "outputs" / "mclmc"),
        inferred_prior_out=str(tmp_path / "outputs" / "inferred_prior"),
    )

    assert plan.blockers == ()
    assert plan.schema == "diffsky_dsps_closure_full"
    assert len(plan.parameters) == 18
    artifact_details = {item.name: item.detail for item in plan.artifacts}
    assert artifact_details["train_dataset"] == "rows=2, expected=2"
    stages = {stage.name: stage for stage in plan.stages}
    assert stages["03_supervised_nf_prior"].ready is True
    assert stages["04_nn_dsps_nf_train"].ready is True
    assert stages["05_nn_redshift_infer"].ready is False
    assert f"--dataset {test}" in stages["05_nn_redshift_infer"].command
    assert f"--dataset {test}" in stages["07_map_under_learned_prior"].command
    assert f"--dataset {test}" in stages["08_flat_mclmc_calibration"].command

    outputs = write_feniks_prior_workflow_plan(plan, tmp_path / "report")
    assert Path(outputs["json"]).exists()
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "FENIKS Prior Workflow Plan" in markdown
    assert "NF-prior MCLMC needs a posterior-target extension" in markdown


def test_feniks_prior_workflow_plan_reports_schema_blocker(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train = dataset_dir / "train.parquet"
    _write_closure_truth_parquet(
        train,
        rows=1,
        drop_column="dust_delta_true",
    )

    plan = build_feniks_prior_workflow_plan(
        {"synthetic_diffsky": {"output_dir": str(dataset_dir)}},
        prior_config={
            "prior_learning": {
                "train_dataset": str(train),
                "validation_dataset": str(dataset_dir / "validation.parquet"),
                "test_dataset": str(dataset_dir / "test.parquet"),
                "schema": "diffsky_dsps_closure_full",
            },
        },
        amortized_config={},
    )

    assert any("dust_delta" in blocker for blocker in plan.blockers)


def _write_closure_truth_parquet(
    path: Path,
    *,
    rows: int,
    drop_column: str | None = None,
) -> None:
    columns = [column for column in expected_closure_truth_columns() if column != drop_column]
    frame = pd.DataFrame(
        {
            column: np.linspace(0.2, 0.4, rows, dtype=float)
            for column in columns
        }
    )
    frame.insert(0, "object_id", np.arange(rows, dtype=np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
