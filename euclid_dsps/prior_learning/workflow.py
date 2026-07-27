"""Preflight planning for the FENIKS prior-learning workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

from .schema import DIFFSKY_DSPS_CLOSURE_FULL_COLUMNS, build_truth_schema

DEFAULT_GENERATION_CONFIG = "configs/diffsky_synthetic_feniks_260617_50k.yaml"
DEFAULT_VALIDATION_CONFIG = DEFAULT_GENERATION_CONFIG
DEFAULT_PRIOR_CONFIG = "configs/prior_diffsky_synthetic_feniks_full_realnvp.yaml"
DEFAULT_AMORTIZED_CONFIG = "configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml"


@dataclass(frozen=True)
class WorkflowArtifact:
    """One file or directory used by the workflow."""

    name: str
    path: str
    kind: str
    required_for: str
    exists: bool
    detail: str = ""


@dataclass(frozen=True)
class WorkflowStage:
    """One executable step in the FENIKS prior workflow."""

    name: str
    purpose: str
    ready: bool
    command: str
    sbatch: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorWorkflowPlan:
    """Resolved workflow contract, preflight status, and launch recipes."""

    dataset_dir: str
    schema: str
    parameters: tuple[str, ...]
    artifacts: tuple[WorkflowArtifact, ...]
    stages: tuple[WorkflowStage, ...]
    warnings: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def dataset_contract_ready(self) -> bool:
        """Whether required FENIKS splits/schema are present and usable."""
        return not self.blockers

    @property
    def ready(self) -> bool:
        """Backward-compatible alias for the dataset contract readiness gate."""
        return self.dataset_contract_ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "schema": self.schema,
            "parameters": list(self.parameters),
            "artifacts": [asdict(item) for item in self.artifacts],
            "stages": [asdict(item) for item in self.stages],
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "dataset_contract_ready": self.dataset_contract_ready,
            "ready": self.ready,
        }


def build_feniks_prior_workflow_plan(
    generation_config: dict[str, Any],
    *,
    validation_config: dict[str, Any] | None = None,
    prior_config: dict[str, Any],
    amortized_config: dict[str, Any],
    generation_config_path: str = DEFAULT_GENERATION_CONFIG,
    validation_config_path: str = DEFAULT_VALIDATION_CONFIG,
    prior_config_path: str = DEFAULT_PRIOR_CONFIG,
    amortized_config_path: str = DEFAULT_AMORTIZED_CONFIG,
    prior_out: str = "outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp",
    amortized_out: str = "outputs/runs/amortized_diffsky_synthetic_feniks_full",
    inference_out: str = (
        "outputs/runs/amortized_diffsky_synthetic_feniks_full_test_infer"
    ),
    map_out: str = "outputs/runs/map_diffsky_synthetic_feniks_under_prior",
    mclmc_out: str = "outputs/runs/mclmc_diffsky_synthetic_feniks_flat",
    inferred_prior_out: str = (
        "outputs/runs/prior_diffsky_synthetic_feniks_from_inferred"
    ),
    repo_root: str | Path = ".",
) -> PriorWorkflowPlan:
    """Build a cheap, file-aware plan for the FENIKS prior-learning ladder."""
    del validation_config
    root = Path(repo_root)
    dataset_dir = _dataset_dir(generation_config, prior_config)
    prior_learning = dict(prior_config.get("prior_learning", {}) or {})
    schema = str(prior_learning.get("schema", "diffsky_dsps_closure_full"))
    train_dataset = _dataset_path(
        prior_learning,
        "train_dataset",
        dataset_dir / "train.parquet",
    )
    validation_dataset = _dataset_path(
        prior_learning,
        "validation_dataset",
        dataset_dir / "validation.parquet",
    )
    test_dataset = _dataset_path(
        prior_learning,
        "test_dataset",
        dataset_dir / "test.parquet",
    )
    prior_checkpoint = _amortized_prior_checkpoint(
        amortized_config,
        Path(prior_out) / "checkpoints" / "best.eqx",
    )
    nn_checkpoint = Path(amortized_out) / "checkpoints" / "best.eqx"
    feature_stats = Path(amortized_out) / "feature_stats.json"
    inference_summary = Path(inference_out) / "posterior_summary.parquet"
    map_estimates = Path(map_out) / "map_estimates.parquet"
    split_sizes = dict(
        (generation_config.get("synthetic_diffsky", {}) or {}).get("split_sizes", {})
        or {}
    )

    artifacts = [
        _artifact(
            "generation_config",
            Path(generation_config_path),
            "file",
            "generation",
            root=root,
        ),
        _artifact(
            "validation_config",
            Path(validation_config_path),
            "file",
            "validation",
            root=root,
        ),
        _artifact("prior_config", Path(prior_config_path), "file", "prior", root=root),
        _artifact(
            "amortized_config",
            Path(amortized_config_path),
            "file",
            "amortized",
            root=root,
        ),
        _artifact("dataset_dir", dataset_dir, "directory", "validation", root=root),
        _parquet_artifact(
            "train_dataset",
            train_dataset,
            "supervised prior and NN training",
            expected_rows=split_sizes.get("train"),
            root=root,
        ),
        _parquet_artifact(
            "validation_dataset",
            validation_dataset,
            "supervised prior validation",
            expected_rows=split_sizes.get("validation"),
            root=root,
        ),
        _parquet_artifact(
            "test_dataset",
            test_dataset,
            "held-out inference evaluation",
            expected_rows=split_sizes.get("test"),
            root=root,
        ),
        _artifact(
            "manifest", dataset_dir / "manifest.yaml", "file", "validation", root=root
        ),
        _artifact(
            "schema", dataset_dir / "schema.json", "file", "validation", root=root
        ),
        _artifact(
            "validation_report",
            dataset_dir / "validation_report.json",
            "file",
            "prior training gate",
            root=root,
        ),
        _artifact(
            "supervised_prior_checkpoint",
            prior_checkpoint,
            "file",
            "amortized training",
            root=root,
        ),
        _artifact(
            "amortized_checkpoint",
            nn_checkpoint,
            "file",
            "amortized inference and MAP under prior",
            root=root,
        ),
        _artifact(
            "amortized_feature_stats",
            feature_stats,
            "file",
            "amortized inference and MAP under prior",
            root=root,
        ),
        _artifact(
            "heldout_inference_summary",
            inference_summary,
            "file",
            "inference evaluation",
            root=root,
        ),
        _artifact(
            "map_estimates",
            map_estimates,
            "file",
            "post-hoc inferred prior",
            root=root,
        ),
    ]

    warnings: list[str] = []
    blockers: list[str] = []
    schema_parameters = tuple(DIFFSKY_BASIC_PARAMETER_NAMES)
    schema_blocker = _schema_blocker(train_dataset, schema, root=root)
    if schema_blocker:
        blockers.append(schema_blocker)
    if schema != "diffsky_dsps_closure_full":
        warnings.append(
            "The production FENIKS closure prior should use "
            "schema=diffsky_dsps_closure_full."
        )
    if _path_exists(dataset_dir / "validation_report.json", root=root) is False:
        warnings.append(
            "Closure validation report is missing; run validation before training "
            "or promoting a prior."
        )
    if not _path_exists(prior_checkpoint, root=root):
        warnings.append(
            "Supervised prior checkpoint is not present yet; amortized training "
            "must wait for diffsky-train-supervised-prior."
        )
    if not _path_exists(nn_checkpoint, root=root):
        warnings.append(
            "Amortized checkpoint is not present yet; inference and MAP-under-prior "
            "must wait for amortized-train-diffsky."
        )
    if not _path_exists(feature_stats, root=root):
        warnings.append(
            "Feature statistics are not present yet; they are produced by "
            "amortized training and required for inference/MAP recipes."
        )
    if not _path_exists(train_dataset, root=root):
        blockers.append(f"Missing training dataset: {train_dataset}")
    if not _path_exists(validation_dataset, root=root):
        blockers.append(f"Missing validation dataset: {validation_dataset}")
    if not _path_exists(test_dataset, root=root):
        blockers.append(f"Missing test dataset: {test_dataset}")

    stage_inputs = {
        "has_train": _path_exists(train_dataset, root=root),
        "has_validation": _path_exists(validation_dataset, root=root),
        "has_test": _path_exists(test_dataset, root=root),
        "has_prior_checkpoint": _path_exists(prior_checkpoint, root=root),
        "has_nn_checkpoint": _path_exists(nn_checkpoint, root=root),
        "has_feature_stats": _path_exists(feature_stats, root=root),
        "has_inference_summary": _path_exists(inference_summary, root=root),
        "has_map_estimates": _path_exists(map_estimates, root=root),
    }
    stages = _workflow_stages(
        generation_config_path=Path(generation_config_path),
        validation_config_path=Path(validation_config_path),
        prior_config_path=Path(prior_config_path),
        amortized_config_path=Path(amortized_config_path),
        dataset_dir=dataset_dir,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        prior_checkpoint=prior_checkpoint,
        nn_checkpoint=nn_checkpoint,
        feature_stats=feature_stats,
        prior_out=Path(prior_out),
        amortized_out=Path(amortized_out),
        inference_out=Path(inference_out),
        map_out=Path(map_out),
        mclmc_out=Path(mclmc_out),
        inferred_prior_out=Path(inferred_prior_out),
        stage_inputs=stage_inputs,
    )
    return PriorWorkflowPlan(
        dataset_dir=str(dataset_dir),
        schema=schema,
        parameters=schema_parameters,
        artifacts=tuple(artifacts),
        stages=tuple(stages),
        warnings=tuple(dict.fromkeys(warnings)),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def write_feniks_prior_workflow_plan(
    plan: PriorWorkflowPlan,
    out_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown renderings of a workflow plan."""
    out = ensure_dir(out_dir)
    json_path = out / "workflow_plan.json"
    md_path = out / "workflow_plan.md"
    write_json(json_path, plan.to_dict())
    md_path.write_text(render_feniks_prior_workflow_markdown(plan), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_feniks_prior_workflow_markdown(plan: PriorWorkflowPlan) -> str:
    """Render a concise runbook from a resolved workflow plan."""
    lines = [
        "# FENIKS Prior Workflow Plan",
        "",
        f"- Dataset directory: `{plan.dataset_dir}`",
        f"- Truth schema: `{plan.schema}`",
        f"- Parameter count: `{len(plan.parameters)}`",
        f"- Dataset contract ready: `{str(plan.dataset_contract_ready).lower()}`",
        "",
        "## Artifacts",
        "",
        "| Artifact | Required for | Status | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for item in plan.artifacts:
        status = "present" if item.exists else "missing"
        detail = item.detail or item.path
        lines.append(f"| `{item.name}` | {item.required_for} | {status} | `{detail}` |")
    if plan.blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in plan.blockers)
    if plan.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in plan.warnings)
    lines.extend(["", "## Stages", ""])
    for stage in plan.stages:
        ready = "ready" if stage.ready else "waiting"
        lines.extend(
            [
                f"### {stage.name}",
                "",
                f"- Status: `{ready}`",
                f"- Purpose: {stage.purpose}",
                "",
                "```bash",
                stage.command,
                "```",
            ]
        )
        if stage.sbatch:
            lines.extend(["", "Jean-Zay launcher:", "", "```bash", stage.sbatch, "```"])
        if stage.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"- {note}" for note in stage.notes)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _workflow_stages(
    *,
    generation_config_path: Path,
    validation_config_path: Path,
    prior_config_path: Path,
    amortized_config_path: Path,
    dataset_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
    test_dataset: Path,
    prior_checkpoint: Path,
    nn_checkpoint: Path,
    feature_stats: Path,
    prior_out: Path,
    amortized_out: Path,
    inference_out: Path,
    map_out: Path,
    mclmc_out: Path,
    inferred_prior_out: Path,
    stage_inputs: dict[str, bool],
) -> list[WorkflowStage]:
    generated = (
        stage_inputs["has_train"]
        and stage_inputs["has_validation"]
        and stage_inputs["has_test"]
    )
    trained_nn = stage_inputs["has_nn_checkpoint"] and stage_inputs["has_feature_stats"]
    return [
        WorkflowStage(
            name="00_dataset_smoke",
            purpose="Check the Diffsky/FENIKS generator and DSPS closure path.",
            ready=True,
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {generation_config_path}",
                "diffsky-generate-dsps-closure",
                "--smoke --overwrite",
            ),
            sbatch=(
                "sbatch --export=ALL,STAGE=both,SMOKE=1,MAX_GALAXIES=120,"
                "OVERWRITE=1,RESUME=0 scripts/diffsky_synthetic_feniks_50k_h100.slurm"
            ),
        ),
        WorkflowStage(
            name="01_dataset_generate",
            purpose="Generate independent train/validation/test closure splits.",
            ready=True,
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {generation_config_path}",
                "diffsky-generate-dsps-closure",
                "--split all --resume",
            ),
            sbatch=(
                "sbatch --export=ALL,STAGE=generate,OVERWRITE=1,RESUME=0 "
                "scripts/diffsky_synthetic_feniks_50k_h100.slurm"
            ),
            notes=("Use RESUME=1 and OVERWRITE=0 for compatible interrupted runs.",),
        ),
        WorkflowStage(
            name="02_dataset_validate",
            purpose="Validate exact same-parameter closure before learning priors.",
            ready=generated,
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {validation_config_path}",
                "diffsky-validate-dsps-closure",
                f"--dataset-dir {dataset_dir}",
                "--sample-size 256 --batch-size 256 --runtime gpu",
            ),
            sbatch=(
                "sbatch --export=ALL,STAGE=validate "
                "scripts/diffsky_synthetic_feniks_50k_h100.slurm"
            ),
        ),
        WorkflowStage(
            name="03_supervised_nf_prior",
            purpose="Train p(theta) directly from the 18D closure truths.",
            ready=generated,
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {prior_config_path}",
                "diffsky-train-supervised-prior",
                f"--out {prior_out}",
            ),
        ),
        WorkflowStage(
            name="04_nn_dsps_nf_train",
            purpose="Train amortized NN+DSPS inference with the supervised NF prior.",
            ready=generated and stage_inputs["has_prior_checkpoint"],
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {amortized_config_path}",
                "amortized-train-diffsky",
                f"--dataset {train_dataset}",
                f"--prior-checkpoint {prior_checkpoint}",
                f"--out {amortized_out}",
            ),
            sbatch=(
                "sbatch --export=ALL,"
                f"CONFIG={amortized_config_path},"
                f"DATASET={train_dataset},"
                f"OUT_DIR={amortized_out},"
                f"PRIOR_CHECKPOINT={prior_checkpoint} "
                "scripts/diffsky_amortized_train_h100.slurm"
            ),
        ),
        WorkflowStage(
            name="05_nn_redshift_infer",
            purpose="Run held-out amortized redshift/posterior inference.",
            ready=stage_inputs["has_test"] and trained_nn,
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {amortized_config_path}",
                "amortized-infer-diffsky",
                f"--dataset {test_dataset}",
                f"--checkpoint {nn_checkpoint}",
                f"--feature-stats {feature_stats}",
                f"--out {inference_out}",
                "--shard-outputs",
            ),
            sbatch=(
                "sbatch --export=ALL,"
                f"CONFIG={amortized_config_path},"
                f"DATASET={test_dataset},"
                f"CHECKPOINT={nn_checkpoint},"
                f"FEATURE_STATS={feature_stats},"
                f"OUT_DIR={inference_out} "
                "scripts/diffsky_amortized_infer_h100.slurm"
            ),
        ),
        WorkflowStage(
            name="06_evaluate_nn_inference",
            purpose="Score held-out posterior calibration against closure truths.",
            ready=stage_inputs["has_inference_summary"],
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {amortized_config_path}",
                "diffsky-evaluate-dsps-closure-inference",
                f"--run {inference_out}",
                f"--dataset {test_dataset}",
                f"--out {inference_out}_eval",
            ),
        ),
        WorkflowStage(
            name="07_map_under_learned_prior",
            purpose="Run MAP redshift inference with the learned RealNVP prior.",
            ready=stage_inputs["has_test"] and trained_nn,
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {amortized_config_path}",
                "diffsky-map-adam-prior",
                f"--dataset {test_dataset}",
                f"--checkpoint {nn_checkpoint}",
                f"--feature-stats {feature_stats}",
                f"--out {map_out}",
                "--prior-weight 0.05 --prior-density-space x",
            ),
            sbatch=(
                "sbatch --export=ALL,"
                f"CONFIG={amortized_config_path},"
                f"DATASET={test_dataset},"
                f"CHECKPOINT={nn_checkpoint},"
                f"FEATURE_STATS={feature_stats},"
                f"OUT_DIR={map_out} "
                "scripts/diffsky_map_adam_prior_h100.slurm"
            ),
        ),
        WorkflowStage(
            name="08_flat_mclmc_calibration",
            purpose="Run direct MCLMC on selected rows as a posterior baseline.",
            ready=stage_inputs["has_test"],
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {amortized_config_path}",
                "posterior",
                f"--dataset {test_dataset}",
                "--sampler mclmc --limit 16 --batch-size 4",
                f"--out {mclmc_out}",
            ),
            sbatch=(
                "sbatch --export=ALL,"
                f"CONFIG={amortized_config_path},"
                f"DATASET={test_dataset},"
                "ROWSET_LIMIT=64,"
                f"OUT_ROOT={mclmc_out} "
                "scripts/diffsky_flat_mclmc_calibration_h100.slurm"
            ),
            notes=(
                "This is the current direct-MCLMC path. NF-prior MCLMC needs a "
                "posterior-target extension; MAP-under-NF is already available.",
            ),
        ),
        WorkflowStage(
            name="09_inferred_nf_prior",
            purpose="Train a post-hoc NF prior from MAP/MCLMC theta samples.",
            ready=stage_inputs["has_map_estimates"],
            command=_join_command(
                "python -m euclid_dsps.cli",
                f"--config {amortized_config_path}",
                "diffsky-train-inferred-prior",
                f"--input {map_out / 'map_estimates.parquet'}",
                f"--out {inferred_prior_out}",
            ),
            sbatch=(
                "sbatch --export=ALL,"
                f"CONFIG={amortized_config_path},"
                f"INFERRED_INPUTS={map_out / 'map_estimates.parquet'},"
                f"OUT_DIR={inferred_prior_out} "
                "scripts/diffsky_inferred_prior_h100.slurm"
            ),
        ),
    ]


def _dataset_dir(
    generation_config: dict[str, Any],
    prior_config: dict[str, Any],
) -> Path:
    synthetic = dict(generation_config.get("synthetic_diffsky", {}) or {})
    if synthetic.get("output_dir"):
        return Path(str(synthetic["output_dir"]))
    prior_learning = dict(prior_config.get("prior_learning", {}) or {})
    for key in ("train_dataset", "dataset"):
        if prior_learning.get(key):
            return Path(str(prior_learning[key])).parent
    return Path("Data/diffsky/synthetic/feniks_260617_dsps_closure")


def _dataset_path(prior_learning: dict[str, Any], key: str, default: Path) -> Path:
    return Path(str(prior_learning.get(key) or default))


def _amortized_prior_checkpoint(
    amortized_config: dict[str, Any],
    default: Path,
) -> Path:
    amortized = dict(amortized_config.get("amortized", {}) or {})
    prior = dict(amortized.get("prior", {}) or {})
    return Path(str(prior.get("checkpoint") or default))


def _artifact(
    name: str,
    path: Path,
    kind: str,
    required_for: str,
    *,
    root: Path,
    detail: str = "",
) -> WorkflowArtifact:
    exists = _path_exists(path, root=root)
    return WorkflowArtifact(
        name=name,
        path=str(path),
        kind=kind,
        required_for=required_for,
        exists=exists,
        detail=detail,
    )


def _parquet_artifact(
    name: str,
    path: Path,
    required_for: str,
    *,
    expected_rows: Any,
    root: Path,
) -> WorkflowArtifact:
    exists = _path_exists(path, root=root)
    detail = ""
    if exists:
        rows = _parquet_num_rows(_resolve(path, root=root))
        detail = f"rows={rows}" if rows is not None else "rows=unknown"
        if expected_rows is not None and rows is not None:
            expected = int(expected_rows)
            detail += f", expected={expected}"
            if rows != expected:
                detail += ", row_count_mismatch"
    return WorkflowArtifact(
        name=name,
        path=str(path),
        kind="parquet",
        required_for=required_for,
        exists=exists,
        detail=detail,
    )


def _schema_blocker(path: Path, schema: str, *, root: Path) -> str | None:
    if not _path_exists(path, root=root):
        return None
    names = _parquet_columns(_resolve(path, root=root))
    if names is None:
        return f"Could not read parquet schema for {path}"
    try:
        resolved = build_truth_schema(
            names,
            schema_name=schema,
            missing_policy="fail",
        )
    except ValueError as exc:
        return str(exc)
    missing = [
        name
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        if name not in {param.name for param in resolved.parameters}
    ]
    if missing and schema == "diffsky_dsps_closure_full":
        return "Closure schema is missing parameters: " + ", ".join(missing)
    return None


def _parquet_num_rows(path: Path) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _parquet_columns(path: Path) -> list[str] | None:
    try:
        return list(pq.read_schema(path).names)
    except Exception:
        return None


def _path_exists(path: Path, *, root: Path) -> bool:
    return _resolve(path, root=root).exists()


def _resolve(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _join_command(*parts: str) -> str:
    return " \\\n  ".join(part for part in parts if part)


def expected_closure_truth_columns() -> tuple[str, ...]:
    """Return the first-choice parquet columns for the 18D closure schema."""
    return tuple(
        DIFFSKY_DSPS_CLOSURE_FULL_COLUMNS[name][0]
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
    )
