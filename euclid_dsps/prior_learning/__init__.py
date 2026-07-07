"""Supervised population-prior learning from Diffsky truth parameters."""

from .schema import ParameterSpec, TruthSchema, build_truth_schema
from .workflow import (
    PriorWorkflowPlan,
    WorkflowArtifact,
    WorkflowStage,
    build_feniks_prior_workflow_plan,
)

__all__ = [
    "ParameterSpec",
    "PriorWorkflowPlan",
    "TruthDataset",
    "TruthSchema",
    "WorkflowArtifact",
    "WorkflowStage",
    "build_truth_schema",
    "build_feniks_prior_workflow_plan",
    "load_truth_dataset",
]


def __getattr__(name: str):
    if name in {"TruthDataset", "load_truth_dataset"}:
        from . import data

        return getattr(data, name)
    raise AttributeError(name)
