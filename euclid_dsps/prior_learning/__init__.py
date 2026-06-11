"""Supervised population-prior learning from Diffsky truth parameters."""

from .data import TruthDataset, load_truth_dataset
from .schema import ParameterSpec, TruthSchema, build_truth_schema

__all__ = [
    "ParameterSpec",
    "TruthDataset",
    "TruthSchema",
    "build_truth_schema",
    "load_truth_dataset",
]
