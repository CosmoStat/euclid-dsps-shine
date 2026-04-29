"""Simple photometric likelihoods."""

from __future__ import annotations

import numpy as np

from .io import GalaxyObservation
from .model import ModelResult


def chi2_mag(observation: GalaxyObservation, result: ModelResult) -> float:
    chi2 = 0.0
    for band in observation.bands:
        model_mag = result.photometry[band.name]["model_mag_ab"]
        if np.isfinite(band.mag_ab) and np.isfinite(model_mag):
            chi2 += ((band.mag_ab - model_mag) / band.sigma_mag) ** 2
    return float(chi2)


def log_likelihood_mag(observation: GalaxyObservation, result: ModelResult) -> float:
    return -0.5 * chi2_mag(observation, result)
