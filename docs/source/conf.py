"""Sphinx configuration for Euclid DSPS SHINE."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as metadata_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "Euclid DSPS SHINE"
author = "Euclid DSPS SHINE contributors"

try:
    release = metadata_version("euclid-dsps-shine")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "astropy",
    "astropy.io",
    "astropy.io.fits",
    "corner",
    "dsps",
    "dsps.cosmology",
    "dsps.dust",
    "dsps.dust.att_curves",
    "h5py",
    "jax",
    "jax.numpy",
    "jax.scipy",
    "jax.scipy.optimize",
    "matplotlib",
    "matplotlib.pyplot",
    "numpyro",
    "numpyro.distributions",
    "numpyro.infer",
    "numpyro.infer.initialization",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
