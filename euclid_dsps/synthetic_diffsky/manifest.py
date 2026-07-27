"""Reproducibility manifest helpers for synthetic closure catalogs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from euclid_dsps.filters import FilterCurve
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES


def stable_hash_payload(payload: Any) -> str:
    """Return a stable SHA256 of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    """Return SHA256 of a local file, or None if it is absent."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filters_hash_payload(filters: dict[str, FilterCurve]) -> dict[str, dict[str, Any]]:
    """Hash loaded filter curve arrays, independent of source file format."""
    payload: dict[str, dict[str, Any]] = {}
    for name, curve in filters.items():
        digest = hashlib.sha256()
        digest.update(np.asarray(curve.wave, dtype=np.float64).tobytes())
        digest.update(np.asarray(curve.transmission, dtype=np.float64).tobytes())
        payload[str(name)] = {
            "source": str(curve.source),
            "sha256": digest.hexdigest(),
            "n_wave": int(len(curve.wave)),
            "effective_wavelength": float(curve.effective_wavelength),
        }
    return payload


def package_version(name: str) -> str | None:
    """Return installed package version if available."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_git_sha(module_name: str) -> str | None:
    """Best-effort git SHA for an installed editable package."""
    try:
        module = __import__(module_name)
    except Exception:
        return None
    file = getattr(module, "__file__", None)
    if not file:
        return None
    root = Path(file).resolve()
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                return None
    return None


def repo_git_sha(repo_root: str | Path = ".") -> str | None:
    """Return current repository SHA if available."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a YAML manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def base_manifest(
    *,
    config: dict[str, Any],
    config_hash: str,
    ssp_hash: str | None,
    filter_hashes: dict[str, dict[str, Any]],
    calibration_hash: str | None,
) -> dict[str, Any]:
    """Return manifest fields common to every generation run."""
    model = config.get("model", {}) or {}
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "repo_git_sha": repo_git_sha(),
        "diffsky_version": package_version("diffsky"),
        "diffsky_git_sha": package_git_sha("diffsky"),
        "dsps_version": package_version("dsps"),
        "diffstar_version": package_version("diffstar"),
        "diffmah_version": package_version("diffmah"),
        "jax_version": package_version("jax"),
        "config_hash": config_hash,
        "calibration": {
            "name": (config.get("synthetic_diffsky", {}) or {}).get(
                "calibration_name", "feniks_260617"
            ),
            "hash": calibration_hash,
        },
        "ssp": {
            "path": str(config.get("ssp_path")),
            "sha256": ssp_hash,
        },
        "filters": filter_hashes,
        "z_sun": float(model.get("z_sun", 0.0142)),
        "stellar_metallicity_model": str(
            model.get("stellar_metallicity_model", "single")
        ),
        "stellar_metallicity_scatter_dex": float(
            model.get("stellar_metallicity_scatter_dex", 0.2)
        ),
        "flux_units": "fnu_cgs",
        "parameter_order": list(DIFFSKY_BASIC_PARAMETER_NAMES),
    }


def calibration_hash(calibration_dir: str | Path, calibration_name: str) -> str | None:
    """Best-effort hash for a calibration directory or HDF5 file."""
    root = Path(calibration_dir)
    candidates = _calibration_candidates(root, calibration_name)
    for candidate in candidates:
        if candidate.is_file():
            return file_sha256(candidate)
        if candidate.is_dir():
            digest = hashlib.sha256()
            files = sorted(path for path in candidate.rglob("*") if path.is_file())
            if not files:
                continue
            for path in files:
                digest.update(str(path.relative_to(candidate)).encode("utf-8"))
                file_hash = file_sha256(path)
                if file_hash:
                    digest.update(file_hash.encode("ascii"))
            return digest.hexdigest()
    return None


def _calibration_candidates(calibration_dir: Path, calibration_name: str) -> list[Path]:
    """Return local and installed-Diffsky candidate calibration paths."""
    bname = f"diffsky_{calibration_name}_param_collection.hdf5"
    candidates = [calibration_dir / bname, calibration_dir]
    if not calibration_dir.is_absolute():
        try:
            import diffsky
        except Exception:
            return candidates
        package_root = Path(diffsky.__file__).resolve().parent
        param_utils_root = package_root / "param_utils"
        candidates.extend(
            [
                param_utils_root / calibration_dir / bname,
                param_utils_root / calibration_dir,
            ]
        )
    return candidates
