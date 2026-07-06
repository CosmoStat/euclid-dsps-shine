"""End-to-end generation of synthetic Diffsky/FENIKS DSPS closure catalogs."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir
from euclid_dsps.model import load_context
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.photometric_uncertainty import flux_error_model_payload

from .backend import generate_proposal_shard
from .config import (
    SyntheticDiffskyConfig,
    load_synthetic_diffsky_config,
    selected_splits,
)
from .manifest import (
    base_manifest,
    calibration_hash,
    file_sha256,
    filters_hash_payload,
    stable_hash_payload,
    write_manifest,
)
from .metallicity import metallicity_summary_payload
from .photometry import GROUND_TRUTH_COLUMNS, add_dsps_closure_photometry
from .population_diagnostics import run_generation_population_diagnostics
from .resampling import resample_weighted_proposals, resampling_summary
from .selection import (
    append_snr_selection_columns,
    apply_photometric_selection,
    apply_proposal_selection,
    normalize_selection,
    photometric_selection_enabled,
)


LAYER_OBJECT_ID_OFFSETS = {
    "inference_ready": 0,
    "survey_like": 10_000_000_000,
}


def generate_dsps_closure_dataset(
    config: dict[str, Any],
    *,
    split: str = "all",
    max_galaxies: int | None = None,
    smoke: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    verbose: bool = True,
) -> Path:
    """Generate weighted proposals and final unweighted closure catalogs."""
    gen_cfg = load_synthetic_diffsky_config(
        config,
        smoke=bool(smoke),
        max_galaxies=max_galaxies,
    )
    out = gen_cfg.output_dir
    chosen = selected_splits(split)
    _progress(
        verbose,
        f"start output={out} splits={','.join(chosen)} smoke={bool(smoke)} "
        f"z=[{gen_cfg.z_min}, {gen_cfg.z_max}]",
    )
    if out.exists() and overwrite and not resume:
        _progress(verbose, f"overwrite requested; removing {out}")
        shutil.rmtree(out)
    ensure_dir(out)
    ensure_dir(out / "proposals")
    ensure_dir(out / "diagnostics")

    _progress(verbose, "loading filters and SSP context")
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int((config.get("model", {}) or {}).get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    ssp_lgmet = np.asarray(context.ssp_lgmet_jax, dtype=float)
    config_hash = stable_hash_payload(config)
    manifest = base_manifest(
        config=config,
        config_hash=config_hash,
        ssp_hash=file_sha256(config["ssp_path"]),
        filter_hashes=filters_hash_payload(filters),
        calibration_hash=calibration_hash(
            gen_cfg.calibration_dir,
            gen_cfg.calibration_name,
        ),
    )
    manifest["synthetic_diffsky"] = {
        "proposal_backend": gen_cfg.proposal_backend,
        "n_host_halos_per_shard": int(gen_cfg.n_host_halos_per_shard),
        "max_shards": int(gen_cfg.max_shards),
        "z_min": float(gen_cfg.z_min),
        "z_max": float(gen_cfg.z_max),
        "lgmp_min": float(gen_cfg.lgmp_min),
        "lgmp_max": float(gen_cfg.lgmp_max),
        "sky_area_degsq": float(gen_cfg.sky_area_degsq),
        "mc_merge": int(gen_cfg.mc_merge),
        "metallicity_grid_policy": gen_cfg.metallicity_grid_policy,
        "max_duplication_fraction": float(gen_cfg.max_duplication_fraction),
        "duplication_gate": str(gen_cfg.duplication_gate),
        "selection": normalize_selection(gen_cfg.selection),
        "output_layers": _manifest_output_layers(gen_cfg.output_layers),
        "flux_error_model": flux_error_model_payload(gen_cfg.flux_error_model),
        "smoke": bool(smoke),
    }
    manifest["splits"] = {}

    for split_name in chosen:
        split_cfg = gen_cfg.splits[split_name]
        split_summary = _generate_one_split(
            config,
            gen_cfg,
            split_cfg,
            filters=filters,
            ssp_lgmet=ssp_lgmet,
            z_sun=float((config.get("model", {}) or {}).get("z_sun", context.z_sun)),
            overwrite=overwrite,
            resume=resume,
            verbose=verbose,
        )
        manifest["splits"][split_name] = split_summary
        write_manifest(out / "manifest.yaml", manifest)

    _progress(verbose, "writing schema and combined catalog")
    _write_schema(out / "schema.json", config)
    _write_all_catalog(out)
    _write_layer_all_catalogs(out, gen_cfg.output_layers)
    _progress(verbose, "running post-generation population diagnostics")
    diagnostics_path = run_generation_population_diagnostics(
        config,
        dataset_dir=out,
        smoke=bool(smoke),
    )
    if diagnostics_path is not None:
        manifest["population_diagnostics"] = str(diagnostics_path)
        _progress(verbose, f"population diagnostics -> {diagnostics_path}")
    write_manifest(out / "manifest.yaml", manifest)
    _progress(verbose, f"done -> {out}")
    return out


def _generate_one_split(
    config: dict[str, Any],
    gen_cfg: SyntheticDiffskyConfig,
    split_cfg,
    *,
    filters,
    ssp_lgmet: np.ndarray,
    z_sun: float,
    overwrite: bool,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    out = gen_cfg.output_dir
    final_path = out / f"{split_cfg.name}.parquet"
    if final_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"{final_path} already exists; pass --overwrite or --resume explicitly"
        )
    if final_path.exists() and resume:
        frame = pd.read_parquet(final_path)
        if len(frame) == int(split_cfg.n_final):
            _progress(
                verbose,
                f"{split_cfg.name}: reusing existing final {final_path} rows={len(frame)}",
            )
            return {
                "status": "reused_existing_final",
                "final_path": str(final_path),
                "final_size": int(len(frame)),
            }
        _progress(
            verbose,
            f"{split_cfg.name}: ignoring stale final {final_path} rows={len(frame)} "
            f"expected={int(split_cfg.n_final)}",
        )
    _progress(
        verbose,
        f"{split_cfg.name}: target={int(split_cfg.n_final)} "
        f"pool_factor={gen_cfg.pool_size_factor} min_ess_factor={gen_cfg.min_ess_fraction}",
    )
    proposal_dir = ensure_dir(out / "proposals" / split_cfg.name)
    if overwrite and proposal_dir.exists() and not resume:
        shutil.rmtree(proposal_dir)
        proposal_dir = ensure_dir(proposal_dir)

    bands = [str(band["name"]) for band in config["bands"]]
    selection_cfg = normalize_selection(gen_cfg.selection)
    output_layers = _resolve_output_layers(
        gen_cfg.output_layers,
        fallback_selection=selection_cfg,
        split_name=split_cfg.name,
        n_final=int(split_cfg.n_final),
    )
    use_photometric_selection = photometric_selection_enabled(selection_cfg)
    proposals: list[pd.DataFrame] = []
    shard_metadata: list[dict[str, Any]] = []
    result = None
    selected_pool = None
    proposal_selection_summary: dict[str, Any] = {}
    for shard_index in range(int(gen_cfg.max_shards)):
        shard_path = proposal_dir / f"shard_{shard_index:05d}.parquet"
        _progress(
            verbose,
            f"{split_cfg.name}: shard {shard_index + 1}/{gen_cfg.max_shards} "
            f"hosts={gen_cfg.n_host_halos_per_shard}",
        )
        if shard_path.exists() and resume:
            shard = pd.read_parquet(shard_path)
            meta = {"backend": gen_cfg.proposal_backend, "reused": True}
            _progress(
                verbose,
                f"{split_cfg.name}: reused {shard_path} proposals={len(shard)}",
            )
        else:
            shard, meta = generate_proposal_shard(
                gen_cfg,
                split_cfg,
                shard_index,
                filters=filters,
                ssp_lgmet=ssp_lgmet,
                z_sun=z_sun,
                ssp_path=config["ssp_path"],
            )
            shard = _stamp_proposal_ids(shard, split_cfg, shard_index)
            shard.to_parquet(shard_path, index=False)
            _progress(
                verbose,
                f"{split_cfg.name}: wrote {shard_path} proposals={len(shard)}",
            )
        proposals.append(shard)
        shard_metadata.append({**meta, "path": str(shard_path)})
        pool = pd.concat(proposals, ignore_index=True)
        selected_pool, proposal_selection_summary = apply_proposal_selection(
            pool,
            selection_cfg,
        )
        if len(selected_pool) == 0:
            _progress(
                verbose,
                f"{split_cfg.name}: raw_pool={len(pool)} selected_pool=0",
            )
            continue
        result = resample_weighted_proposals(selected_pool, split_cfg)
        _progress(
            verbose,
            f"{split_cfg.name}: raw_pool={len(pool)} selected_pool={result.pool_size} "
            f"ESS={result.ess:.3g} dup={result.duplicate_fraction:.3g}",
        )
        final_attempt = shard_index + 1 >= int(gen_cfg.max_shards)
        if _pool_is_sufficient(
            result,
            split_cfg,
            gen_cfg,
            final_attempt=final_attempt,
        ):
            if result.duplicate_fraction > float(gen_cfg.max_duplication_fraction):
                _progress(
                    verbose,
                    f"{split_cfg.name}: duplication={result.duplicate_fraction:.3g} "
                    f"exceeds threshold={gen_cfg.max_duplication_fraction:.3g}; "
                    f"continuing because duplication_gate={gen_cfg.duplication_gate}",
                )
            _progress(verbose, f"{split_cfg.name}: pool targets reached")
            break
    if result is None:
        raise ValueError(
            f"No selected proposals generated for split {split_cfg.name}. "
            "Relax synthetic_diffsky.selection or increase max_shards."
        )
    if not _pool_is_sufficient(
        result,
        split_cfg,
        gen_cfg,
        final_attempt=True,
    ):
        raise RuntimeError(
            f"Split {split_cfg.name} did not reach ESS/duplication targets after "
            f"{gen_cfg.max_shards} shards: ESS={result.ess:.3g}, "
            f"duplication={result.duplicate_fraction:.3g}"
        )
    if selected_pool is None:
        raise ValueError(f"No selected proposals generated for split {split_cfg.name}")
    layer_summaries: dict[str, Any] = {}
    if output_layers:
        final, layer_summaries, candidate_size, candidate_resampling_summary = (
            _build_layered_final_catalogs(
                config,
                gen_cfg,
                split_cfg,
                selected_pool,
                output_layers,
                bands=bands,
                verbose=verbose,
            )
        )
        photometric_summary = dict(
            layer_summaries.get("inference_ready", {}).get("selection_summary", {})
        )
        photometric_summary.setdefault("enabled", bool(photometric_summary))
    else:
        photometric_summary = {"enabled": False}
        candidate_size = int(split_cfg.n_final)
        candidate_resampling_summary: dict[str, Any] = {}
    if not output_layers and use_photometric_selection:
        candidate_size = max(
            int(split_cfg.n_final),
            int(
                np.ceil(
                    float(selection_cfg["photometric_oversample_factor"])
                    * int(split_cfg.n_final)
                )
            ),
        )
        candidate_split = replace(
            split_cfg,
            n_final=candidate_size,
            resample_seed=int(split_cfg.resample_seed) + 104_729,
        )
        candidate_result = resample_weighted_proposals(selected_pool, candidate_split)
        candidate_resampling_summary = resampling_summary(candidate_result)
        candidate = add_dsps_closure_photometry(
            candidate_result.frame,
            config,
            batch_size=int(gen_cfg.jax_batch_size),
            noise_seed=int(split_cfg.noise_seed),
            flux_error_model=gen_cfg.flux_error_model,
            verbose=verbose,
        )
        selected_candidate, photometric_summary = apply_photometric_selection(
            candidate,
            bands,
            selection_cfg,
        )
        photometric_summary["enabled"] = True
        _progress(
            verbose,
            f"{split_cfg.name}: photometric selected "
            f"{len(selected_candidate)}/{len(candidate)} candidates",
        )
        if len(selected_candidate) < int(split_cfg.n_final):
            raise RuntimeError(
                f"Split {split_cfg.name} selected only {len(selected_candidate)} "
                f"photometric candidates for target {int(split_cfg.n_final)}. "
                "Increase synthetic_diffsky.selection.photometric_oversample_factor "
                "or relax the S/N selection."
            )
        final = selected_candidate.sample(
            n=int(split_cfg.n_final),
            replace=False,
            random_state=int(split_cfg.resample_seed) + 271_828,
        ).reset_index(drop=True)
        final = _reset_final_identity(final, split_cfg)
    else:
        if not output_layers:
            final = add_dsps_closure_photometry(
                result.frame,
                config,
                batch_size=int(gen_cfg.jax_batch_size),
                noise_seed=int(split_cfg.noise_seed),
                flux_error_model=gen_cfg.flux_error_model,
                verbose=verbose,
            )
            final = append_snr_selection_columns(final, bands, selection_cfg)
    final_duplicate_fraction = _duplicate_fraction(final)
    _progress(verbose, f"{split_cfg.name}: writing final parquet {final_path}")
    final.to_parquet(final_path, index=False)
    raw_pool = pd.concat(proposals, ignore_index=True)
    summary = {
        "status": "generated",
        "source_seed": int(split_cfg.source_seed),
        "noise_seed": int(split_cfg.noise_seed),
        "resample_seed": int(split_cfg.resample_seed),
        "final_path": str(final_path),
        "proposal_dir": str(proposal_dir),
        "n_shards": int(len(proposals)),
        "shards": shard_metadata,
        "raw_pool_size": int(len(raw_pool)),
        "selected_pool_size": int(len(selected_pool)),
        "proposal_selection": proposal_selection_summary,
        "photometric_selection": photometric_summary,
        "output_layers": layer_summaries,
        "candidate_size": int(candidate_size),
        "candidate_resampling": candidate_resampling_summary,
        "duplication_gate": str(gen_cfg.duplication_gate),
        "max_duplication_fraction": float(gen_cfg.max_duplication_fraction),
        "pool_duplicate_fraction": float(result.duplicate_fraction),
        "resampling_duplicate_warning": bool(
            result.duplicate_fraction > float(gen_cfg.max_duplication_fraction)
        ),
        **resampling_summary(result),
        "duplicate_fraction": float(final_duplicate_fraction),
        "final_size": int(len(final)),
        "metallicity": _metallicity_summary(final),
        "selected_pool_metallicity": _metallicity_summary(selected_pool),
        "raw_metallicity": _metallicity_summary(raw_pool),
    }
    return summary


def _manifest_output_layers(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {"enabled": False}
    payload = dict(raw)
    payload["enabled"] = bool(payload.get("enabled", False))
    return payload


def _resolve_output_layers(
    raw: dict[str, Any],
    *,
    fallback_selection: dict[str, Any],
    split_name: str,
    n_final: int,
) -> dict[str, dict[str, Any]]:
    if not bool((raw or {}).get("enabled", False)):
        return {}
    layers: dict[str, dict[str, Any]] = {}
    for layer_name in ("survey_like", "inference_ready"):
        layer_raw = dict((raw or {}).get(layer_name, {}) or {})
        if not bool(layer_raw.get("enabled", layer_name == "inference_ready")):
            continue
        selection = dict(fallback_selection)
        selection.update(dict(layer_raw.get("selection", {}) or {}))
        split_sizes = dict(layer_raw.get("split_sizes", {}) or {})
        if split_name in split_sizes:
            target_size = int(split_sizes[split_name])
        else:
            size_factor = float(layer_raw.get("size_factor", 1.0))
            target_size = int(np.ceil(float(n_final) * size_factor))
        layers[layer_name] = {
            "name": layer_name,
            "directory": str(layer_raw.get("directory", layer_name)),
            "selection": normalize_selection(selection),
            "target_size": max(0, int(target_size)),
            "allow_smaller": bool(
                layer_raw.get("allow_smaller", layer_name != "inference_ready")
            ),
            "mirror_to_root": bool(
                layer_raw.get("mirror_to_root", layer_name == "inference_ready")
            ),
        }
    if not any(layer.get("mirror_to_root") for layer in layers.values()):
        raise ValueError(
            "synthetic_diffsky.output_layers must mark one layer with "
            "mirror_to_root=true so train/validation/test parquets remain defined"
        )
    return layers


def _build_layered_final_catalogs(
    config: dict[str, Any],
    gen_cfg: SyntheticDiffskyConfig,
    split_cfg,
    selected_pool: pd.DataFrame,
    layers: dict[str, dict[str, Any]],
    *,
    bands: list[str],
    verbose: bool,
) -> tuple[pd.DataFrame, dict[str, Any], int, dict[str, Any]]:
    candidate_size = 0
    for layer in layers.values():
        target = int(layer["target_size"])
        if target <= 0:
            continue
        oversample = float(layer["selection"]["photometric_oversample_factor"])
        candidate_size = max(candidate_size, int(np.ceil(target * oversample)))
    candidate_size = max(candidate_size, int(split_cfg.n_final))
    candidate_split = replace(
        split_cfg,
        n_final=candidate_size,
        resample_seed=int(split_cfg.resample_seed) + 104_729,
    )
    candidate_result = resample_weighted_proposals(selected_pool, candidate_split)
    candidate_resampling_summary = resampling_summary(candidate_result)
    candidate = add_dsps_closure_photometry(
        candidate_result.frame,
        config,
        batch_size=int(gen_cfg.jax_batch_size),
        noise_seed=int(split_cfg.noise_seed),
        flux_error_model=gen_cfg.flux_error_model,
        verbose=verbose,
    )
    summaries: dict[str, Any] = {}
    root_final: pd.DataFrame | None = None
    for layer_index, (layer_name, layer) in enumerate(layers.items()):
        target = int(layer["target_size"])
        selection_cfg = layer["selection"]
        if photometric_selection_enabled(selection_cfg):
            selected, selection_summary = apply_photometric_selection(
                candidate,
                bands,
                selection_cfg,
            )
            selection_summary["enabled"] = True
        else:
            selected = append_snr_selection_columns(candidate, bands, selection_cfg)
            selection_summary = {
                "enabled": False,
                "input_size": int(len(candidate)),
                "selected_size": int(len(selected)),
                "selected_fraction": 1.0 if len(candidate) else 0.0,
            }
        _progress(
            verbose,
            f"{split_cfg.name}: layer {layer_name} selected "
            f"{len(selected)}/{len(candidate)} candidates target={target}",
        )
        if len(selected) < target and not bool(layer["allow_smaller"]):
            raise RuntimeError(
                f"Split {split_cfg.name} layer {layer_name} selected only "
                f"{len(selected)} candidates for target {target}. Increase "
                "photometric_oversample_factor or relax the observable selection."
            )
        n_write = min(target, len(selected)) if bool(layer["allow_smaller"]) else target
        if n_write > 0 and len(selected) > n_write:
            layer_frame = selected.sample(
                n=n_write,
                replace=False,
                random_state=int(split_cfg.resample_seed) + 271_828 + 1009 * layer_index,
            )
        else:
            layer_frame = selected.head(n_write)
        layer_split = replace(
            split_cfg,
            object_id_start=int(split_cfg.object_id_start)
            + int(LAYER_OBJECT_ID_OFFSETS.get(layer_name, 20_000_000_000)),
        )
        layer_frame = _reset_final_identity(layer_frame, layer_split)
        layer_frame["sample_layer"] = layer_name
        layer_dir = ensure_dir(gen_cfg.output_dir / str(layer["directory"]))
        layer_path = layer_dir / f"{split_cfg.name}.parquet"
        _progress(
            verbose,
            f"{split_cfg.name}: writing layer {layer_name} parquet {layer_path}",
        )
        layer_frame.to_parquet(layer_path, index=False)
        summaries[layer_name] = {
            "path": str(layer_path),
            "directory": str(layer_dir),
            "target_size": int(target),
            "final_size": int(len(layer_frame)),
            "allow_smaller": bool(layer["allow_smaller"]),
            "mirror_to_root": bool(layer["mirror_to_root"]),
            "selection": normalize_selection(selection_cfg),
            "selection_summary": selection_summary,
            "duplicate_fraction": float(_duplicate_fraction(layer_frame)),
        }
        if bool(layer["mirror_to_root"]):
            root_final = layer_frame.copy()
    if root_final is None:
        raise RuntimeError("No output layer was configured with mirror_to_root=true")
    return root_final, summaries, int(candidate_size), candidate_resampling_summary


def _progress(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[diffsky][generate] {message}", flush=True)


def _stamp_proposal_ids(
    frame: pd.DataFrame,
    split_cfg,
    shard_index: int,
) -> pd.DataFrame:
    stamped = frame.copy()
    stamped["source_seed"] = int(split_cfg.source_seed)
    stamped["source_split"] = split_cfg.name
    stamped["source_shard"] = int(shard_index)
    stamped["source_proposal_id"] = [
        f"{split_cfg.name}:{split_cfg.source_seed}:{shard_index}:{i}"
        for i in range(len(stamped))
    ]
    return stamped


def _reset_final_identity(frame: pd.DataFrame, split_cfg) -> pd.DataFrame:
    final = frame.copy().reset_index(drop=True)
    final = final.drop(columns=["object_id", "split"], errors="ignore")
    final.insert(0, "split", split_cfg.name)
    final.insert(
        0,
        "object_id",
        np.arange(
            int(split_cfg.object_id_start),
            int(split_cfg.object_id_start) + len(final),
            dtype=np.int64,
        ),
    )
    return final


def _duplicate_fraction(frame: pd.DataFrame) -> float:
    if len(frame) == 0 or "source_proposal_id" not in frame:
        return 0.0
    return 1.0 - (
        float(pd.Series(frame["source_proposal_id"]).nunique()) / float(len(frame))
    )


def _pool_is_sufficient(
    result,
    split_cfg,
    gen_cfg: SyntheticDiffskyConfig,
    *,
    final_attempt: bool = False,
) -> bool:
    if int(split_cfg.n_final) == 0:
        return True
    min_pool = int(np.ceil(float(gen_cfg.pool_size_factor) * int(split_cfg.n_final)))
    min_ess = float(gen_cfg.min_ess_fraction) * int(split_cfg.n_final)
    if result.pool_size < min_pool or result.ess < min_ess:
        return False
    if result.duplicate_fraction <= float(gen_cfg.max_duplication_fraction):
        return True
    gate = str(gen_cfg.duplication_gate)
    if gate == "warn_after_max_shards":
        return bool(final_attempt)
    if gate == "warn":
        return True
    if gate != "fail":
        raise ValueError(
            "synthetic_diffsky.duplication_gate must be 'fail', 'warn', "
            "or 'warn_after_max_shards'"
        )
    return False


def _metallicity_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if "metallicity_clipped" not in frame:
        return {}
    mask = frame["metallicity_clipped"].astype(bool).to_numpy()
    low = int(
        frame.get("metallicity_clip_low", pd.Series(False, index=frame.index))
        .astype(bool)
        .sum()
    )
    high = int(
        frame.get("metallicity_clip_high", pd.Series(False, index=frame.index))
        .astype(bool)
        .sum()
    )
    return metallicity_summary_payload(
        type(
            "MetallicityResult",
            (),
            {
                "clipped_mask": mask,
                "clip_low_count": low,
                "clip_high_count": high,
            },
        )()
    )


def _write_schema(path: Path, config: dict[str, Any]) -> None:
    bands = [str(band["name"]) for band in config["bands"]]
    truth_columns = [
        GROUND_TRUTH_COLUMNS[name] for name in DIFFSKY_BASIC_PARAMETER_NAMES
    ]
    payload = {
        "name": "diffsky_dsps_closure_full",
        "parameter_order": list(DIFFSKY_BASIC_PARAMETER_NAMES),
        "ground_truth_columns": {
            name: GROUND_TRUTH_COLUMNS[name] for name in DIFFSKY_BASIC_PARAMETER_NAMES
        },
        "truth_columns": truth_columns,
        "bands": bands,
        "flux_columns": {
            band: {
                "flux_true": f"flux_true_{band}",
                "flux": f"flux_{band}",
                "fluxerr": f"fluxerr_{band}",
                "mask": f"mask_{band}",
            }
            for band in bands
        },
        "units": {
            "flux": "fnu_cgs",
            "redshift_true": "dimensionless",
            "log10_stellar_metallicity_true": "log10(Z/Z_sun)",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_all_catalog(out: Path) -> None:
    paths = [out / "train.parquet", out / "validation.parquet", out / "test.parquet"]
    if not all(path.exists() for path in paths):
        return
    frames = [pd.read_parquet(path) for path in paths]
    all_frame = pd.concat(frames, ignore_index=True)
    all_frame.to_parquet(out / "all_50k.parquet", index=False)


def _write_layer_all_catalogs(out: Path, output_layers: dict[str, Any]) -> None:
    if not bool((output_layers or {}).get("enabled", False)):
        return
    for layer_name in ("survey_like", "inference_ready"):
        spec = dict((output_layers or {}).get(layer_name, {}) or {})
        if not bool(spec.get("enabled", layer_name == "inference_ready")):
            continue
        layer_dir = out / str(spec.get("directory", layer_name))
        paths = [
            layer_dir / "train.parquet",
            layer_dir / "validation.parquet",
            layer_dir / "test.parquet",
        ]
        if not all(path.exists() for path in paths):
            continue
        frames = [pd.read_parquet(path) for path in paths]
        all_frame = pd.concat(frames, ignore_index=True)
        all_frame.to_parquet(layer_dir / "all.parquet", index=False)
        if len(all_frame) == 50_000:
            all_frame.to_parquet(layer_dir / "all_50k.parquet", index=False)
