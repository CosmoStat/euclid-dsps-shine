from pathlib import Path

from euclid_dsps.amortized.catalog_identity import object_id_column_from_config
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_cosmos_config_preserves_farmer_object_ids() -> None:
    config = load_config(
        ROOT / "configs/experiments/popcosmos_a24_rws_joint.yaml"
    )
    assert object_id_column_from_config(config) == "object_id"
    assert config["truth"]["redshift_column"] == "redshift_true"
    wake = config["amortized"]["objective"]["wake"]
    assert wake["start_encoder_epoch"] == 4
    assert wake["every_encoder_epochs"] == 4
    assert wake["calibration_loss_weight"] == 1.0


def test_cosmos_smc_pilot_has_stable_features_and_tempered_wake() -> None:
    config = load_config(
        ROOT / "configs/experiments/popcosmos_a24_rws_smc_v3.yaml"
    )
    amortized = config["amortized"]
    wake = amortized["objective"]["wake"]
    assert (
        amortized["features"]["stats_catalog_path"]
        == "Data/cosmos2020/prepared/farmer_a24_full.parquet"
    )
    assert amortized["encoder"]["base_components"] == 2
    assert wake["sampler"] == "smc"
    assert wake["n_particles"] == 8
    assert wake["smc_temperatures"] == [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    assert wake["calibration_loss_weight"] == 1.0


def test_native_15d_band_ablation_and_h100_array_contract() -> None:
    native26 = load_config(
        ROOT / "configs/experiments/popcosmos_native15d_rws.yaml"
    )
    native24 = load_config(
        ROOT / "configs/experiments/popcosmos_native15d_rws_24band.yaml"
    )
    assert len(native26["bands"]) == 26
    assert native26["dataset"]["band_subset"] == "cosmos26"
    assert len(native24["bands"]) == 24
    assert native24["dataset"]["band_subset"] == "cosmos24_no_irac"
    assert native24["amortized"]["features"]["n_flux_bands"] == 24
    assert native24["amortized"]["encoder"]["input_dim"] == 48

    array_submit = (
        ROOT / "scripts/submit_popcosmos_native15d_array.sh"
    ).read_text()
    array_job = (
        ROOT / "scripts/popcosmos_native15d_array_h100.slurm"
    ).read_text()
    native_job = (ROOT / "scripts/popcosmos_native15d_rws_h100.slurm").read_text()
    assert 'STAGE must be n5k,n20k,n40k,full' in native_job
    assert "submit_stage full" in array_submit
    assert '--array="0-1%' in array_submit
    assert "afterok:" in array_submit
    assert "MAP_26" not in array_submit
    assert "wake_particles=8" in array_submit
    assert "RESUME_N5K" in array_submit
    assert "export BAND_VARIANT=26" in array_job
    assert "export BAND_VARIANT=24" in array_job
    assert "objective=reweighted_wake_sleep k=8" in array_job


def test_native_15d_full_continuation_uses_four_h100s_and_fixed_cohorts() -> None:
    worker = (
        ROOT / "scripts/popcosmos_native15d_continue_full_h100.slurm"
    ).read_text()
    submit = (
        ROOT / "scripts/submit_popcosmos_native15d_continuation.sh"
    ).read_text()

    assert "#SBATCH --gres=gpu:4" in worker
    assert 'EXPECTED_GPUS="${EXPECTED_GPUS:-4}"' in worker
    assert 'START_EPOCH="${START_EPOCH:-33}"' in worker
    assert 'END_EPOCH="${END_EPOCH:-120}"' in worker
    assert '--initial-checkpoint "$SOURCE_CHECKPOINT"' in worker
    assert '--start-epoch "$START_EPOCH"' in worker
    assert '--train-indices-file "$SOURCE_TRAIN_INDICES"' in worker
    assert '--validation-indices-file "$SOURCE_VALIDATION_INDICES"' in worker
    assert "--data-parallel pmap" in worker
    assert '--row-indices-file "$EVAL_INDICES"' in worker
    assert "held-out inference cohort: unchanged" in worker
    assert "feature statistics changed during continuation" in worker
    assert 'optimizer_state_resumed") is not False' in worker
    assert '--array="0-1%${ARRAY_CONCURRENCY}"' in submit
    assert "--gres=gpu:4" in submit
    assert "END_EPOCH must be at least 100" in submit
    assert "shared train/validation/evaluation cohorts: PASS" in submit


def test_jean_zay_wrappers_scale_gpu_and_smoke_arrays() -> None:
    rws = (ROOT / "scripts/cosmos2020_rws_h100.slurm").read_text()
    submit = (ROOT / "scripts/submit_cosmos2020_reproduction.sh").read_text()
    mclmc = (ROOT / "scripts/submit_cosmos2020_mclmc.sh").read_text()
    a24_forward = (
        ROOT / "scripts/popcosmos_a24_forward_audit_h100.slurm"
    ).read_text()
    assert "DATA_PARALLEL=pmap" in rws
    assert "popcosmos_a24_rws_v2" in rws
    assert "smoke) SIZE=512; EPOCHS=4" in rws
    assert "--allow-inference-fail" in rws
    assert 'test -e "$PREVIOUS/DONE"' in rws
    assert '--data-parallel "$DATA_PARALLEL"' in rws
    assert "--selection-mode sequential" in rws
    assert "--selection-mode random" not in rws
    assert 'TRAIN_PER_DEVICE_BATCH_SIZE * N_GPUS' in rws
    assert 'TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"' in rws
    assert "GRES=(gpu:1 gpu:1 gpu:1 gpu:4 gpu:4)" in submit
    assert "TIMES=(00:30:00 01:00:00 04:00:00 08:00:00 20:00:00)" in submit
    assert 'test -e "$ROOT_DIR/$previous/DONE"' in submit
    assert 'test -s "$ROOT_DIR/$previous/DONE"' not in submit
    assert '--config "$CONFIG"' in submit
    assert 'CONFIG="$CONFIG"' in submit
    assert "index<=target_index" in submit
    assert '[[ -e "$ROOT_DIR" || "$stage" == "$THROUGH" ]]' not in submit
    assert "validate_cosmos2020_reproduction.py" in submit
    assert 'if [[ "$MODE" == "smoke" ]]; then ARRAY="0-1%2"; fi' in mclmc
    assert 'test -e "$MODEL_ROOT/DONE"' in mclmc
    assert "audit_popcosmos_a24_dsps_forward.py" in a24_forward
    assert 'touch "$OUT/DONE"' in a24_forward
