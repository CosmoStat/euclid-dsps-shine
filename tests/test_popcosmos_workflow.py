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


def test_jean_zay_wrappers_scale_gpu_and_smoke_arrays() -> None:
    rws = (ROOT / "scripts/cosmos2020_rws_h100.slurm").read_text()
    submit = (ROOT / "scripts/submit_cosmos2020_reproduction.sh").read_text()
    mclmc = (ROOT / "scripts/submit_cosmos2020_mclmc.sh").read_text()
    assert "DATA_PARALLEL=pmap" in rws
    assert "popcosmos_a24_rws_v2" in rws
    assert "smoke) SIZE=512; EPOCHS=4" in rws
    assert "--allow-inference-fail" in rws
    assert 'test -e "$PREVIOUS/DONE"' in rws
    assert '--data-parallel "$DATA_PARALLEL"' in rws
    assert "GRES=(gpu:1 gpu:1 gpu:1 gpu:4 gpu:4)" in submit
    assert "TIMES=(00:30:00 01:00:00 04:00:00 08:00:00 20:00:00)" in submit
    assert 'test -e "$ROOT_DIR/$previous/DONE"' in submit
    assert 'test -s "$ROOT_DIR/$previous/DONE"' not in submit
    assert "validate_cosmos2020_reproduction.py" in submit
    assert 'if [[ "$MODE" == "smoke" ]]; then ARRAY="0-1%2"; fi' in mclmc
    assert 'test -e "$MODEL_ROOT/DONE"' in mclmc
