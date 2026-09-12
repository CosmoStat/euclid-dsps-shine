#!/bin/bash
#SBATCH --job-name=feniks_expert_audit
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
set -Eeuo pipefail
module purge
module load arch/h100
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
cd "${AVI_NEXT_CODE:?}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$SCRATCH/feniks_sc_drws_runtime/mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
TRAINING=$(python - "${AVI_NEXT_ROOT:?}" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]) / "MANIFEST.json").read_text())["upstream_training"])
PY
)
python -m scripts.audit_feniks_avi_experts \
  --training "$TRAINING" --arm B_experts \
  --out "$AVI_NEXT_ROOT/expert_audits/B_experts"
for ARM in H2_experts H8_experts; do
  python -m scripts.audit_feniks_avi_experts \
    --training "$AVI_NEXT_ROOT" --arm "$ARM" \
    --out "$AVI_NEXT_ROOT/expert_audits/$ARM"
done
python - "$AVI_NEXT_ROOT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "expert_audits").glob("*/EXPERT_AUDIT.json")):
    data = json.loads(path.read_text())
    rows.append({
        "arm": data["arm"], "physical_5d_mira_raw": data["physical_5d_mira_raw"],
        **data["support"], **data["gate"], **data["responsibilities"],
    })
pd.DataFrame(rows).to_json(root / "expert_audits/summary.json", orient="records", indent=2)
(root / "EXPERT_AUDIT_COMPLETE.json").write_text(json.dumps({
    "status": "EXPERT_AUDIT_COMPLETE", "arms": [r["arm"] for r in rows],
    "truth_used_for_mira_only": True, "scientific_promotion": False,
}, indent=2, sort_keys=True))
print(pd.DataFrame(rows).to_string(index=False))
PY
