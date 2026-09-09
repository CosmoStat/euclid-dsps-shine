# Gated overnight local reverse/wake extension

This is a longer local flow training experiment, NOT a global amortized NN
training job or population RWS. The same 16 development objects and two starts
are retained. No independent-cohort generalization or posterior qualification
is claimed. Source checkpoints, parent, context and target stay frozen; local
flow parameters are trainable under `conditional_transport_float64_v1`.

## Allocation and protocol

One job, one node, one H100 (one GPU/node; peak concurrency one GPU after the
pilot). Slurm limit 10 hours, internal limit 9 hours, 5,000,000 counted decoder
evaluations. Adaptation checkpoints: 4096, 16384 and 32768 decoder draws per
arm/start, versus 4096 in the short pilot. Reverse batches 32, wake batches
256; wake preconditions and learning rate remain unchanged. Both arms restart
from the same fixed long-VI source, not a selected pilot winner. Evaluation
and optimization seeds are offset by 100,000,000. Intermediate evaluations use
two K512 replicates, final evaluation two K2048 replicates (K4096 total).

## Automatic resource gate

The job waits for the exact pilot Slurm ID with `afterany`. It then checks
the pilot receipt before loading the model. Failure or missing evidence is a
durable `NIGHT_EXTENSION_NOT_STARTED`, not permission to continue. This gate
briefly occupies the allocated GPU even if it rejects the run.

Required: complete transport64 pilot, 16 cases, 32 distinct PASS audits with
matching hashes, complete final wake checkpoints/contracts/summary receipts,
and matching K4096 source evaluations. In EACH observed/simulated group:

- median paired final wake/source ESS fraction ratio >=1.10;
- at least half of the 16 trajectories have >=4 accepted wake updates;
- no more than half of trajectories regress in ESS fraction.

These are preregistered resource-allocation conditions for further development,
not a scientific support certificate. Neither Pareto-k nor calibration is
certified by passing them. Every original object/start stays in the experiment.
Fresh transport64 audits still gate all optimizer construction in the long run.

## Submit while the pilot runs

Update the branch, activate shine, set REPO_DIR and CACHE_ROOT as in the pilot.

```bash
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"
bash scripts/submit_feniks_sc_drws_objective_night.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_objective_transport64_pilot_v1" \
  1962310 \
  "$BASE/frozen_parent_objective_transport64_night_v1"
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

The dependency ID is checked against the pilot SUBMISSION.json. A new root is
mandatory. Submission changes the latest diagnostic env to the NIGHT job.
The existing pilot keeps running from its own immutable snapshot.

## Readback

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python - "$DIAGNOSTIC_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("NIGHT_GATE.json", "FINAL.json"):
    path = root / name
    print(name, json.loads(path.read_text()) if path.exists() else "pending")
PY
python scripts/summarize_feniks_sc_drws_objective_pilot.py "$DIAGNOSTIC_ROOT"
```

NIGHT_GATE.json records thresholds, group statistics and pinned pilot evidence;
night_gate_pairs.csv records every pair. A started run ends with the standard
OBJECTIVE_PILOT_COMPLETE only when all cases finish, or a failure/budget receipt.
No multi-GPU or population job is chained afterward. Inspect support, rejection
rates and both independent replicates before any next expansion.
