#!/bin/bash
set -Eeuo pipefail
export LOCAL_VI_NIGHT_PILOT_ROOT="${2:?provide transport64 pilot root}"
export LOCAL_VI_NIGHT_DEPENDENCY="${3:?provide pilot Slurm ID}"
export DIAGNOSTIC_ROOT="${4:?provide NEW night root}"
exec bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh "${1:?provide balanced env}"
