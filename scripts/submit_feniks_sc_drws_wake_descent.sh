#!/bin/bash
set -Eeuo pipefail
export LOCAL_VI_WAKE_DESCENT_ROOT="${2:?provide completed transport64 pilot}"
export DIAGNOSTIC_ROOT="${3:?provide NEW descent pilot root}"
exec bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh "${1:?provide balanced env}"
