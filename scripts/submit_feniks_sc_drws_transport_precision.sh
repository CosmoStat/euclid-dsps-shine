#!/bin/bash
set -Eeuo pipefail
export LOCAL_VI_OBJECTIVE_PRECISION_ROOT="${4:?provide completed objective pilot root}"
exec bash scripts/submit_feniks_sc_drws_objective_pilot.sh \
  "${1:?provide balanced env}" "${2:?provide precision night root}" \
  "${3:?provide long local VI root}" "${5:?provide NEW precision audit root}"
