#!/bin/bash
set -Eeuo pipefail
export LOCAL_VI_WAKE_HOLDOUT=1
exec bash scripts/submit_feniks_sc_drws_wake_descent.sh "${1:?balanced env}" "${2:?original completed transport64 pilot}" "${3:?new output root}"
