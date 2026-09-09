#!/bin/bash
set -Eeuo pipefail
cd "${REPO_DIR:-$PWD}"
SOURCE_ENV="${1:-outputs/logs/feniks_sc_drws_balanced_npe_latest.env}"
source "$SOURCE_ENV"
export REPO_DIR="$(pwd -P)"
unset LOCAL_VI_GRADIENT_ISOLATION LOCAL_VI_REDSHIFT_DECOMPOSITION LOCAL_VI_PHOTOMETRY_REFERENCE
unset LOCAL_VI_FULL_DECODER_REFERENCE LOCAL_VI_MDF_PRECISION_REFERENCE
unset LOCAL_VI_TARGET_RESOLUTION_REFERENCE LOCAL_VI_REDSHIFT_PRECISION_REFERENCE
unset LOCAL_VI_OBJECTS LOCAL_VI_STEPS LOCAL_VI_DRAWS
unset DIAGNOSTIC_LOG_ROOT
export LOCAL_VI_PRECISION_NIGHT_REFERENCE="${2:?provide the completed redshift_precision_v1 root}"
export DIAGNOSTIC_ROOT="${3:?provide a NEW output root}"
exec bash scripts/submit_feniks_sc_drws_local_vi_diagnostic.sh "$SOURCE_ENV"
