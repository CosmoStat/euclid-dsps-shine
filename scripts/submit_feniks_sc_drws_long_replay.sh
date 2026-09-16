#!/bin/bash
set -Eeuo pipefail
SOURCE_ENV="${1:?provide balanced env}"
NIGHT_ROOT="${2:?provide precision night root}"
export LOCAL_VI_LONG_REPLAY_ROOT="${3:?provide completed long local VI root}"
NEW_ROOT="${4:?provide NEW replay root}"
unset LOCAL_VI_CONTROLLED_OPTIMIZATION LOCAL_VI_SUPPORT_PROBE_ROOT LOCAL_VI_LONG_OPTIMIZATION
exec bash scripts/submit_feniks_sc_drws_qualified_local_vi.sh "$SOURCE_ENV" "$NIGHT_ROOT" "$NEW_ROOT"
