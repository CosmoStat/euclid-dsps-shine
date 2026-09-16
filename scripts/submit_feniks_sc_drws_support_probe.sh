#!/bin/bash
set -Eeuo pipefail
SOURCE_ENV="${1:?balanced environment required}"
NIGHT_ROOT="${2:?completed night root required}"
export LOCAL_VI_SUPPORT_PROBE_ROOT="${3:?completed controlled local root required}"
NEW_ROOT="${4:?new output root required}"
unset LOCAL_VI_CONTROLLED_OPTIMIZATION
exec bash scripts/submit_feniks_sc_drws_qualified_local_vi.sh "$SOURCE_ENV" "$NIGHT_ROOT" "$NEW_ROOT"
