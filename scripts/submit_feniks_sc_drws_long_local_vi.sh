#!/bin/bash
set -Eeuo pipefail
unset LOCAL_VI_CONTROLLED_OPTIMIZATION LOCAL_VI_SUPPORT_PROBE_ROOT
export LOCAL_VI_LONG_OPTIMIZATION=1
exec bash scripts/submit_feniks_sc_drws_qualified_local_vi.sh "$@"
