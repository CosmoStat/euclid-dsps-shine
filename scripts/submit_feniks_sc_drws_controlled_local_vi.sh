#!/bin/bash
set -Eeuo pipefail
export LOCAL_VI_CONTROLLED_OPTIMIZATION=1
exec bash scripts/submit_feniks_sc_drws_qualified_local_vi.sh "$@"
