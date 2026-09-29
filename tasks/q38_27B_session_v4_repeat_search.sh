#!/usr/bin/env bash
# 601 original V4 sessions + 6 diverse search/recovery cases (v2).
# First launch: --refresh. Later no-argument launches resume this profile only.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SESSION_PROFILE=repeat_search_v2
exec bash "${SCRIPT_DIR}/q38_27B_session_v4_grpo.sh" "$@"
