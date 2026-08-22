#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V="$ROOT/external/vendor"
mkdir -p "$V"
clone_or_update () {
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then git -C "$dest" fetch --depth 1 origin && git -C "$dest" reset --hard origin/HEAD; else git clone --depth 1 "$url" "$dest"; fi
}
clone_or_update https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series.git "$V/RDEx-Series"
clone_or_update https://github.com/P-N-Suganthan/CEC2017.git "$V/CEC2017"
clone_or_update https://github.com/CMA-ES/pycma.git "$V/pycma" || true
printf 'Verified sources fetched. Optional DRL-AEOSF/constraint-consensus slots remain disabled unless a verified author repository URL is supplied.
'
