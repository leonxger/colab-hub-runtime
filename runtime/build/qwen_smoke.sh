#!/bin/bash
# Builds the CPU llama-server of the CI smoke from the patched b12015 tree (same series as the Qwen runtime, no CUDA): qwen_smoke.sh [--print-cmake]; a build needs WORK.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime/build/common.sh
. "$HERE/common.sh"

read -r -d '' CMAKE_LINE << 'LINE' || true
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
LINE

case "${1:-build}" in
  --print-cmake) printf '%s\n' "$CMAKE_LINE" ;;
  build)
    : "${WORK:?set WORK to a writable absolute build directory}"
    [[ "$WORK" = /* ]] || rt_die "WORK must be an absolute path"
    export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-2G}
    for t in git python3 cmake ninja ccache gcc g++; do command -v "$t" > /dev/null || rt_die "$t is not installed"; done
    rt_fresh "$WORK/src" "$WORK/build"
    mkdir -p "$WORK"
    rt_clone qwen "$WORK/src"
    rt_apply qwen "$WORK/src"
    (cd "$WORK" && eval "$CMAKE_LINE") || rt_die "cmake configure (cpu)"
    cmake --build "$WORK/build" -j "${JOBS:-4}" --target llama-server || rt_die "build (cpu)"
    [ -x "$WORK/build/bin/llama-server" ] || rt_die "no llama-server in $WORK/build/bin"
    rt_log "built $WORK/build/bin/llama-server" ;;
  *) echo "usage: $(basename "$0") [--print-cmake]; a build needs WORK and optionally JOBS" >&2; exit 2 ;;
esac
