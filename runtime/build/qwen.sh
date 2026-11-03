#!/bin/bash
# Builds the Qwen-family runtime tarball (llama.cpp at QWEN_TAG, both head-trim diffs, CUDA sm_80 and sm_120a): qwen.sh [--print-cmake | --list-package | --package-only]; a build needs WORK.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime/build/common.sh
. "$HERE/common.sh"

FAMILY=qwen
NAME_TAG=$QWEN_TAG
ARCH_LABEL=sm80-sm120a
ARCHS="sm_80 sm_120a"
TARGETS="llama-server llama-fit-params test-backend-ops"
TOP=(bin py BUILDINFO.json LICENSE)
REQUIRED=(bin/llama-server bin/llama-fit-params bin/test-backend-ops bin/libggml-cuda.so py/convert_hf_to_gguf.py py/conversion/__init__.py py/gguf-py/gguf/__init__.py BUILDINFO.json LICENSE)

read -r -d '' CMAKE_LINE << 'LINE' || true
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_CCCL_VERSION=@CCCL@ -DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real" -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
LINE
CMAKE_LINE=${CMAKE_LINE//@CCCL@/$CCCL}

# binaries and backends from build/bin, the converter with gguf-py from the patched source
rt_stage_family() {
  local b=$WORK/build/bin s=$WORK/stage src=$WORK/src
  mkdir -p "$s/bin" "$s/py/gguf-py"
  cp -a "$b/llama-server" "$b/llama-fit-params" "$b/test-backend-ops" "$s/bin/"
  cp -a "$b"/*.so* "$s/bin/"
  cp "$src/convert_hf_to_gguf.py" "$s/py/"
  cp -a "$src/conversion" "$s/py/"
  cp -a "$src/gguf-py/gguf" "$src/gguf-py/pyproject.toml" "$src/gguf-py/LICENSE" "$s/py/gguf-py/"
  cp "$src/LICENSE" "$s/LICENSE"
}

rt_main "$@"
