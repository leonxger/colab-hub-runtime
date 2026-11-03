#!/bin/bash
# Builds the DeepSeek runtime tarball (llama.cpp at DS_TAG with the patch series, CUDA sm_120a, AVX-512 CPU experts, libpin.so): deepseek.sh [--print-cmake | --list-package | --package-only]; a build needs WORK.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime/build/common.sh
. "$HERE/common.sh"

FAMILY=deepseek
rt_family "$FAMILY"
NAME_TAG=${DS_TAG}p$(rt_series | wc -l)
ARCH_LABEL=sm120a
ARCHS="sm_120a"
TARGETS="llama-server llama-fit-params"
TOP=(bin lib BUILDINFO.json LICENSE)
REQUIRED=(bin/llama-server bin/llama-fit-params bin/libggml-cuda.so lib/libpin.so BUILDINFO.json LICENSE)

read -r -d '' CMAKE_LINE << 'LINE' || true
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_AVX_VNNI=ON -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120a-real -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
LINE

# binaries and libraries from build/bin, libpin.so compiled from runtime/src/pin.c
rt_stage_family() {
  local b=$WORK/build/bin s=$WORK/stage
  mkdir -p "$s/bin" "$s/lib"
  cp -a "$b/llama-server" "$b/llama-fit-params" "$s/bin/"
  cp -a "$b"/*.so* "$s/bin/"
  gcc -O2 -shared -fPIC "$RT/src/pin.c" -ldl -lpthread -o "$s/lib/libpin.so"
  cp "$WORK/src/LICENSE" "$s/LICENSE"
}

rt_main "$@"
