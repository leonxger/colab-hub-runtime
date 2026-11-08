#!/bin/bash
# Builds the voice runtime tarball (audio.cpp audiocpp_server and qwentts.cpp tts-server, CUDA sm_80 and sm_120a, vendored num2words): voice.sh [--print-cmake audiocpp|qwentts|audiocpp-cpu | --list-package | --package-only | --cpu-audiocpp]; a build needs WORK.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime/build/common.sh
. "$HERE/common.sh"

ARCHS="sm_80 sm_120a"
TOP=(audiocpp qwentts py licenses BUILDINFO.json LICENSE)
REQUIRED=(audiocpp/bin/audiocpp_server audiocpp/bin/libggml-cuda.so qwentts/bin/tts-server qwentts/bin/libggml-cuda.so py/num2words/__init__.py py/num2words/COPYING py/num2words/SOURCE.txt licenses/audio.cpp.LICENSE licenses/qwentts.cpp.LICENSE licenses/NOTICE.txt BUILDINFO.json LICENSE)

LAUNCHERS='-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache'
# ENGINE_ENABLE_CPU_ALL_VARIANTS makes the backends loadable modules next to the binary (the Dockerfile of the engine does the same); only the named models are compiled
AUDIOCPP_COMMON="cmake -S src/audiocpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON -DENGINE_ENABLE_OPENMP=ON -DENGINE_ENABLE_VULKAN=OFF -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_tts,voxcpm2 -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_VERSION=$AUDIOCPP_TAG"
CMAKE_AUDIOCPP="$AUDIOCPP_COMMON -B build/audiocpp -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=\"80-real;120a-real\" -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined $LAUNCHERS"
CMAKE_AUDIOCPP_CPU="$AUDIOCPP_COMMON -B build/audiocpp-cpu -DENGINE_ENABLE_CUDA=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
CMAKE_QWENTTS="cmake -S src/qwentts -B build/qwentts -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=\"80-real;120a-real\" -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda' -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda' $LAUNCHERS"

voice_cmake_line() {
  case "$1" in
    audiocpp) printf '%s\n' "$CMAKE_AUDIOCPP" ;;
    audiocpp-cpu) printf '%s\n' "$CMAKE_AUDIOCPP_CPU" ;;
    qwentts) printf '%s\n' "$CMAKE_QWENTTS" ;;
    *) rt_die "--print-cmake takes audiocpp, qwentts or audiocpp-cpu" ;;
  esac
}

voice_pkg_name() { echo "voice-$TREE12.tar.gz"; }

# shallow clone of the pinned tag, commit checked
voice_clone_audiocpp() {
  local head
  [ -z "$(ls -A "$1" 2> /dev/null)" ] || rt_die "$1 exists and is not empty"
  timeout "${CLONE_TIMEOUT:-600}" git clone -q --depth 1 --branch "$AUDIOCPP_TAG" "$AUDIOCPP_URL" "$1" || rt_die "clone of audio.cpp $AUDIOCPP_TAG failed"
  head=$(git -C "$1" rev-parse HEAD) || rt_die "no HEAD in $1"
  [ "$head" = "$AUDIOCPP_COMMIT" ] || rt_die "commit mismatch for audio.cpp $AUDIOCPP_TAG: got $head, want $AUDIOCPP_COMMIT"
  rt_log "cloned audio.cpp $AUDIOCPP_TAG at $head"
}

# qwentts.cpp has no tags: fetch the pinned commit, then its ggml submodule, both commits checked
voice_clone_qwentts() {
  local head ggml
  [ -z "$(ls -A "$1" 2> /dev/null)" ] || rt_die "$1 exists and is not empty"
  mkdir -p "$1"
  git -C "$1" init -q || rt_die "git init in $1"
  git -C "$1" remote add origin "$QWENTTS_URL"
  timeout "${CLONE_TIMEOUT:-600}" git -C "$1" fetch -q --depth 1 origin "$QWENTTS_COMMIT" || rt_die "fetch of qwentts.cpp $QWENTTS_COMMIT failed"
  git -C "$1" checkout -q FETCH_HEAD || rt_die "checkout of qwentts.cpp $QWENTTS_COMMIT failed"
  head=$(git -C "$1" rev-parse HEAD)
  [ "$head" = "$QWENTTS_COMMIT" ] || rt_die "commit mismatch for qwentts.cpp: got $head, want $QWENTTS_COMMIT"
  timeout "${CLONE_TIMEOUT:-600}" git -C "$1" submodule update --init --depth 1 ggml || rt_die "qwentts.cpp ggml submodule fetch failed"
  ggml=$(git -C "$1/ggml" rev-parse HEAD) || rt_die "no HEAD in the ggml submodule"
  [ "$ggml" = "$QWENTTS_GGML_COMMIT" ] || rt_die "ggml commit mismatch for qwentts.cpp: got $ggml, want $QWENTTS_GGML_COMMIT"
  rt_log "fetched qwentts.cpp at $head with ggml $ggml"
}

# the pinned pure-Python wheel into DIR/py: package files, the LGPL text and the source URL
voice_vendor_num2words() {
  local dir=$1 whl got
  whl=$(mktemp "${TMPDIR:-/tmp}/num2words.XXXXXX.whl") || rt_die "no temp file"
  timeout "${DOWNLOAD_TIMEOUT:-180}" curl -fsSL --retry 3 --max-time 120 -o "$whl" "$NUM2WORDS_URL" || { rm -f "$whl"; rt_die "download of num2words $NUM2WORDS_VERSION failed"; }
  got=$(sha256sum "$whl" | cut -c1-64)
  [ "$got" = "$NUM2WORDS_SHA256" ] || { rm -f "$whl"; rt_die "num2words wheel has sha256 $got, want $NUM2WORDS_SHA256"; }
  mkdir -p "$dir/py"
  python3 -I - "$whl" "$dir/py" "$NUM2WORDS_VERSION" "$NUM2WORDS_SOURCE" "$NUM2WORDS_SHA256" << 'PY' || { rm -f "$whl"; rt_die "cannot unpack the num2words wheel"; }
import sys, zipfile
whl, out, version, source, sha = sys.argv[1:6]
z = zipfile.ZipFile(whl)
for n in z.namelist():
    if n.startswith("num2words/") and not n.endswith("/") and ".." not in n and not n.startswith("/"):
        z.extract(n, out)
copying = "num2words-%s.dist-info/COPYING" % version
with open(out + "/num2words/COPYING", "wb") as f:
    f.write(z.read(copying))
with open(out + "/num2words/SOURCE.txt", "w") as f:
    f.write("num2words %s, GNU Lesser General Public License 2.1 (COPYING in this directory), unmodified.\nSource: %s\nWheel sha256: %s\n" % (version, source, sha))
PY
  rm -f "$whl"
  rt_log "vendored num2words $NUM2WORDS_VERSION"
}

voice_notice() {
  cat > "$1" << EOF
Components of this tarball and their licences:
audio.cpp $AUDIOCPP_TAG ($AUDIOCPP_URL): Apache-2.0, licenses/audio.cpp.LICENSE
qwentts.cpp $QWENTTS_COMMIT ($QWENTTS_URL) with its ggml fork $QWENTTS_GGML_COMMIT: MIT, licenses/qwentts.cpp.LICENSE
num2words $NUM2WORDS_VERSION ($NUM2WORDS_SOURCE): LGPL-2.1, py/num2words/COPYING, unmodified, loaded by the relay from py/ on PYTHONPATH
The CUDA runtime libraries are not in this tarball (system CUDA of the VM or the cudart asset).
EOF
}

voice_stage() {
  local w=$WORK s=$WORK/stage
  mkdir -p "$s/audiocpp/bin" "$s/qwentts/bin" "$s/licenses"
  cp -a "$w/build/audiocpp/bin/audiocpp_server" "$s/audiocpp/bin/"
  cp -a "$w"/build/audiocpp/bin/*.so* "$s/audiocpp/bin/"
  cp -a "$w/build/qwentts/tts-server" "$s/qwentts/bin/"
  cp -a "$w"/build/qwentts/*.so* "$s/qwentts/bin/"
  cp "$w/src/audiocpp/LICENSE" "$s/LICENSE"
  cp "$w/src/audiocpp/LICENSE" "$s/licenses/audio.cpp.LICENSE"
  cp "$w/src/qwentts/LICENSE" "$s/licenses/qwentts.cpp.LICENSE"
  voice_notice "$s/licenses/NOTICE.txt"
  voice_vendor_num2words "$s"
}

# BUILDINFO.json without timestamps, so the same inputs give the same bytes
voice_buildinfo() {
  TREE12="$TREE12" CMAKE_AUDIOCPP="$CMAKE_AUDIOCPP" CMAKE_QWENTTS="$CMAKE_QWENTTS" CUDA_VERSION="$(rt_cuda_version)" MAX_GLIBC="${MAX_GLIBC:-2.39}" \
    AUDIOCPP_URL="$AUDIOCPP_URL" AUDIOCPP_TAG="$AUDIOCPP_TAG" AUDIOCPP_COMMIT="$AUDIOCPP_COMMIT" QWENTTS_URL="$QWENTTS_URL" QWENTTS_COMMIT="$QWENTTS_COMMIT" \
    QWENTTS_GGML_COMMIT="$QWENTTS_GGML_COMMIT" NUM2WORDS_VERSION="$NUM2WORDS_VERSION" NUM2WORDS_SHA256="$NUM2WORDS_SHA256" NUM2WORDS_SOURCE="$NUM2WORDS_SOURCE" \
    VOICE_GGUF_REPO="$VOICE_GGUF_REPO" VOICE_GGUF_REV="$VOICE_GGUF_REV" VOICE_GGUF_FILE="$VOICE_GGUF_FILE" VOICE_GGUF_SIZE="$VOICE_GGUF_SIZE" VOICE_GGUF_SHA256="$VOICE_GGUF_SHA256" \
    CUDA_IMAGE="$CUDA_IMAGE" CUDA_IMAGE_DIGEST="$CUDA_IMAGE_DIGEST" ARCHS="$ARCHS" \
    python3 -I - "$1" << 'PY'
import json, os, sys
e = os.environ
info = {
    "schema": 1,
    "family": "voice",
    "tree12": e["TREE12"],
    "audiocpp": {"url": e["AUDIOCPP_URL"], "tag": e["AUDIOCPP_TAG"], "commit": e["AUDIOCPP_COMMIT"], "server": "audiocpp/bin/audiocpp_server", "target": "audiocpp_server", "models": ["qwen3_tts", "voxcpm2"]},
    "qwentts": {"url": e["QWENTTS_URL"], "commit": e["QWENTTS_COMMIT"], "ggml_commit": e["QWENTTS_GGML_COMMIT"], "server": "qwentts/bin/tts-server", "target": "tts-server"},
    "num2words": {"version": e["NUM2WORDS_VERSION"], "wheel_sha256": e["NUM2WORDS_SHA256"], "source": e["NUM2WORDS_SOURCE"], "path": "py/num2words"},
    "gguf": {"repo": e["VOICE_GGUF_REPO"], "revision": e["VOICE_GGUF_REV"], "file": e["VOICE_GGUF_FILE"], "size": int(e["VOICE_GGUF_SIZE"]), "sha256": e["VOICE_GGUF_SHA256"]},
    "cuda": {"version": e["CUDA_VERSION"], "image": e["CUDA_IMAGE"], "digest": e["CUDA_IMAGE_DIGEST"]},
    "archs": e["ARCHS"].split(),
    "max_glibc": e["MAX_GLIBC"],
    "cmake": {"audiocpp": e["CMAKE_AUDIOCPP"], "qwentts": e["CMAKE_QWENTTS"]},
}
with open(sys.argv[1], "w") as f:
    json.dump(info, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

voice_main() {
  local mode=${1:-build} stage out
  umask 022
  case "$mode" in
    --print-cmake) voice_cmake_line "${2:-}" ;;
    --list-package)
      TREE12=$(rt_tree12) || exit 1
      voice_pkg_name
      printf '%s\n' "${REQUIRED[@]}" ;;
    --cpu-audiocpp)
      : "${WORK:?set WORK to a writable absolute build directory}"
      [[ "$WORK" = /* ]] || rt_die "WORK must be an absolute path"
      export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-2G}
      for t in git cmake ninja ccache gcc g++ curl; do command -v "$t" > /dev/null || rt_die "$t is not installed"; done
      rt_fresh "$WORK/src/audiocpp" "$WORK/build/audiocpp-cpu"
      mkdir -p "$WORK/src" "$WORK/build"
      voice_clone_audiocpp "$WORK/src/audiocpp"
      (cd "$WORK" && eval "$CMAKE_AUDIOCPP_CPU") || rt_die "cmake configure (cpu)"
      cmake --build "$WORK/build/audiocpp-cpu" -j "${JOBS:-4}" --target audiocpp_server || rt_die "build (cpu)"
      [ -x "$WORK/build/audiocpp-cpu/bin/audiocpp_server" ] || rt_die "no audiocpp_server in $WORK/build/audiocpp-cpu/bin"
      rt_log "built $WORK/build/audiocpp-cpu/bin/audiocpp_server" ;;
    build | --package-only)
      : "${WORK:?set WORK to a writable absolute build directory}"
      [[ "$WORK" = /* ]] || rt_die "WORK must be an absolute path"
      TREE12=$(rt_tree12) || exit 1
      stage=$WORK/stage; out=$WORK/out
      export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-2G}
      if [ "$mode" = build ]; then
        rt_need_tools
        command -v curl > /dev/null || rt_die "curl is not installed in the build image"
        rt_check_nvcc
        rt_fresh "$WORK/src/audiocpp" "$WORK/src/qwentts" "$WORK/build/audiocpp" "$WORK/build/qwentts" "$stage" "$out"
        mkdir -p "$WORK/src" "$WORK/build"
        voice_clone_audiocpp "$WORK/src/audiocpp"
        voice_clone_qwentts "$WORK/src/qwentts"
        (cd "$WORK" && eval "$CMAKE_AUDIOCPP") || rt_die "cmake configure (audio.cpp)"
        cmake --build "$WORK/build/audiocpp" -j "${JOBS:-4}" --target audiocpp_server || rt_die "build (audio.cpp)"
        (cd "$WORK" && eval "$CMAKE_QWENTTS") || rt_die "cmake configure (qwentts.cpp)"
        cmake --build "$WORK/build/qwentts" -j "${JOBS:-4}" --target tts-server || rt_die "build (qwentts.cpp)"
      else
        rt_fresh "$stage" "$out"
      fi
      mkdir -p "$stage" "$out"
      voice_stage
      voice_buildinfo "$stage/BUILDINFO.json"
      rt_require "$stage" "${REQUIRED[@]}"
      if [ "$mode" = build ]; then
        # shellcheck disable=SC2086
        bash "$RT/build/check_elf.sh" "$stage/audiocpp" $ARCHS || rt_die "check_elf.sh (audiocpp)"
        # shellcheck disable=SC2086
        bash "$RT/build/check_elf.sh" "$stage/qwentts" $ARCHS || rt_die "check_elf.sh (qwentts)"
      fi
      rt_tar "$stage" "$out/$(voice_pkg_name)" "${TOP[@]}"
      ls -l "$out" >&2
      rt_log "packed $out/$(voice_pkg_name)" ;;
    *) echo "usage: $(basename "$0") [--print-cmake audiocpp|qwentts|audiocpp-cpu | --list-package | --package-only | --cpu-audiocpp]; a build needs WORK and optionally TREE12, JOBS" >&2; exit 2 ;;
  esac
}

[ "${BASH_SOURCE[0]}" != "$0" ] || voice_main "$@"
