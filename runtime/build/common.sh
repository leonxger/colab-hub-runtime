#!/bin/bash
# Shared steps of the runtime builds: common.sh --apply-only qwen|deepseek DIR clones the pinned tag into DIR and applies the series; qwen.sh, deepseek.sh and cudart.sh source it.
RT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT_VERSIONS="${RT_VERSIONS:-$RT/versions.env}"
RT_PATCHES="${RT_PATCHES:-$RT/patches}"
# shellcheck source=/dev/null
. "$RT_VERSIONS"

rt_log() { echo "runtime: $*" >&2; }
rt_die() { rt_log "FAILED: $*"; exit 1; }

# sets TAG COMMIT SERIES_DIR for qwen or deepseek
rt_family() {
  case "$1" in
    qwen) TAG=$QWEN_TAG; COMMIT=$QWEN_COMMIT ;;
    deepseek) TAG=$DS_TAG; COMMIT=$DS_COMMIT ;;
    *) rt_die "unknown family '$1' (qwen or deepseek)" ;;
  esac
  SERIES_DIR=$RT_PATCHES/$1
  [ -f "$SERIES_DIR/series" ] || rt_die "no series file in $SERIES_DIR"
}

# the entries of the series file in order, without comments and blank lines
rt_series() { grep -v '^[[:space:]]*\(#\|$\)' "$SERIES_DIR/series"; }

# shallow clone of the pinned tag into the empty or missing DIR, commit checked
rt_clone() {
  local head
  rt_family "$1"
  [ -z "$(ls -A "$2" 2> /dev/null)" ] || rt_die "$2 exists and is not empty"
  timeout "${CLONE_TIMEOUT:-600}" git clone -q --depth 1 --branch "$TAG" "$LLAMA_URL" "$2" || rt_die "clone of $TAG failed"
  head=$(git -C "$2" rev-parse HEAD) || rt_die "no HEAD in $2"
  [ "$head" = "$COMMIT" ] || rt_die "commit mismatch for $TAG: got $head, want $COMMIT"
  rt_log "cloned $TAG at $head"
}

# applies the series to the tree DIR in order, diffs with git apply and scripts with python3 SCRIPT DIR, stops at the first failure; sets PATCH_COUNT
rt_apply() {
  local p f
  rt_family "$1"
  PATCH_COUNT=0
  while read -r p; do
    f=$SERIES_DIR/$p
    [ -f "$f" ] || rt_die "series names $p but $f is missing"
    case "$p" in
      *.diff)
        if git -C "$2" apply --reverse --check "$f" > /dev/null 2>&1; then
          rt_log "already applied $p"
        else
          git -C "$2" apply "$f" || rt_die "patch $p"
          rt_log "applied $p"
        fi ;;
      *.py)
        python3 -I "$f" "$2" >&2 || rt_die "patch $p"
        rt_log "applied $p" ;;
      *) rt_die "series entry $p is neither .diff nor .py" ;;
    esac
    PATCH_COUNT=$((PATCH_COUNT + 1))
  done < <(rt_series)
  [ "$PATCH_COUNT" -gt 0 ] || rt_die "series of $1 is empty"
  rt_log "series ok: $PATCH_COUNT entries applied to $2"
}

# first 12 hex of the runtime/ tree hash: TREE12 from the workflow, else from HEAD
rt_tree12() {
  local t=${TREE12:-}
  [ -n "$t" ] || t=$(git -C "$RT" rev-parse "HEAD:./" 2> /dev/null | cut -c1-12)
  [[ "$t" =~ ^[0-9a-f]{12}$ ]] || rt_die "TREE12 must be 12 hex characters (got '$t')"
  echo "$t"
}

# 13.0.2 from CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
rt_cuda_version() { local v=${CUDA_IMAGE#*:}; echo "${v%%-*}"; }

# the toolkit must report the image's CUDA release (13.0): a newer minor loaded CPU-only on the Colab image
rt_check_nvcc() {
  local want got
  want="release $(rt_cuda_version | cut -d. -f1,2)"
  got=$(nvcc --version 2> /dev/null | grep -o 'release [0-9][0-9]*\.[0-9]*' | head -1) || true
  [ "$got" = "$want" ] || rt_die "nvcc reports '${got:-nothing}', want '$want'"
}

# fresh work directories only: a stale build tree would hide a configure change
rt_fresh() {
  local d
  for d in "$@"; do
    [ -z "$(ls -A "$d" 2> /dev/null)" ] || rt_die "$d exists and is not empty, use a fresh WORK"
  done
}

# BUILDINFO.json without timestamps, so the same inputs give the same bytes; reads FAMILY TAG COMMIT TREE12 CMAKE_LINE TARGETS ARCHS from the environment
rt_buildinfo() {
  local out=$1 lines=() p
  while read -r p; do lines+=("$p:$(sha256sum < "$SERIES_DIR/$p" | cut -c1-64)"); done < <(rt_series)
  PATCHES="${lines[*]}" SERIES_SHA256="$(sha256sum < "$SERIES_DIR/series" | cut -c1-64)" CUDA_VERSION="$(rt_cuda_version)" \
    MAX_GLIBC="${MAX_GLIBC:-2.39}" LLAMA_URL="$LLAMA_URL" CUDA_IMAGE="$CUDA_IMAGE" CUDA_IMAGE_DIGEST="$CUDA_IMAGE_DIGEST" CCCL="$CCCL" \
    FAMILY="$FAMILY" TAG="$TAG" COMMIT="$COMMIT" TREE12="$TREE12" CMAKE_LINE="$CMAKE_LINE" TARGETS="$TARGETS" ARCHS="$ARCHS" \
    python3 -I - "$out" << 'PY'
import json, os, sys
e = os.environ
info = {
    "schema": 1,
    "family": e["FAMILY"],
    "tree12": e["TREE12"],
    "llama_cpp": {"url": e["LLAMA_URL"], "tag": e["TAG"], "commit": e["COMMIT"]},
    "series_sha256": e["SERIES_SHA256"],
    "patches": [{"file": p.split(":")[0], "sha256": p.split(":")[1]} for p in e["PATCHES"].split()],
    "cuda": {"version": e["CUDA_VERSION"], "image": e["CUDA_IMAGE"], "digest": e["CUDA_IMAGE_DIGEST"], "cccl": e["CCCL"]},
    "archs": e["ARCHS"].split(),
    "max_glibc": e["MAX_GLIBC"],
    "targets": e["TARGETS"].split(),
    "cmake": e["CMAKE_LINE"],
}
with open(sys.argv[1], "w") as f:
    json.dump(info, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

# deterministic gzip tar of the entries (relative to DIR) into OUT, mtime SOURCE_DATE_EPOCH or 0; writes OUT.sha256
rt_tar() {
  local dir=$1 out=$2 epoch=${SOURCE_DATE_EPOCH:-0}
  shift 2
  [[ "$epoch" =~ ^[0-9]+$ ]] || rt_die "SOURCE_DATE_EPOCH must be a number (got '$epoch')"
  tar -C "$dir" --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner --exclude=__pycache__ -cf - -- "$@" | gzip -9n > "$out" || rt_die "tar of $out"
  (cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
}

# every REQUIRED path must exist in the staged tree DIR
rt_require() {
  local dir=$1 f
  shift
  for f in "$@"; do [ -e "$dir/$f" ] || rt_die "staged tree lacks $f"; done
}

# tarball file name; needs TREE12 and the family variables
rt_pkg_name() { echo "llama-$FAMILY-$NAME_TAG-cu$(rt_cuda_version)-$ARCH_LABEL-$TREE12.tar.gz"; }

rt_need_tools() {
  local t
  for t in git python3 cmake ninja ccache gcc nvcc tar gzip; do command -v "$t" > /dev/null || rt_die "$t is not installed in the build image"; done
}

# family scripts define FAMILY NAME_TAG ARCH_LABEL ARCHS TARGETS CMAKE_LINE TOP REQUIRED and rt_stage_family, then call: rt_main "$@"
# modes: (none) build and pack into $WORK/out, --print-cmake, --list-package, --package-only (packs an existing $WORK/src and $WORK/build, no ELF checks)
rt_main() {
  local mode=${1:-build} stage out
  umask 022
  rt_family "$FAMILY"
  case "$mode" in
    --print-cmake) printf '%s\n' "$CMAKE_LINE" ;;
    --list-package)
      TREE12=$(rt_tree12) || exit 1
      rt_pkg_name
      printf '%s\n' "${REQUIRED[@]}" ;;
    build | --package-only)
      : "${WORK:?set WORK to a writable absolute build directory}"
      [[ "$WORK" = /* ]] || rt_die "WORK must be an absolute path"
      TREE12=$(rt_tree12) || exit 1
      stage=$WORK/stage; out=$WORK/out
      export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-2G}
      if [ "$mode" = build ]; then
        rt_need_tools
        rt_check_nvcc
        rt_fresh "$WORK/src" "$WORK/build" "$stage" "$out"
        mkdir -p "$WORK"
        rt_clone "$FAMILY" "$WORK/src"
        rt_apply "$FAMILY" "$WORK/src"
        (cd "$WORK" && eval "$CMAKE_LINE") || rt_die "cmake configure"
        # shellcheck disable=SC2086
        cmake --build "$WORK/build" -j "${JOBS:-4}" --target $TARGETS || rt_die "build"
      else
        rt_fresh "$stage" "$out"
      fi
      mkdir -p "$stage" "$out"
      rt_stage_family
      rt_buildinfo "$stage/BUILDINFO.json"
      rt_require "$stage" "${REQUIRED[@]}"
      if [ "$mode" = build ]; then
        # shellcheck disable=SC2086
        bash "$RT/build/check_elf.sh" "$stage" $ARCHS || rt_die "check_elf.sh"
      fi
      rt_tar "$stage" "$out/$(rt_pkg_name)" "${TOP[@]}"
      ls -l "$out" >&2
      rt_log "packed $out/$(rt_pkg_name)" ;;
    *) echo "usage: $(basename "$0") [--print-cmake | --list-package | --package-only]; a build needs WORK and optionally TREE12, JOBS" >&2; exit 2 ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  case "${1:-}" in
    --apply-only)
      [ $# = 3 ] || { echo "usage: common.sh --apply-only qwen|deepseek DIR" >&2; exit 2; }
      rt_clone "$2" "$3"
      rt_apply "$2" "$3" ;;
    *) echo "usage: common.sh --apply-only qwen|deepseek DIR" >&2; exit 2 ;;
  esac
fi
