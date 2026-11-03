#!/bin/bash
# Packs libcudart, libcublas and libcublasLt (major 13) of the build image into the optional fallback asset cudart-<CUDA>-x64-<tree12>.tar.gz: cudart.sh needs WORK; CUDA_LIB defaults to /usr/local/cuda/lib64.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime/build/common.sh
. "$HERE/common.sh"

umask 022
: "${WORK:?set WORK to a writable absolute build directory}"
[[ "$WORK" = /* ]] || rt_die "WORK must be an absolute path"
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
TREE12=$(rt_tree12) || exit 1
stage=$WORK/stage-cudart
out=$WORK/out
rt_fresh "$stage" "$out"
mkdir -p "$stage/lib" "$out"
for l in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
  ls "$CUDA_LIB/$l"* > /dev/null 2>&1 || rt_die "$CUDA_LIB has no $l*"
  cp -a "$CUDA_LIB/$l"* "$stage/lib/"
done
eula=$(dirname "$CUDA_LIB")/EULA.txt
if [ -f "$eula" ]; then cp "$eula" "$stage/LICENSE"; else rt_log "no $eula, the asset carries no LICENSE"; fi
CUDA_VERSION=$(rt_cuda_version) CUDA_IMAGE_DIGEST=$CUDA_IMAGE_DIGEST CUDA_IMAGE=$CUDA_IMAGE TREE12=$TREE12 python3 -I - "$stage" << 'PY'
import hashlib, json, os, sys
root, e = sys.argv[1], os.environ
files = {}
for name in sorted(os.listdir(os.path.join(root, "lib"))):
    p = os.path.join(root, "lib", name)
    files["lib/" + name] = os.readlink(p) if os.path.islink(p) else hashlib.sha256(open(p, "rb").read()).hexdigest()
info = {"schema": 1, "family": "cudart", "tree12": e["TREE12"], "cuda": {"version": e["CUDA_VERSION"], "image": e["CUDA_IMAGE"], "digest": e["CUDA_IMAGE_DIGEST"]}, "files": files}
with open(os.path.join(root, "BUILDINFO.json"), "w") as f:
    json.dump(info, f, indent=2, sort_keys=True)
    f.write("\n")
PY
top=(lib BUILDINFO.json)
[ ! -f "$stage/LICENSE" ] || top+=(LICENSE)
name=cudart-$(rt_cuda_version)-x64-$TREE12.tar.gz
rt_tar "$stage" "$out/$name" "${top[@]}"
ls -l "$out" >&2
rt_log "packed $out/$name"
