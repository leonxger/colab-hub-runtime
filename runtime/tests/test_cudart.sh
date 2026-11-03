#!/bin/bash
# cudart.sh from a fake CUDA library directory: only the three major-13 library families go in, symlinks stay, name follows interface 7, BUILDINFO.json hashes the files, a repeat is byte-identical, a missing library fails.
. "$(dirname "$0")/lib.sh"
export TREE12=0123456789ab SOURCE_DATE_EPOCH=1700000000

mkcuda() { # mkcuda ROOT: ROOT/lib64 with libraries and ROOT/EULA.txt
  local l=$1/lib64
  mkdir -p "$l"
  for n in libcudart.so.13.0.96 libcublas.so.13.0.2.14 libcublasLt.so.13.0.2.14 libcudart.so.12.8.90 libnvrtc.so.13.0.88; do echo "$n" > "$l/$n"; done
  ln -s libcudart.so.13.0.96 "$l/libcudart.so.13"; ln -s libcudart.so.13 "$l/libcudart.so"
  ln -s libcublas.so.13.0.2.14 "$l/libcublas.so.13"; ln -s libcublasLt.so.13.0.2.14 "$l/libcublasLt.so.13"
  ln -s libcudart.so.12.8.90 "$l/libcudart.so.12"
  echo "EULA text" > "$1/EULA.txt"
}
pack() { env WORK="$1" CUDA_LIB="$2/lib64" bash "$BUILD/cudart.sh" > "$SCRATCH/pack.out" 2>&1; }

c1="$SCRATCH/cuda1"; mkcuda "$c1"
name=cudart-13.0.2-x64-0123456789ab.tar.gz
if pack "$SCRATCH/w1" "$c1"; then
  ok "cudart.sh packs a tarball"
  tarball="$SCRATCH/w1/out/$name"
  [ -f "$tarball" ] && ok "file name is $name" || bad "expected $tarball, found $(ls "$SCRATCH/w1/out" | tr '\n' ' ')"
  list=$(tar tzf "$tarball")
  for f in lib/libcudart.so.13 lib/libcudart.so.13.0.96 lib/libcublas.so.13 lib/libcublas.so.13.0.2.14 lib/libcublasLt.so.13 lib/libcublasLt.so.13.0.2.14 BUILDINFO.json LICENSE; do
    grep -qx "$f" <<< "$list" && ok "tarball has $f" || bad "tarball lacks $f"
  done
  ! grep -q 'libcudart.so.12\|libnvrtc\|libcudart.so$' <<< "$list" && ok "no CUDA 12 library, no other library, no dev symlink" || bad "unwanted library in the tarball: $(grep 'so.12\|nvrtc' <<< "$list" | tr '\n' ' ')"
  tar tvzf "$tarball" | grep -q 'libcudart.so.13 -> libcudart.so.13.0.96' && ok "symlinks stay symlinks" || bad "symlink lost"
  (cd "$SCRATCH/w1/out" && sha256sum -c --quiet "$name.sha256") && ok ".sha256 file verifies" || bad ".sha256 does not verify"
  python3 -I - "$(tar xzf "$tarball" -O BUILDINFO.json | python3 -I -c 'import sys; print(sys.stdin.read().replace(chr(10), " "))')" << 'EOF' && ok "BUILDINFO.json names the image digest and the library hashes" || bad "BUILDINFO.json wrong"
import hashlib, json, sys
info = json.loads(sys.argv[1])
assert info["family"] == "cudart" and info["tree12"] == "0123456789ab" and info["cuda"]["version"] == "13.0.2"
assert info["cuda"]["digest"].startswith("sha256:") and info["cuda"]["image"].startswith("nvidia/cuda:13.0.2")
assert info["files"]["lib/libcudart.so.13.0.96"] == hashlib.sha256(b"libcudart.so.13.0.96\n").hexdigest()
assert info["files"]["lib/libcudart.so.13"] == "libcudart.so.13.0.96"
EOF
  c2="$SCRATCH/cuda2"; mkcuda "$c2"
  if pack "$SCRATCH/w2" "$c2"; then [ "$(sha256sum < "$tarball")" = "$(sha256sum < "$SCRATCH/w2/out/$name")" ] && ok "a repeat from the same inputs is byte-identical" || bad "repeat differs"; else bad "repeat pack failed"; fi
else
  bad "cudart.sh failed: $(tail -3 "$SCRATCH/pack.out" | tr '\n' ' ')"
fi

c3="$SCRATCH/cuda3"; mkcuda "$c3"; rm -f "$c3"/lib64/libcublasLt.so.13*
if pack "$SCRATCH/w3" "$c3"; then bad "a missing libcublasLt was packed"; else grep -q 'no libcublasLt.so.13' "$SCRATCH/pack.out" && ok "a missing library fails and is named" || bad "missing library not named"; fi
c4="$SCRATCH/cuda4"; mkcuda "$c4"; rm -f "$c4/EULA.txt"
if pack "$SCRATCH/w4" "$c4" && ! tar tzf "$SCRATCH/w4/out/$name" | grep -qx LICENSE; then ok "without an EULA file the asset is packed without LICENSE and the log says so"; else bad "missing EULA case wrong"; fi
grep -q 'carries no LICENSE' "$SCRATCH/pack.out" && ok "the missing EULA is logged" || bad "missing EULA not logged"
finish test_cudart.sh
