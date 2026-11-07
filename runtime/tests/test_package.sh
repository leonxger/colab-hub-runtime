#!/bin/bash
# Packaging of both families from fake build trees (--package-only): file lists (bin/test-backend-ops for P18), names of the interface 7, BUILDINFO.json, symlinks, sha256 file, byte-identical repeat, failure on a missing binary; --list-package against the goldens.
. "$(dirname "$0")/lib.sh"
export TREE12=0123456789ab SOURCE_DATE_EPOCH=1700000000

mkbuild() { # mkbuild WORK FAMILY: a fake build/bin and src
  local w=$1 fam=$2 b=$1/build/bin s=$1/src
  mkdir -p "$b" "$s/conversion" "$s/gguf-py/gguf" "$s/gguf-py/tests"
  for f in llama-server llama-fit-params; do printf '#!/bin/sh\necho %s\n' "$f" > "$b/$f"; chmod +x "$b/$f"; done
  [ "$fam" = qwen ] && { printf '#!/bin/sh\necho t\n' > "$b/test-backend-ops"; chmod +x "$b/test-backend-ops"; }
  echo cuda > "$b/libggml-cuda.so"; echo base > "$b/libggml-base.so.0.0.1"; ln -s libggml-base.so.0.0.1 "$b/libggml-base.so.0"; ln -s libggml-base.so.0 "$b/libggml-base.so"
  echo cpu > "$b/libggml-cpu-haswell.so"
  echo MIT > "$s/LICENSE"; echo 'print(1)' > "$s/convert_hf_to_gguf.py"; echo '' > "$s/conversion/__init__.py"; echo 'x=1' > "$s/conversion/qwen.py"
  echo '' > "$s/gguf-py/gguf/__init__.py"; echo '[project]' > "$s/gguf-py/pyproject.toml"; echo MIT > "$s/gguf-py/LICENSE"; echo t > "$s/gguf-py/tests/test_x.py"
  mkdir -p "$s/conversion/__pycache__"; echo junk > "$s/conversion/__pycache__/x.pyc"
}
pack() { # pack WORK FAMILY
  env WORK="$1" bash "$BUILD/$2.sh" --package-only > "$SCRATCH/pack.out" 2>&1
}

for fam in qwen deepseek; do
  if [ "$fam" = deepseek ] && ! command -v gcc > /dev/null; then skip "gcc not installed, deepseek packaging (libpin.so) not tested"; continue; fi
  w="$SCRATCH/w-$fam"; mkbuild "$w" "$fam"
  name=$(TREE12=$TREE12 bash "$BUILD/$fam.sh" --list-package | head -1)
  if pack "$w" "$fam"; then
    ok "$fam: --package-only packs a tarball"
    tarball="$w/out/$name"
    [ -f "$tarball" ] && ok "$fam: file name is $name" || bad "$fam: expected $tarball, found $(ls "$w/out" | tr '\n' ' ')"
    list=$(tar tzf "$tarball")
    while read -r req; do grep -qx "$req" <<< "$list" && ok "$fam: tarball has $req" || bad "$fam: tarball lacks $req"; done < <(TREE12=$TREE12 bash "$BUILD/$fam.sh" --list-package | tail -n +2)
    grep -qx 'bin/libggml-cpu-haswell.so' <<< "$list" && ok "$fam: every bin/*.so is in the tarball" || bad "$fam: bin/libggml-cpu-haswell.so missing"
    tar tvzf "$tarball" | grep -q 'libggml-base.so.0 -> libggml-base.so.0.0.1' && ok "$fam: symlinks stay symlinks" || bad "$fam: library symlinks not preserved"
    ! grep -q '__pycache__\|tests/' <<< "$list" && ok "$fam: no __pycache__ and no gguf-py tests" || bad "$fam: unwanted files in the tarball"
    [ -z "$(tar tzf "$tarball" | grep '^/\|^\./')" ] && ok "$fam: paths are relative without ./" || bad "$fam: absolute or ./ paths in the tarball"
    (cd "$w/out" && sha256sum -c --quiet "$name.sha256") && ok "$fam: .sha256 file verifies" || bad "$fam: .sha256 does not verify"
    tar xzf "$tarball" -O BUILDINFO.json > "$SCRATCH/bi-$fam.json"
    python3 -I - "$SCRATCH/bi-$fam.json" "$RT" "$fam" "$GOLDEN/$fam-cmake.txt" << 'EOF' && ok "$fam: BUILDINFO.json holds tag, commit, patch hashes, CUDA pins, cmake line" || bad "$fam: BUILDINFO.json wrong: $(python3 -I -c 'import sys; print(open(sys.argv[1]).read()[:200])' "$SCRATCH/bi-$fam.json" | tr '\n' ' ')"
import hashlib, json, os, sys
info = json.load(open(sys.argv[1]))
rt, fam, golden = sys.argv[2], sys.argv[3], sys.argv[4]
env = dict(l.strip().split("=", 1) for l in open(os.path.join(rt, "versions.env")) if "=" in l and not l.startswith("#"))
tag, commit = (env["QWEN_TAG"], env["QWEN_COMMIT"]) if fam == "qwen" else (env["DS_TAG"], env["DS_COMMIT"])
series = [l.strip() for l in open(os.path.join(rt, "patches", fam, "series")) if l.strip() and not l.startswith("#")]
assert info["family"] == fam and info["tree12"] == "0123456789ab"
assert info["llama_cpp"]["tag"] == tag and info["llama_cpp"]["commit"] == commit
assert [p["file"] for p in info["patches"]] == series
for p in info["patches"]:
    assert p["sha256"] == hashlib.sha256(open(os.path.join(rt, "patches", fam, p["file"]), "rb").read()).hexdigest(), p["file"]
assert info["series_sha256"] == hashlib.sha256(open(os.path.join(rt, "patches", fam, "series"), "rb").read()).hexdigest()
assert info["cuda"] == {"version": "13.0.2", "image": env["CUDA_IMAGE"], "digest": env["CUDA_IMAGE_DIGEST"], "cccl": env["CCCL"]}
assert info["cmake"] == open(golden).read().strip()
assert info["archs"] == (["sm_80", "sm_120a"] if fam == "qwen" else ["sm_120a"]) and info["max_glibc"] == "2.39"
assert ("test-backend-ops" in info["targets"]) == (fam == "qwen")
EOF
    w2="$SCRATCH/w2-$fam"; mkbuild "$w2" "$fam"
    if pack "$w2" "$fam"; then
      [ "$(sha256sum < "$tarball")" = "$(sha256sum < "$w2/out/$name")" ] && ok "$fam: a repeat from the same inputs is byte-identical" || bad "$fam: repeat differs"
    else bad "$fam: repeat pack failed"; fi
  else
    bad "$fam: --package-only failed: $(tail -3 "$SCRATCH/pack.out" | tr '\n' ' ')"
  fi
  w3="$SCRATCH/w3-$fam"; mkbuild "$w3" "$fam"; rm -f "$w3/build/bin/libggml-cuda.so"
  if pack "$w3" "$fam"; then bad "$fam: a build without libggml-cuda.so was packed"; else ok "$fam: a missing libggml-cuda.so stops the packaging"; fi
  [ -z "$(ls "$w3/out" 2> /dev/null)" ] && ok "$fam: no tarball is left after the failure" || bad "$fam: tarball left after a failed packaging"
done

w4="$SCRATCH/w4"; mkbuild "$w4" qwen; rm -f "$w4/build/bin/test-backend-ops"
if pack "$w4" qwen; then bad "qwen: a tarball without bin/test-backend-ops was packed (P18)"; else ok "qwen: a build without test-backend-ops is refused (P18)"; fi

for fam in qwen deepseek; do
  [ "$(TREE12=$TREE12 bash "$BUILD/$fam.sh" --list-package)" = "$(cat "$GOLDEN/$fam-package.txt")" ] && ok "$fam: --list-package equals golden/$fam-package.txt" || bad "$fam: --list-package differs from the golden"
done
grep -qx 'bin/test-backend-ops' "$GOLDEN/qwen-package.txt" && ok "qwen golden lists bin/test-backend-ops (P18)" || bad "qwen golden lacks bin/test-backend-ops"
[ "$(head -1 "$GOLDEN/qwen-package.txt")" = "llama-qwen-b12015-cu13.0.2-sm80-sm120a-0123456789ab.tar.gz" ] && ok "qwen tarball name follows interface 7" || bad "qwen tarball name differs from interface 7"
[ "$(head -1 "$GOLDEN/deepseek-package.txt")" = "llama-deepseek-b11963p24-cu13.0.2-sm120a-0123456789ab.tar.gz" ] && ok "deepseek tarball name follows interface 7" || bad "deepseek tarball name differs from interface 7"
finish test_package.sh
