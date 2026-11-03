#!/bin/bash
# check_elf.sh with a stub cuobjdump and ldd and a real readelf on host ELF files: each rule passes on a good tree and fails when broken.
. "$(dirname "$0")/lib.sh"
CHECK="$BUILD/check_elf.sh"
command -v readelf > /dev/null || { skip "readelf is not installed, check_elf.sh not tested"; exit 0; }
host_elf=$(readlink -f /bin/true)
readelf -h "$host_elf" > /dev/null 2>&1 || { skip "$host_elf is not an ELF file readelf can read"; exit 0; }

stub="$SCRATCH/stub"; mkdir -p "$stub"
cat > "$stub/cuobjdump" << 'EOF'
#!/bin/bash
cat "$STUB_CUBINS"
EOF
cat > "$stub/ldd" << 'EOF'
#!/bin/bash
cat "$STUB_LDD"
EOF
chmod +x "$stub/cuobjdump" "$stub/ldd"

tree="$SCRATCH/tree"
mkdir -p "$tree/bin" "$tree/lib" "$tree/py"
cp "$host_elf" "$tree/bin/libggml-cuda.so"; cp "$host_elf" "$tree/bin/llama-server"; cp "$host_elf" "$tree/lib/libpin.so"
echo 'print(1)' > "$tree/py/convert_hf_to_gguf.py"
cuda=$SCRATCH/cuda/lib64
cat > "$SCRATCH/cubins-good" << 'EOF'
ELF file    1: libggml-cuda.1.sm_80.cubin
ELF file    2: libggml-cuda.1.sm_120a.cubin
ELF file    3: libggml-cuda.2.sm_80.cubin
ELF file    4: libggml-cuda.2.sm_120a.cubin
EOF
cat > "$SCRATCH/ldd-good" << EOF
	linux-vdso.so.1 (0x00007ffc)
	libcudart.so.13 => $cuda/libcudart.so.13 (0x00007f01)
	libcublas.so.13 => $cuda/libcublas.so.13 (0x00007f02)
	libcublasLt.so.13 => $cuda/libcublasLt.so.13 (0x00007f03)
	libcuda.so.1 => not found
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f04)
EOF
run() { # run CUBINS LDD CMD...: the command with the stubs in place
  local cub=$1 ld=$2; shift 2
  env PATH="$stub:$PATH" CUOBJDUMP="$stub/cuobjdump" LDD="$stub/ldd" STUB_CUBINS="$cub" STUB_LDD="$ld" CUDA_LIB="$cuda" MAX_GLIBC=99 "$@" > "$SCRATCH/out" 2>&1
}
good() { run "$SCRATCH/cubins-good" "$SCRATCH/ldd-good" bash "$CHECK" "$tree" "$@"; }

if good sm_80 sm_120a; then ok "a good tree passes (cubins sm_80 and sm_120a, libcuda.so.1 may be missing)"; else bad "good tree failed: $(grep FAIL "$SCRATCH/out" | tr '\n' ' ')"; fi
grep -q 'cubins of libggml-cuda.so: sm_120a sm_80' "$SCRATCH/out" && ok "the cubin set is reported" || bad "cubin set line missing"

if good sm_80; then bad "an extra cubin architecture accepted"; else
  grep -q "cubins of libggml-cuda.so are 'sm_120a sm_80', expected 'sm_80'" "$SCRATCH/out" && ok "an extra cubin architecture fails" || bad "extra cubin message wrong"
fi
if good sm_80 sm_90 sm_120a; then bad "an absent cubin architecture accepted"; else ok "a missing cubin architecture fails"; fi
printf 'ELF file    1: libggml-cuda.1.sm_80.cubin\n' > "$SCRATCH/cubins-one"
if run "$SCRATCH/cubins-one" "$SCRATCH/ldd-good" bash "$CHECK" "$tree" sm_80 sm_120a; then bad "sm_120a missing from the library accepted"; else ok "sm_120a missing from the library fails"; fi
printf 'ELF file    1: libggml-cuda.1.ssm_conv.sm_80.cubin\nELF file    2: libggml-cuda.1.sm_120a.cubin\n' > "$SCRATCH/cubins-names"
if run "$SCRATCH/cubins-names" "$SCRATCH/ldd-good" bash "$CHECK" "$tree" sm_80 sm_120a; then ok "kernel names with sm_ inside do not confuse the cubin parse"; else bad "name with sm_ confused the parser"; fi
: > "$SCRATCH/cubins-none"
if run "$SCRATCH/cubins-none" "$SCRATCH/ldd-good" bash "$CHECK" "$tree" sm_80; then bad "empty cuobjdump output accepted"; else ok "empty cuobjdump output fails"; fi

if run "$SCRATCH/cubins-good" "$SCRATCH/ldd-good" env MAX_GLIBC=2.10 bash "$CHECK" "$tree" sm_80 sm_120a; then bad "GLIBC above the limit accepted"; else
  grep -q 'bin/llama-server needs GLIBC_.*above 2.10' "$SCRATCH/out" && ok "a GLIBC symbol above the limit fails and names the file" || bad "GLIBC failure message wrong"
fi

sed 's/libcudart.so.13/libcudart.so.12/' "$SCRATCH/ldd-good" > "$SCRATCH/ldd-12"
if run "$SCRATCH/cubins-good" "$SCRATCH/ldd-12" bash "$CHECK" "$tree" sm_80 sm_120a; then bad "libcudart.so.12 accepted"; else
  grep -q 'links libcudart.so.12' "$SCRATCH/out" && ok "libcudart.so.12 fails" || bad "libcudart.so.12 message missing"
fi

printf '\tlibcudart.so.13 => /usr/lib/x86_64-linux-gnu/libcudart.so.13 (0x1)\n' > "$SCRATCH/ldd-outside"
if run "$SCRATCH/cubins-good" "$SCRATCH/ldd-outside" bash "$CHECK" "$tree" sm_80 sm_120a; then bad "CUDA library outside CUDA_LIB accepted"; else
  grep -q 'not under' "$SCRATCH/out" && ok "a CUDA library resolved outside CUDA_LIB fails" || bad "outside-path message missing"
fi

{ cat "$SCRATCH/ldd-good"; printf '\tlibssl.so.3 => not found\n'; } > "$SCRATCH/ldd-miss"
if run "$SCRATCH/cubins-good" "$SCRATCH/ldd-miss" bash "$CHECK" "$tree" sm_80 sm_120a; then bad "an unresolved library accepted"; else
  grep -q 'unresolved libssl.so.3' "$SCRATCH/out" && ok "an unresolved library other than libcuda.so.1 fails" || bad "unresolved message missing"
fi

printf '\tlibc.so.6 => /lib/libc.so.6 (0x1)\n' > "$SCRATCH/ldd-nocuda"
if run "$SCRATCH/cubins-good" "$SCRATCH/ldd-nocuda" bash "$CHECK" "$tree" sm_80 sm_120a; then bad "a tree that links no libcudart accepted"; else
  grep -q 'no ELF links libcudart.so.13' "$SCRATCH/out" && ok "a tree without libcudart.so.13 fails" || bad "no-cudart message missing"
fi

rm -f "$tree/bin/libggml-cuda.so"
if good sm_80 sm_120a; then bad "missing libggml-cuda.so accepted"; else ok "a missing libggml-cuda.so fails"; fi
mkdir -p "$SCRATCH/py-only/py"
if run "$SCRATCH/cubins-good" "$SCRATCH/ldd-good" bash "$CHECK" "$SCRATCH/py-only" sm_80; then bad "a tree without ELF files accepted"; else ok "a tree without ELF files fails"; fi
bash "$CHECK" > /dev/null 2>&1; [ $? = 2 ] && ok "usage error exits 2" || bad "usage error exit code is not 2"
finish test_check_elf.sh
