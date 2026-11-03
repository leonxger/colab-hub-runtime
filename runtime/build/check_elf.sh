#!/bin/bash
# Checks an unpacked runtime tree: check_elf.sh DIR ARCH... (for example sm_80 sm_120a). libggml-cuda.so holds exactly these cubins, no ELF needs a GLIBC symbol above MAX_GLIBC, every ELF resolves its CUDA libraries in CUDA_LIB with no libcudart.so.12.
set -uo pipefail
dir=${1:-}
[ -d "$dir" ] && [ $# -ge 2 ] || { echo "usage: check_elf.sh DIR ARCH..." >&2; exit 2; }
shift
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
READELF=${READELF:-readelf}
LDD=${LDD:-ldd}
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
MAX_GLIBC=${MAX_GLIBC:-2.39}
fails=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fails=$((fails + 1)); }

elfs=()
while IFS= read -r f; do "$READELF" -h "$f" > /dev/null 2>&1 && elfs+=("$f"); done < <(find "$dir/bin" "$dir/lib" -type f 2> /dev/null | sort)
if [ "${#elfs[@]}" -gt 0 ]; then ok "${#elfs[@]} ELF files in bin and lib"; else bad "no ELF file in $dir/bin or $dir/lib"; fi

# cubins: the set of sm_ architectures in libggml-cuda.so equals the expected set
lib=$(find "$dir/bin" -name libggml-cuda.so 2> /dev/null | head -1)
if [ -z "$lib" ]; then
  bad "bin/libggml-cuda.so is missing"
else
  want=$(printf '%s\n' "$@" | sort -u | paste -sd' ')
  got=$("$CUOBJDUMP" --list-elf "$lib" 2> /dev/null | grep '^ELF file' | grep -oE '\.sm_[0-9]+[a-z]?\b' | sed 's/^\.//' | sort -u | paste -sd' ')
  if [ "$got" = "$want" ]; then ok "cubins of libggml-cuda.so: $got"; else bad "cubins of libggml-cuda.so are '${got:-none}', expected '$want'"; fi
fi

# the highest GLIBC symbol version of every ELF in bin and lib
top=""
glibc_bad=0
for f in "${elfs[@]}"; do
  v=$("$READELF" -V "$f" 2> /dev/null | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -V | tail -1)
  [ -n "$v" ] || continue
  [ "$(printf '%s\n%s\n' "$v" "${top:-0}" | sort -V | tail -1)" = "$v" ] && top=$v
  if [ "$(printf '%s\n%s\n' "$v" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
    bad "${f#"$dir"/} needs GLIBC_$v, above $MAX_GLIBC"
    glibc_bad=1
  fi
done
[ "$glibc_bad" = 0 ] && ok "highest GLIBC symbol version ${top:-none}, limit $MAX_GLIBC"

# libraries: nothing unresolved except the driver's libcuda.so.1, CUDA libraries only from CUDA_LIB, none of CUDA 12
seen13=0
for f in "${elfs[@]}"; do
  out=$(LD_LIBRARY_PATH="$dir/bin:$dir/lib:$CUDA_LIB" "$LDD" "$f" 2>&1) || true
  rel=${f#"$dir"/}
  while IFS= read -r l; do bad "$rel: unresolved $l"; done < <(grep 'not found' <<< "$out" | grep -v 'libcuda\.so\.1' | sed 's/^[[:space:]]*//')
  grep -q 'libcudart\.so\.12' <<< "$out" && bad "$rel links libcudart.so.12"
  grep -q 'libcudart\.so\.13' <<< "$out" && seen13=1
  while read -r name path; do
    case "$path" in "$CUDA_LIB"/*) ;; *) bad "$rel: $name resolves to $path, not under $CUDA_LIB" ;; esac
  done < <(sed -n 's/^[[:space:]]*\(libcudart\.so\.[0-9]*\|libcublas\.so\.[0-9]*\|libcublasLt\.so\.[0-9]*\) => \([^ ]*\).*/\1 \2/p' <<< "$out" | grep -v ' not$')
done
[ "$seen13" = 1 ] && ok "libcudart.so.13 is linked, libraries resolve under $CUDA_LIB" || bad "no ELF links libcudart.so.13"

if [ "$fails" = 0 ]; then echo "PASS check_elf.sh $dir"; else echo "FAIL check_elf.sh: $fails problems"; exit 1; fi
