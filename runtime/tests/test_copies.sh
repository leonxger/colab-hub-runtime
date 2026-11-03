#!/bin/bash
# The copied patches and pin.c equal the manifest golden/patches-sha256.txt and, when the lab repo is present, their sources byte for byte; every series entry exists and every patch file is listed.
. "$(dirname "$0")/lib.sh"
LAB=${LAB_REPO:-$HOME/colab-hub}

if (cd "$RT" && sha256sum --quiet -c "$GOLDEN/patches-sha256.txt" > "$SCRATCH/sum.out" 2>&1); then
  ok "all $(wc -l < "$GOLDEN/patches-sha256.txt") copied files match golden/patches-sha256.txt"
else
  bad "a copied file differs from golden/patches-sha256.txt: $(head -3 "$SCRATCH/sum.out" | tr '\n' ' ')"
fi

listed=$(awk '{print $2}' "$GOLDEN/patches-sha256.txt" | sort)
present=$(cd "$RT" && find patches src -type f | sort)
[ "$listed" = "$present" ] && ok "the manifest lists exactly the files under patches/ and src/" || bad "manifest and files differ: $(diff <(echo "$listed") <(echo "$present") | head -4 | tr '\n' ' ')"

for fam in qwen deepseek; do
  d=$RT/patches/$fam
  missing=""; unlisted=""
  entries=$(grep -v '^[[:space:]]*\(#\|$\)' "$d/series")
  for p in $entries; do [ -f "$d/$p" ] || missing="$missing $p"; done
  for f in "$d"/*; do
    b=$(basename "$f")
    case "$b" in series | LICENSE.llama.cpp) continue ;; esac
    grep -qx "$b" <<< "$entries" && continue
    # a diff that a listed script applies itself (pr30966.diff) counts as used
    for p in $entries; do [[ "$p" = *.py ]] && grep -q "$b" "$d/$p" && continue 2; done
    unlisted="$unlisted $b"
  done
  [ -z "$missing" ] && ok "$fam: every series entry exists" || bad "$fam: series entries without a file:$missing"
  [ -z "$unlisted" ] && ok "$fam: every patch file is in the series" || bad "$fam: files not in the series:$unlisted"
  [ -f "$d/LICENSE.llama.cpp" ] && ok "$fam: LICENSE.llama.cpp present" || bad "$fam: LICENSE.llama.cpp missing"
done
[ "$(grep -vc '^[[:space:]]*\(#\|$\)' "$RT/patches/deepseek/series")" = 22 ] && ok "deepseek series has 22 entries" || bad "deepseek series is not 22 entries"
[ "$(grep -vc '^[[:space:]]*\(#\|$\)' "$RT/patches/qwen/series")" = 2 ] && ok "qwen series has 2 entries" || bad "qwen series is not 2 entries"
! grep -rlq $'\r' "$RT/patches" "$RT/src" && ok "no CRLF in patches or pin.c" || bad "CRLF found in patches or pin.c"

if [ ! -d "$LAB/vm/qwen27/patches" ] || [ ! -f "$LAB/vm/deepseek/pin.c" ]; then
  skip "lab repo not found at $LAB (set LAB_REPO), copies not compared with their sources"
else
  pairs=("$LAB/vm/deepseek/pin.c:$RT/src/pin.c")
  for f in series pr31037-mtp-head-trim.diff dflash2-head-trim.diff LICENSE.llama.cpp; do pairs+=("$LAB/vm/qwen27/patches/$f:$RT/patches/qwen/$f"); done
  for f in "$LAB"/vm/deepseek/patches/*; do pairs+=("$f:$RT/patches/deepseek/$(basename "$f")"); done
  same=0
  for pr in "${pairs[@]}"; do
    src=${pr%%:*}; dst=${pr#*:}
    if cmp -s "$src" "$dst" && [ "$(sha256sum < "$src")" = "$(sha256sum < "$dst")" ]; then same=$((same + 1)); else bad "$dst differs from $src"; fi
  done
  [ "$same" = "${#pairs[@]}" ] && ok "sha256 of all $same copied files equals the lab source"
  extra=$(comm -13 <(cd "$LAB/vm/deepseek/patches" && ls | sort) <(cd "$RT/patches/deepseek" && ls | sort))
  [ -z "$extra" ] && ok "no file in patches/deepseek that the lab lacks" || bad "extra files in patches/deepseek: $extra"
fi
finish test_copies.sh
