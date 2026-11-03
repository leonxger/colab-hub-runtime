#!/bin/bash
# common.sh against a local fake upstream: shallow clone at a tag, commit check, series order, diffs and scripts, stop at the first failure, refusal of a used directory, nvcc and TREE12 guards.
. "$(dirname "$0")/lib.sh"
COMMON="$BUILD/common.sh"
GITQ=(git -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false -c init.defaultBranch=main)

up="$SCRATCH/upstream"
mkdir -p "$up" && "${GITQ[@]}" -C "$up" init -q
printf 'l1\nl2\nl3\nl4\nl5\nl6\nl7\n' > "$up/a.txt"
"${GITQ[@]}" -C "$up" add a.txt && "${GITQ[@]}" -C "$up" commit -q -m first && "${GITQ[@]}" -C "$up" tag vT1
commit=$("${GITQ[@]}" -C "$up" rev-parse HEAD)

# fixture: versions file and patch dirs; entries are written per case
fx="$SCRATCH/fx"
mkdir -p "$fx/patches/qwen" "$fx/patches/deepseek"
cat > "$fx/versions.env" << VEOF
LLAMA_URL=file://$up
QWEN_TAG=vT1
QWEN_COMMIT=$commit
DS_TAG=vT1
DS_COMMIT=$commit
CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
CUDA_IMAGE_DIGEST=sha256:0
CCCL=v3.4.3
VEOF
cat > "$fx/patches/qwen/add-three.diff" << 'DEOF'
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -2,6 +2,7 @@
 l2
 l3
 l4
+three
 l5
 l6
 l7
DEOF
cat > "$fx/patches/qwen/append.py" << 'PEOF'
import sys
open(sys.argv[1] + '/a.txt', 'a').write('four\n')
PEOF
cat > "$fx/patches/qwen/needs-four.py" << 'PEOF'
import sys
p = sys.argv[1] + '/a.txt'
s = open(p).read()
assert s.endswith('four\n'), 'anchor missing'
open(p, 'w').write(s + 'five\n')
PEOF
cat > "$fx/patches/qwen/fail.py" << 'PEOF'
import sys
sys.exit(3)
PEOF
cat > "$fx/patches/qwen/mark.py" << 'PEOF'
import sys
open(sys.argv[1] + '/MARK', 'w').write('x')
PEOF
run() { # run FAMILY DIR [extra env...]: common.sh --apply-only against the fixture
  local fam=$1 dir=$2; shift 2
  env RT_VERSIONS="$fx/versions.env" RT_PATCHES="$fx/patches" "$@" bash "$COMMON" --apply-only "$fam" "$dir" > "$SCRATCH/out" 2>&1
}
series() { printf '%s\n' "$@" > "$fx/patches/qwen/series"; }

series '# comment' '' add-three.diff append.py needs-four.py
if run qwen "$SCRATCH/t1"; then
  ok "series applies: diff, script, dependent script"
  [ "$(cat "$SCRATCH/t1/a.txt")" = $'l1\nl2\nl3\nl4\nthree\nl5\nl6\nl7\nfour\nfive' ] && ok "patches applied in series order" || bad "a.txt after the series is wrong: $(tr '\n' '|' < "$SCRATCH/t1/a.txt")"
  grep -q 'series ok: 3 entries' "$SCRATCH/out" && ok "comments and blank lines are skipped, 3 entries counted" || bad "entry count line missing"
  [ "$(git -C "$SCRATCH/t1" rev-parse HEAD)" = "$commit" ] && ok "clone sits at the pinned commit" || bad "clone commit differs"
  [ "$(git -C "$SCRATCH/t1" rev-list --count HEAD)" = 1 ] && ok "clone is shallow (one commit)" || bad "clone is not shallow"
else
  bad "series did not apply: $(tail -3 "$SCRATCH/out" | tr '\n' ' ')"
fi

series needs-four.py append.py
rm -rf "$SCRATCH/t2"
check_not "wrong order fails" run qwen "$SCRATCH/t2"
grep -q 'FAILED: patch needs-four.py' "$SCRATCH/out" && ok "the failing entry is named" || bad "failure message lacks the entry name: $(tail -2 "$SCRATCH/out" | tr '\n' ' ')"
[ "$(tail -c 5 "$SCRATCH/t2/a.txt" 2> /dev/null)" != "four" ] && ok "the entry after the failure was not run" || bad "append.py ran after a failure"

series fail.py mark.py
check_not "a script exiting 3 fails the run" run qwen "$SCRATCH/t3"
[ ! -e "$SCRATCH/t3/MARK" ] && ok "stops at the first failure, later entries are not run" || bad "mark.py ran after fail.py"

series add-three.diff
sed 's/^QWEN_COMMIT=.*/QWEN_COMMIT=0000000000000000000000000000000000000000/' "$fx/versions.env" > "$fx/versions-bad.env"
check_not "a wrong commit fails" run qwen "$SCRATCH/t4a" RT_VERSIONS="$fx/versions-bad.env"
grep -q 'commit mismatch' "$SCRATCH/out" && ok "a wrong commit is refused before any patch" || bad "no commit mismatch message"
[ "$(cat "$SCRATCH/t4a/a.txt" 2> /dev/null | wc -l)" = 7 ] && ok "nothing was applied after the mismatch" || bad "patches ran despite a commit mismatch"

series ghost.py
check_not "a series entry without a file fails" run qwen "$SCRATCH/t5"
grep -q 'ghost.py' "$SCRATCH/out" && ok "the missing file is named" || bad "missing file not named"

series add-three.diff
mkdir -p "$SCRATCH/t6" && touch "$SCRATCH/t6/x"
check_not "a used directory is refused" run qwen "$SCRATCH/t6"
check_not "unknown family fails" run nobody "$SCRATCH/t7"
check_not "no arguments exits 2" bash "$COMMON"
bash "$COMMON" > /dev/null 2>&1; [ $? = 2 ] && ok "usage error exit code is 2" || bad "usage error exit code is not 2"

echo 'x' > "$fx/patches/qwen/odd.sh"; series odd.sh
check_not "an entry that is neither diff nor py fails" run qwen "$SCRATCH/t8"

# a second apply of the same series: the diff reports already applied, scripts stay idempotent by their own rules
series add-three.diff
( . "$COMMON"; RT_PATCHES="$fx/patches"; rt_apply qwen "$SCRATCH/t1" ) > "$SCRATCH/out" 2>&1
grep -q 'already applied add-three.diff' "$SCRATCH/out" && ok "a diff that is already applied is skipped" || bad "second apply of a diff did not say already applied: $(tail -2 "$SCRATCH/out" | tr '\n' ' ')"

# nvcc guard: the toolkit must report the image's release
stub="$SCRATCH/stub"; mkdir -p "$stub"
nvcc_stub() { printf '#!/bin/bash\necho "Cuda compilation tools, release %s, V%s.48"\n' "$1" "$1" > "$stub/nvcc"; chmod +x "$stub/nvcc"; }
nvcc_stub 13.0
( . "$COMMON"; PATH="$stub:$PATH" rt_check_nvcc ) > /dev/null 2>&1 && ok "nvcc release 13.0 passes for the 13.0.2 image" || bad "nvcc 13.0 refused"
nvcc_stub 13.4
( . "$COMMON"; PATH="$stub:$PATH" rt_check_nvcc ) > /dev/null 2>&1 && bad "nvcc 13.4 accepted" || ok "nvcc release 13.4 is refused"
rm -f "$stub/nvcc"
( . "$COMMON"; PATH="$stub:/usr/bin:/bin" rt_check_nvcc ) > /dev/null 2>&1 && bad "missing nvcc accepted" || ok "missing nvcc is refused"

( . "$COMMON"; TREE12=0123456789ab; [ "$(rt_tree12)" = 0123456789ab ] ) && ok "a 12 hex TREE12 is taken as given" || bad "TREE12 not taken"
( . "$COMMON"; TREE12=xyz; rt_tree12 ) > /dev/null 2>&1 && bad "bad TREE12 accepted" || ok "a bad TREE12 is refused"
# unset TREE12 falls back to the HEAD tree of runtime/ in a git checkout
repo="$SCRATCH/repo"; mkdir -p "$repo/runtime/build"
cp "$COMMON" "$repo/runtime/build/common.sh"; cp "$RT/versions.env" "$repo/runtime/versions.env"
"${GITQ[@]}" -C "$repo" init -q && "${GITQ[@]}" -C "$repo" add runtime && "${GITQ[@]}" -C "$repo" commit -q -m x
tree=$("${GITQ[@]}" -C "$repo" rev-parse HEAD:runtime | cut -c1-12)
[ "$(env -u TREE12 bash -c '. "$1"; rt_tree12' _ "$repo/runtime/build/common.sh")" = "$tree" ] && ok "TREE12 falls back to the first 12 hex of the runtime/ tree hash" || bad "TREE12 fallback gave a different value than git rev-parse HEAD:runtime"
( . "$COMMON"; [ "$(rt_cuda_version)" = 13.0.2 ] ) && ok "CUDA version 13.0.2 comes from the image tag" || bad "rt_cuda_version wrong"
finish test_common.sh
