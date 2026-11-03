#!/bin/bash
# --print-cmake of both families equals the golden lines, needs no environment and touches nothing; with RT_SRC_QWEN and RT_SRC_DEEPSEEK (patched clones at the pinned tags) every -D option and target name is checked against the tree.
. "$(dirname "$0")/lib.sh"
MILESTONE="$RT/../.orchestrate/milestones/06-vm-runtime.md"

for fam in qwen deepseek; do
  work="$SCRATCH/cwd-$fam"; mkdir -p "$work"
  out=$(cd "$work" && env -i PATH="$PATH" bash "$BUILD/$fam.sh" --print-cmake 2> "$SCRATCH/err-$fam")
  rc=$?
  [ "$rc" = 0 ] && ok "$fam.sh --print-cmake exits 0 without WORK or TREE12" || bad "$fam.sh --print-cmake exit $rc: $(cat "$SCRATCH/err-$fam")"
  [ "$out" = "$(cat "$GOLDEN/$fam-cmake.txt")" ] && ok "$fam: printed line equals golden/$fam-cmake.txt" || bad "$fam: printed line differs from golden/$fam-cmake.txt"
  [ -z "$(ls -A "$work")" ] && ok "$fam: --print-cmake writes nothing" || bad "$fam: --print-cmake created files"
  [ "$(wc -l < "$GOLDEN/$fam-cmake.txt")" = 1 ] && ok "$fam: golden is one line" || bad "$fam: golden is not one line"
done

grep -q "DGGML_CUDA_CCCL_VERSION=$(sed -n 's/^CCCL=//p' "$RT/versions.env")" "$GOLDEN/qwen-cmake.txt" && ok "qwen golden carries CCCL of versions.env" || bad "qwen golden and versions.env disagree on CCCL"

if [ -f "$MILESTONE" ]; then
  # the milestone text with the one documented change: test-backend-ops needs LLAMA_BUILD_TESTS=ON at b12015
  want=$(python3 -I -c "
import re, sys
t = open(sys.argv[1]).read()
line = re.search(r'\*\*Qwen CMake line\*\*[^\`]*\`(cmake -S src[^\`]*)\`', t).group(1)
print(line.replace('-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_TESTS=ON'))" "$MILESTONE")
  [ "$want" = "$(cat "$GOLDEN/qwen-cmake.txt")" ] && ok "qwen golden is the Technical decisions 4 line with LLAMA_BUILD_TESTS=ON" || bad "qwen golden drifted from Technical decisions 4"
else
  skip "milestone file not found, qwen golden not compared with Technical decisions 4"
fi

# every -DNAME= option appears in the tree's CMake files (CMAKE_* are CMake's own)
names_in_tree() { # names_in_tree LABEL TREE LINEFILE
  local label=$1 tree=$2 line=$3 n miss=""
  for n in $(grep -o -- '-D[A-Z0-9_]*=' "$line" | sed 's/^-D//; s/=$//' | grep -v '^CMAKE_'); do
    grep -rqw --include=CMakeLists.txt --include='*.cmake' -- "$n" "$tree" || miss="$miss $n"
  done
  [ -z "$miss" ] && ok "$label: every -D option name exists in the tree" || bad "$label: options not found in the tree:$miss"
}
if [ -n "${RT_SRC_QWEN:-}" ] && [ -d "$RT_SRC_QWEN/ggml" ]; then
  names_in_tree qwen "$RT_SRC_QWEN" "$GOLDEN/qwen-cmake.txt"
  grep -q 'llama_build(test-backend-ops.cpp' "$RT_SRC_QWEN/tests/CMakeLists.txt" && ok "qwen: test-backend-ops is a target of tests/CMakeLists.txt" || bad "qwen: no test-backend-ops target in tests/CMakeLists.txt"
  grep -q 'LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS' "$RT_SRC_QWEN/CMakeLists.txt" && ok "qwen: tests/ is added only with LLAMA_BUILD_TESTS (so ON is needed)" || bad "qwen: the LLAMA_BUILD_TESTS guard of tests/ changed"
  grep -q 'set(TARGET llama-fit-params)' "$RT_SRC_QWEN/tools/fit-params/CMakeLists.txt" && ok "qwen: llama-fit-params target exists" || bad "qwen: no llama-fit-params target"
else
  skip "RT_SRC_QWEN not set, option names of the qwen line not checked against a b12015 tree"
fi
if [ -n "${RT_SRC_DEEPSEEK:-}" ] && [ -d "$RT_SRC_DEEPSEEK/ggml" ]; then
  names_in_tree deepseek "$RT_SRC_DEEPSEEK" "$GOLDEN/deepseek-cmake.txt"
  grep -q 'set(TARGET llama-fit-params)' "$RT_SRC_DEEPSEEK/tools/fit-params/CMakeLists.txt" && ok "deepseek: llama-fit-params target exists" || bad "deepseek: no llama-fit-params target"
else
  skip "RT_SRC_DEEPSEEK not set, option names of the deepseek line not checked against a b11963 tree"
fi
finish test_print_cmake.sh
