#!/bin/bash
# loadwait_test.cpp (M15-T12) and, for DeepSeek, streamprefill_test.cpp (M15-T13) against CPU-only builds of the patched trees RT_SRC_QWEN and RT_SRC_DEEPSEEK (common.sh --apply-only); build dirs RT_LOADWAIT_BUILD_QWEN and RT_LOADWAIT_BUILD_DEEPSEEK (default TREE-loadwait), CMAKE (cmake), JOBS (2); without a tree that family is skipped.
. "$(dirname "$0")/lib.sh"

for fam in qwen deepseek; do
  tree_var=RT_SRC_${fam^^}
  build_var=RT_LOADWAIT_BUILD_${fam^^}
  tree=${!tree_var:-}
  if [ -z "$tree" ]; then
    skip "$fam: $tree_var is not set, loadwait_test.cpp not built"
    continue
  fi
  out=${!build_var:-$tree-loadwait}
  if ! grep -q LOADWAIT_PATCH "$tree/src/llama-mmap.cpp" 2> /dev/null; then
    bad "$fam: $tree has no patch_loadwait.py applied"
    continue
  fi
  if [ ! -f "$out/build.ninja" ]; then
    "${CMAKE:-cmake}" -S "$tree" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE=None -DCMAKE_C_FLAGS=-O0 -DCMAKE_CXX_FLAGS=-O0 -DBUILD_SHARED_LIBS=OFF \
      -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF > "$SCRATCH/cmake-$fam.log" 2>&1 ||
      { bad "$fam: configure failed: $(tail -n 3 "$SCRATCH/cmake-$fam.log" | tr '\n' ' ')"; continue; }
  fi
  ninja -C "$out" -j "${JOBS:-2}" llama > "$SCRATCH/ninja-$fam.log" 2>&1 || { bad "$fam: libllama failed to build: $(tail -n 3 "$SCRATCH/ninja-$fam.log" | tr '\n' ' ')"; continue; }
  defs=()
  [ "$fam" = deepseek ] && defs=(-DLOADWAIT_MOE=1)
  if ! g++ -std=c++17 -O0 "${defs[@]}" -I"$tree/src" -I"$tree/include" -I"$tree/ggml/include" -I"$tree/ggml/src" "$RT/tests/loadwait_test.cpp" -o "$SCRATCH/loadwait-$fam" \
    -Wl,--start-group "$out/src/libllama.a" "$out/ggml/src/libggml.a" "$out/ggml/src/libggml-cpu.a" "$out/ggml/src/libggml-base.a" -Wl,--end-group -lpthread -ldl -lm > "$SCRATCH/cc-$fam.log" 2>&1; then
    bad "$fam: loadwait_test.cpp failed to compile: $(grep -m 3 error "$SCRATCH/cc-$fam.log" | tr '\n' ' ')"
    continue
  fi
  timeout 200 "$SCRATCH/loadwait-$fam" > "$SCRATCH/run-$fam.log" 2>&1
  rc=$?
  sed "s/^/     $fam | /" "$SCRATCH/run-$fam.log"
  [ "$rc" = 0 ] && ok "$fam: loadwait_test passed ($(grep -c '^ok' "$SCRATCH/run-$fam.log") checks)" || bad "$fam: loadwait_test exit $rc"
  [ "$fam" = deepseek ] || continue
  # streamprefill_test.cpp (M15-T13): the streaming MoE prefill on a fake GPU, then checksizes.resident_set and punch.allowed on its fixture
  grep -q STREAMPREFILL_PATCH "$tree/src/llama-moe-cache.cpp" 2> /dev/null || { bad "$fam: $tree has no patch_streamprefill.py applied"; continue; }
  if ! g++ -std=c++17 -O0 -I"$tree/src" -I"$tree/include" -I"$tree/ggml/include" -I"$tree/ggml/src" "$RT/tests/streamprefill_test.cpp" -o "$SCRATCH/streamprefill" \
    -Wl,--start-group "$out/src/libllama.a" "$out/ggml/src/libggml.a" "$out/ggml/src/libggml-cpu.a" "$out/ggml/src/libggml-base.a" -Wl,--end-group -lpthread -ldl -lm > "$SCRATCH/cc-sp.log" 2>&1; then
    bad "$fam: streamprefill_test.cpp failed to compile: $(grep -m 3 error "$SCRATCH/cc-sp.log" | tr '\n' ' ')"
    continue
  fi
  mkdir -p "$SCRATCH/sp"
  STREAMPREFILL_KEEP=$SCRATCH/sp timeout 200 "$SCRATCH/streamprefill" > "$SCRATCH/run-sp.log" 2>&1
  rc=$?
  sed "s/^/     $fam stream | /" "$SCRATCH/run-sp.log"
  [ "$rc" = 0 ] && ok "$fam: streamprefill_test passed ($(grep -c '^ok' "$SCRATCH/run-sp.log") checks)" || bad "$fam: streamprefill_test exit $rc"
  python3 -I - "$RT/../vm/deepseek" "$SCRATCH/sp" > "$SCRATCH/sim.log" 2>&1 << 'PY'
import os
import sys
sys.path.insert(0, sys.argv[1])
import checksizes
import punch
keep = sys.argv[2]
cache, dyn, cpu_re = open(os.path.join(keep, 'params.txt')).read().split()
fixture, resident = os.path.join(keep, 'fixture.gguf'), os.path.join(keep, 'resident.txt')
pinned, groups = checksizes.resident_set([fixture], cpu_re, os.path.join(keep, 'profile.txt'), int(cache), int(dyn))
print('groups', [(g['layers'], g['n_slots']) for g in groups], 'pinned', len(pinned))
assert pinned == punch.resident(resident), 'checksizes.resident_set differs from the prefill\'s resident.txt'
want = sorted((off, nb) for _, off, nb in punch.allowed(cpu_re, [fixture], resident))
got = sorted((int(w[2]), int(w[3])) for w in (line.split() for line in open(os.path.join(keep, 'uploaded.txt'))))
assert want == got, 'uploaded lines %d, punch.allowed ranges %d' % (len(got), len(want))
print('uploaded ranges equal punch.allowed:', len(got))
PY
  rc=$?
  sed "s/^/     $fam stream | /" "$SCRATCH/sim.log"
  [ "$rc" = 0 ] && ok "$fam: checksizes.resident_set equals resident.txt and the uploaded ranges equal punch.allowed on the same fixture" || bad "$fam: the Python plan differs from the prefill: $(tail -n 1 "$SCRATCH/sim.log")"
done
finish test_loadwait.sh
