#!/bin/bash
# Runs every runtime/tests/test_* file, one PASS or FAIL line each, exit 1 on any failure: run.sh [name-substring...]. RT_SRC_QWEN and RT_SRC_DEEPSEEK (patched clones) enable the option-name checks of test_print_cmake.sh.
HERE="$(cd "$(dirname "$0")" && pwd)"
PER_FILE_S=${RUN_TIMEOUT_S:-240}
fails=0; n=0
log="$(mktemp)"; trap 'rm -f "$log"' EXIT

for f in "$HERE"/test_*.sh; do
  [ -f "$f" ] || continue
  name="$(basename "$f")"
  if [ $# -gt 0 ]; then
    match=0; for pat in "$@"; do [[ "$name" == *"$pat"* ]] && match=1; done
    [ "$match" = 1 ] || continue
  fi
  n=$((n + 1)); start=$SECONDS
  timeout "$PER_FILE_S" bash "$f" > "$log" 2>&1 < /dev/null; rc=$?
  secs=$((SECONDS - start))
  if [ "$rc" = 0 ]; then
    echo "PASS $name (${secs} s)"
    grep '^SKIP' "$log" | sed 's/^/     | /'
  else
    [ "$rc" = 124 ] && why="timeout after $PER_FILE_S s" || why="exit $rc"
    echo "FAIL $name ($why, ${secs} s)"; tail -n 15 "$log" | sed 's/^/     | /'
    fails=$((fails + 1))
  fi
done

[ "$n" -gt 0 ] || { echo "FAIL run.sh: no test_* file matched"; exit 1; }
if [ "$fails" = 0 ]; then echo "PASS run.sh: $n test files"; else echo "FAIL run.sh: $fails of $n test files failed"; exit 1; fi
