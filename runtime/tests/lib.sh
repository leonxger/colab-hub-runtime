#!/bin/bash
# Helpers of the runtime tests: source it, call ok, bad, skip and finish NAME.
RT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$RT/build"
GOLDEN="$RT/tests/golden"
fails=0
checks=0
ok() { checks=$((checks + 1)); echo "ok   $*"; }
bad() { checks=$((checks + 1)); fails=$((fails + 1)); echo "FAIL $*"; }
skip() { echo "SKIP $*"; }
# check LABEL CMD...: ok when the command succeeds
check() { local label=$1; shift; if "$@" > /dev/null 2>&1; then ok "$label"; else bad "$label"; fi; }
# check_not LABEL CMD...: ok when the command fails
check_not() { local label=$1; shift; if "$@" > /dev/null 2>&1; then bad "$label"; else ok "$label"; fi; }
finish() {
  if [ "$fails" = 0 ]; then echo "PASS $1 ($checks checks)"; exit 0; fi
  echo "FAIL $1: $fails of $checks checks failed"; exit 1
}
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/runtime-test.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
