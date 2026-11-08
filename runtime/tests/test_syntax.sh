#!/bin/bash
# bash -n on every script, Python syntax of every patch script, pin.c compiles, scripts are executable, no dashes in our own files, the dev repo name only in versions.env.
. "$(dirname "$0")/lib.sh"

for f in "$RT"/build/*.sh "$RT"/smoke/*.sh "$RT"/tests/*.sh; do
  bash -n "$f" 2> "$SCRATCH/err" && ok "bash -n ${f#"$RT"/}" || bad "bash -n ${f#"$RT"/}: $(head -2 "$SCRATCH/err" | tr '\n' ' ')"
done
for f in "$RT"/build/*.sh "$RT"/smoke/*.sh "$RT"/tests/run.sh "$RT"/tests/test_*.sh; do
  [ -x "$f" ] && ok "executable ${f#"$RT"/}" || bad "not executable: ${f#"$RT"/}"
done
bad_py=""
for f in "$RT"/patches/deepseek/*.py; do python3 -I -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$f" 2> /dev/null || bad_py="$bad_py $(basename "$f")"; done
[ -z "$bad_py" ] && ok "all patch scripts parse as Python" || bad "patch scripts that do not parse:$bad_py"
if command -v gcc > /dev/null; then
  gcc -std=gnu11 -fsyntax-only -Wall -Werror=implicit-function-declaration "$RT/src/pin.c" 2> "$SCRATCH/gcc.err" && ok "pin.c passes gcc -fsyntax-only" || bad "pin.c does not compile: $(head -3 "$SCRATCH/gcc.err" | tr '\n' ' ')"
else
  skip "gcc not installed, pin.c not syntax-checked"
fi
if command -v shellcheck > /dev/null; then
  shellcheck -S warning "$RT"/build/*.sh "$RT"/smoke/*.sh > "$SCRATCH/sc.out" 2>&1 && ok "shellcheck build scripts" || bad "shellcheck: $(head -5 "$SCRATCH/sc.out" | tr '\n' ' ')"
else
  skip "shellcheck not installed, build scripts not linted"
fi

# patches are byte copies and exempt; our own files carry no em or en dash
dashes=$(grep -rln --exclude-dir=patches --exclude=pin.c $'\xe2\x80\x94\|\xe2\x80\x93' "$RT" 2> /dev/null)
[ -z "$dashes" ] && ok "no em or en dash in our own files" || bad "dash characters in: $dashes"
# the name of the dev repo lives in versions.env only (the pattern is split so that this file does not match)
hits=$(grep -rl "colab-hub-""runtime" "$RT" 2> /dev/null | sed "s#^$RT/##")
[ "$hits" = "versions.env" ] && ok "the dev repo name appears only in versions.env" || bad "the dev repo name appears in: $hits"
finish test_syntax.sh
