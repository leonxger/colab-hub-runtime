#!/bin/bash
# Smoke test of the per-request "checkpoints" switch on CPU: llama-server with the pinned hybrid Qwen3.5 GGUF logs a context checkpoint for a plain request and none when the request sends "checkpoints": false; env LLAMA_SERVER, QWEN_SMOKE_GGUF, SMOKE_TMP, SMOKE_PORT.
set -euo pipefail
: "${LLAMA_SERVER:?path of llama-server}" "${QWEN_SMOKE_GGUF:?path of the GGUF}"
tmp=${SMOKE_TMP:-$(mktemp -d)}
mkdir -p "$tmp"
port=${SMOKE_PORT:-18090}
[ -x "$LLAMA_SERVER" ] && [ -f "$QWEN_SMOKE_GGUF" ] || { echo "qwen_checkpoints: missing server or GGUF" >&2; exit 1; }
marker='created context checkpoint'
patch_text='no context checkpoints are created'

"$LLAMA_SERVER" -m "$QWEN_SMOKE_GGUF" --host 127.0.0.1 --port "$port" -c 8192 -np 1 -t 4 -lv 4 --no-warmup > "$tmp/server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2> /dev/null || true; wait "$pid" 2> /dev/null || true' EXIT
fail() { echo "qwen_checkpoints: FAILED: $*" >&2; echo "---- server.log (tail)" >&2; tail -n 60 "$tmp/server.log" >&2; exit 1; }

# the server code lives in libllama-server-impl.so: the one the binary loads must hold the patch
impl=$(ldd "$LLAMA_SERVER" 2> /dev/null | awk '/libllama-server-impl/ {print $3}') || true
holders=("$LLAMA_SERVER")
if [ -n "$impl" ]; then
  [ "$(dirname "$(readlink -f "$impl")")" = "$(dirname "$(readlink -f "$LLAMA_SERVER")")" ] || fail "libllama-server-impl resolves to $impl, outside the binary's directory"
  holders+=("$impl")
fi
grep -qa "$patch_text" "${holders[@]}" || fail "the loaded server code (${holders[*]}) lacks the checkpoints field"
echo "qwen_checkpoints: patched server code in ${impl:-the binary itself}"

up=0
for _ in $(seq 1 300); do
  kill -0 "$pid" 2> /dev/null || fail "server exited before it was healthy"
  if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" > /dev/null 2>&1; then up=1; break; fi
  sleep 1
done
[ "$up" = 1 ] || fail "no /health answer within 300 s"
grep -q 'context checkpoints enabled' "$tmp/server.log" || fail "the log does not show 'context checkpoints enabled' (checkpoints off or verbosity too low)"

filler='the ferries come in at dawn and the fish market opens before the first bell rings in the old customs house by the water, so the traders shout their prices across the cobbles while the gulls circle overhead and wait for scraps, and nobody in the harbour town has ever hurried.'
created() { grep -c "$marker" "$tmp/server.log" || true; }

# request NAME FIRSTWORD N_PREDICT FIELD_JSON EXPECT (zero or some): prints the checkpoint lines the request added
request() {
  local name=$1 word=$2 n=$3 field=$4 expect=$5 before after delta
  before=$(created)
  printf '{"prompt": "%s %s", "n_predict": %s, "temperature": 0, "seed": 7%s}' "$word" "$filler" "$n" "$field" > "$tmp/$name.json"
  curl -fsS --max-time 300 -H 'Content-Type: application/json' -d @"$tmp/$name.json" -o "$tmp/$name.out" "http://127.0.0.1:$port/completion" || fail "$name: /completion request"
  sleep 1
  after=$(created)
  delta=$((after - before))
  python3 -I - "$tmp/$name.out" "$name" "$delta" "$n" << 'PY' || fail "$name: the response is not a completion"
import json, sys
r = json.load(open(sys.argv[1]))
n = int(sys.argv[4])
assert r["tokens_evaluated"] > 40, "prompt of %d tokens is too short to checkpoint" % r["tokens_evaluated"]
assert n == 0 or r["content"].strip(), "empty answer"
print("qwen_checkpoints: %s: %d prompt tokens, %.0f ms prompt, %s checkpoint lines" % (sys.argv[2], r["tokens_evaluated"], r["timings"]["prompt_ms"], sys.argv[3]))
PY
  case $expect in
    zero) [ "$delta" = 0 ] || fail "$name: $delta checkpoint lines for a request that switched them off" ;;
    some) [ "$delta" -ge 1 ] || fail "$name: no checkpoint line for a request that should make them" ;;
  esac
}

request off1 Harbour 8 ', "checkpoints": false' zero
request plain1 Quarry 8 '' some
request off2 Orchard 8 ', "checkpoints": false' zero
request on1 Lantern 8 ', "checkpoints": true' some
request warm Glacier 0 '' some
request warmoff Meadow 0 ', "checkpoints": false' zero
echo "qwen_checkpoints: PASS"
