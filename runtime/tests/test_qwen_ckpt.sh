#!/bin/bash
# M16-T02 offline: ckpt-per-request.diff (field parse, guard, no env switch, series order), the CPU smoke build line, fetch_gguf.sh with the QWEN_SMOKE pin, qwen_checkpoints.sh against a fake llama-server, workflow facts; RT_SRC_QWEN (patched clone) also checks the tree.
. "$(dirname "$0")/lib.sh"
PATCH="$RT/patches/qwen/ckpt-per-request.diff"
SERIES="$RT/patches/qwen/series"
SMOKE="$RT/smoke"
fx="$SCRATCH/fx"; mkdir -p "$fx"

# ---- the patch text ----
files=$(grep '^+++ b/' "$PATCH" | sed 's#^+++ b/##' | sort | tr '\n' ' ')
[ "$files" = "tools/server/server-context.cpp tools/server/server-schema.cpp tools/server/server-task.h " ] && ok "the patch touches exactly the server context, schema and task header" || bad "the patch touches: $files"
grep -q '^+    bool checkpoints     = true;' "$PATCH" && ok "task_params gets checkpoints, default true" || bad "no 'bool checkpoints = true' member"
grep -q '^+    add((new field_bool("checkpoints", params.checkpoints))' "$PATCH" && ok "the schema parses the request field checkpoints into the task params" || bad "no field_bool for checkpoints"
grep -q '^+                    do_checkpoint = do_checkpoint && slot.task->params.checkpoints;' "$PATCH" && ok "the checkpoint site is guarded by the per-request flag" || bad "no do_checkpoint guard"
check_not "no environment switch in the patch" grep -qi 'getenv' "$PATCH"
check_not "no n_predict condition in the patch (warm requests keep their checkpoints)" grep -q '^+.*n_predict' "$PATCH"
[ "$(grep -c '^+ *//' "$PATCH")" = 1 ] && ok "one added comment line" || bad "the patch adds $(grep -c '^+ *//' "$PATCH") comment lines"
added=$(( $(grep -c '^+' "$PATCH") - $(grep -c '^+++ ' "$PATCH") ))
[ "$added" = 7 ] && ok "the patch adds 7 lines" || bad "the patch adds $added lines, want 7"
check_not "no dash characters in the patch" grep -q $'\xe2\x80\x94\|\xe2\x80\x93' "$PATCH"
grep -qa 'no context checkpoints are created' "$PATCH" && ok "the field description carries the string the smoke and the workflow look for in the binaries" || bad "the description string changed"

# ---- series ----
entries=$(grep -v '^[[:space:]]*\(#\|$\)' "$SERIES" | tr '\n' ' ')
[ "$entries" = "pr31037-mtp-head-trim.diff dflash2-head-trim.diff patch_loadwait.py ckpt-per-request.diff " ] && ok "series: the checkpoint patch is last, after the head-trim diffs and patch_loadwait.py" || bad "series is: $entries"

# ---- the diff applies to a file built from its own pre-image ----
if python3 -I - "$PATCH" "$fx/pre" << 'PY'
import os, sys
patch, out = sys.argv[1:3]
cur, lines = None, {}
for l in open(patch).read().split("\n"):
    if l.startswith("+++ b/"):
        cur = l[6:]; lines[cur] = []
    elif cur and l and l[0] in " -" and not l.startswith("--- "):
        lines[cur].append(l[1:])
for f, body in lines.items():
    p = os.path.join(out, f); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("\n".join(body) + "\n")
PY
then
  git -C "$fx/pre" init -q 2> /dev/null
  git -C "$fx/pre" apply --check "$PATCH" 2> "$SCRATCH/apply.err" && ok "the diff is well formed and applies to its pre-image" || bad "the diff does not apply to its pre-image: $(head -2 "$SCRATCH/apply.err" | tr '\n' ' ')"
else
  bad "could not build the pre-image of the patch"
fi

# ---- the patched tree (RT_SRC_QWEN is a clone with the series applied) ----
if [ -n "${RT_SRC_QWEN:-}" ] && [ -d "$RT_SRC_QWEN/tools/server" ]; then
  git -C "$RT_SRC_QWEN" apply --reverse --check "$PATCH" 2> /dev/null && ok "the patched tree has the patch applied" || bad "the tree does not carry ckpt-per-request.diff"
  site=$(grep -n 'do_checkpoint = do_checkpoint && slot.task->params.checkpoints;' "$RT_SRC_QWEN/tools/server/server-context.cpp" | cut -d: -f1)
  seq=$(grep -n 'ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||' "$RT_SRC_QWEN/tools/server/server-context.cpp" | cut -d: -f1 | awk -v s="$site" '$1 > s {print; exit}')
  [ -n "$site" ] && [ -n "$seq" ] && [ "$site" -lt "$seq" ] && ok "the guard sits before the sequence-removal conditions" || bad "guard position wrong (guard $site, conditions $seq)"
  [ "$(grep -c 'params.checkpoints' "$RT_SRC_QWEN/tools/server/server-context.cpp")" = 1 ] && ok "params.checkpoints is read at one site only" || bad "params.checkpoints is read at several sites"
  grep -q 'field_bool("checkpoints", params.checkpoints)' "$RT_SRC_QWEN/tools/server/server-schema.cpp" && ok "the field is in the llama completion schema that /completion evaluates" || bad "field missing in server-schema.cpp"
  miss=""
  for n in $(grep -o -- '-D[A-Z0-9_]*=' "$GOLDEN/qwen-cpu-cmake.txt" | sed 's/^-D//; s/=$//' | grep -v '^CMAKE_'); do
    grep -rqw --include=CMakeLists.txt --include='*.cmake' -- "$n" "$RT_SRC_QWEN" || miss="$miss $n"
  done
  [ -z "$miss" ] && ok "every -D option of the CPU smoke line exists in the tree" || bad "smoke line options not found in the tree:$miss"
else
  skip "RT_SRC_QWEN not set, the patched tree is not inspected"
fi

# ---- CPU smoke build line ----
[ "$(bash "$BUILD/qwen_smoke.sh" --print-cmake)" = "$(cat "$GOLDEN/qwen-cpu-cmake.txt")" ] && ok "qwen_smoke.sh --print-cmake equals golden/qwen-cpu-cmake.txt" || bad "the CPU smoke line differs from its golden"
! grep -q 'GGML_CUDA=ON\|CUDA_ARCHITECTURES' "$GOLDEN/qwen-cpu-cmake.txt" && grep -q 'DGGML_CUDA=OFF' "$GOLDEN/qwen-cpu-cmake.txt" && ok "the smoke server is built without CUDA" || bad "the smoke line is not CPU-only"
grep -q 'DLLAMA_BUILD_SERVER=ON' "$GOLDEN/qwen-cpu-cmake.txt" && grep -q 'DLLAMA_USE_PREBUILT_UI=OFF' "$GOLDEN/qwen-cpu-cmake.txt" && ok "same server options as the Qwen runtime" || bad "server options differ from the Qwen runtime line"

# ---- pins ----
python3 -I - "$RT/versions.env" << 'PY' && ok "versions.env pins the smoke GGUF by revision, size and sha256" || bad "the QWEN_SMOKE pin is incomplete"
import re, sys
env = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l and not l.startswith("#"))
assert env["QWEN_SMOKE_GGUF_REPO"] == "unsloth/Qwen3.5-0.8B-GGUF"
assert re.fullmatch(r"[0-9a-f]{40}", env["QWEN_SMOKE_GGUF_REV"])
assert re.fullmatch(r"[0-9a-f]{64}", env["QWEN_SMOKE_GGUF_SHA256"])
assert int(env["QWEN_SMOKE_GGUF_SIZE"]) > 100_000_000
assert env["QWEN_SMOKE_GGUF_FILE"].endswith(".gguf") and env["QWEN_SMOKE_GGUF_BASE"].startswith("https://")
PY

# ---- fetch_gguf.sh with the QWEN_SMOKE prefix ----
d="$SCRATCH/hf/org/repo/resolve/REV1"; mkdir -p "$d" "$SCRATCH/out"; head -c 2048 /dev/urandom > "$d/m.gguf"
gsha=$(sha256sum "$d/m.gguf" | cut -c1-64)
cat > "$fx/versions.env" << EOFV
QWEN_SMOKE_GGUF_BASE=https://huggingface.co
QWEN_SMOKE_GGUF_REPO=org/repo
QWEN_SMOKE_GGUF_REV=REV1
QWEN_SMOKE_GGUF_FILE=m.gguf
QWEN_SMOKE_GGUF_SIZE=2048
QWEN_SMOKE_GGUF_SHA256=$gsha
EOFV
fetch() { env RT_VERSIONS="$1" QWEN_SMOKE_GGUF_BASE="file://$SCRATCH/hf" bash "$SMOKE/fetch_gguf.sh" "$2" QWEN_SMOKE > "$SCRATCH/fetch.out" 2>&1; }
if fetch "$fx/versions.env" "$SCRATCH/out/q.gguf"; then cmp -s "$SCRATCH/out/q.gguf" "$d/m.gguf" && ok "fetch_gguf.sh QWEN_SMOKE downloads and verifies the file" || bad "bytes differ"; else bad "fetch_gguf.sh QWEN_SMOKE failed: $(tail -2 "$SCRATCH/fetch.out" | tr '\n' ' ')"; fi
sed "s/^QWEN_SMOKE_GGUF_SHA256=.*/QWEN_SMOKE_GGUF_SHA256=$(printf '0%.0s' $(seq 1 64))/" "$fx/versions.env" > "$fx/versions-sha.env"
check_not "a wrong sha256 is refused" fetch "$fx/versions-sha.env" "$SCRATCH/out/w.gguf"
[ ! -e "$SCRATCH/out/w.gguf" ] && [ ! -e "$SCRATCH/out/w.gguf.part" ] && ok "a refused file leaves nothing behind" || bad "a refused file stayed"
check_not "an unset pin is refused" env RT_VERSIONS="$fx/versions.env" bash "$SMOKE/fetch_gguf.sh" "$SCRATCH/out/x.gguf" NOPE_PIN
check_not "a malformed prefix is refused" env RT_VERSIONS="$fx/versions.env" bash "$SMOKE/fetch_gguf.sh" "$SCRATCH/out/x.gguf" 'a;b'

# ---- qwen_checkpoints.sh against a fake llama-server ----
fake="$SCRATCH/fake"; mkdir -p "$fake"; echo gguf > "$fake/m.gguf"
cat > "$fake/llama-server" << 'PY'
#!/usr/bin/env python3
# no context checkpoints are created: the string the smoke looks for in the loaded server code
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
assert "-m" in a and "-lv" in a
mode = os.environ.get("FAKE_MODE", "good")
if mode == "exit":
    sys.exit(3)
print("srv    load_model: context checkpoints enabled, max = 32, min spacing = 8192", file=sys.stderr, flush=True)

class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        make = {"good": body.get("checkpoints", True), "ignore": True, "never": False}[mode]
        if make:
            print("slot update_slots: id  0 | task 3 | created context checkpoint 1 of 32 (pos_min = 1, pos_max = 60, n_tokens = 61, size = 20.000 MiB)", file=sys.stderr, flush=True)
        out = json.dumps({"content": "" if body["n_predict"] == 0 else "ok", "tokens_evaluated": 62, "timings": {"prompt_ms": 35.0}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
HTTPServer(("127.0.0.1", int(a[a.index("--port") + 1])), H).serve_forever()
PY
chmod +x "$fake/llama-server"
sed '2d' "$fake/llama-server" > "$fake/unpatched"; chmod +x "$fake/unpatched"
port=$((18200 + RANDOM % 80))
smoke() { # smoke MODE SERVER
  env FAKE_MODE="$1" LLAMA_SERVER="$2" QWEN_SMOKE_GGUF="$fake/m.gguf" SMOKE_TMP="$SCRATCH/smoke-$1" SMOKE_PORT=$port timeout 100 bash "$SMOKE/qwen_checkpoints.sh" > "$SCRATCH/smoke.out" 2>&1
}
if smoke good "$fake/llama-server"; then ok "qwen_checkpoints.sh passes when the field switches checkpoints off and plain requests make them"; else bad "qwen_checkpoints.sh failed on a good server: $(tail -4 "$SCRATCH/smoke.out" | tr '\n' ' ')"; fi
grep -q 'qwen_checkpoints: PASS' "$SCRATCH/smoke.out" && ok "it ends with a PASS line" || bad "no PASS line"
[ "$(grep -c 'checkpoint lines' "$SCRATCH/smoke.out")" = 6 ] && ok "it reports six requests with their checkpoint line counts" || bad "wrong number of report lines"
if smoke ignore "$fake/llama-server"; then bad "passed although the server ignores the field"; else grep -q 'qwen_checkpoints: FAILED: off1' "$SCRATCH/smoke.out" && ok "a server that ignores the field fails on the first request with the field" || bad "no FAILED line for ignore: $(tail -2 "$SCRATCH/smoke.out" | tr '\n' ' ')"; fi
if smoke never "$fake/llama-server"; then bad "passed although plain requests make no checkpoint"; else grep -q 'qwen_checkpoints: FAILED: plain1' "$SCRATCH/smoke.out" && ok "a server that never checkpoints fails on the plain request" || bad "no FAILED line for never: $(tail -2 "$SCRATCH/smoke.out" | tr '\n' ' ')"; fi
if smoke good "$fake/unpatched"; then bad "passed with a server lacking the patch string"; else grep -q 'lacks the checkpoints field' "$SCRATCH/smoke.out" && ok "a binary without the patch string is refused before any request" || bad "no patch-string failure: $(tail -2 "$SCRATCH/smoke.out" | tr '\n' ' ')"; fi
if smoke exit "$fake/llama-server"; then bad "passed with a server that exits"; else grep -q 'exited before it was healthy' "$SCRATCH/smoke.out" && ok "a server that exits fails the smoke" || bad "no exit failure line"; fi
check_not "qwen_checkpoints.sh refuses a missing GGUF" env LLAMA_SERVER="$fake/llama-server" QWEN_SMOKE_GGUF="$fake/none.gguf" bash "$SMOKE/qwen_checkpoints.sh"
check_not "qwen_checkpoints.sh needs its environment" env -u LLAMA_SERVER bash "$SMOKE/qwen_checkpoints.sh"
sleep 1
[ -z "$(pgrep -f "$fake/llama-server" || true)" ] && ok "no fake server is left running" || bad "a fake server is still running"

# ---- workflow facts ----
WF="$RT/workflows/build.yml"
grep -q "smoke-qwen:" "$WF" && grep -q "if: contains(needs.plan.outputs.families, 'qwen')" "$WF" && ok "a smoke-qwen job runs when the qwen family is built" || bad "no smoke-qwen job"
grep -q 'build/qwen_smoke.sh' "$WF" && grep -q 'fetch_gguf.sh "$RUNNER_TEMP/qwen-smoke.gguf" QWEN_SMOKE' "$WF" && grep -q 'runtime/smoke/qwen_checkpoints.sh' "$WF" && ok "it builds the CPU server, fetches the pinned GGUF and runs qwen_checkpoints.sh" || bad "smoke-qwen misses a step"
grep -q 'needs: \[plan, build, smoke, smoke-qwen\]' "$WF" && grep -q "needs.smoke-qwen.result == 'success'" "$WF" && ok "publishing waits for smoke-qwen" || bad "publish does not wait for smoke-qwen"
grep -q 'no context checkpoints are created' "$WF" && grep -q 'libllama-server-impl.so' "$WF" && ok "the build job greps the packed server library for the patched field" || bad "no tarball check in the workflow"
finish test_qwen_ckpt.sh
