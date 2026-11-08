#!/bin/bash
# runtime/sync.sh against a fake gh and curl on PATH, a local bare git remote and a scratch copy of the repo: tag, dry run, publish, pin, verify; nothing touches GitHub. Also the workflow facts and the runtime.json schema.
. "$(dirname "$0")/lib.sh"
. "$RT/versions.env"
REPO="$RUNTIME_OWNER/$RUNTIME_REPO"
APP="$(cd "$RT/.." && pwd)"
BIN="$SCRATCH/bin"
ST="$SCRATCH/state"
SRC="$SCRATCH/src"
REMOTE="$SCRATCH/remote.git"
mkdir -p "$BIN"

[ -f "$APP/vm/lib/manifest.py" ] && [ -f "$APP/manifests/runtime.schema.json" ] || { echo "FAIL test_sync.sh: needs vm/lib/manifest.py and manifests/runtime.schema.json of the app repo"; exit 1; }

# ---- the workflow and the schema file, read as text ----
WF="$RT/workflows/build.yml"
check "sync.sh passes bash -n" bash -n "$RT/sync.sh"
check "sync.sh is executable" test -x "$RT/sync.sh"
check "workflow uses actions/checkout@v7.0.1" grep -q 'uses: actions/checkout@v7.0.1' "$WF"
check "workflow uses actions/cache@v6.1.0" grep -q 'uses: actions/cache@v6.1.0' "$WF"
check "ccache key is ccache-<family>-<hash of versions.env and patches>" grep -qF "key: ccache-\${{ matrix.family }}-\${{ hashFiles('runtime/versions.env', format('runtime/patches/{0}/**', matrix.family)) }}" "$WF"
check "CCACHE_MAXSIZE is 2G" grep -q 'CCACHE_MAXSIZE: 2G' "$WF"
check "permissions contents write" grep -Pzq 'permissions:\n  contents: write' "$WF"
check "concurrency per tag" grep -qF 'group: runtime-${{ inputs.tag }}' "$WF"
check "build jobs have timeout-minutes 300" grep -q 'timeout-minutes: 300' "$WF"
check "draft release is created with --latest=false" grep -qE 'gh release create .*--draft --latest=false' "$WF"
check "release is published with --latest=false" grep -qE 'gh release edit .*--draft=false --latest=false' "$WF"
check "image comes from the digest in versions.env" grep -q 'CUDA_IMAGE_DIGEST' "$WF"
check_not "workflow holds no sha256 literal" grep -q 'sha256:' "$WF"
check_not "workflow stores no artifact" grep -q 'upload-artifact' "$WF"
check_not "workflow names no secret besides GITHUB_TOKEN" bash -c "grep -o 'secrets\\.[A-Za-z_]*' '$WF' | grep -v '^secrets.GITHUB_TOKEN\$'"
check "workflow uses secrets.GITHUB_TOKEN" grep -q 'secrets.GITHUB_TOKEN' "$WF"
check_not "workflow does not carry the repo name" grep -q "$RUNTIME_REPO" "$WF"
AL=$(command -v actionlint || ls "$HOME/.local/bin/actionlint" 2> /dev/null)
if [ -n "$AL" ]; then
  if "$AL" -no-color "$WF" > "$SCRATCH/al.out" 2>&1; then ok "actionlint $("$AL" -version | head -n 1) passes on the workflow"; else bad "actionlint: $(head -n 5 "$SCRATCH/al.out" | tr '\n' ' ')"; fi
else
  skip "actionlint not installed (tools/dev/fetch-actionlint.sh), workflow not linted"
fi

# ---- the schema, with positive and negative pins ----
cat > "$SCRATCH/schema_check.py" << 'PY'
import copy, json, sys
sys.path.insert(0, sys.argv[1] + "/vm/lib")
import manifest
schema = manifest.load_json(sys.argv[1] + "/manifests/runtime.schema.json")
try:
    import jsonschema
except ImportError:
    jsonschema = None

def errors(doc):
    found = [m for _, m in manifest.check_schema(doc, schema)]
    if jsonschema is not None:
        found += [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(doc)]
    return found

good = {
    "schema": 1, "base_url": "https://github.com/o/r/releases/download", "tag": "rt-0123456789ab",
    "builds": {
        "qwen": {"llama_tag": "b12015", "commit": "a" * 40, "patches": ["x.diff"], "cuda": "13.0.2", "archs": ["80-real", "120a-real"], "gpus": ["G4", "A100"], "file": "llama-qwen-0123456789ab.tar.gz", "size": 5, "sha256": "b" * 64},
        "deepseek": {"llama_tag": "b11963", "commit": "c" * 40, "patches": ["y.py"], "cuda": "13.0.2", "archs": ["120a-real"], "gpus": ["G4"], "file": "llama-deepseek-0123456789ab.tar.gz", "size": 6, "sha256": "d" * 64},
        "voice": {"audiocpp_tag": "v0.9.4", "audiocpp_commit": "f" * 40, "qwentts_commit": "1" * 40, "num2words": "0.5.14", "cuda": "13.0.2", "archs": ["80-real", "120a-real"], "gpus": ["G4", "A100", "L4"], "file": "voice-0123456789ab.tar.gz", "size": 8, "sha256": "2" * 64},
        "cudart": {"file": "cudart-0123456789ab.tar.gz", "size": 7, "sha256": "e" * 64},
    },
}
if len(sys.argv) > 2:
    doc = manifest.load_json(sys.argv[2])
    found = errors(doc)
    print("\n".join(found) or "VALID")
    sys.exit(1 if found else 0)

def mutate(label, fn):
    doc = copy.deepcopy(good)
    fn(doc)
    print(label, "rejected" if errors(doc) else "ACCEPTED")

print("good", "VALID" if not errors(good) else "REJECTED " + "; ".join(errors(good)))
mutate("tag-not-hex", lambda d: d.update(tag="rt-XYZ"))
mutate("tag-13-hex", lambda d: d.update(tag="rt-0123456789abc"))
mutate("schema-2", lambda d: d.update(schema=2))
mutate("http-base", lambda d: d.update(base_url="http://github.com/o/r"))
mutate("extra-top-field", lambda d: d.update(latest=True))
mutate("no-cudart", lambda d: d["builds"].pop("cudart"))
mutate("extra-build", lambda d: d["builds"].update(other=d["builds"]["cudart"]))
mutate("sha-63", lambda d: d["builds"]["qwen"].update(sha256="b" * 63))
mutate("sha-upper", lambda d: d["builds"]["qwen"].update(sha256="B" * 64))
mutate("size-0", lambda d: d["builds"]["qwen"].update(size=0))
mutate("size-string", lambda d: d["builds"]["qwen"].update(size="5"))
mutate("gpu-unknown", lambda d: d["builds"]["qwen"].update(gpus=["V100"]))
mutate("gpus-empty", lambda d: d["builds"]["qwen"].update(gpus=[]))
mutate("patches-empty", lambda d: d["builds"]["qwen"].update(patches=[]))
mutate("arch-bare", lambda d: d["builds"]["qwen"].update(archs=["80"]))
mutate("cuda-short", lambda d: d["builds"]["qwen"].update(cuda="13.0"))
mutate("commit-short", lambda d: d["builds"]["qwen"].update(commit="abc"))
mutate("file-path", lambda d: d["builds"]["qwen"].update(file="../x.tar.gz"))
mutate("file-not-tar", lambda d: d["builds"]["cudart"].update(file="x.zip"))
mutate("cudart-extra-field", lambda d: d["builds"]["cudart"].update(archs=["80-real"]))
mutate("qwen-missing-gpus", lambda d: d["builds"]["qwen"].pop("gpus"))
mutate("voice-sha-63", lambda d: d["builds"]["voice"].update(sha256="2" * 63))
mutate("voice-no-qwentts-commit", lambda d: d["builds"]["voice"].pop("qwentts_commit"))
mutate("voice-extra-field", lambda d: d["builds"]["voice"].update(patches=["x.diff"]))
mutate("voice-audiocpp-tag-bare", lambda d: d["builds"]["voice"].update(audiocpp_tag="0.9.4"))
mutate("voice-file-path", lambda d: d["builds"]["voice"].update(file="a/b.tar.gz"))
mutate("voice-gpu-unknown", lambda d: d["builds"]["voice"].update(gpus=["V100"]))
mutate("build-tag-not-hex", lambda d: d["builds"]["qwen"].update(tag="rt-nothex"))
mutate("cudart-tag-short", lambda d: d["builds"]["cudart"].update(tag="rt-0123"))
mutate("tag-int", lambda d: d["builds"]["voice"].update(tag=5))
def positive(label, fn):
    doc = copy.deepcopy(good)
    fn(doc)
    print("pos-" + label, "VALID" if not errors(doc) else "REJECTED " + "; ".join(errors(doc)))

positive("voice-optional", lambda d: d["builds"].pop("voice"))
positive("build-own-tag", lambda d: (d["builds"]["voice"].update(tag="rt-aaaaaaaaaaaa"), d["builds"]["qwen"].update(tag="rt-bbbbbbbbbbbb"), d["builds"]["cudart"].update(tag="rt-cccccccccccc")))
PY
schema_out=$(python3 -I -B "$SCRATCH/schema_check.py" "$APP" 2>&1)
grep -q '^good VALID$' <<< "$schema_out" && ok "a well-formed pin validates against runtime.schema.json" || bad "good pin: $(head -n 3 <<< "$schema_out" | tr '\n' ' ')"
while read -r label verdict; do
  case "$label" in
    good) continue ;;
    pos-*) [ "$verdict" = VALID ] && ok "schema accepts ${label#pos-}" || bad "schema rejects ${label#pos-}: $verdict" ;;
    *) [ "$verdict" = rejected ] && ok "schema rejects $label" || bad "schema accepts $label" ;;
  esac
done <<< "$schema_out"

# ---- the fakes ----
cat > "$BIN/gh" << 'PY'
#!/usr/bin/env python3
import hashlib, json, os, subprocess, sys

st = os.environ["FAKE_ST"]
args = sys.argv[1:]
with open(os.path.join(st, "log"), "a") as f:
    f.write(" ".join(args) + "\n")

def path(name):
    return os.path.join(st, name)

def load(name, default):
    try:
        with open(path(name)) as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def save(name, value):
    with open(path(name), "w") as f:
        json.dump(value, f)

def fail(message, code=1):
    print(message, file=sys.stderr)
    sys.exit(code)

def asset_names(tag, families):
    env = dict(os.environ, TREE12=tag[3:])
    out = []
    for family in ("qwen", "deepseek", "voice"):
        if family not in families:
            continue
        text = subprocess.run(["bash", "%s/build/%s.sh" % (os.environ["FAKE_RT"], family), "--list-package"], env=env, capture_output=True, text=True, check=True).stdout
        out.append(text.splitlines()[0])
    if "qwen" in families:
        out.append("cudart-13.0.2-x64-%s.tar.gz" % tag[3:])
    return out

def finish_build(tag, success, families):
    releases = load("releases.json", {})
    rel = releases.setdefault(tag, {"isDraft": True, "assets": []})
    skip = os.environ.get("FAKE_SKIP_ASSET", "")
    os.makedirs(path("assets"), exist_ok=True)
    for index, name in enumerate(asset_names(tag, families)):
        if not success and index > 0:
            continue
        if skip and name.startswith(skip):
            continue
        data = (name + "\n").encode() * (100 * (index + 1))
        for fname, content in ((name, data), (name + ".sha256", ("%s  %s\n" % (hashlib.sha256(data).hexdigest(), name)).encode())):
            with open(path("assets/" + fname), "wb") as f:
                f.write(content)
            rel["assets"] = [a for a in rel["assets"] if a["name"] != fname] + [{"name": fname, "size": len(content), "state": "uploaded"}]
    if success and not os.environ.get("FAKE_NO_PUBLISH"):
        rel["isDraft"] = False
    save("releases.json", releases)

cmd = args[:2]
if cmd == ["repo", "view"]:
    if not os.path.exists(path("repo")):
        fail("GraphQL: Could not resolve to a Repository with the name '%s'. (repository)" % args[2])
    print(json.dumps({"visibility": open(path("repo")).read().strip()}))
elif cmd == ["repo", "create"]:
    open(path("repo"), "w").write("PRIVATE" if "--private" in args else "PUBLIC")
elif cmd == ["release", "view"]:
    rel = load("releases.json", {}).get(args[2])
    if rel is None:
        fail("release not found")
    print(json.dumps(rel))
elif cmd == ["release", "download"]:
    name = args[args.index("-p") + 1]
    if args[args.index("-O") + 1] != "-" or not os.path.exists(path("assets/" + name)):
        fail("no assets match the file pattern")
    sys.stdout.buffer.write(open(path("assets/" + name), "rb").read())
elif cmd == ["workflow", "run"]:
    left = int(open(path("dispatch_fails")).read()) if os.path.exists(path("dispatch_fails")) else 0
    if left > 0:
        open(path("dispatch_fails"), "w").write(str(left - 1))
        fail("could not find any workflows named build.yml")
    fields = dict(a.split("=", 1) for i, a in enumerate(args) if i > 0 and args[i - 1] == "-f")
    tag = fields["tag"]
    families = fields.get("families", "qwen,deepseek,voice").split(",")
    git = ["git", "--git-dir=" + os.environ["FAKE_REMOTE"]]
    if subprocess.run(git + ["cat-file", "-e", "main:.github/workflows/build.yml"], capture_output=True).returncode != 0:
        fail("could not find any workflows named build.yml")
    tree = subprocess.run(git + ["rev-parse", "main:runtime"], capture_output=True, text=True).stdout.strip()
    if tree[:12] != tag[3:]:
        fail("tag %s does not match the pushed runtime/ tree %s" % (tag, tree[:12]))
    runs = load("runs.json", [])
    runs.insert(0, {"databaseId": 1000 + len(runs), "displayTitle": "build " + tag, "status": "queued", "conclusion": "", "families": families})
    save("runs.json", runs)
elif cmd == ["run", "list"]:
    print(json.dumps(load("runs.json", [])))
elif cmd == ["run", "watch"]:
    runs = load("runs.json", [])
    run = [r for r in runs if str(r["databaseId"]) == args[2]][0]
    if run["status"] != "completed":
        success = os.environ.get("FAKE_RUN_RESULT", "success") == "success"
        finish_build(run["displayTitle"][len("build "):], success, run.get("families", ["qwen", "deepseek", "voice"]))
        run["status"], run["conclusion"] = "completed", "success" if success else "failure"
        save("runs.json", runs)
    print("fake watch of run %s: %s" % (args[2], run["conclusion"]))
    sys.exit(0 if run["conclusion"] == "success" else 1)
elif cmd == ["run", "view"]:
    run = [r for r in load("runs.json", []) if str(r["databaseId"]) == args[2]][0]
    if "--log-failed" in args:
        print("fake failed log line")
    else:
        print(json.dumps({"conclusion": run["conclusion"]}))
else:
    fail("fake gh: UNEXPECTED call: " + " ".join(args), 99)
PY
chmod +x "$BIN/gh"
cat > "$BIN/curl" << 'SH'
#!/bin/bash
echo "curl $*" >> "$FAKE_ST/log"
url=""
for a in "$@"; do case "$a" in https://*) url=$a ;; esac; done
[ -f "$FAKE_ST/assets/${url##*/}" ] || exit 22
SH
chmod +x "$BIN/curl"

# ---- a scratch copy of the app repo with runtime/, the schema and the checker ----
mkdir -p "$SRC/vm/lib" "$SRC/manifests"
cp -a "$RT" "$SRC/runtime"
cp "$APP/vm/lib/manifest.py" "$SRC/vm/lib/"
cp "$APP/manifests/runtime.schema.json" "$SRC/manifests/"
GIT=(git -C "$SRC" -c user.name=tester -c user.email=tester@example.invalid -c commit.gpgsign=false)
"${GIT[@]}" init -q -b main
"${GIT[@]}" add -A
"${GIT[@]}" commit -q -m "scratch copy"
TREE=$("${GIT[@]}" rev-parse HEAD:runtime)
T12=${TREE:0:12}
TAG=rt-$T12

reset_world() {
  rm -rf "$ST" "$REMOTE" "$SCRATCH/tmp" "$SRC/manifests/runtime.json"
  mkdir -p "$ST/assets" "$SCRATCH/tmp"
  : > "$ST/log"
  git init -q --bare -b main "$REMOTE"
  EXTRA_ENV=()
  REMOTE_URL=""
}
EXTRA_ENV=()
REMOTE_URL=""
RC=0
run_sync() {
  env PATH="$BIN:$PATH" FAKE_ST="$ST" FAKE_RT="$SRC/runtime" FAKE_REMOTE="$REMOTE" \
    SYNC_REMOTE_URL="${REMOTE_URL:-file://$REMOTE}" SYNC_RETRY_SLEEP=0 SYNC_WATCH_INTERVAL=1 TMPDIR="$SCRATCH/tmp" \
    "${EXTRA_ENV[@]}" bash "$SRC/runtime/sync.sh" "$@" > "$SCRATCH/out.txt" 2>&1 < /dev/null
  RC=$?
}
want_rc() { [ "$RC" = "$2" ] && ok "$1 (exit $RC)" || bad "$1: exit $RC, want $2; $(tail -n 4 "$SCRATCH/out.txt" | tr '\n' ' ')"; }
out_has() { grep -qF -- "$2" "$SCRATCH/out.txt" && ok "$1" || bad "$1: output lacks '$2'; $(tail -n 4 "$SCRATCH/out.txt" | tr '\n' ' ')"; }
out_lacks() { grep -qF -- "$2" "$SCRATCH/out.txt" && bad "$1: output has '$2'" || ok "$1"; }
log_count() { grep -cE -- "$1" "$ST/log"; }
log_is() { [ "$(log_count "$2")" = "$3" ] && ok "$1" || bad "$1: $(log_count "$2") calls matching '$2', want $3"; }
remote_refs() { git --git-dir="$REMOTE" for-each-ref | wc -l; }
rtgit() { git --git-dir="$REMOTE" "$@"; }
DISPATCH="workflow run build.yml -R $REPO --ref main -f tag=$TAG"
WRITES='^(repo create|workflow run|release (create|edit|upload|delete)|api)'

# ---- dry run: tag, files, dispatch line, no write anywhere ----
reset_world
printf '#!/bin/sh\ntouch "%s/push-attempted"\nexit 1\n' "$SCRATCH" > "$REMOTE/hooks/pre-receive"
chmod +x "$REMOTE/hooks/pre-receive"
run_sync --dry-run
want_rc "dry run" 0
out_has "dry run prints the tag" "tag $TAG"
out_has "dry run prints the dispatch command" "would run: gh $DISPATCH"
out_has "dry run lists the workflow copy" ".github/workflows/build.yml"
out_has "dry run lists runtime/versions.env" "runtime/versions.env"
out_has "dry run lists runtime/workflows/build.yml" "runtime/workflows/build.yml"
want_files=$(( $("${GIT[@]}" ls-tree -r --name-only HEAD:runtime | wc -l) + 1 ))
out_has "dry run counts the files to push" "would push $want_files files"
out_has "dry run says it wrote nothing" "nothing was pushed, created or dispatched"
[ "$(grep -vcE '^(repo view|release view)' "$ST/log")" = 0 ] && ok "dry run made only read calls to gh" || bad "dry run gh calls: $(tr '\n' ';' < "$ST/log")"
log_is "dry run made no write call" "$WRITES" 0
[ "$(remote_refs)" = 0 ] && [ ! -e "$SCRATCH/push-attempted" ] && ok "dry run never contacted the remote that rejects pushes" || bad "dry run touched the remote"
REMOTE_URL=/nonexistent/dev-repo.git
run_sync --dry-run
want_rc "dry run with an unreachable remote" 0
REMOTE_URL=""
echo PUBLIC > "$ST/repo"
run_sync --dry-run
out_has "dry run reports an existing repo" "dev repo: ok"
out_has "dry run reports a missing release" "release $TAG: none"

# ---- the tag follows the runtime/ tree and nothing else ----
[ "$TAG" = "rt-$(git -C "$SRC" rev-parse HEAD:runtime | cut -c1-12)" ] && ok "tag is rt- plus the first 12 hex of git rev-parse HEAD:runtime" || bad "tag formula"
echo "outside" > "$SRC/OUTSIDE.txt"
"${GIT[@]}" add OUTSIDE.txt && "${GIT[@]}" commit -q -m "outside runtime/"
run_sync --dry-run
out_has "a commit outside runtime/ keeps the tag" "tag $TAG"
cp "$SRC/runtime/README.md" "$SCRATCH/README.keep"
echo "more" >> "$SRC/runtime/README.md"
"${GIT[@]}" add runtime/README.md && "${GIT[@]}" commit -q -m "change runtime"
run_sync --dry-run
out_lacks "a runtime/ change moves the tag" "tag $TAG"
cp "$SCRATCH/README.keep" "$SRC/runtime/README.md"
"${GIT[@]}" add runtime/README.md && "${GIT[@]}" commit -q -m "revert runtime"
run_sync --dry-run
out_has "reverting the change gives the original tag back" "tag $TAG"

# ---- refusals: dirty tree, missing repo, private repo ----
echo "dirty" >> "$SRC/runtime/README.md"
run_sync --dry-run
want_rc "dry run with a dirty runtime/ still works" 0
out_has "dry run warns about the dirty tree" "uncommitted changes"
reset_world
echo PUBLIC > "$ST/repo"
run_sync --wait
want_rc "a dirty runtime/ is refused" 1
out_has "the refusal names the cause" "uncommitted changes"
log_is "refusal made no write call" "$WRITES" 0
"${GIT[@]}" checkout -q -- runtime/README.md
reset_world
run_sync --wait
want_rc "missing repo without --create" 1
out_has "the refusal names --create" "--create"
log_is "no repo was created" "^repo create" 0
reset_world
echo PRIVATE > "$ST/repo"
run_sync --create --wait
want_rc "private repo" 1
out_has "the refusal says public" "must be public"
[ "$(remote_refs)" = 0 ] && ok "nothing was pushed to the private repo" || bad "pushed to a private repo"

# ---- the full flow: create, push, dispatch, wait, pin, verify ----
reset_world
run_sync --create --wait --verify
want_rc "create, wait, verify" 0
log_is "repo created public once" "^repo create $REPO --public" 1
log_is "one dispatch with the tag" "^$DISPATCH\$" 1
[ "$(rtgit rev-parse main:runtime)" = "$TREE" ] && ok "the dev repo's runtime/ tree equals the source tree" || bad "tree in the dev repo differs"
[ "$(rtgit rev-parse main:.github/workflows/build.yml)" = "$("${GIT[@]}" rev-parse HEAD:runtime/workflows/build.yml)" ] && ok "the workflow sits at .github/workflows/build.yml unchanged" || bad "workflow copy differs"
[ "$(rtgit ls-tree --name-only main | paste -sd,)" = ".github,runtime" ] && ok "the dev repo holds only runtime/ and .github/" || bad "top level of the dev repo: $(rtgit ls-tree --name-only main | paste -sd,)"
[ "$(rtgit rev-list --count main)" = 1 ] && ok "one commit pushed" || bad "commit count $(rtgit rev-list --count main)"
[ "$(rtgit log -1 --format='%an <%ae>|%cn <%ce>')" = "leonxger <leonxger@users.noreply.github.com>|leonxger <leonxger@users.noreply.github.com>" ] && ok "commit is by leonxger with the noreply address" || bad "commit identity: $(rtgit log -1 --format='%an <%ae>|%cn <%ce>')"
[ "$(rtgit log -1 --format=%B | sed '/^$/d' | wc -l)" = 1 ] && ! rtgit log -1 --format=%B | grep -qiE 'co-authored|claude|signed-off' && ok "commit message is one line without trailers" || bad "commit message: $(rtgit log -1 --format=%B)"
log_is "a watch of the run" '^run watch 1000 ' 1
PIN="$SRC/manifests/runtime.json"
[ -f "$PIN" ] && ok "manifests/runtime.json written" || bad "no runtime.json"
python3 -I -B "$SCRATCH/schema_check.py" "$APP" "$PIN" > "$SCRATCH/pin.out" 2>&1 && ok "written runtime.json validates against runtime.schema.json" || bad "runtime.json invalid: $(head -n 3 "$SCRATCH/pin.out" | tr '\n' ' ')"
cat > "$SCRATCH/pin_check.py" << 'PY'
import hashlib, json, os, re, sys
pin_path, st, rt, tag, repo = sys.argv[1:6]
pin = json.load(open(pin_path))
env = dict(l.split("=", 1) for l in open(rt + "/versions.env").read().splitlines() if re.match(r"^[A-Z0-9_]+=", l))
errs = []

def eq(label, got, want):
    if got != want:
        errs.append("%s: got %r, want %r" % (label, got, want))

def series(family):
    return [l.strip() for l in open("%s/patches/%s/series" % (rt, family)) if l.strip() and not l.startswith("#")]

eq("schema", pin["schema"], 1)
eq("tag", pin["tag"], tag)
eq("base_url", pin["base_url"], "https://github.com/%s/releases/download" % repo)
eq("builds", list(pin["builds"]), ["qwen", "deepseek", "voice", "cudart"])
for family, b in pin["builds"].items():
    if "tag" in b:
        errs.append(family + " has its own tag in a full build")
v = pin["builds"]["voice"]
eq("voice audiocpp_tag", v["audiocpp_tag"], env["AUDIOCPP_TAG"])
eq("voice audiocpp_commit", v["audiocpp_commit"], env["AUDIOCPP_COMMIT"])
eq("voice qwentts_commit", v["qwentts_commit"], env["QWENTTS_COMMIT"])
eq("voice num2words", v["num2words"], env["NUM2WORDS_VERSION"])
eq("voice cuda", v["cuda"], "13.0.2")
eq("voice archs", v["archs"], ["80-real", "120a-real"])
eq("voice gpus", v["gpus"], ["G4", "A100", "L4"])
eq("voice file", v["file"], "voice-" + tag[3:] + ".tar.gz")
for family, tg, cm, archs, gpus in (("qwen", env["QWEN_TAG"], env["QWEN_COMMIT"], ["80-real", "120a-real"], ["G4", "A100"]), ("deepseek", env["DS_TAG"], env["DS_COMMIT"], ["120a-real"], ["G4"])):
    b = pin["builds"][family]
    eq(family + " llama_tag", b["llama_tag"], tg)
    eq(family + " commit", b["commit"], cm)
    eq(family + " patches", b["patches"], series(family))
    eq(family + " cuda", b["cuda"], "13.0.2")
    eq(family + " archs", b["archs"], archs)
    eq(family + " gpus", b["gpus"], gpus)
for family, b in pin["builds"].items():
    data = open(os.path.join(st, "assets", b["file"]), "rb").read()
    eq(family + " size", b["size"], len(data))
    eq(family + " sha256", b["sha256"], hashlib.sha256(data).hexdigest())
    if not b["file"].endswith("-" + tag[3:] + ".tar.gz"):
        errs.append(family + " file does not end with the tree id")
eq("qwen file", pin["builds"]["qwen"]["file"].split("-")[:3], ["llama", "qwen", env["QWEN_TAG"]])
eq("deepseek file", pin["builds"]["deepseek"]["file"].split("-")[:3], ["llama", "deepseek", env["DS_TAG"] + "p" + str(len(series("deepseek")))])
print("\n".join(errs) or "OK")
PY
pin_out=$(python3 -I -B "$SCRATCH/pin_check.py" "$PIN" "$ST" "$SRC/runtime" "$TAG" "$REPO" 2>&1)
[ "$pin_out" = OK ] && ok "runtime.json fields come from versions.env, the series, the sizes and the .sha256 files of the release" || bad "runtime.json content: $pin_out"
for fam in qwen deepseek voice cudart; do
  file=$(python3 -I -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["builds"][sys.argv[2]]["file"])' "$PIN" "$fam")
  log_is "verify streamed $fam through gh release download" "^release download $TAG -R $REPO -p $file -O -\$" 1
done
out_has "verify says passed" "verify passed for $TAG"
log_is "verify probed each asset anonymously" '^curl .*-r 0-0' 4
[ "$(ls "$SCRATCH/tmp" | wc -l)" = 0 ] && ok "sync.sh left no scratch directory behind" || bad "scratch left: $(ls "$SCRATCH/tmp")"

# ---- a published release ends early; --force builds again ----
pin_sum=$(sha256sum "$PIN" | cut -c1-64)
head_before=$(rtgit rev-parse main)
run_sync --create --wait --verify
want_rc "second run on a published release" 0
out_has "it says the release is published" "already published"
log_is "no further write call" "$WRITES" 2
log_is "still one dispatch" "^$DISPATCH\$" 1
[ "$(rtgit rev-parse main)" = "$head_before" ] && ok "no push on a published release" || bad "remote moved"
[ "$(sha256sum "$PIN" | cut -c1-64)" = "$pin_sum" ] && ok "pin untouched" || bad "pin changed"
out_has "verify still runs on early exit" "verify passed"
rm -f "$PIN"
run_sync
want_rc "published release with a missing pin" 0
[ "$(sha256sum "$PIN" | cut -c1-64)" = "$pin_sum" ] && ok "a missing pin is rewritten without a build" || bad "pin not healed"
log_is "healing the pin dispatched nothing" "^$DISPATCH\$" 1
run_sync --wait --force
want_rc "--force on a published release" 0
out_lacks "--force does not stop early" "already published"
log_is "--force dispatches again" "^$DISPATCH\$" 2
out_has "--force finds the tree already pushed" "nothing to commit"
[ "$(rtgit rev-parse main)" = "$head_before" ] && ok "--force pushes no new commit for an unchanged tree" || bad "remote moved on --force"

# ---- verify alone ----
run_sync --verify
want_rc "--verify alone" 0
out_has "--verify alone passes" "verify passed for $TAG"
log_is "--verify alone wrote nothing" "$WRITES" 3
qfile=$(python3 -I -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["builds"]["qwen"]["file"])' "$PIN")
cp "$ST/assets/$qfile" "$SCRATCH/asset.keep"
python3 -I -B -c 'import sys; p = sys.argv[1]; b = bytearray(open(p, "rb").read()); b[10] ^= 1; open(p, "wb").write(b)' "$ST/assets/$qfile"
run_sync --verify
want_rc "--verify with a flipped byte in an asset" 1
out_has "the failure names the file" "$qfile"
out_has "the failure says FAILED" "verify FAILED"
cp "$SCRATCH/asset.keep" "$ST/assets/$qfile"
cp "$PIN" "$SCRATCH/pin.keep"
python3 -I -B -c 'import json, sys; d = json.load(open(sys.argv[1])); d["builds"]["cudart"]["sha256"] = "0" * 64; json.dump(d, open(sys.argv[1], "w"), indent=2)' "$PIN"
run_sync --verify
want_rc "--verify with a wrong sha256 in the pin" 1
cp "$SCRATCH/pin.keep" "$PIN"
python3 -I -B -c 'import json, sys; d = json.load(open(sys.argv[1])); d["tag"] = "rt-nothex"; json.dump(d, open(sys.argv[1], "w"), indent=2)' "$PIN"
run_sync --verify
want_rc "--verify with a pin that breaks the schema" 1
out_has "the schema failure is named" "schema:"
cp "$SCRATCH/pin.keep" "$PIN"
python3 -I -B -c 'import json, sys; d = json.load(open(sys.argv[1])); d["base_url"] = "https://example.invalid/x"; json.dump(d, open(sys.argv[1], "w"), indent=2)' "$PIN"
run_sync --verify
want_rc "--verify with a foreign base_url" 1
out_has "the base_url failure is named" "base_url"
rm -f "$PIN"
run_sync --verify
want_rc "--verify without a pin" 1
out_has "the missing pin is named" "does not exist"

# ---- a failed run, then a good one ----
reset_world
EXTRA_ENV=(FAKE_RUN_RESULT=failure)
run_sync --create --wait
want_rc "failed build run" 1
out_has "the failed log is shown" "fake failed log line"
[ ! -e "$PIN" ] && ok "no pin after a failed run" || bad "pin written after a failed run"
EXTRA_ENV=()
run_sync --create --wait
want_rc "retry after a failed run" 0
log_is "the retry dispatched again (a finished run does not block)" "^$DISPATCH\$" 2
[ -f "$PIN" ] && ok "pin written after the retry" || bad "no pin after the retry"

# ---- a run that is already going is attached, not doubled ----
reset_world
echo PUBLIC > "$ST/repo"
printf '[{"databaseId": 777, "displayTitle": "build %s", "status": "in_progress", "conclusion": ""}]\n' "$TAG" > "$ST/runs.json"
run_sync --wait
want_rc "attach to a running build" 0
out_has "it says it attaches" "run 777 for $TAG is already going"
log_is "no dispatch while a run is going" "^workflow run" 0
log_is "it watched run 777" '^run watch 777 ' 1
[ -f "$PIN" ] && ok "pin written after attaching" || bad "no pin after attaching"

# ---- dispatch retries, no --wait, drafts and odd releases ----
reset_world
echo PUBLIC > "$ST/repo"
echo 2 > "$ST/dispatch_fails"
run_sync --wait
want_rc "dispatch that fails twice" 0
log_is "three dispatch attempts" '^workflow run' 3
reset_world
run_sync --create
want_rc "no --wait" 0
out_has "it explains how to continue" "not waiting"
log_is "one dispatch" "^$DISPATCH\$" 1
log_is "no watch" '^run watch' 0
[ ! -e "$PIN" ] && ok "no pin without --wait" || bad "pin without --wait"
reset_world
EXTRA_ENV=(FAKE_NO_PUBLISH=1)
run_sync --create --wait
want_rc "run succeeded but the release is still a draft" 1
out_has "the draft is named" "is draft"
[ ! -e "$PIN" ] && ok "no pin for a draft" || bad "pin for a draft"
reset_world
EXTRA_ENV=(FAKE_SKIP_ASSET=cudart)
run_sync --create --wait
want_rc "release without the cudart asset" 1
out_has "the missing asset is named" "missing ['cudart-"
[ ! -e "$PIN" ] && ok "no pin for an incomplete release" || bad "pin for an incomplete release"

# ---- a voice-only build merges into the pin and leaves every other entry as it was ----
reset_world
run_sync --create --wait
want_rc "full build before the voice-only one" 0
cp "$PIN" "$SCRATCH/pin.full"
run_sync --wait --only=voice
want_rc "--only=voice without a change of the tree (release already published)" 0
out_has "it ends early on the published release" "already published"
echo "voice only change" >> "$SRC/runtime/README.md"
"${GIT[@]}" add runtime/README.md && "${GIT[@]}" commit -q -m "runtime change for the voice-only build"
TAG2=rt-$("${GIT[@]}" rev-parse HEAD:runtime | cut -c1-12)
[ "$TAG2" != "$TAG" ] && ok "the runtime change moved the tag" || bad "tag did not move"
run_sync --dry-run --only=voice
out_has "dry run names the families of the dispatch" "-f tag=$TAG2 -f families=voice"
run_sync --wait --verify --only=voice
want_rc "--only=voice after a runtime change" 0
log_is "one dispatch with the families input" "^workflow run build.yml -R $REPO --ref main -f tag=$TAG2 -f families=voice\$" 1
cat > "$SCRATCH/pin_merge.py" << 'PY'
import json, sys
old, new, tag, tag2 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
errs = []
if new["tag"] != tag:
    errs.append("top-level tag moved to " + new["tag"])
for family in ("qwen", "deepseek", "cudart"):
    if new["builds"][family] != old["builds"][family]:
        errs.append(family + " entry changed")
if list(new["builds"]) != ["qwen", "deepseek", "voice", "cudart"]:
    errs.append("order " + str(list(new["builds"])))
v = new["builds"]["voice"]
if v.get("tag") != tag2 or v["file"] != "voice-%s.tar.gz" % tag2[3:]:
    errs.append("voice entry has tag %r file %r" % (v.get("tag"), v["file"]))
if v["sha256"] == old["builds"]["voice"]["sha256"]:
    errs.append("voice entry kept the old sha256")
print("\n".join(errs) or "OK")
PY
merge_out=$(python3 -I -B "$SCRATCH/pin_merge.py" "$SCRATCH/pin.full" "$PIN" "$TAG" "$TAG2" 2>&1)
[ "$merge_out" = OK ] && ok "qwen, deepseek and cudart keep their entries byte for byte; voice has its own tag and file" || bad "merged pin: $merge_out"
python3 -I -B "$SCRATCH/schema_check.py" "$APP" "$PIN" > "$SCRATCH/pin.out" 2>&1 && ok "the merged pin validates against runtime.schema.json" || bad "merged pin invalid: $(head -n 3 "$SCRATCH/pin.out" | tr '\n' ' ')"
diff <(python3 -I -B -c 'import json, sys; d = json.load(open(sys.argv[1])); [d["builds"].pop(k) for k in ("voice",)]; print(json.dumps(d, indent=2))' "$SCRATCH/pin.full") <(python3 -I -B -c 'import json, sys; d = json.load(open(sys.argv[1])); [d["builds"].pop(k) for k in ("voice",)]; print(json.dumps(d, indent=2))' "$PIN") > /dev/null && ok "without the voice entry the pin is identical to the one before" || bad "the merge changed more than the voice entry"
log_is "verify streamed the voice asset of the new release" "^release download $TAG2 -R $REPO -p voice-${TAG2#rt-}.tar.gz -O -\$" 1
log_is "verify streamed the Qwen asset of the old release" "^release download $TAG -R $REPO -p llama-qwen-.* -O -\$" 2
out_has "verify passes with builds of two releases" "verify passed"
pin_sum=$(sha256sum "$PIN" | cut -c1-64)
run_sync --wait --only=voice
want_rc "a second --only=voice run" 0
out_has "it ends early on the published release" "already published"
log_is "no second dispatch" "^workflow run build.yml -R $REPO --ref main -f tag=$TAG2 " 1
[ "$(sha256sum "$PIN" | cut -c1-64)" = "$pin_sum" ] && ok "the pin is untouched by the second run" || bad "pin changed"
rm -f "$PIN"
run_sync --wait --only=voice
want_rc "--only=voice with a published release and no pin" 1
out_has "a partial build needs the pin to merge into" "needs the existing pin"
cp "$SCRATCH/pin.full" "$PIN"
run_sync --only=voice
want_rc "--only=voice heals a pin that lacks the new voice entry" 0
python3 -I -B "$SCRATCH/pin_merge.py" "$SCRATCH/pin.full" "$PIN" "$TAG" "$TAG2" | grep -qx OK && ok "the healed pin has the voice entry of the new release and the old rest" || bad "healed pin wrong"
run_sync --wait --only=bogus
want_rc "--only with an unknown family" 1
out_has "the refusal names the families" "--only takes qwen, deepseek and voice"
run_sync --wait --only=
want_rc "--only with an empty list" 1

finish test_sync.sh
