#!/bin/bash
# Pushes the runtime/ tree to the public dev repo, dispatches the CUDA build and pins the release in manifests/runtime.json: runtime/sync.sh [--create] [--dry-run] [--wait] [--verify] [--force] [--only=FAMILIES]; --verify alone re-checks the pin.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=runtime/build/common.sh
. "$HERE/build/common.sh"

BRANCH=main
WORKFLOW=build.yml
AUTHOR_NAME=leonxger
AUTHOR_EMAIL=leonxger@users.noreply.github.com
# L4 and H100 join after a decision adds 89-real and 90-real
QWEN_GPUS="G4 A100"
DS_GPUS="G4"
# the CUDA archs run on the L4 (sm_89) by binary compatibility of the sm_80 code
VOICE_GPUS="G4 A100 L4"
ALL_FAMILIES="qwen deepseek voice"

log() { echo "sync: $*"; }
warn() { echo "sync: $*" >&2; }
die() { echo "sync: FAILED: $*" >&2; exit 1; }
usage() {
  cat >&2 << 'EOF'
usage: runtime/sync.sh [--create] [--dry-run] [--wait] [--verify] [--force] [--only=FAMILIES]
  --create   make the public dev repo when it does not exist
  --dry-run  print the tag, the files to push and the dispatch command; write nothing anywhere
  --wait     watch the build run, then pin the release in manifests/runtime.json
  --verify   stream every pinned asset through sha256sum and compare (alone: check the existing pin)
  --force    build again although the release of this tree is already published
  --only=L   build only these families (comma separated: qwen, deepseek, voice) and merge them into the existing pin; the others keep the pins of their own release
env: SYNC_REMOTE_URL (git remote of the dev repo), SYNC_WAIT_S (cap of --wait, default 20400), COLAB_HUB_RUNTIME_BASE (base_url of the pin)
EOF
  exit "${1:-2}"
}

CREATE=0 DRY=0 WAIT=0 VERIFY=0 FORCE=0 ONLY=""
for a in "$@"; do
  case "$a" in
    --only=*) ONLY=${a#--only=} ;;
    --create) CREATE=1 ;;
    --dry-run) DRY=1 ;;
    --wait) WAIT=1 ;;
    --verify) VERIFY=1 ;;
    --force) FORCE=1 ;;
    -h | --help) usage 0 ;;
    *) warn "unknown option $a"; usage 2 ;;
  esac
done
FAMS="" FULL=1
if [ -n "$ONLY" ]; then
  [[ "$ONLY" =~ ^(qwen|deepseek|voice)(,(qwen|deepseek|voice))*$ ]] || die "--only takes qwen, deepseek and voice, comma separated (got '$ONLY')"
  for f in $ALL_FAMILIES; do [[ ",$ONLY," == *,$f,* ]] && FAMS="$FAMS $f"; done
  FAMS=${FAMS# }
else
  FAMS=$ALL_FAMILIES
fi
[ "$FAMS" = "$ALL_FAMILIES" ] || FULL=0
FAMS_CSV=${FAMS// /,}
VERIFY_ONLY=0
[ "$VERIFY" = 1 ] && [ "$CREATE$DRY$WAIT$FORCE" = 0000 ] && VERIFY_ONLY=1

need() { command -v "$1" > /dev/null || die "$1 is not installed"; }
need git
need python3
need tar
need timeout
ROOT=$(git -C "$HERE" rev-parse --show-toplevel 2> /dev/null) || die "$HERE is not inside a git checkout"
[ "$HERE" = "$ROOT/runtime" ] || die "sync.sh must sit in <repo>/runtime, it is in $HERE"
[[ "$RUNTIME_OWNER" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] || die "RUNTIME_OWNER in versions.env is not a GitHub user name"
[[ "$RUNTIME_REPO" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUNTIME_REPO in versions.env is not a repository name"
REPO="$RUNTIME_OWNER/$RUNTIME_REPO"
REMOTE_URL=${SYNC_REMOTE_URL:-https://github.com/$REPO.git}
BASE_URL=${COLAB_HUB_RUNTIME_BASE:-https://github.com/$REPO/releases/download}
BASE_URL=${BASE_URL%/}
PIN="$ROOT/manifests/runtime.json"
SCHEMA="$ROOT/manifests/runtime.schema.json"
export GIT_TERMINAL_PROMPT=0

TREE=$(git -C "$ROOT" rev-parse HEAD:runtime 2> /dev/null) || die "HEAD has no runtime/ tree, commit it first"
TREE12=${TREE:0:12}
TAG=rt-$TREE12

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/runtime-sync.XXXXXX") || die "no scratch directory"
trap 'rm -rf "$SCRATCH"' EXIT

gh_() { timeout "${SYNC_GH_TIMEOUT:-120}" gh "$@"; }
# one JSON key of stdin, booleans as True or False
py_get() { python3 -I -B -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

# the files of the HEAD tree, one per line, plus the workflow copy
files_to_push() {
  git -C "$ROOT" ls-tree -r --name-only HEAD:runtime | sed 's#^#runtime/#'
  echo ".github/workflows/$WORKFLOW"
}

# sets REPO_STATE to ok, missing, private or unknown (REPO_NOTE says why)
repo_check() {
  local out vis
  REPO_NOTE=""
  if out=$(gh_ repo view "$REPO" --json visibility 2> "$SCRATCH/err"); then
    vis=$(py_get visibility <<< "$out") || vis=unreadable
    case "$vis" in
      PUBLIC) REPO_STATE=ok ;;
      PRIVATE | INTERNAL) REPO_STATE=private; REPO_NOTE="$REPO is $vis" ;;
      *) REPO_STATE=unknown; REPO_NOTE="unreadable answer of gh repo view" ;;
    esac
  elif grep -qiE 'could not resolve to a repository|not found|404' "$SCRATCH/err"; then
    REPO_STATE=missing
  else
    REPO_STATE=unknown
    REPO_NOTE=$(tail -n 1 "$SCRATCH/err")
  fi
}

# prints published, draft or none; returns 1 on any other gh failure
release_state() {
  local out
  if out=$(gh_ release view "$TAG" -R "$REPO" --json isDraft 2> "$SCRATCH/err"); then
    case "$(py_get isDraft <<< "$out")" in
      True) echo draft ;;
      False) echo published ;;
      *) return 1 ;;
    esac
  elif grep -qiE 'release not found|could not resolve to a repository|not found|404' "$SCRATCH/err"; then
    echo none
  else
    tail -n 1 "$SCRATCH/err" >&2
    return 1
  fi
}

# clones the dev repo, replaces its content by the HEAD tree of runtime/ plus the workflow, commits and pushes when something changed
push_tree() {
  local d="$SCRATCH/repo" i committed=0 got
  for i in 1 2 3; do
    rm -rf "$d"
    timeout 300 git clone -q "$REMOTE_URL" "$d" 2> "$SCRATCH/clone.err" && break
    [ "$i" = 3 ] && die "cannot clone $REMOTE_URL: $(tail -n 1 "$SCRATCH/clone.err")"
    sleep "${SYNC_RETRY_SLEEP:-5}"
  done
  if git -C "$d" rev-parse -q --verify HEAD > /dev/null; then
    git -C "$d" checkout -q -B "$BRANCH" || die "cannot switch the clone to $BRANCH"
  else
    git -C "$d" symbolic-ref HEAD "refs/heads/$BRANCH" || die "cannot name the first branch"
  fi
  find "$d" -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +
  mkdir -p "$d/runtime" "$d/.github/workflows"
  git -C "$ROOT" archive --format=tar HEAD:runtime | tar -x -C "$d/runtime" || die "cannot unpack the runtime/ tree"
  cp "$d/runtime/workflows/$WORKFLOW" "$d/.github/workflows/$WORKFLOW" || die "runtime/workflows/$WORKFLOW is missing"
  git -C "$d" add -A
  if git -C "$d" diff --cached --quiet; then
    log "the dev repo already holds this tree, nothing to commit"
  else
    GIT_AUTHOR_NAME=$AUTHOR_NAME GIT_AUTHOR_EMAIL=$AUTHOR_EMAIL GIT_COMMITTER_NAME=$AUTHOR_NAME GIT_COMMITTER_EMAIL=$AUTHOR_EMAIL \
      git -C "$d" -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q -m "Runtime sources $TAG" || die "commit in the scratch clone failed"
    committed=1
  fi
  got=$(git -C "$d" rev-parse HEAD:runtime) || die "the scratch clone has no runtime/ tree"
  [ "$got" = "$TREE" ] || die "runtime/ tree changed in transit: HEAD has $TREE, the copy has $got"
  if [ "$committed" = 1 ]; then
    timeout 300 git -C "$d" -c core.hooksPath=/dev/null -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
      push -q origin "HEAD:refs/heads/$BRANCH" || die "push to $REMOTE_URL failed"
    log "pushed $(git -C "$d" rev-parse --short HEAD) to $BRANCH of $REPO"
  fi
}

# prints "ACTIVE_ID MAX_ID" of the runs titled "build <tag>": ACTIVE_ID is the newest queued or running one or -, MAX_ID the newest of any state or 0
runs_info() {
  gh_ run list -R "$REPO" --workflow "$WORKFLOW" --limit 30 --json databaseId,displayTitle,status > "$SCRATCH/runs.json" 2> "$SCRATCH/err" || return 1
  python3 -I -B - "$SCRATCH/runs.json" "build $TAG" << 'PY'
import json, sys
runs = [r for r in json.load(open(sys.argv[1])) if r["displayTitle"] == sys.argv[2]]
active = [r["databaseId"] for r in runs if r["status"] in ("queued", "in_progress", "waiting", "requested", "pending")]
print(max(active) if active else "-", max([r["databaseId"] for r in runs], default=0))
PY
}

dispatch() {
  local i extra=()
  [ "$FULL" = 1 ] || extra=(-f "families=$FAMS_CSV")
  for i in 1 2 3 4 5 6 7 8; do
    gh_ workflow run "$WORKFLOW" -R "$REPO" --ref "$BRANCH" -f "tag=$TAG" "${extra[@]}" 2> "$SCRATCH/err" && return 0
    warn "dispatch attempt $i failed: $(tail -n 1 "$SCRATCH/err")"
    sleep "${SYNC_RETRY_SLEEP:-5}"
  done
  die "could not dispatch $WORKFLOW"
}

# watches run ID to its end
watch_run() {
  local id=$1 rc concl
  log "watching run $id (cap ${SYNC_WAIT_S:-20400} s)"
  timeout "${SYNC_WAIT_S:-20400}" gh run watch "$id" -R "$REPO" --exit-status --interval "${SYNC_WATCH_INTERVAL:-60}"
  rc=$?
  [ "$rc" != 124 ] || die "run $id is still going after ${SYNC_WAIT_S:-20400} s; run sync.sh --wait again to attach to it"
  concl=$(gh_ run view "$id" -R "$REPO" --json conclusion 2> /dev/null | py_get conclusion) || concl=unknown
  if [ "$rc" != 0 ] || [ "$concl" != success ]; then
    warn "run $id ended with '$concl' (watch exit $rc); the end of its failed log:"
    gh_ run view "$id" -R "$REPO" --log-failed 2> /dev/null | tail -n 60 >&2
    die "the build run failed, the release stays a draft"
  fi
  log "run $id succeeded"
}

# the expected tarballs of this tree for the built families, one "family file" per line: qwen, deepseek, voice, then cudart (only the Qwen job packs it)
asset_pairs() {
  local f
  for f in $FAMS; do
    echo "$f $(TREE12=$TREE12 bash "$HERE/build/$f.sh" --list-package | sed -n 1p)"
  done
  case " $FAMS " in *" qwen "*) echo "cudart cudart-$(rt_cuda_version)-x64-$TREE12.tar.gz" ;; esac
}

# the CUDA architectures of a family as the CMake line names them, space separated
archs_of() {
  local arg=""
  [ "$1" != voice ] || arg=audiocpp
  bash "$HERE/build/$1.sh" --print-cmake $arg | grep -o 'CMAKE_CUDA_ARCHITECTURES=[^ ]*' | head -n 1 | cut -d= -f2 | tr -d '"' | tr ';' ' '
}

# validates FILE against manifests/runtime.schema.json with the checker of vm/lib/manifest.py
schema_check() {
  [ -f "$ROOT/vm/lib/manifest.py" ] || die "vm/lib/manifest.py (the schema checker) is missing"
  python3 -I -B - "$ROOT/vm/lib" "$SCHEMA" "$1" << 'PY' || return 1
import json, sys
sys.path.insert(0, sys.argv[1])
import manifest
errors = manifest.check_schema(manifest.load_json(sys.argv[3]), manifest.load_json(sys.argv[2]))
for path, message in errors:
    print("sync: schema: %s: %s" % (path, message), file=sys.stderr)
sys.exit(1 if errors else 0)
PY
}

# true when every built family's pin entry (cudart too with qwen) names TAG as its release
pin_current() {
  [ -f "$PIN" ] || return 1
  python3 -I -B - "$PIN" "$TAG" "$FAMS" << 'PY' 2> /dev/null
import json, sys
d = json.load(open(sys.argv[1]))
want = sys.argv[3].split() + (["cudart"] if "qwen" in sys.argv[3].split() else [])
for f in want:
    b = d["builds"].get(f)
    if b is None or b.get("tag", d["tag"]) != sys.argv[2]:
        sys.exit(1)
PY
}

# writes manifests/runtime.json from the published release's asset sizes and .sha256 files; a partial build is merged into the existing pin and keeps every other entry as it is
write_pin() {
  local rel="$SCRATCH/release.json" shadir="$SCRATCH/sha" pairs n out
  gh_ release view "$TAG" -R "$REPO" --json isDraft,assets > "$rel" 2> "$SCRATCH/err" || die "cannot read release $TAG: $(tail -n 1 "$SCRATCH/err")"
  [ "$(py_get isDraft < "$rel")" = False ] || die "release $TAG is still a draft, so the build did not publish it"
  mkdir -p "$shadir"
  pairs=$(asset_pairs) || die "cannot name the expected assets"
  while read -r _ n; do
    [[ "$n" =~ ^[A-Za-z0-9._-]+\.tar\.gz$ ]] || die "odd asset name '$n'"
  done <<< "$pairs"
  python3 -I -B - "$rel" "$pairs" << 'PY' || die "the release assets differ from the expected set"
import json, sys
assets = {a["name"]: a for a in json.load(open(sys.argv[1]))["assets"]}
names = [l.split(" ", 1)[1] for l in sys.argv[2].splitlines() if l]
want = set(names) | {n + ".sha256" for n in names}
missing, extra = sorted(want - set(assets)), sorted(set(assets) - want)
bad = sorted(n for n, a in assets.items() if a.get("state", "uploaded") != "uploaded")
if missing or extra or bad:
    sys.exit("sync: release assets: missing %s, unexpected %s, not uploaded %s" % (missing or "none", extra or "none", bad or "none"))
PY
  while read -r _ n; do
    timeout "${SYNC_DOWNLOAD_TIMEOUT:-1800}" gh release download "$TAG" -R "$REPO" -p "$n.sha256" -O - < /dev/null > "$shadir/$n.sha256" 2> "$SCRATCH/err" || die "cannot read $n.sha256: $(tail -n 1 "$SCRATCH/err")"
  done <<< "$pairs"
  out="$SCRATCH/runtime.json"
  RT_DIR="$HERE" SHA_DIR="$shadir" REL_JSON="$rel" OUT="$out" OLD_PIN="$PIN" TAG="$TAG" TREE12="$TREE12" BASE_URL="$BASE_URL" FAMS="$FAMS" FILES="$pairs" \
    CUDA_VERSION="$(rt_cuda_version)" QWEN_TAG="$QWEN_TAG" QWEN_COMMIT="$QWEN_COMMIT" DS_TAG="$DS_TAG" DS_COMMIT="$DS_COMMIT" \
    AUDIOCPP_TAG="$AUDIOCPP_TAG" AUDIOCPP_COMMIT="$AUDIOCPP_COMMIT" QWENTTS_COMMIT="$QWENTTS_COMMIT" NUM2WORDS_VERSION="$NUM2WORDS_VERSION" \
    QWEN_GPUS="$QWEN_GPUS" DS_GPUS="$DS_GPUS" VOICE_GPUS="$VOICE_GPUS" QWEN_ARCHS="$(archs_of qwen)" DS_ARCHS="$(archs_of deepseek)" VOICE_ARCHS="$(archs_of voice)" \
    python3 -I -B - << 'PY' || die "cannot build the pin"
import json, os, re, sys
e = os.environ
assets = {a["name"]: a for a in json.load(open(e["REL_JSON"]))["assets"]}
files = dict(l.split(" ", 1) for l in e["FILES"].splitlines() if l)
built = e["FAMS"].split()
order = ["qwen", "deepseek", "voice", "cudart"]

def sha_of(file):
    text = open(os.path.join(e["SHA_DIR"], file + ".sha256")).read()
    m = re.fullmatch(r"([0-9a-f]{64})  (\S+)\n?", text)
    if not m or m.group(2) != file:
        sys.exit("%s.sha256 is not '<sha256>  %s'" % (file, file))
    return m.group(1)

def series(family):
    with open(os.path.join(e["RT_DIR"], "patches", family, "series")) as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]

def entry(family, **fields):
    file = files[family]
    if not file.endswith("-" + e["TREE12"] + ".tar.gz"):
        sys.exit("%s does not end with the tree id %s" % (file, e["TREE12"]))
    return {**fields, "file": file, "size": assets[file]["size"], "sha256": sha_of(file)}

new = {}
if "qwen" in built:
    new["qwen"] = entry("qwen", llama_tag=e["QWEN_TAG"], commit=e["QWEN_COMMIT"], patches=series("qwen"), cuda=e["CUDA_VERSION"], archs=e["QWEN_ARCHS"].split(), gpus=e["QWEN_GPUS"].split())
    new["cudart"] = entry("cudart")
if "deepseek" in built:
    new["deepseek"] = entry("deepseek", llama_tag=e["DS_TAG"], commit=e["DS_COMMIT"], patches=series("deepseek"), cuda=e["CUDA_VERSION"], archs=e["DS_ARCHS"].split(), gpus=e["DS_GPUS"].split())
if "voice" in built:
    new["voice"] = entry("voice", audiocpp_tag=e["AUDIOCPP_TAG"], audiocpp_commit=e["AUDIOCPP_COMMIT"], qwentts_commit=e["QWENTTS_COMMIT"], num2words=e["NUM2WORDS_VERSION"], cuda=e["CUDA_VERSION"], archs=e["VOICE_ARCHS"].split(), gpus=e["VOICE_GPUS"].split())

if set(built) == {"qwen", "deepseek", "voice"}:
    doc = {"schema": 1, "base_url": e["BASE_URL"], "tag": e["TAG"], "builds": {}}
else:
    if not os.path.exists(e["OLD_PIN"]):
        sys.exit("a partial build needs the existing pin to merge into; build every family once first")
    doc = json.load(open(e["OLD_PIN"]))
    if doc["base_url"] != e["BASE_URL"]:
        sys.exit("the pin's base_url is %s, expected %s" % (doc["base_url"], e["BASE_URL"]))
    for family, b in new.items():
        if e["TAG"] != doc["tag"]:
            new[family] = {"tag": e["TAG"], **b}
builds = {**doc["builds"], **new}
doc["builds"] = {k: builds[k] for k in order if k in builds}
with open(e["OUT"], "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
PY
  schema_check "$out" || die "the new pin does not match $SCHEMA"
  mkdir -p "$ROOT/manifests"
  if [ -f "$PIN" ] && cmp -s "$out" "$PIN"; then
    log "manifests/runtime.json already pins $TAG"
  else
    cp "$out" "$PIN.new" && mv -f "$PIN.new" "$PIN" || die "cannot write $PIN"
    log "wrote manifests/runtime.json for $TAG"
  fi
}

# re-checks the pin against its releases: schema, published, sizes, an anonymous GET, and every asset streamed through sha256sum
verify_pin() {
  local rows ptag pbase fam btag file size sha rel got seen=" "
  need curl
  need sha256sum
  [ -f "$PIN" ] || die "$PIN does not exist; run runtime/sync.sh --wait first"
  schema_check "$PIN" || die "$PIN does not match $SCHEMA"
  rows=$(python3 -I -B - "$PIN" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("pin", d["tag"], d["base_url"])
for family, b in d["builds"].items():
    print(family, b.get("tag", d["tag"]), b["file"], b["size"], b["sha256"])
PY
  ) || die "cannot read $PIN"
  read -r _ ptag pbase <<< "$(sed -n 1p <<< "$rows")"
  [ "$pbase" = "$BASE_URL" ] || die "the pin's base_url is $pbase, expected $BASE_URL"
  grep -qE "^[a-z]+ $TAG " <<< "$rows" || warn "no pinned build comes from $TAG, HEAD's runtime/ tree; verifying the pin"
  while read -r fam btag file size sha <&3; do
    rel="$SCRATCH/verify-release-$btag.json"
    if [[ "$seen" != *" $btag "* ]]; then
      gh_ release view "$btag" -R "$REPO" --json isDraft,assets > "$rel" 2> "$SCRATCH/err" || die "cannot read release $btag: $(tail -n 1 "$SCRATCH/err")"
      [ "$(py_get isDraft < "$rel")" = False ] || die "release $btag is a draft"
      seen="$seen$btag "
    fi
    python3 -I -B - "$rel" "$file" "$size" << 'PY' || die "release asset $file does not match the pin (size or the .sha256 asset)"
import json, sys
assets = {a["name"]: a for a in json.load(open(sys.argv[1]))["assets"]}
name, size = sys.argv[2], int(sys.argv[3])
sys.exit(0 if name in assets and name + ".sha256" in assets and assets[name]["size"] == size else 1)
PY
    curl -fsSL --proto '=https' --proto-redir '=https' --max-time 120 -r 0-0 -o /dev/null "$pbase/$btag/$file" || die "anonymous GET of $pbase/$btag/$file failed"
    got=$(set -o pipefail; timeout "${SYNC_DOWNLOAD_TIMEOUT:-1800}" gh release download "$btag" -R "$REPO" -p "$file" -O - | sha256sum | cut -d ' ' -f 1) || die "download of $file failed"
    if [ "$got" = "$sha" ]; then
      log "verify ok: $fam $file $size bytes sha256 ${sha:0:12}"
    else
      die "verify FAILED: $file has sha256 $got, the pin says $sha"
    fi
  done 3< <(sed 1d <<< "$rows")
  log "verify passed for $ptag"
}

print_plan() {
  local n bytes
  n=$(files_to_push | wc -l)
  bytes=$(git -C "$ROOT" ls-tree -r -l HEAD:runtime | awk '{s += $4} END {print s + 0}')
  log "tag $TAG (runtime/ tree $TREE of HEAD $(git -C "$ROOT" rev-parse --short HEAD))"
  log "dev repo $REPO (public, development only), branch $BRANCH, remote $REMOTE_URL"
  log "would push $n files, $bytes bytes in runtime/, as $AUTHOR_NAME <$AUTHOR_EMAIL>, no trailers:"
  files_to_push | sed 's/^/  /'
  if [ "$FULL" = 1 ]; then
    log "would run: gh workflow run $WORKFLOW -R $REPO --ref $BRANCH -f tag=$TAG"
  else
    log "would run: gh workflow run $WORKFLOW -R $REPO --ref $BRANCH -f tag=$TAG -f families=$FAMS_CSV"
  fi
}

if [ "$VERIFY_ONLY" = 1 ]; then
  need gh
  verify_pin
  exit 0
fi

if [ "$DRY" = 1 ]; then
  print_plan
  if command -v gh > /dev/null; then
    repo_check
    log "dev repo: $REPO_STATE"
    case "$REPO_STATE" in
      ok) log "release $TAG: $(release_state 2> /dev/null || echo unknown)" ;;
      missing) log "release $TAG: none" ;;
      *) log "release $TAG: unknown" ;;
    esac
  fi
  [ -z "$(git -C "$ROOT" status --porcelain -- runtime)" ] || warn "runtime/ has uncommitted changes; a real run refuses them"
  log "dry run: nothing was pushed, created or dispatched"
  exit 0
fi

need gh
[ -z "$(git -C "$ROOT" status --porcelain -- runtime)" ] || die "runtime/ has uncommitted changes; commit them, the tag follows HEAD's tree"
log "tag $TAG for $REPO"
repo_check
case "$REPO_STATE" in
  private) die "$REPO_NOTE; the dev repo must be public (a private repo bills Actions minutes)" ;;
  unknown) die "gh repo view $REPO: $REPO_NOTE" ;;
esac
if [ "$REPO_STATE" = missing ]; then
  [ "$CREATE" = 1 ] || die "$REPO does not exist; run with --create to make it (public, development only)"
  gh_ repo create "$REPO" --public --description "Development build repo for prebuilt llama.cpp runtimes" > /dev/null || die "gh repo create $REPO failed"
  log "created $REPO (public)"
  STATE=none
else
  STATE=$(release_state) || die "cannot read the state of release $TAG"
fi

if [ "$STATE" = published ] && [ "$FORCE" = 0 ]; then
  log "release $TAG is already published, nothing to build (--force builds it again)"
  if ! pin_current; then write_pin; fi
  if [ "$VERIFY" = 1 ]; then verify_pin; fi
  exit 0
fi

log "release $TAG: $STATE"
push_tree
# GitHub registers a freshly pushed workflow a few seconds late (404 until then)
for i in $(seq 1 12); do
  info=$(runs_info) && break
  [ "$i" = 12 ] && die "cannot list the runs of $WORKFLOW: $(tail -n 1 "$SCRATCH/err")"
  sleep 5
done
read -r ACTIVE BEFORE <<< "$info"
if [ "${ACTIVE:--}" != "-" ]; then
  log "run $ACTIVE for $TAG is already going, not dispatching again"
  RUN_ID=$ACTIVE
else
  dispatch
  log "dispatched $WORKFLOW for $TAG"
  RUN_ID=""
  if [ "$WAIT" = 1 ]; then
    for i in $(seq 1 24); do
      info=$(runs_info) || die "cannot list the runs of $WORKFLOW: $(tail -n 1 "$SCRATCH/err")"
      read -r _ now <<< "$info"
      if [ "${now:-0}" -gt "${BEFORE:-0}" ]; then RUN_ID=$now; break; fi
      sleep "${SYNC_RETRY_SLEEP:-5}"
    done
    [ -n "$RUN_ID" ] || die "the run for $TAG did not show up in gh run list"
  fi
fi
if [ "$WAIT" = 0 ]; then
  log "not waiting; run the same command with --wait (and --verify) to attach, pin and check"
  exit 0
fi

watch_run "$RUN_ID"
STATE=$(release_state) || die "cannot read the state of release $TAG"
[ "$STATE" = published ] || die "the run succeeded but release $TAG is $STATE"
write_pin
if [ "$VERIFY" = 1 ]; then verify_pin; fi
log "done: $TAG is published and pinned"
