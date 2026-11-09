#!/bin/bash
# voice.sh and the voice smoke scripts offline: CMake goldens, package list, packaging from fake build trees (layout, num2words, licences, BUILDINFO, byte-identical repeat, failures), clones from fake upstreams, fetch_gguf.sh, voice_tts.sh against a fake server, workflow facts.
. "$(dirname "$0")/lib.sh"
VOICE="$BUILD/voice.sh"
SMOKE="$RT/smoke"
export TREE12=0123456789ab SOURCE_DATE_EPOCH=1700000000
GITQ=(git -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false -c init.defaultBranch=main)

# ---- goldens ----
for k in audiocpp qwentts audiocpp-cpu; do
  [ "$(bash "$VOICE" --print-cmake "$k")" = "$(cat "$GOLDEN/voice-$k-cmake.txt")" ] && ok "--print-cmake $k equals golden/voice-$k-cmake.txt" || bad "--print-cmake $k differs from its golden"
done
check_not "--print-cmake refuses an unknown key" bash "$VOICE" --print-cmake nothing
for k in audiocpp qwentts; do
  line=$(bash "$VOICE" --print-cmake "$k")
  grep -qF 'CMAKE_CUDA_ARCHITECTURES="80-real;120a-real"' <<< "$line" && ok "$k builds real cubins for sm_80 and sm_120a only (sm_89 runs the sm_80 code)" || bad "$k architectures differ from 80-real;120a-real"
  ! grep -q 'NATIVE=ON' <<< "$line" && ok "$k has no native CPU flag" || bad "$k line enables native CPU code"
  grep -qF "CMAKE_INSTALL_RPATH='\$ORIGIN'" <<< "$line" && ok "$k binaries carry RUNPATH \$ORIGIN" || bad "$k lacks the RUNPATH"
done
grep -q 'DENGINE_ENABLE_CPU_ALL_VARIANTS=ON' "$GOLDEN/voice-audiocpp-cmake.txt" && grep -q 'DAUDIOCPP_DEPLOYMENT_BUILD=ON' "$GOLDEN/voice-audiocpp-cmake.txt" && ok "audio.cpp: dynamic backends and compiled-in model specs" || bad "audio.cpp line lacks CPU_ALL_VARIANTS or DEPLOYMENT_BUILD"
grep -q 'DENGINE_ENABLE_CUDA=OFF' "$GOLDEN/voice-audiocpp-cpu-cmake.txt" && ! grep -q 'CUDA_ARCHITECTURES' "$GOLDEN/voice-audiocpp-cpu-cmake.txt" && ok "the smoke line builds without CUDA" || bad "the smoke line is not CPU-only"
[ "$(TREE12=$TREE12 bash "$VOICE" --list-package)" = "$(cat "$GOLDEN/voice-package.txt")" ] && ok "--list-package equals golden/voice-package.txt" || bad "--list-package differs from the golden"
[ "$(head -1 "$GOLDEN/voice-package.txt")" = "voice-0123456789ab.tar.gz" ] && ok "tarball name is voice-<tree12>.tar.gz" || bad "tarball name differs"
check_not "a bad TREE12 is refused" env TREE12=xyz bash "$VOICE" --list-package

# ---- a fixture versions file with a fake wheel ----
fx="$SCRATCH/fx"; mkdir -p "$fx"
python3 -I - "$fx/num2words-0.5.14-py3-none-any.whl" << 'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr("num2words/__init__.py", "MARK = 'fake num2words'\n")
    z.writestr("num2words/lang_DE.py", "x = 1\n")
    z.writestr("num2words/__pycache__/junk.pyc", "junk")
    z.writestr("num2words-0.5.14.dist-info/COPYING", "GNU Lesser General Public License\n\nVersion 2.1, February 1999\n")
    z.writestr("num2words-0.5.14.dist-info/METADATA", "Name: num2words\n")
PY
wsha=$(sha256sum "$fx/num2words-0.5.14-py3-none-any.whl" | cut -c1-64)
cp "$RT/versions.env" "$fx/versions.env"
cat >> "$fx/versions.env" << EOF
NUM2WORDS_URL=file://$fx/num2words-0.5.14-py3-none-any.whl
NUM2WORDS_SHA256=$wsha
EOF

mkbuild() { # mkbuild WORK: fake build trees of both engines and their sources
  local w=$1 a=$1/build/audiocpp/bin q=$1/build/qwentts
  mkdir -p "$a" "$q" "$w/src/audiocpp" "$w/src/qwentts"
  for f in "$a/audiocpp_server" "$q/tts-server"; do printf '#!/bin/sh\necho %s\n' "$(basename "$f")" > "$f"; chmod +x "$f"; done
  for d in "$a" "$q"; do
    echo cuda > "$d/libggml-cuda.so"; echo base > "$d/libggml-base.so.0.0.1"; ln -s libggml-base.so.0.0.1 "$d/libggml-base.so.0"; ln -s libggml-base.so.0 "$d/libggml-base.so"
    echo cpu > "$d/libggml-cpu-haswell.so"
  done
  echo "Apache license" > "$w/src/audiocpp/LICENSE"; echo "MIT license" > "$w/src/qwentts/LICENSE"
}
pack() { env WORK="$1" RT_VERSIONS="$fx/versions.env" "${@:2}" bash "$VOICE" --package-only > "$SCRATCH/pack.out" 2>&1; }

name=voice-$TREE12.tar.gz
w1="$SCRATCH/w1"; mkbuild "$w1"
if pack "$w1"; then
  ok "--package-only packs a tarball"
  tarball="$w1/out/$name"
  [ -f "$tarball" ] && ok "file name is $name" || bad "expected $tarball, found $(ls "$w1/out" | tr '\n' ' ')"
  list=$(tar tzf "$tarball")
  while read -r req; do grep -qx "$req" <<< "$list" && ok "tarball has $req" || bad "tarball lacks $req"; done < <(tail -n +2 "$GOLDEN/voice-package.txt")
  for f in audiocpp/bin/libggml-cpu-haswell.so audiocpp/bin/libggml-base.so qwentts/bin/libggml-cpu-haswell.so qwentts/bin/libggml-base.so.0 py/num2words/lang_DE.py; do
    grep -qx "$f" <<< "$list" && ok "tarball has $f" || bad "tarball lacks $f"
  done
  tar tvzf "$tarball" | grep -q 'audiocpp/bin/libggml-base.so.0 -> libggml-base.so.0.0.1' && ok "symlinks stay symlinks" || bad "symlinks not preserved"
  ! grep -q '__pycache__' <<< "$list" && ok "no __pycache__" || bad "__pycache__ in the tarball"
  [ -z "$(grep '^/\|^\./' <<< "$list")" ] && ok "paths are relative without ./" || bad "absolute or ./ paths in the tarball"
  [ "$(grep -c 'libggml-cuda.so$' <<< "$list")" = 2 ] && ok "each engine has its own libggml-cuda.so (the two ggml versions never share a directory)" || bad "libggml-cuda.so count"
  (cd "$w1/out" && sha256sum -c --quiet "$name.sha256") && ok ".sha256 file verifies" || bad ".sha256 does not verify"
  x="$SCRATCH/x1"; mkdir -p "$x"; tar xzf "$tarball" -C "$x"
  grep -q 'Version 2.1' "$x/py/num2words/COPYING" && ok "the LGPL-2.1 text sits beside the package" || bad "COPYING is not the LGPL-2.1 text"
  . "$RT/versions.env"
  grep -qF "$NUM2WORDS_SOURCE" "$x/py/num2words/SOURCE.txt" && grep -qF "$NUM2WORDS_VERSION" "$x/py/num2words/SOURCE.txt" && ok "SOURCE.txt names the version and the source URL" || bad "SOURCE.txt incomplete"
  [ "$(env -C "$SCRATCH" PYTHONPATH="$x/py" python3 -c 'import num2words; print(num2words.MARK)')" = "fake num2words" ] && ok "import num2words works with PYTHONPATH=<runtime>/py" || bad "num2words does not import from py/"
  python3 -I - "$x/BUILDINFO.json" "$fx" << 'EOF' && ok "BUILDINFO.json holds the engine pins, the GGUF pin, CUDA pins and both CMake lines" || bad "BUILDINFO.json wrong"
import json, os, sys
info = json.load(open(sys.argv[1]))
rt = sys.argv[2]
env = dict(l.strip().split("=", 1) for l in open(os.path.join(rt, "versions.env")) if "=" in l and not l.startswith("#"))
assert info["family"] == "voice" and info["tree12"] == "0123456789ab"
assert info["audiocpp"]["tag"] == env["AUDIOCPP_TAG"] and info["audiocpp"]["commit"] == env["AUDIOCPP_COMMIT"]
assert info["qwentts"]["commit"] == env["QWENTTS_COMMIT"] and info["qwentts"]["ggml_commit"] == env["QWENTTS_GGML_COMMIT"]
assert info["num2words"]["version"] == env["NUM2WORDS_VERSION"] and info["num2words"]["wheel_sha256"] == env["NUM2WORDS_SHA256"]
g = info["gguf"]
assert (g["repo"], g["revision"], g["file"], g["size"], g["sha256"]) == (env["VOICE_GGUF_REPO"], env["VOICE_GGUF_REV"], env["VOICE_GGUF_FILE"], int(env["VOICE_GGUF_SIZE"]), env["VOICE_GGUF_SHA256"])
assert info["cuda"] == {"version": "13.0.2", "image": env["CUDA_IMAGE"], "digest": env["CUDA_IMAGE_DIGEST"]}
assert info["archs"] == ["sm_80", "sm_120a"] and info["max_glibc"] == "2.39"
assert "audiocpp_server" in info["cmake"]["audiocpp"] or "-S src/audiocpp" in info["cmake"]["audiocpp"]
assert "-S src/qwentts" in info["cmake"]["qwentts"]
EOF
  w2="$SCRATCH/w2"; mkbuild "$w2"
  if pack "$w2"; then [ "$(sha256sum < "$tarball")" = "$(sha256sum < "$w2/out/$name")" ] && ok "a repeat from the same inputs is byte-identical" || bad "repeat differs"; else bad "repeat pack failed"; fi
else
  bad "--package-only failed: $(tail -3 "$SCRATCH/pack.out" | tr '\n' ' ')"
fi

for gone in audiocpp/bin/libggml-cuda.so qwentts/bin/libggml-cuda.so qwentts/bin/tts-server audiocpp/bin/audiocpp_server; do
  w3="$SCRATCH/w3"; rm -rf "$w3"; mkbuild "$w3"
  case "$gone" in audiocpp/*) rm -f "$w3/build/audiocpp/bin/${gone#audiocpp/bin/}" ;; *) rm -f "$w3/build/qwentts/${gone#qwentts/bin/}" ;; esac
  if pack "$w3"; then bad "a build without $gone was packed"; else ok "a build without $gone is refused"; fi
  [ -z "$(ls "$w3/out" 2> /dev/null)" ] && ok "no tarball is left after the failure ($gone)" || bad "tarball left after a failed packaging ($gone)"
done
w4="$SCRATCH/w4"; mkbuild "$w4"
sed "s/^NUM2WORDS_SHA256=.*/NUM2WORDS_SHA256=$(printf '0%.0s' $(seq 1 64))/" "$fx/versions.env" > "$fx/versions-badsha.env"
if env WORK="$w4" RT_VERSIONS="$fx/versions-badsha.env" bash "$VOICE" --package-only > "$SCRATCH/pack.out" 2>&1; then bad "a wheel with the wrong sha256 was vendored"; else grep -q 'num2words wheel has sha256' "$SCRATCH/pack.out" && ok "a wheel with the wrong sha256 is refused and named" || bad "bad wheel not named"; fi
w5="$SCRATCH/w5"; mkbuild "$w5"; mkdir -p "$w5/out"; echo x > "$w5/out/old"
check_not "a used out directory is refused" env WORK="$w5" RT_VERSIONS="$fx/versions.env" bash "$VOICE" --package-only
check_not "a relative WORK is refused" env WORK=relative RT_VERSIONS="$fx/versions.env" bash "$VOICE" --package-only
grep -q 'check_elf.sh" "\$stage/audiocpp"' "$VOICE" && grep -q 'check_elf.sh" "\$stage/qwentts"' "$VOICE" && ok "a real build runs check_elf.sh on both engine trees" || bad "voice.sh does not check both trees with check_elf.sh"

# ---- clones from fake upstreams ----
up="$SCRATCH/up-audiocpp"; mkdir -p "$up"; "${GITQ[@]}" -C "$up" init -q
echo "Apache" > "$up/LICENSE"; "${GITQ[@]}" -C "$up" add LICENSE && "${GITQ[@]}" -C "$up" commit -q -m first && "${GITQ[@]}" -C "$up" tag vT9
ac=$("${GITQ[@]}" -C "$up" rev-parse HEAD)
gg="$SCRATCH/up-ggml"; mkdir -p "$gg"; "${GITQ[@]}" -C "$gg" init -q
echo "ggml" > "$gg/g.txt"; "${GITQ[@]}" -C "$gg" add g.txt && "${GITQ[@]}" -C "$gg" commit -q -m ggml
ggc=$("${GITQ[@]}" -C "$gg" rev-parse HEAD)
qw="$SCRATCH/up-qwentts"; mkdir -p "$qw"; "${GITQ[@]}" -C "$qw" init -q
echo "MIT" > "$qw/LICENSE"
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always "${GITQ[@]}" -C "$qw" submodule add -q "file://$gg" ggml
"${GITQ[@]}" -C "$qw" add -A && "${GITQ[@]}" -C "$qw" commit -q -m first
qc=$("${GITQ[@]}" -C "$qw" rev-parse HEAD)
cat > "$fx/versions-clone.env" << EOF
$(cat "$RT/versions.env")
AUDIOCPP_URL=file://$up
AUDIOCPP_TAG=vT9
AUDIOCPP_COMMIT=$ac
QWENTTS_URL=file://$qw
QWENTTS_COMMIT=$qc
QWENTTS_GGML_COMMIT=$ggc
EOF
clone() { # clone FN DIR [env...]
  local fn=$1 dir=$2; shift 2
  ( env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always RT_VERSIONS="$fx/versions-clone.env" "$@" bash -c '. "$1"; "$2" "$3"' _ "$VOICE" "$fn" "$dir" ) > "$SCRATCH/clone.out" 2>&1
}
if clone voice_clone_audiocpp "$SCRATCH/c1"; then ok "audio.cpp is cloned at its tag with the commit checked"; [ -f "$SCRATCH/c1/LICENSE" ] && ok "the clone has the sources" || bad "clone is empty"; else bad "audio.cpp clone failed: $(tail -2 "$SCRATCH/clone.out" | tr '\n' ' ')"; fi
sed "s/^AUDIOCPP_COMMIT=.*/AUDIOCPP_COMMIT=$(printf '1%.0s' $(seq 1 40))/" "$fx/versions-clone.env" > "$fx/versions-badcommit.env"
if ( env RT_VERSIONS="$fx/versions-badcommit.env" bash -c '. "$1"; voice_clone_audiocpp "$2"' _ "$VOICE" "$SCRATCH/c2" ) > "$SCRATCH/clone.out" 2>&1; then bad "a wrong audio.cpp commit was accepted"; else grep -q 'commit mismatch for audio.cpp' "$SCRATCH/clone.out" && ok "a wrong audio.cpp commit is refused" || bad "audio.cpp commit mismatch not reported"; fi
if clone voice_clone_qwentts "$SCRATCH/c3"; then ok "qwentts.cpp is fetched by commit with its ggml submodule"; [ -f "$SCRATCH/c3/ggml/g.txt" ] && ok "the ggml submodule is checked out" || bad "ggml submodule empty"; else bad "qwentts.cpp fetch failed: $(tail -3 "$SCRATCH/clone.out" | tr '\n' ' ')"; fi
sed "s/^QWENTTS_GGML_COMMIT=.*/QWENTTS_GGML_COMMIT=$(printf '2%.0s' $(seq 1 40))/" "$fx/versions-clone.env" > "$fx/versions-badggml.env"
if ( env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always RT_VERSIONS="$fx/versions-badggml.env" bash -c '. "$1"; voice_clone_qwentts "$2"' _ "$VOICE" "$SCRATCH/c4" ) > "$SCRATCH/clone.out" 2>&1; then bad "a wrong ggml commit was accepted"; else grep -q 'ggml commit mismatch' "$SCRATCH/clone.out" && ok "a wrong ggml commit is refused" || bad "ggml mismatch not reported"; fi
mkdir -p "$SCRATCH/c5" && echo x > "$SCRATCH/c5/x"
check_not "a used clone directory is refused" clone voice_clone_audiocpp "$SCRATCH/c5"

# ---- fetch_gguf.sh ----
repo_dir="$SCRATCH/hf/audio-cpp/audio.cpp-gguf/resolve/REV1/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
mkdir -p "$repo_dir"; head -c 4096 /dev/urandom > "$repo_dir/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf"
gsha=$(sha256sum "$repo_dir/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf" | cut -c1-64)
cat > "$fx/versions-gguf.env" << EOF
VOICE_GGUF_BASE=https://huggingface.co
VOICE_GGUF_REPO=audio-cpp/audio.cpp-gguf
VOICE_GGUF_REV=REV1
VOICE_GGUF_FILE=Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf
VOICE_GGUF_SIZE=4096
VOICE_GGUF_SHA256=$gsha
EOF
fetch() { env RT_VERSIONS="$1" VOICE_GGUF_BASE="file://$SCRATCH/hf" bash "$SMOKE/fetch_gguf.sh" "$2" > "$SCRATCH/fetch.out" 2>&1; }
if fetch "$fx/versions-gguf.env" "$SCRATCH/out/g.gguf"; then ok "fetch_gguf.sh downloads and verifies the file"; cmp -s "$SCRATCH/out/g.gguf" "$repo_dir/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf" && ok "the downloaded bytes match" || bad "bytes differ"; else bad "fetch_gguf.sh failed: $(tail -2 "$SCRATCH/fetch.out" | tr '\n' ' ')"; fi
sed 's/^VOICE_GGUF_SIZE=.*/VOICE_GGUF_SIZE=4097/' "$fx/versions-gguf.env" > "$fx/versions-gguf-size.env"
if fetch "$fx/versions-gguf-size.env" "$SCRATCH/out/s.gguf"; then bad "a wrong size was accepted"; else grep -q 'size 4096, want 4097' "$SCRATCH/fetch.out" && [ ! -e "$SCRATCH/out/s.gguf" ] && [ ! -e "$SCRATCH/out/s.gguf.part" ] && ok "a wrong size is refused and leaves nothing behind" || bad "size mismatch handling wrong"; fi
sed "s/^VOICE_GGUF_SHA256=.*/VOICE_GGUF_SHA256=$(printf '0%.0s' $(seq 1 64))/" "$fx/versions-gguf.env" > "$fx/versions-gguf-sha.env"
if fetch "$fx/versions-gguf-sha.env" "$SCRATCH/out/h.gguf"; then bad "a wrong sha256 was accepted"; else grep -q 'sha256' "$SCRATCH/fetch.out" && [ ! -e "$SCRATCH/out/h.gguf" ] && [ ! -e "$SCRATCH/out/h.gguf.part" ] && ok "a wrong sha256 is refused and leaves nothing behind" || bad "sha mismatch handling wrong"; fi
sed 's/^VOICE_GGUF_REV=.*/VOICE_GGUF_REV=NOPE/' "$fx/versions-gguf.env" > "$fx/versions-gguf-rev.env"
check_not "a missing file fails" fetch "$fx/versions-gguf-rev.env" "$SCRATCH/out/m.gguf"
python3 -I - "$RT/versions.env" << 'EOF' && ok "versions.env pins the GGUF at revision 73ce4162f6cc... with 2695186240 bytes" || bad "the GGUF pin in versions.env changed"
import sys
env = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l and not l.startswith("#"))
assert env["VOICE_GGUF_REV"] == "73ce4162f6cce370dd533c6782b772135c1b2a8e"
assert env["VOICE_GGUF_SIZE"] == "2695186240"
assert env["VOICE_GGUF_SHA256"] == "92641a8022d521efd79a5db45e952589b3f2b781680f9abfcf26f075d37b86c6"
assert env["VOICE_GGUF_FILE"] == "Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf"
EOF

# ---- voice_tts.sh against a fake audiocpp_server ----
fake="$SCRATCH/fake"; mkdir -p "$fake/src/assets/resources"
echo RIFFfake > "$fake/src/assets/resources/b.wav"; echo gguf > "$fake/voice.gguf"
cat > "$fake/audiocpp_server" << 'PY'
#!/usr/bin/env python3
import json, os, struct, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1]))
assert "--no-ui" in sys.argv
mode = os.environ.get("FAKE_MODE", "good")
if mode == "exit":
    sys.exit(3)

def wav(rate, nbytes):
    data = b"\x01\x00" * (nbytes // 2)
    return b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(data)) + data

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["voice_ref"].endswith("b.wav") and body["options"]["x_vector_only_mode"] is True and body["options"]["language"] == "de"
        ids = [m["id"] for m in cfg["models"]]
        assert body["model"] in ids
        if body.get("stream_format") == "audio":
            assert body["response_format"] == "pcm"
            out = b"" if mode == "empty-pcm" else b"\x02\x00" * 24000
            self.send_response(200); self.send_header("Content-Type", "audio/pcm"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
        else:
            out = wav(16000 if mode == "bad-rate" else 24000, 0 if mode == "empty-wav" else 48000)
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
HTTPServer(("127.0.0.1", cfg["port"]), H).serve_forever()
PY
chmod +x "$fake/audiocpp_server"
port=$((18100 + RANDOM % 80))
smoke() { # smoke MODE
  env FAKE_MODE="$1" AUDIOCPP_SERVER="$fake/audiocpp_server" VOICE_GGUF="$fake/voice.gguf" AUDIOCPP_SRC="$fake/src" SMOKE_TMP="$SCRATCH/smoke-$1" SMOKE_PORT=$port timeout 100 bash "$SMOKE/voice_tts.sh" > "$SCRATCH/smoke.out" 2>&1
}
if smoke good; then ok "voice_tts.sh passes against a server that answers 24000 Hz WAV and PCM"; else bad "voice_tts.sh failed on a good server: $(tail -4 "$SCRATCH/smoke.out" | tr '\n' ' ')"; fi
grep -q 'voice_tts: PASS' "$SCRATCH/smoke.out" && ok "it ends with a PASS line" || bad "no PASS line"
for m in bad-rate empty-wav empty-pcm exit; do
  if smoke "$m"; then bad "voice_tts.sh passed with a $m server"; else grep -q 'voice_tts: FAILED' "$SCRATCH/smoke.out" && ok "voice_tts.sh fails on a $m server" || bad "no FAILED line for $m: $(tail -2 "$SCRATCH/smoke.out" | tr '\n' ' ')"; fi
done
check_not "voice_tts.sh refuses a missing GGUF" env AUDIOCPP_SERVER="$fake/audiocpp_server" VOICE_GGUF="$fake/none.gguf" AUDIOCPP_SRC="$fake/src" bash "$SMOKE/voice_tts.sh"
check_not "voice_tts.sh needs its environment" env -u AUDIOCPP_SERVER bash "$SMOKE/voice_tts.sh"
[ -z "$(pgrep -f "$fake/audiocpp_server" || true)" ] && ok "no fake server is left running" || bad "a fake server is still running"

# ---- workflow facts ----
WF="$RT/workflows/build.yml"
grep -q 'voice) *$' "$WF" && grep -q 'build/voice.sh' "$WF" && ok "the build job has a voice family that runs voice.sh" || bad "no voice branch in the build job"
grep -q 'runtime/smoke/voice_\*.sh' "$WF" && ok "the smoke job runs every runtime/smoke/voice_*.sh" || bad "no smoke loop over runtime/smoke/voice_*.sh"
grep -q 'build/voice.sh --cpu-audiocpp' "$WF" && grep -q 'runtime/smoke/fetch_gguf.sh' "$WF" && ok "the smoke job builds audiocpp_server for the CPU and fetches the pinned GGUF" || bad "smoke job misses the CPU build or the GGUF fetch"
grep -q 'needs: \[plan, build, smoke, smoke-qwen\]' "$WF" && ok "publishing waits for the smoke job" || bad "publish does not wait for smoke"
grep -q 'families:' "$WF" && grep -q 'default: qwen,deepseek,voice' "$WF" && ok "the workflow takes a families input that defaults to every family" || bad "no families input"
check_not "the smoke job runs on a standard runner (no larger runner label)" grep -qE 'runs-on: .*(large|xlarge|[0-9]+-core|gpu)' "$WF"
check_not "the smoke job asks for no paid runner" grep -q 'ubuntu-24.04-' "$WF"

finish test_voice_build.sh
