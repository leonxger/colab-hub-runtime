#!/bin/bash
# Smoke test of the voice runtime on CPU: starts audiocpp_server with the pinned Qwen3-TTS GGUF, synthesizes one German sentence as WAV (header 24000 Hz) and as streamed PCM; env AUDIOCPP_SERVER, VOICE_GGUF, AUDIOCPP_SRC (source tree with assets/resources/b.wav), SMOKE_TMP.
set -euo pipefail
: "${AUDIOCPP_SERVER:?path of audiocpp_server}" "${VOICE_GGUF:?path of the GGUF}" "${AUDIOCPP_SRC:?path of the audio.cpp source tree}"
tmp=${SMOKE_TMP:-$(mktemp -d)}
mkdir -p "$tmp"
port=${SMOKE_PORT:-18081}
ref=$AUDIOCPP_SRC/assets/resources/b.wav
[ -x "$AUDIOCPP_SERVER" ] && [ -f "$VOICE_GGUF" ] && [ -f "$ref" ] || { echo "voice_tts: missing server, GGUF or reference clip" >&2; exit 1; }
sentence='Guten Morgen, wie geht es dir heute?'

cat > "$tmp/server.json" << JSON
{
  "host": "127.0.0.1",
  "port": $port,
  "backend": "cpu",
  "threads": 4,
  "lazy_load": true,
  "max_loaded_models": 2,
  "models": [
    {"id": "tts-offline", "family": "qwen3_tts", "path": "$VOICE_GGUF", "task": "tts", "mode": "offline"},
    {"id": "tts-stream", "family": "qwen3_tts", "path": "$VOICE_GGUF", "task": "tts", "mode": "streaming"}
  ]
}
JSON
"$AUDIOCPP_SERVER" --config "$tmp/server.json" --no-ui > "$tmp/server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2> /dev/null || true; wait "$pid" 2> /dev/null || true' EXIT
fail() { echo "voice_tts: FAILED: $*" >&2; echo "---- server.log (tail)" >&2; tail -n 60 "$tmp/server.log" >&2; exit 1; }

up=0
for _ in $(seq 1 60); do
  kill -0 "$pid" 2> /dev/null || fail "server exited before it was healthy"
  if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" > "$tmp/health.json" 2> /dev/null; then up=1; break; fi
  sleep 1
done
[ "$up" = 1 ] || fail "no /health answer within 60 s"
echo "voice_tts: healthy: $(head -c 300 "$tmp/health.json")"

request() { # request ID EXTRA_JSON OUT
  cat << JSON
{"model": "$1", "input": "$sentence", "voice_ref": "$ref", "options": {"language": "de", "x_vector_only_mode": true, "max_tokens": 60, "seed": 7}$2}
JSON
}

request tts-offline '' > "$tmp/offline.json"
curl -fsS --max-time 900 -H 'Content-Type: application/json' -d @"$tmp/offline.json" -o "$tmp/out.wav" "http://127.0.0.1:$port/v1/audio/speech" || fail "offline speech request"
python3 -I - "$tmp/out.wav" << 'PY' || fail "the WAV is empty or its sample rate is not 24000"
import struct, sys
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "no RIFF/WAVE header"
rate = struct.unpack("<I", b[24:28])[0]
assert rate == 24000, "sample rate %d" % rate
assert len(b) > 44 + 24000 * 2 // 4, "only %d bytes" % len(b)
print("voice_tts: WAV ok, %d bytes, %d Hz" % (len(b), rate))
PY

request tts-stream ', "response_format": "pcm", "stream_format": "audio"' > "$tmp/stream.json"
curl -fsS -N --max-time 900 -H 'Content-Type: application/json' -d @"$tmp/stream.json" -o "$tmp/out.pcm" "http://127.0.0.1:$port/v1/audio/speech" || fail "streaming speech request"
bytes=$(stat -c %s "$tmp/out.pcm")
[ "$bytes" -gt $((24000 * 2 / 4)) ] || fail "streamed PCM has only $bytes bytes"
echo "voice_tts: streamed PCM ok, $bytes bytes (16-bit mono 24 kHz, $((bytes / 48000)) s)"
echo "voice_tts: PASS"
