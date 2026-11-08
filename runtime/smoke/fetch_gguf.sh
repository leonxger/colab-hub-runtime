#!/bin/bash
# Downloads the pinned Qwen3-TTS GGUF (VOICE_GGUF_* of versions.env) to DEST and checks size and sha256: fetch_gguf.sh DEST; VOICE_GGUF_BASE and RT_VERSIONS override the host and the pins (tests).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RT_VERSIONS="${RT_VERSIONS:-$HERE/../versions.env}"
base_override=${VOICE_GGUF_BASE:-}
# shellcheck source=/dev/null
. "$RT_VERSIONS"
[ -z "$base_override" ] || VOICE_GGUF_BASE=$base_override
dest=${1:?usage: fetch_gguf.sh DEST}
url="$VOICE_GGUF_BASE/$VOICE_GGUF_REPO/resolve/$VOICE_GGUF_REV/$VOICE_GGUF_FILE"
mkdir -p "$(dirname "$dest")"
rm -f "$dest.part"
timeout "${FETCH_TIMEOUT:-1500}" curl -fsSL --retry 4 --retry-delay 5 --max-time "${FETCH_TIMEOUT:-1500}" -o "$dest.part" "$url" || { rm -f "$dest.part"; echo "fetch_gguf: download of $url failed" >&2; exit 1; }
size=$(stat -c %s "$dest.part")
[ "$size" = "$VOICE_GGUF_SIZE" ] || { rm -f "$dest.part"; echo "fetch_gguf: size $size, want $VOICE_GGUF_SIZE" >&2; exit 1; }
sha=$(sha256sum "$dest.part" | cut -c1-64)
[ "$sha" = "$VOICE_GGUF_SHA256" ] || { rm -f "$dest.part"; echo "fetch_gguf: sha256 $sha, want $VOICE_GGUF_SHA256" >&2; exit 1; }
mv "$dest.part" "$dest"
echo "fetch_gguf: $dest $size bytes sha256 ${sha:0:12}"
