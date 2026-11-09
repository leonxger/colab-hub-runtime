#!/bin/bash
# Downloads a pinned GGUF (PREFIX_GGUF_* of versions.env, PREFIX defaults to VOICE) to DEST and checks size and sha256: fetch_gguf.sh DEST [PREFIX]; PREFIX_GGUF_BASE and RT_VERSIONS override the host and the pins (tests).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RT_VERSIONS="${RT_VERSIONS:-$HERE/../versions.env}"
dest=${1:?usage: fetch_gguf.sh DEST [PREFIX]}
p=${2:-VOICE}
[[ "$p" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "fetch_gguf: bad PREFIX '$p'" >&2; exit 2; }
base_var=${p}_GGUF_BASE
base_override=${!base_var:-}
# shellcheck source=/dev/null
. "$RT_VERSIONS"
[ -z "$base_override" ] || printf -v "$base_var" '%s' "$base_override"
pin() { local n=${p}_GGUF_$1; echo "${!n:?$n is not set in $RT_VERSIONS}"; }
base=$(pin BASE); repo=$(pin REPO); rev=$(pin REV); file=$(pin FILE); want_size=$(pin SIZE); want_sha=$(pin SHA256)
url="$base/$repo/resolve/$rev/$file"
mkdir -p "$(dirname "$dest")"
rm -f "$dest.part"
timeout "${FETCH_TIMEOUT:-1500}" curl -fsSL --retry 4 --retry-delay 5 --max-time "${FETCH_TIMEOUT:-1500}" -o "$dest.part" "$url" || { rm -f "$dest.part"; echo "fetch_gguf: download of $url failed" >&2; exit 1; }
size=$(stat -c %s "$dest.part")
[ "$size" = "$want_size" ] || { rm -f "$dest.part"; echo "fetch_gguf: size $size, want $want_size" >&2; exit 1; }
sha=$(sha256sum "$dest.part" | cut -c1-64)
[ "$sha" = "$want_sha" ] || { rm -f "$dest.part"; echo "fetch_gguf: sha256 $sha, want $want_sha" >&2; exit 1; }
mv "$dest.part" "$dest"
echo "fetch_gguf: $dest $size bytes sha256 ${sha:0:12}"
