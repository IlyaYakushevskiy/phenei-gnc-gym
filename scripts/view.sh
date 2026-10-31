#!/usr/bin/env bash
# Open a Rerun recording in the browser-based viewer.
#
#   scripts/view.sh                        # default recording
#   scripts/view.sh studies/other.rrd
#
# WHERE TO OPEN IT
#   Local (WSL2 -> Windows browser): localhost is forwarded automatically.
#   Remote (e.g. a MacBook over a tunnel): forward BOTH ports first --
#       ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <host>
#     then open the URL below on the laptop. The server binds 0.0.0.0, so
#     plain TCP forwarding is enough.
#   Simpler for a remote laptop: just copy the .rrd over (they are small) and
#   run a local viewer there -- no ports, no tunnel latency, GPU-accelerated.
#
# Ports: 9090 = viewer app, 9876 = gRPC data. The ?url= parameter is required;
# bare localhost:9090 loads the app with no data source attached.
#
# NOTE: the viewer rewrites the file it loads, and interrupting it mid-load
# (Ctrl-C during "Loading ...") can truncate the recording. So we always serve
# a throwaway copy and leave the original alone.
set -euo pipefail
cd "$(dirname "$0")/.."
RRD="${1:-studies/reference_descent.rrd}"
[ -f "$RRD" ] || { echo "no such recording: $RRD" >&2; exit 1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cp "$RRD" "$TMP/$(basename "$RRD")"

echo "serving a copy of $RRD (original is left untouched)"
echo "open:  http://localhost:9090?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy"
exec .venv/bin/rerun --web-viewer --web-viewer-port 9090 --port 9876 "$TMP/$(basename "$RRD")"
