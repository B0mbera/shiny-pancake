#!/usr/bin/env bash
# A tour of the HTTP API. Start the hub first:  shiny-pancake serve
set -euo pipefail
HUB=${HUB:-http://127.0.0.1:8000}
AUTH=()
if [[ -n "${SP_API_KEY:-}" ]]; then AUTH=(-H "x-api-key: ${SP_API_KEY}"); fi

echo "== health";  curl -fsS "$HUB/health"; echo

echo "== synchronous agent run"
curl -fsS "${AUTH[@]}" -X POST "$HUB/run" -H 'content-type: application/json' \
     -d '{"goal": "What is 12 * (3 + 4)? Check it."}'; echo

echo "== enqueue a multi-worker summary job"
JOB=$(curl -fsS "${AUTH[@]}" -X POST "$HUB/jobs" -H 'content-type: application/json' \
     -d '{"kind": "summarize", "payload": {"path": "ai-integration-playbook.md", "strategy": "ensemble"}}' \
     | python3 -c 'import json, sys; print(json.load(sys.stdin)["job"]["id"])')
echo "job: $JOB"

echo "== follow it live (Server-Sent Events)"
curl -fsSN "${AUTH[@]}" "$HUB/jobs/$JOB/stream" | grep -m1 '^event: done' >/dev/null && echo "done"

echo "== result"
curl -fsS "${AUTH[@]}" "$HUB/jobs/$JOB" | python3 -c 'import json, sys; print(json.load(sys.stdin)["result"]["output"])'
