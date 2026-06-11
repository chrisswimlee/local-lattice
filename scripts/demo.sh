#!/usr/bin/env bash
# 60-second capability-routing demo.
#
# Shows the core promise: agent code never names a model. It asks for
# `role:fast` / `role:coder`, and the gateway resolves whatever is
# actually loaded on this box — swap models and the same calls keep working.
#
# Usage:
#   ./scripts/demo.sh                       # against http://127.0.0.1:5000
#   BASE_URL=http://127.0.0.1:5001 ./scripts/demo.sh
#   MIDDLE_LAYER_API_KEY=... ./scripts/demo.sh   # if the gateway has auth on

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
AUTH=()
if [[ -n "${MIDDLE_LAYER_API_KEY:-}" ]]; then
  AUTH=(-H "X-API-Key: ${MIDDLE_LAYER_API_KEY}")
fi

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '\033[2m%s\033[0m\n' "$*"; }

pyjson() { python3 -c "$1"; }

chat() { # chat <model> <prompt> -> prints "resolved-model: reply"
  local model="$1" prompt="$2"
  curl -sS -m 120 ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
    "$BASE_URL/v1/chat/completions" \
    -d "{\"model\":\"$model\",\"max_tokens\":2000,\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}" \
  | pyjson '
import json, sys
d = json.load(sys.stdin)
err = d.get("error")
if err:
    print("  error:", err); sys.exit(0)
text = d["choices"][0]["message"]["content"].strip().replace("\n", " ")
print("  resolved to ->", d.get("model", "?"))
print("  reply       ->", text[:160])
'
}

say "1. The gateway is just the OpenAI API"
note "GET $BASE_URL/v1/models"
curl -sS -m 10 ${AUTH[@]+"${AUTH[@]}"} "$BASE_URL/v1/models" | pyjson '
import json, sys
ids = [m["id"] for m in json.load(sys.stdin)["data"]]
print("  %d models visible; first few: %s" % (len(ids), ids[:4]))
'

say "2. Agents ask for a capability, not a model id"
note 'model: "role:fast" — gateway picks the best loaded match'
chat "role:fast" "In one short sentence: why is the sky blue?"

say "3. Same code, different capability"
note 'model: "role:coder"'
chat "role:coder" "Write a one-line Python lambda that reverses a string."

say "4. Second opinions are one HTTP call"
note 'POST /swarm/vote — fan out to the loaded set, judge picks a winner'
curl -sS -m 300 ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
  "$BASE_URL/swarm/vote" \
  -d '{"models":"auto","messages":[{"role":"user","content":"Best name for a coffee-shop wifi network? One suggestion."}],"max_tokens":2000}' \
| pyjson '
import json, sys
d = json.load(sys.stdin)
err = d.get("error")
if err:
    print("  error:", err); sys.exit(0)
swarm = d.get("swarm", {})
for c in swarm.get("candidates", []):
    status = "ok " if c.get("ok") else "fail"
    print("  [%s] %s" % (status, c.get("model", "?")))
print("  winner ->", swarm.get("winner", "?"))
reply = d["choices"][0]["message"]["content"].strip().replace("\n", " ")
print("  reply  ->", reply[:160])
'

say "Done"
note "Swap what is loaded in LM Studio and run this again — the same"
note "role:* calls resolve to the new models with zero code changes."
