# Capability protocol

> **Status:** descriptive (matches the resolver shipped in 0.2.0). The
> grammar will not change incompatibly before 1.0; new capability hints
> may be added.

Local Lattice exposes the OpenAI HTTP API. The only thing that differs
from a vanilla OpenAI server is **what an agent is allowed to put in
`"model"`** and **how that value is resolved against the set of models
currently available on disk**.

This document specifies that protocol.

## 1. The `"model"` field grammar

The string sent in `"model"` is resolved by
[`resolve_model_alias()`](../middle_layerMLX.py) against the live list of
available aliases. The grammar is:

```
model        := empty | placeholder | spec-list
spec-list    := spec ("," spec)*
spec         := role-spec | wildcard-spec | exact-spec
role-spec    := "role:" role-name           ; e.g. role:coder
wildcard-spec:= alias-with-stars            ; e.g. *coder*, qwen*
exact-spec   := literal-alias-or-substring  ; e.g. qwen3-coder-32b
placeholder  := "" | "auto" | "default" | configured placeholder ids
```

Resolution rules:

1. **Placeholders, empty, `auto`, `default`** → auto-pick using
   `mlx_dashboard`'s runtime default → `DEFAULT_MODEL` env →
   `role:default` registry entry → first available alias. Auto-pick is
   what makes Lattice "just work" out of the box: agents that don't care
   which model they get send `""` or `"auto"` and get a sensible answer.
2. **`role:<name>`** → look up `<name>` in the role registry
   (`mlx_roles.json` by default). The registry value is an ordered list
   of substrings; the first one that matches a currently-available alias
   wins. Unknown roles fall through to the next spec in the list.
3. **`spec,spec,spec`** → priority list. Each spec is tried in order;
   first match wins. Mix freely: `"role:coder,qwen3-coder-32b,*coder*"`.
4. **Wildcard substrings** (`"*foo*"` or `"foo*"`) → substring match
   against the available alias list.
5. **Exact / substring** → first exact id match, then a substring
   fallback.

If nothing matches, the call returns an OpenAI-shaped 4xx with the list
of available aliases in the error message.

### Examples

| Sent by agent                                    | Resolves to                                  |
|--------------------------------------------------|----------------------------------------------|
| `""`, `"auto"`, `"default"`                      | Operator default / `role:default` / first    |
| `"role:coder"`                                   | Best available coding model (see registry)   |
| `"role:reasoner,role:coder,role:fast"`           | Tiered fallback                              |
| `"role:reasoner+anthropic-claude-4.6-opus"`*     | Local first, cloud escalation if missing    |
| `"qwen3-coder-32b"`                              | Exact alias match                             |
| `"*coder*"`                                      | Any alias containing `coder`                 |

\* `+` is treated as part of a comma list when combined; canonical form
is the priority list `"role:reasoner,anthropic-claude-4.6-opus"`.

## 2. The role registry

Roles live in [`mlx_roles.json`](../mlx_roles.json). Each role maps to an
ordered list of substrings. First match against the available alias list
wins.

The shipped registry covers:

| Role       | Intended capability                              |
|------------|--------------------------------------------------|
| `fast`     | Low-latency, small (≤ ~10 GB) models             |
| `coder`    | Code-tuned models                                |
| `reasoner` | Large / reasoning-oriented models                |
| `vision`   | Vision-capable models                            |
| `default`  | Operator's general-purpose default               |

Operators override or extend the registry by editing the JSON file.
Capability matching is **substring**, so adding a new model with a
matching alias automatically picks up the role without restarting the
agent.

## 3. Automatic capability inference

In addition to whatever the agent sent in `"model"`, Lattice **infers
capability requirements from the request itself** and filters the
candidate pool before resolution:

| Hint                  | How it's inferred                                                       | What it filters on                                       |
|-----------------------|--------------------------------------------------------------------------|----------------------------------------------------------|
| `needs_vision`        | The `messages` array contains an `image_url` content part.              | Aliases where `model_profiles.json` sets `supports_vision: true`. |
| `min_context_window`  | Total prompt characters exceed `MLX_ROUTE_LONG_PROMPT_CHARS` (default ~28k chars). Estimated tokens become the minimum context window. | Aliases whose profile `context_window` is ≥ the estimate. |
| `prefers_fast`        | Request header `X-MLX-Latency-Tier: fast`.                              | Aliases with `latency_tier: fast` OR `memory_gb_estimate ≤ 10`. |

The filter is **soft** when the hint is "soft" (e.g. `prefers_fast`):
if no alias passes, Lattice falls back to the full pool with a warning
on the dashboard.

The filter is **strict** when the hint is hard (e.g. `needs_vision` or
an explicit `min_context_window`): if nothing matches, the request
returns a 4xx with an explanation instead of silently picking a
non-vision model.

## 4. Model profile metadata

Profiles live in [`model_profiles.json`](../model_profiles.json) and
describe what each alias (or substring pattern) is capable of:

```json
{
  "defaults": {
    "context_window": 128000,
    "supports_vision": false,
    "supports_tools": true,
    "latency_tier": "medium",
    "memory_gb_estimate": 8.0
  },
  "aliases":  { "exact/alias": { ... } },
  "patterns": [ { "substring": "vl", "profile": { "supports_vision": true } } ]
}
```

Profile lookup is: `defaults` → first `patterns` match → exact `aliases`
override. Operators tune this file to teach Lattice about their fleet.

## 5. Swarm endpoints

The gateway exposes swarm functionality via two surfaces:

> **Human walkthrough:** sequence diagrams and copy-paste Python for vote /
> pipeline / fanout live in the README —
> [Swarm in 60 seconds](../README.md#swarm-in-60-seconds).

1. **Chat meta-models on `POST /v1/chat/completions`** — drop-in for any
   OpenAI-compatible client. Set `model:` to one of the names below and
   optionally pass a `swarm:` extension object alongside the standard
   `messages[]`/`max_tokens`/`temperature` fields.
2. **Dedicated `POST /swarm/*` endpoints** — richer request/response shapes
   (return all candidates, sequential pipelines, multi-round debate). Use
   when the OpenAI chat shape can't carry what you need.

Swarm endpoints accept the same capability spec in their `models` array,
so an agent can request *N opinions from the best available local
coder*, etc.

### Chat meta-models (LM Studio gateway)

| `model:` value      | intent     | strategy default            | notes                                                                                  |
| ------------------- | ---------- | --------------------------- | -------------------------------------------------------------------------------------- |
| `swarmCouncil`      | council    | `best-of-n` (judge picks)   | **Canonical.** What most callers want.                                                 |
| `swarmVote`         | council    | `best-of-n`                 | Alias of `swarmCouncil`.                                                               |
| `swarm/vote`        | council    | `best-of-n`                 | Path-style alias of `swarmCouncil`.                                                    |
| `swarmIntelligence` | council    | `best-of-n`                 | **Deprecated** alias kept for the openclaw runtime; emits `DeprecationWarning`, removed in 0.2.0. |
| `swarm/fanout`      | fanout     | `fanout` (no judge)         | Returns the first successful candidate without a judge round. Faster than council.     |
| `swarm/pipeline`    | pipeline   | (rejected with 400)         | Pipelines need `stages[]` which the OpenAI chat shape can't carry — send to `POST /swarm/pipeline` instead. |

Explicit `swarm.strategy` in the request body always wins over the intent's
default. Recognized strategies: `best-of-n`, `first-success`, `longest`,
`fanout`.

Successful chat-meta-model responses set:

- `X-Model-Routed-To: swarm/<winner-id>`
- `X-Swarm-Intent: council|fanout`
- `X-Swarm-Canonical-Name: swarmCouncil` (only when the request used a non-canonical alias)

The full alias map is also surfaced on `/healthz` under
`swarm_chat_canonical` and `swarm_chat_aliases`.

### `POST /swarm/fanout`

Run N specs in parallel, return all of them.

```json
{
  "models":   ["role:coder", "qwen3-coder-32b", "anthropic-claude-4.6-opus"],
  "messages": [{ "role": "user", "content": "..." }],
  "max_parallel": 3
}
```

Each entry is independently resolved. Unresolved entries return an
error in their slot instead of failing the whole request.

### `POST /swarm/vote`

Fanout + judge. The judge spec is resolved the same way:

```json
{
  "models":   ["role:reasoner", "role:coder", "role:fast"],
  "judge":    "role:reasoner",
  "messages": [{ "role": "user", "content": "..." }]
}
```

The judge sees all anonymized candidate answers and picks the winner.

### `POST /swarm/pipeline`

Sequential. Each step's output is available to later steps via
`{{step_name}}` or `{{previous}}` in that step's `system` / `user`
templates:

```json
{
  "messages": [{ "role": "user", "content": "Build a word-count CLI." }],
  "steps": [
    {
      "name": "plan",
      "model": "role:reasoner",
      "system": "Outline the approach.",
      "max_tokens": 512
    },
    {
      "name": "code",
      "model": "role:coder",
      "system": "Implement:\n\n{{plan}}"
    },
    {
      "name": "review",
      "model": "role:reasoner",
      "system": "Critique:\n\n{{code}}"
    }
  ]
}
```

See also the README walkthrough:
[Swarm in 60 seconds](../README.md#swarm-in-60-seconds).

### `POST /swarm/debate`

Multi-round. Each agent sees the others' previous-round answers, then a
judge selects:

```json
{
  "models":   ["role:reasoner", "role:fast", "role:coder"],
  "messages": [{ "role": "user", "content": "..." }],
  "rounds":   2,
  "judge":    "role:reasoner"
}
```

Defaults (timeouts, per-call budgets, judge model) are tunable via the
`SWARM_*` env vars. See `middle_layerMLX.py` for the full list until
Pass 2 lands the typed `Settings` object.

### Per-agent error metadata (LM Studio gateway)

When the LM Studio gateway (`middle_layer.py`) runs a swarm meta-model
(`swarmCouncil`, `swarmVote`, `swarm/fanout`, `swarm/pipeline`), every
candidate in the response carries structured error metadata so callers can
fail soft on a known-bad agent without parsing prose:

```jsonc
{
  "swarm": {
    "candidates": [
      {
        "agent_id": "role:reasoner",         // verbatim spec the caller sent
        "model":    "qwen3.5-122b-a10b",     // resolved id (or "?" on miss)
        "ok":           false,
        "error_kind":   "oom",               // see enum below
        "http_status":  400,                 // upstream status when known
        "error_detail": "Failed to load ...",// upstream payload, prefix-stripped
        "latency_ms":   312,
        "error":        "LM Studio 400: ..." // legacy prose, kept for back-compat
      }
    ]
  }
}
```

When **every** agent fails, the gateway returns HTTP 502 with both the
legacy prose summary and a structured `error_details` object:

```jsonc
{
  "error": "Swarm routing failed: all swarm agents failed: ...",
  "error_details": {
    "summary":       "all swarm agents failed",
    "agent_count":   3,
    "kinds":         { "oom": 2, "model_crashed": 1 },
    "upstream_statuses": { "400": 2, "500": 1 },
    "agents": [ /* one entry per failed candidate, same shape as above */ ]
  }
}
```

The response also sets the `X-Swarm-Error-Kinds` header (e.g.
`oom=2,model_crashed=1`) for clients that route on headers alone.

`error_kind` is a stable, finite enum. Treat unknown values as `"unknown"`:

| kind                 | meaning                                                      |
| -------------------- | ------------------------------------------------------------ |
| `no_models_loaded`   | LM Studio reachable but zero models loaded.                  |
| `model_not_resolved` | Spec didn't match anything loaded or installed.              |
| `oom`                | LM Studio refused to JIT-load (insufficient system resources).|
| `model_crashed`      | Loaded model crashed mid-call.                               |
| `empty_response`     | Upstream returned 200 with empty assistant `content`. Common with reasoning models when `max_tokens` is consumed entirely by `reasoning_content`. The full `response` is still on the candidate so callers can recover the chain-of-thought. |
| `timeout`            | `SWARM_PER_CALL_TIMEOUT` tripped.                            |
| `connection_error`   | Couldn't reach LM Studio.                                    |
| `upstream_4xx`       | Other 4xx from LM Studio / Anthropic.                        |
| `upstream_5xx`       | 5xx from upstream (Internal Server Error, HTML pages, …).    |
| `config_error`       | Local misconfiguration (missing API key, LiteLLM, …).        |
| `anthropic_error`    | Anthropic adapter raised a non-HTTP exception.               |
| `litellm_error`      | LiteLLM adapter raised a non-HTTP exception.                 |
| `unknown`            | Unclassified.                                                |

## 6. Stability and versioning

- The HTTP routes listed above are **stable across the 0.x line**.
  Pass-0 regression captures (`docs/_internal/baseline/` on
  `pass/0-discovery`) pin every response byte-for-byte.
- The **resolver grammar** in §1 is stable; new spec forms may be added
  but the listed forms will not be removed before 1.0.
- The **role registry** is operator data; you can change roles freely.
- The **profile schema** in §4 may gain new optional fields; existing
  fields will not change shape.
- **Automatic capability hints** in §3 may grow (e.g. `needs_tools`).
  Existing hints will not change behavior incompatibly.

A formal stability declaration lands in `docs/stability.md` in Pass 9.

## 7. Why this is a "protocol" and not just a router

The thing that makes Lattice different from a normal model router is
that **both sides of the boundary are abstract**:

- The **agent** does not name a model. It names a capability.
- The **operator** does not have to update agents when the model fleet
  changes. They update the registry.

The HTTP transport happens to be the OpenAI API, because that's what
agent frameworks already speak. The capability spec inside the `"model"`
field is the actual protocol. If a future standards body defines a
formal model-capability advertisement protocol, Lattice will adopt it
as an additional transport without breaking the OpenAI surface.

Until then, the spec in this document **is** the protocol.
