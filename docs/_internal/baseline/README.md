# Pass 0 Baseline — Regression Oracle

This folder contains raw `curl -isS` captures of every public route in
`middle_layerMLX.py` against a tiny MLX model. Future passes MUST replay
these requests and confirm the response payload (modulo timestamps,
request IDs, latency, and the `Date` header) is byte-for-byte identical.
Any divergence is a regression and should be reverted in the offending
commit.

## Capture environment

| Item                        | Value                                                              |
|-----------------------------|--------------------------------------------------------------------|
| Date                        | 2026-05-12                                                         |
| Host / Port                 | `127.0.0.1:5099`                                                   |
| Python                      | `3.14.4` (Homebrew, via existing `middle_layer_venv`)              |
| Server module               | `middle_layerMLX.py serve --no-pick-model`                         |
| Backend model               | `mlx-community/Nemotron-Mini-4B-Instruct-4bit-mlx` (smallest local)|
| `MLX_MODEL_ROOT`            | `~/.lmstudio/models` (auto-discovered)                             |
| `MIDDLE_LAYER_API_KEY`      | `baseline-test-key-do-not-use` (test-only)                         |
| `ANTHROPIC_AUTO_ROUTE`      | `0`                                                                |
| `MAX_CONCURRENT_MODELS`     | `1`                                                                |
| `MAX_PARALLEL_MODEL_CALLS`  | `1`                                                                |
| `MLX_PER_MODEL_INFLIGHT_CAP`| `1`                                                                |
| `DEFAULT_MAX_TOKENS`        | `64`                                                               |
| `MAX_TOKENS_CEILING`        | `256`                                                              |

The venv only had `flask` + `requests` initially; `mlx-lm`, `flask-cors`,
`litellm`, `huggingface_hub` were installed into the existing venv (no
source files modified).

## Captures

| #  | File                                  | Endpoint                              | Notes                                                  |
|----|---------------------------------------|---------------------------------------|--------------------------------------------------------|
| 01 | `01_healthz_unauth.txt`               | `GET /healthz` (no key)               | Returns **401**                                        |
| 02 | `02_v1_models_unauth.txt`             | `GET /v1/models` (no key)             | Returns **401**                                        |
| 03 | `03_healthz_authed.txt`               | `GET /healthz`                        | 200 + full status JSON                                 |
| 04 | `04_v1_models_authed.txt`             | `GET /v1/models`                      | 200 + 17 MLX aliases                                   |
| 05 | `05_swarm_models.txt`                 | `GET /swarm/models`                   | 200 + roles + defaults                                 |
| 06 | `06_dash_snapshot_cold.txt`           | `GET /dashboard/api/snapshot`         | Cold snapshot (no events)                              |
| 07 | `07_dash_config.txt`                  | `GET /dashboard/api/config`           | Dashboard public config                                |
| 08 | `08_dash_html.txt`                    | `GET /dashboard/`                     | Static HTML; **no auth required**                      |
| 09 | `09_bearer_auth.txt`                  | `GET /healthz` (Bearer)               | Bearer header path                                     |
| 10 | `10_options_preflight.txt`            | `OPTIONS /v1/chat/completions`        | **No CORS handler** when `CORS_ORIGINS=""`             |
| 11 | `11_chat_nonstream.txt`               | `POST /v1/chat/completions`           | Tiny prompt → `"PONG"` (1.7s, MLX cold load)           |
| 12 | `12_chat_stream.txt`                  | `POST /v1/chat/completions stream=true` | SSE frames + `[DONE]`                                |
| 13 | `13_completions.txt`                  | `POST /v1/completions`                | Legacy completion shape                                |
| 14 | `14_chat_bad_alias_fallback.txt`      | bogus model → fallback                | `X-Model-Resolution: fallback (...)` header present    |
| 15 | `15_chat_role_fast.txt`               | `model: role:fast`                    | Resolved via `mlx_roles.json`                          |
| 16 | `16_swarm_fanout.txt`                 | `POST /swarm/fanout`                  | Schema: `models`/`messages`                            |
| 17 | `17_swarm_vote.txt`                   | `POST /swarm/vote` `first-success`    | Returns OpenAI `chat.completion` shape                 |
| 18 | `18_swarm_pipeline.txt`               | `POST /swarm/pipeline`                | Single-step pipeline                                   |
| 19 | `19_swarm_debate.txt`                 | `POST /swarm/debate` (rounds=1)       | Requires `len(models) >= 2`                            |
| 20 | `20_dash_snapshot_warm.txt`           | dashboard after activity              | Events accumulated                                     |
| 21 | `21_dash_preferences_post.txt`        | `POST /dashboard/api/preferences`     | Save runtime default model + presets                   |
| 22 | `22_unload_model.txt`                 | `DELETE /v1/models/<alias>`           | 404 when not currently resident                        |
| 23 | `23_chat_latency_header.txt`          | `X-MLX-Latency-Tier: fast`            | Capability-aware routing                               |
| 24 | `24_chat_unauth.txt`                  | unauth chat                           | Returns **401**                                        |
| 25 | `25_chat_query_apikey.txt`            | `?api_key=...` query param            | **Ignored** — returns 401 (good)                       |
| 26 | `26_chat_oversize_prompt.txt`         | 60 000-char prompt                    | **Accepted** (no MAX_CONTENT_LENGTH; 30011 prompt tok) |
| 27 | `27_dash_snapshot_final.txt`          | dashboard after all activity          | Events list capped at MLX_DASHBOARD_MAX_EVENTS         |

## Notable behaviors observed (Pass 0 — record only, do not change)

1. Auth via `X-API-Key` header **and** `Authorization: Bearer` are both
   accepted; query-string `?api_key=` is correctly ignored (returns 401).
2. `ON_MODEL_MISS=fallback` resolved a bogus alias to
   `mlx-community/Qwen3.6-27B-8bit` (first registered alias),
   **not** to `DEFAULT_MODEL`. Confirm with maintainers whether this is
   intentional before changing in any later pass.
3. `MAX_CONCURRENT_MODELS=1` causes LRU eviction between requests — a
   `DELETE /v1/models/<alias>` immediately after a different model was
   used returns 404 because the original was already evicted.
4. `OPTIONS` preflight without `CORS_ORIGINS` set returns 405 — there is
   no global preflight handler. Browser-based clients without
   `CORS_ORIGINS` can not invoke chat completions cross-origin.
5. The dashboard HTML and static assets are intentionally
   **auth-exempt** (only `/dashboard/api/*` is gated). This is a
   product decision — Pass 4 should add `Cache-Control: no-store` and a
   strict `Content-Security-Policy`, but should not add auth to the
   static index.
6. Oversized prompts (>= 30 000 tokens) are accepted with no
   `Content-Length` cap, then handled by `MLX_CONTEXT_OVER_BUDGET`
   (`error` default). This is the largest behavior change Pass 4 will
   introduce.
7. Streaming responses always emit a final `data: [DONE]\n\n` sentinel,
   even on error. Future passes must preserve this.
