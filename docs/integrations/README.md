# Agent framework integrations

Local Lattice exposes the OpenAI HTTP API. Any agent framework that lets you
override the OpenAI base URL works with no SDK on Lattice's side.

The pattern is always the same:

1. Start Lattice locally: `local-lattice-mlx serve --host 127.0.0.1 --port 5001`.
2. Point your agent framework at `http://127.0.0.1:5001/v1` with whatever
   API key you configured via `MIDDLE_LAYER_API_KEY`.
3. In your agent code, replace specific model identifiers with capability
   specs:
   - `"role:coder"` instead of `"qwen2.5-coder-32b"`.
   - `"role:reasoner,role:coder,role:fast"` for tiered fallback.
   - `"role:reasoner,anthropic-claude-4.6-opus"` for hybrid local+cloud.
4. For multi-model patterns (vote, debate, pipeline, fanout), POST to
   `/swarm/*` directly instead of `/v1/chat/completions`.

Worked examples:

- [LangGraph](./langgraph.md)
- [OpenAI Agents SDK](./openai-agents-sdk.md)

The same pattern works for CrewAI, Pydantic AI, Mastra, Aider, Cline,
Continue.dev, and anything else that accepts a custom OpenAI base URL.
Contributions of additional worked examples are welcome — see
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## What the framework gets out of this

- The agent author writes `role:coder` once and never thinks about model
  identifiers again.
- The operator (or end user) controls the model fleet via
  [`mlx_roles.json`](../../mlx_roles.json) and
  [`model_profiles.json`](../../model_profiles.json) without modifying
  agent code.
- Vision routing, long-context routing, and latency-tier routing happen
  automatically based on the request — no glue code in the agent.
- Multi-model patterns (vote / debate / pipeline / fanout) are HTTP
  endpoints, not orchestration code the agent has to maintain.

See [`docs/capabilities.md`](../capabilities.md) for the full grammar
each integration is built on top of.
