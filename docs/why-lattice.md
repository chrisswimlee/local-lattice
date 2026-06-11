# Why Local Lattice

## Stop hardcoding model names in your agents

If you've built anything with an agent framework in the last year, you have
this line somewhere:

```python
model = "gpt-4o"                          # or "claude-4.6-sonnet"
# ...or:
model = "qwen2.5-coder-32b-instruct"     # if you're running locally
```

It works. It also breaks the moment any of these change:

- The user runs your agent on a machine with different models on disk.
- The operator wants to swap providers without redeploying the agent.
- A 7B local model could have handled the request, but the agent
  hardcoded the 70B and you burned 10× the compute for no reason.
- One model wasn't enough — you needed a second opinion, or a coder to
  draft and a reasoner to review.

Agents should not name models. They should declare **what they need**, and
the infrastructure should pick.

## What changes with a capability layer

With Lattice in the middle, the agent's "model" field stops being a model
identifier and starts being a capability spec:

```python
model = "role:coder"
# ...or a priority list with fallback:
model = "role:reasoner,role:coder,role:fast"
# ...or hybrid with cloud as backstop:
model = "role:reasoner,anthropic-claude-4.6-opus"
```

The same agent code now:

- Runs against whatever models the user has installed (because Lattice
  reads the live disk).
- Picks a small/fast model when the request is small, a long-context
  model when the prompt is large, and a vision model when an image is
  attached — **automatically**, without the agent author thinking about it.
- Falls back to cloud only when no local model can do the job.
- Can ask for a vote, a debate, or a pipeline of models by hitting
  `/swarm/vote`, `/swarm/debate`, or `/swarm/pipeline` instead of
  `/v1/chat/completions`.

That last point is the one that matters most. The agent author writes the
agent once. The operator (or end user) tunes their model fleet
independently. Neither has to know about the other.

## Why this is becoming urgent

Three things are happening at once in 2026:

1. **Apple Silicon caught up to "good enough for serious agents."** An
   M4 Max can run a 70B-class model at conversational speed. A year ago
   that was a $40k H100. The economic case for local inference flipped.
2. **API costs at agent scale are punishing.** A real agent loop is 50
   to 500 LLM calls per session. At fleet scale, that's a real bill.
   Teams want to push the cheap parts of the loop local.
3. **Privacy regulation is forcing data-on-device.** EU AI Act, state
   privacy laws, healthcare/legal/finance — for a growing set of
   workloads, "send it to OpenAI" is simply not allowed.

The result is that every agent framework user — LangGraph, CrewAI,
OpenAI Agents SDK, Mastra, Pydantic AI, Aider, Cline, Continue.dev,
Claude Code — is going to want a clean, **framework-agnostic** way to
say "use the best local model for this job, fall back to cloud if you
must."

Nobody has shipped that abstraction cleanly. Ollama is a model runner.
LM Studio is a desktop app. LiteLLM normalizes cloud providers. Lattice
is the missing piece: the **capability layer between agents and LLM
compute**, with local-first defaults.

## Why "swarm" is part of the same idea, not a separate feature

Once the agent stops naming models, asking for **more than one** is just
a different capability. You don't say "call GPT-4 and Claude and merge
them." You say:

```http
POST /swarm/vote
{
  "models":   ["role:reasoner", "role:coder", "role:fast"],
  "judge":    "role:reasoner",
  "messages": [...]
}
```

…and Lattice picks the three best available models for each role, runs
them in parallel, and lets the judge pick. The agent author wrote zero
lines of orchestration code.

Self-consistency, mixture-of-agents, debate, and chain-of-thought
ensembling are all well-established techniques for getting GPT-4-class
quality from smaller models. They've been research papers for years.
They're hard to use because every agent framework leaves the wiring as
an exercise. Lattice productizes them as HTTP endpoints.

See [`docs/capabilities.md`](./capabilities.md) for the formal spec and
[`docs/integrations/`](./integrations/) for drop-in examples in
LangGraph and the OpenAI Agents SDK.

## What Lattice is *not*

- It is not another model runner. It runs **in front of** `mlx_lm`, LM
  Studio, or a future vLLM/llama.cpp backend.
- It is not a cloud router. It can escalate to Anthropic when asked,
  but the local fleet is the primary citizen.
- It is not an agent framework. It is the thing your agent framework
  points at via `OPENAI_BASE_URL=http://localhost:5001/v1`.
- It is not opinionated about which models you run. The role registry
  is a JSON file you own.

## Where this is going

The 0.x line of Lattice is about getting the capability protocol stable,
secure, and well-integrated with the major agent frameworks. 1.0
declares the HTTP surface and the resolver grammar formally frozen.

Beyond 1.0, the interesting work is at the protocol layer:

- Additional backends (vLLM for CUDA, llama.cpp for cross-platform).
- A capability advertisement endpoint so agents can negotiate what's
  available before sending a request.
- Tool-calling capability hints (`needs_tools`, JSON-mode, function
  format normalization across backends).
- Hosted Lattice for teams that want managed routing without running
  their own GPU/Mac fleet.

The HTTP surface and the resolver grammar will not change incompatibly
to get there.
