# OpenAI Agents SDK

The [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) accepts
a custom client. Pointing it at Lattice is a two-line change.

## Setup

```bash
pip install openai-agents
```

Start Lattice in another shell:

```bash
export MIDDLE_LAYER_API_KEY="$(uuidgen)"
local-lattice-mlx serve --host 127.0.0.1 --port 5001
```

## Minimal example: capability routing

```python
import os
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

client = AsyncOpenAI(
    base_url="http://127.0.0.1:5001/v1",
    api_key=os.environ["MIDDLE_LAYER_API_KEY"],
)

agent = Agent(
    name="coder",
    instructions="You write small, correct Python functions.",
    model=OpenAIChatCompletionsModel(
        model="role:coder",
        openai_client=client,
    ),
)

result = Runner.run_sync(agent, "Write a function that returns the Nth Fibonacci number.")
print(result.final_output)
```

Notice the `model="role:coder"`. The agent doesn't know which coder model
is actually loaded — Lattice picks the best available one from the role
registry. If the user adds a better coder model to their fleet tomorrow,
the agent picks it up without any code change.

## Tiered fallback example

```python
model=OpenAIChatCompletionsModel(
    model="role:reasoner,role:coder,role:fast",
    openai_client=client,
)
```

Lattice tries `role:reasoner` first; if none is loaded, it tries
`role:coder`, then `role:fast`. The first available wins.

## Hybrid local + cloud

```python
model=OpenAIChatCompletionsModel(
    model="role:reasoner,anthropic-claude-4.6-opus",
    openai_client=client,
)
```

Requires `ANTHROPIC_API_KEY` set in Lattice's environment. Local first;
Claude only if no local reasoner is available (or if Lattice's
auto-escalation policy decides the request needs it — see
`ANTHROPIC_AUTO_ROUTE`).

## Multi-model patterns (vote, debate, pipeline)

The OpenAI Agents SDK doesn't have a native "ensemble" concept, but
Lattice exposes one via HTTP. The simplest pattern is to call
`/swarm/vote` directly from a tool and return the result to the agent:

```python
import httpx
from agents import function_tool

@function_tool
async def ensemble_answer(question: str) -> str:
    """Ask three local models and let a judge pick the best answer."""
    async with httpx.AsyncClient() as http:
        r = await http.post(
            "http://127.0.0.1:5001/swarm/vote",
            headers={"X-API-Key": os.environ["MIDDLE_LAYER_API_KEY"]},
            json={
                "models":   ["role:reasoner", "role:coder", "role:fast"],
                "judge":    "role:reasoner",
                "messages": [{"role": "user", "content": question}],
            },
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
```

Now any agent that has `ensemble_answer` as a tool can opt into
multi-model voting for hard questions, without the agent author writing
any orchestration code.

## See also

- [`docs/capabilities.md`](../capabilities.md) — full capability spec.
- [`docs/why-lattice.md`](../why-lattice.md) — why this pattern exists.
- [LangGraph integration](./langgraph.md) — same pattern, different framework.
