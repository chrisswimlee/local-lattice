# LangGraph

[LangGraph](https://github.com/langchain-ai/langgraph) uses LangChain's
`ChatOpenAI` (or any other chat-model class) as its model interface. Pointing
it at Lattice is a two-line change.

## Setup

```bash
pip install langgraph langchain-openai
```

Start Lattice in another shell:

```bash
export MIDDLE_LAYER_API_KEY="$(uuidgen)"
local-lattice-mlx serve --host 127.0.0.1 --port 5001
```

## Minimal example: capability routing

```python
import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    model="role:coder",
    base_url="http://127.0.0.1:5001/v1",
    api_key=os.environ["MIDDLE_LAYER_API_KEY"],
)

agent = create_react_agent(llm, tools=[])

print(agent.invoke({"messages": [("user", "Write a Python function that reverses a string.")]}))
```

`model="role:coder"` makes the agent capability-aware. Lattice routes to
whatever coding model the operator has loaded. The agent code doesn't
change when the model fleet changes.

## Per-node capability routing

A real LangGraph graph often wants different capabilities at different
nodes — a fast model for routing, a coder for code generation, a
reasoner for review. With Lattice, each node just declares what it
needs:

```python
fast      = ChatOpenAI(model="role:fast",     base_url=BASE, api_key=KEY)
coder     = ChatOpenAI(model="role:coder",    base_url=BASE, api_key=KEY)
reasoner  = ChatOpenAI(model="role:reasoner", base_url=BASE, api_key=KEY)

def route(state):    return fast.invoke(state["messages"])
def write(state):    return coder.invoke(state["messages"])
def review(state):   return reasoner.invoke(state["messages"])
```

No hardcoded model IDs. The graph runs unchanged on any operator's
fleet, as long as their `mlx_roles.json` covers the roles you used.

## Tiered fallback

```python
llm = ChatOpenAI(
    model="role:reasoner,role:coder,role:fast",
    base_url="http://127.0.0.1:5001/v1",
    api_key=os.environ["MIDDLE_LAYER_API_KEY"],
)
```

Lattice tries each spec in order; first available wins. Useful for
agents that need to degrade gracefully across machines with different
fleets.

## Multi-model patterns (vote / debate / pipeline)

LangGraph's strength is graph composition, so vote/debate/pipeline could
be hand-rolled as a subgraph. But for simple cases, it's cleaner to call
the Lattice endpoint directly from a node:

```python
import httpx

def vote_node(state):
    r = httpx.post(
        "http://127.0.0.1:5001/swarm/vote",
        headers={"X-API-Key": os.environ["MIDDLE_LAYER_API_KEY"]},
        json={
            "models":   ["role:reasoner", "role:coder", "role:fast"],
            "judge":    "role:reasoner",
            "messages": [{"role": "user", "content": state["question"]}],
        },
        timeout=120,
    )
    r.raise_for_status()
    return {"answer": r.json()["choices"][0]["message"]["content"]}
```

The node returns a single answer; the graph upstream/downstream doesn't
need to know three models were involved.

## See also

- [`docs/capabilities.md`](../capabilities.md) — full capability spec.
- [`docs/why-lattice.md`](../why-lattice.md) — why this pattern exists.
- [OpenAI Agents SDK integration](./openai-agents-sdk.md) — same pattern, different framework.
