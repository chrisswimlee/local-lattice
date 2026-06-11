"""Direct tests for ``middle_layer.lmstudio_client`` (the Pass-3 extraction).

The gateway-level behavior stays pinned by tests/test_smoke.py and
tests/test_concurrency.py (which monkey-patch the module-level wrappers in
middle_layer.py); these tests exercise the client object directly with a
faked ``requests`` so no LM Studio is needed.
"""

from __future__ import annotations

import requests as real_requests

from middle_layer import lmstudio_client
from middle_layer.lmstudio_client import LMStudioClient


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: object = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _client(**kw) -> LMStudioClient:
    return LMStudioClient("http://127.0.0.1:9999", **kw)


def test_get_model_ids_parses_openai_shape_and_dedupes(monkeypatch) -> None:
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse(
            payload={"data": [{"id": "m1"}, {"id": "m2"}, {"id": "m1"}, {"notid": True}]}
        )

    monkeypatch.setattr(lmstudio_client.requests, "get", fake_get)
    c = _client()
    ids, err = c.get_model_ids()
    assert err is None and ids == ["m1", "m2"]
    assert calls == ["http://127.0.0.1:9999/v1/models"]


def test_get_model_ids_parses_loaded_instances_shape(monkeypatch) -> None:
    payload = {
        "models": [
            {"id": "base-a", "loaded_instances": [{"id": "inst-a1"}, {"id": "inst-a2"}]},
            {"id": "base-b", "loaded_instances": []},
        ]
    }
    monkeypatch.setattr(
        lmstudio_client.requests, "get", lambda url, timeout: _FakeResponse(payload=payload)
    )
    ids, err = _client().get_model_ids()
    assert err is None and ids == ["inst-a1", "inst-a2", "base-b"]


def test_get_model_ids_cache_and_force_refresh(monkeypatch) -> None:
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse(payload={"data": [{"id": f"m{len(calls)}"}]})

    monkeypatch.setattr(lmstudio_client.requests, "get", fake_get)
    c = _client(model_list_ttl=3600)
    assert c.get_model_ids() == (["m1"], None)
    # Cached: no second HTTP call, same result.
    assert c.get_model_ids() == (["m1"], None)
    assert len(calls) == 1
    # force_refresh busts the cache.
    assert c.get_model_ids(force_refresh=True) == (["m2"], None)
    assert len(calls) == 2


def test_get_model_ids_connection_error(monkeypatch) -> None:
    def fake_get(url, timeout):
        raise real_requests.exceptions.ConnectionError()

    monkeypatch.setattr(lmstudio_client.requests, "get", fake_get)
    ids, err = _client().get_model_ids()
    assert ids == []
    assert err is not None and "Cannot connect" in err


def test_loaded_probe_404_disables_endpoint_stickily(monkeypatch) -> None:
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(lmstudio_client.requests, "get", fake_get)
    c = _client()
    assert c.get_loaded_model_ids() == ([], None)
    assert c.loaded_endpoint_supported is False
    # Second call must not probe again.
    assert c.get_loaded_model_ids() == ([], None)
    assert len(calls) == 1


def test_loaded_probe_filters_state_loaded(monkeypatch) -> None:
    payload = {
        "data": [
            {"id": "loaded-1", "state": "loaded"},
            {"id": "not-loaded", "state": "not-loaded"},
            {"id": "loaded-1", "state": "loaded"},  # dupe
            {"id": "text-embedding-nomic", "state": "loaded"},
        ]
    }
    monkeypatch.setattr(
        lmstudio_client.requests, "get", lambda url, timeout: _FakeResponse(payload=payload)
    )
    c = _client()
    assert c.get_loaded_model_ids() == (["loaded-1", "text-embedding-nomic"], None)
    # Chat-capable variant drops the embedding id.
    c2 = _client()
    monkeypatch.setattr(
        lmstudio_client.requests, "get", lambda url, timeout: _FakeResponse(payload=payload)
    )
    assert c2.get_loaded_chat_capable_model_ids() == (["loaded-1"], None)


def test_chat_completion_payload_and_errors(monkeypatch) -> None:
    seen = {}

    def fake_post(url, headers, data, timeout):
        import json

        seen["url"] = url
        seen["payload"] = json.loads(data)
        seen["timeout"] = timeout
        return _FakeResponse(payload={"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(lmstudio_client.requests, "post", fake_post)
    c = _client(default_chat_timeout=42)
    resp, err = c.chat_completion(
        "m1", [{"role": "user", "content": "x"}], max_tokens=7, temperature=None
    )
    assert err is None and resp is not None
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["payload"] == {
        "model": "m1",
        "messages": [{"role": "user", "content": "x"}],
        "stream": False,
        "max_tokens": 7,
    }
    assert seen["timeout"] == 42

    # Upstream HTTP error becomes (None, "LM Studio <code>: ...").
    monkeypatch.setattr(
        lmstudio_client.requests,
        "post",
        lambda url, headers, data, timeout: _FakeResponse(status_code=400, text="bad request"),
    )
    resp, err = c.chat_completion("m1", [])
    assert resp is None
    assert err is not None and err.startswith("LM Studio 400")

    # Timeout maps to the stable string the swarm error classifier keys on.
    def fake_post_timeout(url, headers, data, timeout):
        raise real_requests.exceptions.Timeout()

    monkeypatch.setattr(lmstudio_client.requests, "post", fake_post_timeout)
    resp, err = c.chat_completion("m1", [])
    assert resp is None and err == "Timeout calling LM Studio"
