"""LM Studio HTTP client (Pass 3 extraction).

Everything that talks HTTP to a LM Studio server lives here: the
installed-models probe (``/v1/models``), the loaded-models probe
(``/api/v0/models``, with graceful degradation on older builds), the
chat-completion call, and the embedding-id heuristic used to keep swarm
fanouts off non-chat models.

The client is an object (not module globals) so the probe caches are
per-instance and unit tests can construct throwaway clients against fake
servers. The legacy gateway (``middle_layer.py``) instantiates one client
and keeps thin module-level wrappers so historical monkey-patching of
``mod.get_lmstudio_model_ids`` et al. keeps working.
"""

from __future__ import annotations

import json
import re
import time

import requests

# Heuristic for filtering embedding / non-chat-capable model ids out of
# swarm fanouts. LM Studio's /api/v0/models *does* carry a ``type`` field
# (``llm``/``vlm``/``embeddings``), but the older ``/v1/models`` endpoint
# doesn't, and we want this filter to work regardless of which probe the
# loaded-ids list came from. Matches the common embedding-model naming
# conventions we see on LM Studio (``text-embedding-...``,
# ``nomic-embed-...``, ``bge-...``, ``e5-...``, anything with a literal
# ``embed`` token in its id segment). Erring on the side of false positives
# is acceptable here — operators can always pass an explicit
# ``swarm.models`` list to bypass the filter.
_EMBED_ID_HINT = re.compile(
    r"(?ix) (?:^|[/_-]) (?:embed|embedding|embeddings|nomic-embed|bge|e5) (?:[/_-]|$)"
)


def is_chat_capable_model_id(model_id: object) -> bool:
    """Best-effort guess at whether a loaded LM Studio model id can serve
    ``/v1/chat/completions``. Used by the ``auto``-expansion of
    ``SWARM_CHAT_DEFAULT_MODELS`` so swarm fanouts don't waste a slot on
    embedding models (which return 400 for chat completions and would just
    increment the failure column).
    """
    if not isinstance(model_id, str):
        return False
    return _EMBED_ID_HINT.search(model_id) is None


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


class LMStudioClient:
    """Probe-and-call client for one LM Studio server.

    ``model_list_ttl`` briefly caches both model-list probes so swarm
    fanouts don't hammer the API. Once ``/api/v0/models`` is observed to
    404 (older LM Studio), the loaded probe is disabled for the lifetime
    of the client (``loaded_endpoint_supported``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_list_ttl: float = 30.0,
        probe_timeout: float = 5.0,
        default_chat_timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_list_ttl = model_list_ttl
        self.probe_timeout = probe_timeout
        self.default_chat_timeout = default_chat_timeout
        self.loaded_endpoint_supported: bool = True
        self._cached_model_ids: list[str] | None = None
        self._cached_model_ids_ts: float = 0.0
        self._cached_loaded_ids: list[str] | None = None
        self._cached_loaded_ids_ts: float = 0.0

    # -- installed/visible models (``/v1/models``) --------------------------

    def get_model_ids(self, force_refresh: bool = False) -> tuple[list[str], str | None]:
        """Return ``(list_of_model_ids, error_message)``. Lists every model id
        the LM Studio server reports, in server order. Briefly cached
        (``model_list_ttl``) so swarm fanouts don't hammer the API.
        """
        now = time.time()
        if (
            not force_refresh
            and self._cached_model_ids is not None
            and (now - self._cached_model_ids_ts) < self.model_list_ttl
        ):
            return list(self._cached_model_ids), None

        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=self.probe_timeout)
            if response.status_code != 200:
                return [], f"LM Studio models endpoint returned {response.status_code}"

            data = response.json()
            ids: list[str] = []

            # Shape A: OpenAI-like {"data":[{"id":"..."}]}
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                for entry in data["data"]:
                    if isinstance(entry, dict) and entry.get("id"):
                        ids.append(entry["id"])

            # Shape B: {"models":[{"loaded_instances":[{"id":"..."}], "id": "..."}]}
            if not ids and isinstance(data, dict) and isinstance(data.get("models"), list):
                for entry in data["models"]:
                    if not isinstance(entry, dict):
                        continue
                    loaded = entry.get("loaded_instances", [])
                    if isinstance(loaded, list) and loaded:
                        for inst in loaded:
                            if isinstance(inst, dict) and inst.get("id"):
                                ids.append(inst["id"])
                    elif entry.get("id"):
                        ids.append(entry["id"])

            deduped = _dedupe(ids)
            self._cached_model_ids = deduped
            self._cached_model_ids_ts = now
            return list(deduped), None

        except requests.exceptions.ConnectionError:
            return [], "Cannot connect to LM Studio. Is it running?"
        except requests.exceptions.Timeout:
            return [], "Timeout connecting to LM Studio."
        except Exception as e:  # noqa: BLE001
            return [], f"Error discovering models: {str(e)}"

    # -- truly-loaded models (``/api/v0/models``) ----------------------------

    def get_loaded_model_ids(self, force_refresh: bool = False) -> tuple[list[str], str | None]:
        """Return ``(loaded_ids, error)`` using LM Studio's ``/api/v0/models``
        endpoint (which exposes per-instance ``state``). Degrades to
        ``([], None)`` once the endpoint is observed to be unsupported
        (older LM Studio); the caller should then treat all installed ids
        as candidates.
        """
        if not self.loaded_endpoint_supported:
            return [], None

        now = time.time()
        if (
            not force_refresh
            and self._cached_loaded_ids is not None
            and (now - self._cached_loaded_ids_ts) < self.model_list_ttl
        ):
            return list(self._cached_loaded_ids), None

        try:
            response = requests.get(
                f"{self.base_url}/api/v0/models", timeout=self.probe_timeout
            )
        except requests.exceptions.ConnectionError:
            return [], "Cannot connect to LM Studio. Is it running?"
        except requests.exceptions.Timeout:
            return [], "Timeout connecting to LM Studio."
        except Exception as e:  # noqa: BLE001
            return [], f"Error discovering loaded models: {str(e)}"

        if response.status_code == 404:
            # Older LM Studio without the /api/v0 surface. Stop probing.
            self.loaded_endpoint_supported = False
            self._cached_loaded_ids = []
            self._cached_loaded_ids_ts = now
            return [], None
        if response.status_code != 200:
            return [], f"LM Studio /api/v0/models returned {response.status_code}"

        try:
            data = response.json()
        except Exception as e:  # noqa: BLE001
            return [], f"Error parsing LM Studio /api/v0/models: {e}"

        loaded: list[str] = []
        for entry in (data.get("data") or []) if isinstance(data, dict) else []:
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == "loaded" and entry.get("id"):
                loaded.append(entry["id"])

        deduped = _dedupe(loaded)
        self._cached_loaded_ids = deduped
        self._cached_loaded_ids_ts = now
        return list(deduped), None

    def get_loaded_chat_capable_model_ids(
        self, force_refresh: bool = False
    ) -> tuple[list[str], str | None]:
        """Loaded-ids list filtered to chat-capable ids only (see
        :func:`is_chat_capable_model_id`)."""
        ids, err = self.get_loaded_model_ids(force_refresh=force_refresh)
        if err:
            return ids, err
        return [m for m in ids if is_chat_capable_model_id(m)], None

    def get_current_model(self) -> tuple[str | None, str | None]:
        """Backwards-compatible single-model accessor: first truly-loaded id
        when ``/api/v0/models`` is reachable, else first installed id."""
        loaded, lerr = self.get_loaded_model_ids()
        if not lerr and loaded:
            return loaded[0], None
        ids, err = self.get_model_ids()
        if err:
            return None, err
        if not ids:
            return None, "No model is loaded in LM Studio."
        return ids[0], None

    # -- inference -----------------------------------------------------------

    def chat_completion(
        self, model_id: str, messages: list, **kwargs: object
    ) -> tuple[dict | None, str | None]:
        """Call LM Studio ``/v1/chat/completions`` for a single model.
        Returns ``(openai_shaped_response_json, error_str)``."""
        payload: dict = {"model": model_id, "messages": messages, "stream": False}
        for k in ("max_tokens", "temperature", "top_p", "stop"):
            if kwargs.get(k) is not None:
                payload[k] = kwargs[k]

        try:
            r = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
                timeout=kwargs.get("timeout", self.default_chat_timeout),  # type: ignore[arg-type]
            )
            if r.status_code >= 400:
                return None, f"LM Studio {r.status_code}: {r.text[:300]}"
            return r.json(), None
        except requests.exceptions.Timeout:
            return None, "Timeout calling LM Studio"
        except Exception as e:  # noqa: BLE001
            return None, f"Error calling LM Studio: {e}"


__all__ = [
    "LMStudioClient",
    "is_chat_capable_model_id",
]
