"""Capability resolution for the LM Studio gateway (Pass 3 extraction).

Pure model-id resolution: placeholder detection, exact/substring matching,
role-registry lookups, comma-separated priority lists, ``*wildcard*``
patterns, and the prefer-loaded / strict-loaded policies described in
``docs/capabilities.md``.

No HTTP probing happens here. Callers pass the live ``available`` /
``loaded`` id lists plus a :class:`ResolverPolicy`; the gateway keeps the
LM Studio probes and the env parsing that builds the policy. This keeps
the module unit-testable without a running upstream.

Role-registry *loading* (``MODEL_ROLES_JSON`` / ``MODEL_ROLES_FILE`` / file
autodiscovery) also lives here because both gateways share the format, but
the functions take an explicit ``anchor_dir`` so discovery is relative to
the calling script, not this package.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# Built-in fallback registry, used only when no roles file or env override
# is present. Entries are substring-matched against the live inventory.
DEFAULT_MODEL_ROLES: dict[str, list[str]] = {
    "coder":    ["coder", "code"],
    "reasoner": ["72b", "70b", "qwen2.5", "llama-3.3", "deepseek-r1"],
    "fast":     ["3b", "7b", "phi", "mini", "small"],
    "vision":   ["vl", "vision", "llava"],
    "default":  [],
}


@dataclass(frozen=True)
class ResolverPolicy:
    """Snapshot of the gateway knobs the resolver needs.

    Built per-call by the gateway so tests (and the dashboard) can mutate
    the module-level globals and see the change take effect immediately.
    """

    roles: dict = field(default_factory=dict)
    prefer_loaded: bool = True
    strict_loaded: bool = False
    default_model: str = ""
    placeholder_ids: frozenset[str] = frozenset({"", "auto", "default"})


def is_placeholder(name: object, placeholder_ids: frozenset[str]) -> bool:
    """True when ``name`` is empty / a generic placeholder / a known cloud id."""
    if name is None:
        return True
    if not isinstance(name, str):
        return True
    return name.strip().lower() in placeholder_ids


def match_one(needle: str, haystack: Iterable[str]) -> str | None:
    """First id in ``haystack`` matching ``needle`` (exact then substring,
    case-insensitive)."""
    if not needle:
        return None
    n = needle.strip().lower()
    for mid in haystack:
        if mid.lower() == n:
            return mid
    for mid in haystack:
        if n in mid.lower():
            return mid
    return None


def resolve_role(
    role: str,
    available: Sequence[str],
    loaded: Sequence[str] | None = None,
    *,
    policy: ResolverPolicy,
) -> str | None:
    """First model id matching any preference for ``role``.

    When ``policy.prefer_loaded`` is on (default) and ``loaded`` is non-empty,
    every preference is first matched against the loaded subset; only if
    nothing in the loaded subset matches do we fall back to the full
    ``available`` set (which on LM Studio includes downloaded-but-not-loaded
    ids that would JIT-load on first call). This stops swarm fanouts from
    JIT-loading three different giant models in parallel and OOMing.

    When ``policy.strict_loaded`` is on AND ``loaded`` is non-empty, the
    fall-through is suppressed entirely — a role miss returns ``None`` rather
    than JIT-loading something the operator didn't explicitly stage.
    """
    prefs = policy.roles.get(role.lower(), [])
    if isinstance(prefs, str):
        prefs = [prefs]
    if policy.prefer_loaded and loaded:
        for p in prefs:
            m = match_one(p, loaded)
            if m:
                return m
        if policy.strict_loaded:
            return None
    for p in prefs:
        m = match_one(p, available)
        if m:
            return m
    return None


def resolve_model_id(
    requested: object,
    available: Sequence[str],
    loaded: Sequence[str] | None,
    *,
    policy: ResolverPolicy,
) -> tuple[str | None, str | None]:
    """Decide which model id to use for a request.

    Accepted shapes for ``requested``:
      None / "" / "auto" / "default" / "middleLayer"  -> auto-pick
      "exact-model-id"                                -> exact, else substring
      "a,b,c"                                         -> priority list (first match wins)
      "role:coder"                                    -> registry lookup
      "*coder*" / "qwen*"                             -> wildcard substring
      mix any of the above in a comma-separated list, e.g. "role:coder,qwen*"

    ``available`` must be the non-empty live inventory; probing is the
    caller's job. ``loaded`` may be ``None`` / empty when the upstream
    cannot report a loaded subset.

    Returns ``(model_id, error_message)``. On a soft miss (specific name
    asked but not loaded), error is non-None; the caller decides whether to
    fall back.
    """
    # In strict mode, once the upstream reports at least one loaded model we
    # *only* resolve against that set. The installed-but-not-loaded set is
    # what LM Studio would JIT-load behind the user's back, which is exactly
    # the behavior strict mode is meant to disable.
    strict = policy.strict_loaded and bool(loaded)

    if is_placeholder(requested, policy.placeholder_ids):
        if policy.default_model:
            if policy.prefer_loaded and loaded:
                m = match_one(policy.default_model, loaded)
                if m:
                    return m, None
            if not strict:
                m = match_one(policy.default_model, available)
                if m:
                    return m, None
        # Try the "default" role next, then first loaded id, then first available.
        m = resolve_role("default", available, loaded, policy=policy)
        if m:
            return m, None
        if policy.prefer_loaded and loaded:
            return loaded[0], None
        if strict:
            # Defensive — bool(loaded) implies non-empty so we already
            # returned above, but keep the guard so future edits can't sneak
            # an installed-set fallthrough past us.
            return None, "STRICT_LOADED_MODELS: no loaded LM Studio model available"
        return available[0], None

    candidates = [c.strip() for c in str(requested).split(",") if c.strip()]
    # First pass: loaded-only (if we have a loaded view).
    if policy.prefer_loaded and loaded:
        for cand in candidates:
            cand_lc = cand.lower()
            if cand_lc.startswith("role:"):
                m = resolve_role(cand_lc.split(":", 1)[1], available, loaded, policy=policy)
                if m:
                    return m, None
                continue
            needle = cand.replace("*", "") if "*" in cand else cand
            m = match_one(needle, loaded)
            if m:
                return m, None
        if strict:
            return None, (
                f"STRICT_LOADED_MODELS: '{requested}' did not match any "
                f"loaded LM Studio model. Loaded: {list(loaded)}"
            )
    # Second pass: full available set (allows JIT-load of installed ids).
    for cand in candidates:
        cand_lc = cand.lower()
        if cand_lc.startswith("role:"):
            m = resolve_role(cand_lc.split(":", 1)[1], available, loaded, policy=policy)
            if m:
                return m, None
            continue
        if "*" in cand:
            m = match_one(cand.replace("*", ""), available)
            if m:
                return m, None
            continue
        m = match_one(cand, available)
        if m:
            return m, None

    return None, f"No loaded LM Studio model matched '{requested}'. Available: {list(available)}"


# ---------------------------------------------------------------------------
# Role-registry loading
# ---------------------------------------------------------------------------

def autodiscover_roles_file(anchor_dir: str) -> str | None:
    """Look for ``lmstudio_roles.json`` (preferred) or ``mlx_roles.json`` in
    ``anchor_dir`` and one directory up, so users running the gateway script
    directly still get a tuned role registry without needing to set
    ``MODEL_ROLES_FILE``. Returns the absolute path or None.
    """
    parent = os.path.dirname(anchor_dir)
    candidates = [
        os.path.join(anchor_dir, "lmstudio_roles.json"),
        os.path.join(parent, "lmstudio_roles.json"),
        os.path.join(anchor_dir, "mlx_roles.json"),
        os.path.join(parent, "mlx_roles.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_model_roles(anchor_dir: str) -> tuple[dict, str | None]:
    """Resolve the role registry from env / files / built-in default.

    Precedence: ``MODEL_ROLES_JSON`` env > ``MODEL_ROLES_FILE`` env >
    autodiscovered file next to ``anchor_dir`` > ``DEFAULT_MODEL_ROLES``.
    Returns ``(roles, source_label)``.
    """
    raw = os.environ.get("MODEL_ROLES_JSON")
    if raw:
        try:
            return json.loads(raw), "MODEL_ROLES_JSON"
        except Exception as e:
            print(f"WARN: MODEL_ROLES_JSON is not valid JSON: {e}")
    path = os.environ.get("MODEL_ROLES_FILE")
    source = None
    if not path:
        path = autodiscover_roles_file(anchor_dir)
        if path:
            source = f"auto:{path}"
    else:
        source = f"env:{path}"
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f), source
        except Exception as e:
            print(f"WARN: cannot load MODEL_ROLES_FILE={path}: {e}")
    return dict(DEFAULT_MODEL_ROLES), "default"


__all__ = [
    "DEFAULT_MODEL_ROLES",
    "ResolverPolicy",
    "is_placeholder",
    "match_one",
    "resolve_role",
    "resolve_model_id",
    "autodiscover_roles_file",
    "load_model_roles",
]
