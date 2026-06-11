"""LM Studio resolver behavior: prefer-loaded ranking, role autodiscover.

Pinned regressions:
- The 'all swarm agents failed: ... insufficient system resources' cascade
  happened because role lookups picked a not-loaded giant when a smaller
  loaded id also matched. PREFER_LOADED_MODELS=1 must keep us on the
  loaded id even when a higher-priority role pref is downloadable but
  not-loaded.
- The 'process kept using DEFAULT_MODEL_ROLES' regression happened because
  MODEL_ROLES_FILE wasn't exported. Auto-discovery must find
  lmstudio_roles.json next to middle_layer.py without env help.
"""

from __future__ import annotations

from tests._helpers import _load_middle_layer


def test_resolve_role_prefers_loaded_id_over_not_loaded() -> None:
    """When LM Studio reports both a loaded small model and a not-loaded
    giant that also matches a role preference, the resolver must pick the
    loaded one. This is the exact failure mode that produced the
    'all swarm agents failed: ... insufficient system resources' cascade.
    """
    mod = _load_middle_layer()
    available = [
        "qwen3.5-122b-a10b",
        "nousresearch/hermes-4-70b",
        "qwen/qwen3-coder-next",
    ]
    loaded = ["qwen3.5-122b-a10b"]
    saved_roles, saved_pref = mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS
    try:
        # Reasoner pref puts the not-loaded giant FIRST so a naive resolver
        # would pick it; prefer-loaded must override that ordering.
        mod.MODEL_ROLES = {
            "reasoner": [
                "nousresearch/hermes-4-70b",
                "qwen3.5-122b-a10b",
            ],
            "coder": ["qwen/qwen3-coder-next"],
            "fast": ["qwen3.5-122b-a10b"],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = True
        mid = mod._resolve_role("reasoner", available, loaded=loaded)
        assert mid == "qwen3.5-122b-a10b", (
            f"prefer-loaded should keep us on the loaded id, got {mid!r}"
        )
        rid, err = mod.resolve_model_id("role:reasoner", available, loaded=loaded)
        assert err is None and rid == "qwen3.5-122b-a10b"
    finally:
        mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS = saved_roles, saved_pref


def test_resolve_role_falls_back_to_not_loaded_when_no_loaded_match() -> None:
    """If nothing in the loaded subset matches the role list, resolver may
    still return a not-loaded id (LM Studio will JIT it). We deliberately
    keep this fallback so callers asking for a specific 'role:coder' on a
    machine with no coder loaded still work.
    """
    mod = _load_middle_layer()
    available = ["qwen3.5-122b-a10b", "qwen/qwen3-coder-next"]
    loaded = ["qwen3.5-122b-a10b"]
    saved_roles, saved_pref = mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS
    try:
        mod.MODEL_ROLES = {
            "coder": ["qwen3-coder-next"],
            "reasoner": ["qwen3.5-122b-a10b"],
            "fast": [],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = True
        # Loaded set has no coder, so we should fall back to the not-loaded one.
        mid = mod._resolve_role("coder", available, loaded=loaded)
        assert mid == "qwen/qwen3-coder-next"
    finally:
        mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS = saved_roles, saved_pref


def test_prefer_loaded_disabled_uses_first_match() -> None:
    """With PREFER_LOADED_MODELS off, we keep the legacy 'first match in pref
    list wins' behavior. Important for users who explicitly want JIT.
    """
    mod = _load_middle_layer()
    available = [
        "qwen3.5-122b-a10b",
        "nousresearch/hermes-4-70b",
    ]
    loaded = ["qwen3.5-122b-a10b"]
    saved_roles, saved_pref = mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS
    try:
        mod.MODEL_ROLES = {
            "reasoner": ["nousresearch/hermes-4-70b", "qwen3.5-122b-a10b"],
            "coder": [],
            "fast": [],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = False
        mid = mod._resolve_role("reasoner", available, loaded=loaded)
        assert mid == "nousresearch/hermes-4-70b"
    finally:
        mod.MODEL_ROLES, mod.PREFER_LOADED_MODELS = saved_roles, saved_pref


def test_strict_loaded_models_never_falls_through_to_installed() -> None:
    """When PREFER_LOADED_MODELS=strict and LM Studio reports at least one
    loaded model, the resolver must refuse to return an installed-but-not-loaded
    id — neither for role lookups nor for explicit-id requests. This is the
    knob that makes MiddleLayer ``dynamically take what's in LM Studio``
    rather than silently asking LM Studio to JIT-load a different model.
    """
    mod = _load_middle_layer()
    available = [
        "qwen3.5-122b-a10b",
        "qwen/qwen3-coder-next",
        "gemma-4-31b-it",
    ]
    loaded = ["gemma-4-31b-it"]
    saved_roles = mod.MODEL_ROLES
    saved_pref = mod.PREFER_LOADED_MODELS
    saved_strict = mod.STRICT_LOADED_MODELS
    try:
        mod.MODEL_ROLES = {
            "coder": ["qwen/qwen3-coder-next"],
            "reasoner": ["qwen3.5-122b-a10b"],
            "fast": [],
            "default": ["qwen3.5-122b-a10b"],
        }
        mod.PREFER_LOADED_MODELS = True
        mod.STRICT_LOADED_MODELS = True

        # Role miss must NOT JIT-load the not-loaded coder.
        assert mod._resolve_role("coder", available, loaded=loaded) is None

        # role:coder request as a whole must error rather than fall through.
        rid, err = mod.resolve_model_id("role:coder", available, loaded=loaded)
        assert rid is None
        assert err is not None and "STRICT_LOADED_MODELS" in err

        # Explicit not-loaded id must also fail under strict mode.
        rid, err = mod.resolve_model_id(
            "qwen3.5-122b-a10b", available, loaded=loaded
        )
        assert rid is None
        assert err is not None and "STRICT_LOADED_MODELS" in err

        # Loaded id explicit request still works.
        rid, err = mod.resolve_model_id("gemma-4-31b-it", available, loaded=loaded)
        assert err is None and rid == "gemma-4-31b-it"

        # Placeholder with DEFAULT_MODEL pointing at a not-loaded id must
        # fall back to the first loaded id, not JIT the installed default.
        saved_default = mod.DEFAULT_MODEL
        try:
            mod.DEFAULT_MODEL = "qwen3.5-122b-a10b"
            rid, err = mod.resolve_model_id("auto", available, loaded=loaded)
            assert err is None and rid == "gemma-4-31b-it"
        finally:
            mod.DEFAULT_MODEL = saved_default
    finally:
        mod.MODEL_ROLES = saved_roles
        mod.PREFER_LOADED_MODELS = saved_pref
        mod.STRICT_LOADED_MODELS = saved_strict


def test_strict_loaded_models_falls_through_when_lmstudio_reports_no_loaded() -> None:
    """Strict mode is only meaningful when LM Studio actually reports a loaded
    set. If the /api/v0/models probe failed and loaded is empty, we keep the
    legacy fall-through so the proxy still works on older LM Studio builds.
    """
    mod = _load_middle_layer()
    available = ["qwen3.5-122b-a10b", "qwen/qwen3-coder-next"]
    saved_roles = mod.MODEL_ROLES
    saved_pref = mod.PREFER_LOADED_MODELS
    saved_strict = mod.STRICT_LOADED_MODELS
    try:
        mod.MODEL_ROLES = {
            "coder": ["qwen/qwen3-coder-next"],
            "reasoner": [],
            "fast": [],
            "default": [],
        }
        mod.PREFER_LOADED_MODELS = True
        mod.STRICT_LOADED_MODELS = True
        rid, err = mod.resolve_model_id("role:coder", available, loaded=[])
        assert err is None and rid == "qwen/qwen3-coder-next"
    finally:
        mod.MODEL_ROLES = saved_roles
        mod.PREFER_LOADED_MODELS = saved_pref
        mod.STRICT_LOADED_MODELS = saved_strict


def test_prefer_loaded_models_env_parses_strict_tokens(monkeypatch) -> None:
    """``PREFER_LOADED_MODELS=strict`` (and equivalents) must flip both the
    bool and the new STRICT_LOADED_MODELS flag when the module is imported.
    """
    import importlib.util
    from tests._helpers import REPO_ROOT

    path = REPO_ROOT / "middle_layer.py"

    for token in ("strict", "only", "2", "loaded-only", "loaded_only"):
        monkeypatch.setenv("PREFER_LOADED_MODELS", token)
        spec = importlib.util.spec_from_file_location(
            f"middle_layer_strict_{token}", path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.PREFER_LOADED_MODELS is True, token
        assert mod.STRICT_LOADED_MODELS is True, token

    for token in ("1", "true", "yes", "on"):
        monkeypatch.setenv("PREFER_LOADED_MODELS", token)
        spec = importlib.util.spec_from_file_location(
            f"middle_layer_lenient_{token}", path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.PREFER_LOADED_MODELS is True, token
        assert mod.STRICT_LOADED_MODELS is False, token

    for token in ("0", "false", "no", "off"):
        monkeypatch.setenv("PREFER_LOADED_MODELS", token)
        spec = importlib.util.spec_from_file_location(
            f"middle_layer_off_{token}", path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.PREFER_LOADED_MODELS is False, token
        assert mod.STRICT_LOADED_MODELS is False, token


def test_autodiscover_finds_lmstudio_roles_next_to_script() -> None:
    """If lmstudio_roles.json exists next to middle_layer.py,
    _load_model_roles must pick it up even when neither MODEL_ROLES_JSON
    nor MODEL_ROLES_FILE is set. Prevents the 'process never picked up the
    new file' regression.
    """
    mod = _load_middle_layer()
    found = mod._autodiscover_roles_file()
    # In this repo lmstudio_roles.json sits at the root, so discovery must
    # succeed (not None) and prefer it over mlx_roles.json when both exist.
    assert found is not None and found.endswith("lmstudio_roles.json"), found
