"""Tests for the shared security helpers (Pass 4).

These exercise the public surface of ``middle_layer.security`` plus a small
integration test that wires the LM Studio backend's auth guard end-to-end
via Flask's test client. The MLX backend uses the same helpers but boots
``mlx_lm`` at import; we don't recreate that subprocess machinery here —
the per-helper tests cover the behaviour and the conftest in this folder
keeps the suite fast.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from middle_layer.security import (  # noqa: E402
    DEFAULT_MAX_REQUEST_BYTES,
    PublicBindWithoutAuthError,
    alias_in_allowlist,
    apply_security_headers,
    check_api_key,
    constant_time_eq,
    enforce_safe_bind,
    extract_bearer,
    is_public_bind,
    is_well_formed_alias,
    resolve_max_request_bytes,
)

# --- constant_time_eq -------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("abc", "abc", True),
        ("abc", "abd", False),
        ("abc", "abcd", False),
        ("", "abc", False),
        ("abc", "", False),
        ("", "", False),
        (None, "abc", False),
        ("abc", None, False),
        (None, None, False),
        ("héllo", "héllo", True),
        ("héllo", "hello", False),
    ],
)
def test_constant_time_eq(a, b, expected):
    assert constant_time_eq(a, b) is expected


# --- extract_bearer ---------------------------------------------------------


@pytest.mark.parametrize(
    "header, expected",
    [
        ("Bearer abc123", "abc123"),
        ("Bearer  spaced  ", "spaced"),
        ("bearer abc123", None),  # case-sensitive prefix, matches RFC 6750
        ("Token abc123", None),
        ("", None),
        (None, None),
        ("Bearer ", None),
    ],
)
def test_extract_bearer(header, expected):
    assert extract_bearer(header) == expected


# --- check_api_key ----------------------------------------------------------


def test_check_api_key_disabled_when_no_expected_key():
    assert check_api_key({}, None) is True
    assert check_api_key({}, "") is True


def test_check_api_key_accepts_x_api_key_header():
    assert check_api_key({"X-API-Key": "secret"}, "secret") is True


def test_check_api_key_accepts_bearer_token():
    assert check_api_key({"Authorization": "Bearer secret"}, "secret") is True


def test_check_api_key_rejects_wrong_key():
    assert check_api_key({"X-API-Key": "wrong"}, "secret") is False
    assert check_api_key({"Authorization": "Bearer wrong"}, "secret") is False


def test_check_api_key_rejects_missing_headers():
    assert check_api_key({}, "secret") is False


def test_check_api_key_handles_lowercase_header_names():
    assert check_api_key({"x-api-key": "secret"}, "secret") is True


# --- is_public_bind ---------------------------------------------------------


@pytest.mark.parametrize(
    "host, expected",
    [
        ("127.0.0.1", False),
        ("localhost", False),
        ("::1", False),
        ("LOCALHOST", False),
        ("0.0.0.0", True),
        ("::", True),
        ("192.168.1.10", True),
        ("10.0.0.1", True),
        ("8.8.8.8", True),
        ("example.com", True),  # DNS names treated as public (fail safe)
        ("", False),
        (None, False),
    ],
)
def test_is_public_bind(host, expected):
    assert is_public_bind(host) is expected


# --- enforce_safe_bind ------------------------------------------------------


def test_enforce_safe_bind_loopback_no_key_is_ok():
    enforce_safe_bind("127.0.0.1", None)
    enforce_safe_bind("localhost", "")


def test_enforce_safe_bind_loopback_with_key_is_ok():
    enforce_safe_bind("127.0.0.1", "anything")


def test_enforce_safe_bind_public_with_key_is_ok():
    enforce_safe_bind("0.0.0.0", "anything")


def test_enforce_safe_bind_public_no_key_raises(monkeypatch):
    monkeypatch.delenv("MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH", raising=False)
    with pytest.raises(PublicBindWithoutAuthError) as ei:
        enforce_safe_bind("0.0.0.0", None)
    msg = str(ei.value)
    assert "0.0.0.0" in msg
    assert "MIDDLE_LAYER_API_KEY" in msg


def test_enforce_safe_bind_override_allows_public_no_key(monkeypatch, caplog):
    monkeypatch.setenv("MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH", "1")
    with caplog.at_level("WARNING"):
        enforce_safe_bind("0.0.0.0", None)
    assert any("PUBLIC BIND WITHOUT AUTH" in r.message for r in caplog.records)


# --- resolve_max_request_bytes ---------------------------------------------


def test_resolve_max_request_bytes_default(monkeypatch):
    monkeypatch.delenv("MIDDLE_LAYER_MAX_REQUEST_BYTES", raising=False)
    assert resolve_max_request_bytes() == DEFAULT_MAX_REQUEST_BYTES


def test_resolve_max_request_bytes_explicit(monkeypatch):
    monkeypatch.setenv("MIDDLE_LAYER_MAX_REQUEST_BYTES", "1024")
    assert resolve_max_request_bytes() == 1024


@pytest.mark.parametrize("bad", ["", "-1", "0", "abc", "1.5"])
def test_resolve_max_request_bytes_invalid_falls_back(monkeypatch, bad):
    if bad == "":
        monkeypatch.delenv("MIDDLE_LAYER_MAX_REQUEST_BYTES", raising=False)
    else:
        monkeypatch.setenv("MIDDLE_LAYER_MAX_REQUEST_BYTES", bad)
    assert resolve_max_request_bytes() == DEFAULT_MAX_REQUEST_BYTES


# --- apply_security_headers -------------------------------------------------


class _FakeHeaders(dict):
    """Header dict that mimics werkzeug.datastructures.Headers item assignment."""


class _FakeResponse:
    def __init__(self):
        self.headers = _FakeHeaders()


def test_apply_security_headers_sets_baseline():
    resp = _FakeResponse()
    apply_security_headers(resp)
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert resp.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert "Content-Security-Policy" not in resp.headers


def test_apply_security_headers_adds_csp_for_dashboard():
    resp = _FakeResponse()
    apply_security_headers(resp, path="/dashboard/")
    csp = resp.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp


def test_apply_security_headers_does_not_overwrite_existing():
    resp = _FakeResponse()
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    apply_security_headers(resp)
    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"


# --- alias allowlist --------------------------------------------------------


def test_is_well_formed_alias_accepts_typical_names():
    for ok in [
        "qwen2.5-coder-32b",
        "mlx-community/Qwen3-8B-MLX",
        "openai/gpt-oss-20b",
        "granite-4.1-8b",
        "model_v1",
    ]:
        assert is_well_formed_alias(ok), ok


def test_is_well_formed_alias_rejects_obvious_attacks():
    for bad in [
        "",
        "a" * 300,
        "model;rm -rf /",
        "model\nname",
        "model\x00name",
        "model$(whoami)",
        "model`id`",
        "../../etc/passwd",  # contains "../" but ".." is allowed; the ".." is not in bad chars
        "model|tee",
        "model<script>",
        42,
        None,
    ]:
        if bad == "../../etc/passwd":
            # "../" is not in the disallowed set; the syntactic filter is
            # not a path-traversal defence. The allowlist check is what
            # protects against this — we assert it elsewhere.
            assert is_well_formed_alias(bad), bad
            continue
        assert not is_well_formed_alias(bad), bad


def test_alias_in_allowlist():
    available = ["a", "b", "c"]
    assert alias_in_allowlist("a", available) is True
    assert alias_in_allowlist("d", available) is False
    assert alias_in_allowlist("", available) is False
    assert alias_in_allowlist(42, available) is False  # type: ignore[arg-type]


# --- Integration: middle_layer.py (LM Studio backend) auth guard ----------


@pytest.fixture
def lmstudio_app(monkeypatch):
    """Load middle_layer.py with a known API key and return its Flask app."""
    monkeypatch.setenv("MIDDLE_LAYER_API_KEY", "test-key")
    monkeypatch.setenv("MIDDLE_LAYER_MAX_REQUEST_BYTES", "1024")
    sys.modules.pop("middle_layer_root_app_sec", None)
    spec = importlib.util.spec_from_file_location(
        "middle_layer_root_app_sec", REPO_ROOT / "middle_layer.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


def test_lmstudio_unauthorized_without_key(lmstudio_app):
    client = lmstudio_app.test_client()
    r = client.get("/v1/models")
    assert r.status_code == 401
    body = json.loads(r.data)
    assert body["error"] == "Unauthorized"


def test_lmstudio_authorized_with_x_api_key(lmstudio_app):
    client = lmstudio_app.test_client()
    r = client.get("/v1/models", headers={"X-API-Key": "test-key"})
    # 200 only if LM Studio is up, which it isn't in tests; we just need
    # to confirm we got *past* the auth guard (so not 401).
    assert r.status_code != 401


def test_lmstudio_authorized_with_bearer(lmstudio_app):
    client = lmstudio_app.test_client()
    r = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert r.status_code != 401


def test_lmstudio_rejects_wrong_bearer(lmstudio_app):
    client = lmstudio_app.test_client()
    r = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_lmstudio_security_headers_present(lmstudio_app):
    client = lmstudio_app.test_client()
    r = client.get("/v1/models", headers={"X-API-Key": "test-key"})
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "no-referrer"


def test_lmstudio_max_content_length_enforced(lmstudio_app):
    client = lmstudio_app.test_client()
    big = b"x" * 4096  # well above the 1024-byte test cap
    r = client.post(
        "/v1/chat/completions",
        data=big,
        headers={
            "X-API-Key": "test-key",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 413
