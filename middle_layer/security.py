"""Shared security primitives used by both gateway backends and the dashboard.

This module is deliberately small, dependency-free (stdlib + Flask types only),
and side-effect free at import time. It centralises the operations that were
previously open-coded in three different files:

- constant-time API key comparison (was: ``==`` / ``!=`` — timing oracle);
- public-bind safety check at startup (refuse to listen on a public
  interface without an API key, unless explicitly overridden);
- a request body size cap (Flask's ``MAX_CONTENT_LENGTH``);
- a standard set of security response headers, plus a strict CSP for the
  dashboard.

All knobs honour environment variables but defaults are safe: deny-by-default
where it matters, opt-in for compatibility shims.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from collections.abc import Iterable, Mapping

log = logging.getLogger("middle_layer.security")


# --- API key comparison -----------------------------------------------------

def constant_time_eq(provided: str | None, expected: str | None) -> bool:
    """Constant-time string comparison that tolerates ``None`` / empty inputs.

    Returns ``False`` when either side is empty so an unset key never matches.
    The actual byte-level comparison goes through :func:`hmac.compare_digest`
    on the UTF-8 encoded forms, which is the stdlib recommendation for
    authentication tokens.
    """
    if not provided or not expected:
        return False
    try:
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
    except (AttributeError, TypeError):
        return False


def extract_bearer(authorization_header: str | None) -> str | None:
    """Return the bearer token from an ``Authorization`` header, or ``None``."""
    if not authorization_header:
        return None
    prefix = "Bearer "
    if authorization_header.startswith(prefix):
        return authorization_header[len(prefix):].strip() or None
    return None


def check_api_key(headers: Mapping[str, str], expected_key: str | None) -> bool:
    """True if the request carries a valid API key, or if no key is configured.

    Accepts either ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
    Comparison is constant-time. When ``expected_key`` is falsy, this returns
    ``True`` (auth disabled) — callers that want to *require* auth should
    enforce that at startup via :func:`enforce_safe_bind`.
    """
    if not expected_key:
        return True
    x_api_key = headers.get("X-API-Key") or headers.get("x-api-key")
    bearer = extract_bearer(headers.get("Authorization") or headers.get("authorization"))
    if constant_time_eq(x_api_key, expected_key):
        return True
    if constant_time_eq(bearer, expected_key):
        return True
    return False


# --- Public bind safety -----------------------------------------------------

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "ip6-localhost"})


def is_public_bind(host: str | None) -> bool:
    """Return True if ``host`` is anything other than a loopback interface.

    ``0.0.0.0`` and ``::`` count as public for this purpose, since they bind
    to *all* interfaces. Unrecognised strings are treated as public (fail
    safe) so an unexpected DNS name does not silently weaken the guard.
    """
    if not host:
        return False
    h = host.strip().lower()
    if h in _LOOPBACK_HOSTS:
        return False
    try:
        addr = ipaddress.ip_address(h)
    except ValueError:
        return True
    if addr.is_loopback:
        return False
    return True


class PublicBindWithoutAuthError(RuntimeError):
    """Raised when the gateway would listen on a public interface without auth."""


def enforce_safe_bind(host: str | None, api_key: str | None) -> None:
    """Refuse to bind to a public interface without an API key.

    Set ``MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH=1`` to override (logs a loud
    warning instead of raising). This escape hatch exists for operators who
    run the gateway behind a trusted reverse proxy that does its own auth.

    Raises :class:`PublicBindWithoutAuthError` when the bind is public, the
    key is missing, and the override is not set.
    """
    if not is_public_bind(host):
        return
    if api_key:
        return
    override = os.environ.get("MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        log.warning(
            "PUBLIC BIND WITHOUT AUTH (host=%s). MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH "
            "is set — assuming an upstream proxy is enforcing authentication. "
            "Anyone reaching this port directly can call any endpoint.",
            host,
        )
        return
    raise PublicBindWithoutAuthError(
        f"Refusing to bind on public interface {host!r} without MIDDLE_LAYER_API_KEY set. "
        f"Either set MIDDLE_LAYER_API_KEY=<key> to enable auth, set HOST=127.0.0.1 to "
        f"stay local-only, or set MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH=1 to override "
        f"(use only behind a trusted auth-enforcing reverse proxy)."
    )


# --- Request size cap -------------------------------------------------------

DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MiB


def resolve_max_request_bytes() -> int:
    """Resolve ``MIDDLE_LAYER_MAX_REQUEST_BYTES`` (default 10 MiB)."""
    raw = os.environ.get("MIDDLE_LAYER_MAX_REQUEST_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BYTES
    try:
        v = int(raw)
    except ValueError:
        log.warning("Invalid MIDDLE_LAYER_MAX_REQUEST_BYTES=%r; using default.", raw)
        return DEFAULT_MAX_REQUEST_BYTES
    if v <= 0:
        log.warning("MIDDLE_LAYER_MAX_REQUEST_BYTES=%d is not positive; using default.", v)
        return DEFAULT_MAX_REQUEST_BYTES
    return v


# --- Response headers -------------------------------------------------------

_DEFAULT_SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)

# CSP for the dashboard HTML / static assets. The dashboard is self-hosted,
# so we disallow inline scripts and remote sources outright. ``connect-src
# 'self'`` lets the dashboard's JS poll the same-origin API.
_DASHBOARD_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "font-src 'self' data:; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)


def apply_security_headers(response, *, path: str | None = None) -> None:
    """Stamp standard hardening headers onto a Flask ``Response`` in place.

    Headers added unconditionally:

    - ``X-Content-Type-Options: nosniff``
    - ``X-Frame-Options: DENY``
    - ``Referrer-Policy: no-referrer``
    - ``Cross-Origin-Resource-Policy: same-origin``

    Additionally, when ``path`` looks like a dashboard route, attach a strict
    Content-Security-Policy that disallows inline scripts and remote sources.

    Existing values are not overwritten (so CORS-configured headers are
    respected, and tests can override via direct assignment).
    """
    headers = response.headers
    for name, value in _DEFAULT_SECURITY_HEADERS:
        if name not in headers:
            headers[name] = value
    if path and _is_dashboard_path(path):
        if "Content-Security-Policy" not in headers:
            headers["Content-Security-Policy"] = _DASHBOARD_CSP


def _is_dashboard_path(path: str) -> bool:
    if not path:
        return False
    return path == "/dashboard" or path.startswith("/dashboard/")


# --- Alias allowlist for dashboard model loads ------------------------------

# Aliases (Hugging Face repo ids, MLX local dir names) should be conservative
# in punctuation — letters, digits, dots, slashes, underscores, hyphens, plus.
# This filter exists to reject obviously hostile inputs (newlines, NULs,
# control chars, shell metacharacters) *before* they reach the loader.
_ALIAS_MAX_LEN = 256


def is_well_formed_alias(value: object) -> bool:
    """True if ``value`` looks like a plausible model alias.

    This is a *syntactic* filter — it does not check whether the alias is
    actually loadable. The dashboard endpoint should additionally verify that
    the alias is in ``mlx_manager.get_available_aliases()`` (i.e. an exact
    match against the discovered set on disk) before calling ``load_model``.
    """
    if not isinstance(value, str):
        return False
    if not value or len(value) > _ALIAS_MAX_LEN:
        return False
    for ch in value:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            return False
    bad = ('"', "'", "`", "$", "\\", "<", ">", "|", ";", "*", "?")
    if any(b in value for b in bad):
        return False
    return True


def alias_in_allowlist(value: object, allowlist: Iterable[str]) -> bool:
    """True if ``value`` exactly matches an entry in ``allowlist``."""
    if not isinstance(value, str):
        return False
    allowset = {a for a in allowlist if isinstance(a, str)}
    return value in allowset


__all__ = [
    "constant_time_eq",
    "extract_bearer",
    "check_api_key",
    "is_public_bind",
    "enforce_safe_bind",
    "PublicBindWithoutAuthError",
    "DEFAULT_MAX_REQUEST_BYTES",
    "resolve_max_request_bytes",
    "apply_security_headers",
    "is_well_formed_alias",
    "alias_in_allowlist",
]
