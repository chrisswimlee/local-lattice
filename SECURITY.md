# Security policy

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | Yes                |
| 0.1.x   | Yes (best-effort; upgrade to 0.2.x) |
| Pre-0.1 | No (pre-OSS tags)  |

Security fixes are backported to the latest minor on the current major line when
practical. After 1.0.0, the previous minor receives critical fixes for 90 days.

## Reporting a vulnerability

**Please do not file security-sensitive issues as public GitHub issues.**

**Preferred:** use [GitHub private vulnerability reporting](https://github.com/chrisswimlee/local-lattice/security/advisories/new) for this repository (maintainers only).

If that workflow is unavailable to you, contact the repository owner via their
[GitHub profile](https://github.com/chrisswimlee) using any contact method they
publish there.

Include:

- A short description of the issue and its impact
- Steps to reproduce (or a proof-of-concept)
- Affected version / commit
- Whether you believe it is already exploitable in the wild

We aim to acknowledge within **72 hours** and to ship a fix or mitigation within
**14 days** for confirmed critical issues, subject to responsible disclosure
coordination.

## Scope

In scope: the Local Lattice / MiddleLayer gateway process, its default configuration, bundled
dashboard static assets, and documented deployment patterns (Flask dev server,
future Docker image).

Out of scope: vulnerabilities in upstream runtimes (`mlx-lm`, LM Studio,
Anthropic API, Hugging Face Hub) unless the gateway passes untrusted client
input to them in an unsafe way.

## Hardening roadmap

Known gaps and their remediation passes are tracked in the internal risk
register (`docs/_internal/RISK_REGISTER.md` on the discovery branch).

### Landed in Pass 4 (current `[Unreleased]`)

- ✅ **Constant-time API key comparison** (`hmac.compare_digest`) across
  both backends and the dashboard. Closes **P4-02**.
- ✅ **Refuse public bind without API key** unless
  `MIDDLE_LAYER_ALLOW_PUBLIC_NO_AUTH=1` is set. Closes **P4-01**.
- ✅ **Model-load allowlist** on `/dashboard/api/models/load`:
  syntactic filter on the alias string plus an exact-match allowlist
  against the discovered on-disk model set. Closes **P4-03**.
- ✅ **Request body size cap** via `MIDDLE_LAYER_MAX_REQUEST_BYTES`
  (default 10 MiB) → Flask-native 413 on oversize bodies.
- ✅ **Standard hardening headers** on every response
  (`X-Content-Type-Options`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, `Cross-Origin-Resource-Policy: same-origin`)
  plus a strict **Content-Security-Policy** on the dashboard.

### Still scheduled (Pass 5+)

- Per-IP / per-API-key rate limiting.
- HSTS guidance for deployments behind TLS.
- CSP nonce-mode for the dashboard (currently allows `style-src 'unsafe-inline'`).
- Prompt redaction (regex) when `MLX_DASHBOARD_CAPTURE_PROMPTS=1`.

## `ANTHROPIC_BASE_URL` and similar

Operator-controlled base URLs (`ANTHROPIC_BASE_URL`, `LM_STUDIO_URL`) must **never**
be taken from untrusted client requests. They are server configuration only.
Exposing them through a multi-tenant control plane would be a security defect.
