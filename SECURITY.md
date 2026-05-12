# Security policy

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |
| < 0.1   | No (pre-OSS tags)  |

Security fixes are backported to the latest minor on the current major line when
practical. After 1.0.0, the previous minor receives critical fixes for 90 days.

## Reporting a vulnerability

**Please do not file security-sensitive issues as public GitHub issues.**

Send reports to: **security@middle-layer.invalid**

Replace that address with a working inbox before the repository is made public
or advertised widely. Until then, use a private channel you already trust with
the maintainers (for example a direct message to the repo owner).

Include:

- A short description of the issue and its impact
- Steps to reproduce (or a proof-of-concept)
- Affected version / commit
- Whether you believe it is already exploitable in the wild

We aim to acknowledge within **72 hours** and to ship a fix or mitigation within
**14 days** for confirmed critical issues, subject to responsible disclosure
coordination.

## Scope

In scope: the MiddleLayer gateway process, its default configuration, bundled
dashboard static assets, and documented deployment patterns (Flask dev server,
future Docker image).

Out of scope: vulnerabilities in upstream runtimes (`mlx-lm`, LM Studio,
Anthropic API, Hugging Face Hub) unless MiddleLayer passes untrusted client
input to them in an unsafe way.

## Hardening roadmap

Known gaps and their remediation passes are tracked in the internal risk
register (`docs/_internal/RISK_REGISTER.md` on the discovery branch). Highlights
scheduled for **Pass 4**:

- Constant-time API key comparison
- Refuse public bind without API key (with explicit override)
- Model-load allowlists and path containment for dashboard-driven loads
- Request size limits, rate limits, security headers, CSP on the dashboard

## `ANTHROPIC_BASE_URL` and similar

Operator-controlled base URLs (`ANTHROPIC_BASE_URL`, `LM_STUDIO_URL`) must **never**
be taken from untrusted client requests. They are server configuration only.
Exposing them through a multi-tenant control plane would be a security defect.
