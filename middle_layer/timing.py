"""Request timing helpers for Local Lattice gateways.

Exposes routing overhead in standard ``X-Lattice-*-Ms`` response headers and
optional structured log lines so operators can separate resolve/queue time
from upstream inference latency.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

HEADER_RESOLVE_MS = "X-Lattice-Resolve-Ms"
HEADER_QUEUE_MS = "X-Lattice-Queue-Ms"
HEADER_UPSTREAM_MS = "X-Lattice-Upstream-Ms"
HEADER_TOTAL_MS = "X-Lattice-Total-Ms"

# Legacy MLX headers — kept for one minor; mirror upstream/queue into Lattice names.
LEGACY_MLX_LATENCY_MS = "X-MLX-Latency-Ms"
LEGACY_MLX_QUEUE_MS = "X-MLX-Queue-Wait-Ms"


def log_timing_enabled() -> bool:
    raw = os.environ.get("LATTICE_LOG_TIMING", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


@dataclass
class RequestTimer:
    """Wall-clock timer for a single HTTP handler."""

    _start: float = field(default_factory=time.perf_counter)
    resolve_ms: int = 0
    queue_ms: int = 0
    upstream_ms: int = 0

    @classmethod
    def start(cls) -> RequestTimer:
        return cls()

    def total_ms(self) -> int:
        return max(0, int((time.perf_counter() - self._start) * 1000))

    @contextmanager
    def measure(self, field_name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = max(0, int((time.perf_counter() - t0) * 1000))
            setattr(self, field_name, ms)

    def header_dict(self) -> dict[str, str]:
        total = self.total_ms()
        return {
            HEADER_RESOLVE_MS: str(self.resolve_ms),
            HEADER_QUEUE_MS: str(self.queue_ms),
            HEADER_UPSTREAM_MS: str(self.upstream_ms),
            HEADER_TOTAL_MS: str(total),
        }

    def apply_to_headers(self, headers: dict[str, str]) -> dict[str, str]:
        headers.update(self.header_dict())
        return headers

    def log_line(self, **fields: Any) -> str:
        parts = [
            "lattice.request",
            f"resolve_ms={self.resolve_ms}",
            f"queue_ms={self.queue_ms}",
            f"upstream_ms={self.upstream_ms}",
            f"total_ms={self.total_ms()}",
        ]
        for key, value in sorted(fields.items()):
            if value is None:
                continue
            text = str(value).replace("\n", " ").replace('"', "'")
            if " " in text:
                text = f'"{text}"'
            parts.append(f"{key}={text}")
        return " ".join(parts)

    def maybe_log(self, **fields: Any) -> None:
        if log_timing_enabled():
            logger.info(self.log_line(**fields))


def lattice_timing_headers(
    *,
    resolve_ms: int = 0,
    queue_ms: int = 0,
    upstream_ms: int = 0,
    total_ms: int | None = None,
) -> dict[str, str]:
    """Build ``X-Lattice-*-Ms`` headers from explicit millisecond buckets."""
    if total_ms is None:
        total_ms = max(resolve_ms + queue_ms + upstream_ms, 0)
    return {
        HEADER_RESOLVE_MS: str(max(0, resolve_ms)),
        HEADER_QUEUE_MS: str(max(0, queue_ms)),
        HEADER_UPSTREAM_MS: str(max(0, upstream_ms)),
        HEADER_TOTAL_MS: str(max(0, total_ms)),
    }


def merge_mlx_legacy_timing(
    headers: dict[str, str],
    *,
    resolve_ms: int = 0,
    queue_ms: int = 0,
    upstream_ms: int = 0,
    total_ms: int | None = None,
) -> dict[str, str]:
    """Add Lattice timing headers alongside legacy MLX latency headers."""
    headers.update(
        lattice_timing_headers(
            resolve_ms=resolve_ms,
            queue_ms=queue_ms,
            upstream_ms=upstream_ms,
            total_ms=total_ms,
        )
    )
    if upstream_ms:
        headers.setdefault(LEGACY_MLX_LATENCY_MS, str(upstream_ms))
    if queue_ms:
        headers.setdefault(LEGACY_MLX_QUEUE_MS, str(queue_ms))
    return headers
