"""Unit tests for middle_layer.timing."""

from __future__ import annotations

import logging
import time

import middle_layer.timing as timing


def test_request_timer_header_dict() -> None:
    timer = timing.RequestTimer.start()
    with timer.measure("resolve_ms"):
        time.sleep(0.001)
    with timer.measure("upstream_ms"):
        time.sleep(0.001)
    timer.queue_ms = 5
    headers = timer.header_dict()
    assert int(headers[timing.HEADER_RESOLVE_MS]) >= 0
    assert headers[timing.HEADER_QUEUE_MS] == "5"
    assert int(headers[timing.HEADER_UPSTREAM_MS]) >= 0
    assert int(headers[timing.HEADER_TOTAL_MS]) >= 2


def test_request_timer_measure_context() -> None:
    timer = timing.RequestTimer.start()
    with timer.measure("resolve_ms"):
        pass
    assert timer.resolve_ms >= 0


def test_lattice_timing_headers_defaults_total() -> None:
    headers = timing.lattice_timing_headers(resolve_ms=1, queue_ms=2, upstream_ms=3)
    assert headers[timing.HEADER_TOTAL_MS] == "6"


def test_merge_mlx_legacy_timing() -> None:
    headers: dict[str, str] = {"X-Model-Routed-To": "mlx/foo"}
    timing.merge_mlx_legacy_timing(headers, queue_ms=4, upstream_ms=90)
    assert headers[timing.HEADER_QUEUE_MS] == "4"
    assert headers[timing.HEADER_UPSTREAM_MS] == "90"
    assert headers[timing.LEGACY_MLX_LATENCY_MS] == "90"
    assert headers[timing.LEGACY_MLX_QUEUE_MS] == "4"


def test_maybe_log_respects_env(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LATTICE_LOG_TIMING", "0")
    timer = timing.RequestTimer.start()
    with caplog.at_level(logging.INFO):
        timer.maybe_log(path="/v1/chat/completions", status=200)
    assert not caplog.records

    monkeypatch.setenv("LATTICE_LOG_TIMING", "1")
    with caplog.at_level(logging.INFO):
        timer.maybe_log(path="/v1/chat/completions", status=200)
    assert len(caplog.records) == 1
    assert "lattice.request" in caplog.records[0].message
    assert "path=/v1/chat/completions" in caplog.records[0].message
