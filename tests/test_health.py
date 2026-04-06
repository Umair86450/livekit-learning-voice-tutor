"""Tests for health.py — HealthMonitor and HealthStatus."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from livekit_voice_agent.health import HealthMonitor, HealthStatus, _MAX_LATENCY_SAMPLES


# ── HealthStatus ──────────────────────────────────────────────────────


def test_health_status_is_healthy_below_threshold() -> None:
    s = HealthStatus(
        uptime_seconds=60,
        total_calls=5,
        active_calls=1,
        error_count=9,
        avg_latency_ms=300.0,
        last_error=None,
    )
    assert s.is_healthy() is True


def test_health_status_is_degraded_at_threshold() -> None:
    s = HealthStatus(
        uptime_seconds=60,
        total_calls=5,
        active_calls=0,
        error_count=10,
        avg_latency_ms=None,
        last_error="timeout",
    )
    assert s.is_healthy() is False


def test_health_status_to_dict_ok() -> None:
    s = HealthStatus(
        uptime_seconds=120.0,
        total_calls=3,
        active_calls=1,
        error_count=0,
        avg_latency_ms=450.0,
        last_error=None,
    )
    d = s.to_dict()
    assert d["status"] == "ok"
    assert d["total_calls"] == 3
    assert d["avg_latency_ms"] == 450.0
    assert d["last_error"] is None


def test_health_status_to_dict_degraded() -> None:
    s = HealthStatus(
        uptime_seconds=30.0,
        total_calls=1,
        active_calls=0,
        error_count=15,
        avg_latency_ms=None,
        last_error="APIError",
    )
    d = s.to_dict()
    assert d["status"] == "degraded"
    assert d["avg_latency_ms"] is None
    assert d["last_error"] == "APIError"


# ── HealthMonitor initial state ───────────────────────────────────────


def test_health_monitor_initial_state() -> None:
    m = HealthMonitor()
    s = m.get_status()
    assert s.total_calls == 0
    assert s.active_calls == 0
    assert s.error_count == 0
    assert s.avg_latency_ms is None
    assert s.last_error is None
    assert s.is_healthy() is True


def test_health_monitor_uptime_increases() -> None:
    m = HealthMonitor()
    s1 = m.get_status()
    time.sleep(0.05)
    s2 = m.get_status()
    assert s2.uptime_seconds > s1.uptime_seconds


# ── record_latency ────────────────────────────────────────────────────


def test_record_latency_single_sample() -> None:
    m = HealthMonitor()
    m.record_latency(400.0)
    s = m.get_status()
    assert s.avg_latency_ms == pytest.approx(400.0)


def test_record_latency_average() -> None:
    m = HealthMonitor()
    m.record_latency(200.0)
    m.record_latency(400.0)
    s = m.get_status()
    assert s.avg_latency_ms == pytest.approx(300.0)


def test_record_latency_rolling_window_max() -> None:
    """Deque must not grow beyond _MAX_LATENCY_SAMPLES."""
    m = HealthMonitor()
    for i in range(_MAX_LATENCY_SAMPLES + 20):
        m.record_latency(float(i))
    assert len(m._latencies) == _MAX_LATENCY_SAMPLES


# ── attach() — session event hooks ───────────────────────────────────


def _make_mock_session() -> MagicMock:
    """Return a mock AgentSession that stores .on() callbacks by event name."""
    session = MagicMock()
    _handlers: dict[str, object] = {}

    def on_decorator(event_name: str):  # type: ignore[no-untyped-def]
        def register(fn):  # type: ignore[no-untyped-def]
            _handlers[event_name] = fn
            return fn
        return register

    session.on.side_effect = on_decorator
    session._handlers = _handlers
    return session


def test_attach_increments_total_calls() -> None:
    m = HealthMonitor()
    session = _make_mock_session()
    m.attach(session)

    session._handlers["agent_started_speaking"](MagicMock())
    session._handlers["agent_started_speaking"](MagicMock())

    assert m.get_status().total_calls == 2


def test_attach_active_calls_increase_decrease() -> None:
    m = HealthMonitor()
    session = _make_mock_session()
    m.attach(session)

    session._handlers["agent_started_speaking"](MagicMock())
    session._handlers["agent_started_speaking"](MagicMock())
    assert m.get_status().active_calls == 2

    session._handlers["agent_stopped_speaking"](MagicMock())
    assert m.get_status().active_calls == 1


def test_attach_active_calls_never_negative() -> None:
    m = HealthMonitor()
    session = _make_mock_session()
    m.attach(session)

    # Stop without start — should not go negative
    session._handlers["agent_stopped_speaking"](MagicMock())
    assert m.get_status().active_calls == 0


def test_attach_error_increments_count() -> None:
    m = HealthMonitor()
    session = _make_mock_session()
    m.attach(session)

    err_ev = MagicMock()
    err_ev.error = ValueError("LLM timeout")
    session._handlers["error"](err_ev)

    s = m.get_status()
    assert s.error_count == 1
    assert "LLM timeout" in (s.last_error or "")


def test_attach_error_stores_last_error() -> None:
    m = HealthMonitor()
    session = _make_mock_session()
    m.attach(session)

    for msg in ["first error", "second error", "third error"]:
        ev = MagicMock()
        ev.error = RuntimeError(msg)
        session._handlers["error"](ev)

    assert m.get_status().error_count == 3
    assert "third error" in (m.get_status().last_error or "")


def test_attach_metrics_records_eou_latency() -> None:
    """EOU metrics should feed into the rolling latency average."""

    m = HealthMonitor()
    session = _make_mock_session()
    m.attach(session)

    # Simulate an EOUMetrics event
    from livekit.agents.metrics import EOUMetrics

    eou = MagicMock(spec=EOUMetrics)
    eou.end_of_utterance_delay = 0.08   # 80ms
    eou.transcription_delay = 0.07      # 70ms → v2v = 150ms

    metrics_ev = MagicMock()
    metrics_ev.metrics = eou
    session._handlers["metrics_collected"](metrics_ev)

    assert m.get_status().avg_latency_ms == pytest.approx(150.0)


# ── log_status ────────────────────────────────────────────────────────


def test_log_status_does_not_raise() -> None:
    m = HealthMonitor()
    m.record_latency(300.0)
    m.log_status()  # should not raise


def test_log_status_no_latency_does_not_raise() -> None:
    m = HealthMonitor()
    m.log_status()  # avg is None, should still work
