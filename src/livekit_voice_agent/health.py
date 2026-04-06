"""
Health monitoring for the voice agent.
Tracks: uptime, total calls, active calls, errors, and voice-to-voice latency.
Attach to an AgentSession to start collecting metrics automatically.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("livekit_voice_agent.health")

# Keep last N latency samples for rolling average
_MAX_LATENCY_SAMPLES = 50


@dataclass
class HealthStatus:
    """Snapshot of current agent health."""

    uptime_seconds: float
    total_calls: int
    active_calls: int
    error_count: int
    avg_latency_ms: float | None   # None until at least one call completes
    last_error: str | None

    def is_healthy(self) -> bool:
        """Simple health check: no runaway errors."""
        return self.error_count < 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.is_healthy() else "degraded",
            "uptime_seconds": round(self.uptime_seconds, 1),
            "total_calls": self.total_calls,
            "active_calls": self.active_calls,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1) if self.avg_latency_ms is not None else None,
            "last_error": self.last_error,
        }


class HealthMonitor:
    """
    Attaches to an AgentSession and collects health metrics.

    Usage:
        monitor = HealthMonitor()
        monitor.attach(session)
        ...
        status = monitor.get_status()
        monitor.log_status()
    """

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._total_calls = 0
        self._active_calls = 0
        self._error_count = 0
        self._last_error: str | None = None
        # Rolling window of voice-to-voice latencies (ms)
        self._latencies: deque[float] = deque(maxlen=_MAX_LATENCY_SAMPLES)

    # ── Public API ────────────────────────────────────────────────────

    def attach(self, session: Any) -> None:
        """Hook into AgentSession events to collect metrics automatically."""

        @session.on("agent_started_speaking")
        def _on_agent_start(_ev: Any) -> None:  # type: ignore[no-untyped-def]
            self._total_calls += 1
            self._active_calls += 1

        @session.on("agent_stopped_speaking")
        def _on_agent_stop(_ev: Any) -> None:  # type: ignore[no-untyped-def]
            self._active_calls = max(0, self._active_calls - 1)

        @session.on("metrics_collected")
        def _on_metrics(ev: Any) -> None:  # type: ignore[no-untyped-def]
            # Capture voice-to-voice latency from EOU metrics
            from livekit.agents.metrics import EOUMetrics

            m = ev.metrics
            if isinstance(m, EOUMetrics):
                v2v_ms = (m.end_of_utterance_delay + m.transcription_delay) * 1000
                self._latencies.append(v2v_ms)

        @session.on("error")
        def _on_error(ev: Any) -> None:  # type: ignore[no-untyped-def]
            self._error_count += 1
            self._last_error = str(ev.error)
            logger.error("[HEALTH] Error #%d: %s", self._error_count, self._last_error)

        logger.info("[HEALTH] Monitor attached to session")

    def record_latency(self, v2v_ms: float) -> None:
        """Manually record a voice-to-voice latency sample (ms)."""
        self._latencies.append(v2v_ms)

    def get_status(self) -> HealthStatus:
        """Return current health snapshot."""
        avg = (sum(self._latencies) / len(self._latencies)) if self._latencies else None
        return HealthStatus(
            uptime_seconds=time.monotonic() - self._start_time,
            total_calls=self._total_calls,
            active_calls=self._active_calls,
            error_count=self._error_count,
            avg_latency_ms=avg,
            last_error=self._last_error,
        )

    def log_status(self) -> None:
        """Log a one-line health summary."""
        s = self.get_status()
        avg = f"{s.avg_latency_ms:.0f}ms" if s.avg_latency_ms is not None else "n/a"
        logger.info(
            "[HEALTH] uptime=%.0fs  calls=%d  active=%d  errors=%d  avg_v2v=%s  status=%s",
            s.uptime_seconds,
            s.total_calls,
            s.active_calls,
            s.error_count,
            avg,
            "OK" if s.is_healthy() else "DEGRADED",
        )
