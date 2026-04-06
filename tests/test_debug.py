"""Tests for debug.py — memory leak fix and latency tracking."""
from __future__ import annotations

import time

from livekit_voice_agent.debug import (
    TURN_TTL_SECONDS,
    _TurnData,
    _cleanup_stale_turns,
)


# ── _TurnData ─────────────────────────────────────────────────────────


def test_turn_data_defaults() -> None:
    t = _TurnData()
    assert t.user_audio_sec == 0.0
    assert t.has_eou is False
    assert t.has_llm is False
    assert t.has_tts is False


def test_turn_data_voice_to_voice_ms() -> None:
    t = _TurnData(eou_ms=80, transcript_ms=60, llm_ttft_ms=200, tts_ttfb_ms=150)
    assert t.voice_to_voice_ms() == 490.0


def test_turn_data_is_complete_all_true() -> None:
    t = _TurnData(has_eou=True, has_llm=True, has_tts=True)
    assert t.is_complete() is True


def test_turn_data_is_complete_partial() -> None:
    assert _TurnData(has_eou=True, has_llm=True, has_tts=False).is_complete() is False
    assert _TurnData(has_eou=True, has_llm=False, has_tts=True).is_complete() is False
    assert _TurnData(has_eou=False, has_llm=True, has_tts=True).is_complete() is False


def test_turn_data_created_at_is_recent() -> None:
    before = time.monotonic()
    t = _TurnData()
    after = time.monotonic()
    assert before <= t.created_at <= after


# ── is_stale ─────────────────────────────────────────────────────────


def test_turn_not_stale_when_fresh() -> None:
    t = _TurnData()
    assert t.is_stale(ttl=TURN_TTL_SECONDS) is False


def test_turn_is_stale_with_past_timestamp() -> None:
    t = _TurnData()
    # Fake an old turn by backdating created_at
    t.created_at = time.monotonic() - (TURN_TTL_SECONDS + 1)
    assert t.is_stale() is True


def test_turn_not_stale_just_before_ttl() -> None:
    t = _TurnData()
    t.created_at = time.monotonic() - (TURN_TTL_SECONDS - 1)
    assert t.is_stale() is False


# ── _cleanup_stale_turns ──────────────────────────────────────────────


def test_cleanup_removes_stale_turns() -> None:
    turns: dict[str, _TurnData] = {}

    fresh = _TurnData()
    stale = _TurnData()
    stale.created_at = time.monotonic() - (TURN_TTL_SECONDS + 5)

    turns["fresh-1"] = fresh
    turns["stale-1"] = stale

    removed = _cleanup_stale_turns(turns)

    assert removed == 1
    assert "fresh-1" in turns
    assert "stale-1" not in turns


def test_cleanup_empty_dict() -> None:
    turns: dict[str, _TurnData] = {}
    removed = _cleanup_stale_turns(turns)
    assert removed == 0


def test_cleanup_all_fresh_no_removal() -> None:
    turns = {f"id-{i}": _TurnData() for i in range(5)}
    removed = _cleanup_stale_turns(turns)
    assert removed == 0
    assert len(turns) == 5


def test_cleanup_all_stale() -> None:
    turns: dict[str, _TurnData] = {}
    for i in range(3):
        t = _TurnData()
        t.created_at = time.monotonic() - (TURN_TTL_SECONDS + 10)
        turns[f"id-{i}"] = t

    removed = _cleanup_stale_turns(turns)
    assert removed == 3
    assert len(turns) == 0


def test_cleanup_custom_ttl() -> None:
    turns: dict[str, _TurnData] = {}
    t = _TurnData()
    t.created_at = time.monotonic() - 5  # 5 seconds old
    turns["id-1"] = t

    # With TTL=10s → not stale
    assert _cleanup_stale_turns(turns, ttl=10.0) == 0

    # With TTL=3s → stale
    assert _cleanup_stale_turns(turns, ttl=3.0) == 1
    assert len(turns) == 0


def test_cleanup_preserves_incomplete_fresh_turns() -> None:
    """Incomplete turns that are still fresh must NOT be removed."""
    turns: dict[str, _TurnData] = {}
    t = _TurnData(has_eou=True, has_llm=False, has_tts=False)  # incomplete but fresh
    turns["incomplete-fresh"] = t

    removed = _cleanup_stale_turns(turns)
    assert removed == 0
    assert "incomplete-fresh" in turns
