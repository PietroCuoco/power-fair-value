"""Offline tests for the LLM component (no API key or network needed)."""

from __future__ import annotations

import json

from power_fv import llm


def _fake_caller_json(*_args) -> str:
    # A canned response a model might return for the sample text.
    return json.dumps(
        {
            "events": [
                {
                    "asset": "Grohnde-Nord Unit B",
                    "fuel_type": "nuclear",
                    "capacity_mw": 1400,
                    "start": "2026-03-12",
                    "end": "2026-03-18",
                    "direction": "bullish",
                    "confidence": 0.9,
                },
                {
                    "asset": "Weisweiler-3",
                    "fuel_type": "lignite",
                    "capacity_mw": 600,
                    "start": "2026-04-05",
                    "end": None,
                    "direction": "bearish",
                    "confidence": 0.8,
                },
            ]
        }
    )


def test_build_prompt_includes_items():
    prompt = llm.build_prompt(["Outage at plant X", "  ", "Cold snap next week"])
    assert "plant X" in prompt
    assert "Cold snap" in prompt
    assert "schema" in prompt.lower()


def test_extract_with_mock_caller(tmp_path):
    res = llm.extract_events(
        ["Grohnde-Nord Unit B offline 1400 MW 12-18 March"],
        provider="gemini",
        model="test-model",
        log_dir=tmp_path,
        caller=_fake_caller_json,
    )
    assert len(res.events) == 2
    assert res.events[0].direction == llm.Direction.bullish
    assert res.events[0].capacity_mw == 1400
    assert res.events[1].direction == llm.Direction.bearish
    # An audit log was written.
    logs = list(tmp_path.glob("llm_*.json"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text())
    assert record["status"] == "ok"
    assert record["n_events"] == 2


def test_no_key_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = llm.extract_events(
        ["some outage text"],
        provider="gemini",
        log_dir=tmp_path,
        caller=None,  # real path, but no key -> must not crash
    )
    assert res.events == []
    logs = list(tmp_path.glob("llm_*.json"))
    assert json.loads(logs[0].read_text())["status"] == "skipped_no_api_key"


def test_bad_json_degrades_gracefully(tmp_path):
    res = llm.extract_events(
        ["text"],
        log_dir=tmp_path,
        caller=lambda *_a: "this is not json",
    )
    assert res.events == []
    record = json.loads(list(tmp_path.glob("llm_*.json"))[0].read_text())
    assert record["status"].startswith("error")


def test_schema_rejects_bad_confidence():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        llm.OutageEvent(
            asset="X", fuel_type="gas", direction="bullish", confidence=5.0
        )
