"""Programmatic LLM component: free-text outage/news -> structured features.

Why this exists
---------------
A power desk continuously receives unstructured text - plant-outage notices,
news headlines, REMIT messages - such as "Unit B of the Grohnde nuclear plant
will be offline for 1.4 GW from 12-18 March for unplanned repairs". Reading
these by hand into model inputs is slow and inconsistent. This module uses an
LLM with a *strict output schema* to convert such text into typed, validated
records (asset, fuel, capacity, outage window, price direction, confidence)
that could feed a "supply-disruption" feature for the price model.

Design choices (for reproducibility, safety, and review)
-------------------------------------------------------
* Provider-agnostic: Gemini (default) or Groq, chosen in config. The network
  call is isolated behind a small ``caller`` function so it can be mocked - the
  parsing, validation, and logging are tested offline with no key or network.
* Strict schema: a Pydantic model is passed as the response schema, so the
  output is validated JSON, not free text parsed with brittle heuristics.
* Audit log: every call writes its prompt, raw response, status, and parsed
  result to ``logs/llm/`` for reproducibility and inspection.
* Graceful degradation: with no API key or no connectivity the pipeline does
  not crash - it logs a clear status and returns zero events, so the rest of
  the project runs fully offline.

Note on data: the bundled sample is *synthetic* (made-up plants and dates) so
the component is reproducible and free of news-copyright concerns. In
production the same code would read a real REMIT / outage feed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_INSTRUCTION = (
    "You are a power-market analyst. Extract supply/demand disruption events "
    "from the text into the given JSON schema. Use the 'direction' field for "
    "the price impact: 'bullish' = upward price pressure (e.g. supply lost, "
    "outage, demand surge), 'bearish' = downward pressure (e.g. capacity "
    "returns, demand falls), 'neutral' if unclear. Set capacity_mw and dates "
    "to null when not stated. Report only events grounded in the text; do not "
    "invent details. confidence is your 0-1 certainty in the extraction."
)


class Direction(StrEnum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class OutageEvent(BaseModel):
    """One structured supply/demand disruption extracted from text."""

    asset: str = Field(description="Plant, unit, or line named in the text")
    fuel_type: str = Field(description="e.g. nuclear, lignite, gas, wind, solar, transmission")
    capacity_mw: float | None = Field(default=None, description="Affected capacity in MW, or null")
    start: str | None = Field(default=None, description="ISO date the disruption starts, or null")
    end: str | None = Field(default=None, description="ISO date it ends, or null")
    direction: Direction = Field(description="Price impact: bullish/bearish/neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 certainty in this extraction")


class ExtractionResult(BaseModel):
    """The full set of events extracted from a batch of text."""

    events: list[OutageEvent] = Field(default_factory=list)


# --- prompt -----------------------------------------------------------------

def build_prompt(texts: list[str]) -> str:
    """Assemble the extraction prompt from a batch of input snippets."""
    joined = "\n".join(f"- {t.strip()}" for t in texts if t.strip())
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Extract all disruption events from these items:\n{joined}\n\n"
        "Return JSON matching the schema with an 'events' array."
    )


# --- provider callers (isolated so tests can mock them) ---------------------

def _call_gemini(prompt: str, model: str, api_key: str, temperature: float) -> str:
    from google import genai  # imported lazily so the module loads without the SDK

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractionResult,
            "temperature": temperature,
        },
    )
    return resp.text


def _call_groq(prompt: str, model: str, api_key: str, temperature: float) -> str:
    from groq import Groq  # imported lazily

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


_CALLERS: dict[str, Callable[[str, str, str, float], str]] = {
    "gemini": _call_gemini,
    "groq": _call_groq,
}
_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


# --- logging ----------------------------------------------------------------

def _write_log(log_dir: Path, record: dict) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    path = log_dir / f"llm_{stamp}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


# --- main entry point -------------------------------------------------------

def extract_events(
    texts: list[str],
    *,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    log_dir: str | Path = "logs/llm",
    caller: Callable[[str, str, str, float], str] | None = None,
) -> ExtractionResult:
    """Extract structured disruption events from text.

    Pass ``caller`` to inject a stub (used in tests) - then no key or network is
    needed. Otherwise the provider's real client is used, reading the key from
    the appropriate environment variable. Any missing key or failure degrades
    gracefully to an empty result, always leaving an audit log behind.
    """
    log_path = ROOT / log_dir if not Path(log_dir).is_absolute() else Path(log_dir)
    prompt = build_prompt(texts)
    api_key = os.environ.get(_KEY_ENV.get(provider, ""), "")

    base = {
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": provider,
        "model": model,
        "prompt": prompt,
    }

    if caller is None:
        if not api_key:
            _write_log(log_path, {**base, "status": "skipped_no_api_key", "raw": None})
            return ExtractionResult()
        caller = _CALLERS[provider]

    try:
        raw = caller(prompt, model, api_key, temperature)
        result = ExtractionResult.model_validate_json(raw)
    except Exception as exc:  # network, parse, or validation failure -> degrade
        _write_log(log_path, {**base, "status": f"error: {type(exc).__name__}: {exc}", "raw": None})
        return ExtractionResult()

    _write_log(
        log_path,
        {**base, "status": "ok", "raw": raw, "n_events": len(result.events)},
    )
    return result


def extract_from_config(
    texts: list[str],
    cfg: dict,
    caller: Callable[[str, str, str, float], str] | None = None,
) -> ExtractionResult:
    """Convenience wrapper that reads provider/model/etc. from the config dict."""
    llm = cfg.get("llm", {})
    return extract_events(
        texts,
        provider=llm.get("provider", "gemini"),
        model=llm.get("model", "gemini-2.5-flash"),
        temperature=llm.get("temperature", 0.0),
        log_dir=llm.get("log_dir", "logs/llm"),
        caller=caller,
    )
