"""Gemini text generation for the situation report. One provider, hard-wired by request.

The only caller is `report.generate_report`, and it treats any exception here as "fall back
to the template". So this module's contract is simple: return a string on success, raise on
anything else. It never blocks the mission -- a hard timeout guarantees that a hall full of
teams hammering the same free-tier endpoint cannot stall us past the 180s clock.

KEY HANDLING: GEMINI_API_KEY is read from the environment and nowhere else. It is never
written to a tracked file, never logged, never included in an error message.
"""

from __future__ import annotations

import os


def gemini_complete(
    prompt: str, model: str = "gemini-flash-latest", timeout: float = 20.0
) -> str:
    """One non-streaming Gemini call. Returns the text, or raises on any failure.

    The import is inside the function so the rest of the pipeline runs whether or not
    google-genai is installed -- the report is the only feature that needs it.

    Timeout is 20s. Gemini rejects any deadline under 10s, and the free-tier flash model
    measured ~11s per call -- so 12s sat right on the edge and intermittently timed out. This
    is a SINGLE call after the whole mission is analysed, not per-frame, so a generous 20s
    ceiling is fine, and a genuine hang past it still falls back to the template. (Note: the
    free tier also has a low requests-per-minute cap; back-to-back calls can 429, which also
    falls back cleanly. A real mission makes one call, so it does not hit that.)
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,  # low: we want faithful rewording, not creativity
            max_output_tokens=256,
            # Gemini 2.5 flash "thinks" by default, and the reasoning tokens are drawn from
            # the SAME output budget -- so a 200-token cap gets spent on thinking and the
            # actual sentence comes back truncated to a few characters. Rewording one
            # sentence needs no reasoning, so we switch thinking off: it fixes the
            # truncation, and it is faster and cheaper.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),  # ms
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text
