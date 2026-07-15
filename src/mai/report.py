"""The LLM situation report (종목 4, 7 pts): computed numbers, phrased by a model.

The whole design is a division of labour that protects the points. The report must contain
the runway crater count, the available length, the status and the operability -- and every
one of those is ALREADY computed deterministically by the mission pipeline. So the language
model is never asked to *know* anything; it is asked only to *phrase* numbers we hand it.

That matters because of the scoring (TASK.md 16.5): 5 of the 7 points are the four facts,
and 2 are for using a generative API at all. If we let the model derive the numbers it could
get them wrong and lose content points AND contradict the other JSON files. If instead it
only rewords our numbers -- and we VALIDATE that its output still contains exactly those
numbers -- the worst case is that a bad API call falls back to a template that scores the
same 5 content points. Only the 2 API points ride on the call succeeding.

TASK.md 16.2 also forbids reporting anything not actually detected. The validator enforces
that from the other side: it rejects any sentence that has dropped one of our facts, which
is also what would catch a chatty model inventing extra ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# Available-length -> (status, operability), straight from TASK.md 16.3. Extracted here so
# that runway_status.json and this report can never disagree about the same runway.
_STATUS_BANDS = [
    (2100, "정상", "사용 가능"),
    (1500, "제한 운용", "제한적 사용 가능"),
    (900, "비상 운용", "사용 가능 여부 검토"),
    (0, "운용 불가", "사용 불가, 폐쇄"),
]

# A loose ceiling only. The organizers changed the limit from "50-100" to "about 100", so we
# no longer enforce a tight window -- but a model that returns three paragraphs is still a
# bug, and a runaway output should fail validation and fall back rather than ship.
MAX_CHARS = 130


def status_for(available_m: int) -> tuple[str, str]:
    for threshold, status, operability in _STATUS_BANDS:
        if available_m >= threshold:
            return status, operability
    return _STATUS_BANDS[-1][1], _STATUS_BANDS[-1][2]


@dataclass(frozen=True)
class ReportInputs:
    when: datetime
    runway_crater_count: int
    available_m: int
    status: str
    operability: str

    @classmethod
    def from_mission(
        cls, when: datetime, runway_crater_count: int, available_m: int
    ) -> "ReportInputs":
        status, operability = status_for(available_m)
        return cls(when, runway_crater_count, available_m, status, operability)


def deterministic_report(inputs: ReportInputs) -> str:
    """The example sentence (TASK.md 16.4) with our numbers. Always valid; the safety net."""
    when = inputs.when
    return (
        f"{when.year}년 {when.month:02d}월 {when.day:02d}일 "
        f"{when.hour:02d}시 {when.minute:02d}분 기준, "
        f"활주로 구역에서 폭파구 {inputs.runway_crater_count}개가 탐지되었으며 "
        f"활주로 가용 길이는 {inputs.available_m}미터로 산출된다. "
        f"현재 활주로는 {inputs.status}상태로 {inputs.operability}."
    )


def validate(text: str, inputs: ReportInputs) -> tuple[bool, list[str]]:
    """Does the sentence still carry all four scored facts? Returns (ok, reasons_it_failed).

    Numbers are matched with word boundaries so that "4" does not match inside "1400", which
    is the kind of silent miss that would cost a content point while looking fine.
    """
    reasons: list[str] = []
    stripped = text.strip()

    if len(stripped) > MAX_CHARS:
        reasons.append(f"too long ({len(stripped)} > {MAX_CHARS} chars)")

    # report time: a YYYY년 ... 시 ... 분 stamp with the right year.
    if not re.search(rf"{inputs.when.year}\s*년.*?시.*?분", stripped):
        reasons.append("missing the report time (YYYY년 … 시 … 분)")

    if not re.search(rf"(?<!\d){inputs.runway_crater_count}(?!\d)\s*개", stripped):
        reasons.append(f"missing the crater count ({inputs.runway_crater_count}개)")

    if not re.search(rf"(?<!\d){inputs.available_m}(?!\d)", stripped):
        reasons.append(f"missing the available length ({inputs.available_m})")

    # Status words (정상 / 제한 운용 / 비상 운용 / 운용 불가) are canonical and short, so an
    # exact match is right. But a natural Korean reword inserts particles into the
    # operability -- Gemini writes "사용 가능 여부를 검토 중" for "사용 가능 여부 검토" -- so
    # matching that as an exact substring would reject a perfectly faithful sentence and
    # throw away the 2 API points. Match it by its ordered key tokens with small gaps
    # instead: the meaning must be present, the exact spacing need not be.
    if inputs.status not in stripped:
        reasons.append(f"missing the status ({inputs.status})")
    if not _contains_phrase(stripped, inputs.operability):
        reasons.append(f"missing the operability ({inputs.operability})")

    return (not reasons), reasons


def _contains_phrase(text: str, phrase: str, max_gap: int = 5) -> bool:
    """True if `phrase`'s words appear in order, allowing short filler (particles) between.

    e.g. "사용 가능 여부 검토" matches "사용 가능 여부를 검토 중" but not a sentence that omits
    a word or reorders them.
    """
    tokens = [t for t in re.split(r"[\s,]+", phrase) if t]
    pattern = rf".{{0,{max_gap}}}".join(re.escape(t) for t in tokens)
    return re.search(pattern, text) is not None


def _prompt(inputs: ReportInputs) -> str:
    """Hand the model the exact numbers and the exact format; ask only for rewording.

    We give it the template so a well-behaved model returns something example-shaped, and we
    forbid it from adding anything, because 16.2 penalises unconfirmed content and our
    validator will reject it anyway.
    """
    facts = deterministic_report(inputs)
    return (
        "너는 군 정찰 상황 보고관이다. 아래 확정된 사실만으로 한국어 상황 보고 문장을 "
        "한 문장~두 문장으로 작성하라. 예시 형식과 어투를 따르되, 주어진 숫자와 상태를 "
        "절대 바꾸지 말고, 주어지지 않은 정보는 추가하지 마라. 100자 내외로 작성하라.\n\n"
        f"확정 사실: 보고시각={inputs.when:%Y년 %m월 %d일 %H시 %M분}, "
        f"활주로 폭파구 개수={inputs.runway_crater_count}개, "
        f"가용 길이={inputs.available_m}미터, "
        f"상태={inputs.status}, 운용여부={inputs.operability}\n\n"
        f"예시: {deterministic_report(inputs)}\n\n"
        "보고 문장만 출력하라(따옴표·설명 없이)."
    )


def generate_report(inputs: ReportInputs, client=None) -> tuple[str, str]:
    """A validated report sentence and its source.

    `client(prompt) -> str` is any callable that returns model text (real Gemini in
    production, a stub in tests). Without one, or on any failure or invalid output, we return
    the deterministic template. One retry, because a single reword occasionally drifts.

    Returns (text, source) where source is "llm" or "fallback".
    """
    if client is not None:
        prompt = _prompt(inputs)
        for _ in range(2):
            try:
                candidate = client(prompt).strip().strip('"').strip()
            except Exception:
                break  # network/timeout/etc -> fall back, never raise into the mission
            ok, _reasons = validate(candidate, inputs)
            if ok:
                return candidate, "llm"
    return deterministic_report(inputs), "fallback"
