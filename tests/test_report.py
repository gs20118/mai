"""The LLM situation report: numbers computed in Python, phrased by a model, then validated.

The point of these tests is the SAFETY, not the prose. The report is worth 7 points and 5 of
them are the four facts (time, crater count, available length, status/operability). The model
must never be able to drop or change one of those, and when the API is unreachable the report
must still ship valid. No network here -- the "model" is a stub we control.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mai import report as report_mod
from mai.report import ReportInputs, deterministic_report, generate_report, status_for, validate

WHEN = datetime(2026, 7, 16, 14, 27)


@pytest.mark.parametrize(
    "available_m,status,operability",
    [
        (3000, "정상", "사용 가능"),
        (2100, "정상", "사용 가능"),
        (2099, "제한 운용", "제한적 사용 가능"),
        (1500, "제한 운용", "제한적 사용 가능"),
        (1200, "비상 운용", "사용 가능 여부 검토"),
        (900, "비상 운용", "사용 가능 여부 검토"),
        (600, "운용 불가", "사용 불가, 폐쇄"),
        (0, "운용 불가", "사용 불가, 폐쇄"),
    ],
)
def test_status_bands_match_the_spec(available_m, status, operability):
    """TASK.md 16.3 -- and the boundaries, which are where an off-by-one would hide."""
    assert status_for(available_m) == (status, operability)


@pytest.mark.parametrize("count", [0, 1, 4, 10])
@pytest.mark.parametrize("available_m", [3000, 1200, 600, 0])
def test_deterministic_report_always_carries_every_fact(count, available_m):
    """The fallback must be valid for every state the mission can produce."""
    inputs = ReportInputs.from_mission(WHEN, count, available_m)
    text = deterministic_report(inputs)
    ok, reasons = validate(text, inputs)
    assert ok, reasons


def test_deterministic_report_matches_the_example_shape():
    """The 4-crater / 1200m example from TASK.md 16.4, reproduced from our own inputs."""
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    text = deterministic_report(inputs)
    assert "폭파구 4개" in text
    assert "1200미터" in text
    assert "비상 운용상태" in text
    assert "사용 가능 여부 검토" in text
    assert "2026년 07월 16일 14시 27분 기준" in text


def test_validate_rejects_a_dropped_crater_count():
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    # A fluent sentence that quietly omits the count -- the exact failure that loses a point
    # while reading fine.
    text = "2026년 07월 16일 14시 27분 기준 활주로 가용 길이는 1200미터이며 비상 운용상태로 사용 가능 여부 검토."
    ok, reasons = validate(text, inputs)
    assert not ok
    assert any("crater count" in r for r in reasons)


def test_validate_does_not_match_a_count_inside_a_bigger_number():
    """'4개' must not be satisfied by the '4' inside '1400'."""
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    text = "2026년 07월 16일 14시 27분 기준 활주로 폭파구는 1400미터 지점, 가용 길이 1200미터, 비상 운용상태로 사용 가능 여부 검토."
    ok, reasons = validate(text, inputs)
    assert not ok
    assert any("crater count" in r for r in reasons)


def test_validate_rejects_a_runaway_length():
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    text = deterministic_report(inputs) + " " + "추가 설명. " * 20
    ok, reasons = validate(text, inputs)
    assert not ok
    assert any("too long" in r for r in reasons)


def test_generate_falls_back_when_the_model_raises():
    """Offline / timeout / bad key: the report must still ship, from the template."""
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)

    def dead_client(_prompt):
        raise RuntimeError("network down")

    text, source = generate_report(inputs, client=dead_client)
    assert source == "fallback"
    assert validate(text, inputs)[0]


def test_generate_falls_back_when_the_model_hallucinates():
    """A model that changes a number must be rejected, not shipped."""
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)

    def lying_client(_prompt):
        # says 7 craters, not 4 -- plausible, fluent, and wrong.
        return "2026년 07월 16일 14시 27분 기준 활주로 폭파구 7개, 가용 길이 1200미터, 비상 운용상태로 사용 가능 여부 검토."

    text, source = generate_report(inputs, client=lying_client)
    assert source == "fallback"  # rejected, so the template is used
    assert "4개" in text


def test_validate_accepts_a_particle_in_the_operability():
    """A faithful Korean reword inserts particles -- '여부를 검토 중' for '여부 검토'.

    Rejecting that would fail a correct sentence and forfeit the 2 API points. This is the
    exact output the live Gemini model produced.
    """
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    natural = (
        "2026년 07월 16일 14시 27분 기준, 활주로 폭파구 개수는 4개, 가용 길이는 1200미터로 "
        "확인되었습니다. 현재 상태는 비상 운용으로 사용 가능 여부를 검토 중입니다."
    )
    assert validate(natural, inputs)[0]


def test_validate_rejects_a_wrong_operability_for_the_band():
    """'사용 가능' alone is not the 비상-band operability '사용 가능 여부 검토'."""
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)  # 비상 운용
    text = "2026년 07월 16일 14시 27분 기준 폭파구 4개, 가용 길이 1200미터, 비상 운용상태로 사용 가능."
    ok, reasons = validate(text, inputs)
    assert not ok
    assert any("operability" in r for r in reasons)


def test_generate_uses_a_valid_model_sentence():
    """When the model returns a faithful reword, we ship IT (that is the 2 API points)."""
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    good = (
        "2026년 07월 16일 14시 27분 기준 활주로 폭파구 4개 탐지, "
        "가용 길이 1200미터. 비상 운용상태로 사용 가능 여부 검토 필요."
    )

    def good_client(_prompt):
        return good

    text, source = generate_report(inputs, client=good_client)
    assert source == "llm"
    assert text == good


def test_generate_strips_quotes_and_whitespace_from_the_model():
    inputs = ReportInputs.from_mission(WHEN, 4, 1200)
    valid = deterministic_report(inputs)

    def chatty_client(_prompt):
        return f'  "{valid}"  '

    text, source = generate_report(inputs, client=chatty_client)
    assert source == "llm"
    assert text == valid
