"""Kare 대화형 선택지와 기존 체크인 스키마 연결 테스트."""

from __future__ import annotations

import pytest

from src.checkin_conversation import (
    CHAT_OBSERVATION_FIELDS,
    CHAT_REQUIRED_FIELDS,
    SCALE_QUESTIONS,
    build_review_sections,
    empty_chat_values,
    normalize_analysis_interests,
    scale_response,
    scale_score,
    supported_scale_score,
    validate_conversation_values,
    validate_narrative_values,
)


def test_each_conversational_scale_preserves_one_to_five_values() -> None:
    """자연어 선택지는 모든 기존 1~5 점수와 손실 없이 왕복한다."""

    for question in SCALE_QUESTIONS:
        for expected_score, response in enumerate(question.options, start=1):
            assert scale_score(question.field, response) == expected_score
            assert scale_response(question.field, expected_score) == response


def test_empty_chat_state_tracks_all_required_fields_without_default_scores() -> None:
    """자유대화 시작 시 필수 항목은 유지하되 임의 3점으로 채우지 않는다."""

    values = empty_chat_values()

    assert CHAT_REQUIRED_FIELDS == (
        "natural_language_concern",
        "interest_fields",
        "desired_job",
    )
    assert all(
        values[question.field] is None for question in SCALE_QUESTIONS
    )
    assert set(CHAT_REQUIRED_FIELDS).issubset(CHAT_OBSERVATION_FIELDS)
    assert values["response_mode"] == "narrative"


def test_conversation_values_are_validated_and_summarized() -> None:
    values = {
        question.field: 3 for question in SCALE_QUESTIONS
    }
    values.update(
        {
            "interest_fields": ["데이터·AI", "서비스"],
            "desired_job": "서비스 기획",
            "natural_language_concern": "진로 방향을 구체화하고 싶어요.",
            "consultation_requested": True,
            "explore_other_fields": False,
        }
    )

    validated = validate_conversation_values(values)
    sections = build_review_sections(validated)

    assert set(sections) == {
        "나의 이야기",
        "관심 분야",
        "희망 직무",
        "직접 말한 상태",
        "원하는 지원",
    }
    assert len(sections["직접 말한 상태"]) == 6
    assert sections["관심 분야"] == ("데이터·AI", "서비스")
    assert sections["희망 직무"] == ("서비스 기획",)


def test_incomplete_conversation_is_rejected() -> None:
    with pytest.raises(ValueError, match="아직 완료되지"):
        validate_conversation_values({"major_interest": 3})


def test_narrative_checkin_keeps_missing_scales_empty() -> None:
    """자연어 체크인은 9개 항목을 채우기 위해 점수를 추정하지 않는다."""

    validated = validate_narrative_values(
        {
            **empty_chat_values(),
            "natural_language_concern": "수업이 빨라 과제가 밀리고 진로 준비가 궁금해요.",
            "interest_fields": ["데이터·AI"],
        }
    )

    assert validated["response_mode"] == "narrative"
    assert validated["major_interest"] is None
    assert validated["learning_difficulty"] is None
    assert validated["interest_fields"] == ["데이터·AI"]


def test_analysis_interests_are_normalized_only_with_transcript_evidence() -> None:
    """Gemini의 자유 표현은 원문 근거가 있을 때만 master 분야로 변환한다."""

    allowed = ("데이터·AI", "서비스", "상담·복지", "보건")
    result = normalize_analysis_interests(
        ["데이터", "보건", "서비스", "상담"],
        transcript=(
            "데이터와 보건 분야에 관심이 있고 수업이 버거워 "
            "상담도 받고 싶어요."
        ),
        allowed_interests=allowed,
    )

    assert result == ["데이터·AI", "보건"]


def test_supported_scale_score_requires_unambiguous_evidence() -> None:
    assert supported_scale_score("major_interest", "전공 흥미는 2점이에요") == 2
    assert supported_scale_score("learning_difficulty", "조금 어려운 편이에요") == 4
    assert supported_scale_score("career_clarity", "아직 확실하지 않아요") is None
    assert supported_scale_score("learning_difficulty", "가끔 벅차요") is None


def test_supported_scale_score_accepts_bare_answer_only_for_current_focus() -> None:
    assert supported_scale_score(
        "major_interest", "2점이에요", allow_bare_score=True
    ) == 2
    assert supported_scale_score("major_interest", "2점이에요") is None
