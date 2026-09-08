"""Kare 9개 영역의 근거 기반 의미 상태 테스트."""

from src.checkin_semantic_state import (
    SEMANTIC_DIMENSIONS,
    confirm_semantic_state,
    empty_semantic_state,
    mark_focus_asked,
    merge_semantic_updates,
    next_semantic_focus,
    semantic_coverage,
    semantic_review_lines,
)


def test_empty_semantic_state_contains_all_nine_dimensions() -> None:
    state = empty_semantic_state()

    assert tuple(state) == SEMANTIC_DIMENSIONS
    assert all(item["state"] == "unknown" for item in state.values())
    assert semantic_coverage(state) == ()


def test_multiple_meanings_are_merged_only_with_student_evidence() -> None:
    message = "전공은 재미있지만 수업이 너무 어려워요."
    state = merge_semantic_updates(
        empty_semantic_state(),
        [
            {
                "dimension": "major_interest",
                "state": "positive",
                "evidence": "전공은 재미있지만",
                "confidence": "high",
                "detail": None,
            },
            {
                "dimension": "learning_difficulty",
                "state": "high_difficulty",
                "evidence": "수업이 너무 어려워요",
                "confidence": "high",
                "detail": None,
            },
            {
                "dimension": "career_clarity",
                "state": "not_clear",
                "evidence": "진로가 막막해요",
                "confidence": "high",
                "detail": None,
            },
        ],
        latest_user_message=message,
    )

    assert state["major_interest"]["state"] == "positive"
    assert state["learning_difficulty"]["state"] == "high_difficulty"
    assert state["career_clarity"]["state"] == "unknown"
    assert set(semantic_coverage(state)) == {
        "major_interest",
        "learning_difficulty",
    }


def test_low_confidence_does_not_become_a_strong_student_state() -> None:
    state = merge_semantic_updates(
        empty_semantic_state(),
        [
            {
                "dimension": "major_satisfaction",
                "state": "positive",
                "evidence": "나쁘지는 않아요",
                "confidence": "low",
                "detail": None,
            }
        ],
        latest_user_message="나쁘지는 않아요.",
    )

    assert state["major_satisfaction"]["state"] == "undetermined"


def test_mark_focus_records_only_the_question_that_was_actually_asked() -> None:
    state = empty_semantic_state()
    asked = mark_focus_asked((), "major_interest")

    assert asked == ("major_interest",)
    assert next_semantic_focus(state, asked) == "natural_language_concern"


def test_confirmation_preserves_evidence_and_marks_resolved_states() -> None:
    state = merge_semantic_updates(
        empty_semantic_state(),
        [
            {
                "dimension": "desired_job",
                "state": "identified",
                "evidence": "데이터분석가가 되고 싶어요",
                "confidence": "high",
                "detail": "데이터분석가",
            }
        ],
        latest_user_message="데이터분석가가 되고 싶어요.",
    )

    confirmed = confirm_semantic_state(state)
    lines = semantic_review_lines(confirmed)

    assert confirmed["desired_job"]["confirmed"] is True
    assert confirmed["major_interest"]["confirmed"] is False
    assert any("희망 직무: 명시적으로 확인됨" in line for line in lines)
