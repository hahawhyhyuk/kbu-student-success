"""학생의 현재 이야기 중심 Kare 대화 계획 테스트."""

from src.checkin_semantic_state import empty_semantic_state, merge_semantic_updates
from src.kare_conversation_planner import (
    conversation_memory,
    conversation_ready_for_review,
    plan_kare_conversation,
)


def test_learning_story_is_deepened_before_switching_to_another_topic() -> None:
    plan = plan_kare_conversation(
        history=[],
        user_message="수업이 너무 빠르고 외울 게 많아서 과제가 밀렸어요.",
        semantic_state=empty_semantic_state(),
        allowed_interest_fields=("데이터·AI", "보건"),
    )

    assert plan.active_thread == "learning"
    assert plan.question_intent == "clarify_scene"
    assert plan.next_focus == "learning_difficulty"
    assert "과제를 시작하기 어려운" in plan.fallback_question
    assert "관심 분야" not in plan.fallback_question


def test_same_story_advances_question_intent_instead_of_repeating() -> None:
    first = plan_kare_conversation(
        history=[],
        user_message="수업을 따라가기 어려워요.",
        semantic_state=empty_semantic_state(),
    )
    memory = conversation_memory({}, first)
    second = plan_kare_conversation(
        history=[],
        user_message="설명이 빠르면 내용을 놓쳐요.",
        semantic_state=empty_semantic_state(),
        previous_thread=memory["active_thread"],
        asked_intents=memory["asked_intents"],
    )

    assert second.active_thread == "learning"
    assert second.question_intent == "clarify_impact"
    assert second.fallback_question != first.fallback_question


def test_option_request_uses_only_allowed_catalog_values() -> None:
    plan = plan_kare_conversation(
        history=[],
        user_message="관심 분야에는 뭐가 있어요?",
        semantic_state=empty_semantic_state(),
        allowed_interest_fields=("데이터·AI", "보건"),
    )

    assert plan.question_intent == "offer_options"
    assert "데이터·AI" in plan.fallback_question
    assert "보건" in plan.fallback_question
    assert "법률" not in plan.fallback_question


def test_recommendation_readiness_does_not_require_all_nine_dimensions() -> None:
    message = "과제가 밀려서 걱정이고 실습은 재미있으며 상담을 받고 싶어요."
    state = merge_semantic_updates(
        empty_semantic_state(),
        [
            {
                "dimension": "natural_language_concern",
                "state": "identified",
                "evidence": "과제가 밀려서 걱정",
                "confidence": "high",
                "detail": "과제 지연",
            },
            {
                "dimension": "major_interest",
                "state": "positive",
                "evidence": "실습은 재미있으며",
                "confidence": "high",
                "detail": "실습",
            },
            {
                "dimension": "consultation_intent",
                "state": "wants_support",
                "evidence": "상담을 받고 싶어요",
                "confidence": "high",
                "detail": None,
            },
        ],
        latest_user_message=message,
    )

    assert conversation_ready_for_review(state, substantive_turn_count=2) is False
    assert conversation_ready_for_review(state, substantive_turn_count=3) is True
    assert sum(item["state"] == "unknown" for item in state.values()) == 6
