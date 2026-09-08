"""Kare 구조화 페르소나 프롬프트의 고정 계약을 검증한다."""

from __future__ import annotations

from src.ai.schema import validate_kare_chat_reply
from src.checkin_semantic_state import empty_semantic_state
from src.kare_prompt import (
    KARE_FEW_SHOT_EXAMPLES,
    KARE_FEW_SHOTS_PER_TURN,
    KARE_PROMPT_VERSION,
    KARE_SYSTEM_INSTRUCTION,
    build_kare_turn_prompt,
    select_kare_few_shots,
)


def test_kare_prompt_separates_persona_policy_examples_context_and_output() -> None:
    """한 문자열 안에서도 역할·규칙·예시·입력·출력을 명확히 구획한다."""

    prompt = build_kare_turn_prompt(
        history=[],
        user_message="요즘 과제가 밀려서 걱정이에요.",
        semantic_state=empty_semantic_state(),
        asked_dimensions=[],
        allowed_interest_fields=["데이터·AI", "보건"],
    )

    assert KARE_PROMPT_VERSION in prompt
    assert "[ROLE AND PERSONA]" in prompt
    assert "[CONVERSATION POLICY]" in prompt
    assert "[EXTRACTION RULES]" in prompt
    assert "[FEW-SHOT EXAMPLES]" in prompt
    assert "[CURRENT CONTEXT]" in prompt
    assert "[OUTPUT CONTRACT]" in prompt
    assert "9개 영역을 모두 채우기 위해 대화를 끌지 않는다" in prompt
    assert "1~5 점수를 묻거나" in prompt
    assert "짧은 반영 한 문장과 질문 한 문장" in KARE_SYSTEM_INSTRUCTION
    assert "학생이 말하지 않은 감정·빈도·원인" in KARE_SYSTEM_INSTRUCTION
    assert prompt.count('"case":') == KARE_FEW_SHOTS_PER_TURN


def test_kare_few_shots_are_schema_valid_and_cover_failure_cases() -> None:
    """예시는 다중 추출·모호함·없음·정정·선택지·종료 상황을 포함한다."""

    assert len(KARE_FEW_SHOT_EXAMPLES) >= 6
    validated = [
        validate_kare_chat_reply(example["output"])
        for example in KARE_FEW_SHOT_EXAMPLES
    ]
    cases = {str(example["case"]): example for example in KARE_FEW_SHOT_EXAMPLES}

    multi_dimensions = {
        update["dimension"]
        for update in cases["한 문장에서 여러 학습 맥락 추출"]["output"][
            "dimension_updates"
        ]
    }
    assert multi_dimensions == {
        "major_interest",
        "learning_difficulty",
        "natural_language_concern",
    }
    assert (
        cases["모호한 표현을 상태로 과대 추정하지 않음"]["output"][
            "dimension_updates"
        ]
        == []
    )
    assert (
        cases["관심 분야 예시 요청은 선택으로 간주하지 않음"]["output"][
            "dimension_updates"
        ]
        == []
    )
    assert cases["학생의 종료 요청 존중"]["output"]["next_focus"] == "review"
    assert len(validated) == len(KARE_FEW_SHOT_EXAMPLES)


def test_kare_selects_relevant_few_shots_without_sending_full_catalog() -> None:
    """선택지·정정 발화에는 관련 예시를 우선하고 요청당 예시 수를 제한한다."""

    option_cases = {
        example["case"]
        for example in select_kare_few_shots("관심 분야에는 어떤 게 있어요?")
    }
    correction_cases = {
        example["case"]
        for example in select_kare_few_shots(
            "아까는 없다고 했는데 데이터분석가로 정정할게요."
        )
    }

    assert "관심 분야 예시 요청은 선택으로 간주하지 않음" in option_cases
    assert "학생의 명시적 정정 반영" in correction_cases
    assert len(option_cases) == KARE_FEW_SHOTS_PER_TURN
    assert len(correction_cases) == KARE_FEW_SHOTS_PER_TURN


def test_kare_context_uses_only_allowed_state_and_message_keys() -> None:
    """애플리케이션 식별자나 임의 상태 키를 프롬프트 Context에 넣지 않는다."""

    state = empty_semantic_state()
    state["student_id"] = {"state": "S0003"}
    state["major_interest"]["student_name"] = "가상학생"
    prompt = build_kare_turn_prompt(
        history=[
            {
                "role": "assistant",
                "content": "편하게 이야기해 주세요.",
                "student_id": "S0003",
            }
        ],
        user_message="수업은 어렵지만 데이터 분야에는 관심이 있어요.",
        semantic_state=state,
        asked_dimensions=["learning_difficulty", "invalid_dimension"],
        allowed_interest_fields=["데이터·AI", "데이터·AI", "보건"],
    )

    assert "student_id" not in prompt
    assert "student_name" not in prompt
    assert "invalid_dimension" not in prompt
    assert prompt.count('"데이터·AI"') >= 1
    assert '"asked_dimensions": [' in prompt
    assert '"learning_difficulty"' in prompt
    assert "수업은 어렵지만 데이터 분야에는 관심이 있어요." in prompt
