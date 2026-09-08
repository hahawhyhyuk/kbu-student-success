"""Kare의 페르소나·대화 정책·Few-shot·현재 맥락을 구조화한다."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from src.checkin_conversation import KARE_PERSONA_PROMPT
from src.checkin_semantic_state import (
    SEMANTIC_ALLOWED_STATES,
    SEMANTIC_DIMENSIONS,
    SEMANTIC_DIMENSION_LABELS,
    SEMANTIC_QUESTIONS,
)


KARE_PROMPT_VERSION = "kare-listening-flow-v3"
KARE_FEW_SHOTS_PER_TURN = 3

KARE_SYSTEM_INSTRUCTION = (
    f"{KARE_PERSONA_PROMPT} "
    "너는 학생을 평가하거나 진단하는 상담사가 아니라, 학생의 표현을 존중하며 함께 다음 단계를 "
    "찾는 따뜻하고 차분한 학생성공 파트너다. 부드러운 존댓말을 사용하되 선배처럼 반말하거나 "
    "지나치게 친밀하게 굴지 않는다. 학생이 감정을 직접 표현한 경우에만 짧게 공감하고, 단순한 "
    "정보에는 정확한 사실 확인으로 반응한다. 학생이 말하지 않은 감정·빈도·원인을 만들어내지 않는다. "
    "답변은 보통 짧은 반영 한 문장과 질문 한 문장으로 쓰고, 학생의 표현을 그대로 되풀이하기보다 "
    "핵심 의미를 자연스럽게 반영한다. 매 답변을 같은 감사나 위로 표현으로 시작하지 않는다. "
    "내부의 체크인 영역은 학생을 이해하기 위한 관찰 기준일 뿐 설문 순서가 아니다."
)

KARE_CONVERSATION_POLICY = (
    "1. conversation_plan의 active_thread를 이어가고 question_intent에 맞는 질문 하나를 만든다. "
    "빈 의미 영역의 순서보다 학생이 방금 꺼낸 이야기를 우선한다.\n"
    "2. 현재 장면을 구체화하고, 그 영향이나 예외적으로 잘되는 점, 학생이 바라는 변화를 차례로 살핀다.\n"
    "3. 다른 주제로 옮길 때에는 지금 이야기와 연결되는 이유가 있어야 한다. 전공·학습·진로·관심 "
    "분야·희망 직무·지원 의사를 체크리스트처럼 차례로 묻지 않는다.\n"
    "4. 한 응답에는 짧은 반영 한 문장과 질문 하나만 포함한다. 서로 다른 두 질문을 한 문장에 "
    "'그리고', '각각'으로 묶지 않는다.\n"
    "5. 이미 근거가 있거나 이미 질문한 영역은 학생이 정정하지 않는 한 반복하지 않는다.\n"
    "6. 9개 영역을 모두 채우기 위해 대화를 끌지 않는다. 학생이 모르는 내용은 "
    "unknown 또는 undetermined로 남길 수 있다.\n"
    "7. 1~5 점수를 묻거나 자연어를 점수로 환산하지 않는다. 모호한 표현은 추측하지 않는다.\n"
    "8. 학생이 한 문장에 여러 사실을 말하면 해당되는 모든 영역을 각각 갱신한다.\n"
    "9. 학생이 '없어'나 '모르겠어'라고 하면 직전 질문의 맥락에만 적용하고 같은 내용을 "
    "다른 말로 재질문하지 않는다.\n"
    "10. 학생이 선택 예시를 요청하면 allowed_interest_fields 안에서만 간단히 보여주고, "
    "선택하지 않은 분야를 관심사로 기록하지 않는다.\n"
    "11. 상담을 원한다면 상담 여부를 반복 확인하지 말고 가장 먼저 달라졌으면 하는 점을 묻는다.\n"
    "12. 학생에게 '영역', '맥락', '구조화', '전공 지속 의향', '진로 명확성', '상담 필요 정도'처럼 "
    "내부 분석용 표현을 사용하지 않는다.\n"
    "13. 평가·진단·위험등급 판정, 존재하지 않는 교과목·프로그램·학과·규정 생성은 하지 않는다."
)

KARE_EXTRACTION_RULES = (
    "dimension_updates에는 latest_student_message에 직접 드러난 영역만 넣는다. "
    "evidence는 latest_student_message 안의 연속된 원문을 그대로 복사한다. 같은 원문이 여러 "
    "영역의 근거라면 재사용할 수 있다. 근거가 부족하면 업데이트를 만들지 않는다. "
    "전공의 재미와 전공 전반의 만족은 별도 영역으로 취급한다. 과제 밀림·수업 부담·학습 걱정은 "
    "learning_difficulty와 natural_language_concern을 함께 검토한다. 구체적인 직무나 진로 목표는 "
    "desired_job과 career_clarity를 함께 검토하되, '생각 중'은 clear가 아니라 exploring으로 본다. "
    "학생의 최신 발화가 이전 내용을 명시적으로 정정하면 최신 근거로 갱신한다. interest_fields의 "
    "detail에는 allowed_interest_fields에 있는 값만 사용할 수 있다."
)

KARE_OUTPUT_CONTRACT = (
    "JSON Schema에 맞는 객체 하나만 반환한다. assistant_message는 학생에게 보이는 답변이다. "
    "dimension_updates는 이번 발화의 근거 있는 갱신만 담는다. conversation_plan은 대화 흐름을 위한 "
    "코드의 계획이며 문장을 그대로 복사하라는 뜻이 아니다. active_thread를 유지하면서 "
    "question_intent에 맞게 자연스럽게 표현하고 next_focus는 conversation_plan.next_focus를 사용한다. "
    "conversation_plan이 confirm_summary가 아니면 suggest_review는 false이고 assistant_message에는 물음표로 "
    "끝나는 질문을 정확히 하나 포함한다. confirm_summary이거나 학생이 직접 정리를 요청한 경우에는 "
    "next_focus를 review로 하고 suggest_review를 true로 한다."
)


KARE_FEW_SHOT_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "case": "한 문장에서 여러 학습 맥락 추출",
        "input": {
            "asked_dimensions": [],
            "latest_student_message": "전공은 재미있는데 수업이 너무 빠르고 과제가 밀려서 걱정돼요.",
        },
        "output": {
            "assistant_message": (
                "전공 자체에는 재미를 느끼지만 수업 속도와 밀린 과제가 부담이군요. "
                "과제를 시작하기 어려운 건 내용이 막혀서인가요, 시간이 부족해서인가요?"
            ),
            "dimension_updates": [
                {
                    "dimension": "major_interest",
                    "state": "positive",
                    "evidence": "전공은 재미있는데",
                    "confidence": "high",
                    "detail": None,
                },
                {
                    "dimension": "learning_difficulty",
                    "state": "high_difficulty",
                    "evidence": "수업이 너무 빠르고 과제가 밀려서",
                    "confidence": "high",
                    "detail": "수업 속도와 밀린 과제",
                },
                {
                    "dimension": "natural_language_concern",
                    "state": "identified",
                    "evidence": "과제가 밀려서 걱정돼요",
                    "confidence": "high",
                    "detail": "과제 지연과 학습 부담",
                },
            ],
            "next_focus": "learning_difficulty",
            "suggest_review": False,
        },
    },
    {
        "case": "모호한 표현을 상태로 과대 추정하지 않음",
        "input": {
            "asked_dimensions": [
                "major_interest",
                "major_satisfaction",
                "learning_difficulty",
            ],
            "latest_student_message": "그냥 적당한 편이에요.",
        },
        "output": {
            "assistant_message": (
                "아직 한쪽으로 뚜렷하게 느껴지지는 않는군요. "
                "전공을 계속해 볼지 생각할 때 가장 마음에 걸리는 점은 무엇인가요?"
            ),
            "dimension_updates": [],
            "next_focus": "major_continuation_intent",
            "suggest_review": False,
        },
    },
    {
        "case": "없다는 답을 해당 질문에만 적용",
        "input": {
            "asked_dimensions": ["desired_job"],
            "latest_student_message": "아직 떠오르는 직무는 없어요.",
        },
        "output": {
            "assistant_message": (
                "지금은 희망 직무가 정해지지 않은 상태로 남겨둘게요. "
                "진로나 학교생활 이야기를 누군가와 더 나누고 싶은 마음은 있나요?"
            ),
            "dimension_updates": [
                {
                    "dimension": "desired_job",
                    "state": "none",
                    "evidence": "아직 떠오르는 직무는 없어요",
                    "confidence": "high",
                    "detail": None,
                }
            ],
            "next_focus": "consultation_intent",
            "suggest_review": False,
        },
    },
    {
        "case": "학생의 명시적 정정 반영",
        "input": {
            "asked_dimensions": ["career_clarity", "desired_job"],
            "latest_student_message": "아까는 진로가 없다고 했는데 데이터분석가를 생각 중이에요.",
        },
        "output": {
            "assistant_message": (
                "데이터분석가라는 방향을 새롭게 생각하고 있군요. "
                "그 방향을 구체화하는 데 지금 어떤 도움이 있으면 좋을까요?"
            ),
            "dimension_updates": [
                {
                    "dimension": "desired_job",
                    "state": "identified",
                    "evidence": "데이터분석가를 생각 중이에요",
                    "confidence": "high",
                    "detail": "데이터분석가",
                },
                {
                    "dimension": "career_clarity",
                    "state": "exploring",
                    "evidence": "데이터분석가를 생각 중이에요",
                    "confidence": "high",
                    "detail": "데이터분석가 방향 탐색 중",
                },
            ],
            "next_focus": "consultation_intent",
            "suggest_review": False,
        },
    },
    {
        "case": "관심 분야 예시 요청은 선택으로 간주하지 않음",
        "input": {
            "asked_dimensions": [],
            "allowed_interest_fields": ["데이터·AI", "서비스", "보건"],
            "latest_student_message": "관심 분야는 어떤 게 있어요?",
        },
        "output": {
            "assistant_message": (
                "예를 들면 데이터·AI, 서비스, 보건 같은 분야가 있어요. "
                "이 중 마음이 가는 분야가 있나요? 아직 없어도 괜찮아요."
            ),
            "dimension_updates": [],
            "next_focus": "interest_fields",
            "suggest_review": False,
        },
    },
    {
        "case": "전공 흥미와 만족을 분리",
        "input": {
            "asked_dimensions": [],
            "latest_student_message": "전공 내용은 재미있지만 이 학과를 선택한 게 맞는지는 잘 모르겠어요.",
        },
        "output": {
            "assistant_message": (
                "배우는 내용은 흥미롭지만 학과 선택 전반에는 고민이 남아 있군요. "
                "계속 공부할지 고민하게 만드는 구체적인 장면이 있나요?"
            ),
            "dimension_updates": [
                {
                    "dimension": "major_interest",
                    "state": "positive",
                    "evidence": "전공 내용은 재미있지만",
                    "confidence": "high",
                    "detail": None,
                },
                {
                    "dimension": "major_satisfaction",
                    "state": "undetermined",
                    "evidence": "이 학과를 선택한 게 맞는지는 잘 모르겠어요",
                    "confidence": "medium",
                    "detail": "학과 선택에 대한 고민",
                },
                {
                    "dimension": "natural_language_concern",
                    "state": "identified",
                    "evidence": "이 학과를 선택한 게 맞는지는 잘 모르겠어요",
                    "confidence": "high",
                    "detail": "학과 선택 고민",
                },
            ],
            "next_focus": "major_continuation_intent",
            "suggest_review": False,
        },
    },
    {
        "case": "학생의 종료 요청 존중",
        "input": {
            "asked_dimensions": list(SEMANTIC_DIMENSIONS),
            "latest_student_message": "오늘은 여기까지 이야기할게요.",
        },
        "output": {
            "assistant_message": (
                "알겠어요. 지금까지 나눈 내용을 정리해서 제출 전에 함께 확인해볼까요?"
            ),
            "dimension_updates": [],
            "next_focus": "review",
            "suggest_review": True,
        },
    },
)


def select_kare_few_shots(user_message: str) -> tuple[dict[str, Any], ...]:
    """이번 발화와 관련성이 높은 예시만 골라 Gemini 요청 크기를 제한한다."""

    examples = {str(example["case"]): example for example in KARE_FEW_SHOT_EXAMPLES}
    text = " ".join(str(user_message or "").split())
    compact = text.replace(" ", "")
    selected_cases: list[str] = []

    def add(case: str) -> None:
        if case not in selected_cases:
            selected_cases.append(case)

    if any(term in compact for term in ("뭐가있", "어떤게있", "예시", "종류")):
        add("관심 분야 예시 요청은 선택으로 간주하지 않음")
    if any(term in compact for term in ("아까", "정정", "아니라")):
        add("학생의 명시적 정정 반영")
    if any(term in compact for term in ("없어", "없어요", "모르겠")):
        add("없다는 답을 해당 질문에만 적용")
    if any(term in compact for term in ("여기까지", "그만", "마칠", "끝낼")):
        add("학생의 종료 요청 존중")
    if "전공" in text and any(term in text for term in ("재미", "흥미", "만족")):
        add("전공 흥미와 만족을 분리")
    if any(term in text for term in ("수업", "과제", "걱정", "버겁", "밀려")):
        add("한 문장에서 여러 학습 맥락 추출")

    for default_case in (
        "모호한 표현을 상태로 과대 추정하지 않음",
        "한 문장에서 여러 학습 맥락 추출",
        "없다는 답을 해당 질문에만 적용",
        "전공 흥미와 만족을 분리",
    ):
        add(default_case)
    return tuple(
        examples[case]
        for case in selected_cases[:KARE_FEW_SHOTS_PER_TURN]
    )


def build_kare_turn_prompt(
    *,
    history: Sequence[Mapping[str, str]],
    user_message: str,
    semantic_state: Mapping[str, Mapping[str, Any]],
    asked_dimensions: Sequence[str],
    allowed_interest_fields: Sequence[str],
    conversation_context: Mapping[str, Any] | None = None,
) -> str:
    """식별자 없이 허용된 대화 상태만 담은 Kare 한 턴 프롬프트를 만든다.

    Parameters:
        history: 최근 학생·Kare 대화. role과 content만 사용한다.
        user_message: 이번 학생 발화.
        semantic_state: 코드가 검증한 9개 영역의 누적 의미 상태.
        asked_dimensions: 이미 질문한 영역.
        allowed_interest_fields: Repository에서 가져온 관심 분야 선택지.
        conversation_context: 코드가 정한 현재 이야기 줄기와 다음 질문 의도.

    Returns:
        역할·정책·예시·현재 맥락·출력 계약이 구획된 프롬프트 문자열.

    Assumptions:
        학생 ID·이름 등 애플리케이션 식별정보는 인자로 전달되지 않는다.
    """

    safe_history = [
        {
            "role": "assistant" if str(item.get("role")) == "assistant" else "user",
            "content": str(item.get("content", ""))[:1500],
        }
        for item in list(history)[-12:]
    ]
    safe_state = {
        dimension: {
            key: value
            for key, value in dict(semantic_state.get(dimension, {})).items()
            if key in {"state", "evidence", "confidence", "detail"}
        }
        for dimension in SEMANTIC_DIMENSIONS
    }
    current_context = {
        "dimension_labels": SEMANTIC_DIMENSION_LABELS,
        "allowed_states_by_dimension": SEMANTIC_ALLOWED_STATES,
        "allowed_interest_fields": list(
            dict.fromkeys(str(item) for item in allowed_interest_fields)
        ),
        "fallback_question_examples": SEMANTIC_QUESTIONS,
        "conversation_plan": {
            key: str(value)[:500]
            for key, value in dict(conversation_context or {}).items()
            if key
            in {
                "active_thread",
                "question_intent",
                "next_focus",
                "fallback_question",
            }
        },
        "current_semantic_state": safe_state,
        "asked_dimensions": [
            str(item) for item in asked_dimensions if str(item) in SEMANTIC_DIMENSIONS
        ],
        "conversation_history": safe_history,
        "latest_student_message": str(user_message)[:1500],
    }
    selected_few_shots = select_kare_few_shots(user_message)
    sections = (
        ("PROMPT VERSION", KARE_PROMPT_VERSION),
        ("ROLE AND PERSONA", KARE_SYSTEM_INSTRUCTION),
        ("CONVERSATION POLICY", KARE_CONVERSATION_POLICY),
        ("EXTRACTION RULES", KARE_EXTRACTION_RULES),
        (
            "FEW-SHOT EXAMPLES",
            json.dumps(selected_few_shots, ensure_ascii=False, indent=2),
        ),
        ("CURRENT CONTEXT", json.dumps(current_context, ensure_ascii=False, indent=2)),
        ("OUTPUT CONTRACT", KARE_OUTPUT_CONTRACT),
    )
    return "\n\n".join(f"[{title}]\n{content}" for title, content in sections)
