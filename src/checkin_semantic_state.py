"""Kare 대화에서 9개 체크인 영역을 근거 기반 의미 상태로 관리한다."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


SEMANTIC_DIMENSIONS: tuple[str, ...] = (
    "major_interest",
    "major_satisfaction",
    "major_continuation_intent",
    "learning_difficulty",
    "career_clarity",
    "consultation_intent",
    "interest_fields",
    "desired_job",
    "natural_language_concern",
)

SEMANTIC_DIMENSION_LABELS: dict[str, str] = {
    "major_interest": "전공 흥미",
    "major_satisfaction": "전공 만족",
    "major_continuation_intent": "전공 지속 의향",
    "learning_difficulty": "학습 어려움",
    "career_clarity": "진로 명확성",
    "consultation_intent": "상담 필요",
    "interest_fields": "관심 분야",
    "desired_job": "희망 직무",
    "natural_language_concern": "현재 고민",
}

COMMON_STATES: tuple[str, ...] = ("unknown", "undetermined", "declined")
SEMANTIC_ALLOWED_STATES: dict[str, tuple[str, ...]] = {
    "major_interest": (*COMMON_STATES, "positive", "mixed", "negative"),
    "major_satisfaction": (*COMMON_STATES, "positive", "mixed", "negative"),
    "major_continuation_intent": (
        *COMMON_STATES,
        "continuing",
        "considering",
        "not_continuing",
    ),
    "learning_difficulty": (
        *COMMON_STATES,
        "low_difficulty",
        "moderate_difficulty",
        "high_difficulty",
    ),
    "career_clarity": (
        *COMMON_STATES,
        "clear",
        "exploring",
        "not_clear",
    ),
    "consultation_intent": (
        *COMMON_STATES,
        "wants_support",
        "considering_support",
        "no_support",
    ),
    "interest_fields": (*COMMON_STATES, "identified", "none"),
    "desired_job": (*COMMON_STATES, "identified", "none"),
    "natural_language_concern": (*COMMON_STATES, "identified", "none"),
}

SEMANTIC_STATE_LABELS: dict[str, str] = {
    "unknown": "아직 이야기하지 않음",
    "undetermined": "아직 불명확함",
    "declined": "응답하지 않음",
    "positive": "긍정적",
    "mixed": "좋은 점과 아쉬운 점이 함께 있음",
    "negative": "부정적",
    "continuing": "계속 공부하고 싶음",
    "considering": "계속할지 고민 중",
    "not_continuing": "계속하고 싶은 마음이 낮음",
    "low_difficulty": "큰 어려움 없음",
    "moderate_difficulty": "일부 어려움 있음",
    "high_difficulty": "어려움이 큰 편",
    "clear": "방향이 명확함",
    "exploring": "방향을 탐색 중",
    "not_clear": "방향이 막연함",
    "wants_support": "상담을 원함",
    "considering_support": "상담을 고민 중",
    "no_support": "현재 상담을 원하지 않음",
    "identified": "명시적으로 확인됨",
    "none": "현재 없음",
}

SEMANTIC_FOCUS_GROUPS: dict[str, tuple[str, ...]] = {
    "natural_language_concern": ("natural_language_concern",),
    "major_interest": (
        "major_interest",
        "major_satisfaction",
        "learning_difficulty",
    ),
    "major_satisfaction": ("major_satisfaction",),
    "learning_difficulty": ("learning_difficulty",),
    "major_continuation_intent": (
        "major_continuation_intent",
        "career_clarity",
    ),
    "career_clarity": ("career_clarity",),
    "interest_fields": ("interest_fields", "desired_job"),
    "desired_job": ("desired_job",),
    "consultation_intent": ("consultation_intent",),
}

SEMANTIC_FOCUS_PRIORITY: tuple[str, ...] = (
    "natural_language_concern",
    "major_interest",
    "major_continuation_intent",
    "interest_fields",
    "consultation_intent",
    "learning_difficulty",
    "major_satisfaction",
    "career_clarity",
    "desired_job",
)

SEMANTIC_QUESTIONS: dict[str, str] = {
    "natural_language_concern": "요즘 학교에서 가장 신경 쓰이는 일부터 이야기해 볼까요?",
    "major_interest": "전공에서 재미있거나 눈길이 가는 부분이 있었나요?",
    "major_satisfaction": "전공 생활에서 가장 아쉬운 점은 무엇인가요?",
    "learning_difficulty": "수업이나 과제를 할 때 가장 자주 막히는 순간은 언제인가요?",
    "major_continuation_intent": "이 전공을 앞으로도 계속 해보고 싶은 마음은 어떤가요?",
    "career_clarity": "앞으로 한번 해보고 싶은 일이 떠오르나요?",
    "interest_fields": "요즘 조금이라도 눈길이 가는 분야가 있나요?",
    "desired_job": "한번 해보고 싶은 일이나 역할이 있나요?",
    "consultation_intent": (
        "이 이야기를 혼자 정리해 보고 싶나요, 아니면 누군가와 함께 방법을 찾아보고 싶나요?"
    ),
}


def empty_semantic_state() -> dict[str, dict[str, Any]]:
    """9개 영역이 아직 언급되지 않은 초기 상태를 만든다."""

    return {
        dimension: {
            "state": "unknown",
            "evidence": None,
            "confidence": "low",
            "detail": None,
            "source": None,
            "confirmed": False,
        }
        for dimension in SEMANTIC_DIMENSIONS
    }


def validate_semantic_state(
    value: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """저장·UI에서 사용할 9개 의미 상태를 검증한다."""

    normalized = empty_semantic_state()
    for dimension, raw_item in dict(value or {}).items():
        if dimension not in SEMANTIC_DIMENSIONS or not isinstance(raw_item, Mapping):
            continue
        state = str(raw_item.get("state", "unknown"))
        if state not in SEMANTIC_ALLOWED_STATES[dimension]:
            raise ValueError(f"지원하지 않는 {dimension} 의미 상태입니다: {state}")
        confidence = str(raw_item.get("confidence", "low"))
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"지원하지 않는 확신도입니다: {confidence}")
        source = raw_item.get("source")
        if source not in {None, "student", "student_review"}:
            raise ValueError(f"지원하지 않는 의미 상태 근거입니다: {source}")
        evidence = raw_item.get("evidence")
        detail = raw_item.get("detail")
        normalized[dimension] = {
            "state": state,
            "evidence": str(evidence).strip()[:300] if evidence else None,
            "confidence": confidence,
            "detail": str(detail).strip()[:300] if detail else None,
            "source": source,
            "confirmed": bool(raw_item.get("confirmed", False)),
        }
    return normalized


def merge_semantic_updates(
    current: Mapping[str, Any] | None,
    updates: Sequence[Mapping[str, Any]],
    *,
    latest_user_message: str,
) -> dict[str, dict[str, Any]]:
    """AI 추출 결과 중 최신 학생 발화에 근거가 있는 값만 병합한다."""

    merged = validate_semantic_state(current)
    compact_message = _compact_evidence(latest_user_message)
    for update in updates:
        dimension = str(update.get("dimension", ""))
        state = str(update.get("state", "unknown"))
        evidence = str(update.get("evidence", "")).strip()
        confidence = str(update.get("confidence", "low"))
        if dimension not in SEMANTIC_DIMENSIONS:
            continue
        if state not in SEMANTIC_ALLOWED_STATES[dimension] or state == "unknown":
            continue
        if confidence not in {"low", "medium", "high"}:
            continue
        compact_evidence = _compact_evidence(evidence)
        if not compact_evidence or compact_evidence not in compact_message:
            continue
        if confidence == "low" and state not in {"declined", "none", "undetermined"}:
            state = "undetermined"
        detail = update.get("detail")
        merged[dimension] = {
            "state": state,
            "evidence": evidence[:300],
            "confidence": confidence,
            "detail": str(detail).strip()[:300] if detail else None,
            "source": "student",
            "confirmed": False,
        }
    return merged


def semantic_coverage(state: Mapping[str, Any] | None) -> tuple[str, ...]:
    """unknown이 아닌 근거 있는 영역을 반환한다."""

    checked = validate_semantic_state(state)
    return tuple(
        dimension
        for dimension in SEMANTIC_DIMENSIONS
        if checked[dimension]["state"] != "unknown"
    )


def next_semantic_focus(
    state: Mapping[str, Any] | None,
    asked_dimensions: Sequence[str],
) -> str:
    """ 아직 파악되지 않고 질문하지 않은 다음 대화 주제를 고른다."""

    checked = validate_semantic_state(state)
    asked = set(map(str, asked_dimensions))
    for focus in SEMANTIC_FOCUS_PRIORITY:
        group = SEMANTIC_FOCUS_GROUPS.get(focus, (focus,))
        if any(
            checked[dimension]["state"] == "unknown" and dimension not in asked
            for dimension in group
        ):
            return focus
    return "review"


def mark_focus_asked(
    asked_dimensions: Sequence[str],
    focus: str,
) -> tuple[str, ...]:
    """LLM이 실제로 물은 한 영역만 반복 방지 상태에 기록한다."""

    result = list(dict.fromkeys(map(str, asked_dimensions)))
    dimension = str(focus)
    if dimension in SEMANTIC_DIMENSIONS and dimension not in result:
        result.append(dimension)
    return tuple(result)


def semantic_question(focus: str) -> str:
    """LLM이 반복·척도 질문을 만든 경우의 안전한 후속 질문을 반환한다."""

    return SEMANTIC_QUESTIONS.get(
        str(focus),
        "지금까지 이야기한 내용을 함께 확인해볼까요?",
    )


def semantic_state_label(dimension: str, state: str) -> str:
    """의미 상태를 학생이 확인할 수 있는 표현으로 바꾼다."""

    if dimension not in SEMANTIC_DIMENSIONS:
        raise ValueError(f"지원하지 않는 체크인 영역입니다: {dimension}")
    if state not in SEMANTIC_ALLOWED_STATES[dimension]:
        raise ValueError(f"지원하지 않는 의미 상태입니다: {state}")
    return SEMANTIC_STATE_LABELS[state]


def semantic_review_lines(
    state: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """9개 영역의 상태와 학생 근거를 최종 확인용 문장으로 만든다."""

    checked = validate_semantic_state(state)
    lines = []
    for dimension in SEMANTIC_DIMENSIONS:
        item = checked[dimension]
        line = (
            f"{SEMANTIC_DIMENSION_LABELS[dimension]}: "
            f"{semantic_state_label(dimension, str(item['state']))}"
        )
        evidence = str(item.get("evidence") or "").strip()
        if evidence and item.get("source") == "student":
            line += f" · 근거: “{evidence[:90]}”"
        lines.append(line)
    return tuple(lines)


def semantic_edit_choices(dimension: str) -> tuple[str, ...]:
    """학생이 확인 화면에서 직접 고를 수 있는 상태를 반환한다."""

    if dimension not in SEMANTIC_DIMENSIONS:
        raise ValueError(f"지원하지 않는 체크인 영역입니다: {dimension}")
    return SEMANTIC_ALLOWED_STATES[dimension]


def confirm_semantic_state(
    state: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """학생이 최종 확인한 영역을 표시한다."""

    confirmed = validate_semantic_state(state)
    for item in confirmed.values():
        if item["state"] != "unknown":
            item["confirmed"] = True
    return confirmed


def finalize_asked_semantic_state(
    state: Mapping[str, Any] | None,
    asked_dimensions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """질문했지만 명확한 근거를 얻지 못한 영역을 불명확로 남긴다."""

    finalized = validate_semantic_state(state)
    for dimension in set(map(str, asked_dimensions)):
        if (
            dimension in SEMANTIC_DIMENSIONS
            and finalized[dimension]["state"] == "unknown"
        ):
            finalized[dimension] = {
                "state": "undetermined",
                "evidence": None,
                "confidence": "low",
                "detail": None,
                "source": None,
                "confirmed": False,
            }
    return finalized


def update_semantic_state_from_review(
    state: Mapping[str, Any] | None,
    dimension: str,
    selected_state: str,
) -> dict[str, dict[str, Any]]:
    """확인 화면에서 학생이 직접 수정한 의미 상태를 반영한다."""

    checked = validate_semantic_state(state)
    if selected_state not in SEMANTIC_ALLOWED_STATES.get(dimension, ()):
        raise ValueError("지원하지 않는 확인 값입니다.")
    checked[dimension] = {
        "state": selected_state,
        "evidence": "학생이 확인 화면에서 직접 수정함",
        "confidence": "high",
        "detail": None,
        "source": "student_review",
        "confirmed": True,
    }
    return checked


def _compact_evidence(text: str) -> str:
    """근거 포함 여부 검증에 사용할 최소 문자열을 만든다."""

    return re.sub(r"[\W_]", "", str(text or "").lower(), flags=re.UNICODE)
