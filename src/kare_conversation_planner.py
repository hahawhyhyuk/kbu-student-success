"""Kare가 빈 항목이 아니라 학생의 현재 이야기에서 다음 말을 고르게 한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.checkin_semantic_state import validate_semantic_state


CONVERSATION_THREADS: tuple[str, ...] = (
    "opening",
    "learning",
    "major",
    "career",
    "interests",
    "support",
    "general",
)

CONVERSATION_INTENTS: tuple[str, ...] = (
    "open_story",
    "clarify_scene",
    "clarify_impact",
    "explore_strength",
    "explore_direction",
    "explore_support",
    "offer_options",
    "confirm_summary",
)


@dataclass(frozen=True)
class KareConversationPlan:
    """학생에게 보이지 않는 다음 대화 행동 계획."""

    active_thread: str
    question_intent: str
    next_focus: str
    fallback_question: str

    def as_dict(self) -> dict[str, str]:
        """AI provider에 전달할 식별정보 없는 JSON 호환 맥락을 반환한다."""

        return asdict(self)


def plan_kare_conversation(
    *,
    history: Sequence[Mapping[str, str]],
    user_message: str,
    semantic_state: Mapping[str, Any] | None,
    previous_thread: str | None = None,
    asked_intents: Sequence[str] = (),
    allowed_interest_fields: Sequence[str] = (),
) -> KareConversationPlan:
    """현재 이야기 줄기와 이미 사용한 질문 의도로 다음 행동을 고른다.

    Parameters:
        history: 식별정보가 제거된 기존 대화.
        user_message: 학생의 최신 발화.
        semantic_state: 학생의 직접 근거로 검증된 9개 의미 상태.
        previous_thread: 직전 턴에서 이어가던 이야기 줄기.
        asked_intents: 이전에 사용한 ``이야기 줄기:질문 의도`` 목록.
        allowed_interest_fields: 학생이 예시를 요청할 때 보여줄 DB 기반 분야.

    Returns:
        현재 이야기, 질문 의도, 호환용 의미 초점과 안전한 fallback 질문.

    Assumptions:
        9개 의미 상태는 관찰 기준이며 질문 순서를 결정하는 체크리스트가 아니다.
    """

    del history  # 현재는 세션에 저장된 thread와 intent가 반복 방지 기억이다.
    checked_state = validate_semantic_state(semantic_state)
    message = " ".join(str(user_message or "").strip().split())
    active_thread = detect_active_thread(message, previous_thread)
    if (
        _mentions_allowed_interest(message, allowed_interest_fields)
        and (
            previous_thread == "interests"
            or any(term in message for term in ("관심", "분야", "눈길"))
        )
        and not _mentions_explicit_job(message)
    ):
        active_thread = "interests"
    asked = set(map(str, asked_intents))

    if _is_meta_opening(message):
        intent = "open_story"
        active_thread = "opening"
    elif _asks_for_options(message):
        intent = "offer_options"
        active_thread = "interests"
    elif _is_no_content_reply(message):
        intent, active_thread = _plan_after_no_content(checked_state, asked)
    else:
        intent = _next_thread_intent(active_thread, checked_state, asked)

    next_focus = _focus_for(active_thread, intent, checked_state)
    fallback_question = _fallback_question(
        active_thread=active_thread,
        intent=intent,
        message=message,
        allowed_interest_fields=allowed_interest_fields,
    )
    return KareConversationPlan(
        active_thread=active_thread,
        question_intent=intent,
        next_focus=next_focus,
        fallback_question=fallback_question,
    )


def conversation_ready_for_review(
    semantic_state: Mapping[str, Any] | None,
    *,
    substantive_turn_count: int,
) -> bool:
    """추천에 필요한 최소 맥락이 모였는지 9개 완성과 별도로 판단한다."""

    if substantive_turn_count < 3:
        return False
    state = validate_semantic_state(semantic_state)

    def resolved(dimension: str) -> bool:
        return state[dimension]["state"] != "unknown"

    has_situation = any(
        resolved(dimension)
        for dimension in (
            "natural_language_concern",
            "learning_difficulty",
            "major_satisfaction",
            "major_continuation_intent",
            "career_clarity",
        )
    )
    has_preference = any(
        resolved(dimension)
        for dimension in (
            "major_interest",
            "interest_fields",
            "desired_job",
        )
    )
    has_support_direction = resolved("consultation_intent")
    return has_situation and has_preference and has_support_direction


def conversation_memory(
    previous: Mapping[str, Any] | None,
    plan: KareConversationPlan,
) -> dict[str, Any]:
    """다음 턴 반복 방지에 필요한 최소 대화 기억을 갱신한다."""

    raw = dict(previous or {})
    asked_intents = list(dict.fromkeys(map(str, raw.get("asked_intents", ()))))
    token = f"{plan.active_thread}:{plan.question_intent}"
    if token not in asked_intents:
        asked_intents.append(token)
    return {
        "active_thread": plan.active_thread,
        "last_question_intent": plan.question_intent,
        "asked_intents": asked_intents[-16:],
    }


def detect_active_thread(message: str, previous_thread: str | None = None) -> str:
    """학생의 최신 표현에서 이어갈 이야기 줄기를 찾는다."""

    text = str(message or "")
    term_groups = (
        (
            "support",
            ("상담", "도움", "도와", "같이 정리", "이야기하고 싶"),
        ),
        (
            "learning",
            (
                "수업",
                "과제",
                "시험",
                "공부",
                "학습",
                "진도",
                "외울",
                "암기",
                "성적",
            ),
        ),
        (
            "major",
            (
                "전공",
                "학과",
                "전과",
                "실습",
                "재미있",
                "재밌",
                "재미없",
                "흥미",
            ),
        ),
        (
            "career",
            (
                "진로",
                "직무",
                "취업",
                "졸업 후",
                "되고 싶",
                "해보고 싶은 일",
                "분석가",
                "개발자",
                "기획자",
                "간호사",
                "교사",
            ),
        ),
        (
            "interests",
            ("관심", "분야", "눈길", "흥미 있는 분야"),
        ),
    )
    matches = [
        (thread, min(text.find(term) for term in terms if term in text))
        for thread, terms in term_groups
        if any(term in text for term in terms)
    ]
    if matches:
        # 한 문장에 여러 이야기가 있으면 마지막에 말한 주제를 자연스럽게 이어간다.
        return max(matches, key=lambda item: item[1])[0]
    if previous_thread in CONVERSATION_THREADS and previous_thread != "opening":
        return str(previous_thread)
    return "general"


def thread_for_semantic_focus(focus: str | None) -> str | None:
    """이전 버전의 의미 초점을 현재 이야기 줄기로 안전하게 변환한다."""

    return {
        "natural_language_concern": "general",
        "major_interest": "major",
        "major_satisfaction": "major",
        "major_continuation_intent": "major",
        "learning_difficulty": "learning",
        "career_clarity": "career",
        "interest_fields": "interests",
        "desired_job": "career",
        "consultation_intent": "support",
    }.get(str(focus or ""))


def _next_thread_intent(
    thread: str,
    state: Mapping[str, Mapping[str, Any]],
    asked: set[str],
) -> str:
    sequences = {
        "opening": ("open_story",),
        "learning": (
            "clarify_scene",
            "clarify_impact",
            "explore_strength",
            "explore_support",
            "explore_direction",
        ),
        "major": (
            "clarify_scene",
            "clarify_impact",
            "explore_strength",
            "explore_direction",
            "explore_support",
        ),
        "career": (
            "clarify_scene",
            "explore_direction",
            "explore_support",
        ),
        "interests": (
            "explore_direction",
            "explore_support",
        ),
        "support": (
            "clarify_impact",
            "explore_direction",
        ),
        "general": (
            "open_story",
            "explore_direction",
            "explore_support",
        ),
    }
    for intent in sequences.get(thread, sequences["general"]):
        if f"{thread}:{intent}" not in asked:
            return intent
    if state["consultation_intent"]["state"] == "unknown":
        return "explore_support"
    return "explore_direction"


def _plan_after_no_content(
    state: Mapping[str, Mapping[str, Any]],
    asked: set[str],
) -> tuple[str, str]:
    if (
        state["interest_fields"]["state"] == "unknown"
        and "interests:explore_direction" not in asked
    ):
        return "explore_direction", "interests"
    if (
        state["consultation_intent"]["state"] == "unknown"
        and "support:explore_support" not in asked
    ):
        return "explore_support", "support"
    return "confirm_summary", "general"


def _focus_for(
    thread: str,
    intent: str,
    state: Mapping[str, Mapping[str, Any]],
) -> str:
    if intent == "confirm_summary":
        return "review"
    if intent == "offer_options":
        return "interest_fields"
    if intent == "explore_support":
        return "consultation_intent"
    if intent == "explore_strength":
        return "major_interest"
    if intent == "explore_direction":
        for dimension in ("interest_fields", "desired_job", "career_clarity"):
            if state[dimension]["state"] == "unknown":
                return dimension
        return "consultation_intent"
    if thread == "learning":
        return "learning_difficulty"
    if thread == "major":
        return "major_satisfaction"
    if thread == "career":
        return "career_clarity"
    if thread == "interests":
        return "interest_fields"
    if thread == "support":
        return "consultation_intent"
    return "natural_language_concern"


def _fallback_question(
    *,
    active_thread: str,
    intent: str,
    message: str,
    allowed_interest_fields: Sequence[str],
) -> str:
    if intent == "confirm_summary":
        return "지금까지 나눈 이야기를 한번 정리해 볼까요?"
    if intent == "open_story":
        return "요즘 학교에서 가장 신경 쓰이는 일부터 이야기해 볼까요?"
    if intent == "offer_options":
        examples = list(dict.fromkeys(map(str, allowed_interest_fields)))[:4]
        if examples:
            joined = ", ".join(examples)
            return f"예를 들면 {joined} 같은 분야가 있어요. 이 가운데 눈길이 가는 분야가 있나요?"
        return "최근 수업이나 활동 중에서 조금이라도 눈길이 간 주제가 있었나요?"
    if intent == "explore_support":
        return "지금 가장 먼저 달라졌으면 하는 건 무엇인가요?"
    if intent == "explore_strength":
        return "그래도 수업이나 전공에서 재미있거나 잘 맞는 부분은 있었나요?"
    if intent == "explore_direction":
        if active_thread == "career":
            return "앞으로 한번 해보고 싶은 일이나 역할이 떠오르나요?"
        if active_thread == "interests":
            return "그 관심을 살려서 한번 해보고 싶은 일이나 활동이 있나요?"
        return "이 경험과 연결해 더 알아보고 싶은 분야가 있나요?"
    if intent == "clarify_impact":
        if active_thread == "major" and any(
            term in message for term in ("재미있", "재밌", "흥미", "잘 맞")
        ):
            return "그런 활동을 할 때 평소 수업과 다르게 느껴지는 점이 있나요?"
        return "그 일 때문에 요즘 학교생활에서 달라진 점이 있나요?"
    if active_thread == "learning":
        if "과제" in message and any(term in message for term in ("밀", "시작")):
            return "과제를 시작하기 어려운 건 내용이 막혀서인가요, 시간이 부족해서인가요?"
        if any(term in message for term in ("외울", "암기")):
            return "외운 내용을 정리할 때 어떤 부분에서 가장 많이 막히나요?"
        return "수업을 듣거나 과제를 할 때 가장 자주 막히는 순간은 언제인가요?"
    if active_thread == "major":
        if any(term in message for term in ("재미있", "재밌", "흥미", "잘 맞")):
            return "어떤 점에서 재미있거나 잘 맞는다고 느꼈나요?"
        return "전공이 나와 맞지 않는다고 느끼는 순간은 언제인가요?"
    if active_thread == "career":
        if any(term in message for term in ("되고 싶", "하고 싶", "해보고 싶")):
            return "그 일을 한번 해보고 싶다고 느낀 계기가 있나요?"
        return "진로를 생각할 때 가장 막막하게 느껴지는 부분은 무엇인가요?"
    return "조금 더 구체적으로 떠오르는 상황이 있나요?"


def _is_meta_opening(text: str) -> bool:
    compact = "".join(str(text or "").strip().rstrip(".!? ").split())
    return compact in {
        "요즘학교생활부터이야기할게요",
        "학교생활부터이야기할게요",
        "이야기할게요",
    }


def _is_no_content_reply(text: str) -> bool:
    compact = "".join(str(text or "").strip().rstrip(".!? ").split())
    return compact in {
        "없어",
        "없어요",
        "없음",
        "특별히없어",
        "특별히없어요",
        "모르겠어",
        "모르겠어요",
        "잘모르겠어",
        "잘모르겠어요",
    }


def _asks_for_options(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(term in compact for term in ("뭐가있", "어떤게있", "예시", "종류"))


def _mentions_allowed_interest(
    text: str,
    allowed_interest_fields: Sequence[str],
) -> bool:
    normalized = str(text or "").lower()
    for label in map(str, allowed_interest_fields):
        terms = {label.lower(), *(part.lower() for part in label.split("·"))}
        if any(len(term) >= 2 and term in normalized for term in terms):
            return True
    return False


def _mentions_explicit_job(text: str) -> bool:
    return any(
        term in str(text or "")
        for term in (
            "직무",
            "되고 싶",
            "하고 싶",
            "분석가",
            "개발자",
            "기획자",
            "간호사",
            "교사",
        )
    )
