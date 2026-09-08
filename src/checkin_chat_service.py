"""Kare 자유대화에서 근거 기반 의미 상태를 누적하는 대화 서비스."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.ai.base import AIProvider
from src.ai.resilient_provider import ResilientAIProvider
from src.ai.schema import validate_kare_chat_reply
from src.checkin_semantic_state import (
    SEMANTIC_DIMENSIONS,
    empty_semantic_state,
    mark_focus_asked,
    merge_semantic_updates,
    semantic_coverage,
    validate_semantic_state,
)
from src.kare_conversation_planner import (
    conversation_memory,
    conversation_ready_for_review,
    plan_kare_conversation,
    thread_for_semantic_focus,
)


@dataclass(frozen=True)
class CheckinChatTurnResult:
    """검증된 Kare 답변과 자유대화 종료 상태."""

    assistant_message: str
    ready_for_review: bool
    provider_name: str
    fallback_used: bool
    warning: str | None = None
    values: dict[str, Any] | None = None
    covered_fields: tuple[str, ...] = ()
    next_focus: str = "conversation"
    semantic_state: dict[str, dict[str, Any]] | None = None
    asked_dimensions: tuple[str, ...] = ()
    conversation_thread: str = "general"
    question_intent: str = "open_story"


class CheckinChatService:
    """학생의 말을 점수로 바꾸지 않고 9개 영역의 의미 맥락을 관리한다.

    LLM은 근거 문장과 의미 상태를 추출하고 맥락에 맞는 질문을 만든다.
    코드는 근거 포함 여부, 반복 주제, 척도 질문과 종료 조건을 통제한다.
    """

    MAX_USER_TURNS = 8
    def __init__(
        self,
        provider: AIProvider,
        allowed_interest_fields: Sequence[str] = (),
    ) -> None:
        self.provider = provider
        # DB master에 존재하는 관심 분야만 안내·정규화할 수 있도록 전달한다.
        self.allowed_interest_fields = tuple(
            dict.fromkeys(str(item) for item in allowed_interest_fields)
        )

    def process_turn(
        self,
        *,
        history: Sequence[Mapping[str, str]],
        user_message: str,
        values: Mapping[str, Any] | None = None,
        covered_fields: Sequence[str] = (),
        current_focus: str | None = None,
        semantic_state: Mapping[str, Any] | None = None,
        asked_dimensions: Sequence[str] = (),
    ) -> CheckinChatTurnResult:
        """학생 발화 한 건에 답하고 확인 화면 제안 여부를 반환한다.

        Parameters:
            history: 학생 식별정보를 제외한 최근 브라우저 세션 대화.
            user_message: 이번 학생 발화.

        Returns:
            검증된 Kare 답변, 종료 상태와 실제 provider 메타데이터.

        Assumptions:
            9개 영역은 학생 발화의 직접 근거로만 정성 상태를 갱신한다.
            이 단계에서는 1~5 점수를 생성하거나 기존 위험점수를 변경하지 않는다.
        """

        normalized_message = str(user_message or "").strip()
        if not normalized_message:
            raise ValueError("Kare에게 보낼 내용을 입력해 주세요.")
        if len(normalized_message) > 1500:
            raise ValueError("한 번에 입력할 수 있는 내용은 1,500자까지입니다.")

        checked_state = validate_semantic_state(
            semantic_state or (values or {}).get("semantic_states") or empty_semantic_state()
        )
        prior_conversation = dict((values or {}).get("kare_conversation") or {})
        plan = plan_kare_conversation(
            history=history,
            user_message=normalized_message,
            semantic_state=checked_state,
            previous_thread=(
                str(prior_conversation.get("active_thread") or "")
                or thread_for_semantic_focus(current_focus)
            ),
            asked_intents=tuple(prior_conversation.get("asked_intents") or ()),
            allowed_interest_fields=self.allowed_interest_fields,
        )
        checked = validate_kare_chat_reply(
            self.provider.generate_kare_reply(
                history=history,
                user_message=normalized_message,
                semantic_state=checked_state,
                asked_dimensions=asked_dimensions,
                allowed_interest_fields=self.allowed_interest_fields,
                conversation_context=plan.as_dict(),
            )
        )
        merged_state = merge_semantic_updates(
            checked_state,
            checked["dimension_updates"],
            latest_user_message=normalized_message,
        )
        prior_user_messages = [
            str(item.get("content", ""))
            for item in history
            if str(item.get("role")) == "user"
        ]
        user_turn_count = len(prior_user_messages) + 1
        no_content_streak = self._no_content_streak(
            [*prior_user_messages, normalized_message]
        )
        explicitly_finished = self._requests_finish(normalized_message)
        substantive_turn_count = self._substantive_turn_count(
            [*prior_user_messages, normalized_message]
        )
        coverage = semantic_coverage(merged_state)
        next_focus = plan.next_focus
        all_dimensions_resolved = len(coverage) == len(SEMANTIC_DIMENSIONS)
        ready_for_review = (
            explicitly_finished
            or no_content_streak >= 2
            or user_turn_count >= self.MAX_USER_TURNS
            or all_dimensions_resolved
            or conversation_ready_for_review(
                merged_state,
                substantive_turn_count=substantive_turn_count,
            )
        )
        assistant_message = str(checked["assistant_message"]).strip()
        unsupported_assumption = False
        internal_jargon = False
        if no_content_streak >= 2:
            assistant_message = (
                "알겠어요. 현재는 구체적으로 정리할 고민이나 요청이 없는 것으로 "
                "남기고 제출 전에 확인해볼게요."
            )
        elif ready_for_review:
            assistant_message = (
                "이야기해줘서 고마워요. 지금까지의 대화를 정리해서 "
                "제출 전에 함께 확인해볼까요?"
            )
        else:
            unsupported_assumption = self._contains_unsupported_assumption(
                assistant_message,
                normalized_message,
            )
            internal_jargon = self._contains_internal_jargon(assistant_message)
        if not ready_for_review and no_content_streak < 2 and (
            self._contains_scale_question(assistant_message)
            or str(checked["next_focus"]) != next_focus
            or bool(checked["suggest_review"])
            or not self._has_exactly_one_question(assistant_message)
            or self._repeats_previous_question(assistant_message, history)
            or unsupported_assumption
            or internal_jargon
        ):
            # 모델의 안전한 반영은 살리되 척도·다중·반복 질문만 현재
            # 이야기 계획의 자연스러운 질문으로 교체한다.
            assistant_message = self._repair_with_question(
                assistant_message,
                plan.fallback_question,
                preserve_reflection=not (unsupported_assumption or internal_jargon),
            )

        updated_asked_dimensions = tuple(map(str, asked_dimensions))
        if not ready_for_review and next_focus != "review":
            updated_asked_dimensions = mark_focus_asked(
                updated_asked_dimensions,
                next_focus,
            )

        provider_name, warning = self._provider_status()
        updated_values = dict(values or {})
        updated_values["semantic_states"] = merged_state
        updated_values["kare_conversation"] = conversation_memory(
            prior_conversation,
            plan,
        )
        return CheckinChatTurnResult(
            assistant_message=assistant_message,
            ready_for_review=ready_for_review,
            provider_name=provider_name,
            fallback_used=provider_name == "mock",
            warning=warning,
            values=updated_values,
            covered_fields=coverage,
            next_focus="review" if ready_for_review else next_focus,
            semantic_state=merged_state,
            asked_dimensions=updated_asked_dimensions,
            conversation_thread=plan.active_thread,
            question_intent=plan.question_intent,
        )

    @classmethod
    def build_student_transcript(
        cls,
        history: Sequence[Mapping[str, str]],
    ) -> str:
        """대화 종료 후 분석할 학생 발화 전체를 한 텍스트로 만든다.

        내용 없는 시작 신호는 제외한다. 실질 내용 없이 없음만 확인된
        대화는 평가 가능한 표준문구로 저장한다.
        """

        user_messages = [
            " ".join(str(item.get("content", "")).strip().split())
            for item in history
            if str(item.get("role")) == "user"
        ]
        substantive = [
            item
            for item in user_messages
            if cls._is_substantive_story(item)
        ]
        if substantive:
            return " ".join(dict.fromkeys(substantive))[-3000:]
        if any(cls._is_no_content_reply(item) for item in user_messages):
            return "현재 특별히 이야기하고 싶은 고민이나 요청은 없습니다."
        return ""

    @staticmethod
    def _is_no_content_reply(text: str) -> bool:
        """짧은 없음·모름 답변인지 확인한다."""

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

    @staticmethod
    def _is_meta_opening(text: str) -> bool:
        """실제 상황 없이 대화를 시작하겠다는 발화인지 확인한다."""

        compact = "".join(str(text or "").strip().rstrip(".!? ").split())
        return compact in {
            "요즘학교생활부터이야기할게요",
            "학교생활부터이야기할게요",
            "이야기할게요",
        }

    @classmethod
    def _is_substantive_story(cls, text: str) -> bool:
        """종료 후 구조화에 사용할 실제 학생 이야기인지 확인한다."""

        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return False
        if cls._is_meta_opening(normalized) or cls._is_no_content_reply(normalized):
            return False
        if cls._requests_finish(normalized):
            return False
        return len("".join(normalized.split())) >= 4

    @classmethod
    def _substantive_turn_count(cls, messages: Sequence[str]) -> int:
        """실제 맥락을 담은 학생 발화 수를 반환한다."""

        return sum(cls._is_substantive_story(item) for item in messages)

    @classmethod
    def _no_content_streak(cls, messages: Sequence[str]) -> int:
        """가장 최근부터 연속된 없음·모름 응답 수를 반환한다."""

        streak = 0
        for item in reversed(messages):
            if cls._is_no_content_reply(item):
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _requests_finish(text: str) -> bool:
        """학생이 직접 대화 종료·정리를 요청했는지 확인한다."""

        normalized = str(text or "").replace(" ", "")
        return any(
            term in normalized
            for term in (
                "여기까지",
                "그만할게",
                "그만할래",
                "정리할게",
                "정리해줘",
                "마칠게",
                "끝낼게",
            )
        )

    @staticmethod
    def _contains_scale_question(text: str) -> bool:
        """provider가 생성한 1~5 척도 질문을 검출한다."""

        compact = "".join(str(text or "").split())
        return (
            ("1점" in compact and "5점" in compact)
            or ("1부터" in compact and "5" in compact)
            or "몇점" in compact
            or "점수로" in compact
        )

    @classmethod
    def _repair_with_question(
        cls,
        message: str,
        question: str,
        *,
        preserve_reflection: bool = True,
    ) -> str:
        """모델의 안전한 맥락 반영은 유지하고 잘못된 질문만 교체한다."""

        sentences = re.split(r"(?<=[.!?])\s+", " ".join(str(message).split()))
        reflections: list[str] = []
        for sentence in sentences if preserve_reflection else ():
            compact = sentence.strip()
            if not compact:
                continue
            if "?" in compact or cls._contains_scale_question(compact):
                continue
            if re.search(r"(?:나요|까요|인가요|어떤가요|무엇인가요)[.]?$", compact):
                continue
            reflections.append(compact)
            if len(reflections) == 2:
                break
        reflection = " ".join(reflections).strip()[:350]
        if not reflection:
            reflection = "지금 이야기한 내용을 이어서 살펴볼게요."
        elif reflection[-1] not in ".!?":
            reflection += "."
        return f"{reflection} {question}"

    @staticmethod
    def _has_exactly_one_question(message: str) -> bool:
        """학생에게 보이는 응답이 질문 하나만 담는지 확인한다."""

        return str(message or "").count("?") == 1

    @classmethod
    def _repeats_previous_question(
        cls,
        message: str,
        history: Sequence[Mapping[str, str]],
    ) -> bool:
        """직전 Kare 질문과 거의 같은 질문을 다시 만드는지 확인한다."""

        current = cls._question_tokens(message)
        if not current:
            return False
        previous_messages = [
            str(item.get("content", ""))
            for item in history
            if str(item.get("role")) == "assistant"
        ]
        for previous in previous_messages[-3:]:
            prior = cls._question_tokens(previous)
            if not prior:
                continue
            overlap = len(current & prior) / max(1, len(current | prior))
            if overlap >= 0.82:
                return True
        return False

    @staticmethod
    def _question_tokens(message: str) -> set[str]:
        question = str(message or "").rsplit(".", 1)[-1]
        words = re.findall(r"[가-힣A-Za-z0-9]+", question)
        stopwords = {"그", "이", "저", "좀", "혹시", "어떤", "무엇", "있나요"}
        return {word for word in words if len(word) > 1 and word not in stopwords}

    @staticmethod
    def _contains_unsupported_assumption(reply: str, student_message: str) -> bool:
        """학생이 말하지 않은 강한 감정·빈도·고립 표현을 검출한다."""

        unsupported_terms = (
            "매일",
            "혼자 마음고생",
            "엄청 힘들",
            "분명 힘들",
            "틀림없이",
            "버티고 있었",
        )
        return any(
            term in str(reply) and term not in str(student_message)
            for term in unsupported_terms
        )

    @staticmethod
    def _contains_internal_jargon(message: str) -> bool:
        """학생 대화에 노출하면 설문처럼 들리는 내부 분석 용어를 검출한다."""

        return any(
            term in str(message or "")
            for term in (
                "전공 지속 의향",
                "진로 명확성",
                "상담 필요 정도",
                "의미 영역",
                "구조화 결과",
            )
        )

    def _provider_status(self) -> tuple[str, str | None]:
        """Resilient provider의 실제 backend와 fallback 경고를 반환한다."""

        if isinstance(self.provider, ResilientAIProvider):
            return self.provider.last_provider_name, self.provider.last_error
        provider_name = self.provider.provider_name
        warning = (
            "Gemini API key가 없어 규칙 기반 Kare 대화로 전환했습니다."
            if provider_name == "mock"
            else None
        )
        return provider_name, warning
