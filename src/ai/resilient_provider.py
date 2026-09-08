"""주 provider 실패 시 검증된 Mock 결과로 전환하는 안전장치."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.ai.base import AIProvider
from src.ai.schema import (
    validate_checkin_analysis,
    validate_checkin_chat_turn,
    validate_kare_chat_reply,
    validate_learning_path_explanation,
    validate_microdegree_candidate_description,
    validate_next_action_suggestion,
    validate_staff_briefing,
)


class ResilientAIProvider(AIProvider):
    """Gemini 또는 다른 주 provider의 장애를 fallback provider로 격리한다."""

    provider_name = "resilient"

    def __init__(self, primary: AIProvider, fallback: AIProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_provider_name = fallback.provider_name
        self.last_error: str | None = None

    def analyze_checkin(self, text: str) -> dict[str, Any]:
        """주 provider를 호출하고 모든 실패에서 fallback 결과를 반환한다."""

        try:
            result = validate_checkin_analysis(self.primary.analyze_checkin(text))
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 분석을 사용할 수 없어 Mock 분석으로 전환했습니다. "
                f"원인: {error}"
            )
            return validate_checkin_analysis(self.fallback.analyze_checkin(text))

    def generate_checkin_chat_turn(
        self,
        *,
        history: Sequence[Mapping[str, str]],
        values: Mapping[str, Any],
        covered_fields: Sequence[str],
        user_message: str,
        current_focus: str,
        allowed_interest_fields: Sequence[str],
    ) -> dict[str, Any]:
        """Gemini 대화 실패 시 같은 입력을 로컬 규칙 기반 Kare로 처리한다."""

        allowed_interests = set(map(str, allowed_interest_fields))
        arguments = {
            "history": history,
            "values": values,
            "covered_fields": covered_fields,
            "user_message": user_message,
            "current_focus": current_focus,
            "allowed_interest_fields": allowed_interest_fields,
        }
        try:
            result = validate_checkin_chat_turn(
                self.primary.generate_checkin_chat_turn(**arguments),
                allowed_interests,
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 대화를 사용할 수 없어 외부 전송 없는 규칙 기반 Kare로 "
                f"전환했습니다. 원인: {error}"
            )
            return validate_checkin_chat_turn(
                self.fallback.generate_checkin_chat_turn(**arguments),
                allowed_interests,
            )

    def generate_kare_reply(
        self,
        *,
        history: Sequence[Mapping[str, str]],
        user_message: str,
        semantic_state: Mapping[str, Mapping[str, Any]],
        asked_dimensions: Sequence[str],
        allowed_interest_fields: Sequence[str],
        conversation_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Gemini 맥락 대화 실패 시 같은 의미 상태로 로컬 Kare를 이어간다."""

        arguments = {
            "history": history,
            "user_message": user_message,
            "semantic_state": semantic_state,
            "asked_dimensions": asked_dimensions,
            "allowed_interest_fields": allowed_interest_fields,
            "conversation_context": conversation_context,
        }
        try:
            result = validate_kare_chat_reply(
                self.primary.generate_kare_reply(**arguments)
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 대화를 사용할 수 없어 외부 전송 없는 규칙 기반 Kare로 "
                f"전환했습니다. 원인: {error}"
            )
            return validate_kare_chat_reply(
                self.fallback.generate_kare_reply(**arguments)
            )

    def explain_learning_path(
        self, profile: dict[str, Any], courses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Gemini 설명 실패 시 선택 과목을 유지한 Mock 설명으로 전환한다."""

        allowed_ids = {str(course["course_id"]) for course in courses}
        try:
            result = validate_learning_path_explanation(
                self.primary.explain_learning_path(profile, courses), allowed_ids
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = f"Gemini 학습경로 설명을 사용할 수 없어 Mock 설명으로 전환했습니다. 원인: {error}"
            return validate_learning_path_explanation(
                self.fallback.explain_learning_path(profile, courses), allowed_ids
            )

    def describe_microdegree_candidate(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """Gemini 후보 설명 실패 시 검증된 Mock 설명으로 전환한다."""

        try:
            result = validate_microdegree_candidate_description(
                self.primary.describe_microdegree_candidate(candidate)
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 교육과정 후보 설명을 사용할 수 없어 Mock 설명으로 "
                f"전환했습니다. 원인: {error}"
            )
            return validate_microdegree_candidate_description(
                self.fallback.describe_microdegree_candidate(candidate)
            )

    def describe_microdegree_candidates(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """후보 묶음을 한 번에 설명하고 실패하면 전체를 Mock으로 전환한다."""

        try:
            results = self.primary.describe_microdegree_candidates(candidates)
            if len(results) != len(candidates):
                raise ValueError("교육과정 후보 설명 개수가 입력과 다릅니다.")
            validated = tuple(
                validate_microdegree_candidate_description(result)
                for result in results
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return validated
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 교육과정 후보 설명을 사용할 수 없어 Mock 설명으로 "
                f"전환했습니다. 원인: {error}"
            )
            return tuple(
                validate_microdegree_candidate_description(result)
                for result in self.fallback.describe_microdegree_candidates(
                    candidates
                )
            )

    def generate_staff_briefing(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gemini 브리핑 실패 시 같은 코드 집계를 사용하는 Mock으로 전환한다."""

        focuses = {str(item) for item in context["focus_risk_types"]}
        actions = {str(item) for item in context["allowed_actions"]}
        try:
            result = validate_staff_briefing(
                self.primary.generate_staff_briefing(context), focuses, actions
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 학생성공 브리핑을 사용할 수 없어 Mock 설명으로 "
                f"전환했습니다. 원인: {error}"
            )
            return validate_staff_briefing(
                self.fallback.generate_staff_briefing(context), focuses, actions
            )

    def suggest_next_action(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gemini 다음 조치 설명 실패 시 설정 기반 Mock 설명으로 전환한다."""

        actions = {str(item) for item in context["allowed_actions"]}
        checks = {str(item) for item in context["allowed_checks"]}
        try:
            result = validate_next_action_suggestion(
                self.primary.suggest_next_action(context), actions, checks
            )
            self.last_provider_name = self.primary.provider_name
            self.last_error = None
            return result
        except Exception as error:
            self.last_provider_name = self.fallback.provider_name
            self.last_error = (
                "Gemini 다음 조치 설명을 사용할 수 없어 Mock 설명으로 "
                f"전환했습니다. 원인: {error}"
            )
            return validate_next_action_suggestion(
                self.fallback.suggest_next_action(context), actions, checks
            )
