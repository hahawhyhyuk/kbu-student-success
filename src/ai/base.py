"""생성형 AI 구현체가 지켜야 하는 provider 계약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class AIProviderError(RuntimeError):
    """AI 호출 또는 안전한 응답 해석에 실패했음을 나타낸다."""


class AIProvider(ABC):
    """Gemini와 Mock 구현을 교체 가능하게 만드는 인터페이스."""

    provider_name = "base"

    @abstractmethod
    def analyze_checkin(self, text: str) -> dict[str, Any]:
        """학생 자유서술을 검증 가능한 구조로 변환한다."""

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
        """자유대화 한 턴의 학생 응답을 이해하고 다음 Kare 응답을 생성한다."""

        raise NotImplementedError

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
        """9개 근거 상태와 현재 이야기 계획으로 Kare의 한 턴을 생성한다."""

        raise NotImplementedError

    def explain_intervention(
        self, profile: Mapping[str, Any], programs: Sequence[Mapping[str, Any]]
    ) -> str:
        """후속 확장용 개입 설명 인터페이스."""

        raise NotImplementedError

    def explain_learning_path(
        self, profile: Mapping[str, Any], courses: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """후속 확장용 학습경로 설명 인터페이스."""

        raise NotImplementedError

    def describe_microdegree_candidate(
        self, candidate: Mapping[str, Any]
    ) -> dict[str, Any]:
        """코드가 선정한 교육과정 개발 후보의 이름과 설명을 생성한다."""

        raise NotImplementedError

    def describe_microdegree_candidates(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """여러 교육과정 후보 설명을 입력 순서대로 반환한다.

        기본 구현은 기존 단건 provider와의 호환성을 위해 순차 호출한다.
        외부 API provider는 호출 한도를 줄일 수 있도록 이 메서드를 재정의한다.
        """

        return tuple(
            self.describe_microdegree_candidate(candidate)
            for candidate in candidates
        )

    def generate_staff_briefing(
        self, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """코드가 집계한 최신 신호를 교직원용 브리핑으로 설명한다."""

        raise NotImplementedError

    def suggest_next_action(
        self, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """현재 지원 상태와 허용 행동 안에서 다음 조치를 설명한다."""

        raise NotImplementedError
