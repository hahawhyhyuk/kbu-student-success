"""AI 학생성공 브리핑과 다음 조치의 근거 제한·fallback 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ai import AIProviderError, MockAIProvider, ResilientAIProvider
from src.ai_copilot_service import (
    AICopilotService,
    BRIEFING_ACTIONS,
    NEXT_ACTION_CHECKS,
)


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "student_id": "S0001",
                "risk_level": "고위험",
                "primary_risk_type": "출결위험형",
                "risk_change": 8.0,
                "gender": "테스트값",
            },
            {
                "student_id": "S0002",
                "risk_level": "주의",
                "primary_risk_type": "출결위험형",
                "risk_change": 2.0,
                "gender": "다른값",
            },
            {
                "student_id": "S0003",
                "risk_level": "관심",
                "primary_risk_type": "진로미설정형",
                "risk_change": -1.0,
                "gender": "미사용",
            },
        ]
    )


def test_copilot_context_uses_aggregates_not_student_or_sensitive_fields() -> None:
    service = AICopilotService(MockAIProvider())
    unaddressed = pd.DataFrame([{"student_id": "S0001"}])

    context = service.build_staff_context(_snapshots(), unaddressed)

    assert context["total_students"] == 3
    assert context["unaddressed_count"] == 1
    assert context["increasing_count"] == 2
    assert context["focus_risk_types"] == ["출결위험형", "진로미설정형"]
    assert "student_id" not in context
    assert "gender" not in context


def test_mock_briefing_uses_only_aggregated_focuses_and_allowed_action() -> None:
    service = AICopilotService(MockAIProvider())
    context = service.build_staff_context(
        _snapshots(), pd.DataFrame([{"student_id": "S0001"}])
    )

    result = service.generate_staff_briefing(context)

    assert set(result.content["focus_risk_types"]).issubset(
        set(context["focus_risk_types"])
    )
    assert result.content["recommended_action"] in BRIEFING_ACTIONS
    assert result.provider_name == "mock"
    assert not any(character.isdigit() for character in result.content["summary"])


def test_next_action_is_limited_to_configured_action_and_checks() -> None:
    service = AICopilotService(MockAIProvider())
    allowed_actions = ("전화 안내", "상담 일정 조율", "후속 체크인 요청")

    result = service.generate_next_action(
        intervention_status="상담 예정",
        has_support_plan=True,
        plan_item_statuses=("지원 예정",),
        allowed_actions=allowed_actions,
    )

    assert result.content["action"] == "상담 일정 조율"
    assert set(result.content["check_before_action"]).issubset(
        set(NEXT_ACTION_CHECKS)
    )


class InvalidBriefingProvider(MockAIProvider):
    """집계에 없는 위험유형을 만드는 실패 provider."""

    provider_name = "invalid"

    def generate_staff_briefing(self, context):
        return {
            "headline": "확인되지 않은 신호",
            "summary": "근거에 없는 유형을 생성했습니다.",
            "focus_risk_types": ["존재하지않는위험형"],
            "recommended_action": context["allowed_actions"][0],
        }


def test_unknown_focus_from_ai_is_rejected() -> None:
    service = AICopilotService(InvalidBriefingProvider())
    context = service.build_staff_context(_snapshots())

    with pytest.raises(AIProviderError, match="집계되지 않은 위험유형"):
        service.generate_staff_briefing(context)


class BrokenCopilotProvider(MockAIProvider):
    provider_name = "broken"

    def generate_staff_briefing(self, context):
        raise AIProviderError("provider unavailable")


def test_briefing_provider_failure_uses_validated_mock_fallback() -> None:
    provider = ResilientAIProvider(
        primary=BrokenCopilotProvider(),
        fallback=MockAIProvider(),
    )
    service = AICopilotService(provider)
    context = service.build_staff_context(_snapshots())

    result = service.generate_staff_briefing(context)

    assert result.provider_name == "mock"
    assert result.warning is not None
    assert result.content["recommended_action"] in BRIEFING_ACTIONS

