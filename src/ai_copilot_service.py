"""코드 집계를 안전한 AI 브리핑과 교직원 다음 조치 설명으로 연결한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.ai import AIProvider, create_ai_provider
from src.ai.schema import (
    validate_next_action_suggestion,
    validate_staff_briefing,
)


BRIEFING_ACTIONS = (
    "전체 대시보드에서 분포 확인",
    "학생 종합 확인에서 근거 검토",
    "아직 개입되지 않은 고위험 학생 확인",
    "학생지원 진행 관리에서 계획 확정",
    "지원 후 변화 확인에서 후속 관찰",
)
NEXT_ACTION_CHECKS = (
    "학생 참여 의사",
    "연락 가능 시간",
    "프로그램 운영 일정",
    "담당 부서 인계 여부",
    "최근 체크인 변화",
)


@dataclass(frozen=True)
class CopilotResult:
    """검증된 AI 설명과 실제 사용 provider 정보."""

    content: dict[str, Any]
    provider_name: str
    warning: str | None = None


class AICopilotService:
    """LLM이 코드 집계 값과 설정된 행동 밖으로 벗어나지 않도록 조율한다."""

    REQUIRED_BRIEFING_COLUMNS = {
        "student_id",
        "risk_level",
        "primary_risk_type",
        "risk_change",
    }

    def __init__(self, provider: AIProvider | None = None) -> None:
        self.provider = provider or create_ai_provider()

    def build_staff_context(
        self,
        latest_snapshots: pd.DataFrame,
        unaddressed: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """최신 학생 스냅샷에서 LLM 입력용 최소 집계만 만든다.

        Parameters:
            latest_snapshots: 현재 필터에 포함된 학생별 최신 위험 스냅샷.
            unaddressed: 코드가 선별한 지원 시작 전 고위험 학생 목록.

        Returns:
            개인 원천값 없이 등급 수·집중 위험유형·변화 건수만 포함한 context.

        Assumptions:
            학생 ID는 행 수 확인에만 쓰며 provider 입력에는 전달하지 않는다.
        """

        missing = self.REQUIRED_BRIEFING_COLUMNS.difference(
            latest_snapshots.columns
        )
        if missing:
            raise ValueError(
                f"AI 브리핑 집계에 필요한 컬럼이 없습니다: {sorted(missing)}"
            )
        risk_level_counts = {
            level: int((latest_snapshots["risk_level"] == level).sum())
            for level in ("정상", "관심", "주의", "고위험")
        }
        at_risk = latest_snapshots[
            latest_snapshots["risk_level"] != "정상"
        ]
        focus_counts = at_risk["primary_risk_type"].value_counts()
        focuses = [str(item) for item in focus_counts.head(3).index]
        if not focuses:
            focuses = ["정기 모니터링"]
        return {
            "total_students": int(len(latest_snapshots)),
            "risk_level_counts": risk_level_counts,
            "focus_risk_types": focuses,
            "unaddressed_count": int(
                len(unaddressed) if unaddressed is not None else 0
            ),
            "increasing_count": int(
                (pd.to_numeric(latest_snapshots["risk_change"], errors="coerce") > 0)
                .fillna(False)
                .sum()
            ),
            "allowed_actions": list(BRIEFING_ACTIONS),
        }

    def generate_staff_briefing(
        self, context: Mapping[str, Any]
    ) -> CopilotResult:
        """집계된 위험유형과 화면 행동만 사용하는 AI 브리핑을 반환한다."""

        active_context = dict(context)
        content = validate_staff_briefing(
            self.provider.generate_staff_briefing(active_context),
            {str(item) for item in active_context["focus_risk_types"]},
            {str(item) for item in active_context["allowed_actions"]},
        )
        return self._result(content)

    def generate_next_action(
        self,
        *,
        intervention_status: str,
        has_support_plan: bool,
        plan_item_statuses: Sequence[str],
        allowed_actions: Sequence[str],
    ) -> CopilotResult:
        """현재 진행상태에 맞는 설정 기반 표준 행동 하나를 AI가 설명하게 한다."""

        actions = [str(item) for item in allowed_actions if str(item).strip()]
        if not actions:
            raise ValueError("AI 다음 조치에 사용할 표준 행동이 없습니다.")
        context = {
            "intervention_status": str(intervention_status),
            "has_support_plan": bool(has_support_plan),
            "plan_item_statuses": [
                str(item) for item in plan_item_statuses if str(item).strip()
            ],
            "allowed_actions": actions,
            "allowed_checks": list(NEXT_ACTION_CHECKS),
        }
        content = validate_next_action_suggestion(
            self.provider.suggest_next_action(context),
            set(actions),
            set(NEXT_ACTION_CHECKS),
        )
        return self._result(content)

    def _result(self, content: dict[str, Any]) -> CopilotResult:
        """Resilient provider의 실제 실행 provider와 fallback 안내를 정규화한다."""

        provider_name = str(
            getattr(self.provider, "last_provider_name", self.provider.provider_name)
        )
        warning = getattr(self.provider, "last_error", None)
        return CopilotResult(
            content=content,
            provider_name=provider_name,
            warning=str(warning) if warning else None,
        )
