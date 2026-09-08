"""고위험 학생과 최신 개입 상태를 결합하는 대시보드 우선순위 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.dashboard_priority_service import DashboardPriorityService
from src.intervention_service import InterventionManagementService


def _snapshot(
    student_id: str,
    overall_risk: float,
    risk_level: str = "고위험",
    week: int = 4,
    risk_change: float = 0.0,
) -> dict[str, object]:
    """우선확인 서비스에 필요한 최소 위험 스냅샷을 만든다."""

    return {
        "student_id": student_id,
        "week": week,
        "department": "DEPT01",
        "overall_risk": overall_risk,
        "risk_level": risk_level,
        "primary_risk_type": "전공부적응형",
        "risk_change": risk_change,
    }


def test_high_risk_students_without_intervention_are_prioritized(
    tmp_path: Path,
) -> None:
    service = DashboardPriorityService(database_path=tmp_path / "app.db")
    snapshots = pd.DataFrame(
        [
            _snapshot("S0001", 72, risk_change=3),
            _snapshot("S0002", 86, risk_change=1),
            _snapshot("S0003", 60, risk_level="주의", risk_change=8),
        ]
    )

    result = service.find_unaddressed_high_risk(snapshots)

    assert result["student_id"].tolist() == ["S0002", "S0001"]
    assert set(result["intervention_status"]) == {"개입 미등록"}
    assert set(result["attention_reason"]) == {"개입 이력 없음"}


def test_support_start_status_excludes_student_but_prestart_status_keeps_them(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    intervention_service = InterventionManagementService(
        database_path=database_path
    )
    prestart_id = intervention_service.store.create_intervention(
        "S0010", "BATCH-10", ["P001"]
    )
    started_id = intervention_service.store.create_intervention(
        "S0020", "BATCH-20", ["P001"]
    )
    intervention_service.update_status(started_id, "상담 예정")
    deferred_id = intervention_service.store.create_intervention(
        "S0030", "BATCH-30", ["P001"]
    )
    intervention_service.update_status(deferred_id, "추후 관찰")
    service = DashboardPriorityService(
        intervention_service=intervention_service
    )
    snapshots = pd.DataFrame(
        [
            _snapshot("S0010", 80),
            _snapshot("S0020", 90),
            _snapshot("S0030", 88),
            _snapshot("S0040", 75),
        ]
    )

    result = service.find_unaddressed_high_risk(snapshots)

    assert prestart_id > 0
    assert result["student_id"].tolist() == ["S0010", "S0040"]
    status_by_student = result.set_index("student_id")[
        "intervention_status"
    ].to_dict()
    assert status_by_student == {
        "S0010": "추천 생성",
        "S0040": "개입 미등록",
    }


def test_latest_week_and_configured_display_limit_are_applied(
    tmp_path: Path,
) -> None:
    config = {
        "dashboard": {
            "unaddressed_intervention_statuses": ["추천 생성"],
            "unaddressed_display_limit": 1,
        },
        "intervention_management": {
            "statuses": ["추천 생성"],
            "feedback_types": ["관심 있음"],
        },
    }
    service = DashboardPriorityService(
        database_path=tmp_path / "app.db",
        config=config,
    )
    snapshots = pd.DataFrame(
        [
            _snapshot("S0100", 90, week=3),
            _snapshot("S0100", 20, risk_level="정상", week=4),
            _snapshot("S0200", 82, week=4),
            _snapshot("S0300", 88, week=4),
        ]
    )

    result = service.find_unaddressed_high_risk(snapshots)

    assert result["student_id"].tolist() == ["S0300"]


def test_priority_service_rejects_missing_snapshot_columns(
    tmp_path: Path,
) -> None:
    service = DashboardPriorityService(database_path=tmp_path / "app.db")

    with pytest.raises(ValueError, match="필요한 컬럼"):
        service.find_unaddressed_high_risk(
            pd.DataFrame([{"student_id": "S0001"}])
        )
