"""지원 완료 뒤 synthetic 15주차 관찰을 연결하는 서비스 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_generator import generate_demo_followup_observations
from src.demo_followup_service import DemoFollowupService
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService


def _interventions(status: str) -> pd.DataFrame:
    """대표 학생의 최신 지원 상태 한 건을 만든다."""

    return pd.DataFrame(
        [
            {
                "intervention_id": 1,
                "student_id": "S0003",
                "status": status,
                "updated_at": "2026-09-08T10:00:00",
            }
        ]
    )


def test_followup_is_applied_only_after_support_completion_status() -> None:
    """검토 대기에는 4주차를 유지하고 상담 완료 뒤에만 15주차를 추가한다."""

    repository = get_default_repository()
    baseline = RiskAnalysisService(repository).build_snapshots()
    service = DemoFollowupService(
        repository,
        observations=generate_demo_followup_observations(),
    )

    waiting = service.apply(baseline, _interventions("교직원 검토 대기"))
    completed = service.apply(baseline, _interventions("상담 완료"))

    waiting_student = waiting.snapshots[waiting.snapshots["student_id"] == "S0003"]
    completed_student = completed.snapshots[
        completed.snapshots["student_id"] == "S0003"
    ].sort_values("week")
    followup = completed_student.iloc[-1]

    assert waiting.applied_student_ids == ()
    assert int(waiting_student["week"].max()) == 4
    assert completed.applied_student_ids == ("S0003",)
    assert int(followup["week"]) == 15
    assert followup["checkin_source"] == "synthetic_demo_followup"
    assert float(followup["overall_risk"]) == pytest.approx(25.95)
    assert followup["risk_level"] == "관심"
    assert len(completed.snapshots) == len(baseline) + 1


def test_existing_real_followup_is_never_replaced_by_synthetic_preset() -> None:
    """동일·이후 주차가 이미 있으면 시연용 관찰을 중복 추가하지 않는다."""

    repository = get_default_repository()
    baseline = RiskAnalysisService(repository).build_snapshots()
    service = DemoFollowupService(
        repository,
        observations=generate_demo_followup_observations(),
    )
    actual_followup = service.build_followup_snapshots().copy()
    actual_followup["week"] = 16
    actual_followup["checkin_source"] = "student_submission"
    current = pd.concat([baseline, actual_followup], ignore_index=True)

    result = service.apply(current, _interventions("프로그램 참여"))

    student = result.snapshots[result.snapshots["student_id"] == "S0003"]
    assert result.applied_student_ids == ()
    assert result.skipped_existing_student_ids == ("S0003",)
    assert student["week"].astype(int).tolist().count(15) == 0
    assert int(student["week"].max()) == 16
    assert len(result.snapshots) == len(current)


def test_followup_rejects_invalid_range_or_wrong_student() -> None:
    """고정 시연값도 범위와 대표 synthetic 학생을 벗어나면 사용하지 않는다."""

    repository = get_default_repository()
    invalid_score = generate_demo_followup_observations()
    invalid_score.loc[0, "attendance_rate"] = 120
    wrong_student = generate_demo_followup_observations()
    wrong_student.loc[0, "student_id"] = "S0004"

    with pytest.raises(ValueError, match="attendance_rate"):
        DemoFollowupService(
            repository,
            observations=invalid_score,
        ).load_observations()
    with pytest.raises(ValueError, match="대표 synthetic 학생"):
        DemoFollowupService(
            repository,
            observations=wrong_student,
        ).load_observations()
