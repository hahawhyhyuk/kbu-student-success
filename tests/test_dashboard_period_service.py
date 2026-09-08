"""전체 대시보드의 실제 데이터 기준 주차 조회 테스트."""

import pandas as pd
import pytest

from src.dashboard_period_service import DashboardPeriodService


def _snapshots() -> pd.DataFrame:
    """주차 누락과 늦은 첫 입력을 포함한 조회 테스트 데이터를 반환한다."""

    return pd.DataFrame(
        [
            {"student_id": "S001", "week": 1, "overall_risk": 10},
            {"student_id": "S001", "week": 2, "overall_risk": 20},
            {"student_id": "S001", "week": 4, "overall_risk": 40},
            {"student_id": "S002", "week": 1, "overall_risk": 15},
            {"student_id": "S002", "week": 2, "overall_risk": 25},
            {"student_id": "S003", "week": 4, "overall_risk": 30},
        ]
    )


def test_period_service_uses_weeks_that_exist_in_data() -> None:
    service = DashboardPeriodService()
    snapshots = _snapshots()

    assert service.available_weeks(snapshots) == (1, 2, 4)
    assert service.week_labels(snapshots) == ("1주차", "2주차", "4주차")
    assert service.latest_week(snapshots) == 4
    assert service.resolve_week("2주차", snapshots) == 2


def test_period_filter_is_as_of_week_and_does_not_change_source() -> None:
    service = DashboardPeriodService()
    snapshots = _snapshots()
    original = snapshots.copy(deep=True)

    filtered = service.filter_snapshots_through_week(snapshots, 2)

    assert filtered["week"].max() == 2
    assert set(filtered["student_id"]) == {"S001", "S002"}
    assert snapshots.equals(original)


def test_period_coverage_distinguishes_current_carried_and_unavailable() -> None:
    service = DashboardPeriodService()

    coverage = service.coverage(_snapshots(), 4)

    assert coverage.total_student_count == 3
    assert coverage.visible_student_count == 3
    assert coverage.current_week_count == 2
    assert coverage.carried_forward_count == 1
    assert coverage.unavailable_student_count == 0

    week_two = service.coverage(_snapshots(), 2)
    assert week_two.current_week_count == 2
    assert week_two.carried_forward_count == 0
    assert week_two.unavailable_student_count == 1


def test_period_service_rejects_missing_or_invalid_weeks() -> None:
    service = DashboardPeriodService()

    with pytest.raises(ValueError, match="지원하지 않는"):
        service.resolve_week("3주차", _snapshots())
    with pytest.raises(ValueError, match="1 이상의 정수"):
        service.available_weeks(
            pd.DataFrame(
                [{"student_id": "S001", "week": "미입력"}]
            )
        )
