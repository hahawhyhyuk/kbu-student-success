"""전체 대시보드의 실제 데이터 기준 주차 조회를 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


DASHBOARD_REFERENCE_WEEK_KEY = "dashboard_reference_week"


@dataclass(frozen=True)
class WeekDataCoverage:
    """선택 주차 기준 학생 데이터의 최신성 집계."""

    reference_week: int
    total_student_count: int
    visible_student_count: int
    current_week_count: int
    carried_forward_count: int
    unavailable_student_count: int


class DashboardPeriodService:
    """실제 존재하는 위험 스냅샷 주차를 기준으로 과거 시점을 조회한다."""

    REQUIRED_COLUMNS = frozenset({"student_id", "week"})

    @classmethod
    def _validated_weeks(cls, snapshots: pd.DataFrame) -> pd.Series:
        """스냅샷의 주차를 양의 정수 Series로 검증해 반환한다."""

        missing = cls.REQUIRED_COLUMNS.difference(snapshots.columns)
        if missing:
            raise ValueError(
                "주차 조회에 필요한 컬럼이 없습니다: "
                + ", ".join(sorted(missing))
            )
        if snapshots.empty:
            raise ValueError("조회할 위험 스냅샷이 없습니다.")
        numeric = pd.to_numeric(snapshots["week"], errors="coerce")
        valid = numeric.notna() & numeric.ge(1) & numeric.mod(1).eq(0)
        if not valid.all():
            raise ValueError("위험 스냅샷의 주차는 1 이상의 정수여야 합니다.")
        return numeric.astype(int)

    def available_weeks(self, snapshots: pd.DataFrame) -> tuple[int, ...]:
        """현재 데이터에 실제 존재하는 주차를 오름차순으로 반환한다."""

        weeks = self._validated_weeks(snapshots)
        return tuple(sorted(set(weeks.tolist())))

    def week_labels(self, snapshots: pd.DataFrame) -> tuple[str, ...]:
        """현재 데이터에 존재하는 주차의 UI 라벨을 반환한다."""

        return tuple(f"{week}주차" for week in self.available_weeks(snapshots))

    def latest_week(self, snapshots: pd.DataFrame) -> int:
        """현재 데이터에서 가장 최신인 주차를 반환한다."""

        return self.available_weeks(snapshots)[-1]

    def resolve_week(self, option: Any, snapshots: pd.DataFrame) -> int:
        """UI 주차 라벨 또는 정수를 실제 존재하는 주차로 검증한다."""

        labels = {
            f"{week}주차": week for week in self.available_weeks(snapshots)
        }
        normalized = str(option).strip()
        if normalized in labels:
            return labels[normalized]
        try:
            week = int(normalized)
        except ValueError as error:
            raise ValueError("지원하지 않는 조회 기준 주차입니다.") from error
        if week not in labels.values():
            raise ValueError("지원하지 않는 조회 기준 주차입니다.")
        return week

    def filter_snapshots_through_week(
        self,
        snapshots: pd.DataFrame,
        reference_week: int,
    ) -> pd.DataFrame:
        """선택 주차까지의 복사본만 반환하고 원본은 변경하지 않는다.

        Parameters:
            snapshots: 학생·주차별 전체 위험 스냅샷.
            reference_week: 조회할 실제 데이터 기준 주차.

        Returns:
            기준 주차 이하의 행만 포함한 새 DataFrame.

        Assumptions:
            특정 주차 자료가 없는 학생은 상위 조회 계층에서 직전 자료를 사용한다.
        """

        week = self.resolve_week(reference_week, snapshots)
        numeric_weeks = self._validated_weeks(snapshots)
        return snapshots.loc[numeric_weeks.le(week)].copy().reset_index(drop=True)

    def coverage(
        self,
        snapshots: pd.DataFrame,
        reference_week: int,
    ) -> WeekDataCoverage:
        """선택 주차 입력·직전 자료 유지·자료 없음 학생 수를 집계한다."""

        week = self.resolve_week(reference_week, snapshots)
        working = snapshots.copy()
        working["student_id"] = working["student_id"].astype(str)
        working["week"] = self._validated_weeks(working)
        total_students = {
            student_id
            for student_id in working["student_id"]
            if student_id.strip()
        }
        visible = working[working["week"].le(week)].copy()
        visible = visible.sort_values(["student_id", "week"], kind="stable")
        latest = visible.groupby("student_id", as_index=False).tail(1)
        current_week_count = int(latest["week"].eq(week).sum())
        carried_forward_count = int(latest["week"].lt(week).sum())
        visible_students = set(latest["student_id"])
        return WeekDataCoverage(
            reference_week=week,
            total_student_count=len(total_students),
            visible_student_count=len(visible_students),
            current_week_count=current_week_count,
            carried_forward_count=carried_forward_count,
            unavailable_student_count=len(total_students - visible_students),
        )
