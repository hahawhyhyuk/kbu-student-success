"""Repository 데이터와 순수 위험엔진을 연결하는 애플리케이션 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.database import StudentCheckinDataStore
from src.repositories import StudentSuccessData, StudentSuccessRepository
from src.repositories import get_default_repository
from src.risk_engine import build_risk_snapshots


def select_latest_student_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """학생마다 보유한 가장 최신 주차의 위험 스냅샷을 한 행씩 선택한다.

    Parameters:
        snapshots: 학생·주차별 위험 스냅샷 DataFrame.

    Returns:
        학생별 최신 행만 남기고 인덱스를 재정렬한 DataFrame.

    Assumptions:
        학생별 관찰 주차는 서로 다를 수 있으며 전체 최대 주차를 모든 학생에게 강제하지 않는다.
    """

    required_columns = {"student_id", "week"}
    missing = required_columns.difference(snapshots.columns)
    if missing:
        raise ValueError(
            f"최신 위험 스냅샷 선택에 필요한 컬럼이 없습니다: {sorted(missing)}"
        )
    if snapshots.empty:
        return snapshots.copy().reset_index(drop=True)
    return (
        snapshots.sort_values(["student_id", "week"], kind="stable")
        .drop_duplicates("student_id", keep="last")
        .reset_index(drop=True)
    )


@dataclass
class RiskAnalysisService:
    """저장방식과 무관하게 화면용 위험분석 데이터를 제공한다."""

    repository: StudentSuccessRepository
    config: Mapping[str, Any] | None = None
    checkin_store: StudentCheckinDataStore | None = None

    def load_source_data(self) -> StudentSuccessData:
        """Repository에서 canonical 원천데이터를 로드하고 검증한다."""

        return self.repository.get_all()

    def build_snapshots(
        self, data: StudentSuccessData | None = None
    ) -> pd.DataFrame:
        """Repository 데이터로 학생·주차별 위험 스냅샷을 계산한다."""

        source = data or self.load_source_data()
        snapshots = build_risk_snapshots(
            source.students,
            source.weekly_activity,
            source.checkins,
            self.config,
        )
        if self.checkin_store is None:
            return snapshots
        return self._apply_student_checkins(snapshots, source)

    def _apply_student_checkins(
        self,
        snapshots: pd.DataFrame,
        source: StudentSuccessData,
    ) -> pd.DataFrame:
        """학생 직접 제출 체크인을 학생·주차별 위험 이력에 반영한다.

        Parameters:
            snapshots: Repository 체크인으로 계산한 전체 주차 위험 스냅샷.
            source: 학생·활동·체크인 원천데이터 묶음.

        Returns:
            Repository 주차와 직접 제출 주차를 합친 스냅샷.

        Assumptions:
            활동 데이터보다 늦은 주차의 체크인은 가장 최근 출결·LMS·성취 값을
            유지하고 새 자기보고 값만 반영한다.
        """

        result = snapshots.copy()
        result["checkin_source"] = "repository"
        result["student_checkin_id"] = pd.Series(
            [None] * len(result), dtype="object"
        )
        submitted = self.checkin_store.list_checkins()
        if submitted.empty:
            return result

        known_student_ids = set(source.students["student_id"].astype(str))
        submitted = submitted[
            submitted["student_id"].astype(str).isin(known_student_ids)
        ].copy()
        if submitted.empty:
            return result
        submitted["student_id"] = submitted["student_id"].astype(str)
        canonical_checkin_columns = list(source.checkins.columns)
        students = source.students.copy()
        students["student_id"] = students["student_id"].astype(str)
        activities = source.weekly_activity.copy()
        activities["student_id"] = activities["student_id"].astype(str)

        for submitted_row in submitted.to_dict(orient="records"):
            student_id = str(submitted_row["student_id"])
            week = int(submitted_row["week"])
            student_row = students[students["student_id"] == student_id]
            student_activities = activities[
                activities["student_id"] == student_id
            ].sort_values("week")
            if student_row.empty or student_activities.empty:
                continue
            eligible_activity = student_activities[
                student_activities["week"].astype(int) <= week
            ]
            activity_row = (
                eligible_activity.iloc[[-1]].copy()
                if not eligible_activity.empty
                else student_activities.iloc[[0]].copy()
            )
            activity_row["week"] = week
            checkin_row = pd.DataFrame(
                [
                    {
                        column: submitted_row.get(column)
                        for column in canonical_checkin_columns
                    }
                ]
            )
            checkin_row["week"] = week
            updated = build_risk_snapshots(
                student_row,
                activity_row,
                checkin_row,
                self.config,
            )
            updated["checkin_source"] = "student_submission"
            updated["student_checkin_id"] = int(submitted_row["checkin_id"])
            same_observation = (
                (result["student_id"].astype(str) == student_id)
                & (result["week"].astype(int) == week)
            )
            result = result[~same_observation]
            result = pd.concat([result, updated[result.columns]], ignore_index=True)

        result = result.sort_values(["student_id", "week"]).reset_index(drop=True)
        result["risk_change"] = (
            result.groupby("student_id")["overall_risk"]
            .diff()
            .fillna(0)
            .round(2)
        )
        return result

    def load_students_and_snapshots(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """상세 화면용 학생정보와 위험 스냅샷을 한 번의 Repository 로드로 반환한다."""

        source = self.load_source_data()
        return source.students, self.build_snapshots(source)


def create_integrated_risk_service(
    repository: StudentSuccessRepository | None = None,
    database_path: str | Path | None = None,
) -> RiskAnalysisService:
    """Repository 원천데이터와 SQLite 최신 체크인을 결합하는 기본 서비스를 만든다."""

    return RiskAnalysisService(
        repository=repository or get_default_repository(),
        checkin_store=StudentCheckinDataStore(database_path),
    )


def get_student_checkin_revision(
    database_path: str | Path | None = None,
) -> int:
    """Streamlit 캐시 키에 사용할 최신 학생 체크인 버전을 반환한다."""

    return StudentCheckinDataStore(database_path).get_revision()
