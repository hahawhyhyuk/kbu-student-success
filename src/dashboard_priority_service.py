"""최신 위험 스냅샷과 개입 상태를 결합해 교직원 우선확인 대상을 찾는다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.intervention_service import InterventionManagementService
from src.risk_service import select_latest_student_snapshots
from src.utils import load_app_config


class DashboardPriorityService:
    """아직 실제 지원이 시작되지 않은 고위험 학생을 선별한다."""

    REQUIRED_SNAPSHOT_COLUMNS = {
        "student_id",
        "week",
        "department",
        "overall_risk",
        "risk_level",
        "primary_risk_type",
        "risk_change",
    }

    def __init__(
        self,
        database_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        intervention_service: InterventionManagementService | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_app_config()
        dashboard_config = self.config["dashboard"]
        self.unaddressed_statuses = frozenset(
            str(item)
            for item in dashboard_config["unaddressed_intervention_statuses"]
        )
        self.display_limit = int(dashboard_config["unaddressed_display_limit"])
        if self.display_limit < 1:
            raise ValueError("미개입 학생 표시 수는 1 이상이어야 합니다.")
        self.intervention_service = intervention_service or (
            InterventionManagementService(database_path=database_path)
        )

    def find_unaddressed_high_risk(
        self,
        snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        """최신 상태가 지원 시작 전인 고위험 학생만 반환한다.

        Parameters:
            snapshots: 학생·주차별 위험 스냅샷. 여러 주차가 포함되어도 된다.

        Returns:
            위험도 내림차순의 미개입 고위험 학생 DataFrame.

        Assumptions:
            개입 이력이 없거나 설정된 지원 시작 전 상태이면 미개입으로 본다.
        """

        missing = self.REQUIRED_SNAPSHOT_COLUMNS.difference(snapshots.columns)
        if missing:
            raise ValueError(
                f"우선확인 분석에 필요한 컬럼이 없습니다: {sorted(missing)}"
            )
        latest = select_latest_student_snapshots(snapshots)
        high_risk = latest[latest["risk_level"] == "고위험"].copy()
        if high_risk.empty:
            return self._empty_result()

        interventions = self.intervention_service.list_latest_interventions()
        if interventions.empty:
            high_risk["intervention_status"] = "개입 미등록"
            high_risk["intervention_updated_at"] = ""
        else:
            intervention_state = interventions[
                ["student_id", "status", "updated_at"]
            ].rename(
                columns={
                    "status": "intervention_status",
                    "updated_at": "intervention_updated_at",
                }
            )
            high_risk = high_risk.merge(
                intervention_state,
                on="student_id",
                how="left",
                validate="one_to_one",
            )
            high_risk["intervention_status"] = high_risk[
                "intervention_status"
            ].fillna("개입 미등록")
            high_risk["intervention_updated_at"] = high_risk[
                "intervention_updated_at"
            ].fillna("")

        unaddressed = high_risk[
            (high_risk["intervention_status"] == "개입 미등록")
            | high_risk["intervention_status"].isin(self.unaddressed_statuses)
        ].copy()
        unaddressed["attention_reason"] = unaddressed[
            "intervention_status"
        ].map(
            lambda status: (
                "개입 이력 없음"
                if status == "개입 미등록"
                else "지원 시작 전 상태"
            )
        )
        return (
            unaddressed.sort_values(
                ["overall_risk", "risk_change", "student_id"],
                ascending=[False, False, True],
            )
            .head(self.display_limit)
            .reset_index(drop=True)
        )

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """UI가 컬럼 존재 여부를 분기하지 않도록 빈 결과 스키마를 반환한다."""

        return pd.DataFrame(
            columns=[
                "student_id",
                "department",
                "overall_risk",
                "risk_level",
                "primary_risk_type",
                "risk_change",
                "intervention_status",
                "intervention_updated_at",
                "attention_reason",
            ]
        )
