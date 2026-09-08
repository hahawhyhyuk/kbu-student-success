"""개입 생성 시점과 현재 위험도를 비교해 후속 관찰 지표를 만든다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.intervention_service import InterventionManagementService
from src.utils import load_app_config


DOMAIN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("attendance_risk", "출결"),
    ("engagement_risk", "학습참여"),
    ("achievement_risk", "학업성취"),
    ("major_adaptation_risk", "전공적응"),
    ("career_risk", "진로설계"),
)


@dataclass(frozen=True)
class InterventionEffectAnalysis:
    """개입 스냅샷 비교 결과와 제외 현황을 함께 제공한다."""

    records: pd.DataFrame
    total_interventions: int
    legacy_without_snapshot: int
    invalid_snapshot_links: int
    missing_current_students: int


class InterventionEffectService:
    """저장된 개입 시점 스냅샷과 최신 위험 결과를 개입 건별로 비교한다."""

    REQUIRED_CURRENT_COLUMNS = {
        "student_id",
        "week",
        "department",
        "overall_risk",
        "risk_level",
        "primary_risk_type",
        *(column for column, _ in DOMAIN_COLUMNS),
    }

    def __init__(
        self,
        database_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        intervention_service: InterventionManagementService | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_app_config()
        effect_config = self.config["intervention_effect"]
        self.meaningful_change_threshold = float(
            effect_config["meaningful_change_threshold"]
        )
        self.score_change_epsilon = float(effect_config["score_change_epsilon"])
        if self.meaningful_change_threshold <= 0:
            raise ValueError("의미 있는 변화 기준은 0보다 커야 합니다.")
        if self.score_change_epsilon < 0:
            raise ValueError("점수 변화 오차 기준은 0 이상이어야 합니다.")
        self.intervention_service = intervention_service or (
            InterventionManagementService(database_path=database_path)
        )

    @staticmethod
    def _empty_records() -> pd.DataFrame:
        """스냅샷이 없어도 UI가 같은 컬럼 계약을 사용하게 한다."""

        columns = [
            "intervention_id",
            "student_id",
            "department",
            "status",
            "intervention_created_at",
            "intervention_updated_at",
            "baseline_week",
            "current_week",
            "baseline_overall_risk",
            "current_overall_risk",
            "overall_change",
            "baseline_risk_level",
            "current_risk_level",
            "baseline_primary_risk_type",
            "current_primary_risk_type",
            "effect_status",
            "has_followup_data",
        ]
        for column, _ in DOMAIN_COLUMNS:
            columns.extend(
                [f"baseline_{column}", f"current_{column}", f"{column}_change"]
            )
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _normalize_optional_identifier(value: Any) -> int | None:
        """SQLite·pandas의 nullable ID를 비교 가능한 값으로 바꾼다."""

        if value is None or bool(pd.isna(value)):
            return None
        return int(value)

    def analyze(self, current_snapshots: pd.DataFrame) -> InterventionEffectAnalysis:
        """개입 시점과 학생별 최신 위험 결과를 비교한다.

        Parameters:
            current_snapshots: Repository와 학생 최신 체크인을 반영한
                학생·주차별 위험 스냅샷.

        Returns:
            평가 가능 여부·점수 변화·제외 건수를 포함한 분석 결과.

        Assumptions:
            변화는 후속 관찰 지표이며 개입의 인과효과를 입증하지 않는다.
        """

        missing = self.REQUIRED_CURRENT_COLUMNS.difference(
            current_snapshots.columns
        )
        if missing:
            raise ValueError(
                f"개입효과 비교에 필요한 컬럼이 없습니다: {sorted(missing)}"
            )

        interventions = self.intervention_service.list_interventions()
        total_interventions = len(interventions)
        if interventions.empty:
            return InterventionEffectAnalysis(
                self._empty_records(), 0, 0, 0, 0
            )

        linked = interventions[
            interventions["risk_snapshot_id"].notna()
        ].copy()
        legacy_without_snapshot = total_interventions - len(linked)
        if linked.empty:
            return InterventionEffectAnalysis(
                self._empty_records(),
                total_interventions,
                legacy_without_snapshot,
                0,
                0,
            )

        stored_snapshots = self.intervention_service.list_risk_snapshots()
        if stored_snapshots.empty:
            return InterventionEffectAnalysis(
                self._empty_records(),
                total_interventions,
                legacy_without_snapshot,
                len(linked),
                0,
            )

        baseline_columns = [
            "snapshot_id",
            "student_id",
            "week",
            "overall_risk",
            "risk_level",
            "primary_risk_type",
            "student_checkin_id",
            *(column for column, _ in DOMAIN_COLUMNS),
        ]
        baselines = stored_snapshots[baseline_columns].rename(
            columns={
                "student_id": "snapshot_student_id",
                **{
                    column: f"baseline_{column}"
                    for column in baseline_columns
                    if column not in {"snapshot_id", "student_id"}
                },
            }
        )
        linked["risk_snapshot_id"] = linked["risk_snapshot_id"].astype(int)
        events = linked.merge(
            baselines,
            left_on="risk_snapshot_id",
            right_on="snapshot_id",
            how="left",
            validate="many_to_one",
        )
        valid_snapshot = events["snapshot_id"].notna() & (
            events["student_id"].astype(str)
            == events["snapshot_student_id"].astype(str)
        )
        invalid_snapshot_links = int((~valid_snapshot).sum())
        events = events[valid_snapshot].copy()
        if events.empty:
            return InterventionEffectAnalysis(
                self._empty_records(),
                total_interventions,
                legacy_without_snapshot,
                invalid_snapshot_links,
                0,
            )

        latest = (
            current_snapshots.sort_values(["student_id", "week"])
            .groupby("student_id", as_index=False)
            .tail(1)
            .copy()
        )
        if "student_checkin_id" not in latest.columns:
            latest["student_checkin_id"] = None
        current_columns = [
            "student_id",
            "department",
            "week",
            "overall_risk",
            "risk_level",
            "primary_risk_type",
            "student_checkin_id",
            *(column for column, _ in DOMAIN_COLUMNS),
        ]
        current = latest[current_columns].rename(
            columns={
                column: f"current_{column}"
                for column in current_columns
                if column != "student_id"
            }
        )
        compared = events.merge(
            current,
            on="student_id",
            how="left",
            validate="many_to_one",
        )
        has_current = compared["current_week"].notna()
        missing_current_students = int((~has_current).sum())
        compared = compared[has_current].copy()
        if compared.empty:
            return InterventionEffectAnalysis(
                self._empty_records(),
                total_interventions,
                legacy_without_snapshot,
                invalid_snapshot_links,
                missing_current_students,
            )

        for column in ("overall_risk", *(item[0] for item in DOMAIN_COLUMNS)):
            compared[f"{column}_change"] = (
                compared[f"current_{column}"].astype(float)
                - compared[f"baseline_{column}"].astype(float)
            ).round(2)
        compared = compared.rename(
            columns={
                "overall_risk_change": "overall_change",
                "created_at": "intervention_created_at",
                "updated_at": "intervention_updated_at",
                "current_department": "department",
            }
        )

        domain_change_columns = [
            f"{column}_change" for column, _ in DOMAIN_COLUMNS
        ]
        score_changed = compared[domain_change_columns + ["overall_change"]].abs().max(
            axis=1
        ) > self.score_change_epsilon
        checkin_changed = compared.apply(
            lambda row: (
                self._normalize_optional_identifier(
                    row["current_student_checkin_id"]
                )
                is not None
                and self._normalize_optional_identifier(
                    row["current_student_checkin_id"]
                )
                != self._normalize_optional_identifier(
                    row["baseline_student_checkin_id"]
                )
            ),
            axis=1,
        )
        compared["has_followup_data"] = (
            compared["current_week"].astype(int)
            > compared["baseline_week"].astype(int)
        ) | checkin_changed | score_changed

        threshold = self.meaningful_change_threshold

        def classify(row: pd.Series) -> str:
            if not bool(row["has_followup_data"]):
                return "후속 데이터 대기"
            change = float(row["overall_change"])
            if change <= -threshold:
                return "개선"
            if change >= threshold:
                return "악화"
            return "유지"

        compared["effect_status"] = compared.apply(classify, axis=1)
        compared["baseline_week"] = compared["baseline_week"].astype(int)
        compared["current_week"] = compared["current_week"].astype(int)
        compared["intervention_id"] = compared["intervention_id"].astype(int)
        compared = compared.sort_values(
            ["intervention_updated_at", "intervention_id"],
            ascending=[False, False],
        ).reset_index(drop=True)

        result_columns = list(self._empty_records().columns)
        return InterventionEffectAnalysis(
            records=compared[result_columns],
            total_interventions=total_interventions,
            legacy_without_snapshot=legacy_without_snapshot,
            invalid_snapshot_links=invalid_snapshot_links,
            missing_current_students=missing_current_students,
        )
