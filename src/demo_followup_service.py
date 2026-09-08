"""지원 완료 뒤에만 대표 학생의 synthetic 15주차 관찰을 연결한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data_generator import (
    DATA_DIR,
    DEMO_FOLLOWUP_COLUMNS,
    generate_demo_followup_observations,
)
from src.repositories import StudentSuccessRepository
from src.risk_engine import build_risk_snapshots
from src.utils import load_app_config


FOLLOWUP_ACTIVITY_COLUMNS = (
    "student_id",
    "week",
    "attendance_rate",
    "absence_count",
    "consecutive_absence",
    "late_count",
    "assignment_submission_rate",
    "overdue_assignment_count",
    "lms_login_days",
    "lms_activity_change",
    "video_completion_rate",
    "quiz_score",
)
FOLLOWUP_CHECKIN_COLUMNS = (
    "student_id",
    "week",
    "major_interest",
    "major_satisfaction",
    "major_continuation_intent",
    "career_clarity",
    "learning_difficulty",
    "consultation_intent",
    "interest_fields",
    "desired_job",
    "natural_language_concern",
)
FOLLOWUP_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "attendance_rate": (0, 100),
    "absence_count": (0, 15),
    "consecutive_absence": (0, 15),
    "late_count": (0, 15),
    "assignment_submission_rate": (0, 100),
    "overdue_assignment_count": (0, 15),
    "lms_login_days": (0, 7),
    "lms_activity_change": (-100, 100),
    "video_completion_rate": (0, 100),
    "quiz_score": (0, 100),
    "major_interest": (1, 5),
    "major_satisfaction": (1, 5),
    "major_continuation_intent": (1, 5),
    "career_clarity": (1, 5),
    "learning_difficulty": (1, 5),
    "consultation_intent": (1, 5),
}


@dataclass(frozen=True)
class DemoFollowupApplication:
    """일반 최신 스냅샷과 조건부 synthetic 후속 관찰의 결합 결과."""

    snapshots: pd.DataFrame
    eligible_student_ids: tuple[str, ...]
    applied_student_ids: tuple[str, ...]
    skipped_existing_student_ids: tuple[str, ...]


class DemoFollowupService:
    """실제 후속값을 덮지 않고 시연용 15주차 관찰만 조건부 적용한다."""

    def __init__(
        self,
        repository: StudentSuccessRepository,
        data_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        observations: pd.DataFrame | None = None,
    ) -> None:
        self.repository = repository
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.config = dict(config) if config is not None else load_app_config()
        demo_config = self.config["demo"]
        followup_config = demo_config["followup"]
        self.enabled = bool(demo_config["enabled"]) and bool(
            followup_config["enabled"]
        )
        self.representative_student_id = str(
            demo_config["representative_student_id"]
        ).strip()
        self.baseline_week = int(demo_config["baseline_week"])
        self.followup_week = int(followup_config["week"])
        self.followup_file = str(followup_config["file"]).strip()
        self.eligible_statuses = frozenset(
            str(status).strip()
            for status in followup_config["eligible_intervention_statuses"]
            if str(status).strip()
        )
        self._observations = observations.copy() if observations is not None else None
        if not self.representative_student_id:
            raise ValueError("시연 대표 학생 ID는 비어 있을 수 없습니다.")
        if self.followup_week <= self.baseline_week:
            raise ValueError("시연 후속 주차는 지원 시작 주차보다 늦어야 합니다.")
        if not self.followup_file or Path(self.followup_file).name != self.followup_file:
            raise ValueError("시연 후속 파일은 data 디렉터리의 파일명만 허용합니다.")
        if not self.eligible_statuses:
            raise ValueError("시연 후속 관찰을 적용할 학생지원 상태가 필요합니다.")

    def load_observations(self) -> pd.DataFrame:
        """검증할 synthetic 후속 원천을 파일 또는 고정 생성값에서 읽는다."""

        if self._observations is not None:
            observations = self._observations.copy()
        else:
            path = self.data_dir / self.followup_file
            observations = (
                pd.read_csv(path, keep_default_na=False)
                if path.exists()
                else generate_demo_followup_observations()
            )
        missing = set(DEMO_FOLLOWUP_COLUMNS).difference(observations.columns)
        if missing:
            raise ValueError(
                f"시연 후속 관찰에 필수 열이 없습니다: {sorted(missing)}"
            )
        observations = observations.loc[:, DEMO_FOLLOWUP_COLUMNS].copy()
        observations["student_id"] = observations["student_id"].astype(str).str.strip()
        observations["week"] = pd.to_numeric(
            observations["week"], errors="coerce"
        )
        if observations.empty:
            raise ValueError("시연 후속 관찰이 비어 있습니다.")
        if observations[["student_id", "week"]].isna().any(axis=1).any():
            raise ValueError("시연 후속 관찰의 학생 ID 또는 주차가 비어 있습니다.")
        if observations[["student_id", "week"]].duplicated().any():
            raise ValueError("시연 후속 관찰의 학생·주차는 중복될 수 없습니다.")
        if set(observations["student_id"]) != {self.representative_student_id}:
            raise ValueError("시연 후속 관찰은 설정된 대표 synthetic 학생만 허용합니다.")
        if not observations["week"].eq(self.followup_week).all():
            raise ValueError("시연 후속 관찰 주차가 설정과 일치하지 않습니다.")
        observations["week"] = observations["week"].astype(int)
        for column, (minimum, maximum) in FOLLOWUP_NUMERIC_BOUNDS.items():
            numeric = pd.to_numeric(observations[column], errors="coerce")
            if numeric.isna().any() or not numeric.between(minimum, maximum).all():
                raise ValueError(
                    f"시연 후속 관찰의 {column} 값은 {minimum}~{maximum} 범위여야 합니다."
                )
            observations[column] = numeric
        required_text = ("natural_language_concern", "observation_note")
        if observations[list(required_text)].fillna("").astype(str).apply(
            lambda column: column.str.strip().eq("")
        ).any(axis=None):
            raise ValueError("시연 후속 관찰의 설명과 출처 표시는 비어 있을 수 없습니다.")
        return observations.reset_index(drop=True)

    def build_followup_snapshots(self) -> pd.DataFrame:
        """후속 원천지표를 기존 규칙 엔진으로 계산한 15주차 스냅샷을 만든다."""

        source = self.repository.get_all()
        observations = self.load_observations()
        students = source.students[
            source.students["student_id"].astype(str).isin(
                observations["student_id"]
            )
        ].copy()
        missing_students = set(observations["student_id"]).difference(
            students["student_id"].astype(str)
        )
        if missing_students:
            raise ValueError(
                "시연 후속 관찰의 학생이 synthetic master에 없습니다: "
                + ", ".join(sorted(missing_students))
            )
        snapshots = build_risk_snapshots(
            students,
            observations.loc[:, FOLLOWUP_ACTIVITY_COLUMNS],
            observations.loc[:, FOLLOWUP_CHECKIN_COLUMNS],
        )
        snapshots["checkin_source"] = "synthetic_demo_followup"
        snapshots["student_checkin_id"] = None
        return snapshots.reset_index(drop=True)

    def apply(
        self,
        current_snapshots: pd.DataFrame,
        interventions: pd.DataFrame,
    ) -> DemoFollowupApplication:
        """최신 지원 상태가 완료 조건인 학생에게만 후속 스냅샷을 추가한다.

        Parameters:
            current_snapshots: Repository와 실제 학생 제출을 반영한 현재 위험 이력.
            interventions: SQLite의 전체 학생지원 기록.

        Returns:
            조건부 후속값을 합친 복사본과 적용·건너뜀 학생 ID.

        Assumptions:
            동일 학생에게 15주차 이상의 실제 관찰이 있으면 synthetic 값을 적용하지 않는다.
        """

        original = current_snapshots.copy()
        if not self.enabled or interventions.empty:
            return DemoFollowupApplication(original, (), (), ())
        required_intervention_columns = {
            "intervention_id",
            "student_id",
            "status",
            "updated_at",
        }
        missing = required_intervention_columns.difference(interventions.columns)
        if missing:
            raise ValueError(
                f"시연 후속 연결에 필요한 지원 열이 없습니다: {sorted(missing)}"
            )
        if {"student_id", "week"}.difference(original.columns):
            raise ValueError("시연 후속 연결에 학생 ID와 주차가 필요합니다.")

        latest = interventions.copy()
        latest["student_id"] = latest["student_id"].astype(str)
        latest = latest.sort_values(
            ["student_id", "updated_at", "intervention_id"], kind="stable"
        ).drop_duplicates("student_id", keep="last")
        eligible_student_ids = tuple(
            sorted(
                latest.loc[
                    latest["status"].astype(str).isin(self.eligible_statuses),
                    "student_id",
                ].astype(str)
            )
        )
        if not eligible_student_ids:
            return DemoFollowupApplication(original, (), (), ())

        followups = self.build_followup_snapshots()
        existing_max_week = (
            original.assign(
                student_id=original["student_id"].astype(str),
                week=pd.to_numeric(original["week"], errors="coerce"),
            )
            .groupby("student_id")["week"]
            .max()
            .to_dict()
        )
        applied: list[str] = []
        skipped_existing: list[str] = []
        selected_rows: list[pd.Series] = []
        eligible_set = set(eligible_student_ids)
        for _, row in followups.iterrows():
            student_id = str(row["student_id"])
            if student_id not in eligible_set:
                continue
            if float(existing_max_week.get(student_id, 0) or 0) >= int(row["week"]):
                skipped_existing.append(student_id)
                continue
            applied.append(student_id)
            selected_rows.append(row)
        if not selected_rows:
            return DemoFollowupApplication(
                original,
                eligible_student_ids,
                (),
                tuple(sorted(set(skipped_existing))),
            )

        selected = pd.DataFrame(selected_rows)
        for provenance_column in ("checkin_source", "student_checkin_id"):
            if provenance_column not in original.columns:
                original[provenance_column] = None
        for column in original.columns:
            if column not in selected.columns:
                selected[column] = None
        selected = selected.loc[:, original.columns]
        combined = pd.concat([original, selected], ignore_index=True)
        combined = combined.sort_values(
            ["student_id", "week"], kind="stable"
        ).reset_index(drop=True)
        return DemoFollowupApplication(
            combined,
            eligible_student_ids,
            tuple(sorted(set(applied))),
            tuple(sorted(set(skipped_existing))),
        )
