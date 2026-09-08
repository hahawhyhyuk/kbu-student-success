"""재현 가능한 해커톤 시연 시나리오와 초기화 안전장치를 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.database import DemoDataStore
from src.utils import load_app_config


DEMO_OBSERVATION_WEEK_KEY = "demo_observation_week"


@dataclass(frozen=True)
class DemoResetResult:
    """시연 초기화 후 복구본과 삭제 건수를 전달한다."""

    backup_path: Path
    deleted_counts: dict[str, int]

    @property
    def deleted_total(self) -> int:
        """초기화한 전체 행 수를 반환한다."""

        return sum(self.deleted_counts.values())


class DemoScenarioService:
    """고정 시드 대표 학생·기준 주차·안전 초기화를 제공한다."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        store: DemoDataStore | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_app_config()
        demo_config = self.config["demo"]
        self.enabled = bool(demo_config["enabled"])
        self.representative_student_id = str(
            demo_config["representative_student_id"]
        ).strip()
        self.baseline_week = int(demo_config["baseline_week"])
        self.observation_weeks = tuple(
            int(week)
            for week in demo_config.get(
                "observation_weeks",
                range(1, self.baseline_week + 1),
            )
        )
        self.reset_confirmation_text = str(
            demo_config["reset_confirmation_text"]
        ).strip()
        if not self.representative_student_id:
            raise ValueError("시연 대표 학생 ID는 비어 있을 수 없습니다.")
        if self.baseline_week < 1:
            raise ValueError("시연 기준 주차는 1 이상이어야 합니다.")
        if (
            not self.observation_weeks
            or tuple(sorted(set(self.observation_weeks)))
            != self.observation_weeks
            or self.observation_weeks
            != tuple(range(1, self.baseline_week + 1))
        ):
            raise ValueError("시연 관찰 주차는 1부터 기준 주차까지 연속되어야 합니다.")
        if not self.reset_confirmation_text:
            raise ValueError("초기화 확인 문구는 비어 있을 수 없습니다.")
        self.store = store or DemoDataStore(database_path)

    @property
    def observation_week_labels(self) -> tuple[str, ...]:
        """위험단계를 암시하지 않는 관찰 주차 라벨을 반환한다."""

        return tuple(f"{week}주차" for week in self.observation_weeks)

    def resolve_observation_week(self, option_label: str) -> int:
        """UI 라벨을 검증하고 해당 관찰 주차를 반환한다."""

        normalized = str(option_label).strip()
        by_label = {
            f"{week}주차": week for week in self.observation_weeks
        }
        if normalized not in by_label:
            raise ValueError("지원하지 않는 시연 관찰 주차입니다.")
        return by_label[normalized]

    def filter_snapshots_through_week(
        self,
        snapshots: pd.DataFrame,
        observation_week: int,
    ) -> pd.DataFrame:
        """선택 주차까지의 복사본만 반환하며 원본 스냅샷은 변경하지 않는다.

        Parameters:
            snapshots: 학생·주차별 위험 스냅샷.
            observation_week: UI에서 선택한 synthetic 관찰 주차.

        Returns:
            선택 주차 이하의 행만 가진 새 DataFrame.

        Assumptions:
            이 기능은 발표용 조회 범위만 바꾸며 CSV, SQLite, 시스템 시간은 수정하지 않는다.
        """

        if "week" not in snapshots.columns:
            raise ValueError("시연 주차 필터에 필요한 week 컬럼이 없습니다.")
        week = int(observation_week)
        if week not in self.observation_weeks:
            raise ValueError("지원하지 않는 시연 관찰 주차입니다.")
        return snapshots[
            snapshots["week"].astype(int) <= week
        ].copy().reset_index(drop=True)

    def representative_risk_path_is_ready(
        self,
        snapshots: pd.DataFrame,
    ) -> bool:
        """대표 학생의 모든 관찰 주차와 고위험 관찰값이 존재하는지 확인한다."""

        required = {"student_id", "week", "risk_level"}
        if required.difference(snapshots.columns):
            return False
        representative = snapshots[
            snapshots["student_id"].astype(str)
            == self.representative_student_id
        ].copy()
        representative["week"] = pd.to_numeric(
            representative["week"], errors="coerce"
        )
        representative = representative[
            representative["week"].isin(self.observation_weeks)
        ].sort_values("week")
        if representative["week"].astype(int).tolist() != list(
            self.observation_weeks
        ):
            return False
        return bool((representative["risk_level"] == "고위험").any())

    def find_alert_candidates(
        self,
        latest_snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        """선택한 관찰 시점에서 실제 고위험인 학생을 안내 후보로 반환한다.

        Parameters:
            latest_snapshots: 선택 주차까지 학생별 최신값만 남긴 위험 결과.

        Returns:
            대표 학생을 먼저, 나머지는 종합 위험도순으로 정렬한 고위험 학생.

        Assumptions:
            주차는 위험단계가 아니며 어느 주차든 고위험 학생이 후보가 될 수 있다.
        """

        required = {"student_id", "week", "risk_level", "overall_risk"}
        missing = required.difference(latest_snapshots.columns)
        if missing:
            raise ValueError(
                f"경보 후보 선정에 필요한 컬럼이 없습니다: {sorted(missing)}"
            )
        candidates = latest_snapshots[
            latest_snapshots["risk_level"].astype(str) == "고위험"
        ].copy()
        if candidates.empty:
            return candidates.reset_index(drop=True)
        candidates["_representative_first"] = (
            candidates["student_id"].astype(str)
            == self.representative_student_id
        ).astype(int)
        candidates["_risk_order"] = pd.to_numeric(
            candidates["overall_risk"], errors="coerce"
        ).fillna(-1)
        return (
            candidates.sort_values(
                ["_representative_first", "_risk_order", "student_id"],
                ascending=[False, False, True],
                kind="stable",
            )
            .drop(columns=["_representative_first", "_risk_order"])
            .reset_index(drop=True)
        )

    def get_record_counts(self) -> dict[str, int]:
        """현재 SQLite 시연 기록 현황을 반환한다."""

        return self.store.get_record_counts()

    def reset_demo_records(
        self,
        confirmation_text: str,
        backup_dir: str | Path | None = None,
    ) -> DemoResetResult:
        """설정된 문구가 정확할 때만 백업 후 시연 기록을 초기화한다."""

        if not self.enabled:
            raise ValueError("현재 환경에서는 시연 초기화가 비활성화되어 있습니다.")
        if str(confirmation_text).strip() != self.reset_confirmation_text:
            raise ValueError("시연 초기화 확인 문구가 일치하지 않습니다.")
        backup_path, deleted_counts = self.store.backup_and_reset(backup_dir)
        return DemoResetResult(backup_path, deleted_counts)
