"""추천 생성 이후의 교직원 개입 상태와 학생 피드백을 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.database import InterventionDataStore
from src.intervention_recommender import ProgramRecommendation
from src.utils import load_app_config


REVIEW_ACTIONS: dict[str, tuple[str, str]] = {
    "승인": ("지원계획 작성", "추천 검토 승인"),
    "수정": ("지원계획 작성", "추천 조정 필요"),
    "보류": ("추후 관찰", "보류"),
}
LEARNING_PATH_REVIEW_ACTIONS: dict[str, str] = {
    "승인": "승인",
    "수정": "조정 필요",
    "보류": "보류",
}


@dataclass(frozen=True)
class InterventionContext:
    """한 추천 생성 작업과 SQLite 식별자를 연결한다."""

    intervention_id: int
    batch_id: str
    recommendation_ids: dict[str, int]
    risk_snapshot_id: int | None = None
    source_checkin_id: int | None = None
    generation_version: int = 1


class InterventionManagementService:
    """UI 입력을 검증하고 SQLite 개입 저장소에 전달한다."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        store: InterventionDataStore | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_app_config()
        management_config = self.config["intervention_management"]
        self.statuses = tuple(str(item) for item in management_config["statuses"])
        self.feedback_types = tuple(
            str(item) for item in management_config["feedback_types"]
        )
        self.next_actions = tuple(
            str(item)
            for item in management_config.get(
                "next_actions",
                ("전화 안내", "상담 일정 조율", "프로그램 신청 연결"),
            )
        )
        self.plan_item_statuses = tuple(
            str(item)
            for item in management_config.get(
                "plan_item_statuses",
                ("지원 예정", "연락 완료", "참여 중", "지원 완료", "학생 미희망"),
            )
        )
        self.store = store or InterventionDataStore(database_path)

    def record_recommendation_batch(
        self,
        student_id: str,
        recommendations: Sequence[ProgramRecommendation],
        risk_snapshot: Mapping[str, Any] | None = None,
        source_checkin_id: int | None = None,
        initial_status: str = "추천 생성",
        create_new_version: bool = False,
    ) -> InterventionContext:
        """검증된 프로그램 추천과 최초 개입 상태를 함께 기록한다.

        Parameters:
            student_id: 추천 대상 synthetic 학생 ID.
            recommendations: deterministic recommender가 선정한 DB 프로그램.
            risk_snapshot: 추천 생성 시점의 검증된 위험엔진 결과.
            source_checkin_id: 학생 제출 체크인과 결과를 연결하는 선택적 ID.
            initial_status: 생성 직후 개입 상태.
            create_new_version: 같은 체크인에 명시적인 새 버전을 만들지 여부.

        Returns:
            이후 상태·피드백 저장에 필요한 SQLite 식별자.

        Assumptions:
            각 program_id는 지원프로그램 master 검증을 이미 통과했다.
        """

        if not recommendations:
            raise ValueError("저장할 프로그램 추천이 없습니다.")
        if initial_status not in self.statuses:
            raise ValueError(f"지원하지 않는 개입 상태입니다: {initial_status}")
        program_ids = [str(item.program_id) for item in recommendations]
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("중복 프로그램 추천은 저장할 수 없습니다.")
        existing = (
            self.store.get_intervention_by_checkin(source_checkin_id)
            if source_checkin_id is not None
            else None
        )
        if existing is not None and not create_new_version:
            return self._context_from_intervention(existing)
        generation_version = (
            int(existing.get("generation_version") or 1) + 1
            if existing is not None
            else 1
        )
        if source_checkin_id is not None and risk_snapshot is not None:
            snapshot_checkin_id = risk_snapshot.get("student_checkin_id")
            if (
                snapshot_checkin_id is not None
                and int(snapshot_checkin_id) != int(source_checkin_id)
            ):
                raise ValueError("다른 체크인의 위험 스냅샷을 연결할 수 없습니다.")
        records = [
            {
                "recommendation_type": "support_program",
                "target_id": recommendation.program_id,
                "score": recommendation.score,
                "rank": rank,
                "reason": recommendation.reason,
            }
            for rank, recommendation in enumerate(recommendations, start=1)
        ]
        batch_id, recommendation_ids = self.store.save_recommendation_batch(
            student_id=student_id,
            recommendations=records,
        )
        risk_snapshot_id = (
            self.store.save_risk_snapshot(student_id, risk_snapshot)
            if risk_snapshot is not None
            else None
        )
        intervention_id = self.store.create_intervention(
            student_id=student_id,
            recommendation_batch_id=batch_id,
            recommended_program_ids=program_ids,
            risk_snapshot_id=risk_snapshot_id,
            source_checkin_id=source_checkin_id,
            generation_version=generation_version,
            initial_status=initial_status,
        )
        return InterventionContext(
            intervention_id=intervention_id,
            batch_id=batch_id,
            recommendation_ids=recommendation_ids,
            risk_snapshot_id=risk_snapshot_id,
            source_checkin_id=source_checkin_id,
            generation_version=generation_version,
        )

    def _context_from_intervention(
        self, intervention: Mapping[str, Any]
    ) -> InterventionContext:
        """저장된 개입 행과 추천 배치를 서비스 context로 복원한다."""

        batch_id = str(intervention.get("recommendation_batch_id") or "")
        if not batch_id:
            raise ValueError("저장된 개입에 추천 배치가 연결되어 있지 않습니다.")
        recommendations = self.store.list_recommendations(batch_id)
        recommendation_ids = {
            str(row["target_id"]): int(row["recommendation_id"])
            for row in recommendations.to_dict(orient="records")
        }
        return InterventionContext(
            intervention_id=int(intervention["intervention_id"]),
            batch_id=batch_id,
            recommendation_ids=recommendation_ids,
            risk_snapshot_id=(
                int(intervention["risk_snapshot_id"])
                if intervention.get("risk_snapshot_id") is not None
                else None
            ),
            source_checkin_id=(
                int(intervention["source_checkin_id"])
                if intervention.get("source_checkin_id") is not None
                else None
            ),
            generation_version=int(intervention.get("generation_version") or 1),
        )

    def apply_review_action(
        self,
        intervention_id: int,
        action: str,
        staff_note: str = "",
    ) -> None:
        """승인·수정·보류 버튼을 표준 개입 상태로 변환해 저장한다."""

        if action not in REVIEW_ACTIONS:
            raise ValueError(f"지원하지 않는 검토 동작입니다: {action}")
        status, staff_action = REVIEW_ACTIONS[action]
        self.update_status(
            intervention_id=intervention_id,
            status=status,
            staff_action=staff_action,
            staff_note=staff_note,
        )

    def apply_integrated_review_action(
        self,
        intervention_id: int,
        action: str,
        staff_note: str = "",
    ) -> bool:
        """비교과 추천과 같은 체크인의 교과 학습경로를 함께 검토한다.

        Parameters:
            intervention_id: 검토할 비교과 추천 지원 ID.
            action: 승인·수정·보류 중 하나. 수정은 재추천을 실행하지 않고
                두 추천 유형에 조정 필요 상태와 메모를 기록한다.
            staff_note: 두 추천 유형에 공통으로 남길 교직원 메모.

        Returns:
            같은 체크인의 저장된 학습경로까지 갱신했는지 여부.

        Assumptions:
            학습경로가 아직 생성되지 않은 체크인은 비교과 검토만
            저장하고 ``False``를 반환한다.
        """

        if action not in REVIEW_ACTIONS:
            raise ValueError(f"지원하지 않는 검토 동작입니다: {action}")
        status, staff_action = REVIEW_ACTIONS[action]
        return self.store.update_integrated_recommendation_review(
            intervention_id=intervention_id,
            intervention_status=status,
            staff_action=staff_action,
            staff_note=staff_note,
            learning_path_status=LEARNING_PATH_REVIEW_ACTIONS[action],
        )

    def update_status(
        self,
        intervention_id: int,
        status: str,
        staff_action: str = "",
        staff_note: str = "",
    ) -> None:
        """설정에 등록된 상태만 교직원 개입 기록에 반영한다."""

        if status not in self.statuses:
            raise ValueError(f"지원하지 않는 개입 상태입니다: {status}")
        self.store.update_intervention(
            intervention_id=intervention_id,
            status=status,
            staff_action=staff_action,
            staff_note=staff_note,
        )

    def save_support_plan(
        self,
        intervention_id: int,
        selected_program_ids: Sequence[str],
        valid_program_ids: set[str],
        assigned_staff: str,
        planned_contact_date: date | str,
        next_action: str,
        plan_note: str = "",
        replacement_program_id: str | None = None,
        replaced_program_id: str | None = None,
        replacement_reason: str = "",
    ) -> int:
        """검토한 추천을 선택·교체해 실행 가능한 지원계획으로 확정한다.

        Parameters:
            intervention_id: 검토할 추천 배치의 개입 ID.
            selected_program_ids: AI 추천 중 지원계획에 유지할 ID.
            valid_program_ids: Repository master에 존재하는 프로그램 ID.
            assigned_staff: 실행 담당자 또는 부서.
            planned_contact_date: 학생 연락 예정일.
            next_action: 확정 직후 실행할 표준 조치.
            plan_note: 교직원 검토 메모.
            replacement_program_id: 추천 대신 선택한 DB 프로그램 ID.
            replaced_program_id: 교체되는 원래 추천 ID.
            replacement_reason: 학생 의사·운영여건을 반영한 교체 근거.

        Returns:
            저장된 support_plan_id.

        Assumptions:
            새 프로그램을 생성하지 않고 Repository master ID만 선택한다.
        """

        intervention = self.get_intervention(intervention_id)
        if intervention is None:
            raise ValueError(f"존재하지 않는 개입 ID입니다: {intervention_id}")
        assigned = str(assigned_staff).strip()
        if not assigned:
            raise ValueError("지원계획 담당자를 입력해야 합니다.")
        if next_action not in self.next_actions:
            raise ValueError(f"지원하지 않는 다음 조치입니다: {next_action}")
        try:
            contact_date = date.fromisoformat(str(planned_contact_date))
        except ValueError as error:
            raise ValueError("연락 예정일은 YYYY-MM-DD 형식이어야 합니다.") from error

        batch_id = str(intervention.get("recommendation_batch_id") or "")
        recommendations = self.list_recommendations(batch_id)
        recommendations = recommendations[
            recommendations["recommendation_type"] == "support_program"
        ]
        recommendation_by_program = {
            str(row["target_id"]): int(row["recommendation_id"])
            for row in recommendations.to_dict(orient="records")
        }
        recommended_ids = set(recommendation_by_program)
        selected_ids = [str(item).strip() for item in selected_program_ids]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("선택한 추천 프로그램은 중복될 수 없습니다.")
        unknown_selected = set(selected_ids).difference(recommended_ids)
        if unknown_selected:
            raise ValueError(
                f"현재 추천에 없는 프로그램입니다: {sorted(unknown_selected)}"
            )
        valid_ids = {str(item) for item in valid_program_ids}
        if not recommended_ids.issubset(valid_ids):
            raise ValueError("Repository master에 없는 추천 프로그램이 있습니다.")

        replacement_id = str(replacement_program_id or "").strip()
        replaced_id = str(replaced_program_id or "").strip()
        reason = str(replacement_reason).strip()
        if replacement_id:
            if replacement_id not in valid_ids:
                raise ValueError(
                    f"Repository master에 없는 교체 프로그램입니다: {replacement_id}"
                )
            if replacement_id in recommended_ids:
                raise ValueError("교체 프로그램은 기존 추천 외 항목으로 선택하세요.")
            if replaced_id not in recommended_ids:
                raise ValueError("교체할 기존 추천 프로그램을 선택해야 합니다.")
            if not reason:
                raise ValueError("프로그램 교체 이유를 입력해야 합니다.")
            selected_ids = [item for item in selected_ids if item != replaced_id]
        elif replaced_id or reason:
            raise ValueError("교체 프로그램을 함께 선택해야 합니다.")

        items: list[dict[str, Any]] = [
            {
                "program_id": program_id,
                "source_recommendation_id": recommendation_by_program[program_id],
                "selection_type": "AI 추천 선택",
            }
            for program_id in selected_ids
        ]
        if replacement_id:
            items.append(
                {
                    "program_id": replacement_id,
                    "source_recommendation_id": recommendation_by_program[replaced_id],
                    "selection_type": "교직원 교체",
                    "replaced_program_id": replaced_id,
                    "replacement_reason": reason,
                }
            )
        if not items:
            raise ValueError("지원계획에는 최소 한 개의 프로그램이 필요합니다.")

        plan_id = self.store.save_support_plan(
            intervention_id=int(intervention_id),
            student_id=str(intervention["student_id"]),
            assigned_staff=assigned,
            planned_contact_date=contact_date.isoformat(),
            next_action=next_action,
            plan_note=plan_note,
            items=items,
        )
        if str(intervention.get("status")) in {
            "추천 생성",
            "교직원 검토 대기",
            "교직원 검토",
            "지원계획 작성",
        }:
            self.update_status(
                intervention_id,
                "연락 전",
                staff_action="지원계획 확정",
                staff_note=plan_note,
            )
        return plan_id

    def get_support_plan(self, intervention_id: int) -> dict[str, Any] | None:
        """개입에 연결된 최신 지원계획을 반환한다."""

        return self.store.get_support_plan(intervention_id)

    def list_support_plan_items(self, intervention_id: int) -> pd.DataFrame:
        """프로그램별 선택 근거와 실행 상태를 반환한다."""

        return self.store.list_support_plan_items(intervention_id)

    def update_support_plan_item(
        self,
        plan_item_id: int,
        item_status: str,
        last_action: str = "",
        staff_note: str = "",
    ) -> None:
        """표준 상태만 사용해 프로그램별 실행 기록을 갱신한다."""

        if item_status not in self.plan_item_statuses:
            raise ValueError(f"지원하지 않는 프로그램 상태입니다: {item_status}")
        self.store.update_support_plan_item(
            plan_item_id,
            item_status,
            last_action,
            staff_note,
        )

    def record_feedback(
        self,
        student_id: str,
        recommendation_id: int,
        feedback_type: str,
        comment: str = "",
    ) -> int:
        """설정에 등록된 학생 피드백만 해당 추천에 연결한다."""

        if feedback_type not in self.feedback_types:
            raise ValueError(f"지원하지 않는 피드백 유형입니다: {feedback_type}")
        return self.store.save_feedback(
            student_id=student_id,
            recommendation_id=recommendation_id,
            feedback_type=feedback_type,
            comment=comment,
        )

    def get_intervention(self, intervention_id: int) -> dict[str, Any] | None:
        """개입 한 건의 현재 영속 상태를 반환한다."""

        return self.store.get_intervention(intervention_id)

    def get_latest_intervention(self, student_id: str) -> dict[str, Any] | None:
        """학생의 가장 최근 개입 상태를 반환한다."""

        return self.store.get_latest_intervention(student_id)

    def get_intervention_for_checkin(
        self, checkin_id: int
    ) -> dict[str, Any] | None:
        """학생 체크인에 연결된 최신 추천·개입 버전을 반환한다."""

        return self.store.get_intervention_by_checkin(checkin_id)

    def get_context_for_checkin(
        self, checkin_id: int
    ) -> InterventionContext | None:
        """학생 체크인에 연결된 저장 context를 반환한다."""

        intervention = self.get_intervention_for_checkin(checkin_id)
        return (
            self._context_from_intervention(intervention)
            if intervention is not None
            else None
        )

    def list_interventions(self) -> pd.DataFrame:
        """관리 화면용 전체 개입 이력을 반환한다."""

        return self.store.list_interventions()

    def list_latest_interventions(self) -> pd.DataFrame:
        """학생별 최신 개입 상태를 대시보드 결합용으로 반환한다."""

        return self.store.list_latest_interventions()

    def get_risk_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        """개입 생성 시점의 위험 스냅샷을 반환한다."""

        return self.store.get_risk_snapshot(snapshot_id)

    def list_risk_snapshots(self, student_id: str | None = None) -> pd.DataFrame:
        """선택 학생 또는 전체 개입 시점 위험 스냅샷을 반환한다."""

        return self.store.list_risk_snapshots(student_id)

    def list_recommendations(self, batch_id: str) -> pd.DataFrame:
        """한 개입에 연결된 프로그램 추천을 반환한다."""

        return self.store.list_recommendations(batch_id)

    def list_feedback(self, student_id: str | None = None) -> pd.DataFrame:
        """선택 학생 또는 전체 피드백 이력을 반환한다."""

        return self.store.list_feedback(student_id)
