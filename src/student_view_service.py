"""학생 체크인부터 맞춤 지원·학습경로까지 기존 서비스를 조율한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping

import pandas as pd

from src.ai import AIProvider
from src.checkin_service import CheckinAnalysisResult, CheckinAnalysisService
from src.course_recommender import CourseRecommender
from src.database import (
    LearningPathDataStore,
    StudentCheckinDataStore,
    database_transaction,
)
from src.demo_alert_notification_service import DemoAlertNotificationService
from src.intervention_recommender import (
    InterventionRecommender,
    ProgramRecommendation,
    RecommendationResult,
)
from src.intervention_service import InterventionContext, InterventionManagementService
from src.learning_path_builder import LearningPathBuilder, LearningPathResult
from src.repositories import StudentSuccessRepository
from src.risk_engine import build_risk_snapshots
from src.risk_service import RiskAnalysisService
from src.utils import load_risk_config


ANALYSIS_FEEDBACK_TYPES = ("맞아요", "조금 달라요", "수정하고 싶어요")
STUDENT_RECOMMENDATION_FEEDBACK_MAP = {
    "관심 있어요": "관심 있음",
    "저장할게요": "나중에 보기",
    "관심 없어요": "관심 없음",
    "추천 이유가 맞지 않아요": "추천 이유가 맞지 않음",
    "다른 분야를 추천받고 싶어요": "다른 분야 추천 희망",
}
STUDENT_SUPPORT_STATUS_LABELS = {
    "추천 생성": "맞춤 지원 준비 중",
    "교직원 검토 대기": "담당자 확인 중",
    "교직원 검토": "지원 내용 검토 중",
    "지원계획 작성": "지원 내용 검토 중",
    "연락 전": "연락 준비 중",
    "상담 예정": "상담 예정",
    "상담 완료": "상담 완료",
    "프로그램 참여": "프로그램 참여 중",
    "학생 거절": "현재 참여하지 않음",
    "추후 관찰": "필요할 때 다시 살펴볼 예정",
}


@dataclass(frozen=True)
class SupportArea:
    """위험점수를 노출하지 않는 학생 친화적 지원 영역 상태."""

    key: str
    label: str
    status: str
    guidance: str


@dataclass(frozen=True)
class StudentJourneyResult:
    """한 번의 학생 체크인으로 생성·저장된 전체 맞춤 지원 결과."""

    checkin_id: int
    profile: dict[str, Any]
    analysis_result: CheckinAnalysisResult
    support_areas: tuple[SupportArea, ...]
    recommendation_result: RecommendationResult
    intervention_context: InterventionContext
    learning_path_result: LearningPathResult | None
    learning_path_recommendation_id: int | None
    learning_path_error: str | None = None


@dataclass(frozen=True)
class RecommendationRegenerationResult:
    """교직원 조정 후 새 버전으로 저장된 통합 추천 결과."""

    recommendation_result: RecommendationResult
    intervention_context: InterventionContext
    learning_path_result: LearningPathResult | None
    learning_path_recommendation_id: int | None
    learning_path_error: str | None = None
    program_recommendations_preserved: bool = False
    learning_path_preserved: bool = False


@dataclass(frozen=True)
class StaffStudentContext:
    """교직원 화면에서 확인할 학생 제출 체크인과 피드백 묶음."""

    latest_checkin: dict[str, Any] | None
    analysis_feedback: pd.DataFrame
    recommendation_feedback: pd.DataFrame
    intervention: dict[str, Any] | None
    recommendations: pd.DataFrame
    learning_path: pd.DataFrame
    support_plan: dict[str, Any] | None
    support_plan_items: pd.DataFrame


class StudentViewQueryService:
    """학생 제출 내용을 추천 모델 로딩 없이 조회 화면에 제공한다."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.checkin_store = StudentCheckinDataStore(database_path)
        self.intervention_service = InterventionManagementService(
            database_path=database_path
        )
        self.learning_path_store = LearningPathDataStore(database_path)

    def get_staff_student_context(self, student_id: str) -> StaffStudentContext:
        """학생 제출 내용과 두 종류의 피드백을 교직원 조회용으로 반환한다."""

        latest_checkin = self.checkin_store.get_latest_checkin(student_id)
        intervention: dict[str, Any] | None = None
        recommendations = pd.DataFrame()
        learning_path = pd.DataFrame()
        support_plan: dict[str, Any] | None = None
        support_plan_items = pd.DataFrame()
        if latest_checkin is not None:
            checkin_id = int(latest_checkin["checkin_id"])
            intervention = self.intervention_service.get_intervention_for_checkin(
                checkin_id
            )
            if intervention is not None:
                batch_id = str(intervention.get("recommendation_batch_id") or "")
                if batch_id:
                    recommendations = self.intervention_service.list_recommendations(
                        batch_id
                    )
                intervention_id = int(intervention["intervention_id"])
                support_plan = self.intervention_service.get_support_plan(
                    intervention_id
                )
                support_plan_items = (
                    self.intervention_service.list_support_plan_items(
                        intervention_id
                    )
                )
                learning_path = self.learning_path_store.get_learning_path_version(
                    checkin_id,
                    int(intervention.get("generation_version") or 1),
                )
            else:
                learning_path = self.learning_path_store.get_learning_path_by_checkin(
                    checkin_id
                )
        return StaffStudentContext(
            latest_checkin=latest_checkin,
            analysis_feedback=self.checkin_store.list_analysis_feedback(student_id),
            recommendation_feedback=self.intervention_service.list_feedback(student_id),
            intervention=intervention,
            recommendations=recommendations,
            learning_path=learning_path,
            support_plan=support_plan,
            support_plan_items=support_plan_items,
        )

    def get_student_mypage_context(self, student_id: str) -> StaffStudentContext:
        """마이페이지에서 최신 체크인과 저장된 맞춤 결과를 다시 조회한다."""

        return self.get_staff_student_context(student_id)


def describe_student_support_status(status: str | None) -> str:
    """내부 개입 상태를 학생이 이해하기 쉬운 진행 문구로 변환한다."""

    normalized = str(status or "").strip()
    return STUDENT_SUPPORT_STATUS_LABELS.get(normalized, "진행 상황 확인 중")


def build_student_support_areas(
    profile: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> tuple[SupportArea, ...]:
    """내부 영역 위험점수를 낙인 없는 지원 상태 문구로 변환한다.

    Parameters:
        profile: 기존 risk engine이 계산한 최신 위험 스냅샷.
        config: 선택적인 risk_config. 미지정 시 프로젝트 설정 사용.

    Returns:
        점수 없이 텍스트 상태를 제공하는 다섯 지원 영역.

    Assumptions:
        지원 권장 기준은 risk_types.secondary_min 설정을 재사용한다.
    """

    active_config = config or load_risk_config()
    normal_max = float(active_config["risk_levels"]["normal_max"])
    support_min = float(active_config["risk_types"]["secondary_min"])
    area_definitions = (
        ("major_adaptation", "전공 탐색", "관심과 적성에 맞는 전공 방향을 함께 살펴볼 수 있어요."),
        ("career", "진로 방향 설정", "희망 직무와 필요한 준비를 구체화할 수 있어요."),
        ("engagement", "학습 참여", "수업·과제 참여 흐름을 점검하고 학습 리듬을 되찾을 수 있어요."),
        ("attendance", "출결 상태", "최근 출결 흐름을 확인하고 필요한 도움을 연결할 수 있어요."),
        ("achievement", "학습 지원", "현재 학습 수준에 맞는 코칭과 보완 방법을 찾아볼 수 있어요."),
    )
    areas: list[SupportArea] = []
    for key, label, guidance in area_definitions:
        score = float(profile.get(f"{key}_risk", 0) or 0)
        if score >= support_min:
            status = "지원 권장"
        elif score > normal_max:
            status = "점검 필요"
        else:
            status = "양호"
        areas.append(SupportArea(key, label, status, guidance))
    return tuple(areas)


def describe_student_analysis(analysis: Mapping[str, Any]) -> tuple[str, ...]:
    """구조화 AI 결과를 학생이 확인하기 쉬운 문장으로 변환한다."""

    messages: list[str] = []
    if bool(analysis.get("major_concern")):
        messages.append("전공 탐색이 필요해 보여요")
    if bool(analysis.get("learning_difficulty")):
        messages.append("수업을 따라가기 위한 학습 지원이 도움이 될 수 있어요")
    if bool(analysis.get("career_uncertainty")):
        messages.append("진로 방향을 구체화하는 데 도움이 필요해 보여요")
    interests = [str(item) for item in analysis.get("interests", []) if str(item)]
    if interests:
        messages.append(f"{'·'.join(interests)} 분야에 관심이 있어요")
    if not messages:
        messages.append("현재 응답에서는 긴급한 고민보다 꾸준한 성장을 위한 관심이 확인됐어요")
    return tuple(messages)


class StudentViewService:
    """기존 Repository·위험엔진·AI·추천 서비스를 학생 여정으로 연결한다."""

    def __init__(
        self,
        repository: StudentSuccessRepository,
        ai_provider: AIProvider,
        database_path: str | Path | None = None,
        program_recommender: InterventionRecommender | None = None,
        course_recommender: CourseRecommender | None = None,
    ) -> None:
        self.repository = repository
        self.ai_provider = ai_provider
        self.checkin_store = StudentCheckinDataStore(database_path)
        self.intervention_service = InterventionManagementService(
            database_path=database_path
        )
        self.learning_path_store = LearningPathDataStore(database_path)
        self.demo_alert_notification_service = DemoAlertNotificationService(
            database_path=database_path
        )
        self.program_recommender = program_recommender or InterventionRecommender()
        self.course_recommender = course_recommender or CourseRecommender()

    @staticmethod
    def _canonical_checkin(
        student_id: str,
        week: int,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """학생 UI 값을 기존 risk engine 체크인 스키마로 정규화한다."""

        interests = values.get("interest_fields", [])
        if isinstance(interests, str):
            interest_text = interests
        else:
            interest_text = "|".join(
                str(item).strip() for item in interests if str(item).strip()
            )
        return {
            "student_id": str(student_id),
            "week": int(week),
            "major_interest": int(values["major_interest"]),
            "major_satisfaction": int(values["major_satisfaction"]),
            "major_continuation_intent": int(values["major_continuation_intent"]),
            "career_clarity": int(values["career_clarity"]),
            "learning_difficulty": int(values["learning_difficulty"]),
            "consultation_intent": int(values["consultation_intent"]),
            "interest_fields": interest_text,
            "desired_job": str(values.get("desired_job", "")).strip(),
            "natural_language_concern": str(
                values.get("natural_language_concern", "")
            ).strip(),
        }

    def _build_current_profile(
        self,
        student_id: str,
        values: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        """최신 활동과 새 체크인으로 기존 risk engine 스냅샷을 다시 계산한다."""

        source = self.repository.get_all()
        student_rows = source.students[
            source.students["student_id"].astype(str) == str(student_id)
        ]
        if student_rows.empty:
            raise ValueError(f"존재하지 않는 학생 ID입니다: {student_id}")
        activity_rows = source.weekly_activity[
            source.weekly_activity["student_id"].astype(str) == str(student_id)
        ]
        if activity_rows.empty:
            raise ValueError(f"학생 활동 데이터가 없습니다: {student_id}")
        if str(values.get("response_mode", "scaled")) == "narrative":
            snapshots = RiskAnalysisService(
                self.repository,
                checkin_store=self.checkin_store,
            ).build_snapshots(source)
            current_rows = snapshots[
                snapshots["student_id"].astype(str) == str(student_id)
            ].sort_values("week")
            if current_rows.empty:
                raise ValueError(f"학생 위험분석 데이터가 없습니다: {student_id}")
            profile = current_rows.iloc[-1].to_dict()
            interests = values.get("interest_fields", []) or []
            profile["interest_fields"] = (
                str(interests)
                if isinstance(interests, str)
                else "|".join(str(item) for item in interests if str(item).strip())
            )
            profile["desired_job"] = str(values.get("desired_job", "")).strip()
            profile["natural_language_concern"] = str(
                values.get("natural_language_concern", "")
            ).strip()
            profile["consultation_requested"] = bool(
                values.get("consultation_requested", False)
            )
            profile["explore_other_fields"] = bool(
                values.get("explore_other_fields", False)
            )
            profile["response_mode"] = "narrative"
            return profile, source
        latest_week = int(activity_rows["week"].max())
        previous_checkin = self.checkin_store.get_latest_checkin(student_id)
        if values.get("week") is not None:
            submission_week = int(values["week"])
        elif previous_checkin is not None and previous_checkin.get("week") is not None:
            submission_week = max(latest_week, int(previous_checkin["week"]) + 1)
        else:
            submission_week = latest_week
        if submission_week < latest_week:
            raise ValueError(
                f"체크인 주차는 현재 활동 주차({latest_week}주) 이상이어야 합니다."
            )
        latest_activity = activity_rows[activity_rows["week"] == latest_week].copy()
        latest_activity["week"] = submission_week
        checkin = self._canonical_checkin(student_id, submission_week, values)
        profile = build_risk_snapshots(
            student_rows,
            latest_activity,
            pd.DataFrame([checkin]),
        ).iloc[0].to_dict()
        profile["consultation_requested"] = bool(
            values.get("consultation_requested", False)
        )
        profile["explore_other_fields"] = bool(
            values.get("explore_other_fields", False)
        )
        return profile, source

    def build_and_store_learning_path(
        self,
        student_id: str,
        profile: Mapping[str, Any],
        analysis: Mapping[str, Any],
        source: Any,
        *,
        generation_version: int = 1,
        create_new_version: bool = False,
        excluded_course_ids: Collection[str] = (),
    ) -> tuple[LearningPathResult, int]:
        """DB 과목으로 경로를 생성하고 체크인·버전별로 저장한다."""

        completed_ids = set(
            source.completed_courses[
                (source.completed_courses["student_id"].astype(str) == str(student_id))
                & (source.completed_courses["completion_status"] == "completed")
            ]["course_id"].astype(str)
        )
        excluded_courses = {str(course_id) for course_id in excluded_course_ids}
        available_courses = source.courses[
            ~source.courses["course_id"].astype(str).isin(excluded_courses)
        ].copy()
        selection = self.course_recommender.recommend(
            profile=profile,
            analysis=analysis,
            courses=available_courses,
            completed_course_ids=completed_ids,
        )
        result = LearningPathBuilder(self.ai_provider).build(profile, selection)
        explanation = result.explanation
        roles = {
            str(item["course_id"]): item for item in explanation["course_roles"]
        }
        path_record = {
            "student_id": str(student_id),
            "path_name": str(explanation["path_name"]),
            "related_job": str(profile.get("desired_job", "")),
            "competencies": list(explanation["competencies"]),
            "reason": str(explanation["reason"]),
            "career_connection": str(explanation["career_connection"]),
            "provider_name": result.provider_name,
            "fallback_used": result.fallback_used,
            "courses": [
                {
                    "course_id": course.course_id,
                    "sequence": int(roles[course.course_id]["sequence"]),
                    "score": course.score,
                    "reason": str(roles[course.course_id]["role"]),
                    "selection_reason": course.reason,
                }
                for course in selection.courses
            ],
        }
        with database_transaction(self.learning_path_store.database_path) as connection:
            analysis_batch_id = self.learning_path_store.save_analysis(
                paths=[path_record],
                candidates=[],
                source_checkin_id=int(profile["student_checkin_id"]),
                generation_version=int(generation_version),
                create_new_version=create_new_version,
                connection=connection,
            )
            target_id = f"learning_path:{analysis_batch_id}"
            average_score = sum(course.score for course in selection.courses) / len(
                selection.courses
            )
            _, recommendation_ids = (
                self.intervention_service.store.save_recommendation_batch(
                    student_id=str(student_id),
                    connection=connection,
                    recommendations=[
                        {
                            "recommendation_type": "learning_path",
                            "target_id": target_id,
                            "score": average_score,
                            "rank": 1,
                            "reason": str(explanation["reason"]),
                        }
                    ],
                )
            )
        return result, recommendation_ids[target_id]

    @staticmethod
    def _restore_program_recommendations(
        recommendation_rows: pd.DataFrame,
        programs: pd.DataFrame,
    ) -> RecommendationResult:
        """조정하지 않은 비교과 추천을 이전 버전과 동일하게 복원한다."""

        program_master = programs.copy()
        program_master["program_id"] = program_master["program_id"].astype(str)
        program_by_id = program_master.set_index("program_id").to_dict(
            orient="index"
        )
        restored: list[ProgramRecommendation] = []
        program_rows = recommendation_rows[
            recommendation_rows["recommendation_type"] == "support_program"
        ].sort_values("rank")
        for row in program_rows.to_dict(orient="records"):
            program_id = str(row["target_id"])
            program = program_by_id.get(program_id)
            if program is None:
                raise ValueError(
                    "현재 비교과 master에 없는 기존 추천은 유지할 수 없습니다."
                )
            restored.append(
                ProgramRecommendation(
                    program_id=program_id,
                    program_name=str(program["program_name"]),
                    program_type=str(program["program_type"]),
                    description=str(program["description"]),
                    department_in_charge=str(program["department_in_charge"]),
                    operation_period=str(program["operation_period"]),
                    score=float(row["score"]),
                    reason=str(row["reason"]),
                )
            )
        if not restored:
            raise ValueError("유지할 기존 비교과 추천이 없습니다.")
        return RecommendationResult(
            recommendations=tuple(restored),
            similarity_backend="stored_previous_version",
        )

    def _copy_learning_path_version(
        self,
        current_path: pd.DataFrame,
        checkin_id: int,
        target_version: int,
    ) -> None:
        """조정하지 않은 교과 경로와 검토 판정을 새 버전에 그대로 보존한다."""

        if current_path.empty:
            return
        path = current_path.iloc[0]
        reviewed_at = path.get("reviewed_at")
        if pd.isna(reviewed_at) or not str(reviewed_at).strip():
            reviewed_at = None
        course_rows = current_path.sort_values("sequence").drop_duplicates(
            "course_id"
        )
        self.learning_path_store.save_analysis(
            paths=[
                {
                    "student_id": str(path["student_id"]),
                    "path_name": str(path["path_name"]),
                    "related_job": str(path.get("related_job") or ""),
                    "competencies": list(path.get("competencies") or []),
                    "reason": str(path["path_reason"]),
                    "career_connection": str(
                        path.get("career_connection") or ""
                    ),
                    "provider_name": str(path.get("provider_name") or ""),
                    "fallback_used": bool(path.get("fallback_used")),
                    "review_status": str(
                        path.get("review_status") or "검토 대기"
                    ),
                    "review_note": str(path.get("review_note") or ""),
                    "reviewed_at": reviewed_at,
                    "courses": [
                        {
                            "course_id": str(course["course_id"]),
                            "sequence": int(course["sequence"]),
                            "score": float(course["score"]),
                            "reason": str(course["course_role"]),
                            "selection_reason": str(
                                course.get("selection_reason") or ""
                            ),
                            "review_judgment": str(
                                course.get("review_judgment") or "판단 전"
                            ),
                        }
                        for course in course_rows.to_dict(orient="records")
                    ],
                }
            ],
            candidates=[],
            source_checkin_id=checkin_id,
            generation_version=target_version,
            create_new_version=True,
        )

    def regenerate_recommendations(
        self,
        student_id: str,
        profile: Mapping[str, Any],
        analysis: Mapping[str, Any],
        source: Any,
        *,
        excluded_program_ids: Collection[str] = (),
        excluded_course_ids: Collection[str] = (),
    ) -> RecommendationRegenerationResult:
        """교직원이 지정한 항목을 제외하고 통합 추천을 새 버전으로 생성한다.

        Parameters:
            student_id: 추천 대상 synthetic 학생 ID.
            profile: 원래 체크인의 위험 스냅샷과 학생 응답.
            analysis: JSON Schema 검증을 마친 AI 이해 결과.
            source: Repository가 반환한 현재 master.
            excluded_program_ids: 현재 비교과 추천에서 제외할 ID.
            excluded_course_ids: 현재 교과 경로에서 제외할 ID.

        Returns:
            기존 버전을 보존하고 새로 저장한 비교과·교과 결과.

        Assumptions:
            재추천은 교직원이 해당 비교과 또는 교과 영역을 `조정 필요`로
            저장한 최신 버전에서만 허용하며 검토 메모를 AI 프롬프트로 사용하지 않는다.
        """

        source_checkin_id = profile.get("student_checkin_id")
        if source_checkin_id is None:
            raise ValueError("재추천은 학생 체크인에 연결된 결과에서만 가능합니다.")
        checkin_id = int(source_checkin_id)
        current_intervention = self.intervention_service.get_intervention_for_checkin(
            checkin_id
        )
        if current_intervention is None:
            raise ValueError("재추천할 기존 추천 버전이 없습니다.")
        if str(current_intervention.get("student_id")) != str(student_id):
            raise ValueError("다른 학생의 체크인 추천은 다시 만들 수 없습니다.")
        excluded_programs = {str(item) for item in excluded_program_ids}
        excluded_courses = {str(item) for item in excluded_course_ids}
        if not excluded_programs and not excluded_courses:
            raise ValueError("제외할 비교과 또는 교과목을 하나 이상 선택해 주세요.")

        current_version = int(current_intervention.get("generation_version") or 1)
        current_path = self.learning_path_store.get_learning_path_version(
            checkin_id,
            current_version,
        )
        program_adjustment_requested = (
            str(current_intervention.get("staff_action") or "")
            == "추천 조정 필요"
        )
        course_adjustment_requested = (
            not current_path.empty
            and str(current_path.iloc[0].get("review_status") or "")
            == "조정 필요"
        )
        if excluded_programs and not program_adjustment_requested:
            raise ValueError("비교과를 먼저 '조정 필요'로 저장해 주세요.")
        if excluded_courses and not course_adjustment_requested:
            raise ValueError("교과를 먼저 '조정 필요'로 저장해 주세요.")

        batch_id = str(current_intervention.get("recommendation_batch_id") or "")
        current_program_rows = self.intervention_service.list_recommendations(
            batch_id
        )
        current_program_ids = set(
            current_program_rows[
                current_program_rows["recommendation_type"] == "support_program"
            ]["target_id"].astype(str)
        )
        unknown_program_ids = excluded_programs - current_program_ids
        if unknown_program_ids:
            raise ValueError("현재 추천에 없는 비교과는 제외할 수 없습니다.")

        current_course_ids = (
            set(current_path["course_id"].astype(str))
            if not current_path.empty
            else set()
        )
        unknown_course_ids = excluded_courses - current_course_ids
        if unknown_course_ids:
            raise ValueError("현재 학습경로에 없는 교과목은 제외할 수 없습니다.")

        program_recommendations_preserved = not excluded_programs
        if program_recommendations_preserved:
            recommendation_result = self._restore_program_recommendations(
                current_program_rows,
                source.support_programs,
            )
        else:
            available_programs = source.support_programs[
                ~source.support_programs["program_id"].astype(str).isin(
                    excluded_programs
                )
            ].copy()
            recommendation_result = self.program_recommender.recommend(
                profile=profile,
                analysis=analysis,
                programs=available_programs,
            )
        intervention_context = self.intervention_service.record_recommendation_batch(
            student_id=student_id,
            recommendations=recommendation_result.recommendations,
            risk_snapshot=profile,
            source_checkin_id=checkin_id,
            initial_status="교직원 검토 대기",
            create_new_version=True,
        )
        if program_recommendations_preserved:
            self.intervention_service.update_status(
                intervention_id=intervention_context.intervention_id,
                status=str(current_intervention.get("status") or "교직원 검토 대기"),
                staff_action=str(current_intervention.get("staff_action") or ""),
                staff_note=str(current_intervention.get("staff_note") or ""),
            )

        learning_path_result: LearningPathResult | None = None
        learning_path_recommendation_id: int | None = None
        learning_path_error: str | None = None
        learning_path_preserved = bool(
            not current_path.empty and not excluded_courses
        )
        if not current_path.empty:
            if learning_path_preserved:
                self._copy_learning_path_version(
                    current_path,
                    checkin_id,
                    intervention_context.generation_version,
                )
            else:
                try:
                    (
                        learning_path_result,
                        learning_path_recommendation_id,
                    ) = self.build_and_store_learning_path(
                        student_id,
                        profile,
                        analysis,
                        source,
                        generation_version=intervention_context.generation_version,
                        create_new_version=True,
                        excluded_course_ids=excluded_courses,
                    )
                except Exception as error:
                    learning_path_error = str(error)

        return RecommendationRegenerationResult(
            recommendation_result=recommendation_result,
            intervention_context=intervention_context,
            learning_path_result=learning_path_result,
            learning_path_recommendation_id=learning_path_recommendation_id,
            learning_path_error=learning_path_error,
            program_recommendations_preserved=program_recommendations_preserved,
            learning_path_preserved=learning_path_preserved,
        )

    def submit_checkin(
        self,
        student_id: str,
        values: Mapping[str, Any],
        analysis_result: CheckinAnalysisResult | None = None,
    ) -> StudentJourneyResult:
        """학생 체크인을 저장하고 분석·Top3 지원·조건부 학습경로를 생성한다.

        Parameters:
            student_id: 시연용 학생 계정으로 검증된 synthetic 학생 ID.
            values: 학생이 직접 입력한 체크인 응답.
            analysis_result: 제출 전 학생이 확인한 선택적 AI 구조화 결과.

        Returns:
            학생 화면에 원문 AI 응답 없이 표시할 검증된 전체 결과.

        Assumptions:
            생성형 AI는 해석과 경로 설명만 수행하고 후보 ID는 기존 코드가 결정한다.
            필수 결과와 안내 완료는 한 트랜잭션으로 저장한다. 실패하면 저장 전
            상태로 복원하므로 재시도 시 체크인·주차가 중복 증가하지 않는다.
            선택적 학습경로 실패는 별도 안내하고 저장된 체크인과 지원 추천을 유지한다.
        """

        profile, source = self._build_current_profile(student_id, values)
        response_mode = str(values.get("response_mode", "scaled"))
        stored_values = dict(values)
        stored_values["week"] = (
            None if response_mode == "narrative" else int(profile["week"])
        )
        profile["checkin_source"] = (
            "student_narrative"
            if response_mode == "narrative"
            else "student_submission"
        )
        # AI·추천 계산 중에는 DB 쓰기 잠금을 잡거나 체크인을 먼저 저장하지 않는다.
        analysis_result = analysis_result or self.preview_checkin_analysis(
            str(values.get("natural_language_concern", ""))
        )
        recommendation_result = self.program_recommender.recommend(
            profile=profile,
            analysis=analysis_result.analysis,
            programs=source.support_programs,
        )
        risk_config = load_risk_config()
        support_areas = build_student_support_areas(profile, risk_config)
        path_threshold = float(risk_config["risk_types"]["secondary_min"])
        analysis = analysis_result.analysis
        path_is_recommended = (
            float(profile["major_adaptation_risk"]) >= path_threshold
            or float(profile["career_risk"]) >= path_threshold
            or bool(profile["explore_other_fields"])
            or bool(analysis["major_concern"])
            or bool(analysis["career_uncertainty"])
        )
        with database_transaction(self.checkin_store.database_path) as connection:
            checkin_id = self.checkin_store.save_checkin(
                student_id, stored_values, connection=connection
            )
            profile["student_checkin_id"] = checkin_id
            self.checkin_store.save_analysis(
                checkin_id=checkin_id,
                analysis=analysis_result.analysis,
                provider_name=analysis_result.provider_name,
                fallback_used=analysis_result.fallback_used,
                connection=connection,
            )
            intervention_context = self.intervention_service.record_recommendation_batch(
                student_id=student_id,
                recommendations=recommendation_result.recommendations,
                risk_snapshot=profile,
                source_checkin_id=checkin_id,
                initial_status="교직원 검토 대기",
                connection=connection,
            )
            self.demo_alert_notification_service.mark_checkin_completed(
                student_id, checkin_id, connection=connection
            )

        learning_path_result: LearningPathResult | None = None
        learning_path_recommendation_id: int | None = None
        learning_path_error: str | None = None
        if path_is_recommended:
            try:
                (
                    learning_path_result,
                    learning_path_recommendation_id,
                ) = self.build_and_store_learning_path(
                    student_id, profile, analysis, source
                )
            except Exception as error:
                learning_path_error = str(error)

        return StudentJourneyResult(
            checkin_id=checkin_id,
            profile=profile,
            analysis_result=analysis_result,
            support_areas=support_areas,
            recommendation_result=recommendation_result,
            intervention_context=intervention_context,
            learning_path_result=learning_path_result,
            learning_path_recommendation_id=learning_path_recommendation_id,
            learning_path_error=learning_path_error,
        )

    def preview_checkin_analysis(self, text: str) -> CheckinAnalysisResult:
        """저장 전에 학생이 확인할 검증된 자유서술 AI 분석을 생성한다.

        Parameters:
            text: Kare 대화에서 학생이 작성한 학교생활·전공·진로 고민.

        Returns:
            원문 모델 응답을 제외한 JSON Schema 검증 완료 결과.

        Assumptions:
            반환값을 ``submit_checkin``에 다시 전달하면 동일 내용을 재호출하지 않는다.
        """

        return CheckinAnalysisService(self.ai_provider).analyze(str(text or ""))

    def record_analysis_feedback(
        self,
        student_id: str,
        checkin_id: int,
        feedback_type: str,
        comment: str = "",
    ) -> int:
        """허용된 AI 이해 결과 피드백을 저장한다."""

        if feedback_type not in ANALYSIS_FEEDBACK_TYPES:
            raise ValueError(f"지원하지 않는 분석 피드백입니다: {feedback_type}")
        return self.checkin_store.save_analysis_feedback(
            checkin_id, student_id, feedback_type, comment
        )

    def record_recommendation_feedback(
        self,
        student_id: str,
        recommendation_id: int,
        feedback_label: str,
        comment: str = "",
    ) -> int:
        """학생 친화적 응답 문구를 기존 feedback 유형으로 변환해 저장한다."""

        try:
            stored_type = STUDENT_RECOMMENDATION_FEEDBACK_MAP[feedback_label]
        except KeyError as error:
            raise ValueError(
                f"지원하지 않는 추천 피드백입니다: {feedback_label}"
            ) from error
        return self.intervention_service.record_feedback(
            student_id, recommendation_id, stored_type, comment
        )

    def get_staff_student_context(self, student_id: str) -> StaffStudentContext:
        """학생 제출 내용과 두 종류의 피드백을 교직원 조회용으로 반환한다."""

        return StudentViewQueryService(
            self.checkin_store.database_path
        ).get_staff_student_context(student_id)
