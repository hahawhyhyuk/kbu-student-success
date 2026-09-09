"""학생의 AI 이해 결과와 비교과·교과 추천을 한 화면에서 검토한다."""

from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st

from src.ai import create_ai_provider
from src.checkin_semantic_state import semantic_coverage, semantic_review_lines
from src.checkin_service import CheckinAnalysisResult, CheckinAnalysisService
from src.course_catalog_presenter import format_course_caption
from src.course_recommender import CourseRecommender
from src.intervention_recommender import InterventionRecommender, RecommendationResult
from src.intervention_service import (
    COURSE_REVIEW_JUDGMENTS,
    InterventionContext,
    InterventionManagementService,
)
from src.learning_path_builder import (
    LearningPathBuilder,
    LearningPathResult,
    ordered_courses_by_sequence,
)
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
    select_latest_student_snapshots,
)
from src.student_view_service import StudentViewQueryService, StudentViewService
from src.ui import (
    queue_ai_reveal,
    render_ai_reveal,
    render_footer,
    render_page_header,
    render_submenu,
)


@st.cache_data
def load_review_data(
    checkin_revision: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repository에서 통합 검토에 필요한 master와 스냅샷을 로드한다."""

    repository = get_default_repository()
    source = repository.get_all()
    snapshots = create_integrated_risk_service(repository).build_snapshots(source)
    return (
        source.students,
        snapshots,
        source.support_programs,
        source.courses,
        source.completed_courses,
    )


@st.cache_resource
def get_program_recommender() -> InterventionRecommender:
    """지원프로그램 임베딩 backend를 rerun 간 재사용한다."""

    # 추천 근거 형식이 바뀌면 Streamlit의 이전 class instance를 재사용하지 않는다.
    return InterventionRecommender()


@st.cache_resource
def get_course_recommender() -> CourseRecommender:
    """교과목 임베딩 backend를 rerun 간 재사용한다."""

    # 교과 선정 근거 형식이 바뀌면 이전 class instance를 재사용하지 않는다.
    return CourseRecommender()


def get_intervention_service() -> InterventionManagementService:
    """현재 설정된 SQLite 경로에 연결된 개입 관리 서비스를 만든다."""

    return InterventionManagementService()


def resolve_gemini_api_key() -> str | None:
    """환경변수를 우선하고 Streamlit secrets를 보조로 확인한다."""

    environment_key = os.getenv("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        return secret_key or None
    except Exception:
        return None


def _analysis_from_checkin(checkin: dict[str, object]) -> dict[str, object]:
    """저장된 학생 체크인 분석을 UI와 추천 서비스용 mapping으로 복원한다."""

    return {
        "major_concern": bool(checkin["major_concern"]),
        "learning_difficulty": bool(checkin["analysis_learning_difficulty"]),
        "career_uncertainty": bool(checkin["career_uncertainty"]),
        "interests": list(checkin["interests"]),
        "desired_jobs": list(checkin["desired_jobs"]),
        "support_needs": list(checkin["support_needs"]),
        "summary": str(checkin["summary"]),
    }


def _render_reason_list(title: str, reason: object) -> None:
    """저장 시점이 다른 추천 근거도 짧은 항목으로 나눠 표시한다."""

    reason_text = str(reason or "").strip()
    items = [
        item.strip().rstrip(".")
        for item in re.split(r"\s+·\s+|\.\s+", reason_text)
        if item.strip().rstrip(".")
    ]
    st.markdown(f"**{title}**")
    for item in items or ["저장된 상세 근거 없음"]:
        st.markdown(f"- {item}")


def _render_course_cards(
    path_rows: pd.DataFrame,
    courses: pd.DataFrame,
    completed_ids: set[str],
    home_department: str,
) -> None:
    """저장된 경로의 교과목을 master와 다시 대조해 표시한다."""

    course_master = courses.set_index("course_id").to_dict(orient="index")
    visible_count = 0
    for row in path_rows.sort_values("sequence").to_dict(orient="records"):
        course_id = str(row["course_id"])
        course = course_master.get(course_id)
        if course is None or course_id in completed_ids:
            continue
        visible_count += 1
        with st.container(border=True):
            title_column, score_column = st.columns([4, 1])
            title_column.markdown(
                f"### {int(row['sequence'])}. {course['course_name']}"
            )
            score_column.metric("추천점수", f"{float(row['score']):.1f}")
            st.write(str(course["description"]))
            st.markdown(f"**과목 역할:** {row['course_role']}")
            selection_reason = str(row.get("selection_reason") or "").strip()
            if selection_reason:
                _render_reason_list("교과 선정 근거", selection_reason)
            st.caption(format_course_caption(course, home_department))
    if visible_count == 0:
        st.warning("현재 master와 이수 조건에서 표시할 교과목이 없습니다.")


def _render_generated_course_cards(
    result: LearningPathResult,
    courses: pd.DataFrame,
    completed_ids: set[str],
) -> None:
    """아직 DB에 연결되지 않은 수동 생성 경로를 검증해 표시한다."""

    role_by_id = {
        str(item["course_id"]): item
        for item in result.explanation["course_roles"]
    }
    master_ids = set(courses["course_id"].astype(str))
    for course in ordered_courses_by_sequence(
        result.selection, result.explanation
    ):
        if course.course_id not in master_ids or course.course_id in completed_ids:
            continue
        role = role_by_id[course.course_id]
        with st.container(border=True):
            title_column, score_column = st.columns([4, 1])
            title_column.markdown(f"### {role['sequence']}. {course.course_name}")
            score_column.metric("추천점수", f"{course.score:.1f}")
            st.write(course.description)
            st.markdown(f"**과목 역할:** {role['role']}")
            _render_reason_list("교과 선정 근거", course.reason)
            st.caption(format_course_caption(course))


st.set_page_config(page_title="AI 맞춤 추천 통합 검토", page_icon="🤝", layout="wide")
render_page_header(
    "AI 맞춤 추천 통합 검토",
    "비교과 지원과 교과 학습경로를 검토·승인합니다.",
    audience="학생 지원·교직원 검토용",
)

students, snapshots, support_programs, courses, completed_courses = load_review_data(
    get_student_checkin_revision()
)
students = students.copy()
students["student_id"] = students["student_id"].astype(str)
latest_all = select_latest_student_snapshots(snapshots)
latest_all = latest_all.copy()
latest_all["student_id"] = latest_all["student_id"].astype(str)
target_ids = latest_all[
    (latest_all["major_adaptation_risk"] >= 60) | (latest_all["career_risk"] >= 60)
]["student_id"].astype(str).tolist()
student_ids = target_ids + [
    item for item in sorted(students["student_id"].astype(str)) if item not in target_ids
]
student_lookup = students.set_index("student_id")
student_option_labels = {
    student_id: (
        f"{student_id} · {student_lookup.loc[student_id, 'department']} · "
        f"{int(student_lookup.loc[student_id, 'grade'])}학년"
    )
    for student_id in student_ids
}
demo_student_id = str(st.session_state.get("demo_student_id", ""))
student_index = student_ids.index(demo_student_id) if demo_student_id in student_ids else 0
with st.container(border=True, key="staff_integrated_review_student_selector_bar"):
    selector_column, status_column = st.columns(
        [4, 1],
        gap="large",
        vertical_alignment="bottom",
    )
    selected_student_id = selector_column.selectbox(
        "확인할 학생",
        student_ids,
        index=student_index,
        key="integrated_review_student_id",
        format_func=lambda student_id: student_option_labels[student_id],
        help="학번을 입력하면 해당 학생을 빠르게 찾을 수 있습니다.",
    )
    selected_rows = latest_all[latest_all["student_id"] == selected_student_id]
    if selected_rows.empty:
        st.warning(
            "선택한 학생의 최신 위험분석 데이터가 없어 "
            "추천을 검토할 수 없습니다."
        )
        render_footer()
        st.stop()
    latest = selected_rows.iloc[0]
    status_column.metric("현재 위험등급", str(latest["risk_level"]))
st.session_state["demo_student_id"] = selected_student_id

completed_ids = set(
    completed_courses[
        (completed_courses["student_id"].astype(str) == selected_student_id)
        & (completed_courses["completion_status"] == "completed")
    ]["course_id"].astype(str)
)

active_section = render_submenu(
    ("학생 · AI 이해", "비교과 추천", "교과 학습경로", "최종 검토", "학생 피드백"),
    key="integrated_recommendation_submenu",
)
info_columns = st.columns(4)
info_columns[0].metric("학생 ID", selected_student_id)
info_columns[1].metric("학과", latest["department"])
info_columns[2].metric("학년", f"{int(latest['grade'])}학년")
info_columns[3].metric("주요 위험유형", latest["primary_risk_type"])

intervention_service = get_intervention_service()
staff_context = StudentViewQueryService().get_staff_student_context(selected_student_id)
submitted_checkin = staff_context.latest_checkin
stored_intervention = staff_context.intervention
stored_program_rows = staff_context.recommendations
stored_path = staff_context.learning_path
if not stored_program_rows.empty:
    stored_program_rows = stored_program_rows[
        stored_program_rows["recommendation_type"] == "support_program"
    ].copy()

analysis: dict[str, object] | None = None
provider_name = ""
recommendation_rows: list[dict[str, object]] = []
recommendation_ids: dict[str, int] = {}
intervention_id: int | None = None
generation_version = 1
result_origin = ""
program_result_key = f"ai_intervention_result_{selected_student_id}"
stale_program_ids: list[str] = []

if (
    submitted_checkin is not None
    and submitted_checkin.get("analysis_id") is not None
    and stored_intervention is not None
    and not stored_program_rows.empty
):
    analysis = _analysis_from_checkin(submitted_checkin)
    provider_name = str(submitted_checkin["provider_name"])
    intervention_id = int(stored_intervention["intervention_id"])
    generation_version = int(stored_intervention.get("generation_version") or 1)
    recommendation_ids = {
        str(row["target_id"]): int(row["recommendation_id"])
        for row in stored_program_rows.to_dict(orient="records")
    }
    program_master = support_programs.set_index("program_id").to_dict(orient="index")
    for row in stored_program_rows.sort_values("rank").to_dict(orient="records"):
        program_id = str(row["target_id"])
        program = program_master.get(program_id)
        if program is None:
            continue
        recommendation_rows.append(
            {
                "program_id": program_id,
                "program_name": str(program["program_name"]),
                "program_type": str(program["program_type"]),
                "description": str(program["description"]),
                "department_in_charge": str(program["department_in_charge"]),
                "operation_period": str(program["operation_period"]),
                "score": float(row["score"]),
                "reason": str(row["reason"]),
                "rank": int(row["rank"]),
            }
        )
    stale_program_ids = [
        str(program_id)
        for program_id in recommendation_ids
        if str(program_id) not in program_master
    ]
    if stale_program_ids:
        recommendation_rows.clear()
        st.warning(
            "저장된 추천이 이전 비교과 master를 참조하고 있어 현재 "
            "WINGS 비교과 목록으로 갱신이 필요합니다. 기존 체크인·AI 이해·"
            "이력은 삭제하지 않습니다."
        )
        if active_section == "학생 · AI 이해" and st.button(
            "현재 비교과 목록으로 추천 갱신",
            type="primary",
        ):
            recommendation_result = get_program_recommender().recommend(
                profile=latest,
                analysis=analysis,
                programs=support_programs,
            )
            analysis_result = CheckinAnalysisResult(
                analysis=analysis,
                provider_name=provider_name,
                fallback_used=bool(submitted_checkin["fallback_used"]),
            )
            intervention_context = intervention_service.record_recommendation_batch(
                student_id=selected_student_id,
                recommendations=recommendation_result.recommendations,
                risk_snapshot=latest.to_dict(),
                source_checkin_id=int(submitted_checkin["checkin_id"]),
                initial_status="교직원 검토 대기",
                create_new_version=True,
            )
            st.session_state[program_result_key] = (
                analysis_result,
                recommendation_result,
                intervention_context,
            )
            st.rerun()
    else:
        result_origin = "학생 제출 시 저장된 결과"
        st.success(
            f"체크인 #{int(submitted_checkin['checkin_id'])}의 AI 이해·비교과 추천을 "
            "불러왔습니다. AI를 다시 호출하거나 추천을 중복 생성하지 않았습니다."
        )
else:
    if active_section == "학생 · AI 이해":
        st.caption(
            "저장된 학생 추천 결과가 없는 기존 가상 체크인에만 "
            "교직원이 분석을 생성할 수 있습니다."
        )
    generate_clicked = active_section == "학생 · AI 이해" and st.button(
        "AI 이해 및 비교과 추천 생성", type="primary"
    )
    if generate_clicked:
        with st.spinner("자유서술을 구조화하고 지원프로그램을 검색하고 있습니다..."):
            if submitted_checkin is not None and submitted_checkin.get("analysis_id") is not None:
                analysis_result = CheckinAnalysisResult(
                    analysis=_analysis_from_checkin(submitted_checkin),
                    provider_name=str(submitted_checkin["provider_name"]),
                    fallback_used=bool(submitted_checkin["fallback_used"]),
                )
            else:
                provider = create_ai_provider(api_key=resolve_gemini_api_key())
                analysis_result = CheckinAnalysisService(provider).analyze(
                    str(latest["natural_language_concern"])
                )
            recommendation_result = get_program_recommender().recommend(
                profile=latest,
                analysis=analysis_result.analysis,
                programs=support_programs,
            )
            source_checkin_id = (
                int(submitted_checkin["checkin_id"]) if submitted_checkin is not None else None
            )
            intervention_context = intervention_service.record_recommendation_batch(
                student_id=selected_student_id,
                recommendations=recommendation_result.recommendations,
                risk_snapshot=latest.to_dict(),
                source_checkin_id=source_checkin_id,
                initial_status=(
                    "교직원 검토 대기" if source_checkin_id is not None else "교직원 검토"
                ),
            )
            st.session_state[program_result_key] = (
                analysis_result,
                recommendation_result,
                intervention_context,
            )
            queue_ai_reveal(
                f"staff_analysis_{intervention_context.intervention_id}_{intervention_context.generation_version}"
            )

    generated_result = st.session_state.get(program_result_key)
    if generated_result is not None:
        analysis_result: CheckinAnalysisResult
        recommendation_result: RecommendationResult
        intervention_context: InterventionContext
        analysis_result, recommendation_result, intervention_context = generated_result
        analysis = analysis_result.analysis
        provider_name = analysis_result.provider_name
        intervention_id = intervention_context.intervention_id
        generation_version = intervention_context.generation_version
        recommendation_ids = intervention_context.recommendation_ids
        recommendation_rows = [
            {
                "program_id": item.program_id,
                "program_name": item.program_name,
                "program_type": item.program_type,
                "description": item.description,
                "department_in_charge": item.department_in_charge,
                "operation_period": item.operation_period,
                "score": item.score,
                "reason": item.reason,
                "rank": rank,
            }
            for rank, item in enumerate(recommendation_result.recommendations, start=1)
        ]
        result_origin = "교직원 생성 결과"
        if analysis_result.warning:
            st.warning(analysis_result.warning)
        if recommendation_result.warning:
            st.warning(recommendation_result.warning)

course_result_key = f"learning_path_result_{selected_student_id}"
manual_course_result: LearningPathResult | None = st.session_state.get(course_result_key)

if analysis is not None and intervention_id is not None and recommendation_rows:
    st.caption(
        f"{result_origin} · 분석 provider: {provider_name} · 추천 버전: {generation_version}"
    )
    if active_section == "학생 · AI 이해":
        st.subheader("학생 체크인과 AI 이해")
        st.info(str(latest["natural_language_concern"]))
        concern_columns = st.columns(3)
        concern_columns[0].metric("전공 고민", "확인" if analysis["major_concern"] else "명시 없음")
        concern_columns[1].metric(
            "학습 어려움", "확인" if analysis["learning_difficulty"] else "명시 없음"
        )
        concern_columns[2].metric(
            "진로 불확실성", "확인" if analysis["career_uncertainty"] else "명시 없음"
        )
        render_ai_reveal(
            f"**AI 요약:** {analysis['summary']}",
            key=f"staff_analysis_{intervention_id}_{generation_version}",
            label=f"검증 완료 · {provider_name} AI 요약",
        )
        if submitted_checkin is not None and semantic_coverage(submitted_checkin.get("semantic_states")):
            with st.expander("학생이 확인한 9개 대화 맥락"):
                for line in semantic_review_lines(submitted_checkin.get("semantic_states")):
                    st.markdown(f"- {line}")
        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.markdown("**관심분야**")
            st.write(", ".join(analysis["interests"]) or "명시된 관심분야 없음")
            st.markdown("**희망직무**")
            st.write(", ".join(analysis["desired_jobs"]) or "명시된 희망직무 없음")
        with detail_right:
            st.markdown("**확인된 지원수요**")
            for support_need in analysis["support_needs"]:
                st.markdown(f"- {support_need}")

    elif active_section == "비교과 추천":
        st.subheader("비교과 추천 Top 3")
        program_weights = get_program_recommender().config["recommendation"]["weights"]
        st.caption(
            "선정 방식: 등록된 전체 비교과를 위험영역 일치 "
            f"{float(program_weights['risk_type_match']):.0%} · 학생 서술 유사도 "
            f"{float(program_weights['semantic_similarity']):.0%} · 체크인 지원수요 일치 "
            f"{float(program_weights['support_need_match']):.0%}로 점수화한 상위 3개입니다."
        )
        visible_program_ids = {
            str(recommendation["program_id"])
            for recommendation in recommendation_rows
        }
        visible_programs = support_programs[
            support_programs["program_id"].astype(str).isin(visible_program_ids)
        ]
        current_reason_by_id = get_program_recommender().explain_program_matches(
            profile=latest,
            analysis=analysis,
            programs=visible_programs,
        )
        program_application_urls = support_programs.set_index("program_id").get(
            "application_url", pd.Series(dtype=str)
        ).to_dict()
        for recommendation in recommendation_rows:
            with st.container(border=True):
                title_column, score_column = st.columns([4, 1])
                title_column.markdown(f"### {recommendation['rank']}. {recommendation['program_name']}")
                score_column.metric("추천점수", f"{float(recommendation['score']):.1f}")
                st.write(recommendation["description"])
                program_id = str(recommendation["program_id"])
                _render_reason_list(
                    "비교과 추천 이유",
                    current_reason_by_id.get(program_id, recommendation["reason"]),
                )
                st.caption(
                    f"유형: {recommendation['program_type']} · "
                    f"담당: {recommendation['department_in_charge']} · "
                    f"운영: {recommendation['operation_period']} · ID: {recommendation['program_id']}"
                )
                application_url = str(
                    program_application_urls.get(str(recommendation["program_id"]), "")
                ).strip()
                if application_url:
                    st.link_button("WINGS에서 세부 조건 확인", application_url)

    elif active_section == "교과 학습경로":
        st.subheader("교과 학습경로")
        st.warning(
            "이 결과는 공식 마이크로디그리가 아니라 교직원 검토용 개인 맞춤 학습경로입니다."
        )
        st.caption(
            "선정 방식: 등록 교과목에서 학년·개설 여부·선수과목·기이수 과목을 먼저 "
            "검증한 뒤, 학생 서술·관심분야·희망직무와의 연결성, 소속 학과 우선순위와 "
            "역량 다양성을 반영해 3~4개를 구성합니다. 타과 과목의 실제 수강 가능 여부는 "
            "교직원이 최종 확인합니다."
        )
        if stored_path.empty and manual_course_result is None:
            st.caption(
                "저장된 경로가 없을 때만 이수과목·선수과목·학년·개설 여부를 "
                "반영해 교과목 3~4개를 생성합니다."
            )
            if st.button("교과 학습경로 생성", type="primary"):
                with st.spinner("DB 교과목을 검색하고 경로 설명을 생성하고 있습니다..."):
                    try:
                        provider = create_ai_provider(api_key=resolve_gemini_api_key())
                        if submitted_checkin is not None:
                            repository = get_default_repository()
                            source = repository.get_all()
                            path_service = StudentViewService(
                                repository=repository,
                                ai_provider=provider,
                                course_recommender=get_course_recommender(),
                            )
                            latest_profile = latest.to_dict()
                            latest_profile["student_checkin_id"] = int(submitted_checkin["checkin_id"])
                            latest_profile["interest_fields"] = "|".join(
                                submitted_checkin.get("interest_fields", [])
                            )
                            latest_profile["desired_job"] = str(submitted_checkin.get("desired_job", ""))
                            latest_profile["natural_language_concern"] = str(
                                submitted_checkin.get("natural_language_concern", "")
                            )
                            result, _ = path_service.build_and_store_learning_path(
                                selected_student_id,
                                latest_profile,
                                analysis,
                                source,
                                generation_version=generation_version,
                                create_new_version=generation_version > 1,
                            )
                            st.session_state[course_result_key] = result
                            queue_ai_reveal(
                                f"staff_learning_path_{selected_student_id}_{int(submitted_checkin['checkin_id'])}"
                            )
                            st.rerun()
                        selection = get_course_recommender().recommend(
                            profile=latest,
                            analysis=analysis,
                            courses=courses,
                            completed_course_ids=completed_ids,
                        )
                        result = LearningPathBuilder(provider).build(latest, selection)
                        st.session_state[course_result_key] = result
                        manual_course_result = result
                        queue_ai_reveal(f"staff_learning_path_{selected_student_id}_manual")
                    except Exception as error:
                        st.error(f"학습경로를 생성하지 못했습니다: {error}")

        if not stored_path.empty:
            path = stored_path.iloc[0]
            review_status = str(path.get("review_status") or "검토 대기")
            st.caption(
                f"체크인 #{int(path['source_checkin_id'])} 저장 결과 · "
                f"추천 버전: v{int(path.get('generation_version') or 1)} · "
                f"설명 provider: {path.get('provider_name') or '기록 없음'} · "
                f"검토 상태: {review_status}"
            )
            st.markdown(f"### {path['path_name']}")
            career_connection = str(path.get("career_connection") or "").strip()
            related_job = str(path.get("related_job") or "").strip()
            connection_text = (
                f"**희망직무 연결:** {career_connection}"
                if career_connection
                else f"**연결 희망직무:** {related_job}"
                if related_job
                else ""
            )
            render_ai_reveal(
                f"{path['path_reason']}\n\n{connection_text}".strip(),
                key=f"staff_learning_path_{selected_student_id}_{int(path['source_checkin_id'])}",
                label=f"검증 완료 · {path.get('provider_name') or 'AI'} 학습경로 설명",
            )
            if path["competencies"]:
                st.markdown(
                    "**습득 역량:** "
                    + " · ".join(str(item) for item in path["competencies"])
                )
            else:
                st.info(
                    "원본 교과목에 역량 태그가 없어 과목명과 실제 개설정보로 "
                    "학습경로를 설명합니다."
                )
            _render_course_cards(stored_path, courses, completed_ids, str(latest["department"]))
        elif manual_course_result is not None:
            result = manual_course_result
            if result.selection.warning:
                st.warning(result.selection.warning)
            if result.warning:
                st.warning(result.warning)
            st.caption(
                f"설명 provider: {result.provider_name} · 유사도 backend: {result.selection.similarity_backend}"
            )
            st.markdown(f"### {result.explanation['path_name']}")
            render_ai_reveal(
                f"{result.explanation['reason']}\n\n**희망직무 연결:** {result.explanation['career_connection']}",
                key=f"staff_learning_path_{selected_student_id}_manual",
                label=f"검증 완료 · {result.provider_name} 학습경로 설명",
            )
            if result.explanation["competencies"]:
                st.markdown(
                    "**습득 역량:** "
                    + " · ".join(result.explanation["competencies"])
                )
            else:
                st.info(
                    "원본 교과목에 역량 태그가 없어 과목명과 실제 개설정보로 "
                    "학습경로를 설명합니다."
                )
            _render_generated_course_cards(result, courses, completed_ids)
            st.info(
                "이 경로는 학생 제출 체크인과 연결되지 않은 수동 조회 결과입니다. "
                "최종 검토 결과는 비교과 추천에만 기록됩니다."
            )

    elif active_section == "최종 검토":
        st.subheader("추천 최종 검토")
        review_notice_key = f"integrated_review_notice_{selected_student_id}"
        review_notice = st.session_state.pop(review_notice_key, None)
        if review_notice:
            notice_type = str(review_notice.get("type") or "success")
            notice_message = str(review_notice.get("message") or "")
            if notice_type == "warning":
                st.warning(notice_message)
            else:
                st.success(notice_message)
        readiness_columns = st.columns(3)
        readiness_columns[0].metric("비교과 추천", f"{len(recommendation_rows)}개")
        readiness_columns[1].metric(
            "교과 학습경로",
            f"{stored_path['course_id'].nunique()}개" if not stored_path.empty else "미생성",
        )
        current_intervention = intervention_service.get_intervention(intervention_id)
        current_status = (
            str(current_intervention["status"]) if current_intervention is not None else "기록 확인 필요"
        )
        current_staff_action = (
            str(current_intervention.get("staff_action") or "")
            if current_intervention is not None
            else ""
        )
        current_path_status = (
            str(stored_path.iloc[0].get("review_status") or "검토 대기")
            if not stored_path.empty
            else "경로 없음"
        )
        program_decision_by_action = {
            "추천 검토 승인": "승인",
            "추천 조정 필요": "조정 필요",
            "보류": "보류",
        }
        current_program_decision = program_decision_by_action.get(
            current_staff_action,
            "판단 전",
        )
        current_course_decision = (
            current_path_status
            if current_path_status in {"승인", "조정 필요", "보류"}
            else "판단 전"
        )
        readiness_columns[2].metric("추천 버전", f"v{generation_version}")
        st.caption(
            f"현재 검토 · 비교과 {current_program_decision} · "
            f"교과 {current_course_decision} · 지원 진행 {current_status}"
        )
        if stored_path.empty:
            st.info(
                "이 체크인에 저장된 교과 학습경로가 없습니다. "
                "현재 검토 결과는 비교과 추천에만 저장됩니다."
            )

        decision_options = ("판단 전", "승인", "조정 필요", "보류")
        program_note_default = (
            str(current_intervention.get("staff_note") or "")
            if current_intervention is not None
            else ""
        )
        path_note_default = (
            str(stored_path.iloc[0].get("review_note") or "")
            if not stored_path.empty
            else ""
        )
        course_judgments: dict[str, str] = {}
        with st.form(
            f"separate_review_form_{selected_student_id}_{intervention_id}_{generation_version}"
        ):
            program_column, course_column = st.columns(2, gap="large")
            with program_column:
                st.markdown("#### 비교과 추천")
                st.caption(
                    f"추천 {len(recommendation_rows)}개를 확인하고 연결 여부를 판단합니다."
                )
                program_decision = st.radio(
                    "비교과 판단",
                    decision_options,
                    index=decision_options.index(current_program_decision),
                    horizontal=True,
                    key=f"program_decision_{selected_student_id}_{generation_version}",
                )
                program_note = st.text_area(
                    "비교과 검토 메모",
                    value=program_note_default,
                    placeholder="연결 조건이나 교체할 프로그램과 이유를 남겨주세요.",
                    key=f"program_review_note_{selected_student_id}_{generation_version}",
                )
            with course_column:
                st.markdown("#### 교과 학습경로")
                if stored_path.empty:
                    course_decision = None
                    course_note = ""
                    st.caption("저장된 교과 학습경로가 없습니다.")
                else:
                    st.caption(
                        "각 과목의 연결성을 먼저 판정한 뒤 경로 전체를 판단합니다."
                    )
                    course_decision = st.radio(
                        "교과 판단",
                        decision_options,
                        index=decision_options.index(current_course_decision),
                        horizontal=True,
                        key=f"course_decision_{selected_student_id}_{generation_version}",
                    )
                    course_names = (
                        courses.set_index("course_id")["course_name"]
                        .astype(str)
                        .to_dict()
                    )
                    path_courses = stored_path.sort_values("sequence").drop_duplicates(
                        "course_id"
                    )
                    for path_course in path_courses.to_dict(orient="records"):
                        course_id = str(path_course["course_id"])
                        stored_judgment = str(
                            path_course.get("review_judgment") or "판단 전"
                        )
                        if stored_judgment not in COURSE_REVIEW_JUDGMENTS:
                            stored_judgment = "판단 전"
                        course_judgments[course_id] = st.selectbox(
                            f"{course_names.get(course_id, course_id)} 적절성",
                            COURSE_REVIEW_JUDGMENTS,
                            index=COURSE_REVIEW_JUDGMENTS.index(stored_judgment),
                            key=(
                                f"course_judgment_{selected_student_id}_"
                                f"{generation_version}_{course_id}"
                            ),
                        )
                    course_note = st.text_area(
                        "교과 검토 메모",
                        value=path_note_default,
                        placeholder="수강 가능 여부나 교체할 과목과 이유를 남겨주세요.",
                        key=f"course_review_note_{selected_student_id}_{generation_version}",
                    )
            save_column, hold_column = st.columns(2)
            save_review = save_column.form_submit_button(
                "검토 결과 저장",
                type="primary",
                use_container_width=True,
            )
            hold_all = hold_column.form_submit_button(
                "전체 보류",
                use_container_width=True,
            )

        if save_review or hold_all:
            selected_program_decision = "보류" if hold_all else program_decision
            selected_course_decision = (
                "보류" if hold_all and course_decision is not None else course_decision
            )
            if selected_program_decision == "판단 전":
                st.error("비교과 추천의 검토 판단을 선택해 주세요.")
            elif selected_course_decision == "판단 전":
                st.error("교과 학습경로의 검토 판단을 선택해 주세요.")
            else:
                try:
                    path_updated = intervention_service.apply_separate_review_actions(
                        intervention_id=intervention_id,
                        program_decision=selected_program_decision,
                        program_note=program_note,
                        course_decision=selected_course_decision,
                        course_note=course_note,
                        course_judgments=course_judgments,
                    )
                    included_scope = (
                        "비교과와 교과의 독립 검토" if path_updated else "비교과 검토"
                    )
                    st.session_state[review_notice_key] = {
                        "type": "success",
                        "message": f"{included_scope} 결과를 저장했습니다.",
                    }
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

        if (
            (
                current_staff_action == "추천 조정 필요"
                or current_path_status == "조정 필요"
            )
            and submitted_checkin is not None
        ):
            st.divider()
            st.markdown("#### 추천 다시 만들기")
            st.caption(
                "교체할 현재 추천을 직접 선택하면 해당 항목을 제외하고 "
                f"현재 master에서 v{generation_version + 1}을 다시 계산합니다. "
                f"기존 v{generation_version} 결과와 검토 메모는 이력으로 보존됩니다."
            )
            excluded_program_ids: list[str] = []
            if current_staff_action == "추천 조정 필요":
                program_names = {
                    str(item["program_id"]): str(item["program_name"])
                    for item in recommendation_rows
                }
                excluded_program_ids = st.multiselect(
                    "교체할 비교과 추천",
                    options=list(program_names),
                    format_func=lambda program_id: program_names[program_id],
                    placeholder="기존 추천 중 제외할 항목 선택",
                    key=f"excluded_programs_{selected_student_id}_{generation_version}",
                )
            else:
                st.caption("비교과는 조정 요청이 없어 현재 추천을 유지합니다.")
            excluded_course_ids: list[str] = []
            if not stored_path.empty and current_path_status == "조정 필요":
                course_names = courses.set_index("course_id")["course_name"].astype(str).to_dict()
                path_course_ids = list(
                    dict.fromkeys(stored_path["course_id"].astype(str).tolist())
                )
                default_excluded_course_ids = list(
                    dict.fromkeys(
                        stored_path[
                            stored_path["review_judgment"] == "부적절"
                        ]["course_id"].astype(str).tolist()
                    )
                )
                excluded_course_ids = st.multiselect(
                    "교체할 교과목",
                    options=path_course_ids,
                    default=default_excluded_course_ids,
                    format_func=lambda course_id: course_names.get(course_id, course_id),
                    placeholder="기존 학습경로 중 제외할 교과목 선택",
                    key=f"excluded_courses_{selected_student_id}_{generation_version}",
                )
            elif stored_path.empty:
                st.caption("저장된 교과 학습경로가 없어 비교과만 다시 만듭니다.")
            else:
                st.caption("교과는 조정 요청이 없어 현재 학습경로를 유지합니다.")
            st.caption(
                "검토 메모는 감사 이력으로만 보존하며 AI 명령으로 해석하지 않습니다. "
                "후보 선정은 제외 항목을 뺀 실제 DB 데이터로 다시 수행합니다."
            )
            regenerate_clicked = st.button(
                "선택 항목 제외하고 추천 다시 만들기",
                type="primary",
                disabled=not (excluded_program_ids or excluded_course_ids),
                key=f"regenerate_recommendations_{selected_student_id}_{generation_version}",
            )
            if regenerate_clicked:
                with st.spinner("실제 DB 후보를 다시 검색하고 새 버전을 저장하고 있습니다..."):
                    try:
                        repository = get_default_repository()
                        source = repository.get_all()
                        provider = create_ai_provider(api_key=resolve_gemini_api_key())
                        regeneration_service = StudentViewService(
                            repository=repository,
                            ai_provider=provider,
                            program_recommender=get_program_recommender(),
                            course_recommender=get_course_recommender(),
                        )
                        latest_profile = latest.to_dict()
                        latest_profile["student_checkin_id"] = int(
                            submitted_checkin["checkin_id"]
                        )
                        latest_profile["interest_fields"] = "|".join(
                            submitted_checkin.get("interest_fields", [])
                        )
                        latest_profile["desired_job"] = str(
                            submitted_checkin.get("desired_job", "")
                        )
                        latest_profile["natural_language_concern"] = str(
                            submitted_checkin.get("natural_language_concern", "")
                        )
                        regeneration = regeneration_service.regenerate_recommendations(
                            selected_student_id,
                            latest_profile,
                            analysis,
                            source,
                            excluded_program_ids=excluded_program_ids,
                            excluded_course_ids=excluded_course_ids,
                        )
                        st.session_state.pop(program_result_key, None)
                        st.session_state.pop(course_result_key, None)
                        if regeneration.learning_path_error:
                            notice_type = "warning"
                            notice_message = (
                                f"v{regeneration.intervention_context.generation_version} "
                                "비교과 추천은 저장했지만 교과 학습경로는 "
                                f"생성하지 못했습니다: {regeneration.learning_path_error}"
                            )
                        else:
                            notice_type = "success"
                            preserved_scopes = []
                            if regeneration.program_recommendations_preserved:
                                preserved_scopes.append("비교과")
                            if regeneration.learning_path_preserved:
                                preserved_scopes.append("교과")
                            preserved_text = (
                                f" {'·'.join(preserved_scopes)} 결과는 그대로 유지했습니다."
                                if preserved_scopes
                                else ""
                            )
                            notice_message = (
                                f"v{regeneration.intervention_context.generation_version} 추천을 "
                                f"저장했습니다.{preserved_text} 조정한 영역은 다시 검토해 주세요."
                            )
                        st.session_state[review_notice_key] = {
                            "type": notice_type,
                            "message": notice_message,
                        }
                        st.rerun()
                    except Exception as error:
                        st.error(f"추천을 다시 만들지 못했습니다: {error}")
        st.caption(
            "비교과와 교과의 판단은 독립적으로 저장됩니다. 실제 지원할 비교과 "
            "프로그램·담당자·연락 예정일은 `학생지원 진행 관리`에서 확정합니다. "
            "교과목은 실제 개설·수강 가능 여부를 학사정보로 최종 확인해야 합니다."
        )
        try:
            st.page_link("pages/05_intervention_management.py", label="지원계획 확정으로 이동", icon="📋")
        except KeyError:
            st.caption("왼쪽 메뉴의 `학생지원 진행 관리`에서 지원계획을 확정하세요.")

    else:
        st.subheader("학생 추천 피드백 기록")
        recommendation_names = {
            str(item["program_id"]): str(item["program_name"]) for item in recommendation_rows
        }
        with st.form(f"feedback_form_{selected_student_id}_{intervention_id}"):
            feedback_program_id = st.selectbox(
                "피드백 대상 프로그램",
                list(recommendation_names),
                format_func=lambda program_id: recommendation_names[program_id],
            )
            feedback_type = st.selectbox("학생 반응", intervention_service.feedback_types)
            feedback_comment = st.text_area(
                "추가 의견 (선택)",
                placeholder="학생이 말한 선호나 추천이 맞지 않은 이유를 기록하세요.",
            )
            feedback_submitted = st.form_submit_button("피드백 저장")
        if feedback_submitted:
            intervention_service.record_feedback(
                student_id=selected_student_id,
                recommendation_id=recommendation_ids[feedback_program_id],
                feedback_type=feedback_type,
                comment=feedback_comment,
            )
            st.success("학생 피드백을 비교과 추천에 연결해 저장했습니다.")
elif stale_program_ids:
    st.info("위 갱신 버튼으로 현재 비교과 master와 맞는 추천 버전을 만드세요.")
elif submitted_checkin is not None:
    st.warning(
        "학생 체크인은 저장되어 있지만 연결된 AI 이해·비교과 추천이 없습니다. "
        "생성 버튼으로 누락 결과를 한 번만 복구할 수 있습니다."
    )
else:
    st.info("기존 가상 체크인을 분석하려면 위 생성 버튼을 누르세요.")

st.divider()
st.caption(
    "AI는 학생의 말을 해석하고 추천 이유를 설명합니다. 후보는 Repository에 존재하는 "
    "비교과 프로그램·교과목만 사용하며, 최종 지원과 수강 가능 여부는 교직원이 검토합니다."
)
render_footer()
