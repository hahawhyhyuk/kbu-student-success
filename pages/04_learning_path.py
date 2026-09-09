"""DB 교과목 3~4개로 AI 맞춤형 역량 학습경로를 제공하는 화면."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.ai import create_ai_provider
from src.checkin_service import CheckinAnalysisService
from src.course_catalog_presenter import format_course_caption
from src.course_recommender import CourseRecommender
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
def load_learning_path_data(
    checkin_revision: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repository = get_default_repository()
    source = repository.get_all()
    snapshots = create_integrated_risk_service(repository).build_snapshots(source)
    return source.students, snapshots, source.courses, source.completed_courses


@st.cache_resource
def get_course_recommender() -> CourseRecommender:
    return CourseRecommender()


def resolve_gemini_api_key() -> str | None:
    environment_key = os.getenv("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        return secret_key or None
    except Exception:
        return None


st.set_page_config(
    page_title="AI 맞춤형 학습 경로 검토", page_icon="🧭", layout="wide"
)
render_page_header(
    "AI 맞춤형 학습 경로 검토",
    "소속 학과를 우선해 개인 맞춤 교과목을 검토합니다.",
    audience="학생 지원·교직원 검토용",
)
with st.expander("학습경로 검토 기준", expanded=False):
    st.warning(
        "이 결과는 공식 마이크로디그리가 아니라 교직원 검토용 개인 맞춤 학습경로입니다."
    )
    st.write("관심분야와 희망직무에 필요한 경우 타과 교과목도 함께 제안합니다.")

students, snapshots, courses, completed_courses = load_learning_path_data(
    get_student_checkin_revision()
)
latest_all = select_latest_student_snapshots(snapshots)
target_ids = latest_all[
    (latest_all["major_adaptation_risk"] >= 60) | (latest_all["career_risk"] >= 60)
]["student_id"].tolist()
student_ids = target_ids + [item for item in sorted(students["student_id"]) if item not in target_ids]
demo_student_id = str(st.session_state.get("demo_student_id", ""))
student_index = (
    student_ids.index(demo_student_id) if demo_student_id in student_ids else 0
)
selected_student_id = st.sidebar.selectbox(
    "학생 선택",
    student_ids,
    index=student_index,
    key="learning_path_student_id",
)
st.session_state["demo_student_id"] = selected_student_id
selected_rows = latest_all[
    latest_all["student_id"].astype(str) == str(selected_student_id)
]
if selected_rows.empty:
    st.warning("선택한 학생의 위험분석 데이터가 없어 학습경로를 검토할 수 없습니다.")
    render_footer()
    st.stop()
latest = selected_rows.iloc[0]
completed_ids = set(
    completed_courses[
        (completed_courses["student_id"] == selected_student_id)
        & (completed_courses["completion_status"] == "completed")
    ]["course_id"].astype(str)
)

active_section = render_submenu(
    ("학생 · 생성 조건", "학습 경로 요약", "권장 학습 순서"),
    key="learning_path_submenu",
)
info_columns = st.columns(5)
info_columns[0].metric("학생 ID", selected_student_id)
info_columns[1].metric("학과", latest["department"])
info_columns[2].metric("학년", f"{int(latest['grade'])}학년")
info_columns[3].metric("관심분야", str(latest["interest_fields"]).replace("|", " · "))
info_columns[4].metric("이수과목", f"{len(completed_ids)}개")
st.markdown(f"**희망직무:** {latest['desired_job'] or '아직 설정하지 않음'}")
if selected_student_id not in target_ids:
    st.info("전공적응·진로설계 우선 대상은 아니지만 교직원 수동 선택으로 경로를 생성할 수 있습니다.")

staff_context = StudentViewQueryService().get_staff_student_context(
    selected_student_id
)
submitted_checkin = staff_context.latest_checkin
stored_path = staff_context.learning_path
result_key = f"learning_path_result_{selected_student_id}"
if not stored_path.empty:
    st.success(
        f"학생 체크인 #{int(stored_path.iloc[0]['source_checkin_id'])}에서 생성된 "
        "학습경로를 불러왔습니다. 교과목을 다시 추천하지 않았습니다."
    )
else:
    st.caption(
        "저장된 학생 학습경로가 없을 때만 새 경로를 생성합니다. "
        "학생이 새 체크인을 제출하면 새 checkin_id로 결과가 구분됩니다."
    )

generate_clicked = False
if active_section == "학생 · 생성 조건" and stored_path.empty:
    generate_clicked = st.button("학습경로 생성", type="primary")
if stored_path.empty and generate_clicked:
    with st.spinner("DB 교과목을 검색하고 학습경로 설명을 생성하고 있습니다..."):
        try:
            provider = create_ai_provider(api_key=resolve_gemini_api_key())
            if (
                submitted_checkin is not None
                and submitted_checkin.get("analysis_id") is not None
            ):
                analysis = {
                    "major_concern": bool(submitted_checkin["major_concern"]),
                    "learning_difficulty": bool(
                        submitted_checkin["analysis_learning_difficulty"]
                    ),
                    "career_uncertainty": bool(
                        submitted_checkin["career_uncertainty"]
                    ),
                    "interests": list(submitted_checkin["interests"]),
                    "desired_jobs": list(submitted_checkin["desired_jobs"]),
                    "support_needs": list(submitted_checkin["support_needs"]),
                    "summary": str(submitted_checkin["summary"]),
                }
            else:
                analysis = CheckinAnalysisService(provider).analyze(
                    str(latest["natural_language_concern"])
                ).analysis

            if submitted_checkin is not None:
                repository = get_default_repository()
                source = repository.get_all()
                path_service = StudentViewService(
                    repository=repository,
                    ai_provider=provider,
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
                path_generation_version = int(
                    (staff_context.intervention or {}).get(
                        "generation_version", 1
                    )
                )
                result, _ = path_service.build_and_store_learning_path(
                    selected_student_id,
                    latest_profile,
                    analysis,
                    source,
                    generation_version=path_generation_version,
                    create_new_version=path_generation_version > 1,
                )
                st.session_state[result_key] = result
                queue_ai_reveal(
                    "staff_learning_path_"
                    f"{selected_student_id}_{int(submitted_checkin['checkin_id'])}"
                )
                st.rerun()
            else:
                selection = get_course_recommender().recommend(
                    profile=latest,
                    analysis=analysis,
                    courses=courses,
                    completed_course_ids=completed_ids,
                )
                result = LearningPathBuilder(provider).build(latest, selection)
                st.session_state[result_key] = result
                queue_ai_reveal(
                    f"staff_learning_path_{selected_student_id}_manual"
                )
        except Exception as error:
            st.error(f"학습경로를 생성하지 못했습니다: {error}")

stored_result: LearningPathResult | None = st.session_state.get(result_key)
if not stored_path.empty:
    path = stored_path.iloc[0]
    st.caption(
        f"학생 제출 시 저장된 결과 · 설명 provider: "
        f"{path.get('provider_name') or '기록 없음'}"
    )
    if active_section == "학생 · 생성 조건":
        st.subheader("학생 · 생성 조건")
        st.success(
            "저장된 학습 경로가 있어 새 교과목 추천을 생성하지 않습니다. "
            "다른 체크인 결과는 새 checkin_id로 구분됩니다."
        )
    elif active_section == "학습 경로 요약":
        st.subheader("학습 경로 요약")
        st.header(str(path["path_name"]))
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
            key=(
                "staff_learning_path_"
                f"{selected_student_id}_{int(path['source_checkin_id'])}"
            ),
            label=f"검증 완료 · {path.get('provider_name') or 'AI'} 학습경로 설명",
        )
        competencies = path["competencies"]
        if competencies:
            st.markdown(
                "**습득 역량:** " + " · ".join(str(item) for item in competencies)
            )
        else:
            st.info(
                "원본 교과목에 역량 태그가 없어 과목명과 실제 개설정보로 "
                "학습경로를 설명합니다."
            )
    else:
        st.subheader("권장 학습 순서")
        course_master = courses.set_index("course_id").to_dict(orient="index")
        for row in stored_path.sort_values("sequence").to_dict(orient="records"):
            course_id = str(row["course_id"])
            course = course_master.get(course_id)
            if course is None or course_id in completed_ids:
                continue
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
                    st.markdown(f"**선정 근거:** {selection_reason}")
                st.caption(format_course_caption(course, str(latest["department"])))
elif stored_result:
    explanation = stored_result.explanation
    selection = stored_result.selection
    if selection.warning:
        st.warning(selection.warning)
    if stored_result.warning:
        st.warning(stored_result.warning)
    st.caption(
        f"설명 provider: {stored_result.provider_name} · 유사도 backend: {selection.similarity_backend}"
    )
    if active_section == "학생 · 생성 조건":
        st.subheader("학생 · 생성 조건")
        st.success("현재 선택 학생의 학습 경로가 생성되었습니다.")
    elif active_section == "학습 경로 요약":
        st.subheader("학습 경로 요약")
        st.header(explanation["path_name"])
        render_ai_reveal(
            (
                f"{explanation['reason']}\n\n"
                f"**희망직무 연결:** {explanation['career_connection']}"
            ),
            key=f"staff_learning_path_{selected_student_id}_manual",
            label=f"검증 완료 · {stored_result.provider_name} 학습경로 설명",
        )
        if explanation["competencies"]:
            st.markdown("**습득 역량:** " + " · ".join(explanation["competencies"]))
        else:
            st.info(
                "원본 교과목에 역량 태그가 없어 과목명과 실제 개설정보로 "
                "학습경로를 설명합니다."
            )
    else:
        role_by_id = {
            item["course_id"]: item for item in explanation["course_roles"]
        }
        st.subheader("권장 학습 순서")
        master_ids = set(courses["course_id"].astype(str))
        for course in ordered_courses_by_sequence(selection, explanation):
            if (
                course.course_id not in master_ids
                or course.course_id in completed_ids
            ):
                continue
            role = role_by_id[course.course_id]
            with st.container(border=True):
                title_column, score_column = st.columns([4, 1])
                title_column.markdown(
                    f"### {role['sequence']}. {course.course_name}"
                )
                score_column.metric("추천점수", f"{course.score:.1f}")
                st.write(course.description)
                st.markdown(f"**과목 역할:** {role['role']}")
                st.markdown(f"**선정 근거:** {course.reason}")
                st.caption(format_course_caption(course))
else:
    if active_section == "학생 · 생성 조건":
        st.subheader("학생 · 생성 조건")
        st.info(
            "저장된 경로가 없는 경우 이수과목·선수과목·학년·개설 여부를 "
            "반영해 학습경로를 생성할 수 있습니다."
        )
    else:
        st.info("먼저 `학생 · 생성 조건` 메뉴에서 학습경로를 생성해 주세요.")

st.divider()
st.caption("최종 수강 가능 여부와 교육과정 적용은 교직원이 실제 학사정보를 확인해 결정합니다.")
render_footer()
