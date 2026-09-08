"""학생이 저장된 최신 체크인과 맞춤 결과를 다시 확인하는 화면."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.access_control import scoped_student_id
from src.checkin_semantic_state import semantic_coverage, semantic_review_lines
from src.course_catalog_presenter import format_course_caption
from src.repositories import get_default_repository
from src.student_view_service import (
    StudentViewQueryService,
    describe_student_analysis,
    describe_student_support_status,
)
from src.ui import render_footer, render_page_header


@st.cache_data
def load_mypage_master() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """학생 선택과 저장 결과 표시에 필요한 master 데이터를 Repository에서 로드한다."""

    source = get_default_repository().get_all()
    return source.students, source.support_programs, source.courses


def get_mypage_service() -> StudentViewQueryService:
    """AI를 다시 호출하지 않는 저장 결과 조회 서비스를 만든다."""

    return StudentViewQueryService()


st.set_page_config(page_title="마이페이지", page_icon="👤", layout="wide")
render_page_header(
    "마이페이지",
    "최근 체크인과 저장된 맞춤 지원·학습 경로, 담당자 확인 상태를 다시 볼 수 있어요.",
    audience="학생용",
)

students, support_programs, courses = load_mypage_master()
student_ids = students["student_id"].astype(str).sort_values().tolist()
try:
    authenticated_student_id = scoped_student_id(st.session_state, student_ids)
except PermissionError as error:
    st.error(str(error))
    st.stop()

if authenticated_student_id is not None:
    selected_student_id = authenticated_student_id
else:
    # 개별 페이지 AppTest와 개발 중 단독 실행만 기존 synthetic 선택기를 사용한다.
    demo_student_id = str(st.session_state.get("demo_student_id", ""))
    student_index = (
        student_ids.index(demo_student_id)
        if demo_student_id in student_ids
        else 0
    )
    selected_student_id = st.sidebar.selectbox(
        "학생 모드 데모 ID",
        student_ids,
        index=student_index,
        key="student_mypage_student_id",
    )
st.session_state["demo_student_id"] = selected_student_id
selected_student = students[
    students["student_id"].astype(str) == selected_student_id
].iloc[0]
st.caption(
    f"가상 학생 계정 · {selected_student_id} · {selected_student['department']} · "
    f"{int(selected_student['grade'])}학년"
)

context = get_mypage_service().get_student_mypage_context(selected_student_id)
checkin = context.latest_checkin
if checkin is None:
    st.info(
        "아직 저장된 체크인이 없어요. 왼쪽 학생 서비스의 `나의 체크인`에서 "
        "첫 체크인을 제출하면 결과가 여기에 표시됩니다."
    )
    render_footer()
    st.stop()

intervention_status = (
    str(context.intervention.get("status", ""))
    if context.intervention is not None
    else None
)
student_status = describe_student_support_status(intervention_status)
interest_fields = [str(item) for item in checkin.get("interest_fields", [])]
desired_job = str(checkin.get("desired_job", "")).strip() or "아직 탐색 중"
created_at = str(checkin.get("created_at", ""))[:10].replace("-", ".")

summary_columns = st.columns(3)
summary_columns[0].metric("최근 체크인", created_at or "저장 완료")
summary_columns[1].metric("관심 분야", f"{len(interest_fields)}개")
summary_columns[2].metric("희망 직무", desired_job)
st.info(
    f"현재 맞춤 지원은 **{student_status}** 상태예요. "
    "최종 지원 내용과 일정은 담당자가 확인한 후 안내합니다."
)

st.subheader("최근 체크인")
with st.container(border=True):
    st.markdown(
        "**관심 분야:** "
        + (" · ".join(interest_fields) if interest_fields else "아직 선택하지 않음")
    )
    st.markdown(f"**희망 직무:** {desired_job}")
    st.markdown(
        "**상담 희망:** "
        + ("신청함" if bool(checkin.get("consultation_requested")) else "신청하지 않음")
    )
    concern = str(checkin.get("natural_language_concern", "")).strip()
    st.markdown("**내가 작성한 고민**")
    st.write(concern or "작성한 자유서술 내용이 없습니다.")

    explicit_scores = {
        "전공 흥미": checkin.get("major_interest"),
        "전공 만족": checkin.get("major_satisfaction"),
        "전공 계속 학습 의향": checkin.get("major_continuation_intent"),
        "수업을 따라가는 어려움": checkin.get("learning_difficulty"),
        "진로 명확도": checkin.get("career_clarity"),
        "상담·도움 필요도": checkin.get("consultation_intent"),
    }
    explicit_scores = {
        label: score for label, score in explicit_scores.items() if score is not None
    }
    if explicit_scores:
        with st.expander("대화에서 직접 말한 1~5 값"):
            for label, score in explicit_scores.items():
                st.markdown(f"- {label}: **{int(score)}점**")
    if semantic_coverage(checkin.get("semantic_states")):
        with st.expander("Kare가 대화로 이해한 9개 영역"):
            for line in semantic_review_lines(checkin.get("semantic_states")):
                st.markdown(f"- {line}")

st.subheader("AI가 이해한 나의 고민")
if checkin.get("analysis_id") is None:
    st.info("저장된 AI 분석 결과를 확인하고 있어요.")
else:
    analysis = {
        "major_concern": bool(checkin.get("major_concern")),
        "learning_difficulty": bool(
            checkin.get("analysis_learning_difficulty")
        ),
        "career_uncertainty": bool(checkin.get("career_uncertainty")),
        "interests": list(checkin.get("interests", [])),
    }
    with st.container(border=True):
        for message in describe_student_analysis(analysis):
            st.markdown(f"- {message}")
        st.markdown(f"**요약:** {checkin.get('summary', '')}")
        st.caption("AI의 해석은 담당자가 지원 방향을 검토할 때 함께 확인합니다.")

program_by_id = {
    str(row["program_id"]): row
    for row in support_programs.to_dict(orient="records")
}
st.subheader("나에게 추천된 지원")
if context.recommendations.empty:
    st.info("저장된 지원 추천을 확인하고 있어요.")
else:
    for recommendation in context.recommendations.to_dict(orient="records"):
        program_id = str(recommendation["target_id"])
        program = program_by_id.get(program_id)
        if program is None:
            continue
        with st.container(border=True):
            st.markdown(
                f"### {int(recommendation['rank'])}. {program['program_name']}"
            )
            st.caption(
                f"{program['program_type']} · 담당 {program['department_in_charge']}"
            )
            st.write(program["description"])
            st.markdown(f"**추천 이유:** {recommendation['reason']}")
            application_url = str(program.get("application_url", "")).strip()
            if application_url:
                st.link_button(
                    "WINGS에서 세부 조건 확인",
                    application_url,
                    key=f"mypage_wings_{program_id}",
                )

if context.support_plan is not None and not context.support_plan_items.empty:
    st.subheader("담당자가 확정한 지원계획")
    plan = context.support_plan
    st.success(
        f"담당 **{plan['assigned_staff']}** · 연락 예정 "
        f"**{plan['planned_contact_date']}** · 다음 안내 **{plan['next_action']}**"
    )
    for item in context.support_plan_items.to_dict(orient="records"):
        program = program_by_id.get(str(item["program_id"]))
        if program is None:
            continue
        with st.container(border=True):
            st.markdown(f"### {program['program_name']}")
            st.markdown(f"**현재 상태:** {item['item_status']}")
            st.caption(
                f"{program['program_type']} · 운영 {program['operation_period']} · "
                f"담당 {program['department_in_charge']}"
            )

course_by_id = {
    str(row["course_id"]): row for row in courses.to_dict(orient="records")
}
st.subheader("AI 맞춤형 역량 학습경로")
if context.learning_path.empty:
    st.info(
        "이번 체크인에는 별도 학습 경로가 저장되지 않았어요. 관심 분야를 넓혀보고 싶다면 "
        "새 체크인에서 `다른 분야도 탐색하고 싶어요`를 선택할 수 있어요."
    )
else:
    first_path = context.learning_path.iloc[0]
    st.markdown(f"## {first_path['path_name']}")
    st.write(first_path["path_reason"])
    competencies = [str(item) for item in first_path["competencies"]]
    if competencies:
        st.markdown("**습득 가능한 역량:** " + " · ".join(competencies))
    career_connection = str(first_path.get("career_connection", "")).strip()
    if career_connection:
        st.markdown(f"**희망 직무와의 연결:** {career_connection}")

    for path_course in context.learning_path.to_dict(orient="records"):
        course = course_by_id.get(str(path_course["course_id"]))
        if course is None:
            continue
        with st.container(border=True):
            st.markdown(
                f"### {int(path_course['sequence'])}. {course['course_name']}"
            )
            st.caption(
                format_course_caption(course, str(selected_student["department"]))
            )
            st.markdown(f"**과목별 역할:** {path_course['course_role']}")
            st.markdown(f"**선정 이유:** {path_course['selection_reason']}")

st.divider()
st.caption(
    "마이페이지는 저장된 최신 결과를 다시 보여주며 AI 분석이나 추천을 새로 생성하지 않습니다."
)
render_footer()
