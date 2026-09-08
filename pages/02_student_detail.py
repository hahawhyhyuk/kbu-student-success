"""선택한 학생의 위험 추세와 설명 가능한 위험 원인을 보여준다."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.checkin_semantic_state import semantic_coverage, semantic_review_lines
from src.intervention_service import InterventionManagementService
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
)
from src.student_view_service import StudentViewQueryService
from src.ui import render_footer, render_page_header, render_submenu
from src.utils import generate_risk_reasons


DOMAIN_LABELS = {
    "attendance_risk": "출결",
    "engagement_risk": "학습참여",
    "achievement_risk": "학업성취",
    "major_adaptation_risk": "전공적응",
    "career_risk": "진로설계",
}


@st.cache_data
def load_student_detail_data(
    checkin_revision: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """학생 정적정보와 계산된 주차별 위험 스냅샷을 캐시한다."""

    service = create_integrated_risk_service(get_default_repository())
    return service.load_students_and_snapshots()


@st.cache_resource
def get_intervention_service() -> InterventionManagementService:
    """학생별 SQLite 개입 이력을 조회할 서비스를 재사용한다."""

    return InterventionManagementService()


@st.cache_resource
def get_student_view_query_service() -> StudentViewQueryService:
    """학생용 화면에서 저장한 체크인과 피드백 조회 서비스를 재사용한다."""

    return StudentViewQueryService()


st.set_page_config(page_title="학생 종합 확인", page_icon="🔎", layout="wide")
render_page_header(
    "학생 종합 확인",
    "학생별 위험 신호와 지원 이력을 확인합니다.",
)

students, snapshots = load_student_detail_data(get_student_checkin_revision())
students = students.copy()
snapshots = snapshots.copy()
students["student_id"] = students["student_id"].astype(str)
snapshots["student_id"] = snapshots["student_id"].astype(str)
student_ids = students["student_id"].sort_values().tolist()
student_lookup = students.set_index("student_id")
student_option_labels = {
    student_id: (
        f"{student_id} · {student_lookup.loc[student_id, 'department']} · "
        f"{int(student_lookup.loc[student_id, 'grade'])}학년"
    )
    for student_id in student_ids
}
demo_student_id = str(st.session_state.get("demo_student_id", ""))
student_index = (
    student_ids.index(demo_student_id) if demo_student_id in student_ids else 0
)
with st.container(border=True, key="staff_student_selector_bar"):
    selector_column, status_column = st.columns(
        [4, 1],
        gap="large",
        vertical_alignment="bottom",
    )
    selected_student_id = selector_column.selectbox(
        "확인할 학생",
        student_ids,
        index=student_index,
        key="student_detail_student_id",
        format_func=lambda student_id: student_option_labels[student_id],
        help="학번을 입력하면 해당 학생을 빠르게 찾을 수 있습니다.",
    )
    student = student_lookup.loc[selected_student_id]
    student_history = snapshots[
        snapshots["student_id"] == selected_student_id
    ].sort_values("week")
    latest = student_history.iloc[-1]
    status_column.metric(
        "현재 위험등급",
        str(latest["risk_level"]),
        help=f"{int(latest['week'])}주차 기준",
    )
st.session_state["demo_student_id"] = selected_student_id

active_section = render_submenu(
    (
        "학생 기본정보",
        "최신 초기경보",
        "주차별 위험도 변화",
        "영역별 최신 위험도",
        "위험 원인 설명",
        "최근 체크인",
        "학생 제출 · 피드백",
        "과거 지원 이력",
    ),
    key="student_detail_submenu",
)
if latest.get("checkin_source") == "student_submission":
    st.caption("최신 위험도에는 학생이 직접 제출한 최근 체크인이 반영되었습니다.")

if active_section == "학생 기본정보":
    st.subheader("학생 기본정보")
    info_columns = st.columns(4)
    info_columns[0].metric("학생 ID", selected_student_id)
    info_columns[1].metric("학과", student["department"])
    info_columns[2].metric("학년", f"{int(student['grade'])}학년")
    info_columns[3].metric("직전 GPA", f"{float(student['previous_gpa']):.2f}")

elif active_section == "최신 초기경보":
    st.subheader("최신 초기경보")
    risk_columns = st.columns(3)
    risk_columns[0].metric(
        "전체 위험도",
        f"{float(latest['overall_risk']):.1f} / 100",
        delta=f"{float(latest['risk_change']):+.1f} (전주 대비)",
        delta_color="inverse",
    )
    risk_columns[1].metric("위험등급", latest["risk_level"])
    risk_columns[2].metric("주요 위험유형", latest["primary_risk_type"])
    if bool(latest["is_complex"]):
        st.warning(
            "두 개 이상의 영역이 동시에 기준을 넘어 복합위험형으로 분류되었습니다."
        )

elif active_section == "주차별 위험도 변화":
    st.subheader("주차별 위험도 변화")
    trend_figure = px.line(
        student_history,
        x="week",
        y="overall_risk",
        markers=True,
        labels={"week": "주차", "overall_risk": "전체 위험도"},
    )
    trend_figure.update_xaxes(dtick=1)
    trend_figure.update_yaxes(range=[0, 100])
    trend_figure.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(trend_figure, width="stretch")

elif active_section == "영역별 최신 위험도":
    st.subheader("영역별 최신 위험도")
    domain_frame = pd.DataFrame(
        {
            "영역": list(DOMAIN_LABELS.values()),
            "위험도": [float(latest[column]) for column in DOMAIN_LABELS],
        }
    )
    domain_figure = px.bar(
        domain_frame,
        x="위험도",
        y="영역",
        orientation="h",
        text_auto=".1f",
        color="위험도",
        color_continuous_scale="YlOrRd",
        range_x=[0, 100],
    )
    domain_figure.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(domain_figure, width="stretch")

elif active_section == "위험 원인 설명":
    st.subheader("위험 원인 설명")
    for reason in generate_risk_reasons(latest):
        st.markdown(f"- {reason}")

elif active_section == "최근 체크인":
    st.subheader("최근 체크인")
    checkin_columns = st.columns(3)
    checkin_columns[0].metric(
        "전공 흥미", f"{int(latest['major_interest'])} / 5"
    )
    checkin_columns[1].metric(
        "전공 만족", f"{int(latest['major_satisfaction'])} / 5"
    )
    checkin_columns[2].metric(
        "진로 명확도", f"{int(latest['career_clarity'])} / 5"
    )
    st.markdown("**관심분야**")
    st.write(str(latest["interest_fields"]).replace("|", ", "))
    st.markdown("**희망직무**")
    st.write(latest["desired_job"] or "아직 설정하지 않음")
    st.markdown("**학생 자유서술**")
    st.info(latest["natural_language_concern"])

elif active_section == "학생 제출 · 피드백":
    st.subheader("학생 직접 제출 체크인 · 피드백")
    student_context = get_student_view_query_service().get_staff_student_context(
        selected_student_id
    )
    student_submission = student_context.latest_checkin
    if student_submission is None:
        st.caption("학생용 '나의 체크인' 화면에서 직접 제출한 기록이 없습니다.")
    else:
        submission_columns = st.columns(4)
        submission_columns[0].metric(
            "상담 희망",
            "희망함"
            if student_submission["consultation_requested"]
            else "선택 안 함",
        )
        submission_columns[1].metric(
            "도움 필요 인식",
            (
                f"{int(student_submission['consultation_intent'])} / 5"
                if student_submission.get("consultation_intent") is not None
                else "자연어로 작성"
            ),
        )
        submission_columns[2].metric(
            "다른 분야 탐색",
            "희망함"
            if student_submission["explore_other_fields"]
            else "선택 안 함",
        )
        submission_columns[3].metric(
            "제출 시각", str(student_submission["created_at"])[:10]
        )
        st.markdown("**학생 자유서술 고민**")
        st.info(student_submission["natural_language_concern"] or "작성 내용 없음")
        submitted_interests = " · ".join(student_submission["interest_fields"])
        st.write(
            f"관심분야: {submitted_interests or '선택 안 함'} · "
            f"희망직무: {student_submission['desired_job'] or '아직 설정하지 않음'}"
        )
        if semantic_coverage(student_submission.get("semantic_states")):
            with st.expander("Kare가 대화 근거로 이해한 9개 영역"):
                for line in semantic_review_lines(
                    student_submission.get("semantic_states")
                ):
                    st.markdown(f"- {line}")
        if student_submission.get("analysis_id") is not None:
            st.markdown("**AI 구조화 결과**")
            st.write(student_submission["summary"])
            analysis_columns = st.columns(3)
            analysis_columns[0].metric(
                "전공 탐색 신호",
                "확인" if student_submission["major_concern"] else "명시 없음",
            )
            analysis_columns[1].metric(
                "학습 지원 신호",
                "확인"
                if student_submission["analysis_learning_difficulty"]
                else "명시 없음",
            )
            analysis_columns[2].metric(
                "진로 탐색 신호",
                "확인"
                if student_submission["career_uncertainty"]
                else "명시 없음",
            )
            st.caption(
                f"분석 provider: {student_submission['provider_name']} · "
                f"지원수요: {' · '.join(student_submission['support_needs']) or '명시 없음'}"
            )
        if not student_context.analysis_feedback.empty:
            st.markdown("**AI 이해 결과에 대한 학생 의견**")
            st.dataframe(
                student_context.analysis_feedback[
                    ["feedback_type", "comment", "created_at"]
                ].rename(
                    columns={
                        "feedback_type": "학생 응답",
                        "comment": "수정·추가 의견",
                        "created_at": "등록 시각",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
else:
    st.subheader("과거 지원 이력")
    intervention_history = get_intervention_service().list_interventions()
    if intervention_history.empty:
        st.caption("저장된 지원 이력이 없습니다.")
    else:
        student_interventions = intervention_history[
            intervention_history["student_id"] == selected_student_id
        ]
        if student_interventions.empty:
            st.caption("이 학생에게 저장된 지원 이력이 없습니다.")
        else:
            st.dataframe(
                student_interventions[
                    [
                        "intervention_id",
                        "status",
                        "staff_action",
                        "staff_note",
                        "created_at",
                        "updated_at",
                    ]
                ].rename(
                    columns={
                        "intervention_id": "지원 ID",
                        "status": "상태",
                        "staff_action": "교직원 조치",
                        "staff_note": "교직원 메모",
                        "created_at": "생성 시각",
                        "updated_at": "최종 수정",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

st.caption(
    "자유서술 AI 구조화와 비교과·교과 추천은 "
    "'AI 맞춤 추천 통합 검토' 화면에서 확인할 수 있습니다."
)
render_footer()
