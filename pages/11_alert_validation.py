"""가상 참조 유형과 최신 초기경보 결과의 동작 일치도를 점검한다."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ai import MockAIProvider
from src.alert_validation_service import (
    AlertValidationReport,
    build_synthetic_reference_labels,
    evaluate_initial_alerts,
)
from src.course_recommendation_evaluation_service import (
    CourseRecommendationEvaluationReport,
    course_recommendation_evaluation_catalog,
    evaluate_course_recommendations,
)
from src.course_recommender import CourseRecommender
from src.kare_evaluation_service import (
    KARE_SYNTHETIC_EVALUATION_CASES,
    KareEvaluationReport,
    evaluate_kare_dialogues,
)
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService
from src.streamlit_ai import create_streamlit_ai_provider
from src.ui import render_footer, render_page_header


@st.cache_data
def load_alert_validation() -> tuple[AlertValidationReport, str]:
    """Repository 데이터로 합성 참조 라벨과 초기경보 결과를 비교한다."""

    repository = get_default_repository()
    source = repository.get_all()
    snapshots = RiskAnalysisService(repository).build_snapshots(source)
    reference_labels = build_synthetic_reference_labels(source.students)
    report = evaluate_initial_alerts(snapshots, reference_labels)
    return report, type(repository).__name__


def format_rate(value: float | None) -> str:
    """선택적 백분율을 화면용 문자열로 변환한다."""

    return "해당 없음" if value is None else f"{value:.1f}%"


@st.cache_data
def load_kare_regression() -> KareEvaluationReport:
    """네트워크 없이 15개 synthetic 대화를 Mock Kare로 회귀 점검한다."""

    return evaluate_kare_dialogues(MockAIProvider())


@st.cache_resource(show_spinner=False)
def load_course_recommendation_regression() -> tuple[
    CourseRecommendationEvaluationReport, str
]:
    """현재 Repository 교과 master로 synthetic 추천 관련성을 점검한다."""

    repository = get_default_repository()
    report = evaluate_course_recommendations(
        CourseRecommender(),
        repository.get_courses(),
    )
    return report, type(repository).__name__


st.set_page_config(page_title="초기경보 모델 검증", page_icon="✅", layout="wide")
render_page_header(
    "초기경보 모델 검증",
    "가상학생 생성 유형과 규칙형 위험엔진의 최신 결과를 비교해 내부 동작을 점검합니다.",
)

st.warning(
    "이 화면의 수치는 생성 규칙을 알고 있는 synthetic 데이터의 동작 일치도입니다. "
    "독립적인 실제 학생 데이터의 예측 정확도나 개입 효과로 해석하면 안 됩니다."
)

report, repository_name = load_alert_validation()
metric_columns = st.columns(5)
metric_columns[0].metric("합성 기준 일치율", format_rate(report.agreement_rate))
metric_columns[1].metric(
    "지원 신호 포착률", format_rate(report.support_signal_recall)
)
metric_columns[2].metric(
    "정상 패턴 보존율", format_rate(report.normal_pattern_specificity)
)
metric_columns[3].metric(
    "단일 위험유형 일치율", format_rate(report.primary_type_match_rate)
)
metric_columns[4].metric(
    "복합 패턴 고위험 포착률", format_rate(report.high_risk_recall)
)
st.caption(
    f"검증 학생 {report.student_count}명 · 참조상 지원 필요 "
    f"{report.reference_support_count}명 · 시스템 지원 신호 "
    f"{report.predicted_support_count}명 · 데이터 provider: {repository_name}"
)

left, right = st.columns([0.82, 1.18])
with left:
    st.subheader("지원 신호 비교표")
    confusion_display = report.confusion.rename(
        columns={
            "reference_group": "합성 참조 구분",
            "predicted_normal": "시스템 정상",
            "predicted_support": "시스템 지원 신호",
        }
    )
    st.dataframe(confusion_display, width="stretch", hide_index=True)
    st.caption(
        "지원 신호는 위험등급 ‘관심·주의·고위험’을 합한 값입니다. "
        "실제 운영 임계값은 대학 정책과 검증 결과에 따라 조정해야 합니다."
    )

with right:
    st.subheader("가상학생 유형별 결과")
    breakdown_display = report.segment_breakdown.rename(
        columns={
            "reference_segment": "가상 참조 유형",
            "student_count": "학생 수",
            "expected_primary_risk_type": "기대 주요유형",
            "support_detection_rate": "지원 신호 포착률(%)",
            "high_risk_detection_rate": "고위험 포착률(%)",
            "primary_type_match_rate": "주요유형 일치율(%)",
        }
    )
    st.dataframe(breakdown_display, width="stretch", hide_index=True)

type_mismatches = report.details[
    report.details["expected_primary_risk_type"].notna()
    & ~report.details["primary_type_match"]
]
with st.expander(f"위험유형 불일치 사례 확인 · {len(type_mismatches)}명"):
    if type_mismatches.empty:
        st.success("단일 위험유형의 불일치 사례가 없습니다.")
    else:
        st.caption(
            "여러 신호가 함께 나타나 기대 유형과 시스템 주요유형이 달라진 사례입니다. "
            "오류로 단정하지 않고 원천지표와 함께 검토해야 합니다."
        )
        mismatch_display = type_mismatches[
            [
                "student_id",
                "reference_segment",
                "expected_primary_risk_type",
                "risk_level",
                "primary_risk_type",
            ]
        ].rename(
            columns={
                "student_id": "학생 ID",
                "reference_segment": "가상 참조 유형",
                "expected_primary_risk_type": "기대 주요유형",
                "risk_level": "시스템 위험등급",
                "primary_risk_type": "시스템 주요유형",
            }
        )
        st.dataframe(mismatch_display, width="stretch", hide_index=True)

st.subheader("실제 대학 데이터로 전환할 때 필요한 검증")
real_validation_plan = pd.DataFrame(
    [
        {
            "준비 항목": "독립 참조 라벨",
            "필요 내용": "교직원이 원천자료를 검토해 기록한 지원 필요 여부·주요 위험유형",
        },
        {
            "준비 항목": "평가 대상과 시점",
            "필요 내용": "같은 학기·같은 관찰기간의 학생 ID와 평가 기준일 고정",
        },
        {
            "준비 항목": "정량 지표",
            "필요 내용": "민감도·특이도·정밀도·F1 및 위험유형별 일치율",
        },
        {
            "준비 항목": "오류 검토",
            "필요 내용": "미포착·과다포착 사례를 교직원이 확인하고 임계값 조정 근거 기록",
        },
    ]
)
st.dataframe(real_validation_plan, width="stretch", hide_index=True)
st.info(
    "발표에서는 ‘합성 데이터에서 생성 의도와 위험엔진 동작을 검증했고, 실제 정확도는 "
    "교직원 참조 라벨을 확보한 뒤 같은 구조로 평가한다’고 설명하는 것이 정확합니다."
)

with st.expander("Kare 대화 품질 회귀 검증 · synthetic 15개", expanded=False):
    st.caption(
        "5개 학과의 구체적·모호한·단답형·정정형 발화를 사용합니다. 기본 결과는 "
        "Mock fallback과 서비스 안전장치의 재현성 검증이며 실제 Gemini 정확도가 아닙니다."
    )
    kare_report = load_kare_regression()
    quality_metrics = st.columns(6)
    quality_metrics[0].metric("대화 사례", f"{kare_report.case_count}개")
    quality_metrics[1].metric("전체 조건 통과", f"{kare_report.case_pass_rate:.1f}%")
    quality_metrics[2].metric("의미 상태 일치", f"{kare_report.state_match_rate:.1f}%")
    quality_metrics[3].metric(
        "한 문장 다중 추출",
        f"{kare_report.multi_extraction_recall:.1f}%",
    )
    quality_metrics[4].metric("척도 질문", f"{kare_report.scale_question_count}건")
    quality_metrics[5].metric("반복 주제", f"{kare_report.repeated_focus_count}건")

    quality_display = kare_report.details[
        [
            "case_id",
            "department",
            "tone",
            "turn_count",
            "state_matches",
            "first_turn_matches",
            "unexpected_inference_count",
            "passed",
        ]
    ].rename(
        columns={
            "case_id": "사례",
            "department": "학과",
            "tone": "말투",
            "turn_count": "대화 턴",
            "state_matches": "의미 상태",
            "first_turn_matches": "첫 문장 다중 추출",
            "unexpected_inference_count": "과잉 추론",
            "passed": "통과",
        }
    )
    st.dataframe(quality_display, width="stretch", hide_index=True)

    selected_case_id = st.selectbox(
        "실제 Gemini로 확인할 synthetic 사례",
        [case.case_id for case in KARE_SYNTHETIC_EVALUATION_CASES],
        format_func=lambda case_id: next(
            f"{case.case_id} · {case.title} · {case.department}"
            for case in KARE_SYNTHETIC_EVALUATION_CASES
            if case.case_id == case_id
        ),
        key="kare_live_evaluation_case",
    )
    selected_case = next(
        case
        for case in KARE_SYNTHETIC_EVALUATION_CASES
        if case.case_id == selected_case_id
    )
    st.caption(
        f"선택 사례는 최대 {len(selected_case.messages)}회의 Gemini 요청을 사용합니다. "
        "버튼을 누르기 전에는 외부 API를 호출하지 않습니다."
    )
    if st.button(
        "선택 사례 1건 Gemini 점검",
        key="run_kare_live_evaluation",
        use_container_width=True,
    ):
        with st.spinner("synthetic 대화를 점검하고 있어요..."):
            live_report = evaluate_kare_dialogues(
                create_streamlit_ai_provider(),
                (selected_case,),
            )
        live_result = live_report.results[0]
        if live_result.passed:
            st.success(
                f"{live_result.provider_name} 응답이 설정된 품질 조건을 모두 통과했습니다."
            )
        else:
            st.warning(
                f"{live_result.provider_name} 응답에서 확인할 항목이 있습니다: "
                f"{' · '.join(live_result.mismatches) or live_result.error}"
            )
        for line in live_result.transcript:
            st.markdown(f"- {line}")
    st.caption(
        "실제 운영 정확도는 별도의 학생 동의·교직원 검토 절차로 만든 참조 데이터가 있어야 평가할 수 있습니다."
    )

with st.expander("교과 추천 내부 관련성 검증 · synthetic 10개", expanded=False):
    st.caption(
        "5개 적용 학과의 진로·학습 요구를 두 사례씩 점검합니다. 허용 과목군은 추천 순위에 "
        "사용하지 않고, 추천 후 과목명·공개 강좌 설명의 관련성만 확인합니다."
    )
    evaluation_catalog = course_recommendation_evaluation_catalog().rename(
        columns={
            "case_id": "사례",
            "title": "학습 요구",
            "department": "학과",
            "grade": "학년",
            "interest_fields": "관심 분야",
            "desired_job": "희망 직무",
            "acceptable_families": "사후 확인 과목군",
        }
    )
    st.dataframe(evaluation_catalog, width="stretch", hide_index=True)

    if st.button(
        "현재 교과 master 추천 10건 점검",
        key="run_course_recommendation_regression",
        use_container_width=True,
    ):
        with st.spinner("교과 추천과 과목군 관련성을 점검하고 있어요..."):
            st.session_state["course_recommendation_regression"] = (
                load_course_recommendation_regression()
            )

    course_evaluation = st.session_state.get("course_recommendation_regression")
    if course_evaluation is not None:
        course_report, course_repository_name = course_evaluation
        course_metrics = st.columns(5)
        course_metrics[0].metric("평가 사례", f"{course_report.case_count}개")
        course_metrics[1].metric(
            "조건 통과",
            f"{course_report.case_pass_rate:.1f}%",
        )
        course_metrics[2].metric(
            "과목군 포괄",
            f"{course_report.family_coverage_rate:.1f}%",
        )
        course_metrics[3].metric(
            "관련 과목",
            f"{course_report.relevant_course_rate:.1f}%",
        )
        course_metrics[4].metric(
            "DB·중복 오류",
            f"{course_report.unknown_course_count + course_report.duplicate_course_count}건",
        )
        course_display = course_report.details.rename(
            columns={
                "case_id": "사례",
                "title": "학습 요구",
                "department": "학과",
                "recommended_courses": "추천 교과목",
                "matched_families": "확인된 과목군",
                "family_matches": "과목군 포괄",
                "relevant_courses": "관련 과목",
                "home_cross_courses": "학과 구성",
                "backend": "유사도 방식",
                "passed": "통과",
                "issues": "검토 사항",
                "error": "오류",
            }
        )
        st.dataframe(course_display, width="stretch", hide_index=True)
        st.caption(
            f"데이터 provider: {course_repository_name}. 이 수치는 synthetic 내부 회귀 "
            "기준이며 실제 학생 추천 정확도나 교육 효과를 의미하지 않습니다."
        )

render_footer()
