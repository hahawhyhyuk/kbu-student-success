"""핵심 메시지·AI 역할·구현 근거를 한 화면에 정리한 발표 요약."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.presentation_service import PresentationEvidence, build_presentation_evidence
from src.repositories import get_default_repository
from src.ui import (
    AFFILIATION,
    COMPETITION_NAME,
    TAGLINE,
    TEAM_NAME,
    render_footer,
    render_flow_step,
    render_page_header,
)


@st.cache_data
def load_presentation_evidence() -> tuple[PresentationEvidence, str]:
    """Repository 구현 범위 지표와 현재 provider 이름을 로드한다."""

    repository = get_default_repository()
    return build_presentation_evidence(repository.get_all()), type(repository).__name__


st.set_page_config(page_title="발표 요약", page_icon="🎤", layout="wide")
render_page_header(
    "발표 요약",
    "핵심 문제·AI 활용·안전장치·교육과정 환류까지 심사·발표 관점으로 한 화면에 정리합니다.",
    audience="발표·심사 요약",
)

st.caption(f"{COMPETITION_NAME} · Team {TEAM_NAME} · {AFFILIATION}")
with st.container(border=True):
    st.markdown("### 30초 소개")
    st.markdown(
        f"> **{TAGLINE}**  "
        "\n> 위험학생을 찾는 데서 끝나지 않고, 왜 지원이 필요한지 설명합니다. "
        "학생이 작성한 고민을 AI로 이해해 "
        "실제 대학 지원프로그램과 개인 학습경로를 연결하며, 반복 수요를 교육과정 개선 근거로 환류합니다."
    )

st.subheader("발견에서 교육과정 개선까지")
flow = (
    ("01", "조기발견", "주차별 신호를 계산해 기준 초과 시 경보"),
    ("02", "원인 설명", "규칙형 점수와 위험유형·주차별 변화 제시"),
    ("03", "AI 고민 이해", "자유서술을 검증된 JSON으로 구조화"),
    ("04", "맞춤 추천", "DB 프로그램 Top 3·DB 교과목 3~4개 연결"),
    ("05", "지원계획·실행", "프로그램 선택·교체, 담당자·연락일·실행 상태 기록"),
    ("06", "후속 변화", "4주차 지원 시작과 synthetic 15주차 관찰 비교"),
    (
        "07",
        "교육과정 환류",
        "반복 학습경로를 신규 마이크로디그리 개발 후보로 집계",
    ),
)
first_row = st.columns(4)
for column, item in zip(first_row, flow[:4]):
    with column:
        render_flow_step(*item)
second_row = st.columns(3)
for column, item in zip(second_row, flow[4:]):
    with column:
        render_flow_step(*item)

evidence, repository_name = load_presentation_evidence()
st.subheader("숫자로 확인하는 구현 범위")
metric_columns = st.columns(6)
metric_columns[0].metric("가상학생", f"{evidence.student_count}명")
metric_columns[1].metric("관찰 기간", f"{evidence.week_count}주")
metric_columns[2].metric("위험 영역", f"{evidence.risk_domain_count}개")
metric_columns[3].metric("지원프로그램", f"{evidence.support_program_count}개")
metric_columns[4].metric("교과목", f"{evidence.course_count}개")
metric_columns[5].metric("가상 학과", f"{evidence.department_count}개")
st.caption(
    f"현재 데이터 provider: {repository_name} · "
    f"위험·추천 입력 민감필드 {len(evidence.sensitive_fields_used)}개 · "
    "위 수치는 synthetic 구현 범위이며 예측 정확도·실제 성과 지표가 아닙니다."
)

st.subheader("AI가 하는 일과 코드가 통제하는 일")
ai_roles = pd.DataFrame(
    [
        {
            "단계": "교직원 브리핑",
            "AI 역할": "최신 집계의 의미와 다음 확인 화면 설명",
            "검증·통제": "학생 선별·숫자 계산은 코드, 허용된 위험유형·화면 행동만 출력",
        },
        {
            "단계": "학생 체크인",
            "AI 역할": "자유서술의 고민·관심·지원수요 구조화",
            "검증·통제": "JSON Schema 재검증, 실패 시 Mock fallback",
        },
        {
            "단계": "지원프로그램",
            "AI 역할": "Sentence-Transformers 우선 의미 유사도",
            "검증·통제": "DB 존재 항목만 후보화, deterministic Top 3",
        },
        {
            "단계": "개인 학습경로",
            "AI 역할": "경로명·과목별 역할·희망직무 연결 설명",
            "검증·통제": "이수·선수·개설 조건 코드 적용, DB 교과목만 유지",
        },
        {
            "단계": "학생지원 진행",
            "AI 역할": "현재 상태에 맞는 다음 조치 근거 설명",
            "검증·통제": "설정된 표준 조치만 제안, 교직원 검토·저장 전에는 미반영",
        },
        {
            "단계": "교육과정 후보",
            "AI 역할": "후보명·공통역량·직무 수요 설명",
            "검증·통제": "최소 학생·학과·유사도 조건을 코드가 검증",
        },
    ]
)
st.dataframe(ai_roles, width="stretch", hide_index=True)
st.info(
    "위험점수·등급은 AI 예측이 아니라 YAML 설정 기반 규칙형 초기경보입니다. "
    "AI는 학생의 말을 이해하고 설명하는 역할을 담당하며, 최종 개입은 교직원이 검토합니다."
)

st.subheader("공식 심사기준 대응표")
st.caption(
    "대회 포스터 기준 공통 심사 80점과 주제 01 심사 20점, 총 100점입니다. "
    "현재 구현 근거는 자체 예상점수가 아니라 발표 준비를 위한 대응 내용입니다."
)
common_review_items = (
    (
        "프로젝트 완성도",
        "20점",
        "목표·기능 구현, 문제 해결 과정, 결과물 안정성, 즉시 사용 가능 여부",
        "7단계 지원 흐름, 재현 가능한 시연, 예외 fallback, 자동화 테스트",
    ),
    (
        "기술적 난이도",
        "15점",
        "구현 기술의 난이도·완성도, 독창성·차별성, 시스템 설계·개발 수준",
        "UI·service·repository 분리, AI provider, JSON 검증, SQLite 이력",
    ),
    (
        "효과성 및 확산 가능성",
        "15점",
        "교내 적용 가능성, 기대 효과·실효성, 지속 운영·확산 가능성",
        "개입 상태·효과 추적, 교직원 검토, CSV→Oracle 교체 구조",
    ),
    (
        "AI 기술 활용도",
        "15점",
        "AI 기술의 적절성, 활용 수준·창의성, AI와 결과물의 연계성",
        "체크인 구조화, 의미 유사도 추천, 학습경로 설명, Mock fallback",
    ),
    (
        "문제 정의의 실제성 및 데이터 기반 분석",
        "15점",
        "문제 정의·근거 활용, 문제 우선순위, 근거 자료의 실제성·신뢰성",
        "5개 위험영역·4주 추세·설명 근거 제공, 실제 대학 데이터 검증은 후속 과제",
    ),
)
st.markdown("#### 공통 심사기준 · 80점")
for title, points, criteria, evidence_text in common_review_items:
    with st.container(border=True):
        st.markdown(f"**{title} · {points}**")
        st.write(criteria)
        st.caption(f"현재 구현 근거: {evidence_text}")

st.markdown("#### 주제 01 심사기준 · 20점")
topic_review_items = pd.DataFrame(
    [
        {
            "배점": "10점",
            "공식 평가 항목": "초기 경보 정확도·고위험군 분류 유효성",
            "현재 구현 근거와 남은 검증": (
                "5개 영역 규칙형 점수·등급 경계 테스트·합성 데이터 동작 검증 구현 / "
                "실제 대학 데이터 기반 타당도 검증 필요"
            ),
        },
        {
            "배점": "5점",
            "공식 평가 항목": "개인화 추천·개입 효과 정량 평가",
            "현재 구현 근거와 남은 검증": (
                "지원프로그램 Top 3·개인 학습경로·개입 전후 스냅샷 구현 / "
                "실제 개입 성과의 정량·인과 검증 필요"
            ),
        },
        {
            "배점": "5점",
            "공식 평가 항목": "데이터 연계·통합 범용성 평가",
            "현재 구현 근거와 남은 검증": (
                "공통 Repository와 CsvRepository 구현 / "
                "실제 대학 CSV 매핑 및 OracleRepository 연동은 후속 과제"
            ),
        },
    ]
)
st.dataframe(topic_review_items, width="stretch", hide_index=True)
st.caption(
    "‘초기경보 모델 검증’ 화면에서 합성 참조 유형별 지원 신호 포착과 위험유형 일치를 "
    "확인할 수 있습니다. 이 수치는 실제 학생 데이터의 예측 정확도가 아닙니다."
)

st.subheader("솔직하게 밝히는 현재 한계와 확장 경로")
limit_left, future_right = st.columns(2)
with limit_left:
    with st.container(border=True):
        st.markdown("**현재 한계**")
        st.markdown(
            "- 고정 시드 synthetic 데이터로 실제 예측 정확도를 입증하지 않음\n"
            "- synthetic 15주차 변화는 인과효과가 아닌 시연용 후속 관찰\n"
            "- 학생 데이터는 synthetic이며 학과·비교과·교과는 검증된 실제 master 우선\n"
            "- 공식 교육과정·행정 불이익을 AI가 자동 결정하지 않음"
        )
with future_right:
    with st.container(border=True):
        st.markdown("**실제 대학 적용 경로**")
        st.markdown(
            "- CSV Repository를 Oracle/TNS Repository로 교체\n"
            "- 실제 학사·LMS·교과·프로그램 master 컬럼 정규화\n"
            "- 교직원 타당도 평가와 추천 수용 이력 축적\n"
            "- 실제 개입 결과를 활용한 기준 보정·hybrid recommender 고도화"
        )

st.subheader("발표 마무리 메시지")
st.success(
    "AI를 사용했는지보다 중요한 것은, 위험 신호를 실제로 더 적절한 학생지원과 "
    "학습경로로 연결할 수 있는가입니다."
)
if st.button("나의 체크인 열기", type="primary"):
    st.switch_page("pages/07_student_view.py")

st.divider()
st.caption(
    f"{TEAM_NAME} · {AFFILIATION} · 설명 가능하고 검증 가능한 학생성공 지원"
)
render_footer()
