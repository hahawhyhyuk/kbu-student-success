"""교직원이 조기경보부터 후속 확인까지 따라가는 이용 매뉴얼."""

from __future__ import annotations

import streamlit as st

from src.access_control import ROLE_STAFF
from src.ui import render_footer, render_page_header, render_submenu
from src.user_guide import (
    ServiceGuide,
    guide_categories_for_role,
    guides_for_role,
)


def render_steps(items: tuple[str, ...]) -> None:
    """짧은 사용 항목을 순서 목록으로 표시한다."""

    for index, item in enumerate(items, start=1):
        st.markdown(f"{index}. {item}")


def render_screen_guide(guide: ServiceGuide) -> None:
    """교직원 화면 하나의 입력·실행·결과를 접힌 안내로 표시한다."""

    with st.expander(guide.title, expanded=False):
        st.write(guide.purpose)
        input_column, action_column, output_column = st.columns(3)
        with input_column:
            st.markdown("**준비·입력**")
            render_steps(guide.inputs)
        with action_column:
            st.markdown("**사용 순서**")
            render_steps(guide.actions)
        with output_column:
            st.markdown("**확인할 결과**")
            render_steps(guide.outputs)
        st.info(f"다음 이동: {guide.next_step}")
        if guide.caution:
            st.warning(guide.caution)
        try:
            st.page_link(guide.path, label=f"{guide.title} 열기", icon="↗️")
        except KeyError:
            st.caption(f"왼쪽 메뉴에서 `{guide.title}`를 선택하세요.")


st.set_page_config(page_title="교직원 이용 매뉴얼", page_icon="📖", layout="wide")
render_page_header(
    "교직원 이용 매뉴얼",
    "조기경보 확인부터 학생 안내, AI 추천 검토, 지원 실행과 후속 변화까지 안내합니다.",
)

active_section = render_submenu(
    ("업무 흐름", "화면별 사용법", "경보 · 추천 기준", "데이터 · 시연"),
    key="staff_manual_submenu",
)

if active_section == "업무 흐름":
    st.subheader("교직원 지원은 이 순서로 진행합니다")
    flow = (
        ("1", "전체 대시보드", "주차와 위험 분포를 확인하고 먼저 살펴볼 학생을 찾습니다."),
        ("2", "학생 안내", "경보 근거를 확인한 후 synthetic 교내 알림을 보냅니다."),
        ("3", "학생 체크인", "학생이 알림을 확인하고 Kare 상담 결과를 제출합니다."),
        ("4", "학생 종합 확인", "행동지표·학생 이야기·과거 지원을 함께 확인합니다."),
        ("5", "AI 맞춤 추천 검토", "DB 비교과 Top 3와 교과 학습경로를 승인·수정·보류합니다."),
        ("6", "학생지원 진행 관리", "담당자·연락일·프로그램별 실행 상태를 기록합니다."),
        ("7", "지원 후 변화 확인", "상담 완료 뒤 synthetic 15주차 관찰로 개선·유지·악화를 비교합니다."),
        ("8", "교육과정 개선", "반복 교과 수요를 신규 마이크로디그리 개발 후보로 검토합니다."),
    )
    for number, title, description in flow:
        with st.container(border=True):
            st.markdown(f"**{number}. {title}**")
            st.write(description)
    st.info(
        "학생 선별·위험점수·추천 후보는 규칙과 DB가 결정하고, 생성형 AI는 학생의 "
        "자유서술 해석과 선정된 후보의 설명을 돕습니다. 최종 지원은 교직원이 결정합니다."
    )

elif active_section == "화면별 사용법":
    role_guides = guides_for_role(ROLE_STAFF)
    for category in guide_categories_for_role(ROLE_STAFF):
        category_guides = tuple(
            guide for guide in role_guides if guide.category == category
        )
        st.subheader(category)
        for guide in category_guides:
            render_screen_guide(guide)

elif active_section == "경보 · 추천 기준":
    st.subheader("경보는 LLM이 아니라 설정된 규칙으로 계산합니다")
    with st.container(border=True):
        st.markdown("**다섯 위험 영역**")
        st.markdown(
            "- 출결: 출석률·연속결석\n"
            "- 학습참여: 과제 제출·LMS 로그인·온라인 영상 진도·활동 변화\n"
            "- 학업성취: GPA·퀴즈·재수강·학사경고\n"
            "- 전공적응·진로설계: 학생 체크인에서 확인된 자기보고"
        )
    threshold_column, priority_column = st.columns(2)
    with threshold_column:
        with st.container(border=True):
            st.markdown("**종합 위험등급**")
            st.write(
                "다섯 영역을 설정 가중치로 합산해 정상(0~24), 관심(25~44), "
                "주의(45~64), 고위험(65~100)으로 구분합니다. 기준은 "
                "`config/risk_config.yaml`에서 관리합니다."
            )
    with priority_column:
        with st.container(border=True):
            st.markdown("**안내 우선 대상**")
            st.write(
                "선택한 주차에서 고위험이고 아직 지원이 시작되지 않은 학생을 "
                "종합 위험도와 최근 변화 순으로 표시합니다. 주차 자체가 위험단계는 아닙니다."
            )
    st.subheader("추천 후보는 대학 master 안에서만 고릅니다")
    st.markdown(
        "- 비교과: 학생의 위험 영역·체크인 지원수요·설명 유사도로 전체 WINGS 비교과에서 Top 3 선정\n"
        "- 교과: 실제 개설 교과목 중 이수과목을 제외하고 소속 학과를 우선하되 타과 과목 허용\n"
        "- 마이크로디그리: 학생별 학습경로에서 반복된 실제 교과목 조합을 수요순 개발 후보로 집계\n"
        "- Gemini: 코드가 고른 프로그램·교과목 ID를 추가하거나 교체할 수 없고 설명만 생성"
    )

else:
    st.subheader("현재 시연 데이터와 운영 전환 원칙")
    with st.expander("어떤 데이터가 적용되어 있나요?", expanded=True):
        st.markdown(
            "- 학생 행동·체크인: 개인정보 없는 고정 seed synthetic 데이터\n"
            "- 학과: 실제 안전 master 46개, 학생 모집단은 설정된 5개 학과\n"
            "- 비교과: 실제 WINGS Excel의 비교과 102개\n"
            "- 교과: 2026년 1·2학기 실제 개설강좌 기반 987개 후보"
        )
    with st.expander("비교과 운영 상태는 어떻게 처리하나요?", expanded=False):
        st.write(
            "운영 상태와 무관하게 102개 전체를 추천 후보로 사용합니다. 원본 상태는 "
            "추적용으로만 보존하며, 마감·게시 여부와 신청 일정은 WINGS에서 최종 확인합니다."
        )
    with st.expander("Gemini가 실패하면 서비스가 멈추나요?", expanded=False):
        st.write(
            "아니요. HTTP 429, API key 누락, 응답 검증 실패가 발생하면 규칙 기반 Mock "
            "설명으로 전환합니다. 코드가 선정한 위험·추천 후보와 저장 흐름은 유지됩니다."
        )
    with st.expander("15주차 후속 변화는 언제 표시되나요?", expanded=False):
        st.write(
            "대표 학생의 지원계획을 실행한 뒤 `학생지원 진행 관리`에서 상태를 "
            "`상담 완료`, `프로그램 참여` 또는 `추후 관찰`로 기록하면 표시됩니다. "
            "실제 학생 성과가 아니라 지원 전후 화면을 점검하기 위한 synthetic 관찰입니다."
        )
    with st.expander("시연 기록은 어떻게 초기화하나요?", expanded=False):
        st.write(
            "관리자 메뉴의 `시연 준비`에서 확인 문구를 입력하면 SQLite 백업을 먼저 "
            "만든 뒤 시연 기록만 삭제합니다. CSV·설정·원본 Excel은 변경하지 않습니다."
        )
    with st.expander("실제 대학 운영 전에 무엇이 필요한가요?", expanded=False):
        st.write(
            "대학 SSO와 서버 세션, 실제 출결·e-Class·성취 데이터 연동, 발송 동의와 "
            "감사 로그, 프로그램별 직접 URL 및 실시간 신청 상태 연결이 필요합니다."
        )

st.divider()
st.caption(
    "위험등급은 처벌이나 불이익이 아니라 지원 우선순위를 위한 지표입니다. "
    "학생의 실제 상황과 의사를 확인한 뒤 지원을 확정하세요."
)
render_footer()
