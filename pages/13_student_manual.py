"""학생이 자신의 체크인과 결과 확인 흐름만 볼 수 있는 이용 매뉴얼."""

from __future__ import annotations

import streamlit as st

from src.access_control import ROLE_STUDENT
from src.ui import render_footer, render_page_header, render_submenu
from src.user_guide import ServiceGuide, guides_for_role


def render_steps(items: tuple[str, ...]) -> None:
    """짧은 사용 항목을 순서 목록으로 표시한다."""

    for index, item in enumerate(items, start=1):
        st.markdown(f"{index}. {item}")


def render_screen_guide(guide: ServiceGuide) -> None:
    """학생 화면 하나의 준비·실행·결과를 접힌 안내로 표시한다."""

    with st.expander(guide.title, expanded=False):
        st.write(guide.purpose)
        action_column, result_column = st.columns(2)
        with action_column:
            st.markdown("**이렇게 사용해요**")
            render_steps(guide.actions)
        with result_column:
            st.markdown("**확인할 결과**")
            render_steps(guide.outputs)
        st.info(f"다음 이동: {guide.next_step}")
        if guide.caution:
            st.warning(guide.caution)
        try:
            st.page_link(guide.path, label=f"{guide.title} 열기", icon="↗️")
        except KeyError:
            st.caption(f"왼쪽 메뉴에서 `{guide.title}`를 선택하세요.")


st.set_page_config(page_title="학생 이용 매뉴얼", page_icon="📖", layout="wide")
render_page_header(
    "학생 이용 매뉴얼",
    "Kare와 이야기하고 맞춤 지원 결과를 확인하는 방법을 간단히 안내합니다.",
    audience="학생용",
)

active_section = render_submenu(
    ("처음 시작하기", "화면별 사용법", "개인정보 · AI", "자주 묻는 질문"),
    key="student_manual_submenu",
)

if active_section == "처음 시작하기":
    st.subheader("체크인은 이 순서로 진행해요")
    flow = (
        ("1", "알림 또는 체크인 시작", "학교 알림이 있으면 확인하고 `나의 체크인`을 엽니다."),
        ("2", "대화 방식 선택", "Gemini 대화에 동의하거나 외부 전송 없는 직접 입력을 선택합니다."),
        ("3", "현재 이야기 나누기", "학교생활·전공·학습·진로 중 떠오르는 내용을 자연스럽게 말합니다."),
        ("4", "이해 결과 확인", "Kare가 이해한 내용이 맞는지 확인하고 필요한 부분을 수정합니다."),
        ("5", "제출과 결과 확인", "추천 비교과·역량 학습경로를 확인하고 나중에는 `마이페이지`에서 다시 봅니다."),
    )
    for number, title, description in flow:
        with st.container(border=True):
            st.markdown(f"**{number}. {title}**")
            st.write(description)
    st.info(
        "교직원에게 전달되는 것은 학생이 최종 확인해 제출한 내용과 지원 검토에 "
        "필요한 결과입니다. AI 결과만으로 지원이 자동 확정되지는 않습니다."
    )

elif active_section == "화면별 사용법":
    st.subheader("학생 화면은 두 단계로 구성돼요")
    for guide in guides_for_role(ROLE_STUDENT):
        render_screen_guide(guide)

elif active_section == "개인정보 · AI":
    st.subheader("대화 방식을 학생이 선택할 수 있어요")
    consent_column, direct_column = st.columns(2)
    with consent_column:
        with st.container(border=True):
            st.markdown("**Gemini Kare 대화**")
            st.write(
                "동의한 경우 최근 대화와 현재까지 정리된 상태만 Gemini API에 "
                "전송합니다. 학생 ID·학번·이름은 전송하지 않습니다."
            )
    with direct_column:
        with st.container(border=True):
            st.markdown("**외부 AI 전송 없는 직접 입력**")
            st.write(
                "동의하지 않아도 고민·관심 분야·희망 직무를 직접 입력할 수 있고, "
                "추천과 교직원 검토 흐름은 동일하게 이어집니다."
            )
    st.markdown("**제출 전에 반드시 확인해요**")
    st.write(
        "Kare가 이해한 정성 상태·관심 분야·희망 직무·현재 고민은 학생이 직접 "
        "확인하고 수정한 뒤에만 저장됩니다. Gemini가 실패하면 개인정보를 새로 "
        "전송하지 않고 규칙 기반 Kare로 전환됩니다."
    )

else:
    st.subheader("자주 묻는 질문")
    with st.expander("Kare가 1~5 점수를 계속 물어보나요?", expanded=True):
        st.write(
            "아니요. 현재 대화는 점수를 채우기 위한 문답이 아니라 학생이 말한 맥락과 "
            "근거를 정성 상태로 정리합니다. 분명하지 않은 내용은 판단하지 않은 상태로 남깁니다."
        )
    with st.expander("추천 결과는 어디에서 다시 보나요?", expanded=False):
        st.write(
            "`마이페이지`에서 최신 체크인, 추천 비교과, AI 맞춤형 역량 학습경로와 "
            "교직원 확인 상태를 다시 볼 수 있습니다."
        )
    with st.expander("학교 알림은 실제 이메일인가요?", expanded=False):
        st.write(
            "현재 해커톤 시연에서는 실제 이메일 주소를 사용하지 않습니다. synthetic "
            "학생 계정에 교내 알림 상태만 저장합니다."
        )
    with st.expander("추천된 프로그램은 바로 신청되나요?", expanded=False):
        st.write(
            "아니요. 추천은 선택지를 연결하는 단계이며 교직원이 검토합니다. 프로그램의 "
            "마감·게시 여부와 신청 일정은 WINGS에서 최종 확인해야 합니다."
        )

st.divider()
st.caption(
    "학생의 AI 동의 여부는 위험점수나 지원 우선순위에 사용되지 않습니다. "
    "최종 지원은 학생의 확인 내용과 교직원 검토를 거칩니다."
)
render_footer()
