"""교직원용 KBU AI 학생성공 레이더 서비스 소개 화면."""

import streamlit as st

from src.ai_copilot_service import AICopilotService, CopilotResult
from src.dashboard_priority_service import DashboardPriorityService
from src.database import initialize_database
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
    select_latest_student_snapshots,
)
from src.streamlit_ai import create_streamlit_ai_provider
from src.ui import (
    AFFILIATION,
    CORE_VALUES,
    TAGLINE,
    TEAM_MEMBERS,
    queue_ai_reveal,
    render_ai_reveal,
    render_flow_step,
    render_footer,
    render_home_hero,
)


@st.cache_data
def load_home_snapshots(checkin_revision: int):
    """첫 화면 AI 브리핑에 사용할 학생별 최신 위험 스냅샷을 반환한다."""

    snapshots = create_integrated_risk_service(
        get_default_repository()
    ).build_snapshots()
    latest = select_latest_student_snapshots(snapshots)
    latest_week = int(snapshots["week"].max())
    return latest, latest_week


st.set_page_config(
    page_title="KBU AI 학생성공 레이더",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()
render_home_hero()

st.info(
    "위험분류는 학생을 처벌하기 위한 판정이 아니라, 교직원이 적절한 지원 시점과 방법을 "
    "검토할 수 있도록 돕는 초기경보 정보입니다."
)

st.subheader("바로 시작하기")
quick_starts = (
    (
        "전체 초기경보",
        "최신 신호에서 우선 확인 학생과 지원 흐름을 살펴봅니다.",
        "pages/01_dashboard.py",
        "전체 대시보드 열기",
        "📊",
    ),
    (
        "AI 맞춤 추천 통합 검토",
        "학생이 확인한 AI 이해와 비교과·교과 추천 근거를 한 화면에서 검토합니다.",
        "pages/03_ai_intervention.py",
        "통합 추천 검토 열기",
        "🤝",
    ),
    (
        "시연 준비",
        "발표 전에 synthetic 기준 주차·기록·Gemini 연결 상태를 점검합니다.",
        "pages/09_demo_scenario.py",
        "시연 준비 열기",
        "🎬",
    ),
)
for column, (title, description, path, label, icon) in zip(
    st.columns(3), quick_starts
):
    with column:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption(description)
            try:
                st.page_link(path, label=label, icon=icon)
            except KeyError:
                # AppTest가 시작 페이지를 단독 실행하는 경우를 허용한다.
                pass

st.subheader("AI 학생성공 코파일럿")
with st.container(border=True, key="kbu_ai_copilot"):
    st.markdown("### 안녕하세요. 오늘 학생성공 신호를 함께 살펴볼까요?")
    st.caption(
        "AI는 최신 집계 결과를 설명하고, 학생·프로그램·교과목 후보는 코드와 DB가 통제합니다."
    )
    copilot_prompts = {
        "오늘 우선 확인할 신호": (
            "오늘 우선 확인할 학생 신호를 알려줘",
            "priority",
        ),
        "지원 전 학생 확인": (
            "아직 지원을 시작하지 못한 학생 흐름을 알려줘",
            "unaddressed",
        ),
        "위험유형 흐름 요약": (
            "이번 주 위험유형 흐름을 요약해줘",
            "risk_distribution",
        ),
    }
    selected_copilot_prompt = st.pills(
        "빠른 질문",
        list(copilot_prompts),
        default="오늘 우선 확인할 신호",
        required=True,
        key="home_copilot_prompt",
        label_visibility="collapsed",
    )
    prompt_text, request_focus = copilot_prompts[
        str(selected_copilot_prompt or "오늘 우선 확인할 신호")
    ]
    prompt_index = list(copilot_prompts).index(
        str(selected_copilot_prompt or "오늘 우선 확인할 신호")
    )
    st.text_input(
        "선택한 질문",
        value=prompt_text,
        disabled=True,
        key=f"home_copilot_prompt_preview_{prompt_index}",
    )
    generate_home_briefing = st.button(
        "AI 브리핑 생성",
        type="primary",
        icon="✨",
        key="home_copilot_generate",
    )

home_latest, home_latest_week = load_home_snapshots(
    get_student_checkin_revision()
)
home_unaddressed = DashboardPriorityService().find_unaddressed_high_risk(
    home_latest
)
home_result_key = "home_ai_copilot_result"
if generate_home_briefing:
    try:
        with st.spinner("최신 신호를 집계하고 AI 브리핑을 작성하고 있습니다..."):
            home_service = AICopilotService(create_streamlit_ai_provider())
            home_context = home_service.build_staff_context(
                home_latest, home_unaddressed
            )
            home_context["request_focus"] = request_focus
            home_result = home_service.generate_staff_briefing(home_context)
            st.session_state[home_result_key] = (
                home_result,
                home_context,
                prompt_text,
                home_latest_week,
            )
            queue_ai_reveal("home_copilot_briefing")
    except Exception as error:
        st.error(f"AI 브리핑을 생성하지 못했습니다. ({error})")

stored_home_result = st.session_state.get(home_result_key)
if stored_home_result is not None:
    home_result: CopilotResult
    home_context: dict[str, object]
    stored_prompt: str
    stored_week: int
    home_result, home_context, stored_prompt, stored_week = stored_home_result
    with st.chat_message("assistant", avatar="✨"):
        st.caption(f"질문 · {stored_prompt}")
        render_ai_reveal(
            (
                f"### {home_result.content['headline']}\n\n"
                f"{home_result.content['summary']}"
            ),
            key="home_copilot_briefing",
            label=f"검증 완료 · {home_result.provider_name} AI 브리핑",
        )
        if home_result.warning:
            st.warning(home_result.warning)
        briefing_metrics = st.columns(4)
        briefing_metrics[0].metric("최대 관찰 주차", f"{stored_week}주차")
        briefing_metrics[1].metric(
            "고위험", f"{home_context['risk_level_counts']['고위험']}명"
        )
        briefing_metrics[2].metric(
            "지원 시작 전", f"{home_context['unaddressed_count']}명"
        )
        briefing_metrics[3].metric(
            "전주 대비 상승", f"{home_context['increasing_count']}명"
        )
        st.markdown(
            "**집중 신호:** "
            + " · ".join(home_result.content["focus_risk_types"])
        )
        st.info(f"**AI가 제안한 다음 화면:** {home_result.content['recommended_action']}")
        st.caption(
            "근거: 최신 위험등급 분포·주요 위험유형·전주 대비 변화·지원 시작 전 상태 · 최종 판단은 교직원"
        )
        try:
            st.page_link(
                "pages/01_dashboard.py",
                label="전체 대시보드에서 근거 확인",
                icon="📊",
            )
        except KeyError:
            st.caption("왼쪽 메뉴에서 `전체 대시보드`를 선택하세요.")
else:
    st.caption(
        "빠른 질문을 선택하고 브리핑을 생성하면, AI가 최신 데이터의 의미와 다음 확인 화면을 설명합니다."
    )

with st.expander("서비스 구성·운영 원칙 자세히 보기", expanded=False):
    st.subheader("학생 성공 지원 흐름")
    flow_steps = (
        ("01", "초기경보", "출결·참여·성취·전공·진로 신호 통합"),
        ("02", "위험유형 분석", "지원이 필요한 주요 원인과 변화 설명"),
        ("03", "AI 체크인", "학생 자유서술을 검증된 구조로 해석"),
        ("04", "맞춤 개입", "DB 지원프로그램 Top 3와 근거 제안"),
        ("05", "학습경로", "이수·선수조건을 반영한 3~4개 교과 연결"),
        ("06", "개입·효과 관리", "교직원 검토와 개입 후 변화를 이력으로 관리"),
        (
            "07",
            "교육과정 개선",
            "반복 수요를 신규 마이크로디그리 개발 후보로 집계",
        ),
    )
    first_row = st.columns(4)
    for column, step in zip(first_row, flow_steps[:4]):
        with column:
            render_flow_step(*step)
    second_row = st.columns(4)
    for column, step in zip(second_row, flow_steps[4:]):
        with column:
            render_flow_step(*step)

    st.subheader("서비스 핵심 가치")
    value_columns = st.columns(4)
    for column, (number, title, description) in zip(value_columns, CORE_VALUES):
        with column:
            render_flow_step(number, title, description, label="VALUE")

    scope_column, principle_column = st.columns(2)
    with scope_column:
        with st.container(border=True):
            st.subheader("주요 구현 기능")
            st.markdown(
                """
                - 가상학생 200명, 4주 위험 추세와 설명 가능한 초기경보
                - 가상 참조 유형 대비 지원 신호·위험유형 동작 검증
                - Gemini/Mock 기반 체크인 분석과 지원프로그램 Top 3
                - DB 교과목 기반 AI 맞춤형 역량 학습경로
                - SQLite 개입 상태·학생 피드백 이력 관리
                - 개입 시점·최신 위험도 비교와 후속 관찰
                - 반복 학습경로 기반 교육과정 개발 후보 발굴
                """
            )
    with principle_column:
        with st.container(border=True):
            st.subheader("운영 원칙")
            st.markdown(
                """
                - 최종 개입 여부와 방식은 교직원이 검토합니다.
                - 추천은 DB에 존재하는 프로그램과 교과목으로 제한합니다.
                - 성별·연령·국적 등 민감정보는 점수에 사용하지 않습니다.
                - 개발 후보는 공식 마이크로디그리가 아닙니다.
                - 현재 데이터는 고정 시드로 만든 가상데이터입니다.
                """
            )

    member_summary = " · ".join(
        f"{name}({role})" for name, role, _ in TEAM_MEMBERS
    )
    st.caption(
        f"Team Kare Bridge U · {AFFILIATION} · {member_summary} · {TAGLINE}"
    )

st.sidebar.success("교직원 메뉴에서 검토할 업무를 선택하세요.")
render_footer()
