"""KBU 브랜드와 공통 Streamlit 화면 요소를 관리한다."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
import re
import time
from typing import Sequence
import unicodedata

import streamlit as st

from src.access_control import (
    ROLE_STAFF,
    ROLE_STUDENT,
    STUDENT_HOME_PAGE,
    clear_authenticated_session,
    get_authenticated_identity,
)
from src.utils import PROJECT_ROOT, load_app_config


TEAM_NAME = "Kare Bridge U"
SERVICE_NAME = "KBU AI 학생성공 레이더"
COMPETITION_NAME = "2026년 AI해커톤 대회"
AFFILIATION = "디지털정보처 IR센터"
TAGLINE = (
    "경복대학교 학생의 위험 신호를 조기에 발견하고, "
    "AI 기반 맞춤 지원과 학습경로로 학생 성공을 연결하는 징검다리."
)
LOGO_PATH = PROJECT_ROOT / "assets" / "kbu_logo_horizontal.png"
TEAM_MEMBERS = (
    ("허혁", "개발 및 구현", AFFILIATION),
    ("박지선", "개발 및 데이터 검수", AFFILIATION),
)
CORE_VALUES = (
    ("01", "조기 발견", "다섯 영역에서 지원 신호를 조기에 발견합니다."),
    ("02", "설명 가능한 AI", "점수와 추천 근거를 교직원이 직접 확인합니다."),
    ("03", "맞춤형 학생 지원", "맞춤 프로그램과 개인 학습경로를 연결합니다."),
    ("04", "교육과정 개선", "반복 수요를 교육과정 개발 후보로 확장합니다."),
)
AI_REVEAL_PENDING_STATE = "_kbu_ai_reveal_pending"
DEFAULT_START_PAGE = STUDENT_HOME_PAGE
STUDENT_NAVIGATION_GROUPS: tuple[
    tuple[str, tuple[tuple[str, str, str], ...]], ...
] = (
    (
        "학생 서비스",
        (
            ("pages/07_student_view.py", "나의 체크인", "🌱"),
            ("pages/12_student_mypage.py", "마이페이지", "👤"),
            ("pages/13_student_manual.py", "이용 매뉴얼", "📖"),
        ),
    ),
)
STAFF_NAVIGATION_GROUPS: tuple[
    tuple[str, tuple[tuple[str, str, str], ...]], ...
] = (
    (
        "교직원 핵심 흐름",
        (
            ("pages/01_dashboard.py", "전체 대시보드", "📊"),
            ("pages/02_student_detail.py", "학생 종합 확인", "🔎"),
            (
                "pages/03_ai_intervention.py",
                "AI 맞춤 추천 통합 검토",
                "🤝",
            ),
            (
                "pages/05_intervention_management.py",
                "학생지원 진행 관리",
                "📋",
            ),
            ("pages/08_intervention_effect.py", "지원 후 변화 확인", "📈"),
            ("pages/14_staff_manual.py", "이용 매뉴얼", "📖"),
        ),
    ),
    (
        "교육과정",
        (
            (
                "pages/06_microdegree_candidates.py",
                "신규 마이크로디그리 개발 후보",
                "🧩",
            ),
        ),
    ),
    (
        "관리자",
        (
            ("pages/00_service_intro.py", "서비스 소개", "🏠"),
            ("pages/09_demo_scenario.py", "시연 준비", "🎬"),
            ("pages/10_presentation_summary.py", "발표 요약", "🎤"),
            ("pages/11_alert_validation.py", "초기경보 모델 검증", "✅"),
        ),
    ),
)
DEFAULT_COLLAPSED_NAVIGATION_GROUPS = frozenset({"관리자"})


def navigation_groups_for_role(
    role: str,
) -> tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...]:
    """학생과 교직원에게 서로 겹치지 않는 메뉴 그룹을 반환한다."""

    if role == ROLE_STUDENT:
        return STUDENT_NAVIGATION_GROUPS
    if role == ROLE_STAFF:
        return STAFF_NAVIGATION_GROUPS
    return ()


def navigation_items_for_role(role: str) -> tuple[tuple[str, str, str], ...]:
    """역할별 페이지 메타데이터를 라우터 등록 순서로 펼친다."""

    return tuple(
        item
        for _, group_items in navigation_groups_for_role(role)
        for item in group_items
    )


def navigation_group_expanded_by_default(group_label: str) -> bool:
    """일반 업무는 펼치고 관리자 도구는 기본적으로 접는다."""

    return str(group_label) not in DEFAULT_COLLAPSED_NAVIGATION_GROUPS


def estimate_chat_text_units(text: str) -> float:
    """채팅 문장의 시각적 길이를 한글 1자 폭 기준으로 추정한다."""

    normalized = " ".join(str(text).split())
    units = 0.0
    for character in normalized:
        if character.isspace():
            units += 0.45
        elif unicodedata.east_asian_width(character) in {"W", "F", "A"}:
            units += 1.0
        else:
            units += 0.58
    return max(units, 1.0)


def calculate_chat_bubble_width_rem(text: str) -> float:
    """문장 길이에 따라 1~4줄이 균형을 이루는 말풍선 최대 폭을 계산한다."""

    units = estimate_chat_text_units(text)
    if units <= 28:
        line_units = max(6.0, units)
    elif units <= 72:
        line_units = math.ceil(units / 2)
    elif units <= 120:
        line_units = math.ceil(units / 3)
    else:
        line_units = min(38, math.ceil(units / 4))
    return round(min(33.0, max(7.5, line_units * 0.82 + 2.5)), 1)


def build_kare_chat_bubble_html(role: str, content: str) -> str:
    """안전하게 이스케이프한 Kare/학생용 가변 폭 말풍선 HTML을 만든다."""

    normalized_role = "user" if role == "user" else "assistant"
    label = "나" if normalized_role == "user" else "Kare"
    avatar = "🙂" if normalized_role == "user" else "✨"
    accessible_role = "user" if normalized_role == "user" else "assistant"
    safe_content = escape(str(content)).replace("\n", "<br>")
    width_rem = calculate_chat_bubble_width_rem(str(content))
    return (
        f'<div class="kare-message-row kare-message-{normalized_role}" '
        f'aria-label="Chat message from {accessible_role}">'
        f'<div class="kare-message-avatar" aria-hidden="true">{avatar}</div>'
        f'<div class="kare-message-bubble" '
        f'style="--kare-bubble-max: {width_rem}rem">'
        f'<div class="kare-message-label">{label}</div>'
        f'<div class="kare-message-text">{safe_content}</div>'
        "</div></div>"
    )


def build_kare_typing_indicator_html() -> str:
    """Kare 응답을 기다리는 동안 대화 목록에 표시할 말풍선을 만든다."""

    return (
        '<div class="kare-message-row kare-message-assistant '
        'kare-message-typing" role="status" aria-live="polite" '
        'aria-label="Kare가 답변을 생각하고 있어요">'
        '<div class="kare-message-avatar" aria-hidden="true">✨</div>'
        '<div class="kare-message-bubble">'
        '<div class="kare-message-label">Kare</div>'
        '<div class="kare-typing-content">'
        '<span class="kare-typing-dots" aria-hidden="true">'
        '<span class="kare-typing-dot"></span>'
        '<span class="kare-typing-dot"></span>'
        '<span class="kare-typing-dot"></span>'
        "</span>"
        '<span class="kare-typing-label">답변을 생각하고 있어요</span>'
        "</div></div></div>"
    )


BRAND_CSS = """
<style>
    :root {
        --kbu-pink: #EC008C;
        --kbu-blue: #004085;
        --kbu-blue-2: #006AB6;
        --kbu-ink: #17324D;
        --kbu-muted: #60758A;
        --kbu-line: #D9E3EE;
        --kbu-surface: #FFFFFF;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 92% 4%, rgba(236, 0, 140, 0.055), transparent 22rem),
            linear-gradient(180deg, #F8FAFD 0%, #F5F8FC 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(248, 250, 253, 0.88);
        border-bottom: 1px solid rgba(217, 227, 238, 0.75);
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--kbu-line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #49647D;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: 0;
        background: transparent;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #294A68;
        font-weight: 750;
    }
    [data-testid="stPageLink"] a {
        min-height: 2.55rem;
        margin: 0.08rem 0;
        padding: 0.5rem 0.7rem;
        color: #294A68;
        border-radius: 0.65rem;
        font-weight: 650;
    }
    [data-testid="stPageLink"] a:hover {
        color: #004085;
        background: #E3EEF8;
    }
    [data-testid="stPageLink"] a[aria-current="page"] {
        color: #FFFFFF;
        background: linear-gradient(105deg, #004085, #006AB6);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 4rem;
        padding-bottom: 4.5rem;
    }
    h1, h2, h3 {
        color: var(--kbu-blue);
        letter-spacing: -0.025em;
    }
    h2 {
        margin-top: 1.65rem !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #E5ECF3;
    }
    [data-testid="stMetric"] {
        box-sizing: border-box;
        min-height: 96px;
        height: auto;
        padding: 0.82rem 0.95rem;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--kbu-line);
        border-top: 3px solid var(--kbu-blue-2);
        border-radius: 0.85rem;
        box-shadow: 0 7px 22px rgba(0, 64, 133, 0.055);
    }
    [data-testid="stMetricLabel"] {
        color: var(--kbu-muted);
        font-weight: 650;
    }
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricDelta"] {
        max-width: 100%;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
        word-break: keep-all;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        font-size: clamp(1.18rem, 2.15vw, 2rem) !important;
        line-height: 1.22 !important;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 2.75rem;
        font-weight: 700;
        border: 1px solid #BED0E2;
        box-shadow: 0 4px 12px rgba(0, 64, 133, 0.07);
    }
    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        color: white;
        border: none;
        background: linear-gradient(115deg, #004085 0%, #006AB6 63%, #EC008C 140%);
    }
    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
        color: white;
        border: none;
        background: linear-gradient(115deg, #00356E 0%, #005B9E 63%, #D5007F 140%);
        transform: translateY(-1px);
    }
    [data-testid="stButtonGroup"] [role="radio"] {
        flex: 0 0 auto !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.9rem;
        overflow: hidden;
    }
    [data-testid="stAlert"] {
        border-radius: 0.8rem;
        border-width: 1px;
    }
    .kbu-ai-reveal-label {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        margin: 0.15rem 0 0.45rem;
        padding: 0.3rem 0.58rem;
        color: #315A7E;
        background: #EDF5FC;
        border: 1px solid #D2E3F1;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 750;
    }
    div[class*="st-key-kbu_ai_copilot"] {
        background:
            radial-gradient(circle at 96% 4%, rgba(236, 0, 140, 0.11), transparent 15rem),
            linear-gradient(135deg, rgba(234, 243, 251, 0.94), rgba(255, 255, 255, 0.98));
        border-color: rgba(0, 100, 178, 0.24) !important;
        box-shadow: 0 16px 38px rgba(0, 64, 133, 0.08);
    }
    .kbu-ai-reveal-dot {
        width: 0.48rem;
        height: 0.48rem;
        background: var(--kbu-pink);
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(236, 0, 140, 0.3);
        animation: kbu-ai-pulse 1.35s ease-out infinite;
    }
    @keyframes kbu-ai-pulse {
        0% { box-shadow: 0 0 0 0 rgba(236, 0, 140, 0.3); }
        70%, 100% { box-shadow: 0 0 0 7px rgba(236, 0, 140, 0); }
    }
    .kbu-page-header {
        position: relative;
        overflow: hidden;
        margin: 0 0 1rem 0;
        padding: 1.05rem 1.35rem 1rem;
        color: white;
        border-radius: 1.05rem;
        background: linear-gradient(112deg, #003A78 0%, #005B9E 72%, #006AB6 100%);
        box-shadow: 0 14px 35px rgba(0, 64, 133, 0.16);
    }
    .kbu-page-header::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        top: -145px;
        right: -25px;
        border: 28px solid rgba(236, 0, 140, 0.52);
        border-radius: 50%;
    }
    .kbu-eyebrow {
        position: relative;
        z-index: 1;
        margin-bottom: 0.42rem;
        color: #FFD5EF;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .kbu-page-title {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #FFFFFF;
        font-size: clamp(1.55rem, 2.6vw, 2.05rem);
        font-weight: 850;
        line-height: 1.2;
        letter-spacing: -0.035em;
    }
    .kbu-page-subtitle {
        position: relative;
        z-index: 1;
        max-width: 880px;
        margin: 0.38rem 0 0;
        color: #DCEEFF;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .kbu-hero-copy {
        padding: 1.6rem 0.4rem 1.2rem;
    }
    .kbu-hero-kicker {
        color: var(--kbu-pink);
        font-size: 0.82rem;
        font-weight: 850;
        letter-spacing: 0.08em;
    }
    .kbu-hero-title {
        margin: 0.35rem 0 0.7rem;
        color: var(--kbu-blue);
        font-size: clamp(2rem, 4vw, 3.5rem);
        font-weight: 900;
        line-height: 1.08;
        letter-spacing: -0.055em;
    }
    .kbu-hero-description {
        max-width: 760px;
        margin: 0;
        color: #49647D;
        font-size: 1.08rem;
        line-height: 1.75;
    }
    .kbu-pill {
        display: inline-block;
        margin: 0.85rem 0.35rem 0 0;
        padding: 0.42rem 0.72rem;
        color: #004085;
        background: #EAF3FB;
        border: 1px solid #CDE0F0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 750;
    }
    .kbu-flow-step {
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        min-height: 148px;
        height: 100%;
        padding: 1rem;
        background: white;
        border: 1px solid var(--kbu-line);
        border-radius: 0.85rem;
        box-shadow: 0 6px 20px rgba(0, 64, 133, 0.045);
    }
    .kbu-flow-number {
        color: var(--kbu-pink);
        font-size: 0.75rem;
        font-weight: 850;
        letter-spacing: 0.08em;
    }
    .kbu-flow-title {
        margin-top: 0.3rem;
        min-height: 1.65rem;
        color: var(--kbu-blue);
        font-weight: 800;
        line-height: 1.45;
    }
    .kbu-flow-copy {
        margin-top: 0.42rem;
        min-height: 3rem;
        color: var(--kbu-muted);
        font-size: 0.85rem;
        line-height: 1.65;
        word-break: keep-all;
        overflow-wrap: break-word;
    }
    .kbu-footer {
        margin-top: 3.5rem;
        padding-top: 1.1rem;
        color: #6A7F93;
        border-top: 1px solid var(--kbu-line);
        font-size: 0.82rem;
        text-align: center;
    }
    @media (prefers-reduced-motion: reduce) {
        .kbu-ai-reveal-dot { animation: none !important; }
    }
</style>
"""


def _render_navigation() -> None:
    """현재 인증 역할에 허용된 메뉴를 접을 수 있는 그룹으로 표시한다."""

    identity = get_authenticated_identity(st.session_state)
    if identity is None:
        return
    try:
        for group_label, items in navigation_groups_for_role(identity.role):
            with st.expander(
                group_label,
                expanded=navigation_group_expanded_by_default(group_label),
            ):
                for path, label, icon in items:
                    st.page_link(path, label=label, icon=icon)
    except KeyError:
        # Streamlit AppTest가 개별 page를 단독 실행할 때는 multipage registry가 없다.
        return


def apply_branding() -> None:
    """현재 Streamlit 페이지에 공통 KBU 브랜드 스타일과 사이드바를 적용한다."""

    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH), size="large")
    with st.sidebar:
        st.markdown(f"**{SERVICE_NAME}**")
        st.caption(f"{TEAM_NAME} · {AFFILIATION}")
        identity = get_authenticated_identity(st.session_state)
        if identity is not None:
            role_label = "학생" if identity.role == ROLE_STUDENT else "교직원"
            account_label = (
                identity.student_id
                if identity.role == ROLE_STUDENT
                else identity.account_id
            )
            st.caption(f"{role_label} 계정 · {account_label} · 시연용")
        st.divider()
        st.caption("SERVICE MENU")
        _render_navigation()
        st.divider()
        if identity is not None and st.button(
            "로그아웃",
            icon="↩️",
            key="kbu_logout",
            use_container_width=True,
        ):
            clear_authenticated_session(st.session_state)
            st.rerun()


def render_page_header(
    title: str,
    subtitle: str,
    audience: str = "교직원용",
) -> None:
    """페이지별 제목과 설명을 동일한 브랜드 hero로 표시한다."""

    apply_branding()
    st.markdown(
        f"""
        <section class="kbu-page-header">
            <div class="kbu-eyebrow">{escape(audience)} · {escape(TEAM_NAME)}</div>
            <h1 class="kbu-page-title">{escape(title)}</h1>
            <p class="kbu-page-subtitle">{escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_submenu(options: Sequence[str], key: str) -> str:
    """교직원 페이지에서 현재 선택한 기능 하나만 여는 가로형 2차 메뉴를 표시한다."""

    if not options:
        raise ValueError("소메뉴 항목은 하나 이상이어야 합니다.")
    selected = st.pills(
        "세부 메뉴",
        list(options),
        default=options[0],
        required=True,
        key=key,
        label_visibility="collapsed",
        width="content",
    )
    return str(selected or options[0])


def build_ai_reveal_chunks(
    text: str,
    words_per_chunk: int = 2,
) -> tuple[str, ...]:
    """검증된 AI 문장을 공백과 문단을 보존한 짧은 표시 단위로 나눈다.

    Parameters:
        text: JSON 검증 이후 UI에 표시해도 되는 AI 설명.
        words_per_chunk: 한 번에 나타낼 공백 기준 어절 수.

    Returns:
        다시 합치면 원문과 같은 순서의 문자열 조각.

    Assumptions:
        이 함수에는 provider 원문이 아니라 schema 검증을 통과한 문장만 전달한다.
    """

    if words_per_chunk < 1:
        raise ValueError("AI 설명 표시 단위는 1어절 이상이어야 합니다.")
    normalized = str(text)
    if not normalized:
        return ()
    tokens = re.findall(r"\S+\s*", normalized)
    return tuple(
        "".join(tokens[index : index + words_per_chunk])
        for index in range(0, len(tokens), words_per_chunk)
    )


def queue_ai_reveal(*keys: str) -> None:
    """새로 생성·검증된 AI 결과만 다음 렌더에서 한 번 애니메이션하도록 예약한다."""

    pending = [
        str(item)
        for item in st.session_state.get(AI_REVEAL_PENDING_STATE, [])
        if str(item).strip()
    ]
    for key in keys:
        normalized = str(key).strip()
        if normalized and normalized not in pending:
            pending.append(normalized)
    st.session_state[AI_REVEAL_PENDING_STATE] = pending


def render_ai_reveal(
    text: str,
    *,
    key: str,
    label: str = "검증 완료 · AI 생성 설명",
) -> bool:
    """검증된 AI 문장을 새 생성 시에만 ChatGPT처럼 순차 표시한다.

    Parameters:
        text: 애플리케이션 검증을 통과한 최종 설명 문장.
        key: 같은 결과의 중복 애니메이션을 막는 안정적인 식별자.
        label: AI 생성과 검증 완료 여부를 알리는 짧은 문구.

    Returns:
        이번 렌더에서 스트리밍 애니메이션을 실행했는지 여부.

    Assumptions:
        저장 결과 재조회는 pending key가 없으므로 즉시 표시한다.
    """

    content = str(text).strip()
    if not content:
        return False
    normalized_key = str(key).strip()
    pending = [
        str(item)
        for item in st.session_state.get(AI_REVEAL_PENDING_STATE, [])
    ]
    should_animate = normalized_key in pending
    if should_animate:
        pending.remove(normalized_key)
        st.session_state[AI_REVEAL_PENDING_STATE] = pending

    st.markdown(
        '<div class="kbu-ai-reveal-label">'
        '<span class="kbu-ai-reveal-dot"></span>'
        f"{escape(label)}</div>",
        unsafe_allow_html=True,
    )
    reveal_config = (
        load_app_config().get("ui", {}).get("ai_reveal", {})
    )
    enabled = bool(reveal_config.get("enabled", True))
    words_per_chunk = int(reveal_config.get("words_per_chunk", 2))
    delay_seconds = max(0.0, float(reveal_config.get("delay_seconds", 0.04)))
    if not should_animate or not enabled:
        st.markdown(content)
        return False

    chunks = build_ai_reveal_chunks(content, words_per_chunk=words_per_chunk)

    def stream_chunks():
        for index, chunk in enumerate(chunks):
            yield chunk
            if index < len(chunks) - 1 and delay_seconds:
                time.sleep(delay_seconds)

    st.write_stream(stream_chunks(), cursor="▌")
    return True


def render_home_hero() -> None:
    """대회·소속·서비스 메시지를 포함한 시작 화면 hero를 표시한다."""

    apply_branding()
    logo_column, copy_column = st.columns([1.05, 2.25], vertical_alignment="center")
    with logo_column:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width="stretch")
    with copy_column:
        st.markdown(
            f"""
            <section class="kbu-hero-copy">
                <div class="kbu-hero-kicker">{escape(COMPETITION_NAME)} · {escape(TEAM_NAME)}</div>
                <h1 class="kbu-hero-title">{escape(SERVICE_NAME)}</h1>
                <p class="kbu-hero-description">{escape(TAGLINE)}</p>
                <span class="kbu-pill">설명 가능한 AI</span>
                <span class="kbu-pill">Human-in-the-loop</span>
                <span class="kbu-pill">가상데이터 기반 시연</span>
            </section>
            """,
            unsafe_allow_html=True,
        )


def render_flow_step(
    number: str,
    title: str,
    description: str,
    label: str = "STEP",
) -> None:
    """서비스 흐름 한 단계를 간결한 카드로 표시한다."""

    st.markdown(
        f"""
        <div class="kbu-flow-step">
            <div class="kbu-flow-number">{escape(label)} {escape(number)}</div>
            <div class="kbu-flow-title">{escape(title)}</div>
            <div class="kbu-flow-copy">{escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """모든 페이지 하단에 팀과 데이터 원칙을 표시한다."""

    st.markdown(
        f"""
        <div class="kbu-footer">
            {escape(SERVICE_NAME)} · {escape(TEAM_NAME)} · {escape(AFFILIATION)}<br>
            현재 시연은 가상데이터를 사용하며 최종 지원 결정은 교직원이 검토합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def asset_path(filename: str) -> Path:
    """프로젝트 assets 파일의 절대 경로를 반환한다."""

    return PROJECT_ROOT / "assets" / filename
