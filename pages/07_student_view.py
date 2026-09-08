"""학생이 직접 체크인하고 맞춤 지원과 학습경로를 확인하는 화면."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.access_control import scoped_student_id
from src.ai import MockAIProvider, create_ai_provider
from src.checkin_chat_service import CheckinChatService
from src.checkin_progress import build_kare_context_progress_html
from src.checkin_conversation import (
    SCALE_QUESTIONS,
    build_review_sections,
    empty_chat_values,
    normalize_analysis_interests,
    scale_score,
    validate_narrative_values,
)
from src.checkin_semantic_state import (
    SEMANTIC_DIMENSIONS,
    SEMANTIC_DIMENSION_LABELS,
    confirm_semantic_state,
    semantic_coverage,
    semantic_edit_choices,
    semantic_state_label,
    update_semantic_state_from_review,
)
from src.checkin_service import CheckinAnalysisResult
from src.course_catalog_presenter import format_course_caption
from src.course_recommender import CourseRecommender, INTEREST_COMPETENCY_MAP
from src.demo_alert_notification_service import (
    STATUS_CHECKIN_COMPLETED,
    STATUS_KARE_STARTED,
    DemoAlertNotificationService,
)
from src.intervention_recommender import InterventionRecommender
from src.repositories import get_default_repository
from src.student_view_service import (
    ANALYSIS_FEEDBACK_TYPES,
    STUDENT_RECOMMENDATION_FEEDBACK_MAP,
    StudentJourneyResult,
    StudentViewService,
    describe_student_analysis,
)
from src.ui import (
    apply_branding,
    build_kare_chat_bubble_html,
    build_kare_typing_indicator_html,
    queue_ai_reveal,
    render_ai_reveal,
    render_footer,
)


KARE_CHAT_CSS = """
<style>
div[class*="st-key-kare_welcome_"] {
    box-sizing: border-box;
    max-width: 900px;
    min-height: calc(100vh - 10rem);
    margin: 0 auto;
    padding: clamp(3rem, 8vh, 5.6rem) 1rem 2rem;
    text-align: center;
}

.kare-orb {
    display: grid;
    place-items: center;
    width: 52px;
    height: 52px;
    margin: 0 auto 0.9rem;
    border-radius: 18px;
    color: #ffffff;
    font-size: 1.3rem;
    background: linear-gradient(135deg, #004085, #006ab6 58%, #ec008c);
    box-shadow: 0 12px 28px rgba(0, 64, 133, 0.18);
}

.kare-welcome-kicker {
    margin-bottom: 0.9rem;
    color: #006ab6;
    font-size: 0.76rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-align: center;
}

.kare-welcome-title {
    margin: 0 !important;
    padding: 0;
    color: #172b3d;
    border: 0;
    font-size: clamp(1.8rem, 3vw, 2.45rem);
    font-weight: 780;
    letter-spacing: -0.04em;
    text-align: center;
}

.kare-welcome-copy {
    max-width: 620px;
    margin: 0.8rem auto 1.55rem !important;
    color: #667b8f;
    line-height: 1.65;
    text-align: center;
}

div[class*="st-key-kare_consent_"] {
    align-items: center;
    box-sizing: border-box;
    max-width: 780px;
    margin: 0 auto 0.45rem;
    padding: 0 0.25rem;
    text-align: center;
}

div[class*="st-key-kare_consent_"] [data-testid="stCheckbox"] {
    width: fit-content;
    max-width: 100%;
    margin-inline: auto;
    text-align: left;
}

div[class*="st-key-kare_welcome_"] [data-testid="stChatInput"] {
    max-width: 780px;
    margin: 0 auto 0.85rem;
    border: 1px solid #d9e1e8;
    border-radius: 999px;
    background: #ffffff;
    box-shadow: 0 14px 36px rgba(23, 50, 77, 0.10);
}

div[class*="st-key-kare_welcome_"] [data-testid="stChatInput"] textarea {
    min-height: 3.25rem;
}

div[class*="st-key-kare_welcome_"] div.stButton > button,
div[class*="st-key-kare_welcome_"] [data-testid="stPageLink"] a {
    min-height: 2.65rem;
    border-radius: 999px;
    box-shadow: none;
}

.kare-quick-label {
    margin: 0.2rem 0 0.5rem;
    color: #718397;
    font-size: 0.78rem;
    font-weight: 700;
}

.kare-example-prompt {
    max-width: 780px;
    margin: 0.1rem auto 0.7rem;
    color: #7b8d9f;
    font-size: 0.84rem;
    text-align: center;
}

div[class*="st-key-kare_actions_"] {
    max-width: 600px;
    margin: 0 auto 0.15rem;
}

div[class*="st-key-kare_actions_"] div.stButton > button {
    color: #49647d;
    font-weight: 680;
}

div[class*="st-key-kare_privacy_"] {
    max-width: 780px;
    margin: 0.1rem auto 0;
    text-align: left;
}

.kare-compact-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
    max-width: 940px;
    margin: 0 auto 0.8rem;
    padding: 0.25rem 0.15rem 0.65rem;
    color: #657a8f;
    border-bottom: 1px solid #e3e9ef;
    font-size: 0.8rem;
}

.kare-compact-brand {
    color: #004085;
    font-weight: 850;
    letter-spacing: 0.06em;
}

.kare-chat-heading {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    margin-bottom: 0.65rem;
}

.kare-chat-heading strong {
    color: #173651;
    font-size: 1rem;
}

div[class*="st-key-kare_chat_header_controls_"] {
    margin-bottom: 0.55rem;
}

div[class*="st-key-kare_chat_header_controls_"] .kare-chat-heading {
    margin-bottom: 0;
}

div[class*="st-key-kare_chat_header_controls_"] [data-testid="stPopover"] button {
    min-height: 2.35rem;
    border-radius: 999px;
    box-shadow: none;
}

div[class*="st-key-kare_chat_header_controls_"] [data-testid="stColumn"]:last-child {
    display: flex;
    justify-content: flex-end;
}

div[class*="st-key-kare_chat_shell_"] {
    box-sizing: border-box;
    max-width: 860px;
    margin: 0 auto;
    padding: 1.15rem 1.25rem 1.3rem;
    border: 1px solid rgba(0, 64, 133, 0.12);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 16px 40px rgba(18, 55, 88, 0.06);
}

.kare-context-progress {
    box-sizing: border-box;
    margin: 0.15rem 0 0.9rem;
    padding: 0.15rem 0.05rem 0.1rem;
}

.kare-context-progress-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.kare-context-progress-title {
    color: #60758a;
    font-size: 0.8rem;
    font-weight: 700;
}

.kare-context-progress-stage {
    color: #006ab6;
    font-size: 0.76rem;
    font-weight: 800;
    white-space: nowrap;
}

.kare-context-progress-track {
    width: 100%;
    height: 0.46rem;
    overflow: hidden;
    border-radius: 999px;
    background: #eaf1f7;
}

.kare-context-progress-fill {
    display: block;
    width: var(--kare-context-progress, 0%);
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #006ab6, #59aee0);
    transition: width 520ms cubic-bezier(.22, .8, .32, 1);
    animation: kare-context-progress-in 520ms cubic-bezier(.22, .8, .32, 1) both;
}

div[class*="st-key-kare_messages_"] {
    gap: 0.1rem !important;
    margin: 0.8rem 0 1rem;
}

div[class*="st-key-kare_messages_"] > [data-testid="stElementContainer"] {
    padding-bottom: 0.8rem;
}

div[class*="st-key-kare_finish_action_"] {
    display: flex;
    box-sizing: border-box;
    align-items: flex-start;
    margin: -0.1rem 0 0.15rem;
    padding-left: 2.55rem;
}

div[class*="st-key-kare_finish_action_"] [data-testid="stButton"] {
    width: fit-content;
}

div[class*="st-key-kare_finish_action_"] div.stButton > button {
    min-height: 2.35rem;
    padding-inline: 1rem;
    border-color: #ccdbe8;
    border-radius: 999px;
    color: #49647d;
    background: #ffffff;
    box-shadow: none;
    font-weight: 700;
}

div[class*="st-key-kare_finish_action_"] div.stButton > button:hover {
    border-color: #8fb5d3;
    color: #004085;
    background: #f5f9fc;
}

.kare-message-row {
    display: flex;
    align-items: flex-end;
    box-sizing: border-box;
    width: 100%;
    margin: 0;
    padding: 0.28rem 0;
    gap: 0.55rem;
    animation: kare-message-in 300ms cubic-bezier(.22, .8, .32, 1) both;
}

.kare-message-user {
    flex-direction: row-reverse;
}

.kare-message-avatar {
    display: grid;
    flex: 0 0 2rem;
    place-items: center;
    width: 2rem;
    height: 2rem;
    border: 1px solid #dbe5ee;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 5px 14px rgba(23, 50, 77, 0.08);
}

.kare-message-bubble {
    box-sizing: border-box;
    width: max-content;
    max-width: min(var(--kare-bubble-max, 30rem), calc(100% - 2.65rem));
    padding: 0.78rem 0.95rem 0.82rem;
    overflow-wrap: anywhere;
    border: 1px solid #dbe6ef;
    border-radius: 18px 18px 18px 6px;
    color: #29455f;
    background: #f4f8fb;
    box-shadow: 0 6px 18px rgba(23, 50, 77, 0.06);
}

.kare-message-label {
    margin-bottom: 0.18rem;
    color: #006ab6;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.kare-message-user .kare-message-bubble {
    min-width: 5.75rem;
    border-color: #0061a7;
    border-radius: 18px 18px 6px 18px;
    color: #ffffff;
    background: #006ab6;
    box-shadow: 0 8px 20px rgba(0, 106, 182, 0.18);
}

.kare-message-user .kare-message-label {
    color: rgba(255, 255, 255, 0.78);
}

.kare-message-text {
    font-size: 0.95rem;
    line-height: 1.62;
}

.kare-message-typing .kare-message-bubble {
    min-width: 12.5rem;
    padding-block: 0.68rem 0.72rem;
}

.kare-typing-content,
.kare-typing-dots {
    display: inline-flex;
    align-items: center;
}

.kare-typing-content {
    gap: 0.55rem;
    color: #60758a;
    font-size: 0.84rem;
    line-height: 1.4;
}

.kare-typing-dots {
    gap: 0.22rem;
}

.kare-typing-dot {
    width: 0.34rem;
    height: 0.34rem;
    border-radius: 50%;
    background: #006ab6;
    animation: kare-typing-dot 1.15s ease-in-out infinite;
}

.kare-typing-dot:nth-child(2) { animation-delay: 140ms; }
.kare-typing-dot:nth-child(3) { animation-delay: 280ms; }

@keyframes kare-typing-dot {
    0%, 60%, 100% { opacity: 0.35; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-0.18rem); }
}

@keyframes kare-message-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes kare-context-progress-in {
    from { transform: scaleX(0); transform-origin: left; }
    to { transform: scaleX(1); transform-origin: left; }
}

.kare-persona-label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.45rem;
    color: #004085;
    font-size: 0.82rem;
    font-weight: 800;
}

@media (max-width: 760px) {
    div[class*="st-key-kare_welcome_"] {
        min-height: auto;
        padding: 1.8rem 0.15rem 1.3rem;
    }
    .kare-compact-header { align-items: flex-start; gap: 0.5rem; }
    div[class*="st-key-kare_chat_shell_"] { padding: 0.75rem; }
    div[class*="st-key-kare_chat_header_controls_"] [data-testid="stHorizontalBlock"] {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.5rem;
    }
    div[class*="st-key-kare_chat_header_controls_"] [data-testid="stColumn"] {
        flex: none !important;
        width: auto !important;
    }
    div[class*="st-key-kare_finish_action_"] {
        align-items: center;
        padding-left: 0;
    }
    .kare-context-progress-head {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.15rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    .kare-message-row,
    .kare-typing-dot,
    .kare-context-progress-fill {
        animation: none !important;
        transition: none !important;
    }
}
</style>
"""


def student_checkin_key(student_id: str, field: str) -> str:
    """학생별 체크인 입력과 단계 상태가 섞이지 않도록 widget key를 만든다."""

    return f"student_checkin_{student_id}_{field}"


def append_kare_message(history_key: str, role: str, content: str) -> None:
    """학생별 세션 대화 기록에 짧은 표시용 메시지를 추가한다."""

    history = list(st.session_state.get(history_key, []))
    history.append({"role": str(role), "content": str(content)})
    st.session_state[history_key] = history


def reset_kare_conversation(student_id: str, state_keys: tuple[str, ...]) -> None:
    """한 학생의 임시 대화와 conditional widget 상태만 초기화한다."""

    for key in state_keys:
        st.session_state.pop(key, None)
    widget_prefix = student_checkin_key(student_id, "response_")
    for key in list(st.session_state):
        if str(key).startswith(widget_prefix):
            st.session_state.pop(key, None)
    st.rerun()


@st.cache_data
def load_student_view_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """학생 선택과 결과 표시용 master 데이터를 Repository에서 로드한다."""

    source = get_default_repository().get_all()
    return source.students, source.support_programs, source.courses


@st.cache_resource
def get_program_recommender() -> InterventionRecommender:
    """프로그램 유사도 backend를 rerun 간 재사용한다."""

    return InterventionRecommender()


@st.cache_resource
def get_course_recommender() -> CourseRecommender:
    """교과목 유사도 backend를 rerun 간 재사용한다."""

    return CourseRecommender()


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


def build_ai_provider(allow_external_ai: bool):
    """학생의 명시적 동의 여부에 따라 외부 AI 또는 로컬 Mock을 선택한다."""

    if not allow_external_ai:
        return MockAIProvider()
    return create_ai_provider(api_key=resolve_gemini_api_key())


def get_student_service(allow_external_ai: bool = False) -> StudentViewService:
    """동의 범위에 맞는 AI와 기존 추천기를 결합한 학생 여정 서비스를 만든다."""

    return StudentViewService(
        repository=get_default_repository(),
        ai_provider=build_ai_provider(allow_external_ai),
        program_recommender=get_program_recommender(),
        course_recommender=get_course_recommender(),
    )


def get_checkin_chat_service(allow_external_ai: bool = False) -> CheckinChatService:
    """학생 식별정보 없이 자유대화를 처리할 Kare 서비스 인스턴스를 만든다."""

    return CheckinChatService(
        build_ai_provider(allow_external_ai),
        allowed_interest_fields=list(INTEREST_COMPETENCY_MAP),
    )


def render_student_alert_notification(
    service: DemoAlertNotificationService,
    student_id: str,
    conversation_mode: str,
) -> None:
    """발송된 시연 안내를 학생에게만 보여주고 Kare 행동 상태를 연결한다."""

    notification = service.get_notification(student_id)
    if notification is None:
        return
    notification = service.mark_student_viewed(student_id) or notification
    if conversation_mode != "welcome":
        notification = service.mark_kare_started(student_id) or notification

    with st.container(border=True, key=f"student_alert_{student_id}"):
        title_column, status_column = st.columns(
            [4, 1], vertical_alignment="center"
        )
        title_column.markdown("#### 학교에서 도착한 Kare 체크인 안내")
        status_column.metric("진행 상태", notification.status)
        st.write(notification.body)
        st.caption(
            "이 알림은 synthetic 계정의 시연용 교내 알림입니다. "
            "실제 이메일 주소나 학생 이름을 사용하지 않습니다."
        )
        if notification.status == STATUS_CHECKIN_COMPLETED:
            st.success("안내를 확인하고 Kare 체크인을 완료했어요.")
        elif notification.status == STATUS_KARE_STARTED:
            st.info("아래에서 AI 동의 대화 또는 외부 AI 전송 없는 직접 입력을 선택하세요.")
        elif st.button(
            "Kare와 이야기하기",
            type="primary",
            key=f"open_kare_from_alert_{student_id}",
        ):
            service.mark_kare_started(student_id)
            st.rerun()


def merge_analysis_into_draft(
    values: dict[str, object],
    analysis: dict[str, object],
) -> dict[str, object]:
    """종료 후 전체 대화 분석에서 명시된 선택 정보만 학생 확인값에 반영한다."""

    merged = dict(values)
    transcript = str(merged.get("natural_language_concern") or "")
    detected_interests = normalize_analysis_interests(
        analysis.get("interests", []),
        transcript=transcript,
        allowed_interests=list(INTEREST_COMPETENCY_MAP),
    )
    if detected_interests and not merged.get("interest_fields"):
        merged["interest_fields"] = list(dict.fromkeys(detected_interests))
    desired_jobs = [
        str(item).strip()
        for item in analysis.get("desired_jobs", [])
        if str(item).strip()
        and "".join(str(item).split()).lower()
        in "".join(transcript.split()).lower()
    ]
    if desired_jobs and not str(merged.get("desired_job") or "").strip():
        merged["desired_job"] = desired_jobs[0]
    consultation_terms = (
        "상담받고 싶",
        "상담 받고 싶",
        "상담을 받고 싶",
        "상담도 받고 싶",
        "상담을 받아",
        "상담 받아",
        "상담이 필요",
        "상담 희망",
    )
    if any(term in transcript for term in consultation_terms):
        merged["consultation_requested"] = True
    semantic_consultation = dict(
        dict(merged.get("semantic_states") or {}).get(
            "consultation_intent", {}
        )
    ).get("state")
    if semantic_consultation == "wants_support":
        merged["consultation_requested"] = True
    return merged


def process_kare_message(
    *,
    message: str,
    draft_key: str,
    history_key: str,
    mode_key: str,
    provider_key: str,
    warning_key: str,
    asked_dimensions_key: str,
    external_ai_consent: bool,
) -> None:
    """자유입력 한 턴을 처리하고 익명 대화·의미 상태를 갱신한다."""

    chat_service = get_checkin_chat_service(external_ai_consent)
    draft = dict(st.session_state[draft_key])
    result = chat_service.process_turn(
        history=list(st.session_state.get(history_key, [])),
        user_message=message,
        values=draft,
        semantic_state=draft.get("semantic_states"),
        asked_dimensions=tuple(
            st.session_state.get(asked_dimensions_key, ())
        ),
    )
    append_kare_message(history_key, "user", str(message).strip())
    append_kare_message(history_key, "assistant", result.assistant_message)
    transcript = chat_service.build_student_transcript(
        list(st.session_state.get(history_key, []))
    )
    draft = dict(result.values or draft)
    if transcript:
        draft["natural_language_concern"] = transcript
    st.session_state[draft_key] = draft
    st.session_state[asked_dimensions_key] = result.asked_dimensions
    st.session_state[provider_key] = result.provider_name
    if result.warning:
        st.session_state[warning_key] = result.warning
    else:
        st.session_state.pop(warning_key, None)
    st.session_state[mode_key] = (
        "preview" if result.ready_for_review else "conversation"
    )
    st.rerun()


st.set_page_config(page_title="나의 체크인", page_icon="🌱", layout="wide")
apply_branding()
st.markdown(KARE_CHAT_CSS, unsafe_allow_html=True)

students, support_programs, courses = load_student_view_data()
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
        key="student_view_student_id",
    )
st.session_state["demo_student_id"] = selected_student_id
selected_student = students[
    students["student_id"].astype(str) == selected_student_id
].iloc[0]
alert_notification_service = DemoAlertNotificationService()
st.sidebar.caption(
    f"가상 학생 계정 · {selected_student_id} · {selected_student['department']} · "
    f"{int(selected_student['grade'])}학년"
)

draft_key = student_checkin_key(selected_student_id, "draft")
mode_key = student_checkin_key(selected_student_id, "conversation_mode")
history_key = student_checkin_key(selected_student_id, "conversation_history")
consent_key = student_checkin_key(selected_student_id, "external_ai_consent")
consent_widget_key = student_checkin_key(
    selected_student_id, "external_ai_consent_widget"
)
input_mode_key = student_checkin_key(selected_student_id, "input_mode")
provider_key = student_checkin_key(selected_student_id, "chat_provider")
warning_key = student_checkin_key(selected_student_id, "chat_warning")
asked_dimensions_key = student_checkin_key(
    selected_student_id, "asked_semantic_dimensions"
)
preview_key = student_checkin_key(selected_student_id, "analysis_preview")
result_key = f"student_journey_result_{selected_student_id}"
conversation_state_keys = (
    draft_key,
    mode_key,
    history_key,
    consent_key,
    consent_widget_key,
    input_mode_key,
    provider_key,
    warning_key,
    asked_dimensions_key,
    preview_key,
)

if draft_key not in st.session_state:
    st.session_state[draft_key] = empty_chat_values()
st.session_state.setdefault(asked_dimensions_key, ())
st.session_state.setdefault(mode_key, "welcome")
st.session_state.setdefault(input_mode_key, "undecided")
if consent_key not in st.session_state:
    st.session_state[consent_key] = (
        str(st.session_state[input_mode_key]) == "ai_chat"
    )
submitted = False
conversation_mode = str(st.session_state[mode_key])
draft_values = dict(st.session_state[draft_key])
external_ai_consent = bool(st.session_state.get(consent_key, False))
input_mode = str(st.session_state.get(input_mode_key, "undecided"))

render_student_alert_notification(
    alert_notification_service,
    selected_student_id,
    conversation_mode,
)

if conversation_mode != "welcome":
    st.markdown(
        f"""
        <div class="kare-compact-header">
            <span class="kare-compact-brand">✦ KARE · 나의 체크인</span>
            <span>{selected_student_id} · {selected_student['department']} · {int(selected_student['grade'])}학년</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if conversation_mode == "welcome":
    first_message: str | None = None
    with st.container(key=f"kare_welcome_{selected_student_id}"):
        st.markdown(
            """
            <div class="kare-orb">✦</div>
            <div class="kare-welcome-kicker">KBU AI 학생성공 파트너 · KARE</div>
            <h1 class="kare-welcome-title">오늘은 어떤 이야기를 나눠볼까요?</h1>
            <p class="kare-welcome-copy">
                학교생활·전공·진로 중 지금 떠오르는 이야기부터 편하게 들려주세요.<br>
                한 문장에 여러 이야기를 해도 괜찮아요.
            </p>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key=f"kare_consent_{selected_student_id}"):
            if consent_widget_key not in st.session_state:
                st.session_state[consent_widget_key] = bool(
                    st.session_state[consent_key]
                )
            external_ai_consent = st.checkbox(
                "Kare AI 대화를 위해 입력 내용을 Gemini API로 전송하는 데 동의합니다.",
                key=consent_widget_key,
                help=(
                    "학번·이름은 전송하지 않고, 최근 대화와 "
                    "현재까지 정리된 내용만 사용해요."
                ),
            )
            st.session_state[consent_key] = bool(external_ai_consent)
        first_message = st.chat_input(
            (
                "Kare에게 지금 떠오르는 이야기를 들려주세요"
                if external_ai_consent
                else "AI 대화 이용 동의 후 입력할 수 있어요"
            ),
            disabled=not external_ai_consent,
            key=student_checkin_key(
                selected_student_id, "welcome_chat_input"
            ),
        )
        if external_ai_consent:
            st.markdown(
                '<div class="kare-quick-label">어떤 말부터 할지 고민된다면</div>',
                unsafe_allow_html=True,
            )
            prompt_columns = st.columns(3)
            quick_prompts = (
                "요즘 수업이나 과제가 조금 버거워요",
                "전공이 나와 맞는지 고민이에요",
                "진로를 아직 정하지 못했어요",
            )
            for index, (column, prompt) in enumerate(zip(prompt_columns, quick_prompts)):
                with column:
                    if st.button(
                        prompt,
                        width="stretch",
                        key=student_checkin_key(
                            selected_student_id, f"quick_prompt_{index}"
                        ),
                    ):
                        alert_notification_service.mark_kare_started(
                            selected_student_id
                        )
                        st.session_state[input_mode_key] = "ai_chat"
                        st.session_state[draft_key] = empty_chat_values()
                        st.session_state[asked_dimensions_key] = ()
                        st.session_state[history_key] = [
                            {
                                "role": "assistant",
                                "content": (
                                    "안녕하세요, Kare예요. 한 문장에 여러 이야기를 해도 괜찮아요. "
                                    "학생의 말에서 필요한 내용을 함께 정리해볼게요."
                                ),
                            }
                        ]
                        with st.spinner("Kare가 이야기를 이해하고 있어요..."):
                            process_kare_message(
                                message=prompt,
                                draft_key=draft_key,
                                history_key=history_key,
                                mode_key=mode_key,
                                provider_key=provider_key,
                                warning_key=warning_key,
                                asked_dimensions_key=asked_dimensions_key,
                                external_ai_consent=True,
                            )
        else:
            st.markdown(
                '<div class="kare-example-prompt">💬 예: 요즘 과제가 밀리고 전공이 나와 맞는지 고민이에요.</div>',
                unsafe_allow_html=True,
            )
        with st.container(key=f"kare_actions_{selected_student_id}"):
            action_columns = st.columns([1, 1])
            with action_columns[0]:
                if st.button(
                    "외부 AI 전송 없이 직접 입력",
                    type="tertiary",
                    width="stretch",
                    key=student_checkin_key(
                        selected_student_id, "start_direct"
                    ),
                ):
                    alert_notification_service.mark_kare_started(
                        selected_student_id
                    )
                    st.session_state[input_mode_key] = "direct"
                    st.session_state[draft_key] = empty_chat_values()
                    st.session_state[mode_key] = "direct"
                    st.rerun()
            with action_columns[1]:
                if st.button(
                    "최근 결과 다시 보기",
                    type="tertiary",
                    width="stretch",
                    key=student_checkin_key(
                        selected_student_id, "open_mypage"
                    ),
                ):
                    st.switch_page("pages/12_student_mypage.py")
        with st.container(key=f"kare_privacy_{selected_student_id}"):
            with st.expander("개인정보·AI 이용 안내", expanded=False):
                st.write(
                    "Gemini 동의는 선택입니다. 동의하지 않아도 외부 AI 전송 없이 "
                    "직접 입력할 수 있고, 동의 여부는 지원 우선순위에 사용되지 않아요. "
                    "대화 도중 동의를 철회하면 이후 전송을 중단하고 브라우저의 임시 대화를 삭제해요.\n\n"
                    "Kare는 학생을 평가하거나 진단하지 않으며, 최종 지원은 담당자가 검토해요."
                )
    if first_message:
        alert_notification_service.mark_kare_started(selected_student_id)
        st.session_state[input_mode_key] = "ai_chat"
        st.session_state[draft_key] = empty_chat_values()
        st.session_state[asked_dimensions_key] = ()
        st.session_state[history_key] = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요, Kare예요. 정해진 순서 없이 편하게 이야기해 주세요. "
                    "한 문장에 여러 이야기를 해도 괜찮아요."
                ),
            },
        ]
        with st.spinner("Kare가 이야기를 이해하고 있어요..."):
            process_kare_message(
                message=str(first_message),
                draft_key=draft_key,
                history_key=history_key,
                mode_key=mode_key,
                provider_key=provider_key,
                warning_key=warning_key,
                asked_dimensions_key=asked_dimensions_key,
                external_ai_consent=True,
            )

elif conversation_mode == "conversation":
    with st.container(key=f"kare_chat_shell_{selected_student_id}"):
        with st.container(key=f"kare_chat_header_controls_{selected_student_id}"):
            heading_column, settings_column = st.columns(
                [5, 1.35], vertical_alignment="center"
            )
            with heading_column:
                st.markdown(
                    """
                    <div class="kare-chat-heading">
                        <strong>✦ Kare와 이야기하는 중</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with settings_column:
                with st.popover(
                    "대화 설정",
                    type="tertiary",
                    icon=":material/tune:",
                    width="content",
                    key=student_checkin_key(
                        selected_student_id, "conversation_settings"
                    ),
                ):
                    st.markdown("**현재 대화 연결**")
                    st.caption(
                        "Gemini 대화 · 학생 ID와 이름은 전송하지 않습니다. "
                        "현재 사용: "
                        + str(st.session_state.get(provider_key, "연결 대기"))
                    )
                    if st.button(
                        "동의 철회하고 대화 삭제",
                        width="stretch",
                        key=student_checkin_key(
                            selected_student_id, "withdraw_consent"
                        ),
                    ):
                        reset_kare_conversation(
                            selected_student_id, conversation_state_keys
                        )
                    if st.button(
                        "외부 전송 없이 직접 입력으로 전환",
                        width="stretch",
                        key=student_checkin_key(
                            selected_student_id, "switch_direct"
                        ),
                    ):
                        st.session_state[consent_key] = False
                        st.session_state[input_mode_key] = "direct"
                        st.session_state[draft_key] = empty_chat_values()
                        st.session_state.pop(history_key, None)
                        st.session_state[mode_key] = "direct"
                        st.rerun()
                    st.caption(
                        "동의를 철회하면 이후 Gemini 전송을 중단하고 현재 브라우저의 "
                        "임시 대화를 삭제합니다. 이미 처리된 API 요청은 회수되지 않습니다."
                    )
        st.markdown(
            build_kare_context_progress_html(
                len(semantic_coverage(draft_values.get("semantic_states"))),
                total_count=len(SEMANTIC_DIMENSIONS),
            ),
            unsafe_allow_html=True,
        )
        with st.container(key=f"kare_messages_{selected_student_id}"):
            for message in st.session_state.get(history_key, []):
                st.markdown(
                    build_kare_chat_bubble_html(
                        str(message["role"]), str(message["content"])
                    ),
                    unsafe_allow_html=True,
                )
            typing_placeholder = st.empty()
        if str(draft_values.get("natural_language_concern") or "").strip():
            with st.container(
                key=f"kare_finish_action_{selected_student_id}"
            ):
                if st.button(
                    "이 정도면 정리할게요",
                    type="tertiary",
                    key=student_checkin_key(selected_student_id, "finish_chat"),
                ):
                    transcript = CheckinChatService.build_student_transcript(
                        list(st.session_state.get(history_key, []))
                    )
                    updated_draft = dict(st.session_state[draft_key])
                    updated_draft["natural_language_concern"] = transcript
                    st.session_state[draft_key] = updated_draft
                    st.session_state[mode_key] = "preview"
                    st.rerun()
        if st.session_state.get(warning_key):
            st.warning(st.session_state[warning_key])
    chat_message = st.chat_input(
        "Kare에게 이어서 이야기해 주세요",
        key=student_checkin_key(selected_student_id, "conversation_chat_input"),
    )
    if chat_message:
        typing_placeholder.markdown(
            build_kare_typing_indicator_html(), unsafe_allow_html=True
        )
        try:
            process_kare_message(
                message=str(chat_message),
                draft_key=draft_key,
                history_key=history_key,
                mode_key=mode_key,
                provider_key=provider_key,
                warning_key=warning_key,
                asked_dimensions_key=asked_dimensions_key,
                external_ai_consent=external_ai_consent,
            )
        finally:
            typing_placeholder.empty()

elif conversation_mode == "direct":
    with st.container(key=f"kare_chat_shell_{selected_student_id}"):
        st.subheader("외부 AI 전송 없는 기본 체크인")
        st.caption(
            "Gemini를 호출하지 않아요. 지금 도움받고 싶은 내용을 자연스럽게 작성해 주세요."
        )
        with st.form(f"direct_checkin_form_{selected_student_id}"):
            direct_concern = st.text_area(
                "나의 이야기",
                value=str(draft_values.get("natural_language_concern") or ""),
                height=170,
                placeholder=(
                    "예: 요즘 수업 속도가 빨라서 과제가 밀리고, "
                    "데이터 분석 직무를 어떻게 준비할지 궁금해요."
                ),
            )
            direct_interests = st.multiselect(
                "관심 분야",
                list(INTEREST_COMPETENCY_MAP),
                default=list(draft_values.get("interest_fields") or []),
            )
            direct_explore = st.checkbox(
                "새로운 분야도 탐색하고 싶어요",
                value=bool(draft_values.get("explore_other_fields", False)),
            )
            direct_job = st.text_input(
                "희망 직무", value=str(draft_values.get("desired_job") or "")
            )
            direct_consultation = st.checkbox(
                "진로 또는 학교생활 상담을 희망해요",
                value=bool(draft_values.get("consultation_requested", False)),
            )
            direct_submitted = st.form_submit_button(
                "외부 전송 없이 입력 내용 확인",
                type="primary",
                width="stretch",
            )
        if direct_submitted:
            direct_semantic_states = draft_values.get("semantic_states")
            for dimension, state in (
                (
                    "interest_fields",
                    "identified" if direct_interests else "none",
                ),
                (
                    "desired_job",
                    "identified" if str(direct_job).strip() else "none",
                ),
                (
                    "natural_language_concern",
                    "identified" if str(direct_concern).strip() else "none",
                ),
                (
                    "consultation_intent",
                    "wants_support" if direct_consultation else "no_support",
                ),
            ):
                direct_semantic_states = update_semantic_state_from_review(
                    direct_semantic_states,
                    dimension,
                    state,
                )
            draft_values.update(
                {
                    "interest_fields": list(direct_interests),
                    "explore_other_fields": bool(direct_explore),
                    "desired_job": str(direct_job).strip(),
                    "consultation_requested": bool(direct_consultation),
                    "natural_language_concern": str(direct_concern).strip(),
                    "response_mode": "narrative",
                    "semantic_states": direct_semantic_states,
                }
            )
            try:
                st.session_state[draft_key] = validate_narrative_values(draft_values)
                st.session_state[input_mode_key] = "direct"
                st.session_state[consent_key] = False
                st.session_state[mode_key] = "preview"
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        if st.button(
            "체크인 방식 다시 선택",
            key=student_checkin_key(selected_student_id, "direct_back"),
        ):
            st.session_state[mode_key] = "welcome"
            st.rerun()

elif conversation_mode == "preview":
    with st.container(key=f"kare_chat_shell_{selected_student_id}"):
        with st.container(key=f"kare_messages_{selected_student_id}"):
            for message in st.session_state.get(history_key, []):
                st.markdown(
                    build_kare_chat_bubble_html(
                        str(message["role"]), str(message["content"])
                    ),
                    unsafe_allow_html=True,
                )
        with st.chat_message("assistant", avatar="✨"):
            st.write(
                "지금까지의 이야기를 정리했어요. 저장하기 전에 제가 이해한 내용을 먼저 보여드릴게요."
            )
            st.caption(
                "점수를 추측하지 않았으며, 말씀한 내용은 아래 확인 화면에서 직접 수정할 수 있어요."
            )
        preview_column, reset_column = st.columns([3, 1])
        with preview_column:
            if st.button(
                "Kare가 이해한 내용 확인",
                type="primary",
                width="stretch",
                key=student_checkin_key(selected_student_id, "generate_preview"),
            ):
                try:
                    checked_values = validate_narrative_values(draft_values)
                    with st.spinner("Kare가 대화 내용을 정리하고 있어요..."):
                        allow_external_ai = (
                            input_mode == "ai_chat" and external_ai_consent
                        )
                        preview_result = get_student_service(
                            allow_external_ai=allow_external_ai
                        ).preview_checkin_analysis(
                            checked_values["natural_language_concern"]
                        )
                        if not allow_external_ai:
                            preview_result = CheckinAnalysisResult(
                                analysis=preview_result.analysis,
                                provider_name=preview_result.provider_name,
                                fallback_used=True,
                                warning=(
                                    "외부 AI로 전송하지 않고 로컬 규칙으로 내용을 정리했습니다."
                                ),
                            )
                        checked_values = validate_narrative_values(
                            merge_analysis_into_draft(
                                checked_values,
                                preview_result.analysis,
                            )
                        )
                        st.session_state[draft_key] = checked_values
                        st.session_state[preview_key] = preview_result
                        st.session_state[mode_key] = "review"
                        queue_ai_reveal(
                            f"kare_checkin_preview_{selected_student_id}"
                        )
                    st.rerun()
                except Exception as error:
                    st.error(f"대화 내용을 정리하지 못했습니다. ({error})")
        with reset_column:
            if st.button(
                "처음부터",
                width="stretch",
                key=student_checkin_key(selected_student_id, "preview_restart"),
            ):
                reset_kare_conversation(
                    selected_student_id, conversation_state_keys
                )

elif conversation_mode == "review":
    preview_result: CheckinAnalysisResult | None = st.session_state.get(preview_key)
    if preview_result is None:
        st.session_state[mode_key] = "preview"
        st.rerun()
    checked_values = validate_narrative_values(draft_values)
    review_sections = build_review_sections(checked_values)
    with st.container(key=f"kare_chat_shell_{selected_student_id}"):
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(
                "### 제가 이렇게 이해했어요"
                if input_mode == "ai_chat"
                else "### 입력 내용을 이렇게 정리했어요"
            )
            for message in describe_student_analysis(preview_result.analysis):
                st.markdown(f"- {message}")
            render_ai_reveal(
                f"**Kare의 요약:** {preview_result.analysis['summary']}",
                key=f"kare_checkin_preview_{selected_student_id}",
                label=f"검증 완료 · {preview_result.provider_name} AI 이해 결과",
            )
            if preview_result.warning:
                st.warning(preview_result.warning)
            st.caption(
                "아래 구조화된 응답을 학생이 확인한 후에만 체크인이 저장됩니다."
            )
        with st.expander("내가 답한 내용 자세히 보기", expanded=False):
            for section_title, lines in review_sections.items():
                st.markdown(f"**{section_title}**")
                for line in lines:
                    st.markdown(f"- {line}")
        submit_column, edit_column, restart_column = st.columns(3)
        with submit_column:
            submitted = st.button(
                "맞아요, 제출할게요",
                type="primary",
                width="stretch",
                key=student_checkin_key(selected_student_id, "confirm_submit"),
            )
        with edit_column:
            if st.button(
                "조금 수정할게요",
                width="stretch",
                key=student_checkin_key(selected_student_id, "edit_answers"),
            ):
                st.session_state[mode_key] = "edit"
                st.rerun()
        with restart_column:
            if st.button(
                "다시 이야기할게요",
                width="stretch",
                key=student_checkin_key(selected_student_id, "review_restart"),
            ):
                reset_kare_conversation(
                    selected_student_id, conversation_state_keys
                )

elif conversation_mode == "edit":
    with st.container(key=f"kare_chat_shell_{selected_student_id}"):
        with st.chat_message("assistant", avatar="✨"):
            st.write("물론이에요. 바꾸고 싶은 내용을 수정한 뒤 다시 확인해 주세요.")
        with st.form(f"kare_edit_form_{selected_student_id}"):
            edit_concern = st.text_area(
                "나의 이야기",
                value=str(draft_values.get("natural_language_concern", "")),
                height=180,
            )
            edit_interests = st.multiselect(
                "관심 분야",
                list(INTEREST_COMPETENCY_MAP),
                default=list(draft_values.get("interest_fields", [])),
            )
            edit_explore = st.checkbox(
                "새로운 분야도 탐색하고 싶어요",
                value=bool(draft_values.get("explore_other_fields", False)),
            )
            edit_job = st.text_input(
                "희망 직무", value=str(draft_values.get("desired_job", ""))
            )
            edit_consultation = st.checkbox(
                "상담을 희망해요",
                value=bool(draft_values.get("consultation_requested", False)),
            )
            semantic_edit_responses: dict[str, str] = {}
            semantic_values = draft_values.get("semantic_states", {})
            with st.expander("대화에서 이해한 나의 상태 수정"):
                st.caption(
                    "점수가 아닌 대화 맥락입니다. Kare의 이해가 다르면 "
                    "학생이 직접 바꿔주세요."
                )
                for dimension in SEMANTIC_DIMENSIONS[:6]:
                    choices = semantic_edit_choices(dimension)
                    current_state = str(
                        dict(semantic_values.get(dimension, {})).get(
                            "state", "unknown"
                        )
                    )
                    current_index = (
                        choices.index(current_state)
                        if current_state in choices
                        else 0
                    )
                    semantic_edit_responses[dimension] = st.selectbox(
                        SEMANTIC_DIMENSION_LABELS[dimension],
                        choices,
                        index=current_index,
                        format_func=lambda state, field=dimension: (
                            semantic_state_label(field, state)
                        ),
                        key=student_checkin_key(
                            selected_student_id,
                            f"semantic_edit_{dimension}",
                        ),
                    )
            explicit_scale_fields = [
                question
                for question in SCALE_QUESTIONS
                if draft_values.get(question.field) is not None
            ]
            edit_scale_responses: dict[str, str] = {}
            if explicit_scale_fields:
                with st.expander("대화에서 직접 말한 1~5 값 수정 (선택)"):
                    for question in explicit_scale_fields:
                        edit_scale_responses[question.field] = st.selectbox(
                            question.short_label,
                            question.options,
                            index=int(draft_values[question.field]) - 1,
                        )
            edit_submitted = st.form_submit_button(
                "수정 내용을 반영하고 다시 확인",
                type="primary",
                width="stretch",
            )
        if edit_submitted:
            for field, response in edit_scale_responses.items():
                draft_values[field] = scale_score(field, response)
            updated_semantic_states = draft_values.get("semantic_states")
            for dimension, selected_state in semantic_edit_responses.items():
                updated_semantic_states = update_semantic_state_from_review(
                    updated_semantic_states,
                    dimension,
                    selected_state,
                )
            updated_semantic_states = update_semantic_state_from_review(
                updated_semantic_states,
                "interest_fields",
                "identified" if edit_interests else "none",
            )
            updated_semantic_states = update_semantic_state_from_review(
                updated_semantic_states,
                "desired_job",
                "identified" if str(edit_job).strip() else "none",
            )
            updated_semantic_states = update_semantic_state_from_review(
                updated_semantic_states,
                "natural_language_concern",
                "identified" if str(edit_concern).strip() else "none",
            )
            updated_semantic_states = update_semantic_state_from_review(
                updated_semantic_states,
                "consultation_intent",
                "wants_support" if edit_consultation else "no_support",
            )
            draft_values.update(
                {
                    "interest_fields": list(edit_interests),
                    "explore_other_fields": bool(edit_explore),
                    "desired_job": str(edit_job).strip(),
                    "consultation_requested": bool(edit_consultation),
                    "natural_language_concern": str(edit_concern).strip(),
                    "response_mode": "narrative",
                    "semantic_states": updated_semantic_states,
                }
            )
            st.session_state[draft_key] = validate_narrative_values(
                draft_values
            )
            st.session_state.pop(preview_key, None)
            st.session_state[mode_key] = "preview"
            append_kare_message(
                history_key,
                "assistant",
                "수정한 내용을 반영했어요. AI 이해 결과를 다시 확인해 주세요.",
            )
            st.rerun()

else:
    with st.container(key=f"kare_chat_shell_{selected_student_id}"):
        with st.chat_message("assistant", avatar="✨"):
            st.success(
                "체크인이 저장되었어요. 아래에서 맞춤 지원과 학습경로를 확인해 주세요."
            )
        if st.button(
            "새 체크인 시작",
            key=student_checkin_key(selected_student_id, "start_new_checkin"),
        ):
            reset_kare_conversation(
                selected_student_id, conversation_state_keys
            )

if submitted:
    submission_values = dict(st.session_state[draft_key])
    submission_values["semantic_states"] = confirm_semantic_state(
        submission_values.get("semantic_states")
    )
    st.session_state[draft_key] = submission_values
    values = validate_narrative_values(submission_values)
    try:
        with st.spinner("응답을 이해하고 맞춤 지원을 찾고 있어요..."):
            new_result = get_student_service(
                allow_external_ai=(
                    input_mode == "ai_chat" and external_ai_consent
                )
            ).submit_checkin(
                selected_student_id,
                values,
                analysis_result=st.session_state.get(preview_key),
            )
            st.session_state[result_key] = new_result
            st.session_state[mode_key] = "submitted"
            reveal_keys = [f"student_analysis_{new_result.checkin_id}"]
            if new_result.learning_path_result is not None:
                reveal_keys.append(f"student_learning_path_{new_result.checkin_id}")
            queue_ai_reveal(*reveal_keys)
        st.success(
            "체크인과 맞춤 결과가 저장되어 교직원 검토 대기 상태로 전달되었습니다. "
            "아래에서 같은 결과를 확인해 주세요."
        )
    except Exception as error:
        st.error(f"체크인을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요. ({error})")

stored_result: StudentJourneyResult | None = st.session_state.get(result_key)
if stored_result is None:
    if conversation_mode != "conversation":
        st.caption(
            "체크인을 제출하면 AI가 이해한 고민, 지원 영역과 맞춤 추천이 표시됩니다."
        )
        render_footer()
    st.stop()

analysis_result = stored_result.analysis_result
analysis = analysis_result.analysis
if analysis_result.warning:
    st.warning(analysis_result.warning)
if stored_result.recommendation_result.warning:
    st.warning(stored_result.recommendation_result.warning)

st.subheader("내 고민이 맞춤 지원으로 연결된 과정")
input_column, first_arrow, understanding_column, second_arrow, connection_column = (
    st.columns([1, 0.08, 1, 0.08, 1])
)
with input_column:
    with st.container(border=True):
        st.caption("STEP 01 · 학생의 말")
        st.write(
            str(stored_result.profile.get("natural_language_concern") or "작성 내용 없음")
        )
with first_arrow:
    st.markdown("<div style='padding-top:3.2rem;text-align:center;font-size:1.5rem;color:#EC008C'>→</div>", unsafe_allow_html=True)
with understanding_column:
    with st.container(border=True):
        st.caption("STEP 02 · AI 이해")
        for message in describe_student_analysis(analysis):
            st.markdown(f"- {message}")
        st.caption(f"검증된 구조화 결과 · {analysis_result.provider_name}")
with second_arrow:
    st.markdown("<div style='padding-top:3.2rem;text-align:center;font-size:1.5rem;color:#006AB6'>→</div>", unsafe_allow_html=True)
with connection_column:
    with st.container(border=True):
        st.caption("STEP 03 · 맞춤 연결")
        for recommendation in stored_result.recommendation_result.recommendations:
            st.markdown(f"- {recommendation.program_name}")
        if stored_result.learning_path_result is not None:
            st.markdown(
                f"- 학습경로: {stored_result.learning_path_result.explanation['path_name']}"
            )
        st.caption("프로그램·교과목 ID는 대학 DB master에서만 선택")

st.subheader("AI가 이해한 나의 고민")
with st.container(border=True):
    for message in describe_student_analysis(analysis):
        st.markdown(f"- {message}")
    render_ai_reveal(
        f"**요약:** {analysis['summary']}",
        key=f"student_analysis_{stored_result.checkin_id}",
        label=f"검증 완료 · {analysis_result.provider_name} AI 요약",
    )
    st.caption(
        "AI의 이해가 정확하지 않을 수 있어요. 아래 응답은 담당자가 지원 방향을 검토할 때 함께 확인합니다."
    )

with st.form(f"analysis_feedback_{stored_result.checkin_id}"):
    analysis_feedback = st.radio(
        "AI가 이해한 내용이 맞나요?",
        ANALYSIS_FEEDBACK_TYPES,
        horizontal=True,
    )
    analysis_feedback_comment = st.text_input(
        "다르게 이해한 부분이나 수정할 내용 (선택)"
    )
    analysis_feedback_submitted = st.form_submit_button("AI 이해 결과 의견 보내기")
if analysis_feedback_submitted:
    get_student_service().record_analysis_feedback(
        selected_student_id,
        stored_result.checkin_id,
        analysis_feedback,
        analysis_feedback_comment,
    )
    st.success("의견이 저장되었습니다.")

st.subheader("현재 도움이 필요한 영역")
area_columns = st.columns(5)
for column, area in zip(area_columns, stored_result.support_areas):
    with column:
        with st.container(border=True):
            st.markdown(f"**{area.label}**")
            st.markdown(f"### {area.status}")
            st.caption(area.guidance)

st.subheader("나에게 맞는 지원")
program_by_id = {
    str(row["program_id"]): row for row in support_programs.to_dict(orient="records")
}
recommendations = stored_result.recommendation_result.recommendations
for rank, recommendation in enumerate(recommendations, start=1):
    program = program_by_id.get(recommendation.program_id)
    if program is None:
        continue
    with st.container(border=True):
        st.markdown(f"### {rank}. {recommendation.program_name}")
        st.caption(
            f"{recommendation.program_type} · 담당 {recommendation.department_in_charge}"
        )
        st.write(recommendation.description)
        st.markdown(f"**추천 이유:** {recommendation.reason}")
        competencies = str(program.get("provided_competencies", "")).replace(
            "|", " · "
        )
        st.markdown(f"**제공 지원·역량:** {competencies or '프로그램 상담 시 안내'}")
        application_url = str(program.get("application_url", "")).strip()
        if application_url:
            st.link_button(
                "WINGS에서 세부 조건 확인",
                application_url,
                key=f"student_wings_{recommendation.program_id}",
            )

recommendation_lookup = {
    item.program_id: item for item in recommendations
}
with st.form(f"program_feedback_{stored_result.checkin_id}"):
    selected_program_id = st.selectbox(
        "의견을 남길 지원 프로그램",
        list(recommendation_lookup),
        format_func=lambda program_id: recommendation_lookup[program_id].program_name,
    )
    program_feedback = st.radio(
        "이 추천이 어떤가요?",
        list(STUDENT_RECOMMENDATION_FEEDBACK_MAP),
        horizontal=True,
    )
    program_feedback_comment = st.text_input(
        "추천에 대한 의견 (선택)", key="student_program_feedback_comment"
    )
    program_feedback_submitted = st.form_submit_button("지원 추천 의견 저장")
if program_feedback_submitted:
    recommendation_id = stored_result.intervention_context.recommendation_ids[
        selected_program_id
    ]
    get_student_service().record_recommendation_feedback(
        selected_student_id,
        recommendation_id,
        program_feedback,
        program_feedback_comment,
    )
    st.success("지원 추천에 대한 의견이 저장되었습니다.")

st.subheader("AI 맞춤형 역량 학습경로")
learning_path = stored_result.learning_path_result
if learning_path is None:
    if stored_result.learning_path_error:
        st.warning(
            "학습경로를 준비하는 중 문제가 발생했습니다. 지원 프로그램은 정상적으로 저장되었으며, "
            f"경로는 다시 확인할 수 있어요. ({stored_result.learning_path_error})"
        )
    else:
        st.info(
            "현재 체크인에서는 별도 학습경로를 우선 제안하지 않았어요. 다른 분야를 탐색하고 싶다면 "
            "체크인에서 해당 항목을 선택해 주세요."
        )
else:
    if learning_path.selection.warning:
        st.warning(learning_path.selection.warning)
    if learning_path.warning:
        st.warning(learning_path.warning)
    explanation = learning_path.explanation
    st.markdown(f"## {explanation['path_name']}")
    render_ai_reveal(
        (
            f"{explanation['reason']}\n\n"
            f"**희망 직무와의 연결:** {explanation['career_connection']}"
        ),
        key=f"student_learning_path_{stored_result.checkin_id}",
        label=f"검증 완료 · {learning_path.provider_name} 학습경로 설명",
    )
    if explanation["competencies"]:
        st.markdown("**습득 가능한 역량:** " + " · ".join(explanation["competencies"]))
    else:
        st.info(
            "원본 교과목에 역량 태그가 없어 과목명과 실제 개설정보로 "
            "학습경로를 설명합니다."
        )
    roles = {
        str(item["course_id"]): item for item in explanation["course_roles"]
    }
    course_master_ids = set(courses["course_id"].astype(str))
    for course in learning_path.selection.courses:
        if course.course_id not in course_master_ids:
            continue
        role = roles[course.course_id]
        with st.container(border=True):
            st.markdown(f"### {role['sequence']}. {course.course_name}")
            st.caption(format_course_caption(course))
            st.markdown(f"**과목별 역할:** {role['role']}")
            if course.competencies:
                st.markdown(f"**습득 역량:** {' · '.join(course.competencies)}")
            st.markdown(f"**선정 이유:** {course.reason}")

    if stored_result.learning_path_recommendation_id is not None:
        with st.form(f"path_feedback_{stored_result.checkin_id}"):
            path_feedback = st.radio(
                "이 학습경로가 어떤가요?",
                list(STUDENT_RECOMMENDATION_FEEDBACK_MAP),
                horizontal=True,
                key="student_path_feedback",
            )
            path_feedback_comment = st.text_input(
                "학습경로에 대한 의견 (선택)", key="student_path_feedback_comment"
            )
            path_feedback_submitted = st.form_submit_button("학습경로 의견 저장")
        if path_feedback_submitted:
            get_student_service().record_recommendation_feedback(
                selected_student_id,
                stored_result.learning_path_recommendation_id,
                path_feedback,
                path_feedback_comment,
            )
            st.success("학습경로에 대한 의견이 저장되었습니다.")

st.divider()
st.caption(
    "추천 결과는 선택을 돕기 위한 안내입니다. 실제 프로그램 참여와 수강 가능 여부는 담당 부서와 확인해 주세요."
)
render_footer()
