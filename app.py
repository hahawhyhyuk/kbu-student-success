"""역할을 확인한 뒤 학생·교직원 화면을 분리해 실행하는 Streamlit 진입점."""

from __future__ import annotations

import streamlit as st

from src.access_control import (
    DEMO_STAFF_ACCOUNT_ID,
    allowed_page_paths,
    authenticate_demo_account,
    default_page_for_role,
    get_authenticated_identity,
    set_authenticated_identity,
)
from src.repositories import get_default_repository
from src.ui import (
    SERVICE_NAME,
    apply_branding,
    navigation_items_for_role,
    render_footer,
)


@st.cache_data
def load_demo_student_ids() -> tuple[str, ...]:
    """역할 검증에 사용할 synthetic 학생 ID를 Repository에서 읽는다."""

    students = get_default_repository().get_students()
    return tuple(students["student_id"].astype(str).sort_values().tolist())


def sign_in_demo_account(account_id: str) -> None:
    """입력 계정을 검증해 세션에 저장하고 역할별 첫 화면으로 이동한다."""

    try:
        identity = authenticate_demo_account(account_id, load_demo_student_ids())
    except ValueError as error:
        st.error(str(error))
        return
    set_authenticated_identity(st.session_state, identity)
    st.rerun()


def render_role_entry() -> None:
    """실제 개인정보 없이 사용할 시연용 역할 진입 화면을 표시한다."""

    apply_branding()
    st.markdown(
        """
        <div style="max-width:760px;margin:2.2rem auto 1.8rem;text-align:center;">
            <div style="color:#006AB6;font-weight:800;letter-spacing:.08em;">
                KARE BRIDGE U · SYNTHETIC DEMO
            </div>
            <h1 style="margin:.7rem 0 .55rem;">이용할 계정을 확인해 주세요</h1>
            <p style="color:#60758A;line-height:1.7;">
                학번은 학생 서비스, 사번은 교직원 서비스로 연결됩니다.<br>
                현재 화면은 가상데이터만 사용하는 해커톤 시연용입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, entry_column, _ = st.columns([1, 1.25, 1])
    with entry_column:
        with st.container(border=True):
            with st.form("demo_account_sign_in"):
                account_id = st.text_input(
                    "시연용 학번 또는 사번",
                    placeholder="예: S0003 또는 STAFF001",
                )
                submitted = st.form_submit_button(
                    "계정으로 시작",
                    type="primary",
                    use_container_width=True,
                )
            if submitted:
                sign_in_demo_account(account_id)

            st.caption("빠른 시연 계정")
            student_column, staff_column = st.columns(2)
            if student_column.button(
                "학생 · S0003",
                use_container_width=True,
                key="quick_student_sign_in",
            ):
                sign_in_demo_account("S0003")
            if staff_column.button(
                f"교직원 · {DEMO_STAFF_ACCOUNT_ID}",
                use_container_width=True,
                key="quick_staff_sign_in",
            ):
                sign_in_demo_account(DEMO_STAFF_ACCOUNT_ID)

        st.info(
            "학생 계정은 자신의 체크인·결과만, 교직원 계정은 전체 학생 지원 화면만 볼 수 있습니다."
        )
        st.caption(
            "운영 전환 시 이 시연용 계정 확인은 대학 SSO와 서버 세션 검증으로 교체해야 합니다."
        )
    render_footer()


def build_role_pages(role: str) -> list[st.Page]:
    """현재 역할의 allowlist만 Streamlit 라우터에 등록한다."""

    allowed_paths = set(allowed_page_paths(role))
    page_items = navigation_items_for_role(role)
    configured_paths = {path for path, _, _ in page_items}
    if configured_paths != allowed_paths:
        raise RuntimeError("역할별 메뉴와 페이지 접근 정책이 일치하지 않습니다.")
    default_path = default_page_for_role(role)
    return [
        st.Page(
            path,
            title=label,
            icon=icon,
            default=path == default_path,
        )
        for path, label, icon in page_items
    ]


st.set_page_config(
    page_title=SERVICE_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

identity = get_authenticated_identity(st.session_state)
if identity is None:
    selected_page = st.navigation(
        [
            st.Page(
                render_role_entry,
                title="계정 확인",
                icon="🔐",
                default=True,
            )
        ],
        position="hidden",
    )
else:
    selected_page = st.navigation(build_role_pages(identity.role), position="hidden")

selected_page.run()
