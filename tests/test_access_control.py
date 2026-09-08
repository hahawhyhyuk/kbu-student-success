"""시연용 역할 검증과 학생 데이터 범위 제한 테스트."""

from __future__ import annotations

import pytest

from src.access_control import (
    AUTH_IDENTITY_KEY,
    DEMO_STAFF_ACCOUNT_ID,
    ROLE_STAFF,
    ROLE_STUDENT,
    STAFF_HOME_PAGE,
    STAFF_PAGE_PATHS,
    STUDENT_HOME_PAGE,
    STUDENT_PAGE_PATHS,
    allowed_page_paths,
    authenticate_demo_account,
    clear_authenticated_session,
    default_page_for_role,
    get_authenticated_identity,
    is_page_allowed,
    scoped_student_id,
    set_authenticated_identity,
)


def test_account_role_uses_master_membership_not_id_prefix() -> None:
    """학생 역할은 S 접두사가 아니라 Repository에 존재하는 ID로만 결정한다."""

    identity = authenticate_demo_account(" s0003 ", ["S0001", "S0003"])

    assert identity.role == ROLE_STUDENT
    assert identity.student_id == "S0003"
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        authenticate_demo_account("S9999", ["S0001", "S0003"])


def test_registered_staff_account_has_no_student_scope() -> None:
    """명시된 교직원 데모 계정만 교직원 역할로 인증한다."""

    identity = authenticate_demo_account(DEMO_STAFF_ACCOUNT_ID, ["S0003"])

    assert identity.role == ROLE_STAFF
    assert identity.student_id is None
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        authenticate_demo_account("STAFF999", ["S0003"])


def test_role_page_allowlists_are_disjoint_and_have_separate_defaults() -> None:
    """학생 페이지와 교직원 페이지는 라우터에서 서로 겹치지 않는다."""

    assert set(STUDENT_PAGE_PATHS).isdisjoint(STAFF_PAGE_PATHS)
    assert allowed_page_paths(ROLE_STUDENT) == STUDENT_PAGE_PATHS
    assert allowed_page_paths(ROLE_STAFF) == STAFF_PAGE_PATHS
    assert default_page_for_role(ROLE_STUDENT) == STUDENT_HOME_PAGE
    assert default_page_for_role(ROLE_STAFF) == STAFF_HOME_PAGE
    assert is_page_allowed(ROLE_STUDENT, "pages/07_student_view.py")
    assert is_page_allowed(ROLE_STUDENT, "pages/13_student_manual.py")
    assert not is_page_allowed(ROLE_STUDENT, "pages/01_dashboard.py")
    assert not is_page_allowed(ROLE_STUDENT, "pages/14_staff_manual.py")
    assert is_page_allowed(ROLE_STAFF, "pages/01_dashboard.py")
    assert is_page_allowed(ROLE_STAFF, "pages/03_ai_intervention.py")
    assert is_page_allowed(ROLE_STAFF, "pages/14_staff_manual.py")
    assert not is_page_allowed(ROLE_STAFF, "pages/13_student_manual.py")
    assert not is_page_allowed(ROLE_STAFF, "pages/04_learning_path.py")
    assert not is_page_allowed(ROLE_STAFF, "pages/12_student_mypage.py")


def test_student_scope_ignores_mutated_demo_selection() -> None:
    """학생 로그인 뒤 demo 선택값을 바꿔도 자기 ID만 조회한다."""

    session_state: dict[str, object] = {"demo_student_id": "S0001"}
    identity = authenticate_demo_account("S0003", ["S0001", "S0003"])
    set_authenticated_identity(session_state, identity)
    session_state["demo_student_id"] = "S0001"

    assert scoped_student_id(session_state, ["S0001", "S0003"]) == "S0003"
    assert get_authenticated_identity(session_state) == identity


def test_staff_cannot_receive_student_page_scope() -> None:
    """교직원 세션이 학생 전용 페이지에 도달하면 학생 데이터 조회를 중단한다."""

    session_state: dict[str, object] = {}
    set_authenticated_identity(
        session_state,
        authenticate_demo_account(DEMO_STAFF_ACCOUNT_ID, ["S0003"]),
    )

    with pytest.raises(PermissionError, match="학생 계정"):
        scoped_student_id(session_state, ["S0003"])


def test_logout_clears_role_and_previous_page_state() -> None:
    """역할 전환 시 이전 학생 대화나 선택 상태를 남기지 않는다."""

    session_state: dict[str, object] = {
        AUTH_IDENTITY_KEY: {
            "role": ROLE_STUDENT,
            "account_id": "S0003",
            "student_id": "S0003",
        },
        "demo_student_id": "S0003",
        "student_checkin_S0003_conversation_history": ["synthetic message"],
    }

    clear_authenticated_session(session_state)

    assert session_state == {}
