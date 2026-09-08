"""시연용 계정 식별과 역할별 화면 접근 범위를 관리한다.

현재 MVP는 실제 대학 인증시스템 대신 synthetic 학생 계정과 고정된 교직원
데모 계정을 사용한다. 운영 전환 시에는 ``authenticate_demo_account``만 대학
SSO/서버 세션 검증으로 교체하고, 아래 역할·페이지 정책은 그대로 재사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping


ROLE_STUDENT = "student"
ROLE_STAFF = "staff"
AUTH_IDENTITY_KEY = "_kbu_authenticated_identity"
DEMO_STAFF_ACCOUNT_ID = "STAFF001"

STUDENT_HOME_PAGE = "pages/07_student_view.py"
STAFF_HOME_PAGE = "pages/01_dashboard.py"
STUDENT_PAGE_PATHS: tuple[str, ...] = (
    "pages/07_student_view.py",
    "pages/12_student_mypage.py",
    "pages/13_student_manual.py",
)
STAFF_PAGE_PATHS: tuple[str, ...] = (
    "pages/00_service_intro.py",
    "pages/01_dashboard.py",
    "pages/02_student_detail.py",
    "pages/03_ai_intervention.py",
    "pages/05_intervention_management.py",
    "pages/06_microdegree_candidates.py",
    "pages/08_intervention_effect.py",
    "pages/09_demo_scenario.py",
    "pages/10_presentation_summary.py",
    "pages/11_alert_validation.py",
    "pages/14_staff_manual.py",
)


@dataclass(frozen=True)
class AuthenticatedIdentity:
    """검증을 통과해 브라우저 세션에 보관하는 최소 계정 정보."""

    role: str
    account_id: str
    student_id: str | None = None

    def as_session_value(self) -> dict[str, str | None]:
        """Streamlit session state에 저장할 직렬화 가능한 값을 반환한다."""

        return {
            "role": self.role,
            "account_id": self.account_id,
            "student_id": self.student_id,
        }


def authenticate_demo_account(
    account_id: str,
    valid_student_ids: Iterable[str],
) -> AuthenticatedIdentity:
    """synthetic 학번 또는 고정 교직원 사번을 역할로 해석한다.

    Parameters:
        account_id: 사용자가 입력한 시연용 학번 또는 사번.
        valid_student_ids: Repository 학생 master에 실제 존재하는 synthetic ID.

    Returns:
        검증된 역할과 학생 범위를 담은 ``AuthenticatedIdentity``.

    Assumptions:
        ID 접두사만으로 역할을 추측하지 않는다. 학생은 Repository 존재 여부,
        교직원은 명시적으로 등록한 데모 계정과 정확히 일치해야 한다.
    """

    normalized = str(account_id).strip().upper()
    if not normalized:
        raise ValueError("시연용 학번 또는 사번을 입력해 주세요.")

    students_by_normalized_id = {
        str(student_id).strip().upper(): str(student_id).strip()
        for student_id in valid_student_ids
        if str(student_id).strip()
    }
    if normalized in students_by_normalized_id:
        student_id = students_by_normalized_id[normalized]
        return AuthenticatedIdentity(
            role=ROLE_STUDENT,
            account_id=student_id,
            student_id=student_id,
        )
    if normalized == DEMO_STAFF_ACCOUNT_ID:
        return AuthenticatedIdentity(
            role=ROLE_STAFF,
            account_id=DEMO_STAFF_ACCOUNT_ID,
        )
    raise ValueError("등록된 시연용 계정을 찾지 못했습니다.")


def get_authenticated_identity(
    session_state: Mapping[str, object],
) -> AuthenticatedIdentity | None:
    """세션 값이 유효할 때만 인증된 계정으로 복원한다."""

    raw_identity = session_state.get(AUTH_IDENTITY_KEY)
    if not isinstance(raw_identity, Mapping):
        return None
    role = str(raw_identity.get("role", "")).strip()
    account_id = str(raw_identity.get("account_id", "")).strip()
    student_raw = raw_identity.get("student_id")
    student_id = str(student_raw).strip() if student_raw is not None else None
    if role not in {ROLE_STUDENT, ROLE_STAFF} or not account_id:
        return None
    if role == ROLE_STUDENT and (
        not student_id or account_id != student_id
    ):
        return None
    if role == ROLE_STAFF and (
        student_id or account_id != DEMO_STAFF_ACCOUNT_ID
    ):
        return None
    return AuthenticatedIdentity(
        role=role,
        account_id=account_id,
        student_id=student_id,
    )


def set_authenticated_identity(
    session_state: MutableMapping[str, object],
    identity: AuthenticatedIdentity,
) -> None:
    """검증된 계정을 세션에 저장하고 학생 조회 범위를 자기 ID로 고정한다."""

    session_state[AUTH_IDENTITY_KEY] = identity.as_session_value()
    if identity.role == ROLE_STUDENT and identity.student_id:
        session_state["demo_student_id"] = identity.student_id


def clear_authenticated_session(
    session_state: MutableMapping[str, object],
) -> None:
    """로그아웃할 때 역할 간 데이터가 남지 않도록 브라우저 상태를 모두 비운다."""

    for key in list(session_state.keys()):
        del session_state[key]


def allowed_page_paths(role: str) -> tuple[str, ...]:
    """역할별로 라우터에 등록할 페이지 경로를 반환한다."""

    if role == ROLE_STUDENT:
        return STUDENT_PAGE_PATHS
    if role == ROLE_STAFF:
        return STAFF_PAGE_PATHS
    return ()


def default_page_for_role(role: str) -> str:
    """로그인 직후 표시할 역할별 첫 화면을 반환한다."""

    if role == ROLE_STUDENT:
        return STUDENT_HOME_PAGE
    if role == ROLE_STAFF:
        return STAFF_HOME_PAGE
    raise ValueError(f"지원하지 않는 역할입니다: {role}")


def is_page_allowed(role: str, page_path: str) -> bool:
    """주어진 화면이 역할별 allowlist에 속하는지 확인한다."""

    return str(page_path) in allowed_page_paths(role)


def scoped_student_id(
    session_state: Mapping[str, object],
    available_student_ids: Iterable[str],
) -> str | None:
    """학생 계정이면 자기 ID만 반환하고 미인증 단독 테스트는 ``None``을 반환한다.

    인증된 교직원이 학생 전용 페이지에 도달하거나 로그인 학생이 master에서
    사라진 경우에는 데이터 노출을 막기 위해 ``PermissionError``를 발생시킨다.
    """

    identity = get_authenticated_identity(session_state)
    if identity is None:
        return None
    if identity.role != ROLE_STUDENT or not identity.student_id:
        raise PermissionError("학생 계정에서만 이용할 수 있는 화면입니다.")
    canonical_ids = {
        str(student_id).strip(): str(student_id).strip()
        for student_id in available_student_ids
    }
    if identity.student_id not in canonical_ids:
        raise PermissionError("현재 학생 계정을 master에서 확인할 수 없습니다.")
    return canonical_ids[identity.student_id]
