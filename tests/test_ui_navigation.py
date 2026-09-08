"""핵심 업무와 보조 도구를 구분한 공통 내비게이션 테스트."""

from __future__ import annotations

import tomllib

from src.ui import (
    BRAND_CSS,
    DEFAULT_START_PAGE,
    STAFF_NAVIGATION_GROUPS,
    STUDENT_NAVIGATION_GROUPS,
    navigation_group_expanded_by_default,
    navigation_items_for_role,
)
from src.access_control import ROLE_STAFF, ROLE_STUDENT
from src.utils import PROJECT_ROOT


EXPECTED_PAGE_PATHS = {
    "pages/00_service_intro.py",
    "pages/01_dashboard.py",
    "pages/02_student_detail.py",
    "pages/03_ai_intervention.py",
    "pages/05_intervention_management.py",
    "pages/06_microdegree_candidates.py",
    "pages/07_student_view.py",
    "pages/08_intervention_effect.py",
    "pages/09_demo_scenario.py",
    "pages/10_presentation_summary.py",
    "pages/11_alert_validation.py",
    "pages/12_student_mypage.py",
    "pages/13_student_manual.py",
    "pages/14_staff_manual.py",
}


def test_navigation_separates_student_and_staff_pages_without_omissions() -> None:
    """학생·교직원 메뉴를 분리하며 통합된 현재 서비스만 등록한다."""

    student_items = navigation_items_for_role(ROLE_STUDENT)
    staff_items = navigation_items_for_role(ROLE_STAFF)
    all_items = [*student_items, *staff_items]
    paths = [path for path, _, _ in all_items]

    assert [label for label, _ in STUDENT_NAVIGATION_GROUPS] == [
        "학생 서비스"
    ]
    assert [label for label, _ in STAFF_NAVIGATION_GROUPS] == [
        "교직원 핵심 흐름",
        "교육과정",
        "관리자",
    ]
    assert len(student_items) == 3
    assert len(staff_items) == 11
    assert set(path for path, _, _ in student_items).isdisjoint(
        path for path, _, _ in staff_items
    )
    assert set(paths) == EXPECTED_PAGE_PATHS
    assert len(paths) == len(set(paths))
    assert student_items[0][1] == "나의 체크인"
    assert staff_items[0][1] == "전체 대시보드"
    assert student_items[-1][1] == "이용 매뉴얼"
    assert staff_items[5][1] == "이용 매뉴얼"
    assert "pages/04_learning_path.py" not in paths
    assert any(label == "AI 맞춤 추천 통합 검토" for _, label, _ in staff_items)

    groups = dict(STAFF_NAVIGATION_GROUPS)
    assert [label for _, label, _ in groups["교육과정"]] == [
        "신규 마이크로디그리 개발 후보"
    ]
    assert {
        label for _, label, _ in groups["관리자"]
    } == {
        "서비스 소개",
        "시연 준비",
        "발표 요약",
        "초기경보 모델 검증",
    }


def test_navigation_groups_are_collapsible_with_admin_closed_by_default() -> None:
    """업무 그룹은 바로 보이고 관리자 도구는 사용자가 펼칠 때만 보인다."""

    assert navigation_group_expanded_by_default("학생 서비스")
    assert navigation_group_expanded_by_default("교직원 핵심 흐름")
    assert navigation_group_expanded_by_default("교육과정")
    assert not navigation_group_expanded_by_default("관리자")


def test_student_role_defaults_to_checkin() -> None:
    """학생 계정의 첫 화면은 대표 기능인 나의 체크인이다."""

    assert DEFAULT_START_PAGE == "pages/07_student_view.py"


def test_global_layout_hides_deploy_toolbar_and_reserves_header_space() -> None:
    """개발 툴바가 페이지 헤더를 가리지 않도록 공통 설정을 유지한다."""

    config = tomllib.loads(
        (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(
            encoding="utf-8"
        )
    )

    assert config["client"]["toolbarMode"] == "minimal"
    assert "padding-top: 4rem;" in BRAND_CSS
    assert '[data-testid="stMetricValue"]' in BRAND_CSS
    assert "white-space: normal !important;" in BRAND_CSS
    assert "overflow-wrap: anywhere;" in BRAND_CSS
    assert "text-overflow: clip !important;" in BRAND_CSS
