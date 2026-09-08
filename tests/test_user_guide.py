"""역할별 이용 매뉴얼 범위와 Streamlit 표시 테스트."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.access_control import ROLE_STAFF, ROLE_STUDENT
from src.user_guide import (
    GUIDE_CATEGORIES,
    MANUAL_PAGE_PATHS,
    SERVICE_GUIDES,
    documented_paths,
    guide_categories_for_role,
    guides_for_role,
)
from src.utils import PROJECT_ROOT


EXPECTED_DOCUMENTED_PATHS = {
    "app.py",
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


def test_role_guides_document_every_active_page_without_cross_role_content() -> None:
    """두 매뉴얼과 모든 업무 화면을 문서화하되 역할별 안내는 섞지 않는다."""

    assert documented_paths() == EXPECTED_DOCUMENTED_PATHS
    assert MANUAL_PAGE_PATHS.issubset(documented_paths())
    assert len(SERVICE_GUIDES) == len(EXPECTED_DOCUMENTED_PATHS) - 2
    assert {guide.category for guide in SERVICE_GUIDES} == set(GUIDE_CATEGORIES)

    student_guides = guides_for_role(ROLE_STUDENT)
    staff_guides = guides_for_role(ROLE_STAFF)
    assert {guide.title for guide in student_guides} == {
        "나의 체크인",
        "마이페이지",
    }
    assert len(staff_guides) == 10
    assert {guide.path for guide in student_guides}.isdisjoint(
        guide.path for guide in staff_guides
    )
    assert guide_categories_for_role(ROLE_STUDENT) == ("시작 · 학생 서비스",)
    assert guide_categories_for_role(ROLE_STAFF) == (
        "교직원 서비스",
        "교육과정",
        "관리자",
    )
    for guide in SERVICE_GUIDES:
        assert guide.purpose
        assert guide.inputs
        assert guide.actions
        assert guide.outputs
        assert guide.next_step


def test_student_manual_renders_student_flow_and_privacy_choices() -> None:
    """학생 매뉴얼은 체크인·결과 확인과 AI 동의 선택만 안내한다."""

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "13_student_manual.py")
    ).run(timeout=30)

    assert not page.exception
    assert page.pills[0].options == [
        "처음 시작하기",
        "화면별 사용법",
        "개인정보 · AI",
        "자주 묻는 질문",
    ]
    assert any("체크인은 이 순서" in item.value for item in page.subheader)

    page.pills[0].set_value("화면별 사용법").run(timeout=30)
    assert not page.exception
    assert {item.label for item in page.expander} == {
        "나의 체크인",
        "마이페이지",
    }

    page.pills[0].set_value("개인정보 · AI").run(timeout=30)
    assert not page.exception
    rendered = "\n".join(str(item.value) for item in page.markdown)
    assert "학생 ID·학번·이름" in rendered
    assert "외부 AI 전송 없는 직접 입력" in rendered


def test_staff_manual_renders_support_flow_and_rule_based_alert_help() -> None:
    """교직원 매뉴얼은 경보부터 후속 확인까지와 판정 근거를 안내한다."""

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "14_staff_manual.py")
    ).run(timeout=30)

    assert not page.exception
    assert page.pills[0].options == [
        "업무 흐름",
        "화면별 사용법",
        "경보 · 추천 기준",
        "데이터 · 시연",
    ]
    assert any("교직원 지원은 이 순서" in item.value for item in page.subheader)

    page.pills[0].set_value("화면별 사용법").run(timeout=30)
    assert not page.exception
    assert len(page.expander) == 10
    assert any("AI 맞춤 추천 통합 검토" in item.label for item in page.expander)
    assert any("초기경보 모델 검증" in item.label for item in page.expander)

    page.pills[0].set_value("경보 · 추천 기준").run(timeout=30)
    assert not page.exception
    rendered = "\n".join(str(item.value) for item in page.markdown)
    assert "정상(0~24)" in rendered
    assert "주차 자체가 위험단계는 아닙니다" in rendered
    assert "전체 WINGS 비교과에서 Top 3" in rendered
