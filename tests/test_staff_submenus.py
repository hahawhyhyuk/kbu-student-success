"""교직원 서비스별 2차 메뉴와 단일 본문 렌더링 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.utils import PROJECT_ROOT


STAFF_SUBMENUS = (
    (
        "01_dashboard.py",
        [
            "위험등급 분포",
            "위험유형 분포",
            "학과별 위험학생 수",
            "주차별 평균 위험도 변화",
            "우선 확인 학생",
            "아직 개입되지 않은 고위험 학생",
        ],
    ),
    (
        "02_student_detail.py",
        [
            "학생 기본정보",
            "최신 초기경보",
            "주차별 위험도 변화",
            "영역별 최신 위험도",
            "위험 원인 설명",
            "최근 체크인",
            "학생 제출 · 피드백",
            "과거 지원 이력",
        ],
    ),
    (
        "03_ai_intervention.py",
        ["학생 · AI 이해", "비교과 추천", "교과 학습경로", "최종 검토", "학생 피드백"],
    ),
    (
        "05_intervention_management.py",
        ["지원 현황", "지원 계획 확정", "프로그램 실행 기록", "추천 근거 · 피드백"],
    ),
    (
        "08_intervention_effect.py",
        ["변화 요약", "변화 차트", "지원별 비교 결과", "선택 지원 상세"],
    ),
)


@pytest.mark.parametrize(("page_name", "expected_options"), STAFF_SUBMENUS)
def test_each_staff_page_has_its_own_submenu(
    page_name: str,
    expected_options: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """교직원 핵심 화면은 페이지 목적에 맞는 2차 메뉴를 하나씩 제공한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / f"{page_name}.db"
    )
    page = AppTest.from_file(str(PROJECT_ROOT / "pages" / page_name)).run(
        timeout=30
    )

    assert not page.exception
    submenu = next(
        pills for pills in page.pills if pills.options == expected_options
    )
    assert submenu.value == expected_options[0]
    for option in expected_options[1:]:
        submenu.set_value(option).run(timeout=30)
        assert not page.exception
        submenu = next(
            pills for pills in page.pills if pills.options == expected_options
        )
        assert submenu.value == option


def test_dashboard_submenu_renders_only_selected_section() -> None:
    """대시보드는 선택한 분포 하나만 본문에 표시한다."""

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "01_dashboard.py")
    ).run(timeout=30)

    assert [item.value for item in page.subheader] == ["위험등급 분포"]
    submenu = next(
        pills
        for pills in page.pills
        if "위험유형 분포" in pills.options
    )
    submenu.set_value("위험유형 분포").run(timeout=30)
    assert not page.exception
    assert [item.value for item in page.subheader] == ["위험유형 분포"]


def test_student_detail_does_not_duplicate_recommendation_feedback() -> None:
    """학생 종합 확인은 AI 이해 결과에 대한 학생 의견만 표시한다."""

    source = (PROJECT_ROOT / "pages" / "02_student_detail.py").read_text(
        encoding="utf-8"
    )

    assert "AI 이해 결과에 대한 학생 의견" in source
    assert "지원·학습경로 추천에 대한 학생 의견" not in source


def test_dashboard_uses_top_query_filters_instead_of_sidebar() -> None:
    """전체 대시보드는 주차·학과·위험등급을 본문 상단에서 조정한다."""

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "01_dashboard.py")
    ).run(timeout=30)

    assert not page.exception
    assert any(
        selectbox.label == "조회 기준 주차" for selectbox in page.selectbox
    )
    assert any(selectbox.label == "학과" for selectbox in page.selectbox)
    assert any(multiselect.label == "위험등급" for multiselect in page.multiselect)
    assert not any(
        selectbox.label == "학과" for selectbox in page.sidebar.selectbox
    )
    assert not any(
        multiselect.label == "위험등급"
        for multiselect in page.sidebar.multiselect
    )

    department_filter = next(
        selectbox for selectbox in page.selectbox if selectbox.label == "학과"
    )
    department_filter.set_value("간호학과").run(timeout=30)

    assert not page.exception
    total_students = next(
        metric for metric in page.metric if metric.label == "전체학생"
    )
    assert int(total_students.value) > 0
    assert int(total_students.value) < 200


def test_integrated_recommendation_review_page_uses_updated_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """교직원 본문에 비교과·교과 통합 검토 명칭을 사용한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "program_review.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "03_ai_intervention.py")
    ).run(timeout=30)

    assert not page.exception
    rendered_markdown = "\n".join(str(item.value) for item in page.markdown)
    assert "AI 맞춤 추천 통합 검토" in rendered_markdown
    assert "AI 추천 프로그램 검토" not in rendered_markdown


def test_student_detail_uses_searchable_top_student_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """학생 종합 확인은 사이드바 대신 제목 아래에서 학생을 변경한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "student_detail.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "02_student_detail.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.run(timeout=30)

    assert not page.exception
    selector = next(
        selectbox
        for selectbox in page.selectbox
        if selectbox.label == "확인할 학생"
    )
    assert selector.value == "S0003"
    assert any(
        option.startswith("S0003 ·") and "학년" in option
        for option in selector.options
    )
    assert not any(
        selectbox.label in {"학생 선택", "확인할 학생"}
        for selectbox in page.sidebar.selectbox
    )
    assert any(metric.label == "현재 위험등급" for metric in page.metric)

    selector.set_value("S0002").run(timeout=30)

    assert not page.exception
    assert page.session_state["demo_student_id"] == "S0002"
    assert any(
        metric.label == "학생 ID" and metric.value == "S0002"
        for metric in page.metric
    )


def test_integrated_review_uses_searchable_top_student_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """비교과·교과 통합 검토는 사이드바 대신 제목 아래에서 학생을 변경한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "integrated_review.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "03_ai_intervention.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.run(timeout=30)

    assert not page.exception
    selector = next(
        selectbox
        for selectbox in page.selectbox
        if selectbox.label == "확인할 학생"
    )
    assert selector.value == "S0003"
    assert any(
        option.startswith("S0003 ·") and "학년" in option
        for option in selector.options
    )
    assert not any(
        selectbox.label in {"학생 선택", "확인할 학생"}
        for selectbox in page.sidebar.selectbox
    )
    assert any(metric.label == "현재 위험등급" for metric in page.metric)

    selector.set_value("S0002").run(timeout=30)

    assert not page.exception
    assert page.session_state["demo_student_id"] == "S0002"
    assert any(
        metric.label == "학생 ID" and metric.value == "S0002"
        for metric in page.metric
    )
