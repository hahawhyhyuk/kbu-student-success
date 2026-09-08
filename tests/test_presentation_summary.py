"""발표 요약 구현 근거 집계와 Streamlit 화면 테스트."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from src.presentation_service import build_presentation_evidence
from src.repositories import StudentSuccessData, get_default_repository
from src.ui import TAGLINE
from src.utils import PROJECT_ROOT


def test_presentation_evidence_matches_active_master() -> None:
    data = get_default_repository().get_all()

    evidence = build_presentation_evidence(data)

    assert evidence.student_count == 200
    assert evidence.week_count == 4
    assert evidence.department_count == 5
    assert evidence.risk_domain_count == 5
    assert evidence.support_program_count == data.support_programs["program_id"].nunique()
    assert evidence.course_count == data.courses["course_id"].nunique()
    assert evidence.sensitive_fields_used == ()


def test_presentation_evidence_reports_sensitive_columns() -> None:
    source = get_default_repository().get_all()
    students = source.students.copy()
    students["gender"] = "not-used-test-field"
    data = StudentSuccessData(
        students=students,
        weekly_activity=source.weekly_activity,
        checkins=source.checkins,
        support_programs=source.support_programs,
        courses=source.courses,
        completed_courses=source.completed_courses,
    )

    evidence = build_presentation_evidence(data)

    assert evidence.sensitive_fields_used == ("gender",)


def test_presentation_summary_page_renders_official_judging_criteria() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "10_presentation_summary.py")
    ).run(timeout=30)

    assert not app.exception
    assert any(metric.label == "가상학생" and metric.value == "200명" for metric in app.metric)
    assert any("30초 소개" in item.value for item in app.markdown)
    assert any(TAGLINE in item.value for item in app.markdown)
    rendered_markdown = " ".join(item.value for item in app.markdown)
    assert "지원계획·실행" in rendered_markdown
    assert "4주차 지원 시작과 synthetic 15주차 관찰 비교" in rendered_markdown
    assert "STEP 07" in rendered_markdown
    assert any("공식 심사기준 대응표" in item.value for item in app.subheader)
    assert any("공통 심사기준 · 80점" in item.value for item in app.markdown)
    assert any("주제 01 심사기준 · 20점" in item.value for item in app.markdown)
    captions = " ".join(item.value for item in app.caption)
    assert "총 100점" in captions
    assert "자체 예상점수가 아니라" in captions
