"""5개 적용 학과 설정과 synthetic 데이터 반영 테스트."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.data_generator import generate_courses, generate_students
from src.department_catalog import (
    REQUIRED_DEPARTMENT_COUNT,
    configured_department_names,
    load_department_catalog,
    validate_configured_departments,
)
from src.utils import PROJECT_ROOT


EXPECTED_DEPARTMENTS = (
    "친환경건축과",
    "유아교육과",
    "실용음악과(3년제)",
    "간호학과",
    "소프트웨어융합과",
)


def test_app_config_defines_exactly_five_target_departments() -> None:
    """회의에서 선택한 5개 학과를 설정의 단일 기준으로 유지한다."""

    catalog = load_department_catalog()

    assert len(catalog) == REQUIRED_DEPARTMENT_COUNT
    assert tuple(item.code for item in catalog) == (
        "DEPT01",
        "DEPT02",
        "DEPT03",
        "DEPT04",
        "DEPT05",
    )
    assert configured_department_names() == EXPECTED_DEPARTMENTS


def test_department_catalog_rejects_wrong_count_and_duplicates() -> None:
    """누락·추가·중복 설정이 조용히 화면에 반영되지 않게 한다."""

    with pytest.raises(ValueError, match="정확히 5개"):
        load_department_catalog(
            {"departments": [{"code": "D1", "name": "학과1"}]}
        )
    duplicated = {
        "departments": [
            {"code": f"D{index}", "name": "중복학과"}
            for index in range(1, 6)
        ]
    }
    with pytest.raises(ValueError, match="name은 중복"):
        load_department_catalog(duplicated)


def test_synthetic_students_and_courses_use_only_configured_departments() -> None:
    """대시보드와 교과 master에 DEPT06이나 임의 학과가 남지 않는다."""

    students = generate_students(student_count=200)
    courses = generate_courses()

    assert set(students["department"]) == set(EXPECTED_DEPARTMENTS)
    assert set(courses["department"]) == set(EXPECTED_DEPARTMENTS)
    assert len(courses) == 40
    assert courses.groupby("department").size().eq(8).all()


def test_unknown_department_value_is_rejected() -> None:
    """설정 변경 뒤 오래된 DEPT 코드 CSV를 명확한 오류로 차단한다."""

    with pytest.raises(ValueError, match="설정되지 않은 학과"):
        validate_configured_departments(
            ["친환경건축과", "DEPT06"],
            source_name="students.csv",
        )


def test_dashboard_department_filter_uses_configured_names() -> None:
    """교직원 화면에는 내부 DEPT 코드 대신 5개 학과명이 표시된다."""

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "01_dashboard.py")
    ).run(timeout=30)

    assert not page.exception
    department_filter = next(
        selectbox for selectbox in page.selectbox if selectbox.label == "학과"
    )
    assert department_filter.options == ["전체", *sorted(EXPECTED_DEPARTMENTS)]
