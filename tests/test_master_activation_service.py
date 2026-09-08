"""실제 master 선택 활성화·백업·Repository fallback 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_generator import generate_and_save_data, generate_support_programs
from src.course_offering_import_service import ACTUAL_COURSE_CATALOG_SOURCE
from src.department_catalog import configured_department_names
from src.master_activation_service import (
    MasterActivationService,
    prepare_activation_ready_courses,
    validate_department_activation_data,
)
from src.master_import_service import COURSE_OUTPUT_COLUMNS
from src.master_import_service import (
    MasterDryRunResult,
    MasterImportDryRunReport,
)
from src.repositories import CsvRepository
from src.utils import PROJECT_ROOT


def _departments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "department_code": f"REAL{index:02d}",
                "department_name": name,
            }
            for index, name in enumerate(configured_department_names(), start=1)
        ]
        + [
            {
                "department_code": "REAL99",
                "department_name": "연계학과",
            }
        ]
    )


def _reviewed_courses() -> pd.DataFrame:
    rows = []
    for index in range(1, 4):
        rows.append(
            {
                "course_id": f"REAL-C{index:03d}",
                "course_name": f"검토 교과목 {index}",
                "department": (
                    configured_department_names()[0]
                    if index < 3
                    else "연계학과"
                ),
                "description": "검토된 실제 교과목 개요",
                "learning_objectives": "기초 개념을 실제 과제로 적용",
                "competencies": "문제해결|협업",
                "related_jobs": "서비스 기획",
                "related_interests": "서비스",
                "grade_level": 1,
                "semester": 1 if index % 2 else 2,
                "difficulty": "기초",
                "prerequisites": "",
                "credit": 3,
                "is_available": True,
            }
        )
    return pd.DataFrame(rows, columns=COURSE_OUTPUT_COLUMNS)


def _programs() -> pd.DataFrame:
    programs = generate_support_programs().head(3).copy()
    programs["program_id"] = ["WINGS-A", "WINGS-B", "WINGS-C"]
    programs["application_url"] = "https://wings.kbu.ac.kr"
    programs["source_system"] = "WINGS"
    programs["source_status"] = "신청/운영중"
    programs["academic_year"] = "2026"
    programs["semester"] = "2학기"
    return programs


def _source_backed_courses() -> pd.DataFrame:
    """실제 개설 필드만 있고 인위적인 의미 태그는 없는 교과 후보를 만든다."""

    rows = []
    for index, department in enumerate(configured_department_names(), start=1):
        rows.append(
            {
                "course_id": f"ACTUAL{index:03d}",
                "course_name": f"실제 개설교과 {index}",
                "department": department,
                "description": f"개설학과 {department} · 영역 전공 · 수업방식 대면",
                "learning_objectives": "",
                "competencies": "",
                "related_jobs": "",
                "related_interests": "",
                "grade_level": 1,
                "semester": 1,
                "difficulty": "원본 미제공",
                "prerequisites": "",
                "credit": 3,
                "is_available": True,
                "catalog_source": ACTUAL_COURSE_CATALOG_SOURCE,
                "source_course_id": f"ACTUAL{index:03d}",
                "academic_years": "2026",
                "offered_semesters": "1",
                "offering_departments": department,
                "offering_department_codes": f"D{index:02d}",
                "grade_levels": "1",
                "course_area": "전공",
                "class_method": "대면",
                "course_type": "일반교과",
                "ncs_type": "NCS",
                "cross_department_status": "allowed",
                "online_status": "offline",
                "recommendation_basis": "교과목명·개설학과·NCS",
                "prerequisite_status": "원본 미제공",
                "completion_data_status": "synthetic 학생 이수정보와 미연결",
                "registration_check_required": True,
                "master_match_method": "code",
                "linked_master_course_id": f"ACTUAL{index:03d}",
            }
        )
    rows.append({**rows[-1], "course_id": "ACTUAL999", "source_course_id": "ACTUAL999", "department": "연계학과", "offering_departments": "연계학과"})
    return pd.DataFrame(rows)


def test_activation_requires_departments_before_reviewed_courses(
    tmp_path: Path,
) -> None:
    service = MasterActivationService(data_dir=tmp_path)
    courses = _reviewed_courses()

    with pytest.raises(ValueError, match="학과 master를 먼저"):
        service.activate_reviewed_courses(courses, courses["course_id"].tolist())

    department_result = service.activate_departments(_departments())
    course_result = service.activate_reviewed_courses(
        courses, courses["course_id"].tolist()
    )

    assert department_result.destination_path.name == "departments_master.csv"
    assert course_result.destination_path.name == "courses_reviewed.csv"
    assert course_result.row_count == 3
    assert set(
        pd.read_csv(course_result.destination_path)["course_id"].astype(str)
    ) == set(courses["course_id"])


def test_activation_backs_up_existing_master_before_atomic_replacement(
    tmp_path: Path,
) -> None:
    service = MasterActivationService(data_dir=tmp_path)
    first = service.activate_departments(_departments())
    updated = _departments().copy()
    updated.loc[updated["department_name"] == "연계학과", "department_name"] = (
        "새 연계학과"
    )

    second = service.activate_departments(updated)

    assert first.backup_path is None
    assert second.backup_path is not None
    assert second.backup_path.exists()
    backed_up = pd.read_csv(second.backup_path, keep_default_na=False)
    assert "연계학과" in set(backed_up["department_name"])
    active = pd.read_csv(second.destination_path, keep_default_na=False)
    assert "새 연계학과" in set(active["department_name"])


def test_repository_prefers_valid_actual_masters_and_filters_old_completions(
    tmp_path: Path,
) -> None:
    generate_and_save_data(output_dir=tmp_path, student_count=20)
    service = MasterActivationService(data_dir=tmp_path)
    service.activate_departments(_departments())
    courses = _reviewed_courses()
    service.activate_reviewed_courses(courses, courses["course_id"].tolist())
    service.activate_programs(_programs())

    repository = CsvRepository(data_dir=tmp_path, auto_generate=False)
    loaded_courses = repository.get_courses()
    loaded_programs = repository.get_support_programs()
    completed = repository.get_completed_courses()

    assert loaded_courses["course_id"].tolist() == courses["course_id"].tolist()
    assert loaded_programs["program_id"].tolist() == [
        "WINGS-A",
        "WINGS-B",
        "WINGS-C",
    ]
    assert completed.empty
    assert repository.get_all().courses["course_id"].tolist() == courses[
        "course_id"
    ].tolist()


def test_repository_falls_back_when_active_master_is_corrupted(
    tmp_path: Path,
) -> None:
    generate_and_save_data(output_dir=tmp_path, student_count=20)
    MasterActivationService(data_dir=tmp_path).activate_departments(
        _departments()
    )
    pd.DataFrame({"bad": ["course"]}).to_csv(
        tmp_path / "courses_reviewed.csv", index=False
    )
    pd.DataFrame({"bad": ["program"]}).to_csv(
        tmp_path / "support_programs_wings.csv", index=False
    )

    repository = CsvRepository(data_dir=tmp_path, auto_generate=False)

    assert len(repository.get_courses()) == 40
    assert len(repository.get_support_programs()) == 12


def test_incomplete_or_sensitive_courses_never_become_activation_ready() -> None:
    courses = _reviewed_courses()
    courses.loc[0, "competencies"] = ""
    courses.loc[1, "description"] = "문의 test@example.com"
    courses.loc[2, "is_available"] = False

    ready = prepare_activation_ready_courses(courses)

    assert ready.empty


def test_department_activation_rejects_ambiguous_codes() -> None:
    departments = _departments()
    departments.loc[0, "department_code"] = "???"

    with pytest.raises(ValueError, match="미확정 학과 코드"):
        validate_department_activation_data(departments)


def test_source_backed_course_catalog_activates_without_invented_tags(
    tmp_path: Path,
) -> None:
    generate_and_save_data(output_dir=tmp_path, student_count=20)
    service = MasterActivationService(data_dir=tmp_path)
    service.activate_departments(_departments())
    courses = _source_backed_courses()

    result = service.activate_course_offering_catalog(courses)
    loaded = CsvRepository(data_dir=tmp_path, auto_generate=False).get_courses()

    assert result.row_count == len(courses)
    assert loaded["course_id"].is_unique
    assert set(loaded["course_id"]) == set(courses["course_id"])
    assert loaded["competencies"].eq("").all()
    assert loaded["related_jobs"].eq("").all()
    assert loaded["offering_departments"].str.contains("연계학과").any()


def test_demo_preparation_renders_reviewed_master_activation_controls() -> None:
    report = MasterImportDryRunReport(
        departments=MasterDryRunResult(
            "departments",
            "학과",
            "departments.xlsx",
            len(_departments()),
            _departments(),
            (),
            0,
            0,
            0,
            0,
            (),
        ),
        courses=MasterDryRunResult(
            "courses",
            "교과목",
            "courses.xlsx",
            len(_reviewed_courses()),
            _reviewed_courses(),
            (),
            0,
            0,
            0,
            0,
            (),
        ),
        programs=MasterDryRunResult(
            "programs",
            "비교과",
            "programs.xlsx",
            len(_programs()),
            _programs(),
            (),
            0,
            0,
            0,
            0,
            (),
        ),
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "09_demo_scenario.py")
    )
    page.session_state["master_import_dry_run_report"] = report
    page.session_state["demo_show_advanced_tools"] = True
    page.run(timeout=30)

    assert not page.exception
    labels = {button.label for button in page.button}
    assert {
        "학과 안전 후보 활성화",
        "비교과 안전 후보 활성화",
        "선택한 교과목 활성화",
    }.issubset(labels)
    assert any(
        "검토 완료 데이터만 활성화" in checkbox.label
        for checkbox in page.checkbox
    )
    assert any(
        expander.label == "검토 완료 master 활성화"
        for expander in page.expander
    )
