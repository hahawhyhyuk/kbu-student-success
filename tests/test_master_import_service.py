"""실제 master dry-run의 개인정보 차단·중복·연결 검증 테스트."""

from __future__ import annotations

import pandas as pd

from src.department_catalog import configured_department_names
from src.master_import_service import (
    COURSE_OUTPUT_COLUMNS,
    DEPARTMENT_OUTPUT_COLUMNS,
    load_course_excel_safe,
    validate_course_frame,
    validate_department_frame,
    validate_program_frame,
)
from src.repositories.base import COURSE_COLUMNS
from src.support_program_importer import SAFE_SOURCE_COLUMNS


def test_department_dry_run_flags_missing_and_duplicate_rows() -> None:
    rows = [
        {
            "department_code": f"D{index:02d}",
            "department_name": name,
        }
        for index, name in enumerate(configured_department_names(), start=1)
    ]
    rows.extend(
        [
            {"department_code": "D01", "department_name": "중복 학과"},
            {"department_code": "", "department_name": "코드 누락 학과"},
        ]
    )

    result = validate_department_frame(
        pd.DataFrame(rows),
        source_name="departments.xlsx",
        blocked_columns=("담당자명",),
    )

    assert not result.ready
    assert result.accepted_row_count == 4
    assert result.duplicate_row_count == 2
    assert result.missing_required_row_count == 1
    assert tuple(result.candidate_data.columns) == DEPARTMENT_OUTPUT_COLUMNS
    assert "중복 학과" not in set(result.candidate_data["department_name"])
    assert configured_department_names()[0] not in set(
        result.candidate_data["department_name"]
    )
    assert {issue.code for issue in result.issues}.issuperset(
        {
            "pii_columns_blocked",
            "duplicate_departments",
            "missing_required_values",
            "configured_departments_missing",
        }
    )


def test_course_dry_run_excludes_invalid_links_and_redacts_contacts() -> None:
    target_department = configured_department_names()[0]
    source = pd.DataFrame(
        [
            {
                "course_id": "C001",
                "course_name": "기초 설계",
                "department": target_department,
                "description": "문의 010-1234-5678 test@example.com",
                "credit": 3,
                "_source_status": "개설",
            },
            {
                "course_id": "C002",
                "course_name": "타과 융합",
                "department": "연계학과",
                "description": "융합 프로젝트",
                "credit": 3,
                "_source_status": "개설중",
            },
            {
                "course_id": "C003",
                "course_name": "연결 오류",
                "department": "없는학과",
                "_source_status": "개설",
            },
            {
                "course_id": "C004",
                "course_name": "중복 A",
                "department": target_department,
                "_source_status": "개설",
            },
            {
                "course_id": "C004",
                "course_name": "중복 B",
                "department": target_department,
                "_source_status": "개설",
            },
            {
                "course_id": "",
                "course_name": "코드 없음",
                "department": target_department,
                "_source_status": "개설",
            },
        ]
    )

    result = validate_course_frame(
        source,
        [*configured_department_names(), "연계학과"],
        source_name="courses.xlsx",
    )

    assert not result.ready
    assert result.accepted_row_count == 2
    assert result.unknown_department_row_count == 1
    assert result.duplicate_row_count == 2
    assert result.missing_required_row_count == 1
    assert set(result.candidate_data.columns) == set(COURSE_COLUMNS)
    serialized = " ".join(
        result.candidate_data.fillna("").astype(str).to_numpy().ravel()
    )
    assert "010-1234-5678" not in serialized
    assert "test@example.com" not in serialized
    assert set(result.candidate_data["department"]) == {
        target_department,
        "연계학과",
    }
    assert {issue.code for issue in result.issues}.issuperset(
        {
            "contacts_redacted",
            "missing_required_values",
            "duplicate_courses",
            "unknown_course_departments",
            "course_semantics_missing",
        }
    )


def test_course_excel_loader_never_requests_pii_columns(monkeypatch) -> None:
    headers = [
        "교과목코드",
        "교과목명",
        "학과",
        "학점",
        "교과목개요",
        "담당자명",
        "전화번호",
        "이메일",
    ]
    captured_usecols: list[str] = []

    def fake_read_excel(_source, **kwargs):
        if kwargs.get("nrows") == 0:
            return pd.DataFrame(columns=headers)
        captured_usecols.extend(kwargs["usecols"])
        return pd.DataFrame(
            {
                column: [
                    {
                        "교과목코드": "C001",
                        "교과목명": "데이터 기초",
                        "학과": configured_department_names()[0],
                        "학점": 3,
                        "교과목개요": "기초 과목",
                    }[column]
                ]
                for column in kwargs["usecols"]
            }
        )

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    loaded = load_course_excel_safe(b"not-read-by-mock")

    assert set(captured_usecols) == {
        "교과목코드",
        "교과목명",
        "학과",
        "학점",
        "교과목개요",
    }
    assert set(loaded.blocked_columns) == {"담당자명", "전화번호", "이메일"}
    assert not {"담당자명", "전화번호", "이메일"}.intersection(
        loaded.frame.columns
    )


def test_wings_dry_run_keeps_safe_program_regardless_of_status() -> None:
    row = {
        "program_group": "학생성공",
        "program_name": "학습전략 특강",
        "program_overview": "문의 010-1234-5678 test@example.com",
        "large_category": "학습지원",
        "middle_category": "특강",
        "program_format": "비교과 (비학점)",
        "status": "신청/운영중",
        "academic_year": 2026,
        "semester": "2학기",
        "operation_start_date": "2026-09-10",
        "operation_end_date": "2026-10-10",
        "application_start_date": "2026-09-01",
        "application_end_date": "2026-09-09",
        "operating_institution": "학생성공처",
        "operating_department": "학생성공센터",
        "primary_competency": "실무 전문성 역량",
        "secondary_competency": "창의융합역량",
    }
    result = validate_program_frame(
        pd.DataFrame([row], columns=SAFE_SOURCE_COLUMNS),
        source_name="programs.xlsx",
        blocked_columns=("운영자", "사번", "전화번호", "이메일"),
    )

    assert result.ready
    assert result.accepted_row_count == 1
    assert "pii_columns_blocked" in {issue.code for issue in result.issues}
    serialized = " ".join(
        result.candidate_data.fillna("").astype(str).to_numpy().ravel()
    )
    assert "010-1234-5678" not in serialized
    assert "test@example.com" not in serialized
    assert "email" not in result.candidate_data.columns


def test_wings_dry_run_does_not_require_or_filter_row_status() -> None:
    base = {
        "program_group": "학생성공",
        "program_name": "진로 탐색",
        "program_overview": "진로 탐색 프로그램",
        "large_category": "취·창업지원",
        "middle_category": "진로",
        "program_format": "비교과 (비학점)",
        "status": "마감완료",
        "academic_year": 2026,
        "semester": "2학기",
        "operation_start_date": "2026-09-10",
        "operation_end_date": "2026-10-10",
        "application_start_date": "2026-09-01",
        "application_end_date": "2026-09-09",
        "operating_institution": "학생성공처",
        "operating_department": "학생성공센터",
        "primary_competency": "실무 전문성 역량",
        "secondary_competency": "",
    }
    status_unknown = dict(base, program_name="상태 미정 프로그램", status="")

    result = validate_program_frame(
        pd.DataFrame([base, status_unknown], columns=SAFE_SOURCE_COLUMNS),
        source_name="programs.xlsx",
    )

    assert result.ready
    assert result.accepted_row_count == 2
    assert result.missing_required_row_count == 0
    assert set(result.candidate_data["source_status"]) == {"마감완료", ""}


def test_wings_dry_run_reports_duplicate_and_missing_required_rows() -> None:
    row = {
        "program_group": "학생성공",
        "program_name": "진로 탐색",
        "program_overview": "진로 탐색 프로그램",
        "large_category": "취·창업지원",
        "middle_category": "진로",
        "program_format": "비교과 (비학점)",
        "status": "신청/운영중",
        "academic_year": 2026,
        "semester": "2학기",
        "operation_start_date": "2026-09-10",
        "operation_end_date": "2026-10-10",
        "application_start_date": "2026-09-01",
        "application_end_date": "2026-09-09",
        "operating_institution": "학생성공처",
        "operating_department": "학생성공센터",
        "primary_competency": "실무 전문성 역량",
        "secondary_competency": "",
    }
    missing_name = dict(row, program_name="")

    result = validate_program_frame(
        pd.DataFrame([row, row, missing_name], columns=SAFE_SOURCE_COLUMNS),
        source_name="programs.xlsx",
    )

    assert not result.ready
    assert result.accepted_row_count == 1
    assert result.duplicate_row_count == 2
    assert result.missing_required_row_count == 1
    assert {issue.code for issue in result.issues}.issuperset(
        {"duplicate_programs", "missing_required_values"}
    )


def test_course_output_column_order_matches_repository_contract() -> None:
    assert set(COURSE_OUTPUT_COLUMNS) == set(COURSE_COLUMNS)
