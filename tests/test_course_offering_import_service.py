"""전 학과 실제 개설강좌 안전 로딩·분반 통합·master 연결 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.course_offering_import_service import (
    ACTUAL_COURSE_CATALOG_SOURCE,
    COURSE_OFFERING_OUTPUT_COLUMNS,
    SafeOfferingFrame,
    build_course_offering_dry_run,
    build_course_recommendation_catalog,
    find_default_course_offering_files,
    load_course_offering_excel_safe,
)


def _safe_source(
    rows: list[dict[str, object]],
    *,
    semester: int,
) -> SafeOfferingFrame:
    """테스트 행을 이미 allowlist로 읽은 학기 강좌로 감싼다."""

    return SafeOfferingFrame(
        frame=pd.DataFrame(rows),
        source_name=f"2026-{semester}.xlsx",
        academic_year=2026,
        semester=semester,
        source_row_count=len(rows),
        blocked_columns=(),
        ignored_column_count=0,
    )


def test_loader_reads_only_allowlisted_columns_and_blocks_pii(monkeypatch) -> None:
    headers = [
        "개설학과코드",
        "개설학과",
        "교과목 코드",
        "교과목명",
        "개설 학년",
        "교과목 학점",
        "개설강좌 상태구분",
        "수강 분반",
        "담당자명",
        "전화번호",
    ]
    source_row = {
        "개설학과코드": "D001",
        "개설학과": "간호학과[D001]",
        "교과목 코드": "C001",
        "교과목명": "간호정보학",
        "개설 학년": 2,
        "교과목 학점": 3,
        "개설강좌 상태구분": "개설",
        "수강 분반": "01",
        "담당자명": "읽으면 안 됨",
        "전화번호": "010-0000-0000",
    }
    requested_indexes: list[int] = []

    def fake_read_excel(_source, **kwargs):
        if kwargs.get("nrows") == 0:
            return pd.DataFrame(columns=headers)
        requested_indexes.extend(kwargs["usecols"])
        selected_headers = [headers[index] for index in kwargs["usecols"]]
        return pd.DataFrame(
            [{header: source_row[header] for header in selected_headers}]
        )

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = load_course_offering_excel_safe(
        b"safe-test",
        academic_year=2026,
        semester=1,
        source_name="semester-one.xlsx",
    )

    assert set(result.blocked_columns) == {"담당자명", "전화번호"}
    assert headers.index("담당자명") not in requested_indexes
    assert headers.index("전화번호") not in requested_indexes
    assert {"course_id", "course_name", "department_name"}.issubset(
        result.frame.columns
    )
    serialized = " ".join(result.frame.astype(str).to_numpy().ravel())
    assert "읽으면 안 됨" not in serialized
    assert "010-0000-0000" not in serialized


def test_dry_run_preserves_all_departments_and_consolidates_sections() -> None:
    semester_one = _safe_source(
        [
            {
                "department_code": "30100100",
                "department_name": "간호학과[30100100]",
                "course_id": "C001",
                "course_name": "간호정보학",
                "grade_level": 2,
                "section": "01",
                "credit": 3,
                "offering_status": "개설",
                "cancellation_status": "",
                "cross_department_excluded": 0,
                "online": 0,
                "enrolled_count": 20,
                "capacity": 30,
                "lecture_hours": 2,
                "practice_hours": 1,
                "total_hours": 3,
            },
            {
                "department_code": "30100100",
                "department_name": "간호학과[30100100]",
                "course_id": "C001",
                "course_name": "간호정보학",
                "grade_level": 2,
                "section": "02",
                "credit": 3,
                "offering_status": "개설",
                "cancellation_status": "",
                "cross_department_excluded": 0,
                "online": 0,
                "enrolled_count": 21,
                "capacity": 30,
                "lecture_hours": 2,
                "practice_hours": 1,
                "total_hours": 3,
            },
            {
                "department_code": "30909999",
                "department_name": "연계학과[30909999]",
                "course_id": "NEW01",
                "course_name": "서비스 기획",
                "grade_level": 1,
                "section": "01",
                "credit": 2,
                "offering_status": "개설",
                "cross_department_excluded": 0,
            },
        ],
        semester=1,
    )
    semester_two = _safe_source(
        [
            {
                "department_code": "30908888",
                "department_name": "다른학과[30908888]",
                "course_id": "NEW02",
                "course_name": "현장 프로젝트",
                "grade_level": 1,
                "section": "01",
                "credit": 2,
                "offering_status": "개설",
                "cross_department_excluded": 1,
            }
        ],
        semester=2,
    )
    course_master = pd.DataFrame(
        [
            {"course_id": "C001", "course_name": "간호정보학"},
            {"course_id": "MASTER02", "course_name": "서비스기획"},
        ]
    )

    result = build_course_offering_dry_run(
        course_master,
        (semester_one, semester_two),
    )

    assert result.ready
    assert result.valid_section_count == 4
    assert result.course_count == 3
    assert result.consolidated_section_count == 1
    assert result.master_code_match_count == 1
    assert result.master_name_match_count == 1
    assert result.master_unmatched_count == 1
    assert result.target_department_course_count == 1
    assert tuple(result.candidate_data.columns) == COURSE_OFFERING_OUTPUT_COLUMNS
    nursing = result.candidate_data[result.candidate_data["course_id"] == "C001"].iloc[0]
    assert nursing["department_name"] == "간호학과"
    assert nursing["section_count"] == 2
    assert nursing["sections"] == "01|02"
    assert nursing["enrolled_count"] == 41
    assert nursing["capacity"] == 60
    assert nursing["availability_status"] == "available"
    assert nursing["cross_department_status"] == "allowed"
    assert set(result.candidate_data["department_name"]) == {
        "간호학과",
        "연계학과",
        "다른학과",
    }
    assert "master_courses_unmatched" in {issue.code for issue in result.issues}
    assert result.summary_frame()["통합 강좌"].tolist() == [2, 1]


def test_dry_run_keeps_unknown_availability_and_restricted_course() -> None:
    source = _safe_source(
        [
            {
                "department_code": "D01",
                "department_name": "유아교육학과[D01]",
                "course_id": "C001",
                "course_name": "놀이 지도",
                "grade_level": 1,
                "credit": 3,
                "offering_status": "",
                "cross_department_excluded": 1,
            },
            {
                "department_code": "D02",
                "department_name": "소프트웨어융합과(2022)[D02]",
                "course_id": "C002",
                "course_name": "프로그래밍",
                "grade_level": 1,
                "credit": 3,
                "offering_status": "개설",
                "cancellation_status": "폐강",
                "cross_department_excluded": 0,
            },
        ],
        semester=1,
    )
    result = build_course_offering_dry_run(
        pd.DataFrame(
            [
                {"course_id": "C001", "course_name": "놀이 지도"},
                {"course_id": "C002", "course_name": "프로그래밍"},
            ]
        ),
        (source,),
    )

    by_id = result.candidate_data.set_index("course_id")
    assert by_id.loc["C001", "department_name"] == "유아교육과"
    assert by_id.loc["C001", "availability_status"] == "unknown"
    assert by_id.loc["C001", "cross_department_status"] == "restricted"
    assert by_id.loc["C002", "department_name"] == "소프트웨어융합과"
    assert by_id.loc["C002", "availability_status"] == "unavailable"
    assert "availability_unknown" in {issue.code for issue in result.issues}


def test_default_file_discovery_is_unicode_safe_and_ignores_lock_files(
    tmp_path: Path,
) -> None:
    expected = {
        "course": tmp_path / "2. 전체 교과목.xlsx",
        "one": tmp_path / "26년1학기 개설강좌.xlsx",
        "two": tmp_path / "26년2학기 개설강좌.xlsx",
    }
    for path in expected.values():
        path.write_bytes(b"")
    (tmp_path / "~$2. 전체 교과목.xlsx").write_bytes(b"")

    result = find_default_course_offering_files(tmp_path)

    assert result.course_master == expected["course"]
    assert result.semester_one == expected["one"]
    assert result.semester_two == expected["two"]


def test_recommendation_catalog_keeps_official_id_and_existing_fields_only() -> None:
    semester_one = _safe_source(
        [
            {
                "department_code": "D01",
                "department_name": "간호학과[D01]",
                "course_id": "REAL01",
                "course_name": "간호정보학",
                "grade_level": 1,
                "section": "01",
                "credit": 3,
                "offering_status": "개설",
                "cross_department_excluded": 0,
                "class_method": "대면",
                "course_area": "전공",
                "course_type": "일반교과",
                "ncs_type": "NCS",
            },
            {
                "department_code": "D02",
                "department_name": "소프트웨어융합과[D02]",
                "course_id": "REAL02",
                "course_name": "데이터분석",
                "grade_level": 1,
                "section": "01",
                "credit": 3,
                "offering_status": "개설",
                "cross_department_excluded": 0,
                "class_method": "혼합",
                "course_area": "전공",
                "course_type": "일반교과",
                "ncs_type": "KBUCS",
            },
        ],
        semester=1,
    )
    semester_two = _safe_source(
        [
            {
                "department_code": "D99",
                "department_name": "연계학과[D99]",
                "course_id": "REAL01",
                "course_name": "간호정보학",
                "grade_level": 2,
                "section": "01",
                "credit": 3,
                "offering_status": "개설",
                "cross_department_excluded": 0,
                "class_method": "원격",
                "course_area": "전공",
                "course_type": "일반교과",
                "ncs_type": "NCS",
            }
        ],
        semester=2,
    )
    report = build_course_offering_dry_run(
        pd.DataFrame(
            [
                {
                    "course_id": "REAL01",
                    "course_name": "간호정보학",
                    "description": "간호 데이터의 안전한 활용을 배운다.",
                },
                {"course_id": "REAL02", "course_name": "데이터분석"},
            ]
        ),
        (semester_one, semester_two),
    )

    catalog = build_course_recommendation_catalog(report)

    assert catalog.course_count == 2
    assert catalog.source_offering_count == 3
    assert catalog.description_linked_count == 1
    assert set(catalog.candidate_data["course_id"]) == {"REAL01", "REAL02"}
    real_one = catalog.candidate_data.set_index("course_id").loc["REAL01"]
    assert real_one["source_course_id"] == "REAL01"
    assert real_one["catalog_source"] == ACTUAL_COURSE_CATALOG_SOURCE
    assert set(real_one["offering_departments"].split("|")) == {
        "간호학과",
        "연계학과",
    }
    assert real_one["offered_semesters"] == "1|2"
    assert real_one["grade_levels"] == "1|2"
    assert real_one["description"] == "간호 데이터의 안전한 활용을 배운다."
    assert real_one["competencies"] == ""
    assert real_one["related_jobs"] == ""
    assert real_one["related_interests"] == ""
