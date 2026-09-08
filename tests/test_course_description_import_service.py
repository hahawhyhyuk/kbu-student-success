"""강좌기본정보의 안전 로딩과 현재 개설강좌 설명 보강 테스트."""

from pathlib import Path

import pandas as pd

from src.course_description_import_service import (
    DESCRIPTION_SOURCE,
    SafeCourseDescriptionFrame,
    build_course_description_enrichment,
    course_catalog_signature,
    find_default_course_description_file,
    load_course_description_excel_safe,
)


def _catalog() -> pd.DataFrame:
    """설명 보강 테스트용 현재 개설강좌 후보를 반환한다."""

    return pd.DataFrame(
        [
            {
                "course_id": "C001",
                "course_name": "데이터 분석",
                "description": "기존 설명",
                "offering_departments": "소프트웨어융합과",
                "recommendation_basis": "과목명·개설학과",
                "offered_semesters": "1|2",
            },
            {
                "course_id": "C002",
                "course_name": "서비스 기획",
                "description": "기존 서비스 설명",
                "offering_departments": "경영학과",
                "recommendation_basis": "과목명·개설학과",
                "offered_semesters": "2",
            },
            {
                "course_id": "C003",
                "course_name": "진로 탐색",
                "description": "충돌 시 지켜야 하는 기존 설명",
                "offering_departments": "교양학부",
                "recommendation_basis": "과목명·개설학과",
                "offered_semesters": "1",
            },
            {
                "course_id": "C004",
                "course_name": "신규 과목",
                "description": "원본 개설 필드로 만든 설명",
                "offering_departments": "간호학과",
                "recommendation_basis": "과목명·개설학과",
                "offered_semesters": "1",
            },
        ]
    )


def _description_source() -> SafeCourseDescriptionFrame:
    """학과 우선·고유 명칭·충돌 사례를 포함한 안전 원본을 반환한다."""

    rows = [
        {
            "academic_year": 2025,
            "semester": 1,
            "course_name": "데이터분석",
            "description": "데이터를 정제하고 시각화하여 문제를 해결하는 방법을 학습합니다.",
            "department_name": "소프트웨어융합과(2022)",
        },
        {
            "academic_year": 2025,
            "semester": 2,
            "course_name": "데이터 분석",
            "description": "데이터를 정제하고 시각화하여 문제를 해결하는 방법을 학습합니다.",
            "department_name": "소프트웨어융합과",
        },
        {
            "academic_year": 2025,
            "semester": 2,
            "course_name": "서비스기획",
            "description": "사용자 문제를 발견하고 서비스 흐름과 요구사항을 설계하는 과목입니다.",
            "department_name": "콘텐츠학과",
        },
        {
            "academic_year": 2025,
            "semester": 1,
            "course_name": "진로탐색",
            "description": "직업 세계를 조사하고 개인의 진로 계획을 구체화하는 과목입니다.",
            "department_name": "교양학부",
        },
        {
            "academic_year": 2025,
            "semester": 2,
            "course_name": "진로 탐색",
            "description": "산업 정보를 조사하고 취업 준비 활동을 실습하는 과목입니다.",
            "department_name": "교양학부",
        },
    ]
    columns = (
        "lecture_id",
        "academic_year",
        "semester",
        "course_name",
        "lecture_name",
        "description",
        "department_code",
        "department_name",
        "grade_level",
        "course_type",
        "online",
        "ncs_code",
        "hours",
        "credit",
    )
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return SafeCourseDescriptionFrame(
        frame=frame.loc[:, list(columns)],
        source_name="강좌기본정보.xlsx",
        source_row_count=len(frame),
        blocked_columns=(),
        ignored_column_count=0,
    )


def test_safe_loader_reads_only_allowlisted_columns(monkeypatch) -> None:
    headers = [
        "강좌구분번호",
        "개설연도",
        "개설학기",
        "과목명",
        "강좌설명",
        "학과명",
        "담당자명",
        "전화번호",
    ]
    source_row = {
        "강좌구분번호": "L001",
        "개설연도": 2025,
        "개설학기": 1,
        "과목명": "데이터분석",
        "강좌설명": "데이터를 정리하고 분석 결과를 설명하는 방법을 학습합니다.",
        "학과명": "소프트웨어융합과",
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

    result = load_course_description_excel_safe(
        b"safe-test",
        source_name="강좌기본정보.xlsx",
    )

    assert set(result.blocked_columns) == {"담당자명", "전화번호"}
    assert headers.index("담당자명") not in requested_indexes
    assert headers.index("전화번호") not in requested_indexes
    serialized = " ".join(result.frame.astype(str).to_numpy().ravel())
    assert "읽으면 안 됨" not in serialized
    assert "010-0000-0000" not in serialized


def test_enrichment_preserves_current_catalog_scope_and_uses_safe_matches() -> None:
    catalog = _catalog()

    result = build_course_description_enrichment(
        catalog,
        _description_source(),
    )

    assert result.ready
    assert result.course_count == len(catalog)
    assert result.enriched_course_count == 2
    assert result.department_match_count == 1
    assert result.unique_name_match_count == 1
    assert result.ambiguous_course_count == 1
    assert result.unmatched_course_count == 1
    assert result.applies_to(catalog)
    assert result.source_catalog_signature == course_catalog_signature(catalog)
    assert result.candidate_data["course_id"].tolist() == catalog[
        "course_id"
    ].tolist()
    assert result.candidate_data["offered_semesters"].tolist() == catalog[
        "offered_semesters"
    ].tolist()

    by_id = result.candidate_data.set_index("course_id")
    assert by_id.loc["C001", "description_match_method"] == (
        "course_name_department"
    )
    assert by_id.loc["C001", "description_source_terms"] == "2025-1|2025-2"
    assert by_id.loc["C002", "description_match_method"] == (
        "unique_course_name"
    )
    assert by_id.loc["C001", "description_source"] == DESCRIPTION_SOURCE
    assert "강좌기본정보 강좌설명" in by_id.loc[
        "C001", "recommendation_basis"
    ]
    assert by_id.loc["C003", "description"] == (
        "충돌 시 지켜야 하는 기존 설명"
    )
    assert by_id.loc["C003", "description_match_method"] == "ambiguous"
    assert by_id.loc["C004", "description"] == "원본 개설 필드로 만든 설명"


def test_short_and_contact_descriptions_are_not_connected() -> None:
    source = _description_source()
    source.frame.loc[0, "description"] = "짧은 설명"
    source.frame.loc[1, "description"] = (
        "자세한 안내는 test@example.com으로 문의하는 데이터 분석 과목입니다."
    )

    result = build_course_description_enrichment(_catalog(), source)

    by_id = result.candidate_data.set_index("course_id")
    assert by_id.loc["C001", "description"] == "기존 설명"
    assert result.short_description_row_count == 1
    assert result.contact_row_count == 1
    assert "short_descriptions_excluded" in {
        issue.code for issue in result.issues
    }
    assert "contact_rows_excluded" in {issue.code for issue in result.issues}


def test_default_description_file_discovery_is_unicode_safe(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "강좌기본정보.xlsx"
    expected.write_bytes(b"")
    (tmp_path / "~$강좌기본정보.xlsx").write_bytes(b"")

    result = find_default_course_description_file(tmp_path)

    assert result == expected
