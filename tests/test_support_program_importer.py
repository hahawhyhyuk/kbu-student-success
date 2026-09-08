"""WINGS 비교과 master의 개인정보 제거·필터·ID 안정성 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.support_program_importer import (
    OUTPUT_COLUMNS,
    SAFE_SOURCE_COLUMNS,
    SAFE_SOURCE_COLUMN_INDEXES,
    WINGS_URL,
    load_wings_programs,
    transform_wings_programs,
)


def _source_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "program_group": "학생성공",
        "program_name": "학생 데이터 특강",
        "program_overview": "데이터 기초 학습과 직무 탐색을 지원합니다.",
        "large_category": "학습지원",
        "middle_category": "특강",
        "program_format": "비교과 (비학점)",
        "status": "신청/운영중",
        "academic_year": 2026,
        "semester": "2학기",
        "operation_start_date": pd.Timestamp("2026-09-10"),
        "operation_end_date": pd.Timestamp("2026-10-10"),
        "application_start_date": pd.Timestamp("2026-09-01"),
        "application_end_date": pd.Timestamp("2026-09-09"),
        "operating_institution": "학생성공처",
        "operating_department": "학생성공센터",
        "primary_competency": "실무 전문성 역량",
        "secondary_competency": "창의융합역량",
    }
    row.update(overrides)
    return row


def test_transform_keeps_all_student_program_statuses_and_removes_contacts() -> None:
    source = pd.DataFrame(
        [
            _source_row(
                program_overview=(
                    "데이터 학습 특강 문의 010-1234-5678 "
                    "test@example.com"
                )
            ),
            _source_row(program_name="교직원 대상 의무교육"),
            _source_row(program_name="마감된 프로그램", status="마감완료"),
            _source_row(program_name="상태 미정 프로그램", status=""),
            _source_row(program_name="기타 행사", program_format="기타"),
        ],
        columns=SAFE_SOURCE_COLUMNS,
    )

    result = transform_wings_programs(source)

    assert len(result) == 3
    assert tuple(result.columns) == OUTPUT_COLUMNS
    assert result["application_url"].eq(WINGS_URL).all()
    assert set(result["source_status"]) == {"신청/운영중", "마감완료", ""}
    assert "교직원 대상 의무교육" not in set(result["program_name"])
    assert "기타 행사" not in set(result["program_name"])
    serialized = " ".join(result.astype(str).to_numpy().ravel())
    assert "010-1234-5678" not in serialized
    assert "test@example.com" not in serialized
    assert "staff_id" not in result.columns
    assert "phone" not in result.columns
    assert "email" not in result.columns


def test_program_ids_are_stable_when_source_order_changes() -> None:
    first = _source_row(program_name="학습전략 특강")
    second = _source_row(
        program_name="진로 탐색 특강",
        large_category="취·창업지원",
    )

    original = transform_wings_programs(
        pd.DataFrame([first, second], columns=SAFE_SOURCE_COLUMNS)
    )
    reversed_result = transform_wings_programs(
        pd.DataFrame([second, first], columns=SAFE_SOURCE_COLUMNS)
    )

    original_ids = original.set_index("program_name")["program_id"].to_dict()
    reversed_ids = reversed_result.set_index("program_name")["program_id"].to_dict()
    assert original_ids == reversed_ids


def test_excel_loader_requests_only_public_source_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workbook = tmp_path / "wings.xlsx"
    workbook.touch()
    captured: dict[str, object] = {}

    def fake_read_excel(path: Path, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            [[_source_row()[column] for column in SAFE_SOURCE_COLUMNS]]
        )

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = load_wings_programs(workbook)

    assert len(result) == 1
    assert captured["usecols"] == list(SAFE_SOURCE_COLUMN_INDEXES)
    assert not {33, 34, 35, 36, 37, 38, 39, 40}.intersection(
        SAFE_SOURCE_COLUMN_INDEXES
    )
