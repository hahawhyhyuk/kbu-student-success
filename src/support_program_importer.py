"""WINGS 비교과 Excel을 개인정보 없는 추천 master로 변환한다."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.repositories.base import SUPPORT_PROGRAM_COLUMNS


WINGS_URL = "https://wings.kbu.ac.kr"
SAFE_SOURCE_COLUMNS = (
    "program_group",
    "program_name",
    "program_overview",
    "large_category",
    "middle_category",
    "program_format",
    "status",
    "academic_year",
    "semester",
    "operation_start_date",
    "operation_end_date",
    "application_start_date",
    "application_end_date",
    "operating_institution",
    "operating_department",
    "primary_competency",
    "secondary_competency",
)
SAFE_SOURCE_COLUMN_INDEXES = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    11,
    12,
    13,
    15,
    17,
    19,
    31,
    32,
    41,
    42,
)
OUTPUT_COLUMNS = (
    "program_id",
    "program_name",
    "program_type",
    "description",
    "target_risk_types",
    "provided_competencies",
    "target_students",
    "department_in_charge",
    "operation_period",
    "application_url",
    "source_system",
    "source_status",
    "academic_year",
    "semester",
)
RISK_ORDER = (
    "attendance",
    "engagement",
    "achievement",
    "major_adaptation",
    "career",
    "complex",
)
CATEGORY_RISK_MAP = {
    "학습지원": {"engagement", "achievement"},
    "학생활동지원": {"engagement", "major_adaptation"},
    "취·창업지원": {"career"},
    "심리상담지원": {"major_adaptation", "career", "complex"},
}
RISK_KEYWORDS = {
    "attendance": ("출석", "결석", "등교"),
    "engagement": ("학습", "튜터링", "멘토링", "동아리", "활동", "특강"),
    "achievement": ("기초", "성적", "시험", "자격", "교과", "경진대회"),
    "major_adaptation": ("전공", "학과", "적응", "동아리"),
    "career": ("진로", "취업", "창업", "직무", "자격"),
    "complex": ("상담", "심리", "마음", "스트레스", "위기"),
}
EXCLUDED_AUDIENCE_PATTERN = re.compile(
    r"교직원|교원\s*대상|직원\s*대상|교수\s*대상|"
    r"고등학생\s*대상|중학생\s*대상|초등학생\s*대상|"
    r"학부모\s*대상|어린이\s*대상",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_PATTERN = re.compile(
    r"(?:01[016789]|02|0[3-6][1-5])[- .)]?\d{3,4}[- .]?\d{4}"
)


def _clean_text(value: Any) -> str:
    """Excel 값을 빈칸 보존 문자열로 정규화한다."""

    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _sanitize_description(value: Any) -> str:
    """프로그램 개요에서 연락처를 제거하고 과도한 길이를 제한한다."""

    text = EMAIL_PATTERN.sub("", _clean_text(value))
    text = PHONE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text if len(text) <= 500 else text[:497].rstrip() + "..."


def _date_text(value: Any) -> str:
    """Excel 날짜를 YYYY-MM-DD 표시용 문자열로 변환한다."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    text = _clean_text(value)
    match = re.search(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", text)
    if not match:
        return text
    return match.group(0).replace(".", "-").replace("/", "-")


def _plain_integer_text(value: Any) -> str:
    """2026.0과 같은 Excel 숫자를 2026 문자열로 변환한다."""

    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return _clean_text(value)


def _unique_join(values: Sequence[Any]) -> str:
    """빈 값과 중복을 제거하고 pipe 문자열로 합친다."""

    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            seen.add(text)
            items.append(text)
    return "|".join(items)


def _target_risk_types(row: pd.Series) -> str:
    """공개 분류·설명에서 초기경보 연결 영역을 결정한다."""

    large_category = _clean_text(row["large_category"])
    targets = set(CATEGORY_RISK_MAP.get(large_category, set()))
    searchable = " ".join(
        _clean_text(row[column])
        for column in (
            "program_name",
            "program_overview",
            "large_category",
            "middle_category",
        )
    ).replace(" ", "")
    for risk_type, keywords in RISK_KEYWORDS.items():
        if any(keyword.replace(" ", "") in searchable for keyword in keywords):
            targets.add(risk_type)
    if not targets:
        targets.add("engagement")
    return "|".join(item for item in RISK_ORDER if item in targets)


def _stable_program_id(row: pd.Series) -> str:
    """원본 순서가 바뀌어도 유지되는 비식별 프로그램 ID를 만든다."""

    identity = "|".join(
        (
            _clean_text(row["program_group"]),
            _clean_text(row["program_name"]),
            _plain_integer_text(row["academic_year"]),
            _clean_text(row["semester"]),
            _date_text(row["operation_start_date"]),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12].upper()
    return f"WINGS-{digest}"


def _operation_period(row: pd.Series) -> str:
    """신청 기간과 운영 기간을 한 줄로 정리한다."""

    application_start = _date_text(row["application_start_date"])
    application_end = _date_text(row["application_end_date"])
    operation_start = _date_text(row["operation_start_date"])
    operation_end = _date_text(row["operation_end_date"])
    periods = []
    if application_start or application_end:
        periods.append(f"신청 {application_start or '?'}~{application_end or '?'}")
    if operation_start or operation_end:
        periods.append(f"운영 {operation_start or '?'}~{operation_end or '?'}")
    return " · ".join(periods) or "WINGS에서 일정 확인"


def transform_wings_programs(source: pd.DataFrame) -> pd.DataFrame:
    """WINGS 공개 프로그램 열만 canonical master로 변환한다.

    Parameters:
        source: ``SAFE_SOURCE_COLUMNS``로 제한된 WINGS 원본 DataFrame.

    Returns:
        운영 상태와 무관한 재학생용 비교과 프로그램 master.

    Assumptions:
        담당자·사번·전화·이메일·예산 열은 입력에도 포함하지 않는다.
        원본 운영 상태는 추적용으로만 보존하고 후보 제외에 사용하지 않는다.
    """

    missing = set(SAFE_SOURCE_COLUMNS).difference(source.columns)
    if missing:
        raise ValueError(f"WINGS 원본에 필수 공개 열이 없습니다: {sorted(missing)}")

    working = source.loc[:, SAFE_SOURCE_COLUMNS].copy()
    names = working["program_name"].map(_clean_text)
    formats = working["program_format"].map(_clean_text)
    audience_names = (
        working["program_group"].map(_clean_text) + " " + names
    )
    mask = (
        names.ne("")
        & formats.str.contains("비교과", regex=False, na=False)
        & ~audience_names.str.contains(EXCLUDED_AUDIENCE_PATTERN, na=False)
    )
    working = working.loc[mask].copy()

    records: list[dict[str, str]] = []
    for _, row in working.iterrows():
        program_name = _clean_text(row["program_name"])
        large_category = _clean_text(row["large_category"])
        middle_category = _clean_text(row["middle_category"])
        description = _sanitize_description(row["program_overview"])
        if not description:
            description = (
                f"{large_category or '학생지원'} 분야의 "
                f"{middle_category or '비교과'} 프로그램입니다."
            )
        department = _clean_text(row["operating_department"])
        institution = _clean_text(row["operating_institution"])
        records.append(
            {
                "program_id": _stable_program_id(row),
                "program_name": program_name,
                "program_type": middle_category or large_category or "비교과",
                "description": description,
                "target_risk_types": _target_risk_types(row),
                "provided_competencies": _unique_join(
                    (row["primary_competency"], row["secondary_competency"])
                )
                or middle_category
                or large_category,
                "target_students": "경복대학교 재학생(세부 참여 조건은 WINGS 확인)",
                "department_in_charge": department or institution or "WINGS 확인",
                "operation_period": _operation_period(row),
                "application_url": WINGS_URL,
                "source_system": "WINGS",
                "source_status": _clean_text(row["status"]),
                "academic_year": _plain_integer_text(row["academic_year"]),
                "semester": _clean_text(row["semester"]),
            }
        )

    output = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if output.empty:
        raise ValueError("재학생용 WINGS 비교과 프로그램이 없습니다.")
    output = output.drop_duplicates(subset=["program_id"]).sort_values(
        ["academic_year", "program_name", "program_id"],
        kind="stable",
    )
    output = output.reset_index(drop=True)
    validate_imported_programs(output)
    return output


def validate_imported_programs(programs: pd.DataFrame) -> None:
    """변환된 master의 스키마·ID·연결 영역·개인정보 제거를 검증한다."""

    missing = SUPPORT_PROGRAM_COLUMNS.difference(programs.columns)
    if missing:
        raise ValueError(f"비교과 master 필수 열이 없습니다: {sorted(missing)}")
    if programs["program_id"].duplicated().any():
        raise ValueError("비교과 master의 program_id는 중복될 수 없습니다.")
    if programs["program_name"].astype(str).str.strip().eq("").any():
        raise ValueError("비교과 master에 프로그램명이 빈 행이 있습니다.")
    valid_risk_types = set(RISK_ORDER)
    for value in programs["target_risk_types"]:
        targets = {item for item in str(value).split("|") if item}
        if not targets or not targets.issubset(valid_risk_types):
            raise ValueError(f"잘못된 추천 연결 영역입니다: {value}")
    searchable = " ".join(
        programs.fillna("").astype(str).to_numpy().ravel().tolist()
    )
    if EMAIL_PATTERN.search(searchable) or PHONE_PATTERN.search(searchable):
        raise ValueError("비교과 master에 이메일 또는 전화번호가 남아 있습니다.")


def load_wings_programs(workbook_path: str | Path) -> pd.DataFrame:
    """Excel에서 공개 프로그램 열만 읽어 변환한다."""

    path = Path(workbook_path)
    if not path.exists():
        raise FileNotFoundError(f"WINGS 비교과 파일을 찾을 수 없습니다: {path}")
    source = pd.read_excel(
        path,
        sheet_name=0,
        header=None,
        skiprows=3,
        usecols=list(SAFE_SOURCE_COLUMN_INDEXES),
        dtype=object,
    )
    source.columns = SAFE_SOURCE_COLUMNS
    return transform_wings_programs(source)


def save_wings_programs(
    workbook_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """검증을 통과한 master만 CSV로 저장한다."""

    programs = load_wings_programs(workbook_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    programs.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(destination)
    return programs


def main() -> None:
    """CLI에서 WINGS Excel을 검증된 CSV로 변환한다."""

    parser = argparse.ArgumentParser(
        description="WINGS 비교과 Excel을 추천용 안전 master로 변환합니다."
    )
    parser.add_argument("workbook", help="WINGS 비교과 Excel 경로")
    parser.add_argument(
        "--output",
        default="data/support_programs_wings.csv",
        help="저장할 canonical CSV 경로",
    )
    args = parser.parse_args()
    programs = save_wings_programs(args.workbook, args.output)
    print(f"{len(programs)}개 WINGS 프로그램을 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
