"""실제 학과·교과·비교과 Excel을 쓰기 없이 안전하게 점검한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterable, Sequence
import unicodedata

import pandas as pd

from src.department_catalog import configured_department_names
from src.repositories.base import COURSE_COLUMNS
from src.support_program_importer import (
    EMAIL_PATTERN,
    EXCLUDED_AUDIENCE_PATTERN,
    OUTPUT_COLUMNS as PROGRAM_OUTPUT_COLUMNS,
    PHONE_PATTERN,
    SAFE_SOURCE_COLUMN_INDEXES,
    SAFE_SOURCE_COLUMNS,
    transform_wings_programs,
)


ExcelSource = str | Path | bytes | bytearray | BinaryIO

DEPARTMENT_HEADER_ROW = 2
DEPARTMENT_DATA_START_ROW = 3
PROGRAM_HEADER_ROW = 1
PROGRAM_DATA_START_ROW = 3

DEPARTMENT_OUTPUT_COLUMNS = ("department_code", "department_name")
COURSE_OUTPUT_COLUMNS = (
    "course_id",
    "course_name",
    "department",
    "description",
    "learning_objectives",
    "competencies",
    "related_jobs",
    "related_interests",
    "grade_level",
    "semester",
    "difficulty",
    "prerequisites",
    "credit",
    "is_available",
)

PII_HEADER_KEYWORDS = (
    "담당자",
    "운영자",
    "성명",
    "이름",
    "사번",
    "학번",
    "전화번호",
    "연락처",
    "휴대전화",
    "핸드폰",
    "이메일",
    "email",
    "phone",
    "주민등록",
    "생년월일",
    "주소",
)

COURSE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "course_id": ("교과목코드", "과목코드", "학수번호", "course_id"),
    "course_name": ("교과목명", "과목명", "course_name"),
    "department": ("학과", "학과명", "개설학과", "주관학과", "department"),
    "description": ("교과목개요", "과목개요", "교과목설명", "강의개요", "description"),
    "learning_objectives": ("학습목표", "교과목학습목표", "learning_objectives"),
    "competencies": ("역량", "핵심역량", "관련역량", "competencies"),
    "related_jobs": ("관련직무", "연계직무", "related_jobs"),
    "related_interests": ("관련관심분야", "관심분야", "related_interests"),
    "grade_level": ("학년", "개설학년", "grade_level"),
    "semester": ("학기", "개설학기", "semester"),
    "difficulty": ("난이도", "difficulty"),
    "prerequisites": ("선수과목", "선수교과목", "prerequisites"),
    "credit": ("학점", "credit"),
    "is_available": ("개설여부", "운영여부", "사용여부", "is_available"),
    "_source_status": ("상태", "개설상태", "운영상태"),
}

COURSE_REQUIRED_SOURCE_COLUMNS = frozenset(
    {"course_id", "course_name", "department"}
)
COURSE_SEMANTIC_COLUMNS = (
    "competencies",
    "related_jobs",
    "related_interests",
)


@dataclass(frozen=True)
class MasterValidationIssue:
    """개별 master 점검에서 발견한 오류·경고·안내 한 건."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class MasterDryRunResult:
    """앱 저장소에 쓰기 전 한 master 파일의 안전 점검 결과."""

    master_type: str
    display_name: str
    source_name: str
    source_row_count: int
    candidate_data: pd.DataFrame
    blocked_columns: tuple[str, ...]
    ignored_column_count: int
    duplicate_row_count: int
    missing_required_row_count: int
    unknown_department_row_count: int
    issues: tuple[MasterValidationIssue, ...]

    @property
    def accepted_row_count(self) -> int:
        """검증 후 쓰기 후보로 남은 비개인 데이터 행 수를 반환한다."""

        return len(self.candidate_data)

    @property
    def ready(self) -> bool:
        """오류가 하나도 없을 때만 활성화 준비 상태로 본다."""

        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def preview(self) -> pd.DataFrame:
        """화면에 노출할 개인정보 없는 최대 5행 미리보기를 반환한다."""

        return self.candidate_data.head(5).copy()


@dataclass(frozen=True)
class MasterImportDryRunReport:
    """학과·교과·비교과 세 파일의 통합 dry-run 결과."""

    departments: MasterDryRunResult
    courses: MasterDryRunResult
    programs: MasterDryRunResult

    @property
    def results(self) -> tuple[MasterDryRunResult, ...]:
        """화면과 CLI에서 사용할 고정 순서 결과를 반환한다."""

        return (self.departments, self.courses, self.programs)

    @property
    def ready(self) -> bool:
        """세 master가 모두 오류 없이 준비됐는지 반환한다."""

        return all(result.ready for result in self.results)

    def summary_frame(self) -> pd.DataFrame:
        """개인정보 없는 파일별 요약표를 만든다."""

        return pd.DataFrame(
            [
                {
                    "구분": result.display_name,
                    "원본 행": result.source_row_count,
                    "안전 후보 행": result.accepted_row_count,
                    "개인정보 열 차단": len(result.blocked_columns),
                    "중복 행": result.duplicate_row_count,
                    "필수값 누락": result.missing_required_row_count,
                    "학과 연결 오류": result.unknown_department_row_count,
                    "판정": "적용 가능" if result.ready else "보완 필요",
                }
                for result in self.results
            ]
        )


@dataclass(frozen=True)
class SafeExcelFrame:
    """allowlist 열만 읽은 원본 DataFrame과 헤더 점검 정보."""

    frame: pd.DataFrame
    headers: tuple[str, ...]
    blocked_columns: tuple[str, ...]
    ignored_column_count: int


def _fresh_excel_source(source: ExcelSource) -> str | Path | BytesIO:
    """pandas가 반복해서 읽을 수 있는 새 Excel 입력을 반환한다."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Excel 파일을 찾을 수 없습니다: {path}")
        return path
    if isinstance(source, (bytes, bytearray)):
        return BytesIO(bytes(source))
    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        return BytesIO(bytes(getvalue()))
    read = getattr(source, "read", None)
    if not callable(read):
        raise TypeError("Excel 입력은 경로, bytes 또는 binary file이어야 합니다.")
    position = source.tell() if hasattr(source, "tell") else None
    payload = read()
    if position is not None and hasattr(source, "seek"):
        source.seek(position)
    return BytesIO(bytes(payload))


def _source_name(source: ExcelSource, explicit_name: str | None) -> str:
    """경로 또는 업로드 파일에서 값이 아닌 표시용 파일명만 얻는다."""

    if explicit_name:
        return Path(explicit_name).name
    if isinstance(source, (str, Path)):
        return Path(source).name
    return Path(str(getattr(source, "name", "업로드 Excel"))).name


def _clean_text(value: Any) -> str:
    """Excel 빈값·정수형 코드를 안정적인 문자열로 정규화한다."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _normalize_header(value: Any) -> str:
    """공백·기호·유니코드 차이를 제거해 공개 열 alias를 비교한다."""

    normalized = unicodedata.normalize("NFKC", _clean_text(value)).lower()
    return re.sub(r"[\s_\-()/.]+", "", normalized)


def detect_blocked_columns(headers: Iterable[Any]) -> tuple[str, ...]:
    """열 이름만 보고 개인정보 가능 열을 찾아 값 읽기 대상에서 제외한다."""

    blocked: list[str] = []
    normalized_keywords = tuple(
        _normalize_header(keyword) for keyword in PII_HEADER_KEYWORDS
    )
    for header in headers:
        text = _clean_text(header)
        normalized = _normalize_header(text)
        if text and any(keyword in normalized for keyword in normalized_keywords):
            blocked.append(text)
    return tuple(dict.fromkeys(blocked))


def _issue(severity: str, code: str, message: str) -> MasterValidationIssue:
    """일관된 validation issue를 생성한다."""

    return MasterValidationIssue(severity, code, message)


def _contact_redaction_count(values: pd.Series) -> int:
    """허용된 설명 열에서 연락처 패턴이 발견된 행 수를 센다."""

    text = values.fillna("").astype(str)
    return int(
        (text.str.contains(EMAIL_PATTERN, na=False)
         | text.str.contains(PHONE_PATTERN, na=False)).sum()
    )


def _sanitize_public_text(value: Any) -> str:
    """허용된 설명 값에도 섞일 수 있는 이메일·전화번호를 제거한다."""

    text = EMAIL_PATTERN.sub("", _clean_text(value))
    text = PHONE_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" ,;")


def load_department_excel_safe(source: ExcelSource) -> SafeExcelFrame:
    """반복 배치된 학과 Excel에서 학과명·코드 값만 읽는다.

    Parameters:
        source: 실제 학과목록 Excel 경로 또는 업로드 bytes.

    Returns:
        학과명·코드만 포함한 ``SafeExcelFrame``.

    Assumptions:
        3행째 헤더 뒤에 ``학과명``과 ``CODE`` 묶음이 가로로 반복된다.
    """

    header_frame = pd.read_excel(
        _fresh_excel_source(source),
        sheet_name=0,
        header=None,
        nrows=DEPARTMENT_HEADER_ROW + 1,
        dtype=object,
    )
    if len(header_frame) <= DEPARTMENT_HEADER_ROW:
        raise ValueError("학과 Excel에서 3행 헤더를 찾을 수 없습니다.")
    raw_headers = tuple(
        _clean_text(value)
        for value in header_frame.iloc[DEPARTMENT_HEADER_ROW].tolist()
    )
    name_indexes = [
        index
        for index, header in enumerate(raw_headers)
        if _normalize_header(header) in {"학과명", "departmentname"}
    ]
    code_indexes = [
        index
        for index, header in enumerate(raw_headers)
        if _normalize_header(header) in {"code", "학과코드", "departmentcode"}
    ]
    if not name_indexes or len(name_indexes) != len(code_indexes):
        raise ValueError("학과 Excel의 `학과명`·`CODE` 열 묶음을 확인해 주세요.")

    parts: list[pd.DataFrame] = []
    for name_index, code_index in zip(name_indexes, code_indexes):
        part = pd.read_excel(
            _fresh_excel_source(source),
            sheet_name=0,
            header=None,
            skiprows=DEPARTMENT_DATA_START_ROW,
            usecols=[name_index, code_index],
            dtype=object,
        )
        part.columns = DEPARTMENT_OUTPUT_COLUMNS[::-1]
        part = part.loc[:, DEPARTMENT_OUTPUT_COLUMNS]
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    nonempty_headers = tuple(header for header in raw_headers if header)
    blocked = detect_blocked_columns(nonempty_headers)
    selected_count = len(name_indexes) + len(code_indexes)
    return SafeExcelFrame(
        frame=frame,
        headers=nonempty_headers,
        blocked_columns=blocked,
        ignored_column_count=max(
            len(nonempty_headers) - selected_count - len(blocked), 0
        ),
    )


def _course_alias_lookup() -> dict[str, str]:
    """교과 원본 alias를 canonical 열 이름으로 펼친다."""

    return {
        _normalize_header(alias): canonical
        for canonical, aliases in COURSE_COLUMN_ALIASES.items()
        for alias in aliases
    }


def load_course_excel_safe(source: ExcelSource) -> SafeExcelFrame:
    """교과 Excel의 헤더를 먼저 확인하고 allowlist 열 값만 읽는다."""

    headers = tuple(
        _clean_text(header)
        for header in pd.read_excel(
            _fresh_excel_source(source),
            sheet_name=0,
            nrows=0,
        ).columns
    )
    alias_lookup = _course_alias_lookup()
    selected: dict[str, str] = {}
    selected_canonical: set[str] = set()
    for header in headers:
        canonical = alias_lookup.get(_normalize_header(header))
        if canonical and canonical not in selected_canonical:
            selected[header] = canonical
            selected_canonical.add(canonical)
    missing = COURSE_REQUIRED_SOURCE_COLUMNS.difference(selected_canonical)
    if missing:
        raise ValueError(
            f"교과 Excel에 필수 공개 열이 없습니다: {sorted(missing)}"
        )

    frame = pd.read_excel(
        _fresh_excel_source(source),
        sheet_name=0,
        usecols=list(selected),
        dtype=object,
    ).rename(columns=selected)
    blocked = detect_blocked_columns(headers)
    return SafeExcelFrame(
        frame=frame,
        headers=headers,
        blocked_columns=blocked,
        ignored_column_count=max(len(headers) - len(selected) - len(blocked), 0),
    )


def load_program_excel_safe(source: ExcelSource) -> SafeExcelFrame:
    """WINGS Excel에서 고정 allowlist 17개 열만 읽는다."""

    header_frame = pd.read_excel(
        _fresh_excel_source(source),
        sheet_name=0,
        header=None,
        nrows=PROGRAM_HEADER_ROW + 1,
        dtype=object,
    )
    if len(header_frame) <= PROGRAM_HEADER_ROW:
        raise ValueError("비교과 Excel에서 공개 열 헤더를 찾을 수 없습니다.")
    headers = tuple(
        _clean_text(value)
        for value in header_frame.iloc[PROGRAM_HEADER_ROW].tolist()
        if _clean_text(value)
    )
    frame = pd.read_excel(
        _fresh_excel_source(source),
        sheet_name=0,
        header=None,
        skiprows=PROGRAM_DATA_START_ROW,
        usecols=list(SAFE_SOURCE_COLUMN_INDEXES),
        dtype=object,
    )
    frame.columns = SAFE_SOURCE_COLUMNS
    blocked = detect_blocked_columns(headers)
    return SafeExcelFrame(
        frame=frame,
        headers=headers,
        blocked_columns=blocked,
        ignored_column_count=max(
            len(headers) - len(SAFE_SOURCE_COLUMNS) - len(blocked), 0
        ),
    )


def validate_department_frame(
    source: pd.DataFrame,
    *,
    source_name: str,
    blocked_columns: Sequence[str] = (),
    ignored_column_count: int = 0,
) -> MasterDryRunResult:
    """학과 코드·명칭의 결측·중복과 5개 적용 학과 존재를 검증한다."""

    required = set(DEPARTMENT_OUTPUT_COLUMNS)
    missing_columns = required.difference(source.columns)
    if missing_columns:
        raise ValueError(f"학과 필수 열이 없습니다: {sorted(missing_columns)}")
    working = source.loc[:, DEPARTMENT_OUTPUT_COLUMNS].copy()
    for column in DEPARTMENT_OUTPUT_COLUMNS:
        working[column] = working[column].map(_clean_text)
    empty_row = working.eq("").all(axis=1)
    working = working.loc[~empty_row].copy()
    missing_mask = working[list(DEPARTMENT_OUTPUT_COLUMNS)].eq("").any(axis=1)
    duplicate_mask = (
        working["department_code"].ne("")
        & working["department_code"].duplicated(keep=False)
    ) | (
        working["department_name"].ne("")
        & working["department_name"].duplicated(keep=False)
    )
    missing_count = int(missing_mask.sum())
    duplicate_count = int(duplicate_mask.sum())

    candidates = working.loc[~missing_mask & ~duplicate_mask].copy()
    target_names = tuple(configured_department_names())
    actual_names = set(candidates["department_name"])
    missing_targets = tuple(name for name in target_names if name not in actual_names)
    target_order = {name: index for index, name in enumerate(target_names)}
    candidates["_target_order"] = candidates["department_name"].map(
        lambda name: target_order.get(name, len(target_order))
    )
    candidates = candidates.sort_values(
        ["_target_order", "department_name"], kind="stable"
    ).drop(columns="_target_order")
    candidates = candidates.reset_index(drop=True)

    issues: list[MasterValidationIssue] = []
    if blocked_columns:
        issues.append(
            _issue(
                "warning",
                "pii_columns_blocked",
                f"개인정보 가능 열 {len(blocked_columns)}개는 값 자체를 읽지 않았습니다.",
            )
        )
    if missing_count:
        issues.append(
            _issue(
                "error",
                "missing_required_values",
                f"학과 코드 또는 학과명이 빈 행 {missing_count}개를 제외했습니다.",
            )
        )
    if duplicate_count:
        issues.append(
            _issue(
                "error",
                "duplicate_departments",
                f"중복 코드·학과명에 해당하는 행 {duplicate_count}개를 확인해야 합니다.",
            )
        )
    if missing_targets:
        issues.append(
            _issue(
                "error",
                "configured_departments_missing",
                "설정된 5개 적용 학과 중 찾지 못한 학과: "
                + ", ".join(missing_targets),
            )
        )
    else:
        issues.append(
            _issue(
                "info",
                "configured_departments_present",
                "설정된 5개 적용 학과가 실제 학과 파일에 모두 존재합니다.",
            )
        )
    return MasterDryRunResult(
        master_type="departments",
        display_name="학과",
        source_name=source_name,
        source_row_count=len(working),
        candidate_data=candidates,
        blocked_columns=tuple(blocked_columns),
        ignored_column_count=ignored_column_count,
        duplicate_row_count=duplicate_count,
        missing_required_row_count=missing_count,
        unknown_department_row_count=0,
        issues=tuple(issues),
    )


def _availability_value(value: Any) -> bool:
    """명시적인 개설·활성 값만 추천 가능 상태로 변환한다."""

    normalized = _normalize_header(value)
    return normalized in {
        "y",
        "yes",
        "true",
        "1",
        "개설",
        "개설중",
        "운영",
        "운영중",
        "활성",
        "사용",
        "사용중",
    }


def validate_course_frame(
    source: pd.DataFrame,
    known_department_names: Iterable[str],
    *,
    source_name: str,
    blocked_columns: Sequence[str] = (),
    ignored_column_count: int = 0,
) -> MasterDryRunResult:
    """교과 ID·필수값·학과 연결·개설 여부와 추천 메타데이터를 검증한다."""

    missing_source = COURSE_REQUIRED_SOURCE_COLUMNS.difference(source.columns)
    if missing_source:
        raise ValueError(f"교과 필수 열이 없습니다: {sorted(missing_source)}")
    working = source.copy()
    source_status = (
        working["is_available"]
        if "is_available" in working.columns
        else working.get("_source_status", pd.Series("", index=working.index))
    )
    availability_known = source_status.map(_clean_text).ne("")
    for column in COURSE_OUTPUT_COLUMNS:
        if column not in working.columns:
            working[column] = False if column == "is_available" else ""
    working = working.loc[:, COURSE_OUTPUT_COLUMNS].copy()
    for column in COURSE_OUTPUT_COLUMNS:
        if column != "is_available":
            working[column] = working[column].map(_clean_text)
    working["is_available"] = source_status.map(_availability_value)

    redaction_count = 0
    for column in (
        "course_name",
        "description",
        "learning_objectives",
        "competencies",
        "related_jobs",
        "related_interests",
        "difficulty",
        "prerequisites",
    ):
        redaction_count += _contact_redaction_count(working[column])
        working[column] = working[column].map(_sanitize_public_text)

    empty_row = working[["course_id", "course_name", "department"]].eq("").all(
        axis=1
    )
    working = working.loc[~empty_row].copy()
    required_columns = ["course_id", "course_name", "department"]
    missing_mask = working[required_columns].eq("").any(axis=1)
    duplicate_mask = (
        working["course_id"].ne("")
        & working["course_id"].duplicated(keep=False)
    )
    known_departments = {
        _clean_text(value) for value in known_department_names if _clean_text(value)
    }
    unknown_mask = (
        working["department"].ne("")
        & ~working["department"].isin(known_departments)
    )
    unknown_names = tuple(
        sorted(working.loc[unknown_mask, "department"].unique().tolist())
    )
    missing_count = int(missing_mask.sum())
    duplicate_count = int(duplicate_mask.sum())
    unknown_count = int(unknown_mask.sum())

    valid_mask = ~missing_mask & ~duplicate_mask & ~unknown_mask
    candidates = working.loc[valid_mask].drop_duplicates(
        subset=["course_id"], keep="first"
    )
    candidates = candidates.sort_values(
        ["department", "course_name", "course_id"], kind="stable"
    ).reset_index(drop=True)

    missing_semantic_columns = tuple(
        column for column in COURSE_SEMANTIC_COLUMNS if column not in source.columns
    )
    missing_semantic_rows = int(
        candidates[list(COURSE_SEMANTIC_COLUMNS)].eq("").all(axis=1).sum()
    )
    missing_description_count = int(candidates["description"].eq("").sum())
    unavailable_status_count = int((~availability_known.loc[working.index]).sum())
    target_names = set(configured_department_names())
    target_departments_with_courses = set(candidates["department"]) & target_names
    missing_target_courses = tuple(sorted(target_names - target_departments_with_courses))

    issues: list[MasterValidationIssue] = []
    if blocked_columns:
        issues.append(
            _issue(
                "warning",
                "pii_columns_blocked",
                f"개인정보 가능 열 {len(blocked_columns)}개는 값 자체를 읽지 않았습니다.",
            )
        )
    if redaction_count:
        issues.append(
            _issue(
                "warning",
                "contacts_redacted",
                f"허용된 설명 열에 섞인 연락처 패턴 {redaction_count}건을 제거했습니다.",
            )
        )
    if missing_count:
        issues.append(
            _issue(
                "error",
                "missing_required_values",
                f"교과목 코드·명·학과 중 필수값이 빈 행 {missing_count}개를 제외했습니다.",
            )
        )
    if duplicate_count:
        issues.append(
            _issue(
                "error",
                "duplicate_courses",
                f"중복 교과목 코드에 해당하는 행 {duplicate_count}개를 확인해야 합니다.",
            )
        )
    if unknown_count:
        sample = ", ".join(unknown_names[:8])
        suffix = f" 외 {len(unknown_names) - 8}개" if len(unknown_names) > 8 else ""
        issues.append(
            _issue(
                "error",
                "unknown_course_departments",
                f"학과 master와 연결되지 않는 교과목 {unknown_count}행이 있습니다: "
                f"{sample}{suffix}",
            )
        )
    if unavailable_status_count:
        issues.append(
            _issue(
                "error",
                "course_availability_missing",
                f"현재 개설·수강 가능 여부를 확인할 수 없는 교과목이 "
                f"{unavailable_status_count}행입니다. 활성화 전 학기별 개설정보가 필요합니다.",
            )
        )
    if missing_semantic_columns or missing_semantic_rows:
        issues.append(
            _issue(
                "warning",
                "course_semantics_missing",
                "역량·관련 직무·관심 분야가 비어 있어 임베딩 추천 품질 검증이 "
                f"필요한 안전 후보가 {missing_semantic_rows}행입니다.",
            )
        )
    if missing_description_count:
        issues.append(
            _issue(
                "warning",
                "course_descriptions_missing",
                f"교과목 개요가 빈 안전 후보 {missing_description_count}행은 "
                "강의계획서 또는 교과목 개요 보완이 필요합니다.",
            )
        )
    if missing_target_courses:
        issues.append(
            _issue(
                "error",
                "target_department_courses_missing",
                "5개 적용 학과 중 연결 교과목이 없는 학과: "
                + ", ".join(missing_target_courses),
            )
        )
    else:
        issues.append(
            _issue(
                "info",
                "cross_department_courses_preserved",
                "5개 소속 학과 교과목을 확인했으며, 연결이 유효한 타과 교과목도 "
                "추천 후보에서 hard filter하지 않았습니다.",
            )
        )
    if set(COURSE_OUTPUT_COLUMNS) != set(COURSE_COLUMNS):
        raise RuntimeError("교과 dry-run 스키마와 Repository 계약이 다릅니다.")
    return MasterDryRunResult(
        master_type="courses",
        display_name="교과목",
        source_name=source_name,
        source_row_count=len(working),
        candidate_data=candidates,
        blocked_columns=tuple(blocked_columns),
        ignored_column_count=ignored_column_count,
        duplicate_row_count=duplicate_count,
        missing_required_row_count=missing_count,
        unknown_department_row_count=unknown_count,
        issues=tuple(issues),
    )


def validate_program_frame(
    source: pd.DataFrame,
    *,
    source_name: str,
    blocked_columns: Sequence[str] = (),
    ignored_column_count: int = 0,
) -> MasterDryRunResult:
    """WINGS 공개 열을 추천 master로 변환하고 중복·연락처를 재검증한다."""

    required_source_columns = {
        "program_group",
        "program_name",
        "program_format",
        "status",
        "academic_year",
        "semester",
        "operation_start_date",
    }
    missing_columns = required_source_columns.difference(source.columns)
    if missing_columns:
        raise ValueError(
            f"비교과 필수 공개 열이 없습니다: {sorted(missing_columns)}"
        )
    normalized = source.copy()
    for column in required_source_columns:
        normalized[column] = normalized[column].map(_clean_text)
    required_values = ["program_name", "program_format"]
    missing_mask = normalized[required_values].eq("").any(axis=1)
    audience = normalized["program_group"] + " " + normalized["program_name"]
    eligible_mask = (
        ~missing_mask
        & normalized["program_format"].str.contains(
            "비교과", regex=False, na=False
        )
        & ~audience.str.contains(EXCLUDED_AUDIENCE_PATTERN, na=False)
    )
    identity_columns = [
        "program_group",
        "program_name",
        "academic_year",
        "semester",
        "operation_start_date",
    ]
    duplicate_mask = normalized.loc[eligible_mask, identity_columns].duplicated(
        keep=False
    )
    duplicate_count = int(duplicate_mask.sum())
    missing_count = int(missing_mask.sum())
    programs = transform_wings_programs(source)
    searchable = " ".join(programs.fillna("").astype(str).to_numpy().ravel())
    issues: list[MasterValidationIssue] = []
    if blocked_columns:
        issues.append(
            _issue(
                "warning",
                "pii_columns_blocked",
                f"개인정보 가능 열 {len(blocked_columns)}개는 값 자체를 읽지 않았습니다: "
                + ", ".join(blocked_columns),
            )
        )
    if duplicate_count:
        issues.append(
            _issue(
                "error",
                "duplicate_programs",
                f"중복 프로그램 ID에 해당하는 행 {duplicate_count}개가 있습니다.",
            )
        )
    if missing_count:
        issues.append(
            _issue(
                "error",
                "missing_required_values",
                f"프로그램명·형식 중 필수값이 빈 행 {missing_count}개를 "
                "확인해야 합니다.",
            )
        )
    if EMAIL_PATTERN.search(searchable) or PHONE_PATTERN.search(searchable):
        issues.append(
            _issue(
                "error",
                "program_contacts_remaining",
                "안전 후보에 이메일 또는 전화번호 패턴이 남아 있습니다.",
            )
        )
    excluded_count = max(len(source) - len(programs), 0)
    if excluded_count:
        issues.append(
            _issue(
                "info",
                "non_extracurricular_programs_excluded",
                f"비교과 외 형식·재학생 외 대상 등 {excluded_count}행을 "
                "추천 후보에서 제외했습니다.",
            )
        )
    if not any(issue.severity == "error" for issue in issues):
        issues.append(
            _issue(
                "info",
                "programs_ready",
                "운영 상태와 무관한 재학생용 WINGS 비교과 전체를 안전 후보로 "
                "확인했습니다.",
            )
        )
    return MasterDryRunResult(
        master_type="programs",
        display_name="비교과",
        source_name=source_name,
        source_row_count=len(source),
        candidate_data=programs.loc[:, list(PROGRAM_OUTPUT_COLUMNS)].copy(),
        blocked_columns=tuple(blocked_columns),
        ignored_column_count=ignored_column_count,
        duplicate_row_count=duplicate_count,
        missing_required_row_count=missing_count,
        unknown_department_row_count=0,
        issues=tuple(issues),
    )


def _failed_result(
    master_type: str,
    display_name: str,
    source_name: str,
    error: Exception,
    columns: Sequence[str],
) -> MasterDryRunResult:
    """파일 하나가 잘못돼도 다른 master 점검을 계속할 수 있게 한다."""

    return MasterDryRunResult(
        master_type=master_type,
        display_name=display_name,
        source_name=source_name,
        source_row_count=0,
        candidate_data=pd.DataFrame(columns=columns),
        blocked_columns=(),
        ignored_column_count=0,
        duplicate_row_count=0,
        missing_required_row_count=0,
        unknown_department_row_count=0,
        issues=(
            _issue(
                "error",
                "file_read_failed",
                f"{display_name} Excel을 안전하게 읽지 못했습니다: {error}",
            ),
        ),
    )


def run_master_import_dry_run(
    department_source: ExcelSource,
    course_source: ExcelSource,
    program_source: ExcelSource,
    *,
    department_name: str | None = None,
    course_name: str | None = None,
    program_name: str | None = None,
) -> MasterImportDryRunReport:
    """세 실제 Excel을 메모리에서만 읽고 통합 점검한다.

    Parameters:
        department_source: 학과목록 Excel.
        course_source: 전체 교과목 Excel.
        program_source: WINGS 전체 프로그램 Excel.
        *_name: 업로드 화면에 표시할 원본 파일명.

    Returns:
        파일별 안전 후보와 오류·경고를 포함한 dry-run 보고서.

    Assumptions:
        함수는 CSV·SQLite·설정 파일을 생성하거나 수정하지 않는다.
    """

    department_source_name = _source_name(department_source, department_name)
    known_department_names: list[str] = []
    try:
        loaded_departments = load_department_excel_safe(department_source)
        known_department_names = [
            _clean_text(value)
            for value in loaded_departments.frame["department_name"]
            if _clean_text(value)
        ]
        departments = validate_department_frame(
            loaded_departments.frame,
            source_name=department_source_name,
            blocked_columns=loaded_departments.blocked_columns,
            ignored_column_count=loaded_departments.ignored_column_count,
        )
    except Exception as error:  # UI에서 파일 하나의 오류로 전체가 멈추지 않게 한다.
        departments = _failed_result(
            "departments",
            "학과",
            department_source_name,
            error,
            DEPARTMENT_OUTPUT_COLUMNS,
        )

    course_source_name = _source_name(course_source, course_name)
    try:
        loaded_courses = load_course_excel_safe(course_source)
        courses = validate_course_frame(
            loaded_courses.frame,
            known_department_names,
            source_name=course_source_name,
            blocked_columns=loaded_courses.blocked_columns,
            ignored_column_count=loaded_courses.ignored_column_count,
        )
    except Exception as error:  # 위와 같은 독립 실패 처리
        courses = _failed_result(
            "courses",
            "교과목",
            course_source_name,
            error,
            COURSE_OUTPUT_COLUMNS,
        )

    program_source_name = _source_name(program_source, program_name)
    try:
        loaded_programs = load_program_excel_safe(program_source)
        programs = validate_program_frame(
            loaded_programs.frame,
            source_name=program_source_name,
            blocked_columns=loaded_programs.blocked_columns,
            ignored_column_count=loaded_programs.ignored_column_count,
        )
    except Exception as error:  # 위와 같은 독립 실패 처리
        programs = _failed_result(
            "programs",
            "비교과",
            program_source_name,
            error,
            PROGRAM_OUTPUT_COLUMNS,
        )
    return MasterImportDryRunReport(departments, courses, programs)


def _print_report(report: MasterImportDryRunReport) -> None:
    """CLI에 개인정보 없는 행 수와 점검 메시지만 출력한다."""

    overall = "적용 준비 완료" if report.ready else "보완 필요"
    print(f"실제 master dry-run: {overall} (어떤 파일도 저장하지 않았습니다.)")
    for result in report.results:
        status = "적용 가능" if result.ready else "보완 필요"
        print(
            f"- {result.display_name}: 원본 {result.source_row_count}행 / "
            f"안전 후보 {result.accepted_row_count}행 / {status}"
        )
        if result.blocked_columns:
            print(f"  개인정보 열 차단 {len(result.blocked_columns)}개")
        for issue in result.issues:
            print(f"  [{issue.severity.upper()}] {issue.message}")


def main() -> None:
    """실제 세 master를 저장 없이 점검하는 CLI 진입점."""

    parser = argparse.ArgumentParser(
        description="학과·교과·비교과 Excel을 앱에 쓰지 않고 안전 점검합니다."
    )
    parser.add_argument("--departments", required=True, help="학과목록 Excel")
    parser.add_argument("--courses", required=True, help="전체 교과목 Excel")
    parser.add_argument("--programs", required=True, help="WINGS 비교과 Excel")
    args = parser.parse_args()
    _print_report(
        run_master_import_dry_run(
            args.departments,
            args.courses,
            args.programs,
        )
    )


if __name__ == "__main__":
    main()
