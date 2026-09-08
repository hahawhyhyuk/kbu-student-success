"""2026년 실제 개설강좌를 원본 수정 없이 안전하게 점검한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any, BinaryIO, Mapping, Sequence
import unicodedata

import pandas as pd

from src.department_catalog import configured_department_names
from src.master_import_service import (
    COURSE_OUTPUT_COLUMNS,
    ExcelSource,
    detect_blocked_columns,
    load_course_excel_safe,
)


OFFERING_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "department_code": ("개설학과코드",),
    "department_name": ("개설학과",),
    "course_id": ("교과목코드", "과목코드", "학수번호"),
    "course_name": ("교과목명", "과목명"),
    "grade_level": ("개설학년", "학년"),
    "section": ("수강분반", "분반"),
    "course_area": ("영역구분",),
    "class_method": ("수업방식",),
    "course_type": ("교과목구분",),
    "ncs_type": ("NCS과목구분",),
    "enrolled_count": ("수강인원",),
    "capacity": ("제한인원", "수강정원"),
    "offering_type": ("개설강좌구분",),
    "day_night": ("주야구분",),
    "offering_status": ("개설강좌상태구분", "개설상태"),
    "cancellation_status": ("개설강좌폐강구분", "폐강구분"),
    "class_weeks": ("수업주수",),
    "credit": ("교과목학점", "학점"),
    "lecture_hours": ("이론강의시간수",),
    "practice_hours": ("실험실습시간수",),
    "total_hours": ("강의시간수",),
    "lecture_type": ("강의유형구분",),
    "evaluation_type": ("성적평가구분",),
    "pass_fail": ("패스교과목여부",),
    "prior_enrollment_required": ("선수강여부",),
    "cross_department_excluded": ("타학과수강제외여부",),
    "online": ("온라인강좌여부",),
    "team_teaching": ("팀티칭여부",),
    "intensive_type": ("집중수업유형내용",),
    "schedule_summary": ("수업편성요약내용",),
    "convergence_type": ("융복합구분",),
    "instruction_progress": ("수업진행구분",),
    "curriculum_year": ("교육과정년도",),
    "curriculum_department_code": ("교육과정부서코드",),
}
REQUIRED_OFFERING_COLUMNS = frozenset(
    {
        "department_code",
        "department_name",
        "course_id",
        "course_name",
        "grade_level",
        "credit",
        "offering_status",
    }
)
COURSE_OFFERING_OUTPUT_COLUMNS: tuple[str, ...] = (
    "academic_year",
    "semester",
    "course_id",
    "course_name",
    "department_code",
    "department_name",
    "is_target_department",
    "grade_levels",
    "credit",
    "section_count",
    "sections",
    "course_area",
    "class_method",
    "course_type",
    "ncs_type",
    "lecture_hours",
    "practice_hours",
    "total_hours",
    "availability_status",
    "cross_department_status",
    "online_status",
    "enrolled_count",
    "capacity",
    "curriculum_year",
    "master_match_method",
    "linked_master_course_id",
    "master_description",
)

ACTUAL_COURSE_CATALOG_SOURCE = "actual_course_offerings_2026"
ACTUAL_COURSE_CATALOG_EXTRA_COLUMNS: tuple[str, ...] = (
    "catalog_source",
    "source_course_id",
    "academic_years",
    "offered_semesters",
    "offering_departments",
    "offering_department_codes",
    "grade_levels",
    "course_area",
    "class_method",
    "course_type",
    "ncs_type",
    "cross_department_status",
    "online_status",
    "recommendation_basis",
    "prerequisite_status",
    "completion_data_status",
    "registration_check_required",
    "master_match_method",
    "linked_master_course_id",
)
ACTUAL_COURSE_CATALOG_COLUMNS: tuple[str, ...] = tuple(COURSE_OUTPUT_COLUMNS) + (
    ACTUAL_COURSE_CATALOG_EXTRA_COLUMNS
)

ACTIVE_STATUS_VALUES = frozenset(
    {"개설", "개설중", "운영", "운영중", "활성", "사용", "사용중"}
)
CANCELLED_STATUS_VALUES = frozenset(
    {"폐강", "폐강확정", "취소", "1", "y", "yes", "true"}
)
TRUE_VALUES = frozenset({"1", "y", "yes", "true", "해당", "예"})
FALSE_VALUES = frozenset({"0", "n", "no", "false", "미해당", "아니오"})

TARGET_DEPARTMENT_ALIASES: dict[str, frozenset[str]] = {
    "친환경건축과": frozenset({"친환경건축과"}),
    "유아교육과": frozenset({"유아교육과", "유아교육학과"}),
    "실용음악과(3년제)": frozenset(
        {"실용음악과(3년제)", "실용음악과", "실용음악학과"}
    ),
    "간호학과": frozenset({"간호학과"}),
    "소프트웨어융합과": frozenset(
        {"소프트웨어융합과(2022)", "소프트웨어융합과", "소프트웨어융합학과"}
    ),
}


@dataclass(frozen=True)
class SafeOfferingFrame:
    """개인정보 가능 열을 읽지 않은 한 학기 개설강좌 원본."""

    frame: pd.DataFrame
    source_name: str
    academic_year: int
    semester: int
    source_row_count: int
    blocked_columns: tuple[str, ...]
    ignored_column_count: int


@dataclass(frozen=True)
class CourseOfferingImportIssue:
    """개설강좌 dry-run에서 발견한 오류·경고·안내 한 건."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class CourseOfferingDryRunResult:
    """두 학기 전 학과 개설강좌와 전체 교과목 연결 점검 결과."""

    candidate_data: pd.DataFrame
    source_files: tuple[SafeOfferingFrame, ...]
    issues: tuple[CourseOfferingImportIssue, ...]
    valid_section_count: int
    excluded_required_count: int
    consolidated_section_count: int
    master_code_match_count: int
    master_name_match_count: int
    master_unmatched_count: int
    target_department_course_count: int

    @property
    def ready(self) -> bool:
        """필수 열 또는 파일 오류가 없으면 연결 점검 가능 상태로 본다."""

        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def course_count(self) -> int:
        """학기와 개설학과를 구분한 통합 강좌 수를 반환한다."""

        return len(self.candidate_data)

    @property
    def preview(self) -> pd.DataFrame:
        """개인정보가 없는 연결 결과 최대 8행을 반환한다."""

        preview_columns = (
            "academic_year",
            "semester",
            "course_id",
            "course_name",
            "department_name",
            "grade_levels",
            "credit",
            "section_count",
            "availability_status",
            "cross_department_status",
            "master_match_method",
        )
        return self.candidate_data.loc[:, list(preview_columns)].head(8).copy()

    def summary_frame(self) -> pd.DataFrame:
        """학기별 원본·분반·통합 과목·연결 현황을 표시한다."""

        records: list[dict[str, Any]] = []
        for source in self.source_files:
            semester_rows = self.candidate_data[
                (self.candidate_data["academic_year"] == source.academic_year)
                & (self.candidate_data["semester"] == source.semester)
            ]
            records.append(
                {
                    "학기": f"{source.academic_year}-{source.semester}",
                    "원본 행": source.source_row_count,
                    "유효 분반": int(semester_rows["section_count"].sum()),
                    "통합 강좌": len(semester_rows),
                    "5개 학과 강좌": int(
                        semester_rows["is_target_department"].sum()
                    ),
                    "코드 연결": int(
                        semester_rows["master_match_method"].eq("code").sum()
                    ),
                    "명칭 연결": int(
                        semester_rows["master_match_method"].eq("name").sum()
                    ),
                    "미연결": int(
                        semester_rows["master_match_method"].eq("none").sum()
                    ),
                }
            )
        return pd.DataFrame(records)


@dataclass(frozen=True)
class CourseRecommendationCatalogResult:
    """실제 개설강좌를 앱의 고유 교과목 추천 후보로 변환한 결과."""

    candidate_data: pd.DataFrame
    source_offering_count: int
    description_linked_count: int
    master_unmatched_count: int
    department_count: int

    @property
    def course_count(self) -> int:
        """중복 없는 실제 교과목 추천 후보 수를 반환한다."""

        return len(self.candidate_data)

    @property
    def preview(self) -> pd.DataFrame:
        """개인정보가 없는 추천 후보 최대 8행을 반환한다."""

        columns = (
            "course_id",
            "course_name",
            "offering_departments",
            "grade_levels",
            "offered_semesters",
            "credit",
            "course_area",
            "class_method",
            "ncs_type",
            "master_match_method",
        )
        return self.candidate_data.loc[:, list(columns)].head(8).copy()


@dataclass(frozen=True)
class DefaultCourseOfferingFiles:
    """프로젝트 상위 폴더에서 발견한 실제 교과 관련 파일."""

    course_master: Path
    semester_one: Path
    semester_two: Path


def _clean_text(value: Any) -> str:
    """Excel 빈값과 정수형 식별자를 안정적인 문자열로 정규화한다."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _normalize_header(value: Any) -> str:
    """공백·괄호·기호·유니코드 차이를 제거해 열 이름을 비교한다."""

    normalized = unicodedata.normalize("NFKC", _clean_text(value)).lower()
    return re.sub(r"[\s_\-()/.]+", "", normalized)


def _normalize_name(value: Any) -> str:
    """교과목명 연결에 사용할 보수적인 정규화 값을 반환한다."""

    normalized = unicodedata.normalize("NFKC", _clean_text(value)).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", normalized)


def _fresh_excel_source(source: ExcelSource) -> str | Path | BytesIO:
    """같은 업로드 파일을 헤더와 허용 열 읽기에 각각 사용할 수 있게 한다."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"개설강좌 Excel을 찾을 수 없습니다: {path}")
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
    """경로 또는 업로드 파일에서 값이 아닌 파일명만 반환한다."""

    if explicit_name:
        return Path(explicit_name).name
    if isinstance(source, (str, Path)):
        return Path(source).name
    return Path(str(getattr(source, "name", "개설강좌.xlsx"))).name


def _alias_lookup() -> dict[str, str]:
    """개설강좌 원본 열 alias를 canonical 열 이름으로 펼친다."""

    return {
        _normalize_header(alias): canonical
        for canonical, aliases in OFFERING_COLUMN_ALIASES.items()
        for alias in aliases
    }


def load_course_offering_excel_safe(
    source: ExcelSource,
    *,
    academic_year: int,
    semester: int,
    source_name: str | None = None,
) -> SafeOfferingFrame:
    """헤더를 먼저 확인하고 공개된 강좌 열 값만 읽는다.

    Parameters:
        source: 개설강좌 Excel 경로 또는 업로드 bytes.
        academic_year: 파일이 나타내는 학년도.
        semester: 파일이 나타내는 1 또는 2학기.
        source_name: 업로드 화면에 표시할 선택적 파일명.

    Returns:
        허용 열 DataFrame과 개인정보 열 차단 현황.

    Assumptions:
        학년도와 학기는 파일명에서 추측하지 않고 호출자가 명시한다.
    """

    if int(academic_year) < 2000 or int(academic_year) > 2100:
        raise ValueError("개설강좌 학년도는 2000~2100 범위여야 합니다.")
    if int(semester) not in {1, 2}:
        raise ValueError("개설강좌 학기는 1 또는 2여야 합니다.")

    headers_frame = pd.read_excel(_fresh_excel_source(source), nrows=0)
    headers = tuple(_clean_text(column) for column in headers_frame.columns)
    blocked_columns = detect_blocked_columns(headers)
    blocked_normalized = {_normalize_header(column) for column in blocked_columns}
    aliases = _alias_lookup()
    selected_indexes: list[int] = []
    selected_columns: list[str] = []
    for index, header in enumerate(headers):
        normalized = _normalize_header(header)
        canonical = aliases.get(normalized)
        if (
            canonical
            and normalized not in blocked_normalized
            and canonical not in selected_columns
        ):
            selected_indexes.append(index)
            selected_columns.append(canonical)

    missing = REQUIRED_OFFERING_COLUMNS.difference(selected_columns)
    if missing:
        raise ValueError(
            "개설강좌 Excel에 필수 공개 열이 없습니다: " + ", ".join(sorted(missing))
        )
    values = pd.read_excel(
        _fresh_excel_source(source),
        usecols=selected_indexes,
        dtype=object,
    )
    values.columns = selected_columns
    values = values.dropna(how="all").reset_index(drop=True)
    return SafeOfferingFrame(
        frame=values,
        source_name=_source_name(source, source_name),
        academic_year=int(academic_year),
        semester=int(semester),
        source_row_count=len(values),
        blocked_columns=tuple(blocked_columns),
        ignored_column_count=max(
            len(headers) - len(selected_columns) - len(blocked_columns), 0
        ),
    )


def _canonical_department_name(value: Any) -> str:
    """강좌 파일의 코드 접미사와 5개 학과 과거 명칭을 정규화한다."""

    name = re.sub(r"\s*\[[^]]+\]\s*$", "", _clean_text(value)).strip()
    for canonical, aliases in TARGET_DEPARTMENT_ALIASES.items():
        if name in aliases:
            return canonical
    return name


def _flag(value: Any) -> bool | None:
    """0/1·예/아니오 계열 값을 삼상 boolean으로 변환한다."""

    normalized = _normalize_header(value)
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _number(value: Any) -> float | None:
    """숫자로 변환 가능한 공개 강좌 값을 반환한다."""

    text = _clean_text(value).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _number_or_none(values: pd.Series) -> int | float | None:
    """분반 전체에서 일관된 숫자 하나만 과목 수준 값으로 사용한다."""

    numbers = sorted({_number(value) for value in values if _number(value) is not None})
    if len(numbers) != 1:
        return None
    value = numbers[0]
    return int(value) if value.is_integer() else value


def _joined(values: pd.Series) -> str:
    """분반별 중복을 제거한 공개 분류값을 한 문자열로 결합한다."""

    unique = sorted({_clean_text(value) for value in values if _clean_text(value)})
    return "|".join(unique)


def _availability(group: pd.DataFrame) -> str:
    """분반 중 하나라도 실제 개설이면 과목을 추천 가능 상태로 표시한다."""

    statuses: list[str] = []
    for status, cancelled in zip(
        group["offering_status"], group["cancellation_status"]
    ):
        status_text = _normalize_header(status)
        cancelled_text = _normalize_header(cancelled)
        if cancelled_text in CANCELLED_STATUS_VALUES:
            statuses.append("unavailable")
        elif status_text in ACTIVE_STATUS_VALUES:
            statuses.append("available")
        elif status_text:
            statuses.append("unavailable")
        else:
            statuses.append("unknown")
    if "available" in statuses:
        return "available"
    if "unknown" in statuses:
        return "unknown"
    return "unavailable"


def _cross_department_status(values: pd.Series) -> str:
    """타학과 제외 플래그를 수강 허용·제한·미확인으로 변환한다."""

    flags = [_flag(value) for value in values]
    known = [value for value in flags if value is not None]
    if False in known:
        return "allowed"
    if known and all(known):
        return "restricted"
    return "unknown"


def _online_status(values: pd.Series) -> str:
    """분반별 온라인 여부를 온라인·대면·미확인으로 합친다."""

    flags = [_flag(value) for value in values]
    known = [value for value in flags if value is not None]
    if True in known:
        return "online"
    if known and not any(known):
        return "offline"
    return "unknown"


def _prepare_sections(source: SafeOfferingFrame) -> tuple[pd.DataFrame, int]:
    """한 학기 강좌 행에서 필수값이 없는 행을 제외하고 정규화한다."""

    working = source.frame.copy()
    for column in OFFERING_COLUMN_ALIASES:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].map(_clean_text)
    working["department_name"] = working["department_name"].map(
        _canonical_department_name
    )
    working["academic_year"] = source.academic_year
    working["semester"] = source.semester
    required = ["department_name", "course_id", "course_name"]
    empty = working[required].eq("").all(axis=1)
    missing = working[required].eq("").any(axis=1)
    excluded_count = int((~empty & missing).sum() + empty.sum())
    return working.loc[~empty & ~missing].reset_index(drop=True), excluded_count


def _consolidate_sections(sections: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """동일 학기·과목·개설학과의 여러 분반을 한 강좌로 통합한다."""

    records: list[dict[str, Any]] = []
    conflict_count = 0
    group_columns = [
        "academic_year",
        "semester",
        "course_id",
        "department_code",
        "department_name",
    ]
    targets = set(configured_department_names())
    for key, group in sections.groupby(group_columns, dropna=False, sort=True):
        academic_year, semester, course_id, department_code, department_name = key
        numeric_fields = {
            column: _number_or_none(group[column])
            for column in (
                "credit",
                "lecture_hours",
                "practice_hours",
                "total_hours",
            )
        }
        for column in numeric_fields:
            values = {
                _number(value)
                for value in group[column]
                if _number(value) is not None
            }
            if len(values) > 1:
                conflict_count += 1
        enrolled = [
            value
            for value in (_number(item) for item in group["enrolled_count"])
            if value is not None
        ]
        capacity = [
            value
            for value in (_number(item) for item in group["capacity"])
            if value is not None
        ]
        records.append(
            {
                "academic_year": int(academic_year),
                "semester": int(semester),
                "course_id": _clean_text(course_id),
                "course_name": _clean_text(group["course_name"].iloc[0]),
                "department_code": _clean_text(department_code),
                "department_name": _clean_text(department_name),
                "is_target_department": _clean_text(department_name) in targets,
                "grade_levels": _joined(group["grade_level"]),
                "credit": numeric_fields["credit"],
                "section_count": len(group),
                "sections": _joined(group["section"]),
                "course_area": _joined(group["course_area"]),
                "class_method": _joined(group["class_method"]),
                "course_type": _joined(group["course_type"]),
                "ncs_type": _joined(group["ncs_type"]),
                "lecture_hours": numeric_fields["lecture_hours"],
                "practice_hours": numeric_fields["practice_hours"],
                "total_hours": numeric_fields["total_hours"],
                "availability_status": _availability(group),
                "cross_department_status": _cross_department_status(
                    group["cross_department_excluded"]
                ),
                "online_status": _online_status(group["online"]),
                "enrolled_count": sum(enrolled) if enrolled else None,
                "capacity": sum(capacity) if capacity else None,
                "curriculum_year": _joined(group["curriculum_year"]),
                "master_match_method": "none",
                "linked_master_course_id": "",
                "master_description": "",
            }
        )
    return (
        pd.DataFrame(records, columns=COURSE_OFFERING_OUTPUT_COLUMNS),
        conflict_count,
    )


def _connect_course_master(
    offerings: pd.DataFrame,
    course_master: pd.DataFrame,
) -> pd.DataFrame:
    """교과목 코드 우선·고유 정규화 명칭 보조로 전체 master를 연결한다."""

    connected = offerings.copy()
    master_codes = {
        _clean_text(value)
        for value in course_master.get("course_id", pd.Series(dtype=object))
        if _clean_text(value)
    }
    name_to_ids: dict[str, set[str]] = {}
    master_descriptions: dict[str, str] = {}
    for course_id, course_name in zip(
        course_master.get("course_id", pd.Series(dtype=object)),
        course_master.get("course_name", pd.Series(dtype=object)),
    ):
        normalized_name = _normalize_name(course_name)
        cleaned_id = _clean_text(course_id)
        if normalized_name and cleaned_id:
            name_to_ids.setdefault(normalized_name, set()).add(cleaned_id)
    if "description" in course_master.columns:
        for course_id, description in zip(
            course_master["course_id"], course_master["description"]
        ):
            cleaned_id = _clean_text(course_id)
            cleaned_description = _clean_text(description)
            if cleaned_id and cleaned_description:
                master_descriptions[cleaned_id] = cleaned_description

    for index, row in connected.iterrows():
        course_id = _clean_text(row["course_id"])
        if course_id in master_codes:
            connected.at[index, "master_match_method"] = "code"
            connected.at[index, "linked_master_course_id"] = course_id
            connected.at[index, "master_description"] = master_descriptions.get(
                course_id, ""
            )
            continue
        matched_ids = name_to_ids.get(_normalize_name(row["course_name"]), set())
        if len(matched_ids) == 1:
            matched_id = next(iter(matched_ids))
            connected.at[index, "master_match_method"] = "name"
            connected.at[index, "linked_master_course_id"] = matched_id
            connected.at[index, "master_description"] = master_descriptions.get(
                matched_id, ""
            )
    return connected


def run_course_offering_dry_run(
    course_master_source: ExcelSource,
    semester_one_source: ExcelSource,
    semester_two_source: ExcelSource,
    *,
    academic_year: int = 2026,
    course_master_name: str | None = None,
    semester_one_name: str | None = None,
    semester_two_name: str | None = None,
) -> CourseOfferingDryRunResult:
    """전 학과 실제 개설강좌를 통합하고 전체 교과목과 연결한다.

    Parameters:
        course_master_source: 전체 교과목 Excel.
        semester_one_source: 해당 학년도 1학기 개설강좌 Excel.
        semester_two_source: 해당 학년도 2학기 개설강좌 Excel.
        academic_year: 두 강좌 파일이 나타내는 학년도.
        *_name: 업로드 화면에 표시할 선택적 파일명.

    Returns:
        분반 통합 후보, 학기별 요약과 연결 경고.

    Assumptions:
        학생 모집단은 5개 학과지만 강좌 후보는 전 학과를 보존한다.
        이 함수는 CSV·SQLite·설정 파일을 생성하거나 수정하지 않는다.
    """

    sources = (
        load_course_offering_excel_safe(
            semester_one_source,
            academic_year=academic_year,
            semester=1,
            source_name=semester_one_name,
        ),
        load_course_offering_excel_safe(
            semester_two_source,
            academic_year=academic_year,
            semester=2,
            source_name=semester_two_name,
        ),
    )
    course_master = load_course_excel_safe(course_master_source).frame
    return build_course_offering_dry_run(course_master, sources)


def build_course_offering_dry_run(
    course_master: pd.DataFrame,
    sources: Sequence[SafeOfferingFrame],
) -> CourseOfferingDryRunResult:
    """이미 안전하게 읽은 강좌와 교과 master로 연결 보고서를 만든다.

    Parameters:
        course_master: 허용 열로 정규화된 전체 교과목 DataFrame.
        sources: 학년도·학기가 명시된 개설강좌 DataFrame 묶음.

    Returns:
        분반 통합 후보, 학기별 요약과 연결 경고.

    Assumptions:
        원본 행을 수정하거나 저장하지 않고 전 학과 강좌를 보존한다.
    """

    if not sources:
        raise ValueError("점검할 개설강좌가 필요합니다.")
    master_missing = {"course_id", "course_name"}.difference(course_master.columns)
    if master_missing:
        raise ValueError(
            "전체 교과목에 연결 필수 열이 없습니다: "
            + ", ".join(sorted(master_missing))
        )
    section_frames: list[pd.DataFrame] = []
    excluded_required_count = 0
    blocked_columns: list[str] = []
    for source in sources:
        sections, excluded = _prepare_sections(source)
        section_frames.append(sections)
        excluded_required_count += excluded
        blocked_columns.extend(source.blocked_columns)
    all_sections = pd.concat(section_frames, ignore_index=True)
    consolidated, conflict_count = _consolidate_sections(all_sections)
    connected = _connect_course_master(consolidated, course_master)
    connected = connected.sort_values(
        ["academic_year", "semester", "department_name", "course_name", "course_id"],
        kind="stable",
    ).reset_index(drop=True)

    code_match_count = int(connected["master_match_method"].eq("code").sum())
    name_match_count = int(connected["master_match_method"].eq("name").sum())
    unmatched_count = int(connected["master_match_method"].eq("none").sum())
    target_count = int(connected["is_target_department"].sum())
    unknown_availability = int(
        connected["availability_status"].eq("unknown").sum()
    )
    issues: list[CourseOfferingImportIssue] = []
    if blocked_columns:
        issues.append(
            CourseOfferingImportIssue(
                "warning",
                "pii_columns_blocked",
                f"개인정보 가능 열 {len(set(blocked_columns))}개는 값 자체를 읽지 않았습니다.",
            )
        )
    if excluded_required_count:
        issues.append(
            CourseOfferingImportIssue(
                "warning",
                "required_rows_excluded",
                f"교과목 코드·명·개설학과가 비어 있는 행 {excluded_required_count}개를 제외했습니다.",
            )
        )
    if conflict_count:
        issues.append(
            CourseOfferingImportIssue(
                "warning",
                "section_value_conflicts",
                f"분반 사이 학점·이론·실습 시간 값이 다른 항목 {conflict_count}건은 미확인으로 남겼습니다.",
            )
        )
    if unknown_availability:
        issues.append(
            CourseOfferingImportIssue(
                "warning",
                "availability_unknown",
                f"개설 상태를 확인할 수 없는 통합 강좌 {unknown_availability}개가 있습니다.",
            )
        )
    if unmatched_count:
        issues.append(
            CourseOfferingImportIssue(
                "warning",
                "master_courses_unmatched",
                f"전체 교과목과 연결되지 않은 실제 개설강좌 {unmatched_count}개도 별도 후보로 보존했습니다.",
            )
        )
    issues.extend(
        (
            CourseOfferingImportIssue(
                "info",
                "all_departments_preserved",
                f"학생 대상은 5개 학과지만 전 학과 통합 강좌 {len(connected):,}개를 보존했습니다.",
            ),
            CourseOfferingImportIssue(
                "info",
                "target_departments_identified",
                f"5개 대상 학과에 연결되는 학기별 강좌 {target_count:,}개를 확인했습니다.",
            ),
            CourseOfferingImportIssue(
                "info",
                "sections_consolidated",
                f"유효 분반 {len(all_sections):,}행을 학기·과목·학과 기준 {len(connected):,}개 강좌로 통합했습니다.",
            ),
        )
    )
    return CourseOfferingDryRunResult(
        candidate_data=connected.loc[:, list(COURSE_OFFERING_OUTPUT_COLUMNS)].copy(),
        source_files=sources,
        issues=tuple(issues),
        valid_section_count=len(all_sections),
        excluded_required_count=excluded_required_count,
        consolidated_section_count=len(all_sections) - len(connected),
        master_code_match_count=code_match_count,
        master_name_match_count=name_match_count,
        master_unmatched_count=unmatched_count,
        target_department_course_count=target_count,
    )


def _split_joined_values(values: pd.Series) -> tuple[str, ...]:
    """파이프로 합쳐진 실제 원본 값을 다시 중복 없는 목록으로 만든다."""

    return tuple(
        sorted(
            {
                item.strip()
                for value in values
                for item in _clean_text(value).split("|")
                if item.strip()
            }
        )
    )


def _source_backed_description(
    group: pd.DataFrame,
    departments: Sequence[str],
) -> str:
    """새 의미를 만들지 않고 원본 설명 또는 개설 필드만으로 설명을 만든다."""

    descriptions = [
        _clean_text(value)
        for value in group["master_description"]
        if _clean_text(value)
    ]
    if descriptions:
        return max(descriptions, key=len)
    details: list[str] = []
    if departments:
        details.append("개설학과 " + ", ".join(departments))
    for label, column in (
        ("영역", "course_area"),
        ("교과목 구분", "course_type"),
        ("수업방식", "class_method"),
        ("NCS 구분", "ncs_type"),
    ):
        values = _split_joined_values(group[column])
        if values:
            details.append(f"{label} {', '.join(values)}")
    return " · ".join(details)


def build_course_recommendation_catalog(
    report: CourseOfferingDryRunResult,
) -> CourseRecommendationCatalogResult:
    """전 학과 개설강좌를 중복 없는 실제 교과목 추천 후보로 변환한다.

    Parameters:
        report: 두 학기 개설강좌와 전체 교과목 연결 dry-run 결과.

    Returns:
        공식 교과목코드를 고유 ID로 사용하는 추천 후보와 품질 집계.

    Assumptions:
        과목명·개설학과·교과구분·NCS·학년·학점·수업방식 등 원본에
        존재하는 값만 사용하며, 역량·직무·관심 태그는 생성하지 않는다.
    """

    if not report.ready:
        raise ValueError("개설강좌 dry-run 오류를 먼저 해결해야 합니다.")
    available = report.candidate_data[
        report.candidate_data["availability_status"].eq("available")
    ].copy()
    if available.empty:
        raise ValueError("추천 후보로 사용할 실제 개설강좌가 없습니다.")

    target_order = {
        name: index for index, name in enumerate(configured_department_names())
    }
    records: list[dict[str, Any]] = []
    for course_id, group in available.groupby("course_id", sort=True):
        names = _split_joined_values(group["course_name"])
        credits = sorted(
            {
                float(value)
                for value in group["credit"]
                if value is not None and not pd.isna(value)
            }
        )
        if len(names) != 1 or len(credits) != 1:
            raise ValueError(
                f"교과목 {course_id}의 명칭 또는 학점이 개설강좌 사이에서 다릅니다."
            )
        departments = _split_joined_values(group["department_name"])
        department_codes = _split_joined_values(group["department_code"])
        grade_levels = tuple(
            sorted(
                {
                    int(item)
                    for value in group["grade_levels"]
                    for item in _clean_text(value).split("|")
                    if item.isdigit() and 1 <= int(item) <= 6
                }
            )
        )
        if not departments or not department_codes or not grade_levels:
            raise ValueError(f"교과목 {course_id}의 개설학과 또는 학년이 없습니다.")
        semesters = tuple(sorted({int(value) for value in group["semester"]}))
        academic_years = tuple(
            sorted({int(value) for value in group["academic_year"]})
        )
        primary_department = min(
            departments,
            key=lambda name: (target_order.get(name, len(target_order)), name),
        )
        match_methods = set(group["master_match_method"].astype(str))
        match_method = (
            "code"
            if "code" in match_methods
            else "name"
            if "name" in match_methods
            else "none"
        )
        linked_ids = _split_joined_values(group["linked_master_course_id"])
        cross_statuses = set(group["cross_department_status"].astype(str))
        cross_status = (
            "allowed"
            if "allowed" in cross_statuses
            else "unknown"
            if "unknown" in cross_statuses
            else "restricted"
        )
        online_statuses = set(group["online_status"].astype(str))
        online_status = (
            "online"
            if "online" in online_statuses
            else "offline"
            if online_statuses == {"offline"}
            else "mixed_or_unknown"
        )
        record = {
            "course_id": _clean_text(course_id),
            "course_name": names[0],
            "department": primary_department,
            "description": _source_backed_description(group, departments),
            "learning_objectives": "",
            "competencies": "",
            "related_jobs": "",
            "related_interests": "",
            "grade_level": min(grade_levels),
            "semester": min(semesters),
            "difficulty": "원본 미제공",
            "prerequisites": "",
            "credit": int(credits[0]),
            "is_available": True,
            "catalog_source": ACTUAL_COURSE_CATALOG_SOURCE,
            "source_course_id": _clean_text(course_id),
            "academic_years": "|".join(map(str, academic_years)),
            "offered_semesters": "|".join(map(str, semesters)),
            "offering_departments": "|".join(departments),
            "offering_department_codes": "|".join(department_codes),
            "grade_levels": "|".join(map(str, grade_levels)),
            "course_area": "|".join(_split_joined_values(group["course_area"])),
            "class_method": "|".join(
                _split_joined_values(group["class_method"])
            ),
            "course_type": "|".join(_split_joined_values(group["course_type"])),
            "ncs_type": "|".join(_split_joined_values(group["ncs_type"])),
            "cross_department_status": cross_status,
            "online_status": online_status,
            "recommendation_basis": (
                "교과목명·개설학과·교과구분·NCS·학년·학점·수업방식"
            ),
            "prerequisite_status": "원본 미제공",
            "completion_data_status": "synthetic 학생 이수정보와 미연결",
            "registration_check_required": True,
            "master_match_method": match_method,
            "linked_master_course_id": "|".join(linked_ids),
        }
        records.append(record)

    candidates = pd.DataFrame(records, columns=ACTUAL_COURSE_CATALOG_COLUMNS)
    candidates = candidates.sort_values(
        ["department", "course_name", "course_id"], kind="stable"
    ).reset_index(drop=True)
    all_departments = {
        item
        for value in candidates["offering_departments"]
        for item in _clean_text(value).split("|")
        if item
    }
    return CourseRecommendationCatalogResult(
        candidate_data=candidates,
        source_offering_count=len(available),
        description_linked_count=int(
            available.groupby("course_id")["master_description"]
            .apply(lambda values: any(_clean_text(value) for value in values))
            .sum()
        ),
        master_unmatched_count=int(
            candidates["master_match_method"].eq("none").sum()
        ),
        department_count=len(all_departments),
    )


def find_default_course_offering_files(root: Path) -> DefaultCourseOfferingFiles:
    """프로젝트 상위 폴더에서 세 교과 Excel을 유니코드 안전하게 찾는다."""

    candidates = [
        path
        for path in Path(root).glob("*.xlsx")
        if not path.name.startswith("~$")
    ]

    def select(predicate, label: str) -> Path:
        matches = [path for path in candidates if predicate(_normalize_header(path.name))]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{label} Excel은 정확히 1개 필요합니다. 현재 {len(matches)}개입니다."
            )
        return matches[0]

    return DefaultCourseOfferingFiles(
        course_master=select(
            lambda name: name.startswith("2") and "전체교과목" in name,
            "전체 교과목",
        ),
        semester_one=select(
            lambda name: "26년1학기개설강좌" in name,
            "2026년 1학기 개설강좌",
        ),
        semester_two=select(
            lambda name: "26년2학기개설강좌" in name,
            "2026년 2학기 개설강좌",
        ),
    )


def _print_report(report: CourseOfferingDryRunResult) -> None:
    """CLI에 개인정보 없는 연결 집계와 점검 메시지만 출력한다."""

    status = "점검 완료" if report.ready else "점검 필요"
    print(f"2026 개설강좌 dry-run: {status} (어떤 파일도 저장하지 않았습니다.)")
    print(report.summary_frame().to_string(index=False))
    for issue in report.issues:
        print(f"[{issue.severity.upper()}] {issue.message}")
    if report.ready:
        catalog = build_course_recommendation_catalog(report)
        print(
            "추천 후보 변환: "
            f"고유 교과목 {catalog.course_count:,}개, "
            f"개설학과 {catalog.department_count:,}개, "
            f"원본 강의개요 연결 {catalog.description_linked_count:,}개, "
            f"교과 master 미연결 {catalog.master_unmatched_count:,}개"
        )


def main() -> None:
    """두 학기 개설강좌 연결 dry-run CLI 진입점."""

    parser = argparse.ArgumentParser(
        description="전 학과 실제 개설강좌를 분반 통합하고 전체 교과목과 연결합니다."
    )
    parser.add_argument("--courses", required=True, help="전체 교과목 Excel")
    parser.add_argument("--semester-1", required=True, help="1학기 개설강좌 Excel")
    parser.add_argument("--semester-2", required=True, help="2학기 개설강좌 Excel")
    parser.add_argument("--year", type=int, default=2026, help="개설 학년도")
    args = parser.parse_args()
    _print_report(
        run_course_offering_dry_run(
            args.courses,
            args.semester_1,
            args.semester_2,
            academic_year=args.year,
        )
    )


if __name__ == "__main__":
    main()
