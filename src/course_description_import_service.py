"""과거 강좌설명을 현재 실제 개설강좌 추천 후보에 안전하게 보강한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re
from typing import Any, BinaryIO, Mapping
import unicodedata

import pandas as pd

from src.master_import_service import ExcelSource, detect_blocked_columns
from src.support_program_importer import EMAIL_PATTERN, PHONE_PATTERN


DESCRIPTION_SOURCE = "course_lecture_basic_information"
MIN_DESCRIPTION_LENGTH = 20
DESCRIPTION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "description_source",
    "description_match_method",
    "description_source_terms",
)
DESCRIPTION_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "lecture_id": ("강좌구분번호", "강좌번호"),
    "academic_year": ("개설연도", "학년도"),
    "semester": ("개설학기", "학기"),
    "course_name": ("과목명", "교과목명"),
    "lecture_name": ("강좌명",),
    "description": ("강좌설명", "강의설명", "강의개요"),
    "department_code": ("학과코드", "개설학과코드"),
    "department_name": ("학과명", "개설학과"),
    "grade_level": ("개설학년", "학년"),
    "course_type": ("강좌유형", "교과목구분"),
    "online": ("원격강좌여부", "온라인강좌여부"),
    "ncs_code": ("NCS코드",),
    "hours": ("수업시간수", "강의시간수"),
    "credit": ("학점",),
}
REQUIRED_DESCRIPTION_COLUMNS = frozenset(
    {"academic_year", "semester", "course_name", "description"}
)
DESCRIPTION_FRAME_COLUMNS: tuple[str, ...] = tuple(
    DESCRIPTION_COLUMN_ALIASES
)


@dataclass(frozen=True)
class SafeCourseDescriptionFrame:
    """개인정보 가능 열을 읽지 않은 강좌설명 원본."""

    frame: pd.DataFrame
    source_name: str
    source_row_count: int
    blocked_columns: tuple[str, ...]
    ignored_column_count: int


@dataclass(frozen=True)
class CourseDescriptionIssue:
    """강좌설명 보강 dry-run에서 발견한 안내 또는 경고."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class CourseDescriptionEnrichmentReport:
    """현재 개설강좌 후보에 과거 강좌설명을 연결한 무저장 결과."""

    candidate_data: pd.DataFrame
    source: SafeCourseDescriptionFrame
    source_catalog_signature: str
    term_summary: pd.DataFrame
    usable_description_row_count: int
    enriched_course_count: int
    department_match_count: int
    unique_name_match_count: int
    ambiguous_course_count: int
    unmatched_course_count: int
    short_description_row_count: int
    contact_row_count: int
    issues: tuple[CourseDescriptionIssue, ...]

    @property
    def ready(self) -> bool:
        """오류가 없고 현재 추천 후보 수가 보존됐는지 반환한다."""

        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def course_count(self) -> int:
        """보강 후에도 유지된 실제 개설 교과목 수를 반환한다."""

        return len(self.candidate_data)

    @property
    def preview(self) -> pd.DataFrame:
        """설명 보강 상태를 개인정보 없이 최대 8행 반환한다."""

        columns = (
            "course_id",
            "course_name",
            "offering_departments",
            "description_match_method",
            "description_source_terms",
        )
        return self.candidate_data.loc[:, list(columns)].head(8).copy()

    def applies_to(self, course_catalog: pd.DataFrame) -> bool:
        """이 보고서가 현재 화면의 추천 후보에서 생성됐는지 확인한다."""

        return self.source_catalog_signature == course_catalog_signature(
            course_catalog
        )


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
    """과목명과 학과명을 보수적으로 연결할 정규화 값을 반환한다."""

    normalized = unicodedata.normalize("NFKC", _clean_text(value)).lower()
    normalized = re.sub(r"\s*\[[^]]+\]\s*$", "", normalized)
    normalized = re.sub(r"\s*\(20\d{2}\)\s*$", "", normalized)
    return re.sub(r"[^0-9a-z가-힣]", "", normalized)


def _normalize_description(value: Any) -> str:
    """동일 설명 판정에 사용할 공백·유니코드 정규화 값을 반환한다."""

    return unicodedata.normalize("NFKC", _clean_text(value)).lower()


def _fresh_excel_source(source: ExcelSource) -> str | Path | BytesIO:
    """헤더와 허용 열을 별도로 읽을 수 있는 Excel 입력을 반환한다."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"강좌기본정보 Excel을 찾을 수 없습니다: {path}")
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
    """업로드 경로나 binary source에서 안전한 파일명만 반환한다."""

    if explicit_name:
        return Path(explicit_name).name
    if isinstance(source, (str, Path)):
        return Path(source).name
    return Path(str(getattr(source, "name", "강좌기본정보.xlsx"))).name


def _alias_lookup() -> dict[str, str]:
    """강좌기본정보 열 alias를 canonical 열 이름으로 펼친다."""

    return {
        _normalize_header(alias): canonical
        for canonical, aliases in DESCRIPTION_COLUMN_ALIASES.items()
        for alias in aliases
    }


def load_course_description_excel_safe(
    source: ExcelSource,
    *,
    source_name: str | None = None,
) -> SafeCourseDescriptionFrame:
    """헤더를 먼저 검사하고 허용된 강좌설명 열 값만 읽는다.

    Parameters:
        source: 강좌기본정보 Excel 경로 또는 업로드 bytes.
        source_name: 업로드 화면에 표시할 선택적 파일명.

    Returns:
        공개 강좌 열만 포함한 DataFrame과 차단 현황.

    Assumptions:
        담당자·성명·사번·전화·이메일 등 개인정보 가능 열은 값 자체를 읽지 않는다.
    """

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

    missing = REQUIRED_DESCRIPTION_COLUMNS.difference(selected_columns)
    if missing:
        raise ValueError(
            "강좌기본정보 Excel에 필수 공개 열이 없습니다: "
            + ", ".join(sorted(missing))
        )
    values = pd.read_excel(
        _fresh_excel_source(source),
        usecols=selected_indexes,
        dtype=object,
    )
    values.columns = selected_columns
    values = values.dropna(how="all").reset_index(drop=True)
    for column in DESCRIPTION_FRAME_COLUMNS:
        if column not in values.columns:
            values[column] = ""
    return SafeCourseDescriptionFrame(
        frame=values.loc[:, list(DESCRIPTION_FRAME_COLUMNS)].copy(),
        source_name=_source_name(source, source_name),
        source_row_count=len(values),
        blocked_columns=tuple(blocked_columns),
        ignored_column_count=max(
            len(headers) - len(selected_columns) - len(blocked_columns), 0
        ),
    )


def _academic_year(value: Any) -> str:
    """연도 원문에서 4자리 학년도를 반환한다."""

    digits = re.sub(r"\D", "", _clean_text(value))
    if len(digits) >= 4:
        year = int(digits[-4:])
    elif len(digits) == 2:
        year = 2000 + int(digits)
    else:
        return ""
    return str(year) if 2000 <= year <= 2100 else ""


def _semester(value: Any) -> str:
    """정규학기와 계절학기 표기를 보존해 정규화한다."""

    text = _clean_text(value)
    normalized = _normalize_header(text)
    if normalized in {"1", "1학기", "01", "01학기"}:
        return "1"
    if normalized in {"2", "2학기", "02", "02학기"}:
        return "2"
    return text


def _term_label(year: Any, semester: Any) -> str:
    """출처 추적용 학년도-학기 문자열을 반환한다."""

    clean_year = _academic_year(year)
    clean_semester = _semester(semester)
    if not clean_year or not clean_semester:
        return ""
    return f"{clean_year}-{clean_semester}"


def _has_contact(value: Any) -> bool:
    """설명 원문에 이메일 또는 전화번호 패턴이 있는지 확인한다."""

    text = _clean_text(value)
    return bool(EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text))


def _prepare_description_rows(
    source: SafeCourseDescriptionFrame,
) -> tuple[pd.DataFrame, int, int, int]:
    """설명 연결에 사용할 유효 행과 제외 사유별 개수를 반환한다."""

    working = source.frame.copy()
    for column in DESCRIPTION_FRAME_COLUMNS:
        working[column] = working[column].map(_clean_text)
    working["academic_year"] = working["academic_year"].map(_academic_year)
    working["semester"] = working["semester"].map(_semester)
    working["term"] = [
        _term_label(year, semester)
        for year, semester in zip(
            working["academic_year"], working["semester"]
        )
    ]
    working["course_name_key"] = working["course_name"].map(_normalize_name)
    working["department_key"] = working["department_name"].map(_normalize_name)
    missing = (
        working["term"].eq("")
        | working["course_name_key"].eq("")
        | working["description"].eq("")
    )
    short = ~missing & working["description"].str.len().lt(
        MIN_DESCRIPTION_LENGTH
    )
    contact = ~missing & ~short & working["description"].map(_has_contact)
    valid = working.loc[~missing & ~short & ~contact].copy()
    valid["description_key"] = valid["description"].map(
        _normalize_description
    )
    return valid, int(missing.sum()), int(short.sum()), int(contact.sum())


def _description_choice(group: pd.DataFrame) -> tuple[str, str] | None:
    """한 연결 그룹의 설명이 하나로 일치하면 설명과 출처 학기를 반환한다."""

    descriptions: dict[str, str] = {}
    for description_key, description in zip(
        group["description_key"], group["description"]
    ):
        current = descriptions.get(description_key, "")
        if len(description) > len(current):
            descriptions[description_key] = description
    if len(descriptions) != 1:
        return None
    terms = tuple(sorted({term for term in group["term"] if term}))
    return next(iter(descriptions.values())), "|".join(terms)


def _build_description_indexes(
    rows: pd.DataFrame,
) -> tuple[
    Mapping[tuple[str, str], tuple[str, str] | None],
    Mapping[str, tuple[str, str] | None],
]:
    """과목명+학과 우선, 고유 과목명 보조 연결 인덱스를 만든다."""

    by_department = {
        key: _description_choice(group)
        for key, group in rows.groupby(
            ["course_name_key", "department_key"], sort=False
        )
        if key[0] and key[1]
    }
    by_name = {
        key: _description_choice(group)
        for key, group in rows.groupby("course_name_key", sort=False)
        if key
    }
    return by_department, by_name


def course_catalog_signature(course_catalog: pd.DataFrame) -> str:
    """설명 외 현재 개설강좌 식별값의 안정적인 서명을 반환한다."""

    required = {"course_id", "course_name"}
    missing = required.difference(course_catalog.columns)
    if missing:
        raise ValueError(
            "추천 후보에 서명 필수 열이 없습니다: " + ", ".join(sorted(missing))
        )
    rows = sorted(
        (
            _clean_text(course_id),
            _clean_text(course_name),
        )
        for course_id, course_name in zip(
            course_catalog["course_id"], course_catalog["course_name"]
        )
    )
    payload = "\n".join(f"{course_id}\t{course_name}" for course_id, course_name in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_course_description_enrichment(
    course_catalog: pd.DataFrame,
    source: SafeCourseDescriptionFrame,
) -> CourseDescriptionEnrichmentReport:
    """현재 개설 후보의 ID·개설정보를 유지하며 강좌설명만 보강한다.

    Parameters:
        course_catalog: 2026년 실제 개설강좌에서 만든 추천 후보.
        source: 허용 열만 읽은 강좌기본정보.

    Returns:
        설명 연결 상태와 보강된 후보를 담은 무저장 보고서.

    Assumptions:
        공식 교과목코드가 없으므로 과목명+학과가 같은 단일 설명만 자동 연결한다.
        학과 연결이 없을 때는 전체 원본에서 설명이 하나인 과목명만 보조 연결한다.
    """

    required_catalog = {
        "course_id",
        "course_name",
        "description",
        "offering_departments",
        "recommendation_basis",
    }
    missing = required_catalog.difference(course_catalog.columns)
    if missing:
        raise ValueError(
            "현재 개설강좌 후보에 필수 열이 없습니다: "
            + ", ".join(sorted(missing))
        )
    if course_catalog.empty:
        raise ValueError("설명을 보강할 현재 개설강좌 후보가 없습니다.")
    if course_catalog["course_id"].astype(str).duplicated().any():
        raise ValueError("현재 개설강좌 후보의 교과목코드는 중복될 수 없습니다.")

    valid, missing_rows, short_rows, contact_rows = _prepare_description_rows(
        source
    )
    if valid.empty:
        raise ValueError("연결 가능한 공개 강좌설명이 없습니다.")
    by_department, by_name = _build_description_indexes(valid)
    enriched = course_catalog.copy()
    enriched["description_source"] = "existing_course_catalog"
    enriched["description_match_method"] = "unmatched"
    enriched["description_source_terms"] = ""

    department_matches = 0
    name_matches = 0
    ambiguous = 0
    unmatched = 0
    for index, row in enriched.iterrows():
        course_name_key = _normalize_name(row["course_name"])
        departments = tuple(
            _normalize_name(value)
            for value in _clean_text(row["offering_departments"]).split("|")
            if _normalize_name(value)
        )
        department_choices = [
            by_department[(course_name_key, department)]
            for department in departments
            if (course_name_key, department) in by_department
        ]
        usable_department_choices = {
            choice for choice in department_choices if choice is not None
        }
        has_department_conflict = any(
            choice is None for choice in department_choices
        ) or len({choice[0] for choice in usable_department_choices}) > 1

        choice: tuple[str, str] | None = None
        method = "unmatched"
        if department_choices and not has_department_conflict:
            if usable_department_choices:
                descriptions = {item[0] for item in usable_department_choices}
                if len(descriptions) == 1:
                    description = next(iter(descriptions))
                    terms = "|".join(
                        sorted(
                            {
                                term
                                for _, source_terms in usable_department_choices
                                for term in source_terms.split("|")
                                if term
                            }
                        )
                    )
                    choice = description, terms
                    method = "course_name_department"
        elif course_name_key in by_name:
            choice = by_name[course_name_key]
            method = "unique_course_name" if choice is not None else "ambiguous"

        if choice is not None:
            description, terms = choice
            enriched.at[index, "description"] = description
            enriched.at[index, "description_source"] = DESCRIPTION_SOURCE
            enriched.at[index, "description_match_method"] = method
            enriched.at[index, "description_source_terms"] = terms
            basis = _clean_text(enriched.at[index, "recommendation_basis"])
            addition = "강좌기본정보 강좌설명"
            if addition not in basis:
                enriched.at[index, "recommendation_basis"] = (
                    f"{basis}·{addition}" if basis else addition
                )
            if method == "course_name_department":
                department_matches += 1
            else:
                name_matches += 1
        elif has_department_conflict or method == "ambiguous" or course_name_key in by_name:
            enriched.at[index, "description_match_method"] = "ambiguous"
            ambiguous += 1
        else:
            unmatched += 1

    term_summary = (
        valid.groupby(["academic_year", "semester"], dropna=False)
        .agg(
            source_rows=("course_name_key", "size"),
            unique_course_names=("course_name_key", "nunique"),
        )
        .reset_index()
    )
    term_summary["학기"] = (
        term_summary["academic_year"] + "-" + term_summary["semester"]
    )
    term_summary = term_summary.rename(
        columns={"source_rows": "사용 가능 설명", "unique_course_names": "고유 과목명"}
    ).loc[:, ["학기", "사용 가능 설명", "고유 과목명"]]

    issues: list[CourseDescriptionIssue] = []
    if source.blocked_columns:
        issues.append(
            CourseDescriptionIssue(
                "warning",
                "pii_columns_blocked",
                f"개인정보 가능 열 {len(source.blocked_columns)}개는 값 자체를 읽지 않았습니다.",
            )
        )
    if missing_rows:
        issues.append(
            CourseDescriptionIssue(
                "warning",
                "required_rows_excluded",
                f"학기·과목명·강좌설명이 비어 있는 {missing_rows}행을 제외했습니다.",
            )
        )
    if short_rows:
        issues.append(
            CourseDescriptionIssue(
                "warning",
                "short_descriptions_excluded",
                f"{MIN_DESCRIPTION_LENGTH}자 미만 강좌설명 {short_rows}행은 의미 검색에서 제외했습니다.",
            )
        )
    if contact_rows:
        issues.append(
            CourseDescriptionIssue(
                "warning",
                "contact_rows_excluded",
                f"연락처 패턴이 포함된 강좌설명 {contact_rows}행을 제외했습니다.",
            )
        )
    if ambiguous:
        issues.append(
            CourseDescriptionIssue(
                "warning",
                "ambiguous_descriptions_preserved",
                f"같은 과목명에 서로 다른 설명이 있는 {ambiguous}개 과목은 기존 설명을 유지했습니다.",
            )
        )
    issues.extend(
        (
            CourseDescriptionIssue(
                "info",
                "descriptions_enriched",
                f"현재 개설 교과목 {department_matches + name_matches:,}개에 검증된 강좌설명을 연결했습니다.",
            ),
            CourseDescriptionIssue(
                "info",
                "catalog_scope_preserved",
                f"현재 개설 교과목 {len(course_catalog):,}개의 ID·학기·개설학과 범위는 변경하지 않았습니다.",
            ),
        )
    )
    return CourseDescriptionEnrichmentReport(
        candidate_data=enriched.reset_index(drop=True),
        source=source,
        source_catalog_signature=course_catalog_signature(course_catalog),
        term_summary=term_summary.reset_index(drop=True),
        usable_description_row_count=len(valid),
        enriched_course_count=department_matches + name_matches,
        department_match_count=department_matches,
        unique_name_match_count=name_matches,
        ambiguous_course_count=ambiguous,
        unmatched_course_count=unmatched,
        short_description_row_count=short_rows,
        contact_row_count=contact_rows,
        issues=tuple(issues),
    )


def run_course_description_enrichment_dry_run(
    description_source: ExcelSource,
    course_catalog: pd.DataFrame,
    *,
    source_name: str | None = None,
) -> CourseDescriptionEnrichmentReport:
    """강좌기본정보를 안전하게 읽고 현재 추천 후보에 무저장 연결한다."""

    source = load_course_description_excel_safe(
        description_source,
        source_name=source_name,
    )
    return build_course_description_enrichment(course_catalog, source)


def find_default_course_description_file(root: Path) -> Path:
    """프로젝트 상위 폴더의 강좌기본정보 Excel 하나를 찾는다."""

    matches = [
        path
        for path in Path(root).glob("*.xlsx")
        if not path.name.startswith("~$")
        and "강좌기본정보" in _normalize_header(path.name)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "강좌기본정보 Excel은 정확히 1개 필요합니다. "
            f"현재 {len(matches)}개입니다."
        )
    return matches[0]


def _print_report(report: CourseDescriptionEnrichmentReport) -> None:
    """CLI에 개인정보 없는 설명 연결 집계만 출력한다."""

    print("강좌설명 보강 dry-run: 점검 완료 (어떤 파일도 저장하지 않았습니다.)")
    print(report.term_summary.to_string(index=False))
    print(
        "현재 개설 교과목 "
        f"{report.course_count:,}개 중 설명 연결 {report.enriched_course_count:,}개, "
        f"설명 충돌 {report.ambiguous_course_count:,}개, "
        f"미연결 {report.unmatched_course_count:,}개"
    )
    for issue in report.issues:
        print(f"[{issue.severity.upper()}] {issue.message}")


def main() -> None:
    """강좌설명 보강 dry-run CLI 진입점."""

    parser = argparse.ArgumentParser(
        description="과거 강좌설명을 현재 실제 개설강좌 추천 후보에 연결합니다."
    )
    parser.add_argument("--descriptions", required=True, help="강좌기본정보 Excel")
    parser.add_argument(
        "--catalog", required=True, help="현재 실제 개설강좌 추천 CSV"
    )
    args = parser.parse_args()
    course_catalog = pd.read_csv(args.catalog, keep_default_na=False)
    _print_report(
        run_course_description_enrichment_dry_run(
            args.descriptions,
            course_catalog,
        )
    )


if __name__ == "__main__":
    main()
