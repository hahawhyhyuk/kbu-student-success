"""검토된 실제 master만 백업 후 원자적으로 활성화한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.course_offering_import_service import (
    ACTUAL_COURSE_CATALOG_EXTRA_COLUMNS,
    ACTUAL_COURSE_CATALOG_SOURCE,
)
from src.department_catalog import configured_department_names
from src.master_import_service import (
    COURSE_OUTPUT_COLUMNS,
    DEPARTMENT_OUTPUT_COLUMNS,
)
from src.repositories.base import COURSE_COLUMNS, SUPPORT_PROGRAM_COLUMNS
from src.support_program_importer import (
    EMAIL_PATTERN,
    OUTPUT_COLUMNS as PROGRAM_OUTPUT_COLUMNS,
    PHONE_PATTERN,
    validate_imported_programs,
)
from src.utils import DATA_DIR, load_app_config


DEFAULT_MASTER_FILES = {
    "departments": "departments_master.csv",
    "courses": "courses_reviewed.csv",
    "programs": "support_programs_wings.csv",
}
DEFAULT_FALLBACK_FILES = {
    "courses": "courses.csv",
    "programs": "support_programs.csv",
}
MASTER_CONFIG_KEYS = {
    "departments": "department_master_file",
    "courses": "course_master_file",
    "programs": "support_program_master_file",
}
FALLBACK_CONFIG_KEYS = {
    "courses": "course_fallback_file",
    "programs": "support_program_fallback_file",
}
COURSE_REQUIRED_TEXT_COLUMNS = (
    "course_id",
    "course_name",
    "department",
    "description",
    "learning_objectives",
    "competencies",
    "related_jobs",
    "related_interests",
    "difficulty",
)


def _is_source_backed_course_catalog(courses: pd.DataFrame) -> bool:
    """실제 개설강좌에서 만든 추천 후보인지 명시적 provenance로 확인한다."""

    if "catalog_source" not in courses.columns or courses.empty:
        return False
    sources = {
        _clean_text(value) for value in courses["catalog_source"] if _clean_text(value)
    }
    return sources == {ACTUAL_COURSE_CATALOG_SOURCE}


def prepare_source_backed_course_catalog(courses: pd.DataFrame) -> pd.DataFrame:
    """실제 개설강좌 추천 후보를 추정 없이 정규화하고 검증한다.

    Parameters:
        courses: ``build_course_recommendation_catalog``가 만든 후보.

    Returns:
        실제 원본 provenance와 추가 공개 필드를 보존한 canonical 교과목.

    Assumptions:
        역량·직무·관심 태그가 비어 있어도 허용하지만 실제 개설·학년·학점과
        수강 확인 필요 상태는 반드시 명시되어야 한다.
    """

    required = set(COURSE_COLUMNS) | set(ACTUAL_COURSE_CATALOG_EXTRA_COLUMNS)
    missing = required.difference(courses.columns)
    if missing:
        raise ValueError(
            "실제 개설강좌 추천 후보 필수 열이 없습니다: "
            + ", ".join(sorted(missing))
        )
    working = courses.copy()
    text_columns = [
        column
        for column in working.columns
        if column not in {"is_available", "registration_check_required"}
    ]
    for column in text_columns:
        working[column] = working[column].map(_clean_text)
    required_text = (
        "course_id",
        "course_name",
        "department",
        "description",
        "catalog_source",
        "source_course_id",
        "academic_years",
        "offered_semesters",
        "offering_departments",
        "offering_department_codes",
        "grade_levels",
        "recommendation_basis",
        "prerequisite_status",
        "completion_data_status",
        "cross_department_status",
    )
    if working[list(required_text)].eq("").any(axis=1).any():
        raise ValueError("실제 개설강좌 추천 후보에 비어 있는 필수 원본 값이 있습니다.")
    if not working["catalog_source"].eq(ACTUAL_COURSE_CATALOG_SOURCE).all():
        raise ValueError("실제 개설강좌 추천 후보의 출처 표시를 확인해 주세요.")
    if not working["course_id"].eq(working["source_course_id"]).all():
        raise ValueError("추천 교과목 ID는 실제 원본 교과목코드와 같아야 합니다.")
    if working["course_id"].duplicated().any():
        raise ValueError("실제 개설강좌 추천 후보의 교과목코드는 중복될 수 없습니다.")
    numeric_values = {
        column: pd.to_numeric(working[column], errors="coerce")
        for column in ("grade_level", "semester", "credit")
    }
    numeric_ready = (
        numeric_values["grade_level"].between(1, 6)
        & numeric_values["semester"].isin({1, 2})
        & numeric_values["credit"].between(1, 9)
        & numeric_values["credit"].mod(1).eq(0)
    )
    if not numeric_ready.all():
        raise ValueError("실제 개설강좌 추천 후보의 학년·학기·학점을 확인해 주세요.")
    available = working["is_available"].map(_explicit_true)
    if not available.all():
        raise ValueError("실제 개설강좌 추천 후보에는 명시적으로 개설된 과목만 허용합니다.")
    registration_check = working["registration_check_required"].map(_explicit_true)
    if not registration_check.all():
        raise ValueError("실제 개설강좌는 최종 수강 가능 확인 필요 상태여야 합니다.")
    allowed_cross_statuses = {"allowed", "restricted", "unknown"}
    if not set(working["cross_department_status"]).issubset(
        allowed_cross_statuses
    ):
        raise ValueError("타과 수강 상태 값이 허용 범위를 벗어났습니다.")
    if _contains_contact(working):
        raise ValueError("실제 개설강좌 추천 후보에 연락처 패턴이 남아 있습니다.")
    working["grade_level"] = numeric_values["grade_level"].astype(int)
    working["semester"] = numeric_values["semester"].astype(int)
    working["credit"] = numeric_values["credit"].astype(int)
    working["is_available"] = True
    working["registration_check_required"] = True
    return working.reset_index(drop=True)


@dataclass(frozen=True)
class MasterActivationResult:
    """실제 master 한 종류의 활성화 결과."""

    master_type: str
    destination_path: Path
    row_count: int
    backup_path: Path | None
    activated_at: str


@dataclass(frozen=True)
class ActiveMasterStatus:
    """현재 앱이 실제 master 또는 fallback을 사용할 수 있는 상태."""

    master_type: str
    display_name: str
    selected_file: str
    row_count: int
    using_actual: bool
    using_fallback: bool
    message: str


def configured_master_file_name(
    master_type: str,
    config: Mapping[str, Any] | None = None,
) -> str:
    """설정에서 경로가 없는 안전한 master 파일명을 반환한다."""

    if master_type not in DEFAULT_MASTER_FILES:
        raise ValueError(f"지원하지 않는 master 유형입니다: {master_type}")
    active_config = config if config is not None else load_app_config()
    data_config = active_config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data 설정은 mapping이어야 합니다.")
    key = MASTER_CONFIG_KEYS[master_type]
    name = str(data_config.get(key, DEFAULT_MASTER_FILES[master_type])).strip()
    if not name or Path(name).name != name:
        raise ValueError(f"{master_type} master 설정은 파일명만 허용합니다.")
    return name


def configured_fallback_file_name(
    master_type: str,
    config: Mapping[str, Any] | None = None,
) -> str:
    """교과·비교과의 안전한 synthetic fallback 파일명을 반환한다."""

    if master_type not in DEFAULT_FALLBACK_FILES:
        raise ValueError(f"fallback이 없는 master 유형입니다: {master_type}")
    active_config = config if config is not None else load_app_config()
    data_config = active_config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data 설정은 mapping이어야 합니다.")
    key = FALLBACK_CONFIG_KEYS[master_type]
    name = str(data_config.get(key, DEFAULT_FALLBACK_FILES[master_type])).strip()
    if not name or Path(name).name != name:
        raise ValueError(f"{master_type} fallback 설정은 파일명만 허용합니다.")
    return name


def _clean_text(value: Any) -> str:
    """CSV 저장·검증에 사용할 문자열을 정규화한다."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def _explicit_true(value: Any) -> bool:
    """검토자가 명시한 활성 값만 True로 해석한다."""

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "개설",
        "개설중",
        "운영",
        "운영중",
        "활성",
        "사용",
    }


def _contains_contact(frame: pd.DataFrame) -> bool:
    """허용 master 값 전체에 연락처 패턴이 남았는지 확인한다."""

    serialized = " ".join(
        frame.fillna("").astype(str).to_numpy().ravel().tolist()
    )
    return bool(EMAIL_PATTERN.search(serialized) or PHONE_PATTERN.search(serialized))


def validate_department_activation_data(departments: pd.DataFrame) -> None:
    """실제 학과 master 안전 후보의 스키마·중복·5개 학과를 검증한다."""

    missing = set(DEPARTMENT_OUTPUT_COLUMNS).difference(departments.columns)
    if missing:
        raise ValueError(f"학과 활성화 필수 열이 없습니다: {sorted(missing)}")
    working = departments.loc[:, list(DEPARTMENT_OUTPUT_COLUMNS)].copy()
    for column in DEPARTMENT_OUTPUT_COLUMNS:
        working[column] = working[column].map(_clean_text)
    if working.empty:
        raise ValueError("활성화할 학과 안전 후보가 없습니다.")
    if working.eq("").any(axis=1).any():
        raise ValueError("학과 코드 또는 학과명이 빈 행은 활성화할 수 없습니다.")
    invalid_codes = working["department_code"].str.fullmatch(r"\?+")
    if invalid_codes.any():
        raise ValueError("`???`와 같은 미확정 학과 코드는 활성화할 수 없습니다.")
    if working["department_code"].duplicated().any():
        raise ValueError("학과 코드는 중복될 수 없습니다.")
    if working["department_name"].duplicated().any():
        raise ValueError("학과명은 중복될 수 없습니다.")
    missing_targets = set(configured_department_names()).difference(
        working["department_name"]
    )
    if missing_targets:
        raise ValueError(
            "설정된 5개 적용 학과가 실제 학과 master에 모두 필요합니다: "
            + ", ".join(sorted(missing_targets))
        )
    if _contains_contact(working):
        raise ValueError("학과 master에 이메일 또는 전화번호 패턴이 남아 있습니다.")


def prepare_activation_ready_courses(courses: pd.DataFrame) -> pd.DataFrame:
    """개설·추천 정보가 모두 있는 교과목 행만 활성화 후보로 반환한다.

    Parameters:
        courses: dry-run이 만든 canonical 교과목 안전 후보.

    Returns:
        유형을 정규화하고 행 단위 필수 검증을 통과한 교과목.

    Assumptions:
        누락 정보를 임의 추정하지 않으며 개설 여부는 명시적 True만 허용한다.
    """

    if _is_source_backed_course_catalog(courses):
        return prepare_source_backed_course_catalog(courses)
    missing = set(COURSE_COLUMNS).difference(courses.columns)
    if missing:
        raise ValueError(f"교과 활성화 필수 열이 없습니다: {sorted(missing)}")
    working = courses.loc[:, list(COURSE_OUTPUT_COLUMNS)].copy()
    for column in COURSE_OUTPUT_COLUMNS:
        if column != "is_available":
            working[column] = working[column].map(_clean_text)
    text_ready = ~working[list(COURSE_REQUIRED_TEXT_COLUMNS)].eq("").any(axis=1)
    numeric_values = {
        column: pd.to_numeric(working[column], errors="coerce")
        for column in ("grade_level", "semester", "credit")
    }
    numeric_ready = (
        numeric_values["grade_level"].between(1, 6)
        & numeric_values["semester"].isin({1, 2})
        & numeric_values["credit"].between(1, 9)
    )
    available = working["is_available"].map(_explicit_true)
    duplicate = working["course_id"].ne("") & working[
        "course_id"
    ].duplicated(keep=False)
    contact_mask = working.apply(
        lambda row: _contains_contact(pd.DataFrame([row])), axis=1
    )
    ready = working.loc[
        text_ready & numeric_ready & available & ~duplicate & ~contact_mask
    ].copy()
    ready["grade_level"] = numeric_values["grade_level"].loc[ready.index].astype(int)
    ready["semester"] = numeric_values["semester"].loc[ready.index].astype(int)
    ready["credit"] = numeric_values["credit"].loc[ready.index].astype(int)
    ready["is_available"] = True
    return ready.reset_index(drop=True)


def validate_course_activation_data(
    courses: pd.DataFrame,
    department_names: Iterable[str],
    *,
    minimum_courses: int = 3,
) -> None:
    """선택된 실제 교과목이 앱 추천에 바로 사용 가능한지 검증한다."""

    if _is_source_backed_course_catalog(courses):
        ready = prepare_source_backed_course_catalog(courses)
        if len(ready) < minimum_courses:
            raise ValueError(
                f"학습경로를 만들려면 실제 개설 교과목이 최소 "
                f"{minimum_courses}개 필요합니다."
            )
        target_departments = set(configured_department_names())
        offering_departments = {
            item.strip()
            for value in ready["offering_departments"]
            for item in str(value).split("|")
            if item.strip()
        }
        missing_targets = target_departments.difference(offering_departments)
        if missing_targets:
            raise ValueError(
                "설정된 5개 학과 중 실제 개설강좌가 없는 학과가 있습니다: "
                + ", ".join(sorted(missing_targets))
            )
        entry_courses = ready[ready["grade_level"] <= 1]
        if len(entry_courses) < minimum_courses:
            raise ValueError(
                f"1학년 추천 후보가 최소 {minimum_courses}개 필요합니다."
            )
        return

    ready = prepare_activation_ready_courses(courses)
    if len(ready) != len(courses):
        raise ValueError(
            "선택 교과목 중 개설 여부·학년·학기·학점·설명·역량·직무·관심 분야 "
            "검증을 통과하지 못한 행이 있습니다."
        )
    if len(ready) < minimum_courses:
        raise ValueError(
            f"학습경로 fallback 없이 사용하려면 검토 완료 교과목이 "
            f"최소 {minimum_courses}개 필요합니다."
        )
    allowed_departments = {
        _clean_text(value) for value in department_names if _clean_text(value)
    }
    unknown = set(ready["department"]).difference(allowed_departments)
    if unknown:
        raise ValueError(
            "활성 학과 master와 연결되지 않는 교과목 학과가 있습니다: "
            + ", ".join(sorted(unknown))
        )
    course_ids = set(ready["course_id"])
    for value in ready["prerequisites"]:
        prerequisites = {
            item.strip() for item in str(value).split("|") if item.strip()
        }
        if not prerequisites.issubset(course_ids):
            raise ValueError(
                "선수과목은 함께 활성화한 교과목 ID만 참조할 수 있습니다."
            )
    entry_courses = ready[
        (ready["grade_level"] <= 1) & ready["prerequisites"].eq("")
    ]
    if len(entry_courses) < minimum_courses:
        raise ValueError(
            f"1학년 학생도 경로를 구성할 수 있도록 선수과목 없는 1학년 교과목이 "
            f"최소 {minimum_courses}개 필요합니다."
        )


def validate_program_activation_data(programs: pd.DataFrame) -> None:
    """실제 비교과 master가 Repository 계약과 개인정보 규칙을 지키는지 검증한다."""

    missing = SUPPORT_PROGRAM_COLUMNS.difference(programs.columns)
    if missing:
        raise ValueError(f"비교과 활성화 필수 열이 없습니다: {sorted(missing)}")
    validate_imported_programs(programs)


class MasterActivationService:
    """실제 master 활성화·백업·상태 확인을 데이터 디렉터리에 한정한다."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.config = config if config is not None else load_app_config()
        self.backup_dir = self.data_dir / "master_backups"

    def _master_path(self, master_type: str) -> Path:
        """설정된 실제 master의 data 디렉터리 내부 경로를 반환한다."""

        return self.data_dir / configured_master_file_name(
            master_type, self.config
        )

    def _fallback_path(self, master_type: str) -> Path:
        """설정된 synthetic fallback 경로를 반환한다."""

        return self.data_dir / configured_fallback_file_name(
            master_type, self.config
        )

    def _backup_existing(self, destination: Path) -> Path | None:
        """기존 활성 파일이 있으면 교체 직전 복구본을 만든다."""

        if not destination.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.backup_dir / f"{destination.stem}_{timestamp}.csv"
        shutil.copy2(destination, backup)
        return backup

    def _activate(
        self,
        master_type: str,
        data: pd.DataFrame,
        validator: Callable[[pd.DataFrame], None],
    ) -> MasterActivationResult:
        """검증·백업·임시파일 재검증 뒤 실제 master를 원자적으로 교체한다."""

        validator(data)
        destination = self._master_path(master_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = self._backup_existing(destination)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            data.to_csv(temporary, index=False, encoding="utf-8")
            reloaded = pd.read_csv(temporary, keep_default_na=False)
            validator(reloaded)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return MasterActivationResult(
            master_type=master_type,
            destination_path=destination,
            row_count=len(data),
            backup_path=backup,
            activated_at=datetime.now(UTC).isoformat(),
        )

    def activate_departments(
        self, departments: pd.DataFrame
    ) -> MasterActivationResult:
        """중복·미확정 코드를 제외한 실제 학과 안전 후보를 활성화한다."""

        clean = departments.loc[:, list(DEPARTMENT_OUTPUT_COLUMNS)].copy()
        validate_department_activation_data(clean)
        return self._activate(
            "departments", clean, validate_department_activation_data
        )

    def activate_programs(self, programs: pd.DataFrame) -> MasterActivationResult:
        """운영 상태와 무관한 실제 WINGS 비교과 안전 후보를 활성화한다."""

        clean = programs.loc[:, list(PROGRAM_OUTPUT_COLUMNS)].copy()
        return self._activate("programs", clean, validate_program_activation_data)

    def activate_reviewed_courses(
        self,
        courses: pd.DataFrame,
        reviewed_course_ids: Sequence[str],
    ) -> MasterActivationResult:
        """교직원이 선택한 행 중 활성화 준비 검증을 통과한 교과목만 저장한다."""

        requested_ids = tuple(
            dict.fromkeys(
                _clean_text(course_id)
                for course_id in reviewed_course_ids
                if _clean_text(course_id)
            )
        )
        if not requested_ids:
            raise ValueError("활성화할 검토 완료 교과목을 선택해 주세요.")
        ready = prepare_activation_ready_courses(courses)
        selected = ready[ready["course_id"].isin(requested_ids)].copy()
        missing_ids = set(requested_ids).difference(selected["course_id"])
        if missing_ids:
            raise ValueError(
                "활성화 준비 검증을 통과하지 못한 교과목 ID가 있습니다: "
                + ", ".join(sorted(missing_ids)[:10])
            )
        department_path = self._master_path("departments")
        if not department_path.exists():
            raise ValueError("교과목보다 실제 학과 master를 먼저 활성화해야 합니다.")
        departments = pd.read_csv(department_path, keep_default_na=False)
        validate_department_activation_data(departments)
        minimum = int(
            self.config.get("course_recommendation", {}).get(
                "minimum_courses", 3
            )
        )

        def validate(selected_courses: pd.DataFrame) -> None:
            validate_course_activation_data(
                selected_courses,
                departments["department_name"],
                minimum_courses=minimum,
            )

        validate(selected)
        return self._activate("courses", selected, validate)

    def activate_course_offering_catalog(
        self,
        courses: pd.DataFrame,
    ) -> MasterActivationResult:
        """전 학과 실제 개설강좌에서 만든 전체 추천 후보를 활성화한다."""

        if not _is_source_backed_course_catalog(courses):
            raise ValueError("실제 개설강좌에서 만든 추천 후보가 아닙니다.")
        department_path = self._master_path("departments")
        if not department_path.exists():
            raise ValueError("교과목보다 실제 학과 master를 먼저 활성화해야 합니다.")
        departments = pd.read_csv(department_path, keep_default_na=False)
        validate_department_activation_data(departments)
        minimum = int(
            self.config.get("course_recommendation", {}).get(
                "minimum_courses", 3
            )
        )

        def validate(candidate_courses: pd.DataFrame) -> None:
            validate_course_activation_data(
                candidate_courses,
                departments["department_name"],
                minimum_courses=minimum,
            )

        clean = prepare_source_backed_course_catalog(courses)
        validate(clean)
        return self._activate("courses", clean, validate)

    def _fallback_status(
        self,
        master_type: str,
        display_name: str,
        reason: str,
    ) -> ActiveMasterStatus:
        """실제 master 대신 synthetic 파일을 사용하는 상태를 만든다."""

        fallback = self._fallback_path(master_type)
        row_count = 0
        message = reason
        if fallback.exists():
            try:
                row_count = len(pd.read_csv(fallback, keep_default_na=False))
                message = f"{reason} synthetic fallback을 사용합니다."
            except Exception as error:
                message = f"{reason} fallback도 읽을 수 없습니다: {error}"
        else:
            message = f"{reason} fallback 파일도 없습니다."
        return ActiveMasterStatus(
            master_type=master_type,
            display_name=display_name,
            selected_file=fallback.name,
            row_count=row_count,
            using_actual=False,
            using_fallback=True,
            message=message,
        )

    def inspect_statuses(self) -> tuple[ActiveMasterStatus, ...]:
        """현재 실제 master 유효성과 fallback 선택 상태를 점검한다."""

        department_path = self._master_path("departments")
        active_department_names: list[str] = []
        if department_path.exists():
            try:
                departments = pd.read_csv(
                    department_path, keep_default_na=False
                )
                validate_department_activation_data(departments)
                active_department_names = departments[
                    "department_name"
                ].astype(str).tolist()
                department_status = ActiveMasterStatus(
                    "departments",
                    "학과",
                    department_path.name,
                    len(departments),
                    True,
                    False,
                    "검증된 실제 학과 master를 사용합니다.",
                )
            except Exception as error:
                department_status = ActiveMasterStatus(
                    "departments",
                    "학과",
                    "config/app_config.yaml",
                    len(configured_department_names()),
                    False,
                    True,
                    f"실제 학과 master 오류로 설정된 5개 학과를 유지합니다: {error}",
                )
        else:
            department_status = ActiveMasterStatus(
                "departments",
                "학과",
                "config/app_config.yaml",
                len(configured_department_names()),
                False,
                True,
                "실제 학과 master가 없어 설정된 5개 학과를 유지합니다.",
            )

        program_path = self._master_path("programs")
        if program_path.exists():
            try:
                programs = pd.read_csv(program_path, keep_default_na=False)
                validate_program_activation_data(programs)
                program_status = ActiveMasterStatus(
                    "programs",
                    "비교과",
                    program_path.name,
                    len(programs),
                    True,
                    False,
                    "검증된 실제 WINGS 비교과 master를 사용합니다.",
                )
            except Exception as error:
                program_status = self._fallback_status(
                    "programs", "비교과", f"실제 비교과 master 오류: {error}."
                )
        else:
            program_status = self._fallback_status(
                "programs", "비교과", "실제 비교과 master가 없습니다."
            )

        course_path = self._master_path("courses")
        if course_path.exists() and active_department_names:
            try:
                courses = pd.read_csv(course_path, keep_default_na=False)
                minimum = int(
                    self.config.get("course_recommendation", {}).get(
                        "minimum_courses", 3
                    )
                )
                validate_course_activation_data(
                    courses,
                    active_department_names,
                    minimum_courses=minimum,
                )
                source_backed = _is_source_backed_course_catalog(courses)
                course_status = ActiveMasterStatus(
                    "courses",
                    "교과목",
                    course_path.name,
                    len(courses),
                    True,
                    False,
                    (
                        "2026년 실제 개설강좌 추천 master를 사용합니다."
                        if source_backed
                        else "검토 완료 실제 교과목 master를 사용합니다."
                    ),
                )
            except Exception as error:
                course_status = self._fallback_status(
                    "courses", "교과목", f"실제 교과목 master 오류: {error}."
                )
        elif course_path.exists():
            course_status = self._fallback_status(
                "courses",
                "교과목",
                "유효한 실제 학과 master가 없어 실제 교과목을 사용할 수 없습니다.",
            )
        else:
            course_status = self._fallback_status(
                "courses", "교과목", "검토 완료 실제 교과목 master가 없습니다."
            )
        return department_status, program_status, course_status
