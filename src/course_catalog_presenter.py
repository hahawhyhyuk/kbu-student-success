"""교과목 master 출처에 맞는 안전한 화면 문구를 만든다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _value(course: Any, key: str, default: Any = "") -> Any:
    if isinstance(course, Mapping):
        return course.get(key, default)
    return getattr(course, key, default)


def _values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = value
    else:
        items = str(value).split("|")
    return tuple(
        str(item).strip()
        for item in items
        if str(item).strip() and str(item).strip().lower() != "nan"
    )


def _is_present(value: Any) -> bool:
    return str(value or "").strip().lower() not in {"", "nan", "none"}


def is_source_backed_course(course: Any) -> bool:
    """실제 개설강좌에서 변환된 교과목인지 반환한다."""

    return bool(
        _is_present(_value(course, "catalog_source", ""))
        or _is_present(_value(course, "recommendation_basis", ""))
        or str(_value(course, "registration_check_required", ""))
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def course_offering_departments(course: Any) -> tuple[str, ...]:
    """복수 개설학과를 우선하고 기존 master의 단일 학과를 보조로 사용한다."""

    departments = list(_values(_value(course, "offering_departments", "")))
    primary = str(_value(course, "department", "")).strip()
    if primary and primary.lower() != "nan" and primary not in departments:
        departments.append(primary)
    return tuple(departments)


def format_course_caption(
    course: Any,
    home_department: str | None = None,
) -> str:
    """실제·synthetic 교과목을 구분해 근거와 수강 조건을 표시한다."""

    departments = course_offering_departments(course)
    embedded_home = bool(_value(course, "is_home_department", False))
    is_home = (
        str(home_department).strip() in departments
        if home_department is not None
        else embedded_home
    )
    scope = "소속 학과" if is_home else "타과 선택"
    course_id = str(
        _value(course, "source_course_id", "")
        or _value(course, "course_id", "")
    ).strip()
    department_text = ", ".join(departments) or "미확인"

    if is_source_backed_course(course):
        grades = _values(
            _value(course, "grade_levels", _value(course, "grade_level", ""))
        )
        semesters = _values(
            _value(
                course,
                "offered_semesters",
                _value(course, "semester", ""),
            )
        )
        source_details: list[str] = []
        for value in (
            _value(course, "course_area", ""),
            _value(course, "course_type", ""),
            _value(course, "ncs_type", ""),
            _value(course, "class_method", ""),
        ):
            for item in _values(value):
                if item not in source_details:
                    source_details.append(item)
        parts = [
            scope,
            f"공식 과목코드: {course_id}",
            f"개설학과: {department_text}",
            f"개설학년: {'·'.join(grades) or '미확인'}",
            f"개설학기: {'·'.join(semesters) or '미확인'}",
        ]
        if source_details:
            parts.append("원본 분류: " + " · ".join(source_details))
        parts.append("선수·이수내역 연결 전 · 실제 수강 가능 여부 확인 필요")
        return " · ".join(parts)

    prerequisites = _values(_value(course, "prerequisites", ""))
    grade = str(_value(course, "grade_level", "")).strip()
    semester = str(_value(course, "semester", "")).strip()
    difficulty = str(_value(course, "difficulty", "")).strip()
    credit = _value(course, "credit", "")
    try:
        credit_text = f"{float(credit):g}학점"
    except (TypeError, ValueError):
        credit_text = "학점 미확인"
    return " · ".join(
        (
            scope,
            f"개설학과 {department_text}",
            f"ID: {course_id}",
            f"{grade}학년",
            f"{semester}학기",
            difficulty,
            credit_text,
            f"선수과목: {', '.join(prerequisites) or '없음'}",
        )
    )
