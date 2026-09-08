"""서비스 적용 대상 5개 학과를 앱 설정에서 읽고 검증한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.utils import load_app_config


REQUIRED_DEPARTMENT_COUNT = 5


@dataclass(frozen=True)
class DepartmentDefinition:
    """학과의 내부 식별 코드와 사용자 표시명을 함께 보관한다."""

    code: str
    name: str


def load_department_catalog(
    config: Mapping[str, Any] | None = None,
) -> tuple[DepartmentDefinition, ...]:
    """앱 설정에서 중복 없는 5개 학과 정의를 반환한다.

    Parameters:
        config: 선택적인 앱 설정. 미지정 시 ``config/app_config.yaml`` 사용.

    Returns:
        설정 순서를 유지한 5개 ``DepartmentDefinition``.

    Assumptions:
        MVP 적용 범위는 정확히 5개 학과이며 코드와 표시명은 모두 필수다.
    """

    active_config = config if config is not None else load_app_config()
    raw_departments = active_config.get("departments")
    if not isinstance(raw_departments, list):
        raise ValueError("departments 설정은 학과 목록이어야 합니다.")

    catalog: list[DepartmentDefinition] = []
    for index, raw_department in enumerate(raw_departments, start=1):
        if not isinstance(raw_department, Mapping):
            raise ValueError(f"departments {index}번째 항목은 mapping이어야 합니다.")
        code = str(raw_department.get("code", "")).strip()
        name = str(raw_department.get("name", "")).strip()
        if not code or not name:
            raise ValueError(f"departments {index}번째 항목의 code와 name이 필요합니다.")
        catalog.append(DepartmentDefinition(code=code, name=name))

    if len(catalog) != REQUIRED_DEPARTMENT_COUNT:
        raise ValueError(
            f"MVP 적용 학과는 정확히 {REQUIRED_DEPARTMENT_COUNT}개여야 합니다."
        )
    codes = [department.code for department in catalog]
    names = [department.name for department in catalog]
    if len(codes) != len(set(codes)):
        raise ValueError("departments code는 중복될 수 없습니다.")
    if len(names) != len(set(names)):
        raise ValueError("departments name은 중복될 수 없습니다.")
    return tuple(catalog)


def configured_department_names(
    config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """화면과 synthetic master에서 사용할 학과 표시명을 반환한다."""

    return tuple(
        department.name for department in load_department_catalog(config)
    )


def validate_configured_departments(
    values: Iterable[object],
    *,
    source_name: str,
    config: Mapping[str, Any] | None = None,
) -> None:
    """데이터의 학과 값이 설정된 5개 학과 밖으로 벗어나지 않게 검증한다."""

    allowed = set(configured_department_names(config))
    actual = {
        str(value).strip()
        for value in values
        if str(value).strip()
    }
    unknown = actual.difference(allowed)
    if unknown:
        raise ValueError(
            f"{source_name}에 설정되지 않은 학과가 있습니다: {sorted(unknown)}. "
            "`python -m src.data_generator`로 synthetic CSV를 다시 생성하세요."
        )
