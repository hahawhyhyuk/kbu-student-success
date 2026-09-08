"""애플리케이션이 구체 저장소 클래스를 직접 알지 않게 하는 provider factory."""

from __future__ import annotations

import os
from pathlib import Path

from src.repositories.base import StudentSuccessRepository
from src.repositories.csv_repository import CsvRepository


DATA_PROVIDER_ENV = "KBU_DATA_PROVIDER"
DEFAULT_DATA_PROVIDER = "csv"


def create_repository(
    provider_name: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> StudentSuccessRepository:
    """설정된 provider 이름에 맞는 Repository를 생성한다.

    Parameters:
        provider_name: provider 식별자. 미지정 시 KBU_DATA_PROVIDER 또는 csv.
        data_dir: CsvRepository에 전달할 선택적 데이터 디렉터리.

    Returns:
        StudentSuccessRepository 인터페이스 구현체.

    Assumptions:
        현재 등록된 구현은 csv뿐이다. 향후 OracleRepository를 등록할 때 이 factory의
        분기만 추가하며 UI와 risk engine은 수정하지 않는다.
    """

    selected_provider = (
        provider_name or os.getenv(DATA_PROVIDER_ENV, DEFAULT_DATA_PROVIDER)
    ).strip().lower()
    if selected_provider == "csv":
        return CsvRepository(data_dir=data_dir)
    raise ValueError(
        f"지원하지 않는 데이터 provider입니다: {selected_provider}. "
        "현재 사용 가능한 provider: csv"
    )


def get_default_repository() -> StudentSuccessRepository:
    """현재 애플리케이션 설정에 해당하는 기본 Repository를 반환한다."""

    return create_repository()
