"""공통 경로, 설정, 데이터 로딩 및 설명 템플릿 유틸리티."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "risk_config.yaml"
DEFAULT_APP_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.yaml"
DATA_DIR = PROJECT_ROOT / "data"


@lru_cache(maxsize=4)
def load_risk_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """위험 계산 설정을 읽는다.

    Parameters:
        config_path: 선택적인 YAML 설정 파일 경로.

    Returns:
        위험 구간, 영역 가중치, 결측 기본값을 담은 사전.

    Assumptions:
        설정 파일은 신뢰할 수 있는 프로젝트 내부 YAML이며 최상위 값은 mapping이다.
    """

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"위험 설정은 mapping이어야 합니다: {path}")
    return config


@lru_cache(maxsize=4)
def load_app_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """AI provider와 추천 설정을 읽는다.

    Parameters:
        config_path: 선택적인 YAML 설정 파일 경로.

    Returns:
        AI 모델, timeout, 추천 top_k와 가중치를 담은 사전.

    Assumptions:
        API key는 설정 파일에 저장하지 않고 환경변수 또는 Streamlit secrets를 사용한다.
    """

    path = Path(config_path) if config_path else DEFAULT_APP_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"앱 설정은 mapping이어야 합니다: {path}")
    return config


def ensure_demo_data(data_dir: str | Path | None = None) -> Path:
    """CsvRepository를 통해 필수 synthetic 데이터 존재를 보장한다.

    Parameters:
        data_dir: CSV를 확인하고 생성할 디렉터리.

    Returns:
        학생·활동·체크인·지원프로그램·교과목·이수내역 CSV가 존재하는 디렉터리.

    Assumptions:
        일부 파일만 존재하는 경우 전체 synthetic CSV를 함께 재생성한다.
    """

    from src.repositories import CsvRepository

    return CsvRepository(data_dir=data_dir).ensure_available()


def load_demo_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CsvRepository 결과를 기존 튜플 형식으로 반환한다.

    Parameters:
        data_dir: CSV가 저장된 디렉터리.

    Returns:
        students, weekly_activity, checkins 순서의 DataFrame 튜플.

    Assumptions:
        빈 문자열은 진로 미설정 여부를 보존하기 위해 NaN으로 자동 변환하지 않는다.
    """

    from src.repositories import CsvRepository

    return CsvRepository(data_dir=data_dir).get_all().as_tuple()


def _display_number(value: Any, digits: int = 0) -> str:
    """화면 설명에 사용할 숫자를 안전하게 문자열로 변환한다."""

    try:
        if pd.isna(value):
            return "정보 없음"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "정보 없음"


def generate_risk_reasons(
    record: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> list[str]:
    """최신 위험 스냅샷을 사람이 이해할 수 있는 문장으로 설명한다.

    Parameters:
        record: 원천 지표와 영역별 위험점수가 포함된 학생 스냅샷.
        config: 선택적인 위험 설정. 미지정 시 프로젝트 기본 설정을 사용한다.

    Returns:
        주요 위험영역별 deterministic 설명 문장 목록.

    Assumptions:
        생성형 AI를 사용하지 않으며, 정상 범위 학생은 안정 상태 안내 한 문장만 반환한다.
    """

    active_config = dict(config) if config is not None else load_risk_config()
    overall = float(record.get("overall_risk", 0) or 0)
    if overall <= float(active_config["risk_levels"]["normal_max"]):
        return [
            "현재 다섯 영역의 종합 지표가 정상 범위입니다. 주차별 변화를 계속 관찰합니다."
        ]

    domain_templates = {
        "attendance": (
            "최근 출석률이 {attendance}%이고 연속결석이 {consecutive}회로 "
            "출결 영역을 우선 확인할 필요가 있습니다."
        ).format(
            attendance=_display_number(record.get("attendance_rate"), 1),
            consecutive=_display_number(record.get("consecutive_absence")),
        ),
        "engagement": (
            "과제 제출률 {assignment}%, 주간 LMS 로그인 {login}일, 활동 변화 {change}%로 "
            "학습참여 흐름을 확인할 필요가 있습니다."
        ).format(
            assignment=_display_number(record.get("assignment_submission_rate"), 1),
            login=_display_number(record.get("lms_login_days")),
            change=_display_number(record.get("lms_activity_change"), 1),
        ),
        "achievement": (
            "직전 GPA {gpa}, 최근 퀴즈 {quiz}점과 학업 이력을 함께 볼 때 "
            "학업성취 지원 여부를 검토할 필요가 있습니다."
        ).format(
            gpa=_display_number(record.get("previous_gpa"), 2),
            quiz=_display_number(record.get("quiz_score"), 1),
        ),
        "major_adaptation": (
            "전공 흥미 {interest}점, 만족 {satisfaction}점, 지속 의향 {continuation}점으로 "
            "전공적응에 관한 대화를 우선 권장합니다."
        ).format(
            interest=_display_number(record.get("major_interest")),
            satisfaction=_display_number(record.get("major_satisfaction")),
            continuation=_display_number(
                record.get("major_continuation_intent")
            ),
        ),
        "career": (
            "진로 명확도가 {clarity}점이고 희망직무가 {job_state}되어 있어 "
            "진로 탐색 지원을 검토할 필요가 있습니다."
        ).format(
            clarity=_display_number(record.get("career_clarity")),
            job_state=(
                "설정"
                if str(record.get("desired_job", "")).strip()
                else "아직 설정되지 않음"
            ),
        ),
    }

    domain_scores = {
        domain: float(record.get(f"{domain}_risk", 0) or 0)
        for domain in domain_templates
    }
    primary = max(domain_scores, key=domain_scores.get)
    secondary_min = float(active_config["risk_types"]["secondary_min"])
    selected = [primary]
    selected.extend(
        domain
        for domain, score in sorted(
            domain_scores.items(), key=lambda item: item[1], reverse=True
        )
        if domain != primary and score >= secondary_min
    )
    return [domain_templates[domain] for domain in selected]
