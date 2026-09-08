"""설정 기반 위험등급 및 위험유형 분류."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.utils import load_risk_config


DOMAIN_LABELS = {
    "attendance": "출결위험형",
    "engagement": "학습참여저하형",
    "achievement": "학업부진형",
    "major_adaptation": "전공부적응형",
    "career": "진로미설정형",
}


def _safe_score(score: Any) -> float:
    """점수를 0~100의 유효한 실수로 정규화한다."""

    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return min(100.0, max(0.0, numeric))


def classify_risk_level(
    score: float, config: Mapping[str, Any] | None = None
) -> str:
    """종합 위험점수를 한국어 위험등급으로 변환한다.

    Parameters:
        score: 0~100 범위가 기대되는 종합 위험점수.
        config: 선택적인 위험 설정.

    Returns:
        정상, 관심, 주의, 고위험 중 하나.

    Assumptions:
        범위를 벗어나거나 유효하지 않은 값은 안전하게 0~100으로 정규화한다.
    """

    active_config = config or load_risk_config()
    levels = active_config["risk_levels"]
    safe_score = _safe_score(score)
    if safe_score <= float(levels["normal_max"]):
        return "정상"
    if safe_score <= float(levels["attention_max"]):
        return "관심"
    if safe_score <= float(levels["caution_max"]):
        return "주의"
    return "고위험"


def classify_risk_types(
    domain_scores: Mapping[str, float],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """다섯 영역 점수에서 주·보조·복합 위험유형을 분류한다.

    Parameters:
        domain_scores: attendance 등 다섯 영역의 0~100 점수 mapping.
        config: 선택적인 위험유형 임계값 설정.

    Returns:
        primary/secondary 영역, 한국어 표시명, 복합위험 여부를 담은 사전.

    Assumptions:
        누락된 영역은 0점으로 처리하며 동점은 DOMAIN_LABELS 선언 순서를 따른다.
    """

    active_config = config or load_risk_config()
    rules = active_config["risk_types"]
    scores = {
        domain: _safe_score(domain_scores.get(domain, 0)) for domain in DOMAIN_LABELS
    }
    primary_domain = max(scores, key=scores.get)
    secondary_domains = [
        domain
        for domain, score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        if domain != primary_domain and score >= float(rules["secondary_min"])
    ]
    complex_domains = [
        domain
        for domain, score in scores.items()
        if score >= float(rules["complex_min"])
    ]
    is_complex = len(complex_domains) >= int(rules["complex_domain_count"])

    return {
        "primary_domain": primary_domain,
        "primary_risk_type": DOMAIN_LABELS[primary_domain],
        "secondary_domains": secondary_domains,
        "secondary_risk_types": [DOMAIN_LABELS[item] for item in secondary_domains],
        "is_complex": is_complex,
        "display_risk_type": (
            "복합위험형" if is_complex else DOMAIN_LABELS[primary_domain]
        ),
    }
