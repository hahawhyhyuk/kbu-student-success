"""원본 master를 수정하지 않고 추천 대상의 관심 분야 근거를 판별한다."""

from __future__ import annotations

import re
from typing import Iterable, Mapping


INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "데이터·AI": (
        "데이터",
        "빅데이터",
        "인공지능",
        "AI",
        "머신러닝",
        "딥러닝",
        "파이썬",
        "코딩",
        "프로그래밍",
        "통계",
        "시각화",
        "컴퓨테이셔널",
    ),
    "경영·마케팅": (
        "경영",
        "마케팅",
        "시장",
        "소비자",
        "고객분석",
        "브랜드",
        "창업",
        "비즈니스",
        "광고",
        "유통",
    ),
    "콘텐츠·디자인": (
        "콘텐츠",
        "디자인",
        "영상",
        "그래픽",
        "시각",
        "UX",
        "UI",
        "촬영",
        "편집",
        "스토리텔링",
        "브랜딩",
    ),
    "서비스": (
        "서비스기획",
        "서비스설계",
        "서비스운영",
        "고객경험",
        "사용자경험",
        "프로젝트관리",
        "운영기획",
        "고객관리",
        "UX",
    ),
    "보건": (
        "보건",
        "건강",
        "의료",
        "간호",
        "환자",
        "임상",
        "치료",
        "재활",
        "위생",
        "안전보건",
    ),
    "상담·복지": (
        "상담",
        "복지",
        "심리",
        "사례관리",
        "사회복지",
        "대인관계",
        "자존감",
        "정서",
        "스트레스",
    ),
}


def _normalized_text(value: object) -> str:
    """띄어쓰기와 기호 차이를 제거한 소문자 비교 문자열을 반환한다."""

    return re.sub(r"[^가-힣a-z0-9]+", "", str(value or "").lower())


def infer_interest_evidence(
    text: object,
    requested_interests: Iterable[str] | None = None,
    *,
    keyword_map: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """과목·프로그램 설명에서 관심 분야별 직접 근거 단어를 찾는다.

    Parameters:
        text: 과목명·강좌설명 또는 프로그램명·설명을 합친 문자열.
        requested_interests: 학생이 명시한 관심 분야. 없으면 전체 분야를 검사한다.
        keyword_map: 테스트나 설정 확장을 위한 선택적 분야별 키워드.

    Returns:
        한 개 이상의 근거가 발견된 분야와 실제 일치 키워드의 매핑.

    Assumptions:
        결과는 실행 중 계산하며 원본 Excel이나 CSV에 태그를 쓰지 않는다.
    """

    active_map = keyword_map or INTEREST_KEYWORDS
    if requested_interests is None:
        labels = tuple(active_map)
    else:
        requested = {str(value).strip() for value in requested_interests}
        labels = tuple(label for label in active_map if label in requested)
    normalized = _normalized_text(text)
    evidence: dict[str, tuple[str, ...]] = {}
    for label in labels:
        matches = tuple(
            dict.fromkeys(
                str(keyword)
                for keyword in active_map[label]
                if _normalized_text(keyword) in normalized
            )
        )
        if matches:
            evidence[label] = matches
    return evidence


def supported_interest_labels(values: Iterable[object]) -> set[str]:
    """자유형 AI 결과에서 시스템이 지원하는 관심 분야만 남긴다."""

    return {
        str(value).strip()
        for value in values
        if str(value).strip() in INTEREST_KEYWORDS
    }
