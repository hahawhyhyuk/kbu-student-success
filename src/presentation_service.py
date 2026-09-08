"""발표 요약 화면에 표시할 구현 근거 지표를 Repository에서 집계한다."""

from __future__ import annotations

from dataclasses import dataclass

from src.repositories import StudentSuccessData


RISK_DOMAIN_COUNT = 5
SENSITIVE_FIELD_NAMES = frozenset(
    {
        "gender",
        "sex",
        "age",
        "nationality",
        "disability",
        "economic_status",
        "health_information",
        "religion",
        "political_orientation",
    }
)


@dataclass(frozen=True)
class PresentationEvidence:
    """발표에서 구현 범위를 객관적으로 제시할 집계 근거."""

    student_count: int
    week_count: int
    department_count: int
    risk_domain_count: int
    support_program_count: int
    course_count: int
    sensitive_fields_used: tuple[str, ...]


def build_presentation_evidence(data: StudentSuccessData) -> PresentationEvidence:
    """Repository master와 활동 데이터로 발표용 구현 지표를 산출한다.

    Parameters:
        data: 스키마 검증을 통과한 학생성공 Repository 데이터.

    Returns:
        학생·주차·학과·프로그램·교과목 수와 민감필드 현황.

    Assumptions:
        이 지표는 구현 범위이며 실제 성과·예측 정확도를 의미하지 않는다.
    """

    all_columns = {
        str(column).strip().lower()
        for frame in data.as_full_tuple()
        for column in frame.columns
    }
    sensitive_fields_used = tuple(sorted(all_columns & SENSITIVE_FIELD_NAMES))
    return PresentationEvidence(
        student_count=int(data.students["student_id"].nunique()),
        week_count=int(data.weekly_activity["week"].nunique()),
        department_count=int(data.students["department"].nunique()),
        risk_domain_count=RISK_DOMAIN_COUNT,
        support_program_count=int(data.support_programs["program_id"].nunique()),
        course_count=int(data.courses["course_id"].nunique()),
        sensitive_fields_used=sensitive_fields_used,
    )
