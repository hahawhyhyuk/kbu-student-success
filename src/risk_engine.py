"""설명 가능한 규칙 기반 초기경보 위험점수 엔진."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

from src.risk_classifier import classify_risk_level, classify_risk_types
from src.utils import load_risk_config


DOMAIN_KEYS = (
    "attendance",
    "engagement",
    "achievement",
    "major_adaptation",
    "career",
)


def _clamp(score: float) -> float:
    """위험점수를 0~100 범위로 제한한다."""

    return round(min(100.0, max(0.0, float(score))), 2)


def _safe_number(value: Any, default: float) -> float:
    """None, NaN, 문자열 결측을 설정 기본값으로 변환한다."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def _score_from_minimum_bands(
    value: float, bands: Sequence[Mapping[str, float]]
) -> float:
    """내림차순 최소값 구간에서 첫 번째로 일치하는 위험점수를 찾는다."""

    for band in bands:
        if value >= float(band["min"]):
            return float(band["score"])
    return float(bands[-1]["score"])


def _weighted_average(
    scores: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """설정 가중치의 합으로 정규화한 가중평균을 계산한다."""

    total_weight = sum(float(weights[key]) for key in scores)
    if total_weight <= 0:
        raise ValueError("위험 가중치 합은 0보다 커야 합니다.")
    total = sum(float(scores[key]) * float(weights[key]) for key in scores)
    return _clamp(total / total_weight)


def calculate_attendance_risk(
    attendance_rate: float | None,
    consecutive_absence: int | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """출석률과 연속결석 횟수로 출결 위험도를 계산한다.

    Parameters:
        attendance_rate: 최근 주차 출석률(0~100).
        consecutive_absence: 최근 연속결석 횟수.
        config: 선택적인 위험 설정.

    Returns:
        0~100 범위 출결 위험점수.

    Assumptions:
        결측값은 YAML의 attendance.missing_defaults를 사용한다.
    """

    active_config = config or load_risk_config()
    rules = active_config["attendance"]
    defaults = rules["missing_defaults"]
    rate = _safe_number(attendance_rate, defaults["attendance_rate"])
    absence = _safe_number(
        consecutive_absence, defaults["consecutive_absence"]
    )
    base_score = _score_from_minimum_bands(rate, rules["rate_bands"])
    bonus = _score_from_minimum_bands(
        absence, rules["consecutive_absence_bonuses"]
    )
    return _clamp(base_score + bonus)


def calculate_engagement_risk(
    assignment_submission_rate: float | None,
    lms_login_days: float | None,
    lms_activity_change: float | None,
    video_completion_rate: float | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """과제·LMS·영상 활동을 결합해 학습참여 위험도를 계산한다.

    Parameters:
        assignment_submission_rate: 과제 제출률(0~100).
        lms_login_days: 최근 1주 LMS 로그인 일수(0~7).
        lms_activity_change: 전주 대비 LMS 활동 변화율(%).
        video_completion_rate: 영상 학습 완료율(0~100).
        config: 선택적인 위험 설정.

    Returns:
        0~100 범위 학습참여 위험점수.

    Assumptions:
        각 입력은 YAML의 구간점수로 변환한 뒤 설정 가중평균한다.
    """

    active_config = config or load_risk_config()
    rules = active_config["engagement"]
    defaults = rules["missing_defaults"]
    values = {
        "assignment_submission_rate": _safe_number(
            assignment_submission_rate,
            defaults["assignment_submission_rate"],
        ),
        "lms_login_days": _safe_number(
            lms_login_days, defaults["lms_login_days"]
        ),
        "video_completion_rate": _safe_number(
            video_completion_rate,
            defaults["video_completion_rate"],
        ),
        "lms_activity_change": _safe_number(
            lms_activity_change,
            defaults["lms_activity_change"],
        ),
    }
    scores = {
        key: _score_from_minimum_bands(values[key], rules[f"{key}_bands"])
        for key in values
    }
    return _weighted_average(scores, rules["weights"])


def calculate_achievement_risk(
    previous_gpa: float | None,
    quiz_score: float | None,
    repeat_course_count: int | None,
    academic_warning_history: int | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """GPA·퀴즈·재수강·학사경고로 학업성취 위험도를 계산한다.

    Parameters:
        previous_gpa: 4.5 만점 직전학기 GPA.
        quiz_score: 최근 퀴즈 점수(0~100).
        repeat_course_count: 재수강 과목 수.
        academic_warning_history: 학사경고 횟수.
        config: 선택적인 위험 설정.

    Returns:
        0~100 범위 학업성취 위험점수.

    Assumptions:
        결측 입력은 YAML에 정의된 중립 또는 안전 기본값을 사용한다.
    """

    active_config = config or load_risk_config()
    rules = active_config["achievement"]
    defaults = rules["missing_defaults"]
    values = {
        "previous_gpa": _safe_number(previous_gpa, defaults["previous_gpa"]),
        "quiz_score": _safe_number(quiz_score, defaults["quiz_score"]),
        "repeat_course_count": _safe_number(
            repeat_course_count, defaults["repeat_course_count"]
        ),
        "academic_warning_history": _safe_number(
            academic_warning_history, defaults["academic_warning_history"]
        ),
    }
    scores = {
        key: _score_from_minimum_bands(values[key], rules[f"{key}_bands"])
        for key in values
    }
    return _weighted_average(scores, rules["weights"])


def _likert_to_risk(value: Any, config: Mapping[str, Any], default: float) -> float:
    """Likert 1~5 값을 설정된 위험점수로 변환한다."""

    numeric = int(round(_safe_number(value, default)))
    numeric = min(5, max(1, numeric))
    risk_map = config["likert_risk_map"]
    return float(risk_map.get(numeric, risk_map.get(str(numeric))))


def calculate_major_adaptation_risk(
    major_interest: float | None,
    major_satisfaction: float | None,
    major_continuation_intent: float | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """세 전공 관련 Likert 문항의 평균으로 전공적응 위험도를 계산한다.

    Parameters:
        major_interest: 전공 흥미 1~5.
        major_satisfaction: 전공 만족 1~5.
        major_continuation_intent: 전공 지속 의향 1~5.
        config: 선택적인 위험 설정.

    Returns:
        0~100 범위 전공적응 위험점수.

    Assumptions:
        Likert 결측값은 설정의 major_adaptation.missing_default를 사용한다.
    """

    active_config = config or load_risk_config()
    default = float(active_config["major_adaptation"]["missing_default"])
    scores = [
        _likert_to_risk(value, active_config, default)
        for value in (
            major_interest,
            major_satisfaction,
            major_continuation_intent,
        )
    ]
    return _clamp(sum(scores) / len(scores))


def calculate_career_risk(
    career_clarity: float | None,
    desired_job: str | None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """진로 명확도와 희망직무 설정 여부로 진로설계 위험도를 계산한다.

    Parameters:
        career_clarity: 진로 명확도 Likert 1~5.
        desired_job: 희망직무 텍스트. 공백이면 추가 위험을 적용한다.
        config: 선택적인 위험 설정.

    Returns:
        0~100 범위 진로설계 위험점수.

    Assumptions:
        진로 명확도 결측값은 설정 기본값, 직무 결측은 빈 문자열로 처리한다.
    """

    active_config = config or load_risk_config()
    rules = active_config["career"]
    score = _likert_to_risk(
        career_clarity, active_config, float(rules["missing_default"])
    )
    job_text = "" if desired_job is None else str(desired_job).strip()
    if not job_text or job_text.lower() == "nan":
        score += float(rules["missing_desired_job_bonus"])
    return _clamp(score)


def calculate_overall_risk(
    domain_scores: Mapping[str, float],
    config: Mapping[str, Any] | None = None,
) -> float:
    """다섯 영역 위험점수의 설정 가중평균을 계산한다.

    Parameters:
        domain_scores: attendance, engagement, achievement,
            major_adaptation, career 점수 mapping.
        config: 선택적인 위험 설정.

    Returns:
        0~100 범위 종합 위험점수.

    Assumptions:
        누락 또는 유효하지 않은 영역 점수는 0점으로 처리한다.
    """

    active_config = config or load_risk_config()
    safe_scores = {
        key: _clamp(_safe_number(domain_scores.get(key), 0))
        for key in DOMAIN_KEYS
    }
    return _weighted_average(safe_scores, active_config["weights"])


def build_risk_snapshots(
    students: pd.DataFrame,
    weekly_activity: pd.DataFrame,
    checkins: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """가상 원천데이터를 학생·주차별 초기경보 스냅샷으로 변환한다.

    Parameters:
        students: 학생 정적 정보 DataFrame.
        weekly_activity: 학생별 1~4주 활동 DataFrame.
        checkins: 학생별 4주차 체크인 DataFrame.
        config: 선택적인 위험 설정.

    Returns:
        원천지표, 영역점수, 종합점수, 등급, 유형, 주차 변화가 포함된 DataFrame.

    Assumptions:
        현재 가상데이터에는 학생당 체크인이 하나이므로 전공·진로 자기보고 점수는 네 주의
        행동 변화 비교에 동일하게 사용한다. 실제 운영에서는 시점별 체크인을 조인한다.
    """

    required_student_columns = {
        "student_id",
        "previous_gpa",
        "academic_warning_history",
        "repeat_course_count",
    }
    required_activity_columns = {
        "student_id",
        "week",
        "attendance_rate",
        "consecutive_absence",
        "assignment_submission_rate",
        "lms_login_days",
        "lms_activity_change",
        "video_completion_rate",
        "quiz_score",
    }
    required_checkin_columns = {
        "student_id",
        "major_interest",
        "major_satisfaction",
        "major_continuation_intent",
        "career_clarity",
        "desired_job",
    }
    for frame_name, frame, required in (
        ("students", students, required_student_columns),
        ("weekly_activity", weekly_activity, required_activity_columns),
        ("checkins", checkins, required_checkin_columns),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{frame_name} 데이터에 필수 컬럼이 없습니다: {sorted(missing)}"
            )

    active_config = config or load_risk_config()
    checkin_columns = [column for column in checkins.columns if column != "week"]
    merged = weekly_activity.merge(students, on="student_id", how="left")
    merged = merged.merge(
        checkins[checkin_columns], on="student_id", how="left"
    )

    snapshot_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        domain_scores = {
            "attendance": calculate_attendance_risk(
                row.get("attendance_rate"),
                row.get("consecutive_absence"),
                active_config,
            ),
            "engagement": calculate_engagement_risk(
                row.get("assignment_submission_rate"),
                row.get("lms_login_days"),
                row.get("lms_activity_change"),
                row.get("video_completion_rate"),
                active_config,
            ),
            "achievement": calculate_achievement_risk(
                row.get("previous_gpa"),
                row.get("quiz_score"),
                row.get("repeat_course_count"),
                row.get("academic_warning_history"),
                active_config,
            ),
            "major_adaptation": calculate_major_adaptation_risk(
                row.get("major_interest"),
                row.get("major_satisfaction"),
                row.get("major_continuation_intent"),
                active_config,
            ),
            "career": calculate_career_risk(
                row.get("career_clarity"), row.get("desired_job"), active_config
            ),
        }
        overall = calculate_overall_risk(domain_scores, active_config)
        risk_types = classify_risk_types(domain_scores, active_config)
        snapshot = dict(row)
        snapshot.update(
            {f"{domain}_risk": score for domain, score in domain_scores.items()}
        )
        snapshot.update(
            {
                "overall_risk": overall,
                "risk_level": classify_risk_level(overall, active_config),
                "primary_risk_type": risk_types["primary_risk_type"],
                "secondary_risk_types": ", ".join(
                    risk_types["secondary_risk_types"]
                ),
                "risk_type": risk_types["display_risk_type"],
                "is_complex": risk_types["is_complex"],
            }
        )
        snapshot_rows.append(snapshot)

    snapshots = pd.DataFrame(snapshot_rows).sort_values(
        ["student_id", "week"]
    )
    snapshots["risk_change"] = (
        snapshots.groupby("student_id")["overall_risk"].diff().fillna(0).round(2)
    )
    return snapshots.reset_index(drop=True)
