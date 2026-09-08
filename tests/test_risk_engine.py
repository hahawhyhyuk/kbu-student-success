"""위험엔진과 가상데이터의 핵심 계약 테스트."""

import math

import numpy as np
import pandas as pd
import pytest

from src.data_generator import ARCHETYPE_PROPORTIONS, generate_synthetic_data
from src.risk_classifier import classify_risk_level, classify_risk_types
from src.risk_engine import (
    build_risk_snapshots,
    calculate_achievement_risk,
    calculate_attendance_risk,
    calculate_career_risk,
    calculate_engagement_risk,
    calculate_major_adaptation_risk,
    calculate_overall_risk,
)


def _normal_domain_scores() -> dict[str, float]:
    """모든 영역이 안정적인 테스트 입력을 생성한다."""

    return {
        "attendance": calculate_attendance_risk(100, 0),
        "engagement": calculate_engagement_risk(100, 7, 5, 100),
        "achievement": calculate_achievement_risk(4.3, 95, 0, 0),
        "major_adaptation": calculate_major_adaptation_risk(5, 5, 5),
        "career": calculate_career_risk(5, "서비스 기획"),
    }


@pytest.mark.parametrize(
    "score",
    [
        calculate_attendance_risk(-50, 99),
        calculate_engagement_risk(-20, -2, -500, 300),
        calculate_achievement_risk(-1, 200, 20, 10),
        calculate_major_adaptation_risk(-5, 9, 3),
        calculate_career_risk(-10, ""),
        calculate_overall_risk(
            {
                "attendance": -20,
                "engagement": 120,
                "achievement": 50,
                "major_adaptation": 50,
                "career": 50,
            }
        ),
    ],
)
def test_all_scores_stay_between_zero_and_one_hundred(score: float) -> None:
    assert 0 <= score <= 100


def test_perfect_attendance_has_very_low_risk() -> None:
    assert calculate_attendance_risk(100, 0) <= 5


def test_low_attendance_and_consecutive_absence_has_high_risk() -> None:
    assert calculate_attendance_risk(70, 3) >= 90


def test_low_major_likert_has_very_high_risk() -> None:
    assert calculate_major_adaptation_risk(1, 1, 1) >= 95


def test_all_normal_inputs_have_low_overall_risk() -> None:
    assert calculate_overall_risk(_normal_domain_scores()) <= 24


@pytest.mark.parametrize(
    "calculator,args",
    [
        (calculate_attendance_risk, (None, np.nan)),
        (calculate_engagement_risk, (None, np.nan, pd.NA, None)),
        (calculate_achievement_risk, (np.nan, None, pd.NA, None)),
        (calculate_major_adaptation_risk, (None, np.nan, pd.NA)),
        (calculate_career_risk, (np.nan, None)),
    ],
)
def test_domain_functions_do_not_crash_on_missing_values(
    calculator, args
) -> None:
    score = calculator(*args)
    assert math.isfinite(score)
    assert 0 <= score <= 100


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "정상"),
        (24, "정상"),
        (25, "관심"),
        (44, "관심"),
        (45, "주의"),
        (64, "주의"),
        (65, "고위험"),
        (100, "고위험"),
    ],
)
def test_risk_level_boundaries(score: float, expected: str) -> None:
    assert classify_risk_level(score) == expected


def test_complex_risk_requires_two_domains_at_or_above_threshold() -> None:
    result = classify_risk_types(
        {
            "attendance": 80,
            "engagement": 70,
            "achievement": 20,
            "major_adaptation": 30,
            "career": 10,
        }
    )
    assert result["primary_risk_type"] == "출결위험형"
    assert result["display_risk_type"] == "복합위험형"
    assert result["is_complex"] is True


def test_synthetic_data_shape_distribution_and_sensitive_field_exclusion() -> None:
    students, activity, checkins = generate_synthetic_data()
    assert len(students) == 200
    assert len(activity) == 800
    assert len(checkins) == 200
    expected_counts = {
        archetype: int(200 * proportion)
        for archetype, proportion in ARCHETYPE_PROPORTIONS.items()
    }
    assert students["archetype"].value_counts().to_dict() == expected_counts
    sensitive_fields = {
        "gender",
        "age",
        "nationality",
        "disability",
        "economic_status",
        "health_information",
        "religion",
        "political_orientation",
    }
    all_columns = set(students) | set(activity) | set(checkins)
    assert sensitive_fields.isdisjoint(all_columns)


def test_synthetic_generation_is_reproducible() -> None:
    first = generate_synthetic_data(seed=2026)
    second = generate_synthetic_data(seed=2026)
    for first_frame, second_frame in zip(first, second):
        pd.testing.assert_frame_equal(first_frame, second_frame)


def test_snapshot_builder_produces_four_valid_weeks_per_student() -> None:
    students, activity, checkins = generate_synthetic_data()
    snapshots = build_risk_snapshots(students, activity, checkins)
    assert len(snapshots) == 800
    assert snapshots.groupby("student_id")["week"].nunique().eq(4).all()
    score_columns = [
        "attendance_risk",
        "engagement_risk",
        "achievement_risk",
        "major_adaptation_risk",
        "career_risk",
        "overall_risk",
    ]
    assert snapshots[score_columns].ge(0).all().all()
    assert snapshots[score_columns].le(100).all().all()
