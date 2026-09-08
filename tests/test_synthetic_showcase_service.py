"""시연용 synthetic 대표 학생 선정 테스트."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

from src.data_generator import generate_synthetic_data
from src.risk_engine import build_risk_snapshots
from src.synthetic_showcase_service import (
    SHOWCASE_DEFINITIONS,
    build_synthetic_showcases,
)


def _build_default_showcases():
    """고정 시드 데이터로 대표 사례를 생성한다."""

    students, activity, checkins = generate_synthetic_data()
    snapshots = build_risk_snapshots(students, activity, checkins)
    return students, snapshots, build_synthetic_showcases(students, snapshots)


def test_showcases_cover_eight_distinct_synthetic_student_stories() -> None:
    _, _, showcases = _build_default_showcases()

    assert len(showcases) == 8
    assert len({case.student_id for case in showcases}) == 8
    assert [case.key for case in showcases] == [
        definition.key for definition in SHOWCASE_DEFINITIONS
    ]
    assert {case.archetype for case in showcases} == {
        "normal",
        "attendance_risk",
        "engagement_risk",
        "achievement_risk",
        "major_mismatch",
        "career_unclear",
        "complex_risk",
    }
    assert sum(case.archetype == "complex_risk" for case in showcases) == 2
    assert next(case for case in showcases if case.key == "complex_main").student_id == "S0003"


def test_single_risk_showcases_match_the_intended_primary_signal() -> None:
    _, _, showcases = _build_default_showcases()
    expected_types = {
        definition.key: definition.expected_primary_risk_type
        for definition in SHOWCASE_DEFINITIONS
        if definition.expected_primary_risk_type
    }

    for case in showcases:
        if case.key in expected_types:
            assert case.primary_risk_type == expected_types[case.key]
    stable = next(case for case in showcases if case.key == "stable")
    assert stable.risk_level == "정상"
    assert all(case.week == 4 for case in showcases)
    assert all(
        [week for week, _ in case.risk_history] == [1, 2, 3, 4]
        for case in showcases
    )
    assert all(
        case.period_risk_change
        == round(case.risk_history[-1][1] - case.risk_history[0][1], 2)
        for case in showcases
    )
    assert all(0 <= case.overall_risk <= 100 for case in showcases)
    assert all(case.natural_language_concern for case in showcases)


def test_showcase_selection_is_deterministic_and_excludes_sensitive_fields() -> None:
    students, snapshots, first = _build_default_showcases()
    second = build_synthetic_showcases(
        students.sample(frac=1, random_state=7),
        snapshots.sample(frac=1, random_state=8),
    )

    assert [case.student_id for case in first] == [
        case.student_id for case in second
    ]
    sensitive_fields = {
        "gender",
        "age",
        "nationality",
        "disability",
        "economic_status",
        "health_information",
        "religion",
        "political_orientation",
        "name",
        "student_number",
    }
    assert sensitive_fields.isdisjoint(asdict(first[0]))


def test_showcase_builder_reports_missing_snapshot_schema() -> None:
    students, _, _ = generate_synthetic_data(student_count=20)

    with pytest.raises(ValueError, match="snapshots 컬럼"):
        build_synthetic_showcases(students, pd.DataFrame({"student_id": []}))
