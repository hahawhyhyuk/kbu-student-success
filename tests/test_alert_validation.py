"""초기경보 참조 라벨 비교 서비스와 Streamlit 화면 테스트."""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.alert_validation_service import (
    build_synthetic_reference_labels,
    evaluate_initial_alerts,
)
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService
from src.utils import PROJECT_ROOT


def test_synthetic_alert_validation_reports_internal_agreement() -> None:
    repository = get_default_repository()
    source = repository.get_all()
    snapshots = RiskAnalysisService(repository).build_snapshots(source)
    references = build_synthetic_reference_labels(source.students)

    report = evaluate_initial_alerts(snapshots, references)

    assert report.student_count == 200
    assert report.reference_support_count == 110
    assert report.predicted_support_count == 110
    assert report.agreement_rate == 100.0
    assert report.support_signal_recall == 100.0
    assert report.normal_pattern_specificity == 100.0
    assert report.high_risk_recall == 100.0
    assert report.primary_type_match_rate == 91.7


def test_alert_validation_counts_false_positive_and_false_negative() -> None:
    references = pd.DataFrame(
        [
            {
                "student_id": "S1",
                "reference_segment": "정상형",
                "expected_support_needed": False,
                "expected_high_risk": False,
                "expected_primary_risk_type": None,
            },
            {
                "student_id": "S2",
                "reference_segment": "출결위험형",
                "expected_support_needed": True,
                "expected_high_risk": False,
                "expected_primary_risk_type": "출결위험형",
            },
            {
                "student_id": "S3",
                "reference_segment": "복합위험형",
                "expected_support_needed": True,
                "expected_high_risk": True,
                "expected_primary_risk_type": None,
            },
            {
                "student_id": "S4",
                "reference_segment": "학업부진형",
                "expected_support_needed": True,
                "expected_high_risk": False,
                "expected_primary_risk_type": "학업부진형",
            },
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "student_id": "S1",
                "week": 4,
                "risk_level": "관심",
                "primary_risk_type": "진로미설정형",
            },
            {
                "student_id": "S2",
                "week": 4,
                "risk_level": "정상",
                "primary_risk_type": "출결위험형",
            },
            {
                "student_id": "S3",
                "week": 4,
                "risk_level": "고위험",
                "primary_risk_type": "출결위험형",
            },
            {
                "student_id": "S4",
                "week": 4,
                "risk_level": "주의",
                "primary_risk_type": "진로미설정형",
            },
        ]
    )

    report = evaluate_initial_alerts(snapshots, references)

    assert report.agreement_rate == 50.0
    assert report.support_signal_recall == 66.7
    assert report.normal_pattern_specificity == 0.0
    assert report.high_risk_recall == 100.0
    assert report.primary_type_match_rate == 0.0
    assert report.confusion.to_dict(orient="records") == [
        {
            "reference_group": "지원 불필요",
            "predicted_normal": 0,
            "predicted_support": 1,
        },
        {
            "reference_group": "지원 필요",
            "predicted_normal": 1,
            "predicted_support": 2,
        },
    ]


def test_alert_validation_rejects_reference_without_prediction() -> None:
    references = pd.DataFrame(
        [
            {
                "student_id": "MISSING",
                "reference_segment": "정상형",
                "expected_support_needed": False,
                "expected_high_risk": False,
                "expected_primary_risk_type": None,
            }
        ]
    )
    snapshots = pd.DataFrame(
        columns=["student_id", "week", "risk_level", "primary_risk_type"]
    )

    with pytest.raises(ValueError, match="초기경보 결과가 없습니다"):
        evaluate_initial_alerts(snapshots, references)


def test_alert_validation_page_renders_with_synthetic_limit() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "11_alert_validation.py")
    ).run(timeout=30)

    assert not app.exception
    assert any(
        metric.label == "합성 기준 일치율" and metric.value == "100.0%"
        for metric in app.metric
    )
    assert any(
        "독립적인 실제 학생 데이터의 예측 정확도" in item.value
        for item in app.warning
    )
    assert any(
        metric.label == "대화 사례" and metric.value == "15개"
        for metric in app.metric
    )
    assert any("Kare 대화 품질 회귀 검증" in item.label for item in app.expander)
