"""Kare synthetic 대화 품질 평가 서비스 테스트."""

from __future__ import annotations

import pytest

from src.ai import MockAIProvider
from src.kare_evaluation_service import (
    KARE_SYNTHETIC_EVALUATION_CASES,
    KareEvaluationCase,
    evaluate_kare_dialogues,
    kare_evaluation_catalog,
)


def test_kare_catalog_has_fifteen_balanced_synthetic_cases() -> None:
    """5개 적용 학과에 세 사례씩 배치하고 식별정보를 사용하지 않는다."""

    catalog = kare_evaluation_catalog()

    assert len(catalog) == 15
    assert set(catalog.groupby("department").size()) == {3}
    assert catalog["case_id"].is_unique
    combined_messages = " ".join(
        message
        for case in KARE_SYNTHETIC_EVALUATION_CASES
        for message in case.messages
    )
    assert "S000" not in combined_messages
    assert "학번" not in combined_messages
    assert "이름" not in combined_messages


def test_mock_kare_passes_repeatable_dialogue_quality_baseline() -> None:
    """Mock과 코드 안전장치는 15개 기준선에서 추론·반복·척도 위반 없이 동작한다."""

    report = evaluate_kare_dialogues(MockAIProvider())

    assert report.case_count == 15
    assert report.passed_case_count == 15
    assert report.case_pass_rate == 100.0
    assert report.state_match_rate == 100.0
    assert report.multi_extraction_recall == 100.0
    assert report.review_completion_rate == 100.0
    assert report.unexpected_inference_count == 0
    assert report.scale_question_count == 0
    assert report.repeated_focus_count == 0
    assert report.details["passed"].all()


def test_kare_evaluation_tracks_correction_and_ambiguous_non_inference() -> None:
    """정정은 최신 상태로 바꾸고 모호한 답은 긍정·부정으로 채우지 않는다."""

    selected = tuple(
        case
        for case in KARE_SYNTHETIC_EVALUATION_CASES
        if case.case_id in {"KARE-02", "KARE-14"}
    )
    report = evaluate_kare_dialogues(MockAIProvider(), selected)
    results = {result.case_id: result for result in report.results}

    assert results["KARE-02"].unexpected_inference_count == 0
    assert results["KARE-14"].matched_state_count == 1
    assert results["KARE-14"].passed


def test_kare_evaluation_rejects_duplicate_case_ids_or_unknown_dimensions() -> None:
    """평가 데이터 자체가 잘못되면 품질 수치를 만들지 않는다."""

    valid = KARE_SYNTHETIC_EVALUATION_CASES[0]
    invalid = KareEvaluationCase(
        "INVALID",
        "잘못된 영역",
        "간호학과",
        "테스트",
        ("여기까지 할게요.",),
        (("unknown_dimension", ("positive",)),),
    )

    with pytest.raises(ValueError, match="중복"):
        evaluate_kare_dialogues(MockAIProvider(), (valid, valid))
    with pytest.raises(ValueError, match="지원하지 않는 의미 영역"):
        evaluate_kare_dialogues(MockAIProvider(), (invalid,))
