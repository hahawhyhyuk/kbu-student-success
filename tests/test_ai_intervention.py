"""AI JSON 검증, fallback, DB 한정 지원프로그램 추천 테스트."""

from __future__ import annotations

import json
import urllib.error
from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from src.ai import (
    AIProvider,
    AIProviderError,
    GeminiProvider,
    MockAIProvider,
    ResilientAIProvider,
    create_ai_provider,
    validate_checkin_analysis,
    validate_kare_chat_reply,
)
from src.data_generator import generate_support_programs
from src.intervention_recommender import InterventionRecommender
from src.similarity import TokenOverlapSimilarityBackend
from src.checkin_semantic_state import empty_semantic_state
from src.utils import load_app_config, load_risk_config


MAJOR_CONCERN_TEXT = (
    "현재 전공이 생각했던 것과 다르고 콘텐츠 분야에도 관심이 있습니다. "
    "전공을 계속 공부해야 할지 고민됩니다."
)


@pytest.mark.parametrize(
    "threshold,expected_scores",
    [
        (69, [33.33, 33.33, 33.33]),
        (70, [50.0, 50.0, 0.0]),
        (80, [100.0, 0.0, 0.0]),
        (90, [100.0, 0.0, 0.0]),
    ],
)
def test_recommendation_scores_and_reasons_use_configured_risk_threshold(
    threshold: int, expected_scores: list[float],
) -> None:
    """보조 위험 경계값 변경이 추천 점수와 화면 설명에 함께 반영된다."""

    risk_config = deepcopy(load_risk_config())
    risk_config["risk_types"]["secondary_min"] = threshold
    app_config = deepcopy(load_app_config())
    app_config["recommendation"]["weights"] = {
        "risk_type_match": 1.0,
        "semantic_similarity": 0.0,
        "support_need_match": 0.0,
    }
    programs = generate_support_programs().iloc[:3].copy()
    programs["target_risk_types"] = ["attendance", "major_adaptation", "career"]
    profile = {
        "attendance_risk": 85,
        "major_adaptation_risk": 72,
        "career_risk": 69,
    }
    recommender = InterventionRecommender(
        similarity_backend=TokenOverlapSimilarityBackend(),
        config=app_config,
        risk_config=risk_config,
    )
    result = recommender.recommend(profile, {}, programs)
    reasons = recommender.explain_program_matches(profile, {}, programs)
    expected_by_id = dict(zip(programs["program_id"].astype(str), expected_scores))
    for recommendation in result.recommendations:
        expected_score = expected_by_id[recommendation.program_id]
        assert recommendation.score == pytest.approx(expected_score)
        assert recommendation.reason == reasons[recommendation.program_id]
        assert ("일치 영역 없음" in recommendation.reason) == (expected_score == 0)


def test_recommender_loads_risk_threshold_from_default_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """위험 설정을 주입하지 않아도 기본 YAML 로더의 값을 사용한다."""

    risk_config = deepcopy(load_risk_config())
    risk_config["risk_types"]["secondary_min"] = 80
    monkeypatch.setattr(
        "src.intervention_recommender.load_risk_config", lambda: risk_config
    )
    recommender = InterventionRecommender(TokenOverlapSimilarityBackend())
    assert recommender._active_risk_domains(
        {"attendance_risk": 90, "career_risk": 75}
    ) == {"attendance"}


class InvalidAIProvider(AIProvider):
    """Schema를 위반하는 테스트용 provider."""

    provider_name = "invalid"

    def analyze_checkin(self, text: str) -> dict[str, Any]:
        return {"raw": text}


class FakeHttpResponse:
    """urllib context manager와 동일한 최소 응답 객체."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_mock_provider_returns_schema_valid_analysis() -> None:
    result = MockAIProvider().analyze_checkin(MAJOR_CONCERN_TEXT)

    assert result["major_concern"] is True
    assert "콘텐츠·디자인" in result["interests"]
    assert validate_checkin_analysis(result) == result


def test_mock_analysis_does_not_treat_counseling_request_as_interest() -> None:
    """상담 요청은 상담·복지 관심 분야로 잘못 분류하지 않는다."""

    result = MockAIProvider().analyze_checkin(
        "수업을 따라가기 버거워서 상담도 받고 싶어요."
    )

    assert "상담·복지" not in result["interests"]
    assert "학습 지원" in result["support_needs"]


def test_schema_rejects_unknown_or_missing_ai_fields() -> None:
    with pytest.raises(AIProviderError, match="JSON Schema"):
        validate_checkin_analysis({"summary": "불완전한 결과", "raw": "노출 금지"})


def test_resilient_provider_falls_back_when_primary_json_is_invalid() -> None:
    provider = ResilientAIProvider(
        primary=InvalidAIProvider(), fallback=MockAIProvider()
    )

    result = provider.analyze_checkin(MAJOR_CONCERN_TEXT)

    assert result["major_concern"] is True
    assert provider.last_provider_name == "mock"
    assert provider.last_error is not None
    assert "raw" not in result


def test_factory_uses_mock_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = create_ai_provider()

    assert isinstance(provider, MockAIProvider)


def test_gemini_provider_parses_and_validates_structured_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = MockAIProvider().analyze_checkin(MAJOR_CONCERN_TEXT)
    api_payload = {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(expected, ensure_ascii=False)}]}}
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeHttpResponse(api_payload),
    )

    result = GeminiProvider(api_key="test-key").analyze_checkin(
        MAJOR_CONCERN_TEXT
    )

    assert result == expected


def test_gemini_microdegree_descriptions_use_one_structured_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """여러 교육과정 후보를 한 번의 Gemini 요청으로 설명한다."""

    candidates = [
        {
            "student_count": 8,
            "department_count": 2,
            "course_names": ["데이터분석", "보건통계"],
            "competencies": [],
            "related_jobs": [],
        },
        {
            "student_count": 5,
            "department_count": 1,
            "course_names": ["서비스기획", "사용자경험"],
            "competencies": [],
            "related_jobs": [],
        },
    ]
    mock = MockAIProvider()
    descriptions = [
        {
            "candidate_index": index,
            **mock.describe_microdegree_candidate(candidate),
        }
        for index, candidate in enumerate(candidates)
    ]
    api_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {"candidates": descriptions},
                                ensure_ascii=False,
                            )
                        }
                    ]
                }
            }
        ]
    }
    call_count = 0
    captured_timeout = 0
    captured_payload: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHttpResponse:
        nonlocal call_count, captured_timeout, captured_payload
        call_count += 1
        captured_timeout = timeout
        captured_payload = json.loads(request.data.decode("utf-8"))
        return FakeHttpResponse(api_payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = GeminiProvider(api_key="test-key").describe_microdegree_candidates(
        candidates
    )

    assert call_count == 1
    assert captured_timeout == 60
    assert captured_payload["generationConfig"]["maxOutputTokens"] == 8192
    assert len(result) == 2
    assert all(item["status"] == "교육과정 검토 후보" for item in result)


def test_gemini_error_names_microdegree_operation_on_http_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429 오류가 체크인이 아닌 실제 교육과정 설명 작업을 가리킨다."""

    def raise_rate_limit(request: Any, timeout: int) -> FakeHttpResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_rate_limit)
    with pytest.raises(
        AIProviderError,
        match="교육과정 후보 설명에 실패했습니다 \\(HTTP 429\\)",
    ):
        GeminiProvider(api_key="test-key").describe_microdegree_candidates(
            [{"course_names": ["데이터분석"]}]
        )


def test_gemini_microdegree_timeout_reports_bulk_request_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """대량 후보 설명의 확장된 제한 시간과 실패 원인을 안내한다."""

    def raise_timeout(request: Any, timeout: int) -> FakeHttpResponse:
        assert timeout == 60
        raise TimeoutError

    monkeypatch.setattr("urllib.request.urlopen", raise_timeout)
    with pytest.raises(
        AIProviderError,
        match=r"교육과정 후보 설명에 실패했습니다 \(시간 초과: 60초\)",
    ):
        GeminiProvider(api_key="test-key").describe_microdegree_candidates(
            [{"course_names": ["데이터분석"]}]
        )


def test_gemini_chat_uses_persona_and_validated_anonymous_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """자유대화 요청은 시스템 페르소나를 사용하고 학생 ID 없이 구조화된다."""

    conversation_context = {
        "active_thread": "learning",
        "question_intent": "clarify_scene",
        "next_focus": "learning_difficulty",
        "fallback_question": "수업에서 언제 가장 자주 막히나요?",
    }
    expected = MockAIProvider().generate_kare_reply(
        history=[],
        user_message="요즘 과제가 자꾸 밀려서 걱정이에요.",
        semantic_state=empty_semantic_state(),
        asked_dimensions=[],
        allowed_interest_fields=["데이터·AI", "보건"],
        conversation_context=conversation_context,
    )
    api_payload = {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(expected, ensure_ascii=False)}]}}
        ]
    }
    captured_request: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: int) -> FakeHttpResponse:
        captured_request["payload"] = json.loads(request.data.decode("utf-8"))
        captured_request["timeout"] = timeout
        return FakeHttpResponse(api_payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = GeminiProvider(api_key="test-key").generate_kare_reply(
        history=[{"role": "assistant", "content": "편하게 이야기해 주세요."}],
        user_message="요즘 과제가 자꾸 밀려서 걱정이에요.",
        semantic_state=empty_semantic_state(),
        asked_dimensions=[],
        allowed_interest_fields=["데이터·AI", "보건"],
        conversation_context=conversation_context,
    )

    assert validate_kare_chat_reply(result) == expected
    request_payload = captured_request["payload"]
    assert "systemInstruction" in request_payload
    serialized_payload = json.dumps(request_payload, ensure_ascii=False)
    assert "학생성공 파트너 Kare" in serialized_payload
    assert "[ROLE AND PERSONA]" in serialized_payload
    assert "[FEW-SHOT EXAMPLES]" in serialized_payload
    assert "[CURRENT CONTEXT]" in serialized_payload
    assert "모호한 표현을 상태로 과대 추정하지 않음" in serialized_payload
    assert "career_clarity" in serialized_payload
    assert "natural_language_concern" in serialized_payload
    assert "clarify_scene" in serialized_payload
    assert "active_thread" in serialized_payload
    assert "student_id" not in serialized_payload


def test_recommender_returns_three_unique_existing_program_ids() -> None:
    programs = generate_support_programs()
    analysis = MockAIProvider().analyze_checkin(
        "과제와 진도를 따라가기 어렵고 공부가 버겁습니다."
    )
    profile = {
        "attendance_risk": 10,
        "engagement_risk": 82,
        "achievement_risk": 68,
        "major_adaptation_risk": 20,
        "career_risk": 15,
        "is_complex": True,
        "natural_language_concern": "과제와 진도를 따라가기 어렵습니다.",
        "interest_fields": "데이터·AI",
        "desired_job": "",
    }
    recommender = InterventionRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    )

    result = recommender.recommend(profile, analysis, programs)

    recommended_ids = [item.program_id for item in result.recommendations]
    assert len(recommended_ids) == 3
    assert len(set(recommended_ids)) == 3
    assert set(recommended_ids).issubset(set(programs["program_id"]))
    assert all(0 <= item.score <= 100 for item in result.recommendations)
    assert all(item.reason for item in result.recommendations)
    assert all("위험영역 일치:" in item.reason for item in result.recommendations)
    assert all(
        "대화·프로그램 설명 의미 유사도 참고값:" in item.reason
        for item in result.recommendations
    )
    assert all(
        "체크인 지원수요 일치:" in item.reason
        for item in result.recommendations
    )
    visible_programs = programs[
        programs["program_id"].isin(recommended_ids)
    ]
    refreshed_reasons = recommender.explain_program_matches(
        profile,
        analysis,
        visible_programs,
    )
    assert set(refreshed_reasons) == set(recommended_ids)
    assert all(
        "대화·프로그램 설명 의미 유사도 참고값:" in reason
        for reason in refreshed_reasons.values()
    )


def test_recommender_respects_configured_top_k() -> None:
    programs = generate_support_programs()
    analysis = MockAIProvider().analyze_checkin(MAJOR_CONCERN_TEXT)
    profile = {
        "attendance_risk": 5,
        "engagement_risk": 10,
        "achievement_risk": 15,
        "major_adaptation_risk": 90,
        "career_risk": 75,
        "is_complex": True,
        "natural_language_concern": MAJOR_CONCERN_TEXT,
        "interest_fields": "콘텐츠·디자인",
        "desired_job": "콘텐츠 기획",
    }

    result = InterventionRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    ).recommend(profile, analysis, programs, top_k=2)

    assert len(result.recommendations) == 2


def test_program_ranking_prefers_purpose_and_home_department_over_keyword_noise() -> None:
    """단일 키워드 열거와 타 학과 프로그램이 직접 지원보다 앞서지 않는다."""

    programs = pd.DataFrame(
        [
            {
                "program_id": "P-AI-1",
                "program_name": "AI 해커톤대회",
                "program_type": "공모전/경진대회",
                "description": "AI 기반 문제해결과 데이터 분석 실무를 경험합니다.",
                "target_risk_types": "engagement|achievement",
                "provided_competencies": "문제해결",
                "target_students": "재학생",
                "department_in_charge": "AI디지털트윈연구원",
                "operation_period": "상시",
            },
            {
                "program_id": "P-AI-2",
                "program_name": "AI 해커톤대회",
                "program_type": "공모전/경진대회",
                "description": "AI 기반 문제해결과 데이터 분석 실무를 경험합니다.",
                "target_risk_types": "engagement|achievement",
                "provided_competencies": "문제해결",
                "target_students": "재학생",
                "department_in_charge": "AI디지털트윈연구원",
                "operation_period": "다음 학기",
            },
            {
                "program_id": "P-LEARN",
                "program_name": "과제·수업 학습 튜터링",
                "program_type": "학습지원",
                "description": "밀린 과제와 수업 진도를 따라가도록 튜터링합니다.",
                "target_risk_types": "engagement|achievement",
                "provided_competencies": "학습관리",
                "target_students": "재학생",
                "department_in_charge": "교수학습혁신센터",
                "operation_period": "상시",
            },
            {
                "program_id": "P-HOME",
                "program_name": "[간호학과] 진로 멘토링",
                "program_type": "진로지원",
                "description": "간호 전공과 직무를 탐색하는 멘토링입니다.",
                "target_risk_types": "major_adaptation|career",
                "provided_competencies": "진로설계",
                "target_students": "간호학과 재학생",
                "department_in_charge": "간호학과",
                "operation_period": "상시",
            },
            {
                "program_id": "P-NOISE",
                "program_name": "경복대학교 오픈캠퍼스 데이",
                "program_type": "홍보대사",
                "description": "보건의료, AI, 로봇, 뷰티 등 다양한 체험을 운영합니다.",
                "target_risk_types": "engagement|major_adaptation|career",
                "provided_competencies": "홍보",
                "target_students": "재학생",
                "department_in_charge": "서비스경영과",
                "operation_period": "상시",
            },
            {
                "program_id": "P-MISMATCH",
                "program_name": "[의료미용과] AI 피부 진단",
                "program_type": "학과행사",
                "description": "피부진단기기를 활용한 전공 체험입니다.",
                "target_risk_types": "engagement|major_adaptation",
                "provided_competencies": "전공실무",
                "target_students": "재학생",
                "department_in_charge": "의료미용과",
                "operation_period": "상시",
            },
        ]
    )
    profile = {
        "department": "간호학과",
        "attendance_risk": 72,
        "engagement_risk": 85,
        "achievement_risk": 60,
        "major_adaptation_risk": 45,
        "career_risk": 55,
        "natural_language_concern": "과제와 수업이 어렵지만 데이터 AI도 배우고 싶어요.",
        "interest_fields": "데이터·AI",
        "desired_job": "데이터분석가",
    }
    analysis = {
        "summary": "학습 지원과 데이터 AI 분야 탐색을 원함",
        "support_needs": ["학습 지원", "진로 탐색"],
        "interests": ["데이터·AI"],
        "desired_jobs": ["데이터분석가"],
    }
    recommender = InterventionRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    )

    result = recommender.recommend(profile, analysis, programs)
    selected_ids = {item.program_id for item in result.recommendations}
    selected_names = [item.program_name for item in result.recommendations]
    reasons = recommender.explain_program_matches(profile, analysis, programs)

    assert selected_ids == {"P-AI-1", "P-LEARN", "P-HOME"}
    assert len(selected_names) == len(set(selected_names))
    assert "관심분야 직접 연결: 직접 연결 없음" in reasons["P-NOISE"]
    assert "P-NOISE" not in selected_ids
    assert "P-MISMATCH" not in selected_ids
