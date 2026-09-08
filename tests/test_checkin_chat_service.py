"""Kare 자유대화·종료·fallback 테스트."""

from __future__ import annotations

from typing import Any

import pytest

from src.ai import AIProviderError, MockAIProvider, ResilientAIProvider
from src.checkin_chat_service import CheckinChatService
from src.checkin_conversation import (
    CHAT_OBSERVATION_FIELDS,
    empty_chat_values,
)


ALLOWED_INTERESTS = (
    "데이터·AI",
    "경영·마케팅",
    "콘텐츠·디자인",
    "서비스",
    "상담·복지",
    "보건",
)


def test_free_chat_keeps_numeric_fields_empty_and_tracks_semantic_evidence() -> None:
    """여러 턴의 맥락은 점수 대신 근거 있는 의미 상태로 남긴다."""

    service = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS)
    values = empty_chat_values()
    covered: tuple[str, ...] = ()
    history: list[dict[str, str]] = []
    focus: str | None = None
    messages = (
        "전공 흥미는 2점이에요.",
        "전공 만족도도 2점이에요.",
        "전공 지속 의향은 2점 정도예요.",
        "학습 어려움은 4점이에요.",
        "진로 명확성은 2점이에요.",
        "상담 필요 정도는 5점이고 상담을 받고 싶어요.",
        "데이터와 서비스 분야에 관심 있어요.",
        "서비스 기획을 하고 싶어요.",
        "전공 선택이 맞는지 고민이고 진로 방향이 막막해요.",
    )

    for message in messages:
        result = service.process_turn(
            history=history,
            values=values,
            covered_fields=covered,
            user_message=message,
            current_focus=focus,
        )
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result.assistant_message},
            ]
        )
        values = result.values
        covered = result.covered_fields
        focus = result.next_focus

    assert result.ready_for_review
    assert "consultation_intent" in covered
    assert "interest_fields" in covered
    assert result.semantic_state["consultation_intent"]["state"] == "wants_support"
    assert all(values[field] is None for field in CHAT_OBSERVATION_FIELDS)
    transcript = service.build_student_transcript(history)
    assert "전공 흥미는 2점" in transcript
    assert "전공 선택이 맞는지" in transcript


def test_one_message_keeps_numeric_values_empty_and_tracks_multiple_meanings() -> None:
    """한 문장의 여러 정보도 점수 없이 의미 영역으로 연결한다."""

    result = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message=(
            "전공 흥미는 2점이고 학습 어려움은 4점이에요. "
            "진로가 막막해서 고민이에요."
        ),
    )

    assert result.values["major_interest"] is None
    assert result.values["learning_difficulty"] is None
    assert result.values["career_clarity"] is None
    assert "career_clarity" in result.covered_fields
    assert "natural_language_concern" in result.covered_fields
    assert not result.ready_for_review


def test_rich_first_message_does_not_force_immediate_structured_completion() -> None:
    """정보가 많은 첫 발화도 바로 슬롯 완료로 처리하지 않는다."""

    result = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message=(
            "전공 흥미는 2점, 전공 만족도는 3점, 계속 공부할 마음은 2점이에요. "
            "학습 어려움은 4점이고 진로 명확성은 2점, 상담 필요 정도는 4점이에요. "
            "데이터와 AI, 서비스 분야에 관심 있고 서비스 기획을 하고 싶어요. "
            "과제를 따라가기 벅차고 진로 방향이 고민이라 상담을 받고 싶어요."
        ),
    )

    assert not result.ready_for_review
    assert result.semantic_state["consultation_intent"]["state"] == "wants_support"
    assert all(result.values[field] is None for field in CHAT_OBSERVATION_FIELDS)


def test_interest_and_job_wait_for_end_of_conversation_analysis() -> None:
    """관심·직무의 의미 유무는 추적하되 최종 master 값은 전체 분석 후 반영한다."""

    result = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message=(
            "경영·마케팅에 관심 있어요. "
            "희망 직무는 서비스 기획이에요."
        ),
    )

    assert result.values["interest_fields"] is None
    assert result.values["desired_job"] is None
    assert result.semantic_state["interest_fields"]["state"] == "identified"
    assert result.semantic_state["desired_job"]["state"] == "identified"


def test_consultation_request_is_not_classified_during_free_chat() -> None:
    """상담 요청도 전체 대화 분석 전에는 관심 분야로 분류하지 않는다."""

    result = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message="이대로 따라갈 수 있을지 걱정돼서 상담도 한번 받아보고 싶어요.",
        current_focus="interest_fields",
    )

    assert result.values["interest_fields"] is None
    assert result.values["consultation_requested"] is False
    assert result.semantic_state["consultation_intent"]["state"] == "wants_support"


def test_ambiguous_language_does_not_guess_scale_scores() -> None:
    """정도의 경계가 불분명한 자연어는 1~5 점수로 임의 환산하지 않는다."""

    result = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message=(
            "전공은 요즘 그냥 그런 편이고 수업도 가끔 벅차요. "
            "진로는 생각 중이지만 아직 확실하지 않아요."
        ),
    )

    assert result.values["major_interest"] is None
    assert result.values["learning_difficulty"] is None
    assert result.values["career_clarity"] is None
    assert result.semantic_state["career_clarity"]["state"] in {
        "exploring",
        "not_clear",
        "undetermined",
    }
    assert result.next_focus in {*CHAT_OBSERVATION_FIELDS, "review"}


def test_student_can_correct_an_explicit_scale_score_in_chat() -> None:
    """같은 발화에서 이전 숫자를 정정하면 '아니라' 뒤의 점수를 사용한다."""

    result = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message="전공 흥미는 2점이 아니라 4점이에요.",
    )

    assert result.values["major_interest"] is None
    assert result.semantic_state["major_interest"]["state"] == "unknown"


class BrokenChatProvider(MockAIProvider):
    """외부 대화 장애를 재현하는 provider."""

    provider_name = "broken"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        raise AIProviderError("chat unavailable")


class OverconfidentAdvancingProvider(MockAIProvider):
    """첫 발화부터 검토 가능하다고 판단하는 AI를 재현한다."""

    provider_name = "overconfident"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        result = super().generate_kare_reply(**kwargs)
        result["suggest_review"] = True
        return result


class RepeatingNaturalQuestionProvider(MockAIProvider):
    """자유대화에서 척도 질문을 생성하는 잘못된 AI를 재현한다."""

    provider_name = "repeating"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "assistant_message": "전공 흥미를 1점부터 5점 중 몇 점으로 생각하나요?",
            "dimension_updates": [],
            "next_focus": "major_interest",
            "suggest_review": False,
        }


class AlwaysContinuingProvider(MockAIProvider):
    """충분한 대화 후에도 질문을 계속하는 AI를 재현한다."""

    provider_name = "continuing"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "assistant_message": "그 이야기를 조금 더 들려줄래요?",
            "dimension_updates": [],
            "next_focus": "natural_language_concern",
            "suggest_review": False,
        }


class NarrowGroupedQuestionProvider(MockAIProvider):
    """묶음 주제 중 일부만 질문하는 AI를 재현한다."""

    provider_name = "narrow"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "assistant_message": "전공 공부에 흥미를 느끼나요?",
            "dimension_updates": [],
            "next_focus": "major_interest",
            "suggest_review": False,
        }


class ContextualNarrowQuestionProvider(MockAIProvider):
    """맥락 반영은 자연스럽지만 묶음 질문 일부를 누락하는 AI를 재현한다."""

    provider_name = "contextual_narrow"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "assistant_message": (
                "전공에는 재미를 느끼지만 수업과 과제가 부담이군요. "
                "현재 전공을 계속 공부하고 싶은 마음은 어떤가요?"
            ),
            "dimension_updates": [],
            "next_focus": "major_continuation_intent",
            "suggest_review": False,
        }


class OverinterpretingProvider(MockAIProvider):
    """학생이 말하지 않은 강한 감정을 덧붙이는 AI를 재현한다."""

    provider_name = "overinterpreting"

    def generate_kare_reply(self, **kwargs: Any) -> dict[str, Any]:
        context = dict(kwargs["conversation_context"])
        return {
            "assistant_message": (
                "매일 혼자 마음고생하며 버티고 있었겠어요. "
                "수업에서 어떤 점이 어려운가요?"
            ),
            "dimension_updates": [],
            "next_focus": context["next_focus"],
            "suggest_review": False,
        }


def test_ambiguous_natural_reply_is_not_forced_into_numeric_scale() -> None:
    """모호한 자연어 응답에 1~5 점수를 요구하지 않는다."""

    result = CheckinChatService(
        RepeatingNaturalQuestionProvider(), ALLOWED_INTERESTS
    ).process_turn(
        history=[
            {"role": "assistant", "content": "안녕하세요, Kare예요."},
            {"role": "user", "content": "학교생활부터 이야기할게요."},
            {"role": "assistant", "content": "전공 공부는 요즘 어떠신가요?"},
        ],
        values=empty_chat_values(),
        covered_fields=[],
        user_message="나쁘지는 않아요. 재밌는 편이에요.",
        current_focus="major_interest",
    )

    assert result.next_focus == "major_satisfaction"
    assert result.values["major_interest"] is None
    assert "1점" not in result.assistant_message
    assert "5점" not in result.assistant_message
    assert "재미있거나 잘 맞는" in result.assistant_message


def test_unsupported_emotional_assumption_is_not_shown_to_student() -> None:
    """학생이 표현하지 않은 고립·고통 서술은 안전한 사실 반영으로 교체한다."""

    result = CheckinChatService(
        OverinterpretingProvider(), ALLOWED_INTERESTS
    ).process_turn(
        history=[],
        values=empty_chat_values(),
        user_message="수업 설명이 빨라요.",
    )

    assert "매일" not in result.assistant_message
    assert "혼자 마음고생" not in result.assistant_message
    assert "수업" in result.assistant_message or "과제" in result.assistant_message
    assert result.assistant_message.count("?") == 1


def test_model_question_that_ignores_the_active_story_is_repaired() -> None:
    """모델이 대화 계획을 벗어나면 현재 이야기 질문 하나로 교체한다."""

    result = CheckinChatService(
        NarrowGroupedQuestionProvider(), ALLOWED_INTERESTS
    ).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message="학교생활부터 이야기할게요.",
    )

    assert "가장 신경 쓰이는 일" in result.assistant_message
    assert result.assistant_message.count("?") == 1
    assert result.asked_dimensions == ("natural_language_concern",)


def test_question_repair_preserves_reflection_and_current_story() -> None:
    """모델이 다른 주제로 옮겨도 반영은 남기고 학습 이야기를 이어간다."""

    result = CheckinChatService(
        ContextualNarrowQuestionProvider(), ALLOWED_INTERESTS
    ).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message="전공은 재미있지만 수업이 빠르고 과제가 밀려요.",
        asked_dimensions=(
            "natural_language_concern",
            "major_interest",
            "major_satisfaction",
            "learning_difficulty",
        ),
    )

    assert "전공에는 재미를 느끼지만 수업과 과제가 부담이군요." in (
        result.assistant_message
    )
    assert "과제를 시작하기 어려운 건" in result.assistant_message
    assert "앞으로의 진로 방향" not in result.assistant_message
    assert result.assistant_message.count("?") == 1


def test_realistic_narrative_adapts_until_context_is_sufficient() -> None:
    """자연어 맥락이 충분해질 때까지 척도 없이 질문을 이어간다."""

    service = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS)
    values = empty_chat_values()
    covered: tuple[str, ...] = ()
    history: list[dict[str, str]] = []
    focus: str | None = None
    asked: tuple[str, ...] = ()
    messages = (
        "요즘 수업 말이 빠르고 외울 게 많아서 과제가 밀리고 있어요.",
        "전공은 재미있지만 이대로 따라갈 수 있을지 걱정돼요.",
        "데이터와 보건 분야가 관심 있고 데이터분석가가 되고 싶어요.",
        "학습 상담을 받아서 밀린 과제부터 해결하고 싶어요.",
    )

    results = []
    for message in messages:
        result = service.process_turn(
            history=history,
            values=values,
            covered_fields=covered,
            user_message=message,
            current_focus=focus,
            asked_dimensions=asked,
        )
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result.assistant_message},
            ]
        )
        values = result.values
        covered = result.covered_fields
        focus = result.next_focus
        asked = result.asked_dimensions
        results.append(result)

    assert not results[0].ready_for_review
    assert not results[1].ready_for_review
    assert not results[2].ready_for_review
    assert result.ready_for_review
    assert result.next_focus == "review"
    assert len(result.covered_fields) >= 7
    assert result.values["major_interest"] is None
    assert result.values["learning_difficulty"] is None
    assert result.semantic_state["major_satisfaction"]["state"] in {
        "unknown",
        "undetermined",
    }
    assert result.semantic_state["major_continuation_intent"]["state"] in {
        "unknown",
        "undetermined",
    }
    assert result.semantic_state["learning_difficulty"]["state"] == "high_difficulty"
    assert result.semantic_state["consultation_intent"]["state"] == "wants_support"
    assert all("1점" not in item.assistant_message for item in results)
    assert result.values["interest_fields"] is None
    assert result.values["desired_job"] is None
    transcript = service.build_student_transcript(history)
    assert "과제가 밀리고" in transcript
    assert "데이터분석가" in transcript


def test_conversation_does_not_force_finish_after_three_substantive_turns() -> None:
    """모델이 질문을 계속해도 3턴을 고정 종료 기준으로 사용하지 않는다."""

    service = CheckinChatService(AlwaysContinuingProvider(), ALLOWED_INTERESTS)
    history: list[dict[str, str]] = []
    for message in (
        "수업이 빨라서 과제가 밀리고 있어요.",
        "전공은 재미있지만 따라갈 수 있을지 걱정돼요.",
        "데이터 분야에 관심이 있고 분석가가 되고 싶어요.",
    ):
        result = service.process_turn(history=history, user_message=message)
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result.assistant_message},
            ]
        )

    assert not result.ready_for_review
    assert result.next_focus != "review"


def test_meta_opening_and_two_no_content_replies_end_without_repeating_topics() -> None:
    """시작 신호 뒤에 '없어'가 반복되면 같은 주제를 묻지 않고 확인으로 간다."""

    service = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS)
    history: list[dict[str, str]] = []
    results = []
    for message in (
        "요즘 학교생활부터 이야기할게요",
        "없어",
        "없어",
    ):
        result = service.process_turn(history=history, user_message=message)
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result.assistant_message},
            ]
        )
        results.append(result)

    assert not results[0].ready_for_review
    assert not results[1].ready_for_review
    assert results[2].ready_for_review
    assert "관심 분야" not in results[1].assistant_message
    assert "희망 직무" not in results[1].assistant_message
    assert service.build_student_transcript(history) == (
        "현재 특별히 이야기하고 싶은 고민이나 요청은 없습니다."
    )


def test_chat_failure_falls_back_to_local_mock() -> None:
    provider = ResilientAIProvider(BrokenChatProvider(), MockAIProvider())
    result = CheckinChatService(provider, ALLOWED_INTERESTS).process_turn(
        history=[],
        values=empty_chat_values(),
        covered_fields=[],
        user_message="전공 흥미는 4점이에요.",
    )

    assert result.values["major_interest"] is None
    assert result.provider_name == "mock"
    assert result.fallback_used
    assert result.warning is not None


def test_chat_rejects_blank_or_oversized_message() -> None:
    service = CheckinChatService(MockAIProvider(), ALLOWED_INTERESTS)

    with pytest.raises(ValueError, match="입력"):
        service.process_turn(
            history=[], values=empty_chat_values(), covered_fields=[], user_message=""
        )
    with pytest.raises(ValueError, match="1,500자"):
        service.process_turn(
            history=[],
            values=empty_chat_values(),
            covered_fields=[],
            user_message="가" * 1501,
        )
