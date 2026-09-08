"""실제 개인정보 없이 Kare 대화 품질을 반복 점검하는 평가 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.ai.base import AIProvider
from src.checkin_chat_service import CheckinChatService
from src.checkin_conversation import empty_chat_values
from src.checkin_semantic_state import SEMANTIC_DIMENSIONS, empty_semantic_state


DEFAULT_INTEREST_FIELDS: tuple[str, ...] = (
    "데이터·AI",
    "경영·마케팅",
    "콘텐츠·디자인",
    "서비스",
    "상담·복지",
    "보건",
)


@dataclass(frozen=True)
class KareEvaluationCase:
    """synthetic 대화 한 사례와 기대 의미 상태."""

    case_id: str
    title: str
    department: str
    tone: str
    messages: tuple[str, ...]
    expected_states: tuple[tuple[str, tuple[str, ...]], ...]
    first_turn_dimensions: tuple[str, ...] = ()
    forbidden_dimensions: tuple[str, ...] = ()
    expect_review: bool = True


@dataclass(frozen=True)
class KareEvaluationCaseResult:
    """대화 사례 하나의 자동 평가 결과."""

    case_id: str
    title: str
    department: str
    tone: str
    turn_count: int
    expected_state_count: int
    matched_state_count: int
    first_turn_expected_count: int
    first_turn_matched_count: int
    unexpected_inference_count: int
    scale_question_count: int
    repeated_focus_count: int
    ready_for_review: bool
    provider_name: str
    passed: bool
    mismatches: tuple[str, ...]
    transcript: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True)
class KareEvaluationReport:
    """여러 synthetic 대화의 품질 지표와 사례별 상세."""

    case_count: int
    passed_case_count: int
    expected_state_count: int
    matched_state_count: int
    first_turn_expected_count: int
    first_turn_matched_count: int
    unexpected_inference_count: int
    scale_question_count: int
    repeated_focus_count: int
    expected_review_count: int
    completed_review_count: int
    details: pd.DataFrame
    results: tuple[KareEvaluationCaseResult, ...]

    @property
    def case_pass_rate(self) -> float:
        """전체 사례 중 모든 기대조건을 만족한 비율."""

        return _percentage(self.passed_case_count, self.case_count)

    @property
    def state_match_rate(self) -> float:
        """명시한 의미 상태 기대값의 일치 비율."""

        return _percentage(self.matched_state_count, self.expected_state_count)

    @property
    def multi_extraction_recall(self) -> float:
        """첫 발화에 함께 담긴 여러 영역을 찾은 비율."""

        return _percentage(
            self.first_turn_matched_count,
            self.first_turn_expected_count,
        )

    @property
    def review_completion_rate(self) -> float:
        """확인 화면 전환을 기대한 사례의 종료 성공률."""

        return _percentage(self.completed_review_count, self.expected_review_count)


KARE_SYNTHETIC_EVALUATION_CASES: tuple[KareEvaluationCase, ...] = (
    KareEvaluationCase(
        "KARE-01",
        "전공 흥미와 밀린 과제",
        "친환경건축과",
        "구체적",
        (
            "전공은 재미있고 만족하지만 과제가 밀려서 걱정돼요.",
            "전공은 계속 공부하고 싶고 진로는 친환경 설계 쪽으로 생각 중이에요.",
            "과제 정리 상담을 받고 싶어요. 오늘은 여기까지 이야기할게요.",
        ),
        (
            ("major_interest", ("positive",)),
            ("major_satisfaction", ("positive",)),
            ("major_continuation_intent", ("continuing",)),
            ("learning_difficulty", ("high_difficulty",)),
            ("career_clarity", ("exploring",)),
            ("consultation_intent", ("wants_support",)),
            ("natural_language_concern", ("identified",)),
        ),
        (
            "major_interest",
            "major_satisfaction",
            "learning_difficulty",
            "natural_language_concern",
        ),
    ),
    KareEvaluationCase(
        "KARE-02",
        "애매한 전공 답변",
        "친환경건축과",
        "짧고 모호함",
        ("전공은 그냥 적당한 편이에요.", "오늘은 여기까지 할게요."),
        (),
        forbidden_dimensions=(
            "major_interest",
            "major_satisfaction",
            "learning_difficulty",
        ),
    ),
    KareEvaluationCase(
        "KARE-03",
        "안정적인 학습과 상담 비희망",
        "친환경건축과",
        "명확함",
        (
            "전공은 재미있고 수업도 할만해요. 상담은 필요 없어요.",
            "특별히 고민 없어요. 여기까지 이야기할게요.",
        ),
        (
            ("major_interest", ("positive",)),
            ("learning_difficulty", ("low_difficulty",)),
            ("consultation_intent", ("no_support",)),
            ("natural_language_concern", ("none",)),
        ),
        (
            "major_interest",
            "learning_difficulty",
            "consultation_intent",
        ),
    ),
    KareEvaluationCase(
        "KARE-04",
        "실습 보고서와 콘텐츠 관심",
        "유아교육과",
        "서술형",
        (
            "전공은 재미있지만 과제 보고서가 밀려서 버거워요.",
            "콘텐츠·디자인 분야에 관심 있고 콘텐츠 기획 직무를 하고 싶어요.",
            "학습 도움을 받고 싶어요. 여기까지 할게요.",
        ),
        (
            ("major_interest", ("positive",)),
            ("learning_difficulty", ("high_difficulty",)),
            ("interest_fields", ("identified",)),
            ("desired_job", ("identified",)),
            ("career_clarity", ("clear",)),
            ("natural_language_concern", ("identified",)),
        ),
        (
            "major_interest",
            "learning_difficulty",
            "natural_language_concern",
        ),
    ),
    KareEvaluationCase(
        "KARE-05",
        "관심 분야 선택지 요청",
        "유아교육과",
        "질문형",
        (
            "관심 분야에는 어떤 게 있어요?",
            "그중에서는 서비스 분야에 관심 있어요.",
            "오늘은 여기까지 이야기할게요.",
        ),
        (("interest_fields", ("identified",)),),
        forbidden_dimensions=("desired_job",),
    ),
    KareEvaluationCase(
        "KARE-06",
        "진로 정정",
        "유아교육과",
        "정정형",
        (
            "진로가 막막해서 고민이에요.",
            "아까는 막막하다고 했는데 서비스 기획 직무도 생각하고 있어요.",
            "진로 상담을 받고 싶어요. 여기까지 할게요.",
        ),
        (
            ("career_clarity", ("clear", "exploring")),
            ("desired_job", ("identified",)),
            ("consultation_intent", ("wants_support",)),
            ("natural_language_concern", ("identified",)),
        ),
    ),
    KareEvaluationCase(
        "KARE-07",
        "전공 만족과 콘텐츠 진로",
        "실용음악과(3년제)",
        "긍정적",
        (
            "전공은 만족스럽고 콘텐츠 분야에 관심 있어요.",
            "콘텐츠 기획 직무를 하고 싶어요. 여기까지 이야기할게요.",
        ),
        (
            ("major_satisfaction", ("positive",)),
            ("interest_fields", ("identified",)),
            ("desired_job", ("identified",)),
            ("career_clarity", ("clear",)),
        ),
        ("major_satisfaction", "interest_fields"),
    ),
    KareEvaluationCase(
        "KARE-08",
        "희망 직무 미정",
        "실용음악과(3년제)",
        "망설임",
        (
            "졸업 후 진로를 아직 못 정해서 고민이에요.",
            "지금은 희망 직무가 없어요.",
            "오늘은 여기까지 할게요.",
        ),
        (
            ("career_clarity", ("not_clear",)),
            ("desired_job", ("none", "undetermined")),
            ("natural_language_concern", ("identified",)),
        ),
    ),
    KareEvaluationCase(
        "KARE-09",
        "전공 이탈 고민",
        "실용음악과(3년제)",
        "직설적",
        (
            "전공이 재미없고 그만둘지 고민이에요.",
            "상담을 받아보고 싶어요. 여기까지 이야기할게요.",
        ),
        (
            ("major_interest", ("negative",)),
            ("major_continuation_intent", ("not_continuing",)),
            ("consultation_intent", ("wants_support",)),
            ("natural_language_concern", ("identified",)),
        ),
        (
            "major_interest",
            "major_continuation_intent",
            "natural_language_concern",
        ),
    ),
    KareEvaluationCase(
        "KARE-10",
        "빠른 수업과 상담 요청",
        "간호학과",
        "긴 문장",
        (
            "전공은 만족하지만 수업이 너무 어렵고 외울 게 많아서 따라가기가 버거워요.",
            "계속 공부하고 싶고 학습 상담도 받고 싶어요. 여기까지 할게요.",
        ),
        (
            ("major_satisfaction", ("positive",)),
            ("major_continuation_intent", ("continuing",)),
            ("learning_difficulty", ("high_difficulty",)),
            ("consultation_intent", ("wants_support",)),
            ("natural_language_concern", ("identified",)),
        ),
        (
            "major_satisfaction",
            "learning_difficulty",
            "natural_language_concern",
        ),
    ),
    KareEvaluationCase(
        "KARE-11",
        "구체적인 간호 진로",
        "간호학과",
        "간결함",
        (
            "진로는 간호사로 정했고 전공도 계속 공부하고 싶어요.",
            "보건 분야에 관심 있어요. 오늘은 여기까지 이야기할게요.",
        ),
        (
            ("career_clarity", ("clear",)),
            ("major_continuation_intent", ("continuing",)),
            ("interest_fields", ("identified",)),
        ),
        ("career_clarity", "major_continuation_intent"),
    ),
    KareEvaluationCase(
        "KARE-12",
        "대화 내용 없음",
        "간호학과",
        "단답형",
        (
            "요즘 학교생활부터 이야기할게요",
            "모르겠어요",
            "없어요",
        ),
        (("natural_language_concern", ("none", "undetermined")),),
    ),
    KareEvaluationCase(
        "KARE-13",
        "데이터 진로와 학습 부담",
        "소프트웨어융합과",
        "정보 밀집형",
        (
            "수업이 너무 어렵고 과제가 밀려서 걱정되지만 데이터 분야에 관심 있고 데이터 분석 직무를 하고 싶어요.",
            "학습 상담을 받고 싶어요. 여기까지 이야기할게요.",
        ),
        (
            ("learning_difficulty", ("high_difficulty",)),
            ("interest_fields", ("identified",)),
            ("desired_job", ("identified",)),
            ("career_clarity", ("clear",)),
            ("consultation_intent", ("wants_support",)),
            ("natural_language_concern", ("identified",)),
        ),
        (
            "learning_difficulty",
            "interest_fields",
            "desired_job",
            "career_clarity",
            "natural_language_concern",
        ),
    ),
    KareEvaluationCase(
        "KARE-14",
        "전공 흥미 정정",
        "소프트웨어융합과",
        "정정형",
        (
            "전공이 재미없다고 느꼈어요.",
            "아까 말은 정정할게요. 프로젝트를 해보니 전공이 재미있어요.",
            "오늘은 여기까지 할게요.",
        ),
        (("major_interest", ("positive",)),),
        first_turn_dimensions=("major_interest",),
    ),
    KareEvaluationCase(
        "KARE-15",
        "안정 상태와 고민 없음",
        "소프트웨어융합과",
        "담담함",
        (
            "전공은 재미있고 수업도 할만해요. 특별히 고민 없어요.",
            "상담은 필요 없어요. 여기까지 이야기할게요.",
        ),
        (
            ("major_interest", ("positive",)),
            ("learning_difficulty", ("low_difficulty",)),
            ("natural_language_concern", ("none",)),
            ("consultation_intent", ("no_support",)),
        ),
        (
            "major_interest",
            "learning_difficulty",
            "natural_language_concern",
        ),
    ),
)


def _percentage(numerator: int, denominator: int) -> float:
    """분모가 없으면 0으로 두고 백분율을 소수점 한 자리로 반환한다."""

    if denominator <= 0:
        return 0.0
    return round(numerator / denominator * 100, 1)


def _validate_case(case: KareEvaluationCase) -> None:
    """평가 사례 자체의 필드·기대값 오류를 실행 전에 차단한다."""

    if not case.case_id or not case.messages:
        raise ValueError("Kare 평가 사례에는 ID와 한 개 이상의 발화가 필요합니다.")
    expected_dimensions = [dimension for dimension, _ in case.expected_states]
    if len(expected_dimensions) != len(set(expected_dimensions)):
        raise ValueError(f"{case.case_id}의 기대 의미 영역이 중복되었습니다.")
    referenced_dimensions = {
        *expected_dimensions,
        *case.first_turn_dimensions,
        *case.forbidden_dimensions,
    }
    unknown_dimensions = referenced_dimensions.difference(SEMANTIC_DIMENSIONS)
    if unknown_dimensions:
        raise ValueError(
            f"{case.case_id}에 지원하지 않는 의미 영역이 있습니다: "
            f"{sorted(unknown_dimensions)}"
        )


def evaluate_kare_case(
    case: KareEvaluationCase,
    provider: AIProvider,
    *,
    allowed_interest_fields: Sequence[str] = DEFAULT_INTEREST_FIELDS,
) -> KareEvaluationCaseResult:
    """Kare 한 사례를 실제 대화 서비스로 실행하고 안전·추출 조건을 평가한다."""

    _validate_case(case)
    service = CheckinChatService(provider, allowed_interest_fields)
    state: Mapping[str, Mapping[str, Any]] = empty_semantic_state()
    values = empty_chat_values()
    history: list[dict[str, str]] = []
    asked_dimensions: tuple[str, ...] = ()
    covered_fields: tuple[str, ...] = ()
    next_focus: str | None = None
    focus_history: list[str] = []
    transcript: list[str] = []
    scale_question_count = 0
    first_turn_covered: set[str] = set()
    ready_for_review = False
    provider_names: list[str] = []

    try:
        for turn_index, message in enumerate(case.messages):
            transcript.append(f"학생: {message}")
            result = service.process_turn(
                history=history,
                user_message=message,
                values=values,
                covered_fields=covered_fields,
                current_focus=next_focus,
                semantic_state=state,
                asked_dimensions=asked_dimensions,
            )
            transcript.append(f"Kare: {result.assistant_message}")
            if service._contains_scale_question(result.assistant_message):
                scale_question_count += 1
            if result.next_focus != "review":
                # 같은 의미 영역을 더 깊게 물어도 질문 의도가 다르면 반복이 아니다.
                focus_history.append(
                    f"{result.conversation_thread}:{result.question_intent}"
                )
            state = result.semantic_state or empty_semantic_state()
            values = result.values or values
            covered_fields = result.covered_fields
            asked_dimensions = result.asked_dimensions
            next_focus = result.next_focus
            provider_names.append(result.provider_name)
            if turn_index == 0:
                first_turn_covered = set(result.covered_fields)
            ready_for_review = result.ready_for_review
            history.extend(
                (
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": result.assistant_message},
                )
            )
            if ready_for_review:
                break
    except Exception as error:
        return KareEvaluationCaseResult(
            case.case_id,
            case.title,
            case.department,
            case.tone,
            len(focus_history),
            len(case.expected_states),
            0,
            len(case.first_turn_dimensions),
            0,
            0,
            scale_question_count,
            0,
            False,
            ", ".join(dict.fromkeys(provider_names)) or provider.provider_name,
            False,
            ("대화 실행 실패",),
            tuple(transcript),
            str(error),
        )

    checked_state = dict(state)
    mismatches: list[str] = []
    matched_states = 0
    for dimension, allowed_states in case.expected_states:
        actual_state = str(checked_state[dimension]["state"])
        if actual_state in allowed_states:
            matched_states += 1
        else:
            mismatches.append(
                f"{dimension}: 기대 {'/'.join(allowed_states)}, 실제 {actual_state}"
            )
    unexpected = 0
    for dimension in case.forbidden_dimensions:
        actual_state = str(checked_state[dimension]["state"])
        if actual_state != "unknown":
            unexpected += 1
            mismatches.append(f"{dimension}: 추론 금지, 실제 {actual_state}")
    first_turn_matched = sum(
        dimension in first_turn_covered for dimension in case.first_turn_dimensions
    )
    repeated_focus_count = len(focus_history) - len(set(focus_history))
    review_matches = ready_for_review == case.expect_review
    if not review_matches:
        mismatches.append(
            f"확인 전환: 기대 {case.expect_review}, 실제 {ready_for_review}"
        )
    passed = (
        matched_states == len(case.expected_states)
        and first_turn_matched == len(case.first_turn_dimensions)
        and unexpected == 0
        and scale_question_count == 0
        and repeated_focus_count == 0
        and review_matches
    )
    provider_name = ", ".join(dict.fromkeys(provider_names)) or provider.provider_name
    return KareEvaluationCaseResult(
        case.case_id,
        case.title,
        case.department,
        case.tone,
        len(history) // 2,
        len(case.expected_states),
        matched_states,
        len(case.first_turn_dimensions),
        first_turn_matched,
        unexpected,
        scale_question_count,
        repeated_focus_count,
        ready_for_review,
        provider_name,
        passed,
        tuple(mismatches),
        tuple(transcript),
    )


def evaluate_kare_dialogues(
    provider: AIProvider,
    cases: Sequence[KareEvaluationCase] = KARE_SYNTHETIC_EVALUATION_CASES,
    *,
    allowed_interest_fields: Sequence[str] = DEFAULT_INTEREST_FIELDS,
) -> KareEvaluationReport:
    """여러 synthetic 사례를 실행해 비교 가능한 Kare 품질 보고서를 만든다."""

    checked_cases = tuple(cases)
    if not checked_cases:
        raise ValueError("Kare 평가 사례가 한 개 이상 필요합니다.")
    if len({case.case_id for case in checked_cases}) != len(checked_cases):
        raise ValueError("Kare 평가 사례 ID는 중복될 수 없습니다.")
    results = tuple(
        evaluate_kare_case(
            case,
            provider,
            allowed_interest_fields=allowed_interest_fields,
        )
        for case in checked_cases
    )
    details = pd.DataFrame(
        [
            {
                "case_id": result.case_id,
                "title": result.title,
                "department": result.department,
                "tone": result.tone,
                "turn_count": result.turn_count,
                "state_matches": (
                    f"{result.matched_state_count}/{result.expected_state_count}"
                ),
                "first_turn_matches": (
                    f"{result.first_turn_matched_count}/"
                    f"{result.first_turn_expected_count}"
                ),
                "unexpected_inference_count": result.unexpected_inference_count,
                "scale_question_count": result.scale_question_count,
                "repeated_focus_count": result.repeated_focus_count,
                "ready_for_review": result.ready_for_review,
                "provider": result.provider_name,
                "passed": result.passed,
                "mismatches": " · ".join(result.mismatches),
                "error": result.error or "",
            }
            for result in results
        ]
    )
    expected_review_results = [
        (case, result)
        for case, result in zip(checked_cases, results)
        if case.expect_review
    ]
    return KareEvaluationReport(
        case_count=len(results),
        passed_case_count=sum(result.passed for result in results),
        expected_state_count=sum(result.expected_state_count for result in results),
        matched_state_count=sum(result.matched_state_count for result in results),
        first_turn_expected_count=sum(
            result.first_turn_expected_count for result in results
        ),
        first_turn_matched_count=sum(
            result.first_turn_matched_count for result in results
        ),
        unexpected_inference_count=sum(
            result.unexpected_inference_count for result in results
        ),
        scale_question_count=sum(result.scale_question_count for result in results),
        repeated_focus_count=sum(result.repeated_focus_count for result in results),
        expected_review_count=len(expected_review_results),
        completed_review_count=sum(
            result.ready_for_review for _, result in expected_review_results
        ),
        details=details,
        results=results,
    )


def kare_evaluation_catalog() -> pd.DataFrame:
    """화면에 표시할 비식별 synthetic 평가 사례 목록을 반환한다."""

    return pd.DataFrame(
        [
            {
                "case_id": case.case_id,
                "title": case.title,
                "department": case.department,
                "tone": case.tone,
                "turns": len(case.messages),
                "first_message": case.messages[0],
            }
            for case in KARE_SYNTHETIC_EVALUATION_CASES
        ]
    )
