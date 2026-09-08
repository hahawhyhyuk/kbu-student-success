"""API key 없이도 동작하는 deterministic 체크인 분석 provider."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from src.ai.base import AIProvider
from src.ai.schema import (
    validate_checkin_analysis,
    validate_learning_path_explanation,
    validate_microdegree_candidate_description,
    validate_next_action_suggestion,
    validate_staff_briefing,
    validate_checkin_chat_turn,
    validate_kare_chat_reply,
)
from src.checkin_conversation import (
    CHAT_OBSERVATION_FIELDS,
    CHAT_REQUIRED_FIELDS,
    SCALE_QUESTION_BY_FIELD,
    chat_question,
    explicit_concern_from_message,
    next_uncovered_field,
    supported_scale_score,
)
from src.checkin_semantic_state import (
    SEMANTIC_DIMENSIONS,
    merge_semantic_updates,
    next_semantic_focus,
)


class MockAIProvider(AIProvider):
    """키워드 규칙으로 JSON Schema 호환 결과를 생성한다."""

    provider_name = "mock"

    INTEREST_TERMS = {
        "데이터·AI": ("데이터", "AI", "인공지능"),
        "경영·마케팅": ("경영", "마케팅", "SNS"),
        "콘텐츠·디자인": ("콘텐츠", "디자인", "영상"),
        "서비스": ("서비스",),
        "상담·복지": ("상담", "복지"),
        "보건": ("보건",),
    }
    JOB_TERMS = {
        "데이터 분석": ("데이터 분석", "데이터분석"),
        "디지털 마케팅": ("디지털 마케팅", "SNS 마케팅", "마케팅"),
        "콘텐츠 기획": ("콘텐츠 기획", "콘텐츠"),
        "서비스 기획": ("서비스 기획",),
    }

    @staticmethod
    def _is_no_content_reply(text: str) -> bool:
        """짧은 부정·모름 응답을 대화 주제 없음 신호로 구분한다."""

        compact = "".join(str(text or "").strip().rstrip(".!? ").split())
        return compact in {
            "없어",
            "없어요",
            "없음",
            "특별히없어",
            "특별히없어요",
            "모르겠어",
            "모르겠어요",
            "잘모르겠어",
            "잘모르겠어요",
        }

    def generate_kare_reply(
        self,
        *,
        history: Sequence[Mapping[str, str]],
        user_message: str,
        semantic_state: Mapping[str, Mapping[str, Any]],
        asked_dimensions: Sequence[str],
        allowed_interest_fields: Sequence[str],
        conversation_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """명시적 표현만 9개 의미 영역에 연결하며 로컬 대화를 이어간다."""

        message = " ".join(str(user_message or "").strip().split())
        context = dict(conversation_context or {})
        active_thread = str(context.get("active_thread") or "general")
        compact = "".join(message.rstrip(".!? ").split())
        meta_opening = compact in {
            "요즘학교생활부터이야기할게요",
            "학교생활부터이야기할게요",
            "이야기할게요",
        }
        updates: list[dict[str, Any]] = []

        def add_update(
            dimension: str,
            state: str,
            *,
            detail: str | None = None,
            confidence: str = "high",
            replace: bool = False,
        ) -> None:
            existing_index = next(
                (
                    index
                    for index, item in enumerate(updates)
                    if item["dimension"] == dimension
                ),
                None,
            )
            update = {
                "dimension": dimension,
                "state": state,
                "evidence": message,
                "confidence": confidence,
                "detail": detail,
            }
            if existing_index is None:
                updates.append(update)
            elif replace:
                updates[existing_index] = update

        if self._is_no_content_reply(message):
            last_focus = next(
                (
                    dimension
                    for dimension in reversed(tuple(asked_dimensions))
                    if dimension in SEMANTIC_DIMENSIONS
                ),
                "natural_language_concern",
            )
            no_state = {
                "major_interest": "negative",
                "major_satisfaction": "negative",
                "major_continuation_intent": "not_continuing",
                "learning_difficulty": "low_difficulty",
                "career_clarity": "not_clear",
                "consultation_intent": "no_support",
                "interest_fields": "none",
                "desired_job": "none",
                "natural_language_concern": "none",
            }[last_focus]
            add_update(last_focus, no_state)
        elif not meta_opening:
            if ("전공" in message or "실습" in message) and any(
                term in message
                for term in ("재미있", "재미 있", "흥미 있", "흥미가 있", "즐거")
            ):
                add_update("major_interest", "positive")
            elif "전공" in message and any(
                term in message for term in ("재미없", "흥미 없", "싫")
            ):
                add_update("major_interest", "negative")
            if "전공" in message and any(
                term in message for term in ("만족", "불만")
            ):
                negative_satisfaction = "불만" in message or bool(
                    re.search(r"만족하(?:지\s*(?:않|못)|진\s*않)", message)
                )
                state = "negative" if negative_satisfaction else "positive"
                add_update("major_satisfaction", state)
            continuation_context = (
                "전공" in message
                or "계속 공부" in message
                or "공부를 계속" in message
                or "major_continuation_intent" in asked_dimensions
            )
            if continuation_context and any(
                term in message for term in ("계속", "그만", "바꾸", "전과")
            ) and "점" not in message:
                if any(term in message for term in ("그만", "바꾸", "전과", "싫")):
                    state = "not_continuing"
                elif any(term in message for term in ("고민", "모르")):
                    state = "considering"
                else:
                    state = "continuing"
                add_update("major_continuation_intent", state)
            if any(term in message for term in ("수업", "과제", "공부", "학습")):
                if any(
                    term in message
                    for term in (
                        "너무 어려",
                        "너무 어렵",
                        "많이 어려",
                        "많이 어렵",
                        "버겁",
                        "밀려",
                        "밀리",
                        "밀렸",
                        "빠르",
                    )
                ):
                    add_update("learning_difficulty", "high_difficulty")
                elif any(term in message for term in ("어렵", "어려", "힘들")):
                    add_update("learning_difficulty", "moderate_difficulty")
                elif any(term in message for term in ("할만", "괜찮", "수월")):
                    add_update("learning_difficulty", "low_difficulty")
            if any(term in message for term in ("진로", "직무", "졸업 후")):
                if any(term in message for term in ("막막", "모르", "못 정", "안 정")):
                    add_update("career_clarity", "not_clear")
                elif any(term in message for term in ("고민", "탐색", "생각 중")):
                    add_update("career_clarity", "exploring")
                elif any(term in message for term in ("정했", "명확", "하고 싶")):
                    add_update("career_clarity", "clear")
            if "상담" in message:
                state = "no_support" if any(
                    term in message for term in ("필요 없", "안 받", "원하지 않")
                ) else "wants_support"
                add_update("consultation_intent", state)
            detected_interests = [
                label
                for label, terms in self.INTEREST_TERMS.items()
                if label in set(map(str, allowed_interest_fields))
                and any(term in message for term in terms)
                and (
                    active_thread == "interests"
                    or any(term in message for term in ("관심", "분야", "마음이 가"))
                )
            ]
            if detected_interests:
                add_update(
                    "interest_fields",
                    "identified",
                    detail=", ".join(detected_interests),
                )
            detected_jobs = [
                label
                for label, terms in self.JOB_TERMS.items()
                if any(term in message for term in terms)
                and (
                    active_thread == "career"
                    or any(term in message for term in ("직무", "되고 싶", "하고 싶"))
                )
            ]
            if detected_jobs:
                add_update("desired_job", "identified", detail=detected_jobs[0])
                add_update("career_clarity", "clear", replace=True)
            elif any(term in message for term in ("희망 직무", "직무")) and any(
                term in message for term in ("없어", "없어요", "정하지 못", "못 정")
            ):
                add_update("desired_job", "none")
            if any(term in message for term in ("고민 없", "걱정 없", "특별히 없")):
                add_update("natural_language_concern", "none")
            elif any(
                term in message
                for term in (
                    "고민",
                    "걱정",
                    "버겁",
                    "막막",
                    "따라가기",
                    "밀려",
                    "밀리",
                    "밀렸",
                )
            ):
                add_update("natural_language_concern", "identified", detail=message)

        proposed_state = merge_semantic_updates(
            semantic_state,
            updates,
            latest_user_message=message,
        )
        next_focus = str(
            context.get("next_focus")
            or next_semantic_focus(proposed_state, asked_dimensions)
        )
        question_intent = str(context.get("question_intent") or "")
        fallback_question = str(context.get("fallback_question") or "").strip()
        suggest_review = next_focus == "review" or question_intent == "confirm_summary"
        if suggest_review:
            assistant_message = (
                "이야기해줘서 고마워요. 지금까지 이해한 내용을 "
                "제출 전에 함께 확인해볼까요?"
            )
        else:
            question = fallback_question or "조금 더 구체적으로 떠오르는 상황이 있나요?"
            reflection = self._natural_reflection(
                message,
                active_thread=str(context.get("active_thread") or "general"),
                meta_opening=meta_opening,
            )
            assistant_message = f"{reflection} {question}"
        reply = {
            "assistant_message": assistant_message,
            "dimension_updates": updates,
            "next_focus": next_focus,
            "suggest_review": suggest_review,
        }
        return validate_kare_chat_reply(reply)

    @classmethod
    def _natural_reflection(
        cls,
        message: str,
        *,
        active_thread: str,
        meta_opening: bool,
    ) -> str:
        """학생이 말하지 않은 감정을 만들지 않는 짧은 사실 반영을 만든다."""

        if meta_opening:
            return "좋아요."
        if cls._is_no_content_reply(message):
            return "괜찮아요. 지금 바로 떠오르지 않을 수 있어요."
        if active_thread == "learning":
            if "수업" in message and "과제" in message and any(
                term in message for term in ("밀려", "밀리")
            ):
                return "수업을 따라가다 보니 과제까지 밀리고 있군요."
            if any(term in message for term in ("외울", "암기")):
                return "외울 내용이 많아 정리하는 데 어려움이 있군요."
            if any(term in message for term in ("어렵", "힘들", "버겁", "밀려")):
                return "수업이나 과제에서 부담을 느끼고 있군요."
            return "요즘 공부하면서 겪는 상황을 이야기해 주셨군요."
        if active_thread == "major":
            if any(term in message for term in ("재미있", "흥미 있", "즐거")):
                return "전공에서 재미를 느끼는 부분이 있군요."
            if any(term in message for term in ("안 맞", "재미없", "싫")):
                return "전공이 나와 잘 맞는지 고민하고 있군요."
            return "전공에 대해 느끼는 점을 이야기해 주셨군요."
        if active_thread == "career":
            if any(term in message for term in ("막막", "모르", "못 정")):
                return "앞으로 어떤 일을 할지 아직 정리하는 중이군요."
            return "앞으로 해보고 싶은 일에 대한 생각이 있군요."
        if active_thread == "interests":
            return "눈길이 가는 분야에 대해 이야기하고 있군요."
        if active_thread == "support":
            if any(term in message for term in ("필요 없", "안 받", "원하지 않")):
                return "지금은 다른 사람의 도움보다 스스로 정리해 보고 싶군요."
            if any(term in message for term in ("상담", "받고 싶", "도와")):
                return "지금은 함께 방법을 찾아보고 싶군요."
            return "어떤 방식으로 정리할지 생각하고 있군요."
        return "지금 떠오르는 이야기를 들려주셨군요."

    def analyze_checkin(self, text: str) -> dict[str, Any]:
        """자유서술의 명시적 표현만 사용해 고민과 지원수요를 구조화한다."""

        normalized = str(text or "").strip()
        difficulty_terms = (
            "어렵",
            "버겁",
            "부족",
            "놓쳤",
            "놓칠",
            "밀린",
            "따라가기",
            "성적",
            "걱정",
        )
        concern_terms = ("고민", "확신", "맞는지", "다른 분야", "적성")
        career_terms = ("진로", "직무", "어떤 일", "졸업 후", "무엇을 하고")

        major_concern = "전공" in normalized and any(
            term in normalized for term in concern_terms
        )
        learning_difficulty = any(
            term in normalized for term in difficulty_terms
        ) and any(
            term in normalized
            for term in ("수업", "과제", "학습", "공부", "퀴즈", "진도", "성적")
        )
        career_uncertainty = any(
            term in normalized for term in career_terms
        ) and any(
            term in normalized
            for term in ("모르", "정하지", "고민", "불확실", "궁금")
        )

        interests = [
            label
            for label, terms in self.INTEREST_TERMS.items()
            if any(term in normalized for term in terms)
        ]
        # "상담을 받고 싶다"는 지원 요청이지 상담·복지 분야에 대한
        # 학습 관심이 아니다. 명시적인 분야 표현이 없으면 후보에서 제외한다.
        consultation_request_terms = (
            "상담받고 싶",
            "상담 받고 싶",
            "상담을 받고 싶",
            "상담도 받고 싶",
            "상담을 받아",
            "상담 받아",
            "상담이 필요",
            "상담 희망",
        )
        explicit_counseling_interest = any(
            term in normalized
            for term in (
                "상담 분야",
                "상담·복지",
                "상담복지",
                "복지 분야",
                "복지에 관심",
                "상담에 관심",
            )
        )
        if (
            "상담·복지" in interests
            and any(term in normalized for term in consultation_request_terms)
            and not explicit_counseling_interest
        ):
            interests.remove("상담·복지")
        desired_jobs = [
            label
            for label, terms in self.JOB_TERMS.items()
            if any(term in normalized for term in terms)
        ]
        support_needs = []
        if major_concern:
            support_needs.append("전공 탐색")
        if learning_difficulty:
            support_needs.append("학습 지원")
        if career_uncertainty:
            support_needs.append("진로 탐색")
        if not support_needs:
            support_needs.append("정기 모니터링")

        detected = []
        if major_concern:
            detected.append("전공 고민")
        if learning_difficulty:
            detected.append("학습 어려움")
        if career_uncertainty:
            detected.append("진로 불확실성")
        summary = (
            "자유서술에서 " + ", ".join(detected) + "이 확인됨"
            if detected
            else "자유서술에서 즉시 지원이 필요한 명시적 고민은 확인되지 않음"
        )
        return validate_checkin_analysis(
            {
                "major_concern": major_concern,
                "learning_difficulty": learning_difficulty,
                "career_uncertainty": career_uncertainty,
                "interests": interests,
                "desired_jobs": desired_jobs,
                "support_needs": support_needs,
                "summary": summary,
            }
        )

    def generate_checkin_chat_turn(
        self,
        *,
        history: Sequence[Mapping[str, str]],
        values: Mapping[str, Any],
        covered_fields: Sequence[str],
        user_message: str,
        current_focus: str,
        allowed_interest_fields: Sequence[str],
    ) -> dict[str, Any]:
        """명시적 점수와 키워드를 이용해 API 없이도 Kare 대화를 이어간다."""

        del history
        message = str(user_message).strip()
        updates: dict[str, Any] = {
            "major_interest": None,
            "major_satisfaction": None,
            "major_continuation_intent": None,
            "learning_difficulty": None,
            "career_clarity": None,
            "consultation_intent": None,
            "interest_fields": None,
            "desired_job": None,
            "consultation_requested": None,
            "explore_other_fields": None,
            "natural_language_concern": None,
        }
        addressed: set[str] = set()

        for field in SCALE_QUESTION_BY_FIELD:
            score = supported_scale_score(
                field,
                message,
                allow_bare_score=field == current_focus,
            )
            if score is not None:
                updates[field] = score
                addressed.add(field)

        interest_context = current_focus == "interest_fields" or any(
            term in message for term in ("관심", "분야", "탐색")
        )
        interest_scopes: list[str] = []
        for sentence in re.split(r"[.!?。]\s*", message):
            marker_positions = [
                sentence.find(marker)
                for marker in ("관심", "분야", "탐색")
                if marker in sentence
            ]
            if not marker_positions:
                continue
            marker_position = min(marker_positions)
            clause_start = max(
                sentence.rfind(";", 0, marker_position),
                sentence.rfind("；", 0, marker_position),
            ) + 1
            explicit_job_starts = [
                sentence.lower().find(label.lower(), marker_position)
                for label in self.JOB_TERMS
                if sentence.lower().find(label.lower(), marker_position) >= 0
            ]
            clause_end = (
                min(explicit_job_starts) if explicit_job_starts else len(sentence)
            )
            interest_scopes.append(sentence[clause_start:clause_end])

        def is_interest_term(term: str) -> bool:
            """상담 희망 같은 다른 문맥의 단어를 관심 분야로 오인하지 않는다."""

            if current_focus == "interest_fields" and not interest_scopes:
                return term.lower() in message.lower()
            return any(term.lower() in scope.lower() for scope in interest_scopes)

        detected_interests = (
            [
                label
                for label in allowed_interest_fields
                for terms in [self.INTEREST_TERMS.get(label, (label,))]
                if any(is_interest_term(term) for term in terms)
            ]
            if interest_context
            else []
        )
        if (
            "상담·복지" in detected_interests
            and "복지" not in message
            and "상담 분야" not in message
            and "상담" in message
            and any(term in message for term in ("받", "필요", "희망", "원해"))
        ):
            detected_interests.remove("상담·복지")
        if detected_interests:
            previous_interests = list(values.get("interest_fields") or [])
            updates["interest_fields"] = list(
                dict.fromkeys([*previous_interests, *detected_interests])
            )
            addressed.add("interest_fields")
        elif current_focus == "interest_fields" and any(
            term in message for term in ("없", "모르", "아직", "정하지")
        ):
            updates["interest_fields"] = []
            addressed.add("interest_fields")

        job_context = current_focus == "desired_job" or any(
            term in message for term in ("직무", "하고 싶", "되고 싶", "희망")
        )
        explicit_job_labels = [
            label
            for label in self.JOB_TERMS
            if label.lower() in message.lower()
        ]
        detected_jobs = (
            explicit_job_labels
            or [
                label
                for label, terms in self.JOB_TERMS.items()
                if any(term.lower() in message.lower() for term in terms)
            ]
            if job_context
            else []
        )
        if detected_jobs:
            updates["desired_job"] = detected_jobs[0]
            addressed.add("desired_job")
        elif current_focus == "desired_job":
            if any(term in message for term in ("없", "모르", "아직", "정하지")):
                updates["desired_job"] = ""
                addressed.add("desired_job")

        explicit_concern = explicit_concern_from_message(message)
        if explicit_concern is not None:
            updates["natural_language_concern"] = explicit_concern
            addressed.add("natural_language_concern")
        elif current_focus == "natural_language_concern":
            updates["natural_language_concern"] = message[:1200]
            addressed.add("natural_language_concern")

        if "상담" in message:
            if any(
                term in message
                for term in (
                    "받고 싶",
                    "받아보고 싶",
                    "받아 보고 싶",
                    "받을래",
                    "원해",
                    "필요",
                    "희망",
                )
            ):
                updates["consultation_requested"] = True
            elif any(term in message for term in ("괜찮", "필요 없", "원하지")):
                updates["consultation_requested"] = False
        if "새로운 분야" in message or "다른 분야" in message:
            updates["explore_other_fields"] = True

        effective_covered = set(map(str, covered_fields))
        for field in addressed:
            if field in SCALE_QUESTION_BY_FIELD and updates[field] is None:
                continue
            effective_covered.add(field)
        next_focus = next_uncovered_field(effective_covered)
        ready = next_focus == "review"
        if ready:
            assistant_message = (
                "이야기해줘서 고마워요. 지금까지 이해한 내용을 제출 전에 함께 확인해볼까요?"
            )
        else:
            assistant_message = f"말해줘서 고마워요. {chat_question(next_focus)}"

        result = {
            "assistant_message": assistant_message,
            "field_updates": updates,
            "addressed_fields": [
                field for field in CHAT_OBSERVATION_FIELDS if field in addressed
            ],
            "next_focus": next_focus,
            "ready_for_review": ready,
        }
        return validate_checkin_chat_turn(result, set(allowed_interest_fields))

    def explain_learning_path(
        self, profile: dict[str, Any], courses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """선정된 DB 교과목만 사용해 deterministic 학습경로 설명을 만든다."""

        interests = str(profile.get("interest_fields", "관심역량")).replace("|", "·")
        competencies = sorted(
            {item for course in courses for item in str(course["competencies"]).split("|") if item}
        )
        has_competencies = bool(competencies)
        result = {
            "path_name": (
                f"{interests or '관심분야'} 역량 연결 경로"
                if has_competencies
                else f"{interests or '관심분야'} 교과 탐색 경로"
            ),
            "reason": (
                "학생의 관심분야와 희망직무를 바탕으로 기초부터 심화 순서로 연결한 교과목 조합입니다."
                if has_competencies
                else "학생의 관심 내용과 실제 개설 교과목명·학과·수업정보를 연결한 탐색 경로입니다."
            ),
            "competencies": competencies,
            "course_roles": [
                {
                    "course_id": str(course["course_id"]),
                    "role": (
                        f"{course['course_name']}을 통해 "
                        f"{str(course['competencies']).replace('|', '·')} 역량을 보완"
                        if str(course["competencies"]).strip()
                        else f"{course['course_name']}의 실제 개설 내용을 확인"
                    ),
                    "sequence": index,
                }
                for index, course in enumerate(courses, start=1)
            ],
            "career_connection": f"희망직무 {profile.get('desired_job') or '탐색 단계'}에 필요한 복합 역량을 단계적으로 확인할 수 있습니다.",
        }
        return validate_learning_path_explanation(result, {str(course["course_id"]) for course in courses})

    def describe_microdegree_candidate(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """집계된 역량과 직무만 사용해 deterministic 후보 설명을 만든다."""

        competencies = [str(item) for item in candidate.get("competencies", [])]
        related_jobs = [str(item) for item in candidate.get("related_jobs", [])]
        course_names = [str(item) for item in candidate.get("course_names", [])]
        name_terms = competencies[:2] or related_jobs[:2] or course_names[:2]
        candidate_name = "·".join(name_terms or ["교과 연계"]) + " 개발 후보"
        evidence = (
            f"{', '.join(competencies[:4])} 역량 수요"
            if competencies
            else f"{', '.join(course_names[:4])} 교과목 반복 선택"
        )
        description = (
            f"{candidate.get('student_count', 0)}명의 반복 학습경로에서 확인된 "
            f"{evidence}를 바탕으로, 교과목 구성과 운영 타당성을 "
            "교직원이 추가 검토할 수 있는 후보입니다."
        )
        return validate_microdegree_candidate_description(
            {
                "candidate_name": candidate_name,
                "description": description,
                "status": "교육과정 검토 후보",
            }
        )

    def generate_staff_briefing(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """코드 집계 위험유형을 이용해 재현 가능한 교직원 브리핑을 만든다."""

        focuses = [str(item) for item in context["focus_risk_types"]]
        if focuses == ["정기 모니터링"]:
            headline = "현재는 안정적인 흐름을 이어서 살펴볼 때예요"
            summary = (
                "뚜렷하게 집중된 지원 신호보다 주차별 변화를 계속 확인하는 것이 좋습니다."
            )
        else:
            headline = "지원 우선 신호를 함께 확인했어요"
            summary = (
                f"{', '.join(focuses)} 신호가 상대적으로 두드러져 원천지표와 "
                "학생의 현재 의사를 함께 살펴보는 것이 좋습니다."
        )
        allowed_actions = [str(item) for item in context["allowed_actions"]]
        request_focus = str(context.get("request_focus", "priority"))
        preferred_action = {
            "risk_distribution": "전체 대시보드에서 분포 확인",
            "unaddressed": "아직 개입되지 않은 고위험 학생 확인",
            "priority": "학생 종합 확인에서 근거 검토",
        }.get(request_focus, allowed_actions[0])
        if (
            request_focus == "unaddressed"
            and int(context["unaddressed_count"]) == 0
        ):
            preferred_action = "전체 대시보드에서 분포 확인"
        action = (
            preferred_action
            if preferred_action in allowed_actions
            else allowed_actions[0]
        )
        return validate_staff_briefing(
            {
                "headline": headline,
                "summary": summary,
                "focus_risk_types": focuses[:3],
                "recommended_action": action,
            },
            set(focuses),
            set(allowed_actions),
        )

    def suggest_next_action(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """지원 상태에 따라 설정된 표준 행동 중 하나를 재현 가능하게 선택한다."""

        allowed_actions = [str(item) for item in context["allowed_actions"]]
        status = str(context["intervention_status"])
        preferred_by_status = {
            "추천 생성": "전화 안내",
            "교직원 검토 대기": "전화 안내",
            "교직원 검토": "전화 안내",
            "지원계획 작성": "전화 안내",
            "연락 전": "전화 안내",
            "상담 예정": "상담 일정 조율",
            "상담 완료": "프로그램 신청 연결",
            "프로그램 참여": "후속 체크인 요청",
            "추후 관찰": "후속 체크인 요청",
        }
        action = preferred_by_status.get(status, allowed_actions[0])
        if action not in allowed_actions:
            action = allowed_actions[0]
        checks = [str(item) for item in context["allowed_checks"]]
        selected_checks = ["학생 참여 의사", "최근 체크인 변화"]
        selected_checks = [item for item in selected_checks if item in checks]
        if not selected_checks:
            selected_checks = checks[:1]
        reason = (
            f"현재 지원 상태가 {status}이므로 학생의 의사와 최근 변화를 확인한 뒤 "
            f"{action}을 검토할 수 있습니다."
        )
        return validate_next_action_suggestion(
            {
                "action": action,
                "reason": reason,
                "check_before_action": selected_checks,
            },
            set(allowed_actions),
            set(checks),
        )
