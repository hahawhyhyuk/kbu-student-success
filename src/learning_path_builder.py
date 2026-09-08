"""선정된 DB 교과목과 AI 설명을 하나의 검증된 학습경로로 결합한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.ai import AIProvider, ResilientAIProvider, validate_learning_path_explanation
from src.course_recommender import CoursePathSelection


@dataclass(frozen=True)
class LearningPathResult:
    selection: CoursePathSelection
    explanation: dict[str, Any]
    provider_name: str
    fallback_used: bool
    warning: str | None = None


class LearningPathBuilder:
    """AI가 과목을 바꾸지 못하도록 ID를 재검증한 뒤 설명을 결합한다."""

    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def build(
        self, profile: Mapping[str, Any], selection: CoursePathSelection
    ) -> LearningPathResult:
        courses = [
            {
                "course_id": course.course_id,
                "course_name": course.course_name,
                "competencies": "|".join(course.competencies),
                "difficulty": course.difficulty,
                "offering_departments": "|".join(course.offering_departments),
                "offered_semesters": "|".join(
                    str(item) for item in course.offered_semesters
                ),
                "course_area": "|".join(course.course_area),
                "class_method": "|".join(course.class_method),
                "course_type": "|".join(course.course_type),
                "ncs_type": "|".join(course.ncs_type),
                "recommendation_basis": course.recommendation_basis,
            }
            for course in selection.courses
        ]
        explanation = validate_learning_path_explanation(
            self.provider.explain_learning_path(dict(profile), courses),
            {course.course_id for course in selection.courses},
        )
        if isinstance(self.provider, ResilientAIProvider):
            provider_name = self.provider.last_provider_name
            warning = self.provider.last_error
        else:
            provider_name = self.provider.provider_name
            warning = (
                "Gemini API key가 없어 Mock 학습경로 설명을 사용했습니다."
                if provider_name == "mock" else None
            )
        return LearningPathResult(
            selection=selection,
            explanation=explanation,
            provider_name=provider_name,
            fallback_used=provider_name == "mock",
            warning=warning,
        )
