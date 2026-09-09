"""DB 교과목만 대상으로 개인 맞춤 역량 학습경로 과목을 선정한다."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from src.recommendation_interest import (
    infer_interest_evidence,
    supported_interest_labels,
)
from src.similarity import ResilientSimilarityBackend, SimilarityBackend, create_similarity_backend
from src.utils import load_app_config


INTEREST_COMPETENCY_MAP = {
    "데이터·AI": {"데이터이해", "데이터활용", "기초분석", "AI이해"},
    "경영·마케팅": {"마케팅", "시장분석", "고객분석", "사업기획"},
    "콘텐츠·디자인": {"콘텐츠기획", "콘텐츠제작", "시각화", "디자인이해"},
    "서비스": {"서비스기획", "서비스설계", "고객경험", "프로젝트관리"},
    "보건": {"보건서비스", "건강정보", "보건기획"},
    "상담·복지": {"상담기초", "공감소통", "사례관리", "복지이해"},
}
DIFFICULTY_ORDER = {"기초": 1, "중급": 2, "심화": 3}


@dataclass(frozen=True)
class CourseRecommendation:
    course_id: str
    course_name: str
    department: str
    description: str
    competencies: tuple[str, ...]
    grade_level: int
    semester: int
    difficulty: str
    credit: int
    prerequisites: tuple[str, ...]
    score: float
    reason: str
    is_home_department: bool
    source_course_id: str = ""
    offering_departments: tuple[str, ...] = ()
    offered_semesters: tuple[int, ...] = ()
    grade_levels: tuple[int, ...] = ()
    course_area: tuple[str, ...] = ()
    class_method: tuple[str, ...] = ()
    course_type: tuple[str, ...] = ()
    ncs_type: tuple[str, ...] = ()
    recommendation_basis: str = ""
    registration_check_required: bool = False


@dataclass(frozen=True)
class CoursePathSelection:
    courses: tuple[CourseRecommendation, ...]
    covered_competencies: tuple[str, ...]
    similarity_backend: str
    covered_source_attributes: tuple[str, ...] = ()
    warning: str | None = None


@dataclass(frozen=True)
class _CourseCandidate:
    """Hard filter를 통과한 개별 교과목의 선정 후보를 보관한다."""

    course: dict[str, Any]
    score: float
    competencies: frozenset[str]
    source_attributes: frozenset[str]
    coverage_terms: frozenset[str]
    reason: str
    is_home_department: bool
    matched_interests: frozenset[str]


def _split_values(value: Any) -> set[str]:
    return {item.strip() for item in str(value or "").split("|") if item.strip()}


def _is_available(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _course_departments(course: Mapping[str, Any]) -> set[str]:
    """실제 개설학과 목록이 있으면 사용하고 기존 master는 단일 학과를 사용한다."""

    departments = _split_values(course.get("offering_departments", ""))
    department = str(course.get("department", "")).strip()
    if department:
        departments.add(department)
    return departments


def _source_attributes(course: Mapping[str, Any]) -> set[str]:
    """실제 원본의 교과 분류값만 다양성 근거로 반환한다."""

    attributes: set[str] = set()
    for column in ("course_area", "class_method", "course_type", "ncs_type"):
        attributes.update(_split_values(course.get(column, "")))
    return attributes


def _course_name_family(value: Any) -> str:
    """과목명 끝의 (1)·(2) 같은 연속편 표기만 제거해 중복 계열을 찾는다."""

    normalized = re.sub(r"\s+", "", str(value or "")).lower()
    return re.sub(r"[\(\[]?\d+[\)\]]?$", "", normalized)


class CourseRecommender:
    """Hard filter와 greedy competency coverage로 3~4개 과목을 선정한다."""

    def __init__(
        self,
        similarity_backend: SimilarityBackend | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or load_app_config()
        self.similarity_backend = similarity_backend or create_similarity_backend(self.config)

    @staticmethod
    def _profile_interests(profile: Mapping[str, Any], analysis: Mapping[str, Any]) -> set[str]:
        interests = _split_values(profile.get("interest_fields", ""))
        interests.update(str(item) for item in analysis.get("interests", []))
        return supported_interest_labels(interests)

    @staticmethod
    def _profile_text(profile: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
        return " ".join(
            str(value).replace("|", " ")
            for value in (
                profile.get("natural_language_concern", ""),
                profile.get("interest_fields", ""),
                profile.get("desired_job", ""),
                analysis.get("summary", ""),
                " ".join(analysis.get("interests", [])),
                " ".join(analysis.get("desired_jobs", [])),
            )
            if str(value).strip()
        )

    @staticmethod
    def _course_text(course: Mapping[str, Any]) -> str:
        return " ".join(
            str(course.get(column, "")).replace("|", " ")
            for column in (
                "course_name",
                "description",
                "learning_objectives",
                "competencies",
                "related_jobs",
                "related_interests",
                "offering_departments",
                "course_area",
                "class_method",
                "course_type",
                "ncs_type",
            )
        )

    def recommend(
        self,
        profile: Mapping[str, Any],
        analysis: Mapping[str, Any],
        courses: pd.DataFrame,
        completed_course_ids: Sequence[str],
        top_k: int | None = None,
    ) -> CoursePathSelection:
        """수강 조건을 필터링하고 소속 학과 우선·타과 허용 경로를 반환한다.

        소속 학과는 후보점수에 설정 가점을 받고, 수강 가능한 소속 학과
        과목이 있으면 설정된 최소 개수를 먼저 확보한다. 타과 과목은
        관심분야·희망직무·필요역량이 더 잘 맞을 때 같은 후보군에서 선정된다.
        """

        config = self.config["course_recommendation"]
        weights = config["weights"]
        limit = int(top_k or config["top_k"])
        limit = min(4, max(int(config["minimum_courses"]), limit))
        department_policy = config.get("department_policy", {})
        allow_cross_department = bool(
            department_policy.get("allow_cross_department", True)
        )
        home_department_bonus = float(
            department_policy.get("home_department_bonus", 0.0)
        )
        minimum_home_courses = int(
            department_policy.get("minimum_home_department_courses", 0)
        )
        source_policy = config.get("source_backed_policy", {})
        source_home_department_bonus = float(
            source_policy.get("home_department_bonus", home_department_bonus)
        )
        source_candidate_pool_size = int(
            source_policy.get("candidate_pool_size", config["candidate_pool_size"])
        )
        source_maximum_cross_courses = int(
            source_policy.get("maximum_cross_department_courses", limit)
        )
        max_general_education_courses = int(
            source_policy.get("max_general_education_courses", 1)
        )
        max_same_name_family = int(
            source_policy.get("max_same_name_family", 1)
        )
        minimum_interest_courses = int(
            config.get("minimum_interest_aligned_courses", 0)
        )
        if not 0.0 <= home_department_bonus <= 1.0:
            raise ValueError("소속 학과 추천 가점은 0과 1 사이여야 합니다.")
        if not 0.0 <= source_home_department_bonus <= 1.0:
            raise ValueError("실제 개설강좌의 소속 학과 가점은 0과 1 사이여야 합니다.")
        if not 0 <= minimum_home_courses <= limit:
            raise ValueError("소속 학과 최소 과목 수가 추천 개수를 벗어났습니다.")
        if not 0 <= source_maximum_cross_courses <= limit:
            raise ValueError("타과 최대 과목 수가 추천 개수를 벗어났습니다.")
        if source_candidate_pool_size < limit:
            raise ValueError("실제 개설강좌 후보군은 추천 개수 이상이어야 합니다.")
        if max_general_education_courses < 0 or max_same_name_family < 1:
            raise ValueError("실제 개설강좌 다양성 설정을 확인해 주세요.")
        if not 0 <= minimum_interest_courses <= limit:
            raise ValueError("관심 분야 우선 과목 수가 추천 개수를 벗어났습니다.")
        grade = int(profile.get("grade", 1))
        home_department = str(profile.get("department", "")).strip()
        completed = {str(course_id) for course_id in completed_course_ids}
        all_ids = set(courses["course_id"].astype(str))
        eligible_records = []
        for course in courses.to_dict(orient="records"):
            course_id = str(course["course_id"])
            prerequisites = _split_values(course.get("prerequisites", ""))
            course_departments = _course_departments(course)
            is_home_department = bool(
                home_department and home_department in course_departments
            )
            if course_id in completed:
                continue
            if int(course["grade_level"]) > grade or not _is_available(course["is_available"]):
                continue
            if not prerequisites.issubset(completed):
                continue
            if not prerequisites.issubset(all_ids):
                continue
            if (
                home_department
                and not allow_cross_department
                and not is_home_department
            ):
                continue
            if (
                home_department
                and not is_home_department
                and str(course.get("cross_department_status", "")).strip()
                == "restricted"
            ):
                continue
            eligible_records.append(course)
        if len(eligible_records) < int(config["minimum_courses"]):
            raise ValueError("필터를 충족하는 교과목이 3개 미만입니다.")

        query = self._profile_text(profile, analysis)
        documents = [self._course_text(course) for course in eligible_records]
        semantic_scores = self.similarity_backend.similarities(query, documents)
        interests = self._profile_interests(profile, analysis)
        desired_jobs = {str(profile.get("desired_job", "")).strip()}
        desired_jobs.update(str(item) for item in analysis.get("desired_jobs", []))
        desired_jobs.discard("")
        needed_competencies: set[str] = set()
        for interest in interests:
            needed_competencies.update(INTEREST_COMPETENCY_MAP.get(interest, set()))

        candidates: list[_CourseCandidate] = []
        for course, semantic_score in zip(eligible_records, semantic_scores):
            explicit_course_interests = _split_values(
                course["related_interests"]
            )
            inferred_interest_evidence = infer_interest_evidence(
                self._course_text(course),
                interests,
            )
            course_interests = explicit_course_interests | set(
                inferred_interest_evidence
            )
            matched_interests = interests & course_interests
            course_jobs = _split_values(course["related_jobs"])
            competencies = _split_values(course["competencies"])
            source_attributes = _source_attributes(course)
            interest_match = len(interests & course_interests) / len(interests) if interests else 0.0
            job_match = (
                sum(any(job in related or related in job for related in course_jobs) for job in desired_jobs)
                / len(desired_jobs)
                if desired_jobs else 0.0
            )
            competency_match = (
                len(needed_competencies & competencies) / len(needed_competencies)
                if needed_competencies else 0.0
            )
            feasibility = 1.0 if int(course["grade_level"]) < grade else 0.85
            weighted_components = [
                (float(weights["semantic_similarity"]), float(semantic_score)),
                (float(weights["feasibility"]), feasibility),
            ]
            if course_interests:
                weighted_components.append(
                    (float(weights["interest_match"]), interest_match)
                )
            if course_jobs:
                weighted_components.append((float(weights["job_match"]), job_match))
            if competencies:
                weighted_components.append(
                    (float(weights["competency_coverage"]), competency_match)
                )
            active_weight = sum(weight for weight, _ in weighted_components)
            score = 100 * sum(
                weight * value for weight, value in weighted_components
            ) / active_weight
            course_departments = _course_departments(course)
            is_home_department = bool(
                home_department and home_department in course_departments
            )
            source_backed = bool(str(course.get("catalog_source", "")).strip())
            if is_home_department:
                if source_backed:
                    score += 100 * source_home_department_bonus / max(
                        1, len(course_departments)
                    )
                else:
                    score += 100 * home_department_bonus
            reasons = ["시스템 보유자료 기준 수강 필터 통과"]
            if is_home_department:
                reasons.append("소속 학과 우선")
            elif home_department:
                reasons.append("타과 과목으로 역량 범위 확장")
            if matched_interests:
                inferred_matches = sorted(
                    matched_interests & set(inferred_interest_evidence)
                )
                if inferred_matches:
                    matched_keywords = sorted(
                        {
                            keyword
                            for label in inferred_matches
                            for keyword in inferred_interest_evidence[label]
                        }
                    )
                    reasons.append(
                        "관심분야 직접 연결: "
                        + "·".join(sorted(matched_interests))
                        + " (과목명·설명 근거: "
                        + "·".join(matched_keywords[:4])
                        + ")"
                    )
                else:
                    reasons.append(
                        "관심분야 직접 연결: "
                        + "·".join(sorted(matched_interests))
                    )
            if job_match:
                reasons.append("희망직무와 연결")
            if needed_competencies & competencies:
                reasons.append("필요역량 보완")
            if source_backed:
                reasons.append("실제 개설정보 확인")
                if (
                    not is_home_department
                    and str(course.get("cross_department_status", "")).strip()
                    == "unknown"
                ):
                    reasons.append("타과 수강 가능 여부 확인 필요")
            reasons.append(
                "대화·과목 정보 의미 유사도 참고값 "
                f"{float(semantic_score):.0%}"
            )
            coverage_terms = set(competencies or source_attributes)
            coverage_terms.update(
                f"관심:{label}" for label in matched_interests
            )
            candidates.append(
                _CourseCandidate(
                    course=course,
                    score=score,
                    competencies=frozenset(competencies),
                    source_attributes=frozenset(source_attributes),
                    coverage_terms=frozenset(coverage_terms),
                    reason=" · ".join(reasons) or "체크인 내용과 연관",
                    is_home_department=is_home_department,
                    matched_interests=frozenset(matched_interests),
                )
            )

        interest_gate_fallback = False
        if interests:
            interest_relevant_candidates = [
                item
                for item in candidates
                if item.is_home_department or item.matched_interests
            ]
            if len(interest_relevant_candidates) >= int(
                config["minimum_courses"]
            ):
                candidates = interest_relevant_candidates
            else:
                interest_gate_fallback = True

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                -item.score,
                not item.is_home_department,
                str(item.course["course_id"]),
            ),
        )
        has_source_backed_candidates = any(
            bool(str(item.course.get("catalog_source", "")).strip())
            for item in ranked_candidates
        )
        pool_size = (
            source_candidate_pool_size
            if has_source_backed_candidates
            else int(config["candidate_pool_size"])
        )
        candidate_pool = ranked_candidates[:pool_size]
        pool_course_ids = {
            str(item.course["course_id"]) for item in candidate_pool
        }
        for item in ranked_candidates:
            course_id = str(item.course["course_id"])
            if item.matched_interests and course_id not in pool_course_ids:
                candidate_pool.append(item)
                pool_course_ids.add(course_id)
        source_home_candidates = [
            item for item in ranked_candidates if item.is_home_department
        ]
        if has_source_backed_candidates and source_home_candidates:
            required_source_home_courses = max(
                minimum_home_courses,
                limit - source_maximum_cross_courses,
            )
            for item in source_home_candidates[:required_source_home_courses]:
                course_id = str(item.course["course_id"])
                if course_id not in pool_course_ids:
                    candidate_pool.append(item)
                    pool_course_ids.add(course_id)
            effective_source_cross_limit = source_maximum_cross_courses
        else:
            # 소속 학과 개설강좌 자체가 없을 때만 타과 경로 fallback을 허용한다.
            effective_source_cross_limit = limit

        selected: list[_CourseCandidate] = []
        covered: set[str] = set()
        covered_source_attributes: set[str] = set()
        covered_terms: set[str] = set()

        def can_select(item: _CourseCandidate) -> bool:
            source_backed = bool(
                str(item.course.get("catalog_source", "")).strip()
            )
            if source_backed and not item.is_home_department:
                selected_cross_count = sum(
                    not selected_item.is_home_department
                    for selected_item in selected
                )
                if selected_cross_count >= effective_source_cross_limit:
                    return False
            if not source_backed:
                return True
            family = _course_name_family(item.course.get("course_name", ""))
            family_count = sum(
                _course_name_family(selected_item.course.get("course_name", ""))
                == family
                for selected_item in selected
            )
            if family and family_count >= max_same_name_family:
                return False
            is_general = "교양" in _split_values(
                item.course.get("course_area", "")
            )
            selected_general_count = sum(
                "교양" in _split_values(
                    selected_item.course.get("course_area", "")
                )
                for selected_item in selected
            )
            return not (
                is_general
                and selected_general_count >= max_general_education_courses
            )

        def selection_value(item: _CourseCandidate) -> tuple[float, int]:
            new_term_ratio = len(item.coverage_terms - covered_terms) / max(
                1, len(item.coverage_terms)
            )
            course_id = str(item.course["course_id"])
            numeric_id = int(course_id[1:]) if course_id[1:].isdigit() else 0
            return (
                item.score
                + 100 * float(config["diversity_bonus"]) * new_term_ratio,
                -numeric_id,
            )

        def select(item: _CourseCandidate) -> None:
            """선정 항목과 이후 다양성 계산용 누적 근거를 함께 갱신한다."""

            selected.append(item)
            covered.update(item.competencies)
            covered_source_attributes.update(item.source_attributes)
            covered_terms.update(item.coverage_terms)

        home_candidates = source_home_candidates
        required_home_courses = min(
            minimum_home_courses,
            len(home_candidates),
            limit,
        )
        while home_candidates and len(selected) < required_home_courses:
            selectable_home = [item for item in home_candidates if can_select(item)]
            if not selectable_home:
                break
            best_home = max(selectable_home, key=selection_value)
            select(best_home)
            home_candidates.remove(best_home)

        remaining = [item for item in candidate_pool if item not in selected]
        available_interest_candidates = [
            item for item in candidate_pool if item.matched_interests
        ]
        required_interest_courses = min(
            minimum_interest_courses,
            len(available_interest_candidates),
            limit,
        )
        while (
            remaining
            and len(selected) < limit
            and sum(bool(item.matched_interests) for item in selected)
            < required_interest_courses
        ):
            selectable_interest = [
                item
                for item in remaining
                if item.matched_interests and can_select(item)
            ]
            if not selectable_interest:
                break
            best_interest = max(selectable_interest, key=selection_value)
            select(best_interest)
            remaining.remove(best_interest)

        while remaining and len(selected) < limit:
            selectable = [item for item in remaining if can_select(item)]
            if not selectable:
                break
            best = max(selectable, key=selection_value)
            select(best)
            remaining.remove(best)
        if len(selected) < int(config["minimum_courses"]):
            raise ValueError("학습경로를 구성할 교과목이 부족합니다.")

        selected.sort(
            key=lambda item: (
                DIFFICULTY_ORDER.get(str(item.course["difficulty"]), 99),
                int(item.course["grade_level"]),
                str(item.course["course_id"]),
            )
        )
        recommendations = tuple(
            CourseRecommendation(
                course_id=str(item.course["course_id"]),
                course_name=str(item.course["course_name"]),
                department=str(item.course["department"]),
                description=str(item.course["description"]),
                competencies=tuple(sorted(item.competencies)),
                grade_level=int(item.course["grade_level"]),
                semester=int(item.course["semester"]),
                difficulty=str(item.course["difficulty"]),
                credit=int(item.course["credit"]),
                prerequisites=tuple(
                    sorted(_split_values(item.course["prerequisites"]))
                ),
                score=round(min(100.0, max(0.0, item.score)), 2),
                reason=item.reason,
                is_home_department=item.is_home_department,
                source_course_id=str(
                    item.course.get("source_course_id", "")
                ),
                offering_departments=tuple(
                    sorted(_course_departments(item.course))
                ),
                offered_semesters=tuple(
                    sorted(
                        int(value)
                        for value in _split_values(
                            item.course.get(
                                "offered_semesters", item.course["semester"]
                            )
                        )
                        if str(value).isdigit()
                    )
                ),
                grade_levels=tuple(
                    sorted(
                        int(value)
                        for value in _split_values(
                            item.course.get(
                                "grade_levels", item.course["grade_level"]
                            )
                        )
                        if str(value).isdigit()
                    )
                ),
                course_area=tuple(
                    sorted(_split_values(item.course.get("course_area", "")))
                ),
                class_method=tuple(
                    sorted(_split_values(item.course.get("class_method", "")))
                ),
                course_type=tuple(
                    sorted(_split_values(item.course.get("course_type", "")))
                ),
                ncs_type=tuple(
                    sorted(_split_values(item.course.get("ncs_type", "")))
                ),
                recommendation_basis=str(
                    item.course.get("recommendation_basis", "")
                ),
                registration_check_required=str(
                    item.course.get("registration_check_required", "")
                ).strip().lower()
                in {"true", "1", "yes"},
            )
            for item in selected
        )
        backend_name = self.similarity_backend.backend_name
        warnings: list[str] = []
        if home_department and minimum_home_courses and not any(
            item.is_home_department for item in candidates
        ):
            warnings.append(
                "수강 조건을 충족하는 소속 학과 과목이 없어 "
                "관심분야·희망직무와 연결된 타과 과목으로 구성했습니다."
            )
        if interest_gate_fallback:
            warnings.append(
                "명시한 관심 분야와 직접 연결되는 수강 가능 과목이 부족해 "
                "일부 과목은 학과·학년·대화 맥락을 기준으로 보완했습니다."
            )
        selected_interest_count = sum(
            bool(item.matched_interests) for item in selected
        )
        if interests and selected_interest_count < required_interest_courses:
            warnings.append(
                "과목 영역·타과 수강 제한으로 관심 분야 우선 과목 수를 모두 "
                "확보하지 못했습니다."
            )
        if isinstance(self.similarity_backend, ResilientSimilarityBackend):
            backend_name = self.similarity_backend.last_backend_name
            if self.similarity_backend.last_warning:
                warnings.append(self.similarity_backend.last_warning)
        if any(item.course.get("catalog_source") for item in selected):
            warnings.append(
                "실제 개설강좌의 과목명·학과·교과구분·NCS·수업방식만으로 "
                "선정했습니다. 강의계획서 기반 역량·직무 정보는 포함하지 않았습니다."
            )
        if any(
            str(item.course.get("completion_data_status", "")).strip()
            for item in selected
        ):
            warnings.append(
                "실제 이수내역과 선수과목 정보가 연결되지 않아 최종 수강 가능 여부는 "
                "교직원이 확인해야 합니다."
            )
        return CoursePathSelection(
            courses=recommendations,
            covered_competencies=tuple(sorted(covered)),
            similarity_backend=backend_name,
            covered_source_attributes=tuple(sorted(covered_source_attributes)),
            warning=" ".join(warnings) or None,
        )
