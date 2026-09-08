"""반복되는 개인 학습경로를 교육과정 개발 후보로 집계한다."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

import pandas as pd

from src.ai import AIProvider, MockAIProvider, ResilientAIProvider
from src.course_recommender import CoursePathSelection, CourseRecommender
from src.database import LearningPathDataStore
from src.risk_service import select_latest_student_snapshots
from src.utils import load_app_config


CANDIDATE_PRIORITY = "수요 후보"


@dataclass(frozen=True)
class LearningPathDemand:
    """학생 한 명에게 추천된 DB 교과목 경로의 집계용 표현."""

    student_id: str
    department: str
    desired_job: str
    selection: CoursePathSelection

    @property
    def course_ids(self) -> tuple[str, ...]:
        return tuple(course.course_id for course in self.selection.courses)

    @property
    def competencies(self) -> tuple[str, ...]:
        return self.selection.covered_competencies


@dataclass(frozen=True)
class MicrodegreeCandidate:
    """교직원이 추가 검토할 수 있는 신규 교육과정 개발 후보."""

    candidate_id: str
    candidate_name: str
    description: str
    student_ids: tuple[str, ...]
    departments: tuple[str, ...]
    course_ids: tuple[str, ...]
    course_names: tuple[str, ...]
    competencies: tuple[str, ...]
    related_jobs: tuple[str, ...]
    average_similarity: float
    priority: str
    demand_share: float
    status: str

    @property
    def student_count(self) -> int:
        return len(self.student_ids)

    @property
    def department_count(self) -> int:
        return len(self.departments)


@dataclass(frozen=True)
class MicrodegreeAnalysisResult:
    """전체 대상 학생 경로와 기준을 통과한 개발 후보 분석 결과."""

    candidates: tuple[MicrodegreeCandidate, ...]
    demands: tuple[LearningPathDemand, ...]
    target_student_count: int
    skipped_student_count: int
    minimum_repeated_courses: int
    provider_name: str
    warning: str | None = None


def jaccard_similarity(
    left: Sequence[str] | set[str], right: Sequence[str] | set[str]
) -> float:
    """두 교과목 집합의 Jaccard 유사도를 0~1로 반환한다."""

    left_set = {str(item) for item in left}
    right_set = {str(item) for item in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


class MicrodegreeCandidateService:
    """개인 경로 생성, 반복 교과목 수요 집계, 후보 검증을 담당한다."""

    def __init__(
        self,
        course_recommender: CourseRecommender | None = None,
        provider: AIProvider | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_app_config()
        self.course_recommender = course_recommender or CourseRecommender(
            config=self.config
        )
        self.provider = provider or MockAIProvider()

    @staticmethod
    def _analysis_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        """대량 분석에서 생성형 AI 호출 없이 명시된 관심·직무를 정규화한다."""

        interests = [
            item.strip()
            for item in str(profile.get("interest_fields", "")).split("|")
            if item.strip()
        ]
        desired_job = str(profile.get("desired_job", "")).strip()
        return {
            "major_concern": False,
            "learning_difficulty": False,
            "career_uncertainty": not bool(desired_job),
            "interests": interests,
            "desired_jobs": [desired_job] if desired_job else [],
            "support_needs": ["진로 탐색"],
            "summary": str(profile.get("natural_language_concern", "")),
        }

    def build_demands(
        self,
        snapshots: pd.DataFrame,
        courses: pd.DataFrame,
        completed_courses: pd.DataFrame,
    ) -> tuple[tuple[LearningPathDemand, ...], int, int]:
        """최신 위험 대상 학생별로 수강 가능한 3~4개 교과 경로를 생성한다.

        Parameters:
            snapshots: 위험 서비스가 만든 주차별 canonical 스냅샷.
            courses: Repository의 교과목 master.
            completed_courses: Repository의 학생별 이수내역.

        Returns:
            생성된 경로, 대상 학생 수, 필터로 경로를 만들지 못한 학생 수.

        Assumptions:
            민감정보는 읽지 않으며 전공적응 또는 진로 위험 기준만 대상 선정에 사용한다.
        """

        if snapshots.empty or courses.empty:
            raise ValueError("학습경로 수요 분석에 필요한 데이터가 없습니다.")
        if courses["course_id"].astype(str).duplicated().any():
            raise ValueError("교과목 master의 course_id는 중복될 수 없습니다.")
        config = self.config["microdegree_candidate"]
        risk_min = float(config["target_risk_min"])
        latest = select_latest_student_snapshots(snapshots)
        targets = latest[
            (latest["major_adaptation_risk"] >= risk_min)
            | (latest["career_risk"] >= risk_min)
        ].sort_values("student_id")

        completed_map = {
            str(student_id): set(group["course_id"].astype(str))
            for student_id, group in completed_courses[
                completed_courses["completion_status"] == "completed"
            ].groupby("student_id")
        }
        demands: list[LearningPathDemand] = []
        skipped = 0
        for profile in targets.to_dict(orient="records"):
            student_id = str(profile["student_id"])
            try:
                selection = self.course_recommender.recommend(
                    profile=profile,
                    analysis=self._analysis_from_profile(profile),
                    courses=courses,
                    completed_course_ids=completed_map.get(student_id, set()),
                )
            except ValueError:
                skipped += 1
                continue
            demands.append(
                LearningPathDemand(
                    student_id=student_id,
                    department=str(profile["department"]),
                    desired_job=str(profile.get("desired_job", "")).strip(),
                    selection=selection,
                )
            )
        return tuple(demands), len(targets), skipped

    def find_repeated_course_groups(
        self, demands: Sequence[LearningPathDemand]
    ) -> tuple[tuple[LearningPathDemand, ...], ...]:
        """같은 교과목 조합이 반복된 학생 묶음을 수요 규모순으로 반환한다.

        4과목 경로 전체 일치를 요구하지 않고 설정된 개수의 반복 교과목만
        집계한다. 여러 학과 후보와 한 학과 안에서 충분히 반복된 특화 후보를
        모두 보존한다.
        """

        config = self.config["microdegree_candidate"]
        repeat_count = int(config["repeated_course_count"])
        minimum_students = int(config["minimum_students"])
        minimum_departments = int(config["minimum_departments"])
        single_department_minimum = int(
            config["single_department_minimum_students"]
        )
        if repeat_count < 2:
            raise ValueError("반복 교과목 기준은 2개 이상이어야 합니다.")
        if minimum_students < 1 or single_department_minimum < minimum_students:
            raise ValueError("마이크로디그리 후보의 최소 수요 인원을 확인해 주세요.")
        if minimum_departments < 2:
            raise ValueError("다학과 후보의 최소 학과 수는 2개 이상이어야 합니다.")

        grouped: dict[tuple[str, ...], list[LearningPathDemand]] = {}
        for demand in sorted(demands, key=lambda item: item.student_id):
            course_ids = tuple(sorted(set(demand.course_ids)))
            for course_group in combinations(course_ids, repeat_count):
                grouped.setdefault(course_group, []).append(demand)

        eligible: list[tuple[LearningPathDemand, ...]] = []
        seen_student_sets: set[tuple[str, ...]] = set()
        ordered_groups = sorted(
            grouped.items(),
            key=lambda item: (
                -len(item[1]),
                -len({demand.department for demand in item[1]}),
                item[0],
            ),
        )
        for _, group in ordered_groups:
            departments = {demand.department for demand in group}
            is_multi_department = (
                len(group) >= minimum_students
                and len(departments) >= minimum_departments
            )
            is_department_specific = (
                len(departments) == 1
                and len(group) >= single_department_minimum
            )
            student_ids = tuple(sorted(demand.student_id for demand in group))
            if (
                not (is_multi_department or is_department_specific)
                or student_ids in seen_student_sets
            ):
                continue
            eligible.append(tuple(group))
            seen_student_sets.add(student_ids)
        return tuple(eligible)

    @staticmethod
    def _average_similarity(cluster: Sequence[LearningPathDemand]) -> float:
        similarities = [
            jaccard_similarity(left.course_ids, right.course_ids)
            for index, left in enumerate(cluster)
            for right in cluster[index + 1 :]
        ]
        return sum(similarities) / len(similarities) if similarities else 1.0

    def _candidate_from_cluster(
        self,
        cluster: Sequence[LearningPathDemand],
        courses: pd.DataFrame,
        candidate_index: int,
        demand_population: int,
        generated_description: Mapping[str, Any] | None = None,
    ) -> MicrodegreeCandidate | None:
        config = self.config["microdegree_candidate"]
        if len(cluster) < int(config["minimum_students"]):
            return None
        departments = tuple(sorted({item.department for item in cluster}))
        is_multi_department = len(departments) >= int(
            config["minimum_departments"]
        )
        is_department_specific = (
            len(departments) == 1
            and len(cluster) >= int(config["single_department_minimum_students"])
        )
        if not (is_multi_department or is_department_specific):
            return None

        minimum_common_count = max(
            2,
            math.ceil(len(cluster) * float(config["common_item_min_ratio"])),
        )
        competency_counts = Counter(
            competency
            for demand in cluster
            for competency in set(demand.competencies)
        )
        job_counts = Counter(
            demand.desired_job for demand in cluster if demand.desired_job
        )
        common_competencies = tuple(
            item
            for item, count in sorted(
                competency_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if count >= minimum_common_count
        )
        common_jobs = tuple(
            item
            for item, count in sorted(
                job_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if count >= minimum_common_count
        )
        course_counts = Counter(
            course_id for demand in cluster for course_id in set(demand.course_ids)
        )
        course_limit = min(4, max(3, int(config["candidate_course_count"])))
        candidate_course_ids = tuple(
            course_id
            for course_id, _ in sorted(
                course_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )[:course_limit]
        )
        course_name_map = courses.set_index("course_id")["course_name"].to_dict()
        if any(course_id not in course_name_map for course_id in candidate_course_ids):
            raise ValueError("후보 군집에 DB에 없는 교과목 ID가 포함되었습니다.")
        candidate_rows = courses[
            courses["course_id"].astype(str).isin(candidate_course_ids)
        ]
        source_backed = bool(
            "catalog_source" in candidate_rows.columns
            and candidate_rows["catalog_source"].astype(str).str.strip().ne("").all()
        )
        if not common_competencies and not common_jobs and not source_backed:
            return None
        course_names = tuple(
            str(course_name_map[course_id])
            for course_id in candidate_course_ids
        )
        description_input = {
            "student_count": len(cluster),
            "department_count": len(departments),
            "course_names": course_names,
            "competencies": common_competencies[:6],
            "related_jobs": common_jobs[:5],
        }
        generated = (
            dict(generated_description)
            if generated_description is not None
            else self.provider.describe_microdegree_candidate(description_input)
        )
        if demand_population < 1 or len(cluster) > demand_population:
            raise ValueError("후보 수요 인원과 전체 경로 인원을 확인해 주세요.")
        demand_share = round(len(cluster) / demand_population, 4)
        return MicrodegreeCandidate(
            candidate_id=f"MDC{candidate_index:03d}",
            candidate_name=str(generated["candidate_name"]),
            description=str(generated["description"]),
            student_ids=tuple(sorted(item.student_id for item in cluster)),
            departments=departments,
            course_ids=candidate_course_ids,
            course_names=course_names,
            competencies=common_competencies[:6],
            related_jobs=common_jobs[:5],
            average_similarity=round(self._average_similarity(cluster), 3),
            priority=CANDIDATE_PRIORITY,
            demand_share=demand_share,
            status=str(generated["status"]),
        )

    def analyze(
        self,
        snapshots: pd.DataFrame,
        courses: pd.DataFrame,
        completed_courses: pd.DataFrame,
    ) -> MicrodegreeAnalysisResult:
        """학생 경로의 반복 교과목 조합을 집계해 개발 후보를 반환한다."""

        demands, target_count, skipped_count = self.build_demands(
            snapshots=snapshots,
            courses=courses,
            completed_courses=completed_courses,
        )
        candidates: list[MicrodegreeCandidate] = []
        seen_course_sets: set[frozenset[str]] = set()
        for cluster in self.find_repeated_course_groups(demands):
            candidate = self._candidate_from_cluster(
                cluster=cluster,
                courses=courses,
                candidate_index=len(candidates) + 1,
                demand_population=len(demands),
                generated_description={
                    "candidate_name": "교육과정 개발 후보",
                    "description": "반복 학습경로 수요를 설명하는 중입니다.",
                    "status": "교육과정 검토 후보",
                },
            )
            candidate_course_set = (
                frozenset(candidate.course_ids) if candidate is not None else frozenset()
            )
            if candidate is not None and candidate_course_set not in seen_course_sets:
                candidates.append(candidate)
                seen_course_sets.add(candidate_course_set)

        description_inputs = [
            {
                "student_count": candidate.student_count,
                "department_count": candidate.department_count,
                "course_names": candidate.course_names,
                "competencies": candidate.competencies,
                "related_jobs": candidate.related_jobs,
            }
            for candidate in candidates
        ]
        generated_descriptions = self.provider.describe_microdegree_candidates(
            description_inputs
        )
        if len(generated_descriptions) != len(candidates):
            raise ValueError("교육과정 후보 설명 개수가 생성된 후보 수와 다릅니다.")
        candidates = [
            replace(
                candidate,
                candidate_name=str(generated["candidate_name"]),
                description=str(generated["description"]),
                status=str(generated["status"]),
            )
            for candidate, generated in zip(candidates, generated_descriptions)
        ]

        provider_name = self.provider.provider_name
        warning = None
        if isinstance(self.provider, ResilientAIProvider):
            provider_name = self.provider.last_provider_name
            warning = self.provider.last_error
        return MicrodegreeAnalysisResult(
            candidates=tuple(candidates),
            demands=demands,
            target_student_count=target_count,
            skipped_student_count=skipped_count,
            minimum_repeated_courses=int(
                self.config["microdegree_candidate"]["repeated_course_count"]
            ),
            provider_name=provider_name,
            warning=warning,
        )

    @staticmethod
    def persist_analysis(
        result: MicrodegreeAnalysisResult,
        store: LearningPathDataStore | None = None,
    ) -> str:
        """검증된 학생별 경로와 개발 후보를 한 SQLite 배치로 저장한다."""

        active_store = store or LearningPathDataStore()
        paths = [
            {
                "student_id": demand.student_id,
                "path_name": "교육과정 수요 분석용 개인 맞춤 역량 경로",
                "related_job": demand.desired_job,
                "competencies": demand.competencies,
                "reason": (
                    "학생의 명시된 관심분야·희망직무와 이수·선수·학년 "
                    "조건을 반영해 코드가 선정한 경로입니다."
                ),
                "courses": [
                    {
                        "course_id": course.course_id,
                        "sequence": sequence,
                        "score": course.score,
                        "reason": course.reason,
                    }
                    for sequence, course in enumerate(
                        demand.selection.courses, start=1
                    )
                ],
            }
            for demand in result.demands
        ]
        candidates = [
            {
                "candidate_code": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "student_count": candidate.student_count,
                "department_count": candidate.department_count,
                "departments": candidate.departments,
                "course_ids": candidate.course_ids,
                "course_names": candidate.course_names,
                "competencies": candidate.competencies,
                "related_jobs": candidate.related_jobs,
                "student_ids": candidate.student_ids,
                "average_similarity": candidate.average_similarity,
                "priority": candidate.priority,
                "demand_share": candidate.demand_share,
                "description": candidate.description,
                "status": candidate.status,
            }
            for candidate in result.candidates
        ]
        return active_store.save_analysis(paths=paths, candidates=candidates)
