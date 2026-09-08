"""학습경로 군집화, 후보 조건, AI 설명, SQLite 저장 테스트."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.ai import (
    AIProvider,
    MockAIProvider,
    ResilientAIProvider,
    validate_microdegree_candidate_description,
)
from src.course_recommender import (
    CoursePathSelection,
    CourseRecommendation,
    CourseRecommender,
)
from src.database import LearningPathDataStore
from src.microdegree_service import (
    CANDIDATE_PRIORITY,
    LearningPathDemand,
    MicrodegreeCandidateService,
    jaccard_similarity,
)
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService
from src.similarity import (
    SentenceTransformerSimilarityBackend,
    TokenOverlapSimilarityBackend,
)
from src.utils import PROJECT_ROOT, load_app_config


class InvalidCandidateProvider(AIProvider):
    """공식 과정으로 잘못 표현하는 테스트용 provider."""

    provider_name = "invalid"

    def analyze_checkin(self, text: str) -> dict[str, Any]:
        return MockAIProvider().analyze_checkin(text)

    def describe_microdegree_candidate(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "candidate_name": "임의 공식 과정",
            "description": "검증되지 않은 설명",
            "status": "공식 마이크로디그리",
        }


class BatchCountingCandidateProvider(MockAIProvider):
    """교육과정 후보 설명이 묶음 메서드 한 번으로 호출되는지 기록한다."""

    provider_name = "batch-counting"

    def __init__(self) -> None:
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def describe_microdegree_candidates(self, candidates):
        self.batch_calls += 1
        self.batch_sizes.append(len(candidates))
        return super().describe_microdegree_candidates(candidates)


@pytest.fixture(scope="module")
def analysis_source():
    repository = get_default_repository()
    source = repository.get_all()
    snapshots = RiskAnalysisService(repository).build_snapshots(source)
    return source, snapshots


@pytest.fixture(scope="module")
def candidate_result(analysis_source):
    source, snapshots = analysis_source
    service = MicrodegreeCandidateService(
        course_recommender=CourseRecommender(
            similarity_backend=TokenOverlapSimilarityBackend()
        ),
        provider=MockAIProvider(),
    )
    return service, service.analyze(
        snapshots=snapshots,
        courses=source.courses,
        completed_courses=source.completed_courses,
    )


def test_jaccard_similarity_boundary() -> None:
    assert jaccard_similarity({"A", "B", "C"}, {"A", "B", "D"}) == 0.5
    assert jaccard_similarity({"A"}, {"B"}) == 0.0
    assert jaccard_similarity([], []) == 0.0


def _repeated_course_demand(
    student_id: str,
    department: str,
    course_ids: tuple[str, ...],
) -> LearningPathDemand:
    recommendations = tuple(
        CourseRecommendation(
            course_id=course_id,
            course_name=f"실제 교과목 {course_id}",
            department=department,
            description="2026년 실제 개설강좌",
            competencies=(),
            grade_level=1,
            semester=1,
            difficulty="원본 미제공",
            credit=3,
            prerequisites=(),
            score=80.0,
            reason="실제 개설 확인",
            is_home_department=True,
            source_course_id=course_id,
        )
        for course_id in course_ids
    )
    return LearningPathDemand(
        student_id,
        department,
        "",
        CoursePathSelection(
            courses=recommendations,
            covered_competencies=(),
            similarity_backend="test",
        ),
    )


def test_repeated_course_groups_include_multi_and_department_specific_demand() -> None:
    service = MicrodegreeCandidateService(provider=MockAIProvider())
    multi = tuple(
        _repeated_course_demand(f"M{index}", department, ("A", "B", f"X{index}"))
        for index, department in enumerate(
            ("간호학과", "소프트웨어융합과", "간호학과"), start=1
        )
    )
    department_specific = tuple(
        _repeated_course_demand(f"D{index}", "유아교육과", ("C", "D", f"Y{index}"))
        for index in range(1, 6)
    )

    groups = service.find_repeated_course_groups(multi + department_specific)
    student_groups = {
        frozenset(demand.student_id for demand in group) for group in groups
    }

    assert frozenset(demand.student_id for demand in multi) in student_groups
    assert (
        frozenset(demand.student_id for demand in department_specific)
        in student_groups
    )


def test_four_students_in_one_department_do_not_become_candidate() -> None:
    demands = tuple(
        _repeated_course_demand(f"S{index}", "유아교육과", ("A", "B", f"C{index}"))
        for index in range(1, 5)
    )

    groups = MicrodegreeCandidateService(
        provider=MockAIProvider()
    ).find_repeated_course_groups(demands)

    assert not groups


def test_one_common_course_does_not_create_repeated_pair_candidate() -> None:
    demands = tuple(
        _repeated_course_demand(
            f"S{index}",
            "유아교육과",
            ("ONLY", f"X{index}", f"Y{index}"),
        )
        for index in range(1, 6)
    )

    groups = MicrodegreeCandidateService(
        provider=MockAIProvider()
    ).find_repeated_course_groups(demands)

    assert not groups


def test_repeated_course_count_below_two_is_rejected() -> None:
    config = copy.deepcopy(load_app_config())
    config["microdegree_candidate"]["repeated_course_count"] = 1
    service = MicrodegreeCandidateService(
        provider=MockAIProvider(), config=config
    )
    demand = _repeated_course_demand("S1", "간호학과", ("A", "B", "C"))

    with pytest.raises(ValueError, match="2개 이상"):
        service.find_repeated_course_groups((demand,))


def test_sentence_transformer_reuses_course_embeddings() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            return np.asarray(
                [[1.0, float(index + 1)] for index, _ in enumerate(texts)]
            )

    backend = SentenceTransformerSimilarityBackend("test-model")
    fake_model = FakeModel()
    backend._model = fake_model

    backend.similarities("첫 질의", ["공통 교과목 A", "공통 교과목 B"])
    backend.similarities("두 번째 질의", ["공통 교과목 A", "공통 교과목 B"])

    assert fake_model.calls[0] == [
        "첫 질의",
        "공통 교과목 A",
        "공통 교과목 B",
    ]
    assert fake_model.calls[1] == ["두 번째 질의"]


def test_candidates_meet_demand_department_and_master_rules(
    analysis_source, candidate_result
) -> None:
    source, _ = analysis_source
    _, result = candidate_result
    master_ids = set(source.courses["course_id"].astype(str))

    assert result.target_student_count == 57
    assert result.skipped_student_count == 0
    assert result.candidates
    assert {candidate.priority for candidate in result.candidates} == {
        CANDIDATE_PRIORITY
    }
    for candidate in result.candidates:
        assert candidate.student_count >= 3
        assert (
            candidate.department_count >= 2
            or candidate.student_count >= 5
        )
        assert len(candidate.course_ids) == 4
        assert len(candidate.course_ids) == len(set(candidate.course_ids))
        assert set(candidate.course_ids).issubset(master_ids)
        candidate_rows = source.courses[
            source.courses["course_id"].astype(str).isin(candidate.course_ids)
        ]
        source_backed = (
            "catalog_source" in candidate_rows.columns
            and candidate_rows["catalog_source"].astype(str).str.strip().ne("").all()
        )
        assert candidate.competencies or candidate.related_jobs or source_backed
        assert candidate.status == "교육과정 검토 후보"
        assert candidate.demand_share == round(
            candidate.student_count / len(result.demands), 4
        )


def test_candidate_analysis_requests_all_descriptions_in_one_batch(
    analysis_source,
) -> None:
    """후보 수가 많아도 외부 provider에는 묶음 설명을 한 번만 요청한다."""

    source, snapshots = analysis_source
    provider = BatchCountingCandidateProvider()
    result = MicrodegreeCandidateService(
        course_recommender=CourseRecommender(
            similarity_backend=TokenOverlapSimilarityBackend()
        ),
        provider=provider,
    ).analyze(
        snapshots=snapshots,
        courses=source.courses,
        completed_courses=source.completed_courses,
    )

    assert result.candidates
    assert provider.batch_calls == 1
    assert provider.batch_sizes == [len(result.candidates)]


def test_candidates_share_configured_number_of_repeated_courses(
    candidate_result,
) -> None:
    _, result = candidate_result
    demand_by_student = {
        demand.student_id: demand for demand in result.demands
    }

    for candidate in result.candidates:
        candidate_demands = [
            demand_by_student[student_id]
            for student_id in candidate.student_ids
        ]
        common_course_ids = set(candidate_demands[0].course_ids)
        for demand in candidate_demands[1:]:
            common_course_ids.intersection_update(demand.course_ids)
        assert len(common_course_ids) >= result.minimum_repeated_courses


def test_sensitive_fields_do_not_change_candidate_analysis(
    analysis_source,
) -> None:
    source, snapshots = analysis_source
    sensitive_snapshots = snapshots.assign(
        gender="테스트", age=99, nationality="테스트"
    )

    def analyze(active_snapshots):
        return MicrodegreeCandidateService(
            course_recommender=CourseRecommender(
                similarity_backend=TokenOverlapSimilarityBackend()
            ),
            provider=MockAIProvider(),
        ).analyze(
            snapshots=active_snapshots,
            courses=source.courses,
            completed_courses=source.completed_courses,
        )

    baseline = analyze(snapshots)
    compared = analyze(sensitive_snapshots)

    assert [candidate.course_ids for candidate in baseline.candidates] == [
        candidate.course_ids for candidate in compared.candidates
    ]
    assert [candidate.student_ids for candidate in baseline.candidates] == [
        candidate.student_ids for candidate in compared.candidates
    ]


def test_mixed_latest_weeks_keep_other_students_in_demand_analysis(
    analysis_source, candidate_result
) -> None:
    """한 학생만 5주차를 제출해도 나머지 학생의 4주차 수요를 누락하지 않는다."""

    source, snapshots = analysis_source
    service, baseline = candidate_result
    followup = snapshots[
        (snapshots["student_id"].astype(str) == "S0003")
        & (snapshots["week"] == 4)
    ].copy()
    followup["week"] = 5
    followup["major_adaptation_risk"] = 0.0
    followup["career_risk"] = 0.0
    mixed_snapshots = pd.concat([snapshots, followup], ignore_index=True)

    compared = service.analyze(
        snapshots=mixed_snapshots,
        courses=source.courses,
        completed_courses=source.completed_courses,
    )

    baseline_students = {item.student_id for item in baseline.demands}
    compared_students = {item.student_id for item in compared.demands}
    assert "S0003" in baseline_students
    assert "S0003" not in compared_students
    assert baseline_students - {"S0003"} == compared_students
    assert compared.target_student_count == baseline.target_student_count - 1


def test_candidate_description_schema_and_fallback() -> None:
    invalid = InvalidCandidateProvider().describe_microdegree_candidate({})
    with pytest.raises(Exception, match="JSON Schema"):
        validate_microdegree_candidate_description(invalid)

    provider = ResilientAIProvider(
        InvalidCandidateProvider(), MockAIProvider()
    )
    result = provider.describe_microdegree_candidate(
        {
            "student_count": 10,
            "competencies": ["데이터활용", "마케팅"],
            "related_jobs": ["디지털 마케팅"],
        }
    )

    assert result["status"] == "교육과정 검토 후보"
    assert provider.last_provider_name == "mock"
    assert provider.last_error is not None


def test_analysis_is_persisted_as_paths_courses_and_candidates(
    tmp_path: Path, candidate_result
) -> None:
    service, result = candidate_result
    store = LearningPathDataStore(tmp_path / "app.db")

    batch_id = service.persist_analysis(result, store)
    stored_paths = store.list_learning_paths(batch_id)
    stored_candidates = store.list_latest_candidates()

    assert stored_paths["student_id"].nunique() == len(result.demands)
    assert set(stored_paths["course_id"]).issubset(
        {
            course.course_id
            for demand in result.demands
            for course in demand.selection.courses
        }
    )
    assert len(stored_candidates) == len(result.candidates)
    assert set(stored_candidates["status"]) == {"교육과정 검토 후보"}
    assert set(stored_candidates["priority"]) == {CANDIDATE_PRIORITY}
    assert list(stored_candidates["priority"]) == [
        candidate.priority for candidate in result.candidates
    ]
    assert stored_candidates["demand_share"].gt(0).all()
    assert set(stored_candidates["analysis_batch_id"]) == {batch_id}


def test_legacy_candidate_payload_is_saved_for_reanalysis(tmp_path: Path) -> None:
    """우선순위 필드가 없던 기존 호출도 저장 실패 없이 보존한다."""

    store = LearningPathDataStore(tmp_path / "legacy-candidate.db")
    store.save_analysis(
        paths=[
            {
                "student_id": "S1",
                "path_name": "기존 경로",
                "related_job": "",
                "competencies": ["기초역량"],
                "reason": "기존 저장 형식",
                "courses": [
                    {
                        "course_id": "C001",
                        "sequence": 1,
                        "score": 0.7,
                        "reason": "기존 경로 과목",
                    }
                ],
            }
        ],
        candidates=[
            {
                "candidate_code": "OLD-01",
                "candidate_name": "기존 후보",
                "student_count": 5,
                "department_count": 2,
                "departments": ["DEPT01", "DEPT02"],
                "course_ids": ["C001"],
                "course_names": ["기존 교과목"],
                "competencies": ["기초역량"],
                "related_jobs": [],
                "student_ids": ["S1", "S2", "S3", "S4", "S5"],
                "average_similarity": 0.7,
                "description": "기존 저장 형식",
                "status": "검토 후보",
            }
        ],
    )

    stored = store.list_latest_candidates().iloc[0]
    assert stored["priority"] == "재분석 필요"
    assert stored["demand_share"] == 0


def test_microdegree_page_explains_demand_candidate_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """교직원 화면에서 반복 수요 기준과 synthetic 한계를 함께 안내한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH",
        tmp_path / "candidate-page.db",
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "06_microdegree_candidates.py")
    ).run(timeout=30)

    assert not page.exception
    visible = "\n".join(
        str(item.value)
        for item in [*page.markdown, *page.caption, *page.warning, *page.info]
    )
    assert "2개 이상 반복된 교과목 조합" in visible
    assert "다학과 후보: 3명 이상·2개 학과 이상" in visible
    assert "학과 특화 후보: 한 학과 5명 이상" in visible
    assert "synthetic 학생 200명" in visible
    assert "공식 마이크로디그리가 아닙니다" in visible
    assert "A · 우선 검토" not in visible
    assert "B · 수요 관찰" not in visible


def test_source_backed_repeated_paths_can_become_candidate_without_tags() -> None:
    recommendations = tuple(
        CourseRecommendation(
            course_id=f"R{index:03d}",
            course_name=f"실제 교과목 {index}",
            department="연계학과",
            description="2026년 실제 개설강좌",
            competencies=(),
            grade_level=1,
            semester=1,
            difficulty="원본 미제공",
            credit=3,
            prerequisites=(),
            score=80.0,
            reason="실제 개설 확인",
            is_home_department=False,
            source_course_id=f"R{index:03d}",
        )
        for index in range(1, 4)
    )
    selection = CoursePathSelection(
        courses=recommendations,
        covered_competencies=(),
        similarity_backend="test",
        covered_source_attributes=("전공", "NCS"),
    )
    demands = (
        LearningPathDemand("S1", "간호학과", "", selection),
        LearningPathDemand("S2", "간호학과", "", selection),
        LearningPathDemand("S3", "소프트웨어융합과", "", selection),
    )
    courses = pd.DataFrame(
        [
            {
                "course_id": course.course_id,
                "course_name": course.course_name,
                "catalog_source": "actual_course_offerings_2026",
            }
            for course in recommendations
        ]
    )
    service = MicrodegreeCandidateService(provider=MockAIProvider())

    candidate = service._candidate_from_cluster(
        demands,
        courses,
        candidate_index=1,
        demand_population=3,
    )

    assert candidate is not None
    assert candidate.course_ids == ("R001", "R002", "R003")
    assert not candidate.competencies
    assert not candidate.related_jobs
    assert "실제 교과목" in candidate.candidate_name
    assert service.find_repeated_course_groups(demands) == (demands,)
