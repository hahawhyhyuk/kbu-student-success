"""Synthetic 교과 추천 내부 관련성 평가 서비스 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src.course_recommendation_evaluation_service import (
    COURSE_RECOMMENDATION_EVALUATION_CASES,
    AcceptableCourseFamily,
    CourseRecommendationEvaluationCase,
    course_recommendation_evaluation_catalog,
    evaluate_course_recommendations,
)
from src.course_recommender import CourseRecommender
from src.similarity import TokenOverlapSimilarityBackend


def _course(
    course_id: str,
    course_name: str,
    description: str,
) -> dict[str, object]:
    return {
        "course_id": course_id,
        "course_name": course_name,
        "department": "소프트웨어융합과",
        "description": description,
        "learning_objectives": "",
        "competencies": "",
        "related_jobs": "",
        "related_interests": "",
        "grade_level": 1,
        "semester": 1,
        "difficulty": "원본 미제공",
        "prerequisites": "",
        "credit": 3,
        "is_available": True,
        "catalog_source": "actual_course_offerings_2026",
        "source_course_id": course_id,
        "offering_departments": "소프트웨어융합과",
        "offered_semesters": "1|2",
        "grade_levels": "1",
        "course_area": "전공",
        "class_method": "대면",
        "course_type": "일반교과",
        "ncs_type": "자율과목",
        "cross_department_status": "allowed",
        "recommendation_basis": "교과목명·개설학과·강좌설명",
        "completion_data_status": "미연결",
        "registration_check_required": True,
    }


def _case(case_id: str = "EVAL-01") -> CourseRecommendationEvaluationCase:
    return CourseRecommendationEvaluationCase(
        case_id=case_id,
        title="데이터 분석과 AI 개발",
        department="소프트웨어융합과",
        grade=2,
        interest_fields=("데이터·AI",),
        desired_job="데이터 분석가",
        concern="파이썬으로 데이터를 분석하고 인공지능을 배우고 싶어요.",
        acceptable_families=(
            AcceptableCourseFamily("프로그래밍", ("파이썬", "프로그래밍")),
            AcceptableCourseFamily("데이터 분석", ("데이터", "분석")),
            AcceptableCourseFamily("인공지능", ("인공지능", "AI")),
        ),
        minimum_family_matches=2,
        minimum_relevant_courses=2,
    )


def _courses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _course("C001", "파이썬프로그래밍", "파이썬 코딩 기초"),
            _course("C002", "데이터분석", "데이터 전처리와 분석"),
            _course("C003", "인공지능개론", "인공지능 모델의 원리"),
            _course("C004", "웹프로그래밍", "웹 서비스 프로그래밍"),
        ]
    )


def test_catalog_has_ten_balanced_non_identifying_cases() -> None:
    """5개 적용 학과에 두 사례씩 배치하고 개인 식별값을 사용하지 않는다."""

    catalog = course_recommendation_evaluation_catalog()

    assert len(catalog) == 10
    assert set(catalog.groupby("department").size()) == {2}
    assert catalog["case_id"].is_unique
    combined = " ".join(
        (case.concern + " " + case.desired_job)
        for case in COURSE_RECOMMENDATION_EVALUATION_CASES
    )
    assert "학번" not in combined
    assert "이름" not in combined
    assert "student_id" not in combined


def test_evaluation_checks_results_without_changing_recommendation_ranking() -> None:
    """평가 과목군은 추천 이후에만 적용되고 DB 존재·관련성도 함께 확인한다."""

    recommender = CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    )
    case = _case()
    courses = _courses()
    direct = recommender.recommend(
        case.profile(), case.analysis(), courses, completed_course_ids=()
    )

    report = evaluate_course_recommendations(recommender, courses, (case,))
    result = report.results[0]

    assert result.selected_course_ids == tuple(
        course.course_id for course in direct.courses
    )
    assert report.case_pass_rate == 100.0
    assert report.family_coverage_rate == 100.0
    assert report.relevant_course_rate == 100.0
    assert report.unknown_course_count == 0
    assert report.duplicate_course_count == 0
    assert result.passed


def test_evaluation_records_weak_relevance_as_review_issue() -> None:
    """기대 과목군을 충분히 충족하지 못하면 오류 대신 검토 대상으로 남긴다."""

    weak_case = CourseRecommendationEvaluationCase(
        **{
            **_case().__dict__,
            "case_id": "EVAL-WEAK",
            "acceptable_families": (
                AcceptableCourseFamily("해양", ("해양", "선박")),
                AcceptableCourseFamily("항해", ("항해", "기관")),
            ),
        }
    )
    report = evaluate_course_recommendations(
        CourseRecommender(similarity_backend=TokenOverlapSimilarityBackend()),
        _courses(),
        (weak_case,),
    )

    assert not report.results[0].passed
    assert report.results[0].error is None
    assert report.results[0].issues
    assert report.family_coverage_rate == 0.0


def test_evaluation_rejects_invalid_case_or_duplicate_course_ids() -> None:
    """평가 기준과 교과 master 자체의 오류를 실행 전에 차단한다."""

    recommender = CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    )
    valid = _case()

    with pytest.raises(ValueError, match="중복"):
        evaluate_course_recommendations(recommender, _courses(), (valid, valid))

    duplicated_courses = pd.concat([_courses(), _courses().iloc[[0]]])
    with pytest.raises(ValueError, match="course_id는 중복"):
        evaluate_course_recommendations(
            recommender,
            duplicated_courses,
            (valid,),
        )
