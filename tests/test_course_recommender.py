"""교과목 hard filter, 다양성, AI 학습경로 설명 검증 테스트."""

import pandas as pd
import pytest

from src.ai import AIProvider, MockAIProvider, ResilientAIProvider
from src.ai.schema import validate_learning_path_explanation
from src.course_recommender import CourseRecommender
from src.data_generator import generate_courses
from src.learning_path_builder import (
    LearningPathBuilder,
    ordered_courses_by_sequence,
)
from src.recommendation_interest import infer_interest_evidence
from src.similarity import TokenOverlapSimilarityBackend


PROFILE = {
    "student_id": "TEST",
    "department": "소프트웨어융합과",
    "grade": 2,
    "interest_fields": "데이터·AI|경영·마케팅|콘텐츠·디자인",
    "desired_job": "디지털 마케팅",
    "natural_language_concern": "데이터와 콘텐츠를 활용하는 마케팅 직무에 관심이 있습니다.",
}
ANALYSIS = {
    "major_concern": True,
    "learning_difficulty": False,
    "career_uncertainty": True,
    "interests": ["데이터·AI", "경영·마케팅", "콘텐츠·디자인"],
    "desired_jobs": ["디지털 마케팅"],
    "support_needs": ["전공 탐색", "진로 탐색"],
    "summary": "데이터와 콘텐츠를 연결한 마케팅 직무를 탐색하고 있음",
}


class InvalidPathProvider(AIProvider):
    provider_name = "invalid"

    def analyze_checkin(self, text: str):
        return ANALYSIS

    def explain_learning_path(self, profile, courses):
        return {
            "path_name": "잘못된 경로",
            "reason": "DB에 없는 과목을 포함",
            "competencies": ["테스트"],
            "course_roles": [
                {"course_id": "UNKNOWN", "role": "잘못된 과목", "sequence": index}
                for index in range(1, 4)
            ],
            "career_connection": "없음",
        }


def _selection(profile=None, completed=None):
    return CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    ).recommend(
        profile=profile or PROFILE,
        analysis=ANALYSIS,
        courses=generate_courses(),
        completed_course_ids=completed or set(),
    )


def test_course_path_has_three_to_four_unique_existing_courses() -> None:
    courses = generate_courses()
    result = _selection()
    ids = [course.course_id for course in result.courses]

    assert 3 <= len(ids) <= 4
    assert len(ids) == len(set(ids))
    assert set(ids).issubset(set(courses["course_id"]))
    assert len(result.covered_competencies) >= 3
    assert all(
        "시스템 보유자료 기준 수강 필터 통과" in course.reason
        for course in result.courses
    )
    assert all(
        "대화·과목 정보 의미 유사도 참고값" in course.reason
        for course in result.courses
    )


def test_completed_courses_are_excluded() -> None:
    initial = _selection()
    completed = {initial.courses[0].course_id, initial.courses[1].course_id}
    result = _selection(completed=completed)

    assert completed.isdisjoint({course.course_id for course in result.courses})


def test_grade_availability_and_prerequisites_are_hard_filters() -> None:
    completed = {"C001", "C002", "C003"}
    result = _selection(completed=completed)

    for course in result.courses:
        assert course.grade_level <= PROFILE["grade"]
        assert set(course.prerequisites).issubset(completed)


def test_home_department_is_prioritized_while_cross_department_stays_allowed() -> None:
    profile = {
        "student_id": "TEST-CROSS",
        "department": "간호학과",
        "grade": 2,
        "interest_fields": "데이터·AI",
        "desired_job": "데이터분석가",
        "natural_language_concern": "데이터 분석 역량을 배우고 싶어요.",
    }
    analysis = {
        **ANALYSIS,
        "interests": ["데이터·AI"],
        "desired_jobs": ["데이터분석가"],
        "summary": "데이터 분석 역량을 배우고 싶음",
    }

    result = CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    ).recommend(
        profile=profile,
        analysis=analysis,
        courses=generate_courses(),
        completed_course_ids=set(),
    )

    home_courses = [course for course in result.courses if course.is_home_department]
    cross_courses = [course for course in result.courses if not course.is_home_department]
    assert home_courses
    assert cross_courses
    assert all(course.department == "간호학과" for course in home_courses)
    assert all("소속 학과 우선" in course.reason for course in home_courses)
    assert all("타과 과목" in course.reason for course in cross_courses)


def test_cross_department_path_is_used_when_no_home_course_is_eligible() -> None:
    courses = generate_courses()
    courses.loc[courses["department"] == "간호학과", "is_available"] = False
    profile = {
        **PROFILE,
        "department": "간호학과",
        "interest_fields": "데이터·AI",
        "desired_job": "데이터분석가",
    }

    result = CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    ).recommend(
        profile=profile,
        analysis=ANALYSIS,
        courses=courses,
        completed_course_ids=set(),
    )

    assert all(not course.is_home_department for course in result.courses)
    assert result.warning is not None
    assert "소속 학과 과목이 없어" in result.warning


def test_sensitive_fields_do_not_change_course_selection() -> None:
    baseline = _selection(PROFILE)
    sensitive_profile = dict(PROFILE, gender="여성", age=99, nationality="TEST")
    compared = _selection(sensitive_profile)

    assert [item.course_id for item in baseline.courses] == [
        item.course_id for item in compared.courses
    ]


def test_learning_path_explanation_rejects_unknown_course_ids() -> None:
    invalid = InvalidPathProvider().explain_learning_path({}, [])
    with pytest.raises(Exception, match="DB에 없는 교과목"):
        validate_learning_path_explanation(invalid, {"C001", "C002", "C003"})


def test_mock_learning_path_keeps_selected_database_ids() -> None:
    selection = _selection()
    result = LearningPathBuilder(MockAIProvider()).build(PROFILE, selection)

    explained_ids = {item["course_id"] for item in result.explanation["course_roles"]}
    assert explained_ids == {course.course_id for course in selection.courses}
    assert result.provider_name == "mock"


def test_learning_path_cards_follow_explanation_sequence() -> None:
    """카드 출력은 추천기 배열이 아니라 검증된 권장 순서를 따른다."""

    selection = _selection()
    reversed_courses = tuple(reversed(selection.courses))
    explanation = {
        "course_roles": [
            {
                "course_id": course.course_id,
                "role": "테스트 역할",
                "sequence": sequence,
            }
            for sequence, course in enumerate(reversed_courses, start=1)
        ]
    }

    ordered = ordered_courses_by_sequence(selection, explanation)

    assert [course.course_id for course in ordered] == [
        course.course_id for course in reversed_courses
    ]


def test_invalid_primary_path_explanation_falls_back_to_mock() -> None:
    selection = _selection()
    provider = ResilientAIProvider(InvalidPathProvider(), MockAIProvider())

    result = LearningPathBuilder(provider).build(PROFILE, selection)

    assert result.fallback_used is True
    assert {item["course_id"] for item in result.explanation["course_roles"]} == {
        course.course_id for course in selection.courses
    }


def test_actual_offerings_use_source_fields_without_inventing_tags() -> None:
    rows = []
    specifications = (
        ("R001", "간호정보학", "간호학과", "allowed"),
        ("R002", "데이터분석", "소프트웨어융합과", "allowed"),
        ("R003", "AI 기초", "소프트웨어융합과", "allowed"),
        ("R004", "보건 데이터", "보건학과", "allowed"),
        ("R005", "데이터 심화", "데이터학과", "restricted"),
    )
    for course_id, name, department, cross_status in specifications:
        rows.append(
            {
                "course_id": course_id,
                "course_name": name,
                "department": department,
                "description": f"{department} 개설 전공 교과목 {name}",
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
                "offering_departments": department,
                "offered_semesters": "1|2",
                "grade_levels": "1",
                "course_area": "전공",
                "class_method": "대면",
                "course_type": "일반교과",
                "ncs_type": "NCS",
                "cross_department_status": cross_status,
                "recommendation_basis": "교과목명·개설학과·NCS",
                "completion_data_status": "미연결",
                "registration_check_required": True,
            }
        )
    profile = {
        **PROFILE,
        "department": "간호학과",
        "interest_fields": "데이터·AI|보건",
        "desired_job": "보건 데이터분석가",
        "natural_language_concern": "보건 데이터와 AI를 배우고 싶어요.",
    }
    analysis = {
        **ANALYSIS,
        "interests": ["데이터·AI", "보건"],
        "desired_jobs": ["보건 데이터분석가"],
        "summary": "보건 데이터와 AI 학습 희망",
    }

    result = CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend()
    ).recommend(
        profile=profile,
        analysis=analysis,
        courses=pd.DataFrame(rows),
        completed_course_ids=set(),
    )

    selected_ids = {course.course_id for course in result.courses}
    assert 3 <= len(selected_ids) <= 4
    assert "R005" not in selected_ids
    assert any(course.is_home_department for course in result.courses)
    assert any(not course.is_home_department for course in result.courses)
    assert all(not course.competencies for course in result.courses)
    assert not result.covered_competencies
    assert result.covered_source_attributes
    assert result.warning is not None
    assert "강의계획서" in result.warning
    assert all(course.source_course_id == course.course_id for course in result.courses)


def test_source_backed_path_keeps_cross_department_limit_when_home_courses_exist() -> None:
    rows = []
    specifications = (
        ("HOME01", "간호기초실습", "간호학과", "기초 간호 술기를 익힙니다."),
        ("HOME02", "건강사정", "간호학과", "대상자의 건강 상태를 사정합니다."),
        ("CROSS01", "데이터서비스기획", "소프트웨어융합과", "데이터 AI 서비스 운영 기획"),
        ("CROSS02", "AI서비스개발", "소프트웨어융합과", "데이터 AI 서비스 운영 개발"),
        ("CROSS03", "디지털서비스운영", "경영학과", "데이터 AI 서비스 운영 관리"),
        ("CROSS04", "서비스분석", "경영학과", "데이터 AI 서비스 운영 분석"),
    )
    for course_id, name, department, description in specifications:
        rows.append(
            {
                "course_id": course_id,
                "course_name": name,
                "department": department,
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
                "offering_departments": department,
                "offered_semesters": "1",
                "grade_levels": "1",
                "course_area": "전공",
                "class_method": "대면",
                "course_type": "일반교과",
                "ncs_type": "NCS",
                "cross_department_status": "allowed",
                "recommendation_basis": "교과목명·개설학과·NCS",
                "completion_data_status": "미연결",
                "registration_check_required": True,
            }
        )
    config = {
        "course_recommendation": {
            "top_k": 4,
            "minimum_courses": 3,
            "minimum_interest_aligned_courses": 2,
            "candidate_pool_size": 4,
            "diversity_bonus": 0.0,
            "department_policy": {
                "allow_cross_department": True,
                "home_department_bonus": 0.0,
                "minimum_home_department_courses": 1,
            },
            "source_backed_policy": {
                "home_department_bonus": 0.0,
                "candidate_pool_size": 4,
                "maximum_cross_department_courses": 2,
                "max_general_education_courses": 1,
                "max_same_name_family": 1,
            },
            "weights": {
                "semantic_similarity": 0.4,
                "interest_match": 0.2,
                "job_match": 0.15,
                "competency_coverage": 0.15,
                "feasibility": 0.1,
            },
        }
    }
    profile = {
        **PROFILE,
        "department": "간호학과",
        "interest_fields": "데이터·AI",
        "desired_job": "서비스 운영",
        "natural_language_concern": "데이터 AI 서비스 운영을 배우고 싶어요.",
    }

    result = CourseRecommender(
        similarity_backend=TokenOverlapSimilarityBackend(),
        config=config,
    ).recommend(
        profile=profile,
        analysis={
            **ANALYSIS,
            "interests": ["데이터·AI"],
            "desired_jobs": ["서비스 운영"],
            "summary": "데이터 AI 서비스 운영을 배우고 싶음",
        },
        courses=pd.DataFrame(rows),
        completed_course_ids=set(),
    )

    assert len(result.courses) == 4
    assert sum(not course.is_home_department for course in result.courses) == 2
    assert sum(course.is_home_department for course in result.courses) == 2
    assert sum(
        "관심분야 직접 연결:" in course.reason
        and "데이터·AI" in course.reason
        for course in result.courses
    ) >= 2


@pytest.mark.parametrize(
    "interest,text",
    [
        ("데이터·AI", "파이썬으로 데이터를 분석하고 시각화하는 수업"),
        ("경영·마케팅", "소비자 시장과 브랜드 마케팅을 배우는 수업"),
        ("콘텐츠·디자인", "영상 편집과 그래픽 디자인 프로젝트"),
        ("서비스", "고객경험을 중심으로 서비스기획을 실습"),
        ("보건", "환자 간호와 임상 보건을 배우는 수업"),
        ("상담·복지", "심리 상담과 사례관리 실습"),
    ],
)
def test_runtime_interest_inference_supports_every_student_interest(
    interest: str, text: str
) -> None:
    """원본 master에 태그가 없어도 6개 관심 분야를 동일한 방식으로 판별한다."""

    evidence = infer_interest_evidence(text, {interest})

    assert interest in evidence
    assert evidence[interest]
