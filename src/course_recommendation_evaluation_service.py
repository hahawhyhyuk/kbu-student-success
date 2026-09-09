"""Synthetic 학생 사례로 교과 추천의 내부 관련성을 반복 점검한다.

평가용 과목군은 추천 입력이나 교과 master에 추가되지 않는다. 추천이 끝난 뒤
과목명·공개 강좌 설명에 기대 개념이 나타나는지만 확인하는 독립적인 회귀 기준이다.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from src.course_recommender import CoursePathSelection, CourseRecommender


@dataclass(frozen=True)
class AcceptableCourseFamily:
    """한 synthetic 진로·학습 요구에 허용할 수 있는 교과 개념군."""

    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class CourseRecommendationEvaluationCase:
    """개인정보가 없는 추천 입력과 사후 평가 기대값."""

    case_id: str
    title: str
    department: str
    grade: int
    interest_fields: tuple[str, ...]
    desired_job: str
    concern: str
    acceptable_families: tuple[AcceptableCourseFamily, ...]
    minimum_family_matches: int = 2
    minimum_relevant_courses: int = 2

    def profile(self) -> dict[str, Any]:
        """기존 CourseRecommender 입력 계약으로 synthetic profile을 변환한다."""

        return {
            "department": self.department,
            "grade": self.grade,
            "interest_fields": "|".join(self.interest_fields),
            "desired_job": self.desired_job,
            "natural_language_concern": self.concern,
        }

    def analysis(self) -> dict[str, Any]:
        """추천 평가에 필요한 최소 AI 분석 구조를 반환한다."""

        return {
            "interests": list(self.interest_fields),
            "desired_jobs": [self.desired_job] if self.desired_job else [],
            "summary": self.concern,
        }


@dataclass(frozen=True)
class CourseRecommendationEvaluationResult:
    """추천 사례 한 건의 사후 과목군 일치 결과."""

    case_id: str
    title: str
    department: str
    selected_course_ids: tuple[str, ...]
    selected_course_names: tuple[str, ...]
    matched_families: tuple[str, ...]
    relevant_course_count: int
    home_course_count: int
    cross_course_count: int
    unknown_course_count: int
    duplicate_course_count: int
    similarity_backend: str
    passed: bool
    issues: tuple[str, ...]
    course_matches: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True)
class CourseRecommendationEvaluationReport:
    """여러 synthetic 추천 사례의 내부 관련성 집계."""

    case_count: int
    passed_case_count: int
    expected_family_count: int
    matched_family_count: int
    selected_course_count: int
    relevant_course_count: int
    unknown_course_count: int
    duplicate_course_count: int
    details: pd.DataFrame
    results: tuple[CourseRecommendationEvaluationResult, ...]

    @property
    def case_pass_rate(self) -> float:
        """사례별 최소 기대조건을 모두 충족한 비율."""

        return _percentage(self.passed_case_count, self.case_count)

    @property
    def family_coverage_rate(self) -> float:
        """기대한 과목군 중 추천 결과에서 확인된 비율."""

        return _percentage(self.matched_family_count, self.expected_family_count)

    @property
    def relevant_course_rate(self) -> float:
        """추천 과목 중 하나 이상의 허용 과목군과 연결된 비율."""

        return _percentage(self.relevant_course_count, self.selected_course_count)


def _family(label: str, *keywords: str) -> AcceptableCourseFamily:
    return AcceptableCourseFamily(label=label, keywords=tuple(keywords))


COURSE_RECOMMENDATION_EVALUATION_CASES: tuple[
    CourseRecommendationEvaluationCase, ...
] = (
    CourseRecommendationEvaluationCase(
        "COURSE-01",
        "친환경 건축설계 탐색",
        "친환경건축과",
        2,
        ("콘텐츠·디자인",),
        "친환경 건축 설계",
        "에너지와 환경을 고려한 건축 설계 방법을 배우고 싶어요.",
        (
            _family("친환경 설계", "친환경", "녹색건축", "지속가능", "제로에너지"),
            _family("건축 계획", "건축설계", "건축계획", "공간설계", "bim"),
            _family("건축 환경", "건축환경", "건축설비", "환경디자인"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-02",
        "건축 시공·디지털 실무",
        "친환경건축과",
        2,
        ("서비스", "콘텐츠·디자인"),
        "건축 시공관리",
        "도면을 이해하고 현장 시공과 공정관리 실무를 익히고 싶어요.",
        (
            _family("시공·공정", "시공", "공법", "공정관리", "건설"),
            _family("도면·디지털", "도면", "cad", "bim", "컴퓨터", "모델링"),
            _family("건축 안전", "건축", "안전", "품질관리"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-03",
        "유아 놀이수업 기획",
        "유아교육과",
        2,
        ("콘텐츠·디자인",),
        "유아교사",
        "아이들의 발달에 맞는 놀이와 수업 활동을 직접 설계하고 싶어요.",
        (
            _family("영유아 발달", "영유아", "유아", "아동", "발달"),
            _family("교수·수업", "교수", "수업", "교육방법", "교재"),
            _family("놀이·활동", "놀이", "활동", "음악", "미술", "동화"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-04",
        "유아 상담·발달 지원",
        "유아교육과",
        2,
        ("상담·복지",),
        "유아 상담·교육",
        "아이의 행동과 발달을 이해하고 보호자와 소통하는 법을 배우고 싶어요.",
        (
            _family("발달·심리", "발달", "심리", "행동", "정서"),
            _family("상담·소통", "상담", "의사소통", "부모", "보호자"),
            _family("아동 지원", "유아", "아동", "복지", "특수교육", "장애"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-05",
        "AI 음악 제작",
        "실용음악과(3년제)",
        2,
        ("데이터·AI", "콘텐츠·디자인"),
        "음악 프로듀서",
        "AI와 디지털 도구를 활용해서 작곡하고 음원을 제작하고 싶어요.",
        (
            _family("작곡·미디", "작곡", "미디", "daw", "시퀀"),
            _family("AI·디지털", "인공지능", "ai", "디지털", "데이터"),
            _family("음원·콘텐츠 제작", "음원", "콘텐츠", "제작", "프로듀"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-06",
        "보컬 무대 역량",
        "실용음악과(3년제)",
        2,
        ("콘텐츠·디자인",),
        "보컬리스트",
        "보컬 기본기를 다지고 밴드와 함께 무대 공연 경험을 쌓고 싶어요.",
        (
            _family("보컬·발성", "보컬", "발성", "가창"),
            _family("연주·공연", "연주", "공연", "무대"),
            _family("앙상블", "앙상블", "합주", "밴드"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-07",
        "임상 간호 기초",
        "간호학과",
        2,
        ("보건",),
        "임상간호사",
        "환자 상태를 정확히 파악하고 안전하게 간호하는 실무를 배우고 싶어요.",
        (
            _family("임상 간호", "간호", "임상", "환자", "대상자"),
            _family("건강 사정", "건강사정", "사정", "진단", "간호과정"),
            _family("환자 안전", "환자안전", "감염", "안전", "응급"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-08",
        "보건 데이터·연구",
        "간호학과",
        2,
        ("보건", "데이터·AI"),
        "보건 데이터 분석",
        "건강과 간호 데이터를 분석해서 근거 있는 보건 서비스를 만들고 싶어요.",
        (
            _family("보건·건강", "보건", "건강", "의료"),
            _family("데이터·연구", "데이터", "통계", "연구", "분석"),
            _family("디지털 헬스", "디지털헬스", "건강정보", "간호정보", "인공지능"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-09",
        "AI 데이터 분석",
        "소프트웨어융합과",
        2,
        ("데이터·AI",),
        "데이터 분석가",
        "파이썬으로 데이터를 분석하고 인공지능 모델을 다뤄보고 싶어요.",
        (
            _family("프로그래밍", "파이썬", "프로그래밍", "코딩", "소프트웨어"),
            _family("인공지능", "인공지능", "ai", "머신러닝", "딥러닝"),
            _family("데이터 분석", "데이터", "분석", "시각화"),
        ),
    ),
    CourseRecommendationEvaluationCase(
        "COURSE-10",
        "클라우드 앱 서비스 개발",
        "소프트웨어융합과",
        2,
        ("서비스", "데이터·AI"),
        "애플리케이션 개발자",
        "클라우드에서 동작하는 앱 서비스를 기획하고 배포해보고 싶어요.",
        (
            _family("앱·웹", "애플리케이션", "모바일", "웹", "앱개발"),
            _family("클라우드·배포", "클라우드", "서버", "네트워크", "배포"),
            _family("서비스 개발", "서비스개발", "소프트웨어개발", "프로그래밍", "프로젝트"),
        ),
    ),
)


_COURSE_TEXT_COLUMNS = (
    "course_name",
    "description",
    "learning_objectives",
    "course_area",
    "class_method",
    "course_type",
    "ncs_type",
)


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _keyword_in_text(keyword: str, compact_text: str, raw_text: str) -> bool:
    """한글 구절은 공백을 무시하고 영문 약어는 단어 경계로 비교한다."""

    normalized_keyword = _normalized_text(keyword)
    if not normalized_keyword:
        return False
    if normalized_keyword.isascii() and normalized_keyword.isalnum():
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
                raw_text.lower(),
            )
        )
    return normalized_keyword in compact_text


def _matched_family_labels(
    course: Mapping[str, Any],
    families: Sequence[AcceptableCourseFamily],
) -> tuple[str, ...]:
    raw_text = " ".join(str(course.get(column, "")) for column in _COURSE_TEXT_COLUMNS)
    compact_text = _normalized_text(raw_text)
    return tuple(
        family.label
        for family in families
        if any(
            _keyword_in_text(keyword, compact_text, raw_text)
            for keyword in family.keywords
        )
    )


def _validate_cases(
    cases: Sequence[CourseRecommendationEvaluationCase],
) -> None:
    if not cases:
        raise ValueError("교과 추천 평가 사례가 한 개 이상 필요합니다.")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("교과 추천 평가 사례 ID는 중복될 수 없습니다.")
    for case in cases:
        if not case.case_id.strip() or not case.title.strip():
            raise ValueError("교과 추천 평가 사례에는 ID와 제목이 필요합니다.")
        if not case.department.strip() or not 1 <= int(case.grade) <= 4:
            raise ValueError(f"{case.case_id}: 학과와 1~4학년 정보가 필요합니다.")
        if not case.concern.strip() or not case.acceptable_families:
            raise ValueError(f"{case.case_id}: 고민과 허용 과목군이 필요합니다.")
        family_labels = [family.label for family in case.acceptable_families]
        if len(family_labels) != len(set(family_labels)):
            raise ValueError(f"{case.case_id}: 허용 과목군 이름이 중복되었습니다.")
        if any(not family.label.strip() or not family.keywords for family in case.acceptable_families):
            raise ValueError(f"{case.case_id}: 과목군 이름과 키워드가 필요합니다.")
        if not 1 <= case.minimum_family_matches <= len(case.acceptable_families):
            raise ValueError(f"{case.case_id}: 최소 과목군 일치 수가 올바르지 않습니다.")
        if not 1 <= case.minimum_relevant_courses <= 4:
            raise ValueError(f"{case.case_id}: 최소 관련 과목 수가 올바르지 않습니다.")


def _evaluate_selection(
    case: CourseRecommendationEvaluationCase,
    selection: CoursePathSelection,
    course_by_id: Mapping[str, Mapping[str, Any]],
) -> CourseRecommendationEvaluationResult:
    selected_ids = tuple(course.course_id for course in selection.courses)
    selected_names = tuple(course.course_name for course in selection.courses)
    matched_families: set[str] = set()
    relevant_course_count = 0
    course_matches: list[str] = []
    unknown_course_count = 0

    for recommendation in selection.courses:
        source = course_by_id.get(recommendation.course_id)
        if source is None:
            unknown_course_count += 1
            course_matches.append(f"{recommendation.course_name}: DB에서 확인되지 않음")
            continue
        labels = _matched_family_labels(source, case.acceptable_families)
        if labels:
            relevant_course_count += 1
            matched_families.update(labels)
            course_matches.append(f"{recommendation.course_name}: {', '.join(labels)}")
        else:
            course_matches.append(f"{recommendation.course_name}: 과목군 수동 확인")

    duplicate_course_count = len(selected_ids) - len(set(selected_ids))
    issues: list[str] = []
    if unknown_course_count:
        issues.append(f"DB 미존재 과목 {unknown_course_count}개")
    if duplicate_course_count:
        issues.append(f"중복 과목 {duplicate_course_count}개")
    if len(matched_families) < case.minimum_family_matches:
        issues.append(
            f"기대 과목군 {case.minimum_family_matches}개 중 {len(matched_families)}개 확인"
        )
    if relevant_course_count < case.minimum_relevant_courses:
        issues.append(
            f"관련 과목 {case.minimum_relevant_courses}개 중 {relevant_course_count}개 확인"
        )
    return CourseRecommendationEvaluationResult(
        case_id=case.case_id,
        title=case.title,
        department=case.department,
        selected_course_ids=selected_ids,
        selected_course_names=selected_names,
        matched_families=tuple(sorted(matched_families)),
        relevant_course_count=relevant_course_count,
        home_course_count=sum(course.is_home_department for course in selection.courses),
        cross_course_count=sum(not course.is_home_department for course in selection.courses),
        unknown_course_count=unknown_course_count,
        duplicate_course_count=duplicate_course_count,
        similarity_backend=selection.similarity_backend,
        passed=not issues,
        issues=tuple(issues),
        course_matches=tuple(course_matches),
    )


def evaluate_course_recommendations(
    recommender: CourseRecommender,
    courses: pd.DataFrame,
    cases: Sequence[CourseRecommendationEvaluationCase] = (
        COURSE_RECOMMENDATION_EVALUATION_CASES
    ),
) -> CourseRecommendationEvaluationReport:
    """실제 추천을 실행하고 synthetic 기대 과목군과의 관련성을 집계한다.

    Parameters:
        recommender: 운영과 동일한 deterministic 후보선정기.
        courses: 현재 Repository가 제공하는 교과목 master.
        cases: 개인정보 없는 고정 평가 사례.

    Returns:
        사례 통과율·과목군 포괄률·관련 과목 비율과 사례별 상세.

    Assumptions:
        과목군은 사후 회귀 점검에만 쓰며 추천 점수나 순위에는 사용하지 않는다.
    """

    _validate_cases(cases)
    if "course_id" not in courses.columns:
        raise ValueError("교과목 데이터에 course_id가 필요합니다.")
    course_records = courses.to_dict(orient="records")
    course_by_id = {str(course["course_id"]): course for course in course_records}
    if len(course_by_id) != len(course_records):
        raise ValueError("교과목 데이터의 course_id는 중복될 수 없습니다.")

    results: list[CourseRecommendationEvaluationResult] = []
    for case in cases:
        try:
            selection = recommender.recommend(
                profile=case.profile(),
                analysis=case.analysis(),
                courses=courses,
                completed_course_ids=(),
            )
            results.append(_evaluate_selection(case, selection, course_by_id))
        except Exception as exc:
            results.append(
                CourseRecommendationEvaluationResult(
                    case_id=case.case_id,
                    title=case.title,
                    department=case.department,
                    selected_course_ids=(),
                    selected_course_names=(),
                    matched_families=(),
                    relevant_course_count=0,
                    home_course_count=0,
                    cross_course_count=0,
                    unknown_course_count=0,
                    duplicate_course_count=0,
                    similarity_backend="오류",
                    passed=False,
                    issues=("추천 실행 오류",),
                    course_matches=(),
                    error=str(exc),
                )
            )

    case_by_id = {case.case_id: case for case in cases}
    details = pd.DataFrame(
        [
            {
                "case_id": result.case_id,
                "title": result.title,
                "department": result.department,
                "recommended_courses": " · ".join(result.selected_course_names),
                "matched_families": " · ".join(result.matched_families) or "확인 필요",
                "family_matches": (
                    f"{len(result.matched_families)}/"
                    f"{len(case_by_id[result.case_id].acceptable_families)}"
                ),
                "relevant_courses": (
                    f"{result.relevant_course_count}/{len(result.selected_course_ids)}"
                ),
                "home_cross_courses": (
                    f"소속 {result.home_course_count} · 타과 {result.cross_course_count}"
                ),
                "backend": result.similarity_backend,
                "passed": result.passed,
                "issues": " · ".join(result.issues),
                "error": result.error or "",
            }
            for result in results
        ]
    )
    return CourseRecommendationEvaluationReport(
        case_count=len(results),
        passed_case_count=sum(result.passed for result in results),
        expected_family_count=sum(len(case.acceptable_families) for case in cases),
        matched_family_count=sum(len(result.matched_families) for result in results),
        selected_course_count=sum(len(result.selected_course_ids) for result in results),
        relevant_course_count=sum(result.relevant_course_count for result in results),
        unknown_course_count=sum(result.unknown_course_count for result in results),
        duplicate_course_count=sum(result.duplicate_course_count for result in results),
        details=details,
        results=tuple(results),
    )


def course_recommendation_evaluation_catalog() -> pd.DataFrame:
    """화면에 표시할 비식별 synthetic 교과 추천 평가 사례를 반환한다."""

    return pd.DataFrame(
        [
            {
                "case_id": case.case_id,
                "title": case.title,
                "department": case.department,
                "grade": case.grade,
                "interest_fields": " · ".join(case.interest_fields),
                "desired_job": case.desired_job,
                "acceptable_families": " · ".join(
                    family.label for family in case.acceptable_families
                ),
            }
            for case in COURSE_RECOMMENDATION_EVALUATION_CASES
        ]
    )
