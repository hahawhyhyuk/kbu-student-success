"""CSV와 향후 대학 DB가 공통으로 지켜야 하는 데이터 공급 계약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


STUDENT_COLUMNS = frozenset(
    {
        "student_id",
        "department",
        "grade",
        "previous_gpa",
        "earned_credits",
        "academic_warning_history",
        "repeat_course_count",
        "archetype",
    }
)
WEEKLY_ACTIVITY_COLUMNS = frozenset(
    {
        "student_id",
        "week",
        "attendance_rate",
        "absence_count",
        "consecutive_absence",
        "late_count",
        "assignment_submission_rate",
        "overdue_assignment_count",
        "lms_login_days",
        "lms_activity_change",
        "video_completion_rate",
        "quiz_score",
    }
)
CHECKIN_COLUMNS = frozenset(
    {
        "student_id",
        "week",
        "major_interest",
        "major_satisfaction",
        "major_continuation_intent",
        "career_clarity",
        "learning_difficulty",
        "consultation_intent",
        "interest_fields",
        "desired_job",
        "natural_language_concern",
    }
)
SUPPORT_PROGRAM_COLUMNS = frozenset(
    {
        "program_id",
        "program_name",
        "program_type",
        "description",
        "target_risk_types",
        "provided_competencies",
        "target_students",
        "department_in_charge",
        "operation_period",
    }
)
COURSE_COLUMNS = frozenset(
    {
        "course_id", "course_name", "department", "description",
        "learning_objectives", "competencies", "related_jobs",
        "related_interests", "grade_level", "semester", "difficulty",
        "prerequisites", "credit", "is_available",
    }
)
COMPLETED_COURSE_COLUMNS = frozenset(
    {"student_id", "course_id", "completion_status", "completion_grade"}
)


@dataclass(frozen=True)
class StudentSuccessData:
    """위험분석과 개입추천에 필요한 canonical DataFrame 묶음.

    OracleRepository를 추가할 때도 DB 컬럼명과 타입을 이 구조에 맞게 정규화한 뒤
    반환해야 한다. 이로써 service와 risk engine은 저장소 종류를 알 필요가 없다.
    """

    students: pd.DataFrame
    weekly_activity: pd.DataFrame
    checkins: pd.DataFrame
    support_programs: pd.DataFrame
    courses: pd.DataFrame
    completed_courses: pd.DataFrame

    def as_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """기존 튜플 기반 호출부와 호환되는 순서로 DataFrame을 반환한다."""

        return self.students, self.weekly_activity, self.checkins

    def as_full_tuple(
        self,
    ) -> tuple[pd.DataFrame, ...]:
        """지원프로그램·교과목·이수내역을 포함한 전체 결과를 반환한다."""

        return (
            self.students,
            self.weekly_activity,
            self.checkins,
            self.support_programs,
            self.courses,
            self.completed_courses,
        )


class StudentSuccessRepository(ABC):
    """원천 저장소와 무관한 학생성공 데이터 접근 인터페이스.

    향후 Oracle 구현은 TNS/credential 처리와 SQL을 구현체 내부에 한정하고,
    아래 메서드에서 canonical pandas DataFrame을 반환하면 된다.
    """

    @abstractmethod
    def get_students(self) -> pd.DataFrame:
        """학생 정적정보를 canonical 컬럼으로 반환한다."""

    @abstractmethod
    def get_weekly_activity(self) -> pd.DataFrame:
        """학생별 주차 활동을 canonical 컬럼으로 반환한다."""

    @abstractmethod
    def get_checkins(self) -> pd.DataFrame:
        """학생 체크인을 canonical 컬럼으로 반환한다."""

    @abstractmethod
    def get_support_programs(self) -> pd.DataFrame:
        """추천 가능한 지원프로그램 master를 canonical 컬럼으로 반환한다."""

    @abstractmethod
    def get_courses(self) -> pd.DataFrame:
        """추천 가능한 교과목 master를 canonical 컬럼으로 반환한다."""

    @abstractmethod
    def get_completed_courses(self) -> pd.DataFrame:
        """학생별 이수과목을 canonical 컬럼으로 반환한다."""

    def get_all(self) -> StudentSuccessData:
        """위험분석에 필요한 전체 데이터셋을 하나의 객체로 반환한다.

        구현체가 transaction 또는 connection 재사용이 필요하면 이 메서드를
        재정의할 수 있다.
        """

        data = StudentSuccessData(
            students=self.get_students(),
            weekly_activity=self.get_weekly_activity(),
            checkins=self.get_checkins(),
            support_programs=self.get_support_programs(),
            courses=self.get_courses(),
            completed_courses=self.get_completed_courses(),
        )
        validate_repository_data(data)
        return data


def validate_repository_data(data: StudentSuccessData) -> None:
    """Repository 결과가 위험엔진의 canonical 스키마를 만족하는지 확인한다.

    Parameters:
        data: Repository가 반환한 세 DataFrame 묶음.

    Returns:
        없음. 유효하지 않으면 ValueError를 발생시킨다.

    Assumptions:
        추가 컬럼은 허용하되 필수 컬럼 누락과 알 수 없는 학생 참조는 허용하지 않는다.
    """

    frames = (
        ("students", data.students, STUDENT_COLUMNS),
        ("weekly_activity", data.weekly_activity, WEEKLY_ACTIVITY_COLUMNS),
        ("checkins", data.checkins, CHECKIN_COLUMNS),
        ("support_programs", data.support_programs, SUPPORT_PROGRAM_COLUMNS),
        ("courses", data.courses, COURSE_COLUMNS),
        ("completed_courses", data.completed_courses, COMPLETED_COURSE_COLUMNS),
    )
    for name, frame, required_columns in frames:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{name}는 pandas DataFrame이어야 합니다.")
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} 데이터에 필수 컬럼이 없습니다: {sorted(missing)}"
            )

    student_ids = set(data.students["student_id"])
    for name, frame in (
        ("weekly_activity", data.weekly_activity),
        ("checkins", data.checkins),
    ):
        unknown_ids = set(frame["student_id"]).difference(student_ids)
        if unknown_ids:
            sample = sorted(str(value) for value in unknown_ids)[:5]
            raise ValueError(
                f"{name} 데이터가 존재하지 않는 student_id를 참조합니다: {sample}"
            )

    if data.support_programs["program_id"].duplicated().any():
        raise ValueError("support_programs의 program_id는 중복될 수 없습니다.")
    if data.courses["course_id"].duplicated().any():
        raise ValueError("courses의 course_id는 중복될 수 없습니다.")
    course_ids = set(data.courses["course_id"])
    unknown_completed_students = set(data.completed_courses["student_id"]).difference(student_ids)
    unknown_completed_courses = set(data.completed_courses["course_id"]).difference(course_ids)
    if unknown_completed_students or unknown_completed_courses:
        raise ValueError("completed_courses가 존재하지 않는 학생 또는 교과목을 참조합니다.")
    if data.completed_courses[["student_id", "course_id"]].duplicated().any():
        raise ValueError("학생별 이수과목은 중복될 수 없습니다.")
