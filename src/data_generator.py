"""학생 archetype을 기반으로 재현 가능한 시연용 가상데이터를 생성한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.department_catalog import (
    DepartmentDefinition,
    configured_department_names,
    load_department_catalog,
)
from src.utils import load_app_config


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


DEFAULT_SEED = 2026
DEFAULT_STUDENT_COUNT = 200
DEPARTMENTS = list(configured_department_names())
DEMO_ALERT_ACTIVITY_PATH: dict[int, dict[str, float | int]] = {
    1: {
        "attendance_rate": 96.0,
        "absence_count": 0,
        "consecutive_absence": 0,
        "late_count": 0,
        "assignment_submission_rate": 95.0,
        "overdue_assignment_count": 0,
        "lms_login_days": 5,
        "lms_activity_change": 0.0,
        "video_completion_rate": 92.0,
        "quiz_score": 82.0,
    },
    2: {
        "attendance_rate": 91.0,
        "absence_count": 1,
        "consecutive_absence": 0,
        "late_count": 1,
        "assignment_submission_rate": 88.0,
        "overdue_assignment_count": 1,
        "lms_login_days": 4,
        "lms_activity_change": -8.0,
        "video_completion_rate": 82.0,
        "quiz_score": 75.0,
    },
    3: {
        "attendance_rate": 84.0,
        "absence_count": 2,
        "consecutive_absence": 1,
        "late_count": 2,
        "assignment_submission_rate": 75.0,
        "overdue_assignment_count": 2,
        "lms_login_days": 3,
        "lms_activity_change": -22.0,
        "video_completion_rate": 70.0,
        "quiz_score": 68.0,
    },
    4: {
        "attendance_rate": 77.0,
        "absence_count": 4,
        "consecutive_absence": 3,
        "late_count": 4,
        "assignment_submission_rate": 60.0,
        "overdue_assignment_count": 3,
        "lms_login_days": 2,
        "lms_activity_change": -40.0,
        "video_completion_rate": 55.0,
        "quiz_score": 62.0,
    },
}
DEMO_FOLLOWUP_COLUMNS = (
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
    "major_interest",
    "major_satisfaction",
    "major_continuation_intent",
    "career_clarity",
    "learning_difficulty",
    "consultation_intent",
    "interest_fields",
    "desired_job",
    "natural_language_concern",
    "observation_note",
)
ARCHETYPE_PROPORTIONS = {
    "normal": 0.45,
    "attendance_risk": 0.10,
    "engagement_risk": 0.10,
    "achievement_risk": 0.10,
    "major_mismatch": 0.10,
    "career_unclear": 0.08,
    "complex_risk": 0.07,
}


CONCERN_TEMPLATES = {
    "normal": [
        "현재 수업은 대체로 잘 따라가고 있으며 앞으로도 학습 계획을 꾸준히 유지하고 싶습니다.",
        "학교생활에 큰 어려움은 없고 이번 학기에는 전공 역량을 더 탄탄히 쌓고 싶습니다.",
        "수업과 과제를 계획대로 수행하고 있으며 관심 분야를 조금 더 탐색해 보고 싶습니다.",
    ],
    "attendance_risk": [
        "최근 일정 관리가 잘되지 않아 수업에 빠지는 날이 생겼고 출석 흐름을 다시 잡고 싶습니다.",
        "몇 차례 연속으로 수업에 참여하지 못해 진도를 놓칠까 걱정되고 보충 방법이 궁금합니다.",
        "요즘 등교와 시간 관리에 어려움이 있어 출석을 안정적으로 회복하는 데 도움이 필요합니다.",
    ],
    "engagement_risk": [
        "최근 수업과 과제를 따라가는 것이 조금 버겁고 공부에 집중하기 어려운 날이 많았습니다.",
        "LMS에 접속하는 횟수가 줄고 과제도 미루게 되어 학습 리듬을 되찾고 싶습니다.",
        "강의 영상과 과제를 제때 끝내지 못하는 일이 늘어 효과적인 학습 계획이 필요합니다.",
    ],
    "achievement_risk": [
        "공부한 만큼 성적이 나오지 않고 기초 내용이 부족한 것 같아 학습 방법을 상담받고 싶습니다.",
        "퀴즈에서 어려운 문제가 많았고 이전에 부족했던 과목까지 함께 보완할 방법이 필요합니다.",
        "수업 내용을 이해하는 데 시간이 오래 걸려 기초부터 다시 공부해야 할지 고민됩니다.",
    ],
    "major_mismatch": [
        "현재 전공이 생각했던 것과 조금 다른 것 같고 다른 분야에도 관심이 있습니다. 앞으로 무엇을 공부해야 할지 고민됩니다.",
        "전공 수업이 제 적성과 맞는지 확신이 없고 콘텐츠와 기획 분야도 함께 탐색해 보고 싶습니다.",
        "지금 전공을 계속 공부할지 고민 중이며 관심 있는 데이터와 디자인 분야를 어떻게 연결할지 궁금합니다.",
    ],
    "career_unclear": [
        "아직 어떤 일을 하고 싶은지 정하지 못했고 지금 전공을 어떻게 진로와 연결해야 할지 잘 모르겠습니다.",
        "관심 분야는 몇 가지 있지만 구체적인 희망직무가 없어 진로 탐색을 어디서 시작해야 할지 고민됩니다.",
        "졸업 후 진로가 선명하지 않아 제 강점과 맞는 직무를 알아보고 싶습니다.",
    ],
    "complex_risk": [
        "최근 결석과 밀린 과제가 함께 늘었고 전공과 진로에 대한 확신도 낮아 무엇부터 해결해야 할지 모르겠습니다.",
        "수업 참여와 성적이 모두 걱정되고 현재 전공을 계속해야 할지도 고민되어 단계적인 도움이 필요합니다.",
        "학습 진도를 놓친 뒤 학교생활 전반에 자신감이 떨어졌고 진로 방향도 함께 상담받고 싶습니다.",
    ],
}


STUDENT_PROFILE_RULES: dict[str, dict[str, Any]] = {
    "normal": {
        "gpa": (3.65, 0.32),
        "warning_probabilities": (0.96, 0.04, 0.0),
        "repeat_probabilities": (0.86, 0.12, 0.02, 0.0),
    },
    "attendance_risk": {
        "gpa": (3.15, 0.42),
        "warning_probabilities": (0.83, 0.16, 0.01),
        "repeat_probabilities": (0.70, 0.22, 0.07, 0.01),
    },
    "engagement_risk": {
        "gpa": (3.05, 0.43),
        "warning_probabilities": (0.80, 0.18, 0.02),
        "repeat_probabilities": (0.66, 0.24, 0.08, 0.02),
    },
    "achievement_risk": {
        "gpa": (2.05, 0.38),
        "warning_probabilities": (0.34, 0.48, 0.18),
        "repeat_probabilities": (0.18, 0.34, 0.31, 0.17),
    },
    "major_mismatch": {
        "gpa": (3.20, 0.38),
        "warning_probabilities": (0.84, 0.15, 0.01),
        "repeat_probabilities": (0.72, 0.22, 0.05, 0.01),
    },
    "career_unclear": {
        "gpa": (3.35, 0.38),
        "warning_probabilities": (0.90, 0.09, 0.01),
        "repeat_probabilities": (0.77, 0.19, 0.04, 0.0),
    },
    "complex_risk": {
        "gpa": (2.15, 0.48),
        "warning_probabilities": (0.28, 0.48, 0.24),
        "repeat_probabilities": (0.16, 0.30, 0.32, 0.22),
    },
}


ACTIVITY_RULES: dict[str, dict[str, tuple[float, float, float]]] = {
    "normal": {
        "attendance_rate": (97, 1.8, -0.2),
        "assignment_submission_rate": (96, 3.0, -0.3),
        "lms_login_days": (5.5, 0.8, 0.0),
        "lms_activity_change": (3, 7.0, -0.5),
        "video_completion_rate": (94, 4.5, -0.3),
        "quiz_score": (87, 6.5, 0.0),
    },
    "attendance_risk": {
        "attendance_rate": (91, 3.0, -3.0),
        "assignment_submission_rate": (91, 5.0, -0.8),
        "lms_login_days": (4.8, 0.9, -0.1),
        "lms_activity_change": (-2, 8.0, -1.0),
        "video_completion_rate": (88, 6.0, -0.8),
        "quiz_score": (78, 8.0, -0.5),
    },
    "engagement_risk": {
        "attendance_rate": (95, 2.2, -0.5),
        "assignment_submission_rate": (89, 5.0, -6.0),
        "lms_login_days": (4.3, 0.8, -0.65),
        "lms_activity_change": (-10, 7.0, -8.0),
        "video_completion_rate": (84, 6.0, -6.0),
        "quiz_score": (76, 8.0, -1.5),
    },
    "achievement_risk": {
        "attendance_rate": (94, 2.5, -0.4),
        "assignment_submission_rate": (87, 6.0, -1.0),
        "lms_login_days": (4.4, 0.9, -0.1),
        "lms_activity_change": (-5, 8.0, -1.5),
        "video_completion_rate": (84, 7.0, -1.0),
        "quiz_score": (65, 7.5, -2.0),
    },
    "major_mismatch": {
        "attendance_rate": (96, 2.0, -0.3),
        "assignment_submission_rate": (91, 5.0, -0.8),
        "lms_login_days": (4.8, 0.8, -0.1),
        "lms_activity_change": (0, 7.0, -1.0),
        "video_completion_rate": (89, 6.0, -0.8),
        "quiz_score": (80, 7.0, -0.5),
    },
    "career_unclear": {
        "attendance_rate": (96, 2.0, -0.2),
        "assignment_submission_rate": (93, 4.5, -0.5),
        "lms_login_days": (5.0, 0.8, -0.1),
        "lms_activity_change": (1, 7.0, -0.8),
        "video_completion_rate": (91, 5.5, -0.5),
        "quiz_score": (82, 7.0, -0.5),
    },
    "complex_risk": {
        "attendance_rate": (90, 3.5, -4.0),
        "assignment_submission_rate": (84, 6.5, -7.0),
        "lms_login_days": (3.8, 0.9, -0.6),
        "lms_activity_change": (-12, 8.0, -7.0),
        "video_completion_rate": (80, 7.0, -6.0),
        "quiz_score": (64, 8.0, -2.0),
    },
}


CHECKIN_RULES: dict[str, dict[str, Any]] = {
    "normal": {
        "major_range": (4, 5),
        "career_range": (4, 5),
        "learning_range": (1, 2),
        "consultation_range": (1, 2),
        "interests": ["데이터·AI", "콘텐츠·디자인", "서비스"],
        "jobs": ["데이터 분석", "콘텐츠 기획", "서비스 기획"],
    },
    "attendance_risk": {
        "major_range": (3, 5),
        "career_range": (3, 5),
        "learning_range": (2, 4),
        "consultation_range": (3, 5),
        "interests": ["서비스", "경영·마케팅", "데이터·AI"],
        "jobs": ["서비스 운영", "마케팅 기획", "데이터 분석"],
    },
    "engagement_risk": {
        "major_range": (3, 4),
        "career_range": (3, 4),
        "learning_range": (3, 5),
        "consultation_range": (3, 5),
        "interests": ["콘텐츠·디자인", "서비스", "데이터·AI"],
        "jobs": ["콘텐츠 기획", "서비스 기획", "데이터 분석"],
    },
    "achievement_risk": {
        "major_range": (3, 4),
        "career_range": (2, 4),
        "learning_range": (4, 5),
        "consultation_range": (4, 5),
        "interests": ["데이터·AI", "서비스", "상담·복지"],
        "jobs": ["데이터 실무", "서비스 운영", "상담 지원"],
    },
    "major_mismatch": {
        "major_range": (1, 2),
        "career_range": (2, 4),
        "learning_range": (2, 4),
        "consultation_range": (4, 5),
        "interests": ["콘텐츠·디자인", "경영·마케팅", "데이터·AI"],
        "jobs": ["콘텐츠 기획", "디지털 마케팅", "데이터 기획"],
    },
    "career_unclear": {
        "major_range": (3, 4),
        "career_range": (1, 2),
        "learning_range": (2, 3),
        "consultation_range": (3, 5),
        "interests": ["데이터·AI", "경영·마케팅", "콘텐츠·디자인"],
        "jobs": [],
    },
    "complex_risk": {
        "major_range": (1, 3),
        "career_range": (1, 2),
        "learning_range": (4, 5),
        "consultation_range": (4, 5),
        "interests": ["콘텐츠·디자인", "서비스", "경영·마케팅"],
        "jobs": ["", "서비스 기획"],
    },
}


SUPPORT_PROGRAMS = [
    {
        "program_id": "P001",
        "program_name": "출석회복 코칭",
        "program_type": "학업상담",
        "description": "결석 원인을 점검하고 출석 계획과 주간 실천 목표를 함께 수립하는 개별 코칭",
        "target_risk_types": "attendance|complex",
        "provided_competencies": "시간관리|출석계획|학업지속",
        "target_students": "결석 또는 연속결석이 증가한 학생",
        "department_in_charge": "DEMO_UNIT01",
        "operation_period": "학기 중 상시",
    },
    {
        "program_id": "P002",
        "program_name": "학습리듬 회복 코칭",
        "program_type": "학습코칭",
        "description": "LMS 활동과 수업 참여가 감소한 학생에게 현실적인 주간 학습 루틴을 설계하도록 지원",
        "target_risk_types": "engagement|complex",
        "provided_competencies": "학습계획|집중관리|자기점검",
        "target_students": "학습참여 지표가 최근 감소한 학생",
        "department_in_charge": "DEMO_UNIT01",
        "operation_period": "학기 중 상시",
    },
    {
        "program_id": "P003",
        "program_name": "과제관리 클리닉",
        "program_type": "학습지원",
        "description": "미제출과 지연 제출을 줄이기 위해 과제 분해, 일정 수립, 제출 점검 방법을 실습",
        "target_risk_types": "engagement|achievement|complex",
        "provided_competencies": "과제관리|일정관리|학습실행",
        "target_students": "과제 제출률이 낮거나 지연 제출이 있는 학생",
        "department_in_charge": "DEMO_UNIT02",
        "operation_period": "4주 단위",
    },
    {
        "program_id": "P004",
        "program_name": "기초학습 튜터링",
        "program_type": "튜터링",
        "description": "기초 개념 이해와 문제풀이가 필요한 학생을 튜터와 연결해 단계별 학습을 지원",
        "target_risk_types": "achievement|complex",
        "provided_competencies": "기초학습|문제해결|학업자신감",
        "target_students": "퀴즈 또는 직전 성취도가 낮은 학생",
        "department_in_charge": "DEMO_UNIT02",
        "operation_period": "학기별 모집",
    },
    {
        "program_id": "P005",
        "program_name": "학업전략 상담",
        "program_type": "학업상담",
        "description": "학업경고, 재수강, 성적 저하 원인을 함께 검토하고 개인별 학업 회복 계획을 수립",
        "target_risk_types": "achievement|complex",
        "provided_competencies": "학업계획|성취관리|자기점검",
        "target_students": "학업성취 회복 계획이 필요한 학생",
        "department_in_charge": "DEMO_UNIT01",
        "operation_period": "학기 중 상시",
    },
    {
        "program_id": "P006",
        "program_name": "전공적응 개별상담",
        "program_type": "전공상담",
        "description": "전공 흥미, 만족도, 지속 의향을 점검하고 현재 전공 안에서 가능한 학습 선택지를 탐색",
        "target_risk_types": "major_adaptation|complex",
        "provided_competencies": "전공이해|자기이해|학업의사결정",
        "target_students": "전공 만족도 또는 지속 의향이 낮은 학생",
        "department_in_charge": "DEMO_UNIT03",
        "operation_period": "학기 중 상시",
    },
    {
        "program_id": "P007",
        "program_name": "관심분야 탐색 워크숍",
        "program_type": "전공탐색",
        "description": "여러 관심분야와 전공 역량의 연결점을 비교하고 후속 탐색 계획을 만드는 소그룹 워크숍",
        "target_risk_types": "major_adaptation|career|complex",
        "provided_competencies": "관심탐색|역량연결|의사결정",
        "target_students": "다른 분야에 관심이 있거나 전공 방향을 고민하는 학생",
        "department_in_charge": "DEMO_UNIT03",
        "operation_period": "월 1회",
    },
    {
        "program_id": "P008",
        "program_name": "진로설계 개별상담",
        "program_type": "진로상담",
        "description": "흥미와 강점을 바탕으로 희망직무 후보를 만들고 학기별 진로 행동계획을 수립",
        "target_risk_types": "career|major_adaptation|complex",
        "provided_competencies": "자기이해|진로설계|목표수립",
        "target_students": "진로 명확도가 낮거나 희망직무를 정하지 못한 학생",
        "department_in_charge": "DEMO_UNIT04",
        "operation_period": "학기 중 상시",
    },
    {
        "program_id": "P009",
        "program_name": "직무탐색 워크숍",
        "program_type": "진로탐색",
        "description": "관심 산업의 직무 정보와 필요 역량을 조사하고 자신에게 맞는 직무 후보를 비교",
        "target_risk_types": "career|major_adaptation",
        "provided_competencies": "직무이해|정보탐색|진로의사결정",
        "target_students": "관심분야는 있으나 구체적 직무가 없는 학생",
        "department_in_charge": "DEMO_UNIT04",
        "operation_period": "월 1회",
    },
    {
        "program_id": "P010",
        "program_name": "복합지원 사례관리",
        "program_type": "통합지원",
        "description": "출결, 학습, 전공, 진로 문제가 함께 나타난 학생에게 지원 순서를 정하고 담당 프로그램을 연계",
        "target_risk_types": "complex",
        "provided_competencies": "지원계획|목표관리|자원연계",
        "target_students": "두 개 이상 위험영역에서 지원이 필요한 학생",
        "department_in_charge": "DEMO_UNIT05",
        "operation_period": "학기 중 상시",
    },
    {
        "program_id": "P011",
        "program_name": "동료 학습멘토링",
        "program_type": "멘토링",
        "description": "학습경험이 있는 동료 멘토와 정기적으로 만나 수업 적응과 학습 실행을 점검",
        "target_risk_types": "engagement|achievement|major_adaptation",
        "provided_competencies": "학습실행|대학생활적응|상호협력",
        "target_students": "학습 또는 대학생활 적응에 동료 지원이 필요한 학생",
        "department_in_charge": "DEMO_UNIT02",
        "operation_period": "학기별 모집",
    },
    {
        "program_id": "P012",
        "program_name": "주간 학습계획 워크숍",
        "program_type": "학습지원",
        "description": "수업, 과제, LMS 학습을 한 주 계획으로 정리하고 실행 가능한 시간표를 완성",
        "target_risk_types": "attendance|engagement|achievement|complex",
        "provided_competencies": "시간관리|학습계획|과제관리",
        "target_students": "주간 학습 계획과 실행 점검이 필요한 학생",
        "department_in_charge": "DEMO_UNIT02",
        "operation_period": "격주 운영",
    },
]


COURSE_TRACK_TEMPLATES: dict[str, list[tuple[Any, ...]]] = {
    "data_ai": [
        ("데이터 사고 기초", "데이터이해|문제정의", "데이터 분석|서비스 기획", "데이터·AI", 1, "기초", None),
        ("스프레드시트 분석", "데이터정리|기초분석", "데이터 분석|마케팅 기획", "데이터·AI|경영·마케팅", 1, "기초", None),
        ("Python 활용 기초", "프로그래밍|자동화", "데이터 분석|AI 서비스", "데이터·AI", 1, "기초", None),
        ("데이터 시각화", "시각화|데이터소통", "데이터 분석|콘텐츠 기획", "데이터·AI|콘텐츠·디자인", 2, "중급", 1),
        ("응용 통계", "통계분석|의사결정", "데이터 분석|시장 분석", "데이터·AI|경영·마케팅", 2, "중급", 1),
        ("머신러닝 이해", "AI이해|예측분석", "AI 서비스|데이터 분석", "데이터·AI", 3, "심화", 3),
        ("데이터 프로젝트", "프로젝트관리|데이터활용", "데이터 분석|서비스 기획", "데이터·AI|서비스", 3, "심화", 4),
        ("AI 서비스 기획", "AI기획|서비스설계", "AI 서비스|서비스 기획", "데이터·AI|서비스", 3, "심화", 6),
    ],
    "business": [
        ("경영 이해", "비즈니스이해|의사결정", "경영 기획|서비스 운영", "경영·마케팅|서비스", 1, "기초", None),
        ("소비자 이해", "고객분석|조사기초", "마케팅 기획|서비스 기획", "경영·마케팅|서비스", 1, "기초", None),
        ("마케팅 기초", "마케팅|시장분석", "마케팅 기획|브랜드 기획", "경영·마케팅", 1, "기초", None),
        ("디지털 마케팅", "디지털마케팅|캠페인기획", "디지털 마케팅|SNS 마케팅", "경영·마케팅|콘텐츠·디자인", 2, "중급", 3),
        ("서비스 운영", "서비스운영|고객경험", "서비스 운영|고객 관리", "서비스|경영·마케팅", 2, "중급", 1),
        ("비즈니스 데이터", "데이터활용|경영분석", "시장 분석|경영 기획", "데이터·AI|경영·마케팅", 2, "중급", 1),
        ("브랜드 콘텐츠", "브랜드기획|콘텐츠기획", "브랜드 기획|콘텐츠 기획", "경영·마케팅|콘텐츠·디자인", 3, "심화", 4),
        ("비즈니스 프로젝트", "프로젝트관리|사업기획", "경영 기획|창업", "경영·마케팅|서비스", 3, "심화", 6),
    ],
    "content_design": [
        ("디자인 기초", "시각표현|디자인이해", "콘텐츠 디자인|브랜드 디자인", "콘텐츠·디자인", 1, "기초", None),
        ("시각 콘텐츠 제작", "콘텐츠제작|시각화", "콘텐츠 제작|디자인", "콘텐츠·디자인", 1, "기초", None),
        ("스토리텔링", "스토리기획|콘텐츠기획", "콘텐츠 기획|마케팅 기획", "콘텐츠·디자인|경영·마케팅", 1, "기초", None),
        ("UX 기초", "사용자이해|UX설계", "UX 기획|서비스 기획", "콘텐츠·디자인|서비스", 2, "중급", 1),
        ("영상 콘텐츠 제작", "영상제작|콘텐츠표현", "영상 콘텐츠|SNS 콘텐츠", "콘텐츠·디자인", 2, "중급", 2),
        ("디지털 콘텐츠 기획", "디지털기획|스토리설계", "콘텐츠 기획|서비스 기획", "콘텐츠·디자인|서비스", 2, "중급", 3),
        ("콘텐츠 마케팅", "콘텐츠마케팅|고객소통", "콘텐츠 마케팅|SNS 마케팅", "콘텐츠·디자인|경영·마케팅", 3, "심화", 6),
        ("콘텐츠 포트폴리오", "포트폴리오|프로젝트관리", "콘텐츠 제작|디자인", "콘텐츠·디자인", 3, "심화", 5),
    ],
    "service": [
        ("서비스 커뮤니케이션", "의사소통|고객응대", "서비스 운영|고객 관리", "서비스", 1, "기초", None),
        ("고객경험 이해", "고객이해|경험분석", "서비스 기획|고객 관리", "서비스|경영·마케팅", 1, "기초", None),
        ("서비스 기획 기초", "서비스기획|문제정의", "서비스 기획|서비스 운영", "서비스", 1, "기초", None),
        ("프로젝트 관리", "프로젝트관리|협업", "서비스 기획|프로젝트 관리", "서비스|경영·마케팅", 2, "중급", 3),
        ("조직 협업", "협업|갈등조정", "조직 운영|서비스 운영", "서비스|상담·복지", 2, "중급", 1),
        ("상담 커뮤니케이션", "상담기초|공감소통", "상담 지원|고객 관리", "상담·복지|서비스", 2, "중급", 1),
        ("서비스 디자인", "서비스설계|UX설계", "서비스 기획|UX 기획", "서비스|콘텐츠·디자인", 3, "심화", 2),
        ("현장 문제해결", "문제해결|프로젝트실행", "서비스 운영|프로젝트 관리", "서비스", 3, "심화", 4),
    ],
    "health": [
        ("보건 의사소통", "보건소통|정보전달", "보건 서비스|건강 콘텐츠", "보건|서비스", 1, "기초", None),
        ("건강정보 이해", "건강정보|자료해석", "보건 정보|보건 서비스", "보건|데이터·AI", 1, "기초", None),
        ("보건서비스 기초", "보건서비스|대상자이해", "보건 서비스|서비스 운영", "보건|서비스", 1, "기초", None),
        ("데이터 기반 보건", "보건데이터|기초분석", "보건 데이터|보건 기획", "보건|데이터·AI", 2, "중급", 2),
        ("건강 콘텐츠 기획", "건강콘텐츠|정보설계", "건강 콘텐츠|콘텐츠 기획", "보건|콘텐츠·디자인", 2, "중급", 1),
        ("현장 안전관리", "안전관리|위험예방", "안전 관리|보건 서비스", "보건|서비스", 2, "중급", 3),
        ("대상자 경험설계", "경험설계|서비스개선", "보건 서비스|서비스 기획", "보건|서비스", 3, "심화", 3),
        ("보건 프로젝트", "프로젝트관리|보건기획", "보건 기획|보건 서비스", "보건|서비스", 3, "심화", 4),
    ],
    "counseling_welfare": [
        ("상담 이해", "상담기초|자기이해", "상담 지원|복지 서비스", "상담·복지", 1, "기초", None),
        ("복지서비스 이해", "복지이해|대상자이해", "복지 서비스|사례 지원", "상담·복지|서비스", 1, "기초", None),
        ("대인관계와 소통", "대인관계|공감소통", "상담 지원|서비스 운영", "상담·복지|서비스", 1, "기초", None),
        ("상담기법 기초", "상담기법|의사소통", "상담 지원|사례 지원", "상담·복지", 2, "중급", 1),
        ("복지 프로그램 기획", "프로그램기획|요구분석", "복지 기획|서비스 기획", "상담·복지|서비스", 2, "중급", 2),
        ("사례관리 기초", "사례관리|자원연계", "사례 지원|복지 서비스", "상담·복지", 2, "중급", 1),
        ("지역사회 프로젝트", "지역분석|프로젝트관리", "복지 기획|프로젝트 관리", "상담·복지|서비스", 3, "심화", 5),
        ("디지털 복지콘텐츠", "디지털콘텐츠|복지소통", "복지 콘텐츠|콘텐츠 기획", "상담·복지|콘텐츠·디자인", 3, "심화", 5),
    ],
}


def _archetype_counts(student_count: int) -> dict[str, int]:
    """비율 합을 보존하며 archetype별 정수 인원수를 계산한다."""

    raw_counts = {
        name: student_count * proportion
        for name, proportion in ARCHETYPE_PROPORTIONS.items()
    }
    counts = {name: int(value) for name, value in raw_counts.items()}
    remainder = student_count - sum(counts.values())
    ranked = sorted(
        raw_counts,
        key=lambda name: raw_counts[name] - counts[name],
        reverse=True,
    )
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def generate_students(
    student_count: int = DEFAULT_STUDENT_COUNT, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """archetype 분포를 먼저 배정하고 학생 정적정보를 생성한다.

    Parameters:
        student_count: 생성할 학생 수.
        seed: NumPy 난수 시드.

    Returns:
        개인정보 필드가 없는 학생 DataFrame.

    Assumptions:
        GPA는 4.5 만점이며 학년은 1~3학년으로 제한한다.
    """

    rng = np.random.default_rng(seed)
    archetypes: list[str] = []
    for archetype, count in _archetype_counts(student_count).items():
        archetypes.extend([archetype] * count)
    rng.shuffle(archetypes)

    records: list[dict[str, Any]] = []
    credit_centers = {1: 18, 2: 52, 3: 86}
    for index, archetype in enumerate(archetypes, start=1):
        rules = STUDENT_PROFILE_RULES[archetype]
        grade = int(rng.choice([1, 2, 3], p=[0.38, 0.34, 0.28]))
        gpa = float(
            np.clip(rng.normal(rules["gpa"][0], rules["gpa"][1]), 1.0, 4.5)
        )
        warning = int(
            rng.choice([0, 1, 2], p=rules["warning_probabilities"])
        )
        repeat_count = int(
            rng.choice([0, 1, 2, 3], p=rules["repeat_probabilities"])
        )
        earned_credits = int(
            np.clip(
                rng.normal(credit_centers[grade] + (gpa - 3.0) * 3, 5),
                6,
                115,
            )
        )
        records.append(
            {
                "student_id": f"S{index:04d}",
                "department": str(rng.choice(DEPARTMENTS)),
                "grade": grade,
                "previous_gpa": round(gpa, 2),
                "earned_credits": earned_credits,
                "academic_warning_history": warning,
                "repeat_course_count": repeat_count,
                "archetype": archetype,
            }
        )
    return pd.DataFrame(records)


def _activity_value(
    rng: np.random.Generator,
    rule: tuple[float, float, float],
    week: int,
    lower: float,
    upper: float,
) -> float:
    """기준값, 노이즈, 주차 추세로 하나의 활동지표를 생성한다."""

    center, standard_deviation, weekly_trend = rule
    return float(
        np.clip(
            rng.normal(center + weekly_trend * (week - 1), standard_deviation),
            lower,
            upper,
        )
    )


def generate_weekly_activity(
    students: pd.DataFrame, seed: int = DEFAULT_SEED + 1
) -> pd.DataFrame:
    """학생 archetype별 1~4주 활동 패턴을 생성한다.

    Parameters:
        students: student_id와 archetype을 포함한 학생 DataFrame.
        seed: NumPy 난수 시드.

    Returns:
        학생당 네 행인 주차별 활동 DataFrame.

    Assumptions:
        위험 archetype의 일부 지표는 주차가 지날수록 악화되며 모든 비율은 0~100이다.
    """

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for student in students[["student_id", "archetype"]].to_dict(
        orient="records"
    ):
        archetype = str(student["archetype"])
        rules = ACTIVITY_RULES[archetype]
        for week in range(1, 5):
            attendance = _activity_value(
                rng, rules["attendance_rate"], week, 55, 100
            )
            assignment = _activity_value(
                rng, rules["assignment_submission_rate"], week, 25, 100
            )
            login_days = _activity_value(
                rng, rules["lms_login_days"], week, 0, 7
            )
            activity_change = _activity_value(
                rng, rules["lms_activity_change"], week, -75, 30
            )
            video = _activity_value(
                rng, rules["video_completion_rate"], week, 20, 100
            )
            quiz = _activity_value(rng, rules["quiz_score"], week, 20, 100)

            expected_absences = max(0.0, (100 - attendance) / 8)
            absence_count = int(
                np.clip(round(rng.normal(expected_absences, 0.6)), 0, 6)
            )
            if archetype in {"attendance_risk", "complex_risk"}:
                consecutive_upper = min(4, 1 + week)
                consecutive_absence = int(
                    rng.integers(max(0, week - 2), consecutive_upper + 1)
                )
            else:
                consecutive_absence = int(
                    rng.choice([0, 1, 2], p=[0.84, 0.14, 0.02])
                )
            consecutive_absence = min(consecutive_absence, absence_count)
            late_count = int(
                np.clip(round(rng.normal(max(0, (96 - attendance) / 6), 0.7)), 0, 5)
            )
            overdue_count = int(
                np.clip(round(rng.normal(max(0, (95 - assignment) / 18), 0.6)), 0, 5)
            )

            records.append(
                {
                    "student_id": student["student_id"],
                    "week": week,
                    "attendance_rate": round(attendance, 1),
                    "absence_count": absence_count,
                    "consecutive_absence": consecutive_absence,
                    "late_count": late_count,
                    "assignment_submission_rate": round(assignment, 1),
                    "overdue_assignment_count": overdue_count,
                    "lms_login_days": int(round(login_days)),
                    "lms_activity_change": round(activity_change, 1),
                    "video_completion_rate": round(video, 1),
                    "quiz_score": round(quiz, 1),
                }
            )
    activity = pd.DataFrame(records)
    demo_student_id = str(
        load_app_config()["demo"]["representative_student_id"]
    ).strip()
    demo_student = students[
        students["student_id"].astype(str) == demo_student_id
    ]
    if (
        not demo_student.empty
        and str(demo_student.iloc[0]["archetype"]) == "complex_risk"
    ):
        for week, values in DEMO_ALERT_ACTIVITY_PATH.items():
            target = (
                (activity["student_id"].astype(str) == demo_student_id)
                & (activity["week"].astype(int) == week)
            )
            for column, value in values.items():
                activity.loc[target, column] = value
    return activity


def _random_likert(
    rng: np.random.Generator, value_range: tuple[int, int]
) -> int:
    """양 끝을 포함하는 Likert 정수를 샘플링한다."""

    return int(rng.integers(value_range[0], value_range[1] + 1))


def generate_checkins(
    students: pd.DataFrame, seed: int = DEFAULT_SEED + 2
) -> pd.DataFrame:
    """학생별 4주차 정량·자유서술 체크인 한 건을 생성한다.

    Parameters:
        students: student_id와 archetype을 포함한 학생 DataFrame.
        seed: NumPy 난수 시드.

    Returns:
        학생당 한 행인 체크인 DataFrame.

    Assumptions:
        major_mismatch의 전공 문항은 1~2, career_unclear의 진로 명확도는 1~2이며
        희망직무는 빈 문자열로 생성한다.
    """

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for student in students[["student_id", "archetype"]].to_dict(
        orient="records"
    ):
        archetype = str(student["archetype"])
        rules = CHECKIN_RULES[archetype]
        interests = list(rules["interests"])
        interest_count = int(rng.integers(1, min(2, len(interests)) + 1))
        selected_interests = rng.choice(
            interests, size=interest_count, replace=False
        ).tolist()
        jobs = list(rules["jobs"])
        desired_job = str(rng.choice(jobs)) if jobs else ""
        records.append(
            {
                "student_id": student["student_id"],
                "week": 4,
                "major_interest": _random_likert(rng, rules["major_range"]),
                "major_satisfaction": _random_likert(
                    rng, rules["major_range"]
                ),
                "major_continuation_intent": _random_likert(
                    rng, rules["major_range"]
                ),
                "career_clarity": _random_likert(rng, rules["career_range"]),
                "learning_difficulty": _random_likert(
                    rng, rules["learning_range"]
                ),
                "consultation_intent": _random_likert(
                    rng, rules["consultation_range"]
                ),
                "interest_fields": "|".join(selected_interests),
                "desired_job": desired_job,
                "natural_language_concern": str(
                    rng.choice(CONCERN_TEMPLATES[archetype])
                ),
            }
        )
    return pd.DataFrame(records)


def generate_demo_followup_observations() -> pd.DataFrame:
    """대표 synthetic 학생의 15주차 후속 관찰 한 건을 생성한다.

    Returns:
        출결·LMS·성취와 학생 자기보고를 함께 가진 비식별 후속 관찰 DataFrame.

    Assumptions:
        이 값은 지원 효과를 입증하는 실제 데이터가 아니라, 상담 완료 뒤 후속 확인
        화면을 재현하기 위한 고정 시나리오다. 일반 1~4주차 원천에는 합치지 않는다.
    """

    demo_config = load_app_config()["demo"]
    followup_config = demo_config["followup"]
    record = {
        "student_id": str(demo_config["representative_student_id"]),
        "week": int(followup_config["week"]),
        "attendance_rate": 92.0,
        "absence_count": 1,
        "consecutive_absence": 0,
        "late_count": 1,
        "assignment_submission_rate": 86.0,
        "overdue_assignment_count": 1,
        "lms_login_days": 4,
        "lms_activity_change": 8.0,
        "video_completion_rate": 82.0,
        "quiz_score": 76.0,
        "major_interest": 4,
        "major_satisfaction": 4,
        "major_continuation_intent": 4,
        "career_clarity": 4,
        "learning_difficulty": 2,
        "consultation_intent": 2,
        "interest_fields": "서비스|보건",
        "desired_job": "헬스케어 서비스 기획",
        "natural_language_concern": (
            "상담 후 출석과 과제 흐름을 회복했고 전공과 진로 방향도 "
            "구체화하고 있습니다."
        ),
        "observation_note": (
            "상담 완료 후 15주차 synthetic 후속 관찰 · 실제 학생 성과가 아님"
        ),
    }
    return pd.DataFrame([record], columns=DEMO_FOLLOWUP_COLUMNS)


def generate_synthetic_data(
    student_count: int = DEFAULT_STUDENT_COUNT, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """서로 참조 무결성이 맞는 세 가상데이터 테이블을 생성한다.

    Parameters:
        student_count: 생성할 학생 수.
        seed: 전체 데이터셋의 기준 난수 시드.

    Returns:
        students, weekly_activity, checkins DataFrame 튜플.

    Assumptions:
        활동과 체크인에는 학생 테이블에 존재하는 ID만 포함한다.
    """

    students = generate_students(student_count=student_count, seed=seed)
    weekly_activity = generate_weekly_activity(students, seed=seed + 1)
    checkins = generate_checkins(students, seed=seed + 2)
    return students, weekly_activity, checkins


def generate_support_programs() -> pd.DataFrame:
    """추천 가능한 synthetic 지원프로그램 master를 생성한다.

    Returns:
        고유 program_id와 추천 메타데이터를 가진 DataFrame.

    Assumptions:
        실제 대학 프로그램이 아니며 DEMO_UNIT 코드로 담당부서를 명시한다.
    """

    return pd.DataFrame(SUPPORT_PROGRAMS)


def _configured_course_tracks() -> tuple[
    tuple[DepartmentDefinition, list[tuple[Any, ...]]], ...
]:
    """5개 학과와 시연용 교과 트랙의 명시적 연결을 검증한다."""

    config = load_app_config()
    raw_tracks = dict(config.get("demo", {}).get("department_course_tracks", {}))
    catalog = load_department_catalog(config)
    department_codes = {department.code for department in catalog}
    if set(raw_tracks) != department_codes:
        raise ValueError(
            "demo.department_course_tracks는 설정된 5개 학과 코드를 모두 한 번씩 포함해야 합니다."
        )
    configured_tracks = []
    for department in catalog:
        track_name = str(raw_tracks[department.code]).strip()
        if track_name not in COURSE_TRACK_TEMPLATES:
            raise ValueError(
                f"알 수 없는 synthetic 교과 트랙입니다: {track_name}"
            )
        configured_tracks.append(
            (department, COURSE_TRACK_TEMPLATES[track_name])
        )
    return tuple(configured_tracks)


def generate_courses() -> pd.DataFrame:
    """설정된 5개 학과의 추천 가능한 synthetic 교과목 40개를 생성한다."""

    records: list[dict[str, Any]] = []
    for department_index, (department, courses) in enumerate(
        _configured_course_tracks(), start=1
    ):
        for course_index, course in enumerate(courses, start=1):
            name, competencies, jobs, interests, grade, difficulty, prerequisite = course
            course_id = f"C{(department_index - 1) * 8 + course_index:03d}"
            prerequisite_id = (
                f"C{(department_index - 1) * 8 + int(prerequisite):03d}"
                if prerequisite
                else ""
            )
            records.append(
                {
                    "course_id": course_id,
                    "course_name": name,
                    "department": department.name,
                    "description": f"{name}의 핵심 개념을 이해하고 실제 과제로 적용하는 synthetic 교과목",
                    "learning_objectives": f"{str(competencies).replace('|', '·')} 역량을 단계적으로 개발",
                    "competencies": competencies,
                    "related_jobs": jobs,
                    "related_interests": interests,
                    "grade_level": grade,
                    "semester": 1 if course_index % 2 else 2,
                    "difficulty": difficulty,
                    "prerequisites": prerequisite_id,
                    "credit": 3,
                    "is_available": True,
                }
            )
    return pd.DataFrame(records)


def generate_completed_courses(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    seed: int = DEFAULT_SEED + 3,
) -> pd.DataFrame:
    """학년과 소속 학과에 맞는 synthetic 이수과목 이력을 생성한다."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    completion_ranges = {1: (0, 2), 2: (3, 5), 3: (5, 7)}
    for student in students[["student_id", "department", "grade"]].to_dict(
        orient="records"
    ):
        eligible = courses[
            (courses["department"] == student["department"])
            & (courses["grade_level"] <= max(1, int(student["grade"]) - 1))
        ]
        minimum, maximum = completion_ranges[int(student["grade"])]
        count = min(len(eligible), int(rng.integers(minimum, maximum + 1)))
        selected_ids = (
            rng.choice(eligible["course_id"].to_numpy(), size=count, replace=False)
            if count
            else []
        )
        for course_id in selected_ids:
            records.append(
                {
                    "student_id": student["student_id"],
                    "course_id": str(course_id),
                    "completion_status": "completed",
                    "completion_grade": str(rng.choice(["A", "B", "C"], p=[0.35, 0.50, 0.15])),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "student_id",
            "course_id",
            "completion_status",
            "completion_grade",
        ],
    )


def generate_and_save_data(
    output_dir: str | Path | None = None,
    student_count: int = DEFAULT_STUDENT_COUNT,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, ...]:
    """가상데이터를 생성해 프로젝트 data 디렉터리에 CSV로 저장한다.

    Parameters:
        output_dir: 저장 디렉터리. 미지정 시 프로젝트 data/.
        student_count: 생성할 학생 수.
        seed: 전체 데이터셋의 기준 난수 시드.

    Returns:
        학생·활동·체크인·지원프로그램·교과목·이수내역 CSV 경로 튜플.

    Assumptions:
        같은 경로의 기존 가상 CSV는 동일한 스키마와 시드 결과로 교체한다.
    """

    target_dir = Path(output_dir) if output_dir else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    students, weekly_activity, checkins = generate_synthetic_data(
        student_count=student_count, seed=seed
    )
    support_programs = generate_support_programs()
    courses = generate_courses()
    completed_courses = generate_completed_courses(students, courses, seed + 3)
    followup_config = load_app_config()["demo"]["followup"]
    followup_path = target_dir / str(followup_config["file"])
    generate_demo_followup_observations().to_csv(
        followup_path,
        index=False,
        encoding="utf-8",
        float_format="%.2f",
    )
    paths = (
        target_dir / "students.csv",
        target_dir / "weekly_activity.csv",
        target_dir / "checkins.csv",
        target_dir / "support_programs.csv",
        target_dir / "courses.csv",
        target_dir / "completed_courses.csv",
    )
    frames = (
        students,
        weekly_activity,
        checkins,
        support_programs,
        courses,
        completed_courses,
    )
    for frame, path in zip(frames, paths):
        frame.to_csv(path, index=False, encoding="utf-8", float_format="%.2f")
    return paths


def main() -> None:
    """CLI 실행 시 기본 200명 가상데이터를 생성한다."""

    paths = generate_and_save_data()
    print("가상데이터 생성 완료")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
