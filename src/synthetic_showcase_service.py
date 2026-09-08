"""200명 synthetic 데이터에서 시연용 대표 학생 사례를 선정한다."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.risk_service import select_latest_student_snapshots


@dataclass(frozen=True)
class SyntheticShowcaseDefinition:
    """시연에서 보여줄 synthetic 학생 유형과 관찰 포인트."""

    key: str
    title: str
    archetype: str
    expected_primary_risk_type: str
    story: str
    review_focus: str


@dataclass(frozen=True)
class SyntheticShowcase:
    """실제 synthetic 행과 시연 설명을 결합한 표시용 사례."""

    key: str
    title: str
    student_id: str
    archetype: str
    department: str
    grade: int
    week: int
    risk_level: str
    primary_risk_type: str
    overall_risk: float
    risk_change: float
    risk_history: tuple[tuple[int, float], ...]
    story: str
    review_focus: str
    natural_language_concern: str

    @property
    def option_label(self) -> str:
        """선택 위젯에서 학생 ID와 사례 제목을 함께 보여준다."""

        return f"{self.student_id} · {self.title}"

    @property
    def period_risk_change(self) -> float:
        """최초 주차 대비 최신 종합 위험도 변화를 반환한다."""

        if not self.risk_history:
            return 0.0
        return round(self.risk_history[-1][1] - self.risk_history[0][1], 2)

    @property
    def risk_history_text(self) -> str:
        """위험도 주차 흐름을 짧은 한 줄 표현으로 반환한다."""

        return " → ".join(
            f"{week}주 {score:.1f}" for week, score in self.risk_history
        )


SHOWCASE_DEFINITIONS: tuple[SyntheticShowcaseDefinition, ...] = (
    SyntheticShowcaseDefinition(
        "stable",
        "안정적인 학업 유지",
        "normal",
        "",
        "출결·학습·전공·진로 신호가 안정적인 정상 보존 사례입니다.",
        "불필요한 개입을 추천하지 않고 안정 상태를 유지하는지 확인합니다.",
    ),
    SyntheticShowcaseDefinition(
        "attendance",
        "출결 흐름 회복",
        "attendance_risk",
        "출결위험형",
        "학습 의지는 유지되지만 결석과 연속결석 신호가 두드러진 사례입니다.",
        "출결 근거를 우선 확인하고 주간 회복 계획으로 연결합니다.",
    ),
    SyntheticShowcaseDefinition(
        "engagement",
        "학습 참여 리듬 회복",
        "engagement_risk",
        "학습참여저하형",
        "출석보다 LMS·과제·강의영상 참여 감소가 더 크게 보이는 사례입니다.",
        "학습 리듬 저하를 출결 문제로 단정하지 않는지 확인합니다.",
    ),
    SyntheticShowcaseDefinition(
        "achievement",
        "학업성취 기초 보완",
        "achievement_risk",
        "학업부진형",
        "참여는 이어가지만 이전 성적·재수강·퀴즈에서 보완 필요가 보이는 사례입니다.",
        "학습 의지와 성취 결과를 구분하고 기초학습 지원을 검토합니다.",
    ),
    SyntheticShowcaseDefinition(
        "major",
        "전공 적응 재탐색",
        "major_mismatch",
        "전공부적응형",
        "성적만으로는 드러나지 않지만 전공 흥미·만족·지속 의사가 낮은 사례입니다.",
        "전공 부적응을 학업 능력 문제로 오해하지 않고 관심 분야를 함께 보아야 합니다.",
    ),
    SyntheticShowcaseDefinition(
        "career",
        "진로 방향 탐색",
        "career_unclear",
        "진로미설정형",
        "학업 신호는 비교적 안정적이지만 진로 명확성과 희망직무가 비어 있는 사례입니다.",
        "위험도만 높이지 않고 진로 탐색과 역량 학습경로를 연결합니다.",
    ),
    SyntheticShowcaseDefinition(
        "complex_main",
        "복합위험 대표 지원",
        "complex_risk",
        "",
        "출결·학습·전공·진로 신호가 겹쳐 지원 우선순위 설계가 필요한 주 시연 사례입니다.",
        "학생의 말을 먼저 확인하고 DB 지원·학습경로를 교직원이 검토하는 전체 흐름을 보여줍니다.",
    ),
    SyntheticShowcaseDefinition(
        "complex_severe",
        "복합위험 우선 사례",
        "complex_risk",
        "",
        "여러 영역의 신호가 함께 높아 즉시 교직원 검토가 필요한 교차 사례입니다.",
        "한 가지 프로그램으로 단순화하지 않고 단계적 지원 계획을 세웁니다.",
    ),
)


def _pick_showcase_row(
    candidates: pd.DataFrame,
    definition: SyntheticShowcaseDefinition,
    selected_student_ids: set[str],
    representative_student_id: str,
) -> pd.Series:
    """정의에 맞는 한 학생을 고정 기준으로 선정한다."""

    available = candidates[
        ~candidates["student_id"].astype(str).isin(selected_student_ids)
    ].copy()
    if definition.key == "complex_main":
        representative = available[
            available["student_id"].astype(str) == representative_student_id
        ]
        if not representative.empty:
            return representative.iloc[0]
    if definition.expected_primary_risk_type:
        matched = available[
            available["primary_risk_type"]
            == definition.expected_primary_risk_type
        ]
        if not matched.empty:
            available = matched
    if available.empty:
        raise ValueError(
            f"{definition.title} 시연에 사용할 synthetic 학생이 없습니다."
        )
    ascending = definition.key == "stable"
    return available.sort_values(
        ["overall_risk", "risk_change", "student_id"],
        ascending=[ascending, ascending, True],
        kind="stable",
    ).iloc[0]


def build_synthetic_showcases(
    students: pd.DataFrame,
    snapshots: pd.DataFrame,
    representative_student_id: str = "S0003",
) -> tuple[SyntheticShowcase, ...]:
    """전체 synthetic 학생에서 서로 다른 8개 시연 사례를 만든다.

    Parameters:
        students: archetype·학과·학년이 포함된 synthetic 학생 데이터.
        snapshots: 학생·주차별 위험 스냅샷.
        representative_student_id: 전체 흐름에 사용할 복합위험 대표 ID.

    Returns:
        정상·5개 단일위험·2개 복합위험 사례.

    Assumptions:
        표시는 학생별 최신 주차를 사용하며 실제 개인정보를 포함하지 않는다.
    """

    required_student_columns = {"student_id", "archetype", "department", "grade"}
    missing_students = required_student_columns.difference(students.columns)
    if missing_students:
        raise ValueError(
            "대표 가상학생 선정에 필요한 students 컬럼이 없습니다: "
            f"{sorted(missing_students)}"
        )
    required_snapshot_columns = {
        "student_id",
        "week",
        "risk_level",
        "primary_risk_type",
        "overall_risk",
        "risk_change",
        "natural_language_concern",
    }
    missing_snapshots = required_snapshot_columns.difference(snapshots.columns)
    if missing_snapshots:
        raise ValueError(
            "대표 가상학생 선정에 필요한 snapshots 컬럼이 없습니다: "
            f"{sorted(missing_snapshots)}"
        )

    latest = select_latest_student_snapshots(snapshots)
    student_frame = students.copy()
    student_frame["student_id"] = student_frame["student_id"].astype(str)
    latest["student_id"] = latest["student_id"].astype(str)
    duplicated_static_columns = [
        column
        for column in ("archetype", "department", "grade")
        if column in latest.columns
    ]
    combined = student_frame[
        ["student_id", "archetype", "department", "grade"]
    ].merge(
        latest.drop(columns=duplicated_static_columns),
        on="student_id",
        how="inner",
    )
    selected_student_ids: set[str] = set()
    showcases: list[SyntheticShowcase] = []
    representative_id = str(representative_student_id).strip()

    for definition in SHOWCASE_DEFINITIONS:
        candidates = combined[
            combined["archetype"].astype(str) == definition.archetype
        ]
        row = _pick_showcase_row(
            candidates,
            definition,
            selected_student_ids,
            representative_id,
        )
        student_id = str(row["student_id"])
        selected_student_ids.add(student_id)
        history_frame = snapshots[
            snapshots["student_id"].astype(str) == student_id
        ].sort_values("week", kind="stable")
        risk_history = tuple(
            (int(history["week"]), float(history["overall_risk"]))
            for history in history_frame.to_dict(orient="records")
        )
        showcases.append(
            SyntheticShowcase(
                key=definition.key,
                title=definition.title,
                student_id=student_id,
                archetype=definition.archetype,
                department=str(row["department"]),
                grade=int(row["grade"]),
                week=int(row["week"]),
                risk_level=str(row["risk_level"]),
                primary_risk_type=str(row["primary_risk_type"]),
                overall_risk=float(row["overall_risk"]),
                risk_change=float(row["risk_change"]),
                risk_history=risk_history,
                story=definition.story,
                review_focus=definition.review_focus,
                natural_language_concern=str(row["natural_language_concern"]),
            )
        )

    return tuple(showcases)
