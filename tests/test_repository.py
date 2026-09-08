"""데이터 접근 경계와 Repository 교체 가능성 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_generator import (
    generate_and_save_data,
    generate_completed_courses,
    generate_courses,
    generate_support_programs,
    generate_synthetic_data,
)
from src.repositories import (
    CsvRepository,
    StudentSuccessRepository,
    create_repository,
)
from src.repositories.base import StudentSuccessData, validate_repository_data
from src.risk_service import RiskAnalysisService, select_latest_student_snapshots


class InMemoryRepository(StudentSuccessRepository):
    """서비스가 CSV에 의존하지 않는지 확인하기 위한 테스트 구현체."""

    def __init__(self, data: StudentSuccessData) -> None:
        self.data = data
        self.calls = {
            "students": 0,
            "activity": 0,
            "checkins": 0,
            "programs": 0,
            "courses": 0,
            "completed": 0,
        }

    def get_students(self) -> pd.DataFrame:
        self.calls["students"] += 1
        return self.data.students.copy()

    def get_weekly_activity(self) -> pd.DataFrame:
        self.calls["activity"] += 1
        return self.data.weekly_activity.copy()

    def get_checkins(self) -> pd.DataFrame:
        self.calls["checkins"] += 1
        return self.data.checkins.copy()

    def get_support_programs(self) -> pd.DataFrame:
        self.calls["programs"] += 1
        return self.data.support_programs.copy()

    def get_courses(self) -> pd.DataFrame:
        self.calls["courses"] += 1
        return self.data.courses.copy()

    def get_completed_courses(self) -> pd.DataFrame:
        self.calls["completed"] += 1
        return self.data.completed_courses.copy()


def test_csv_repository_loads_canonical_data(tmp_path: Path) -> None:
    expected = generate_synthetic_data(student_count=20)
    generate_and_save_data(output_dir=tmp_path, student_count=20)

    actual = CsvRepository(data_dir=tmp_path, auto_generate=False).get_all()

    for expected_frame, actual_frame in zip(expected, actual.as_tuple()):
        pd.testing.assert_frame_equal(expected_frame, actual_frame)


def test_csv_repository_prefers_validated_wings_program_master(
    tmp_path: Path,
) -> None:
    generate_and_save_data(output_dir=tmp_path, student_count=20)
    wings_programs = generate_support_programs().head(3).copy()
    wings_programs["program_id"] = ["WINGS-A", "WINGS-B", "WINGS-C"]
    wings_programs["application_url"] = "https://wings.kbu.ac.kr"
    wings_programs["source_system"] = "WINGS"
    wings_programs.to_csv(
        tmp_path / "support_programs_wings.csv", index=False
    )

    loaded = CsvRepository(
        data_dir=tmp_path, auto_generate=False
    ).get_support_programs()

    assert loaded["program_id"].tolist() == ["WINGS-A", "WINGS-B", "WINGS-C"]
    assert set(loaded["source_system"]) == {"WINGS"}


def test_latest_snapshot_is_selected_per_student_with_mixed_weeks() -> None:
    snapshots = pd.DataFrame(
        [
            {"student_id": "S0001", "week": 3, "overall_risk": 30},
            {"student_id": "S0001", "week": 4, "overall_risk": 40},
            {"student_id": "S0002", "week": 4, "overall_risk": 50},
            {"student_id": "S0002", "week": 5, "overall_risk": 45},
        ]
    )

    latest = select_latest_student_snapshots(snapshots).set_index("student_id")

    assert latest.loc["S0001", "week"] == 4
    assert latest.loc["S0002", "week"] == 5
    assert latest.loc["S0001", "overall_risk"] == 40
    assert latest.loc["S0002", "overall_risk"] == 45


def test_csv_repository_generates_missing_synthetic_data(tmp_path: Path) -> None:
    repository = CsvRepository(data_dir=tmp_path, auto_generate=True)

    data = repository.get_all()

    assert len(data.students) == 200
    assert len(data.weekly_activity) == 800
    assert len(data.checkins) == 200
    assert len(data.courses) == 40
    assert data.courses["course_id"].nunique() == 40
    assert not data.completed_courses.empty
    assert (tmp_path / "students.csv").exists()
    followup_path = tmp_path / "demo_followup_observations.csv"
    assert followup_path.exists()
    followup = pd.read_csv(followup_path, keep_default_na=False)
    assert followup[["student_id", "week"]].to_dict(orient="records") == [
        {"student_id": "S0003", "week": 15}
    ]


def test_csv_repository_reports_missing_files_when_generation_disabled(
    tmp_path: Path,
) -> None:
    repository = CsvRepository(data_dir=tmp_path, auto_generate=False)

    with pytest.raises(FileNotFoundError, match="필수 CSV 파일"):
        repository.get_students()


def test_risk_service_depends_only_on_repository_contract() -> None:
    students, activity, checkins = generate_synthetic_data(student_count=20)
    courses = generate_courses()
    repository = InMemoryRepository(
        StudentSuccessData(
            students,
            activity,
            checkins,
            generate_support_programs(),
            courses,
            generate_completed_courses(students, courses),
        )
    )

    loaded_students, snapshots = RiskAnalysisService(
        repository
    ).load_students_and_snapshots()

    assert len(loaded_students) == 20
    assert len(snapshots) == 80
    assert repository.calls == {
        "students": 1,
        "activity": 1,
        "checkins": 1,
        "programs": 1,
        "courses": 1,
        "completed": 1,
    }


def test_repository_validation_rejects_unknown_student_reference() -> None:
    students, activity, checkins = generate_synthetic_data(student_count=20)
    courses = generate_courses()
    invalid_activity = activity.copy()
    invalid_activity.loc[0, "student_id"] = "UNKNOWN"

    with pytest.raises(ValueError, match="존재하지 않는 student_id"):
        validate_repository_data(
            StudentSuccessData(
                students,
                invalid_activity,
                checkins,
                generate_support_programs(),
                courses,
                generate_completed_courses(students, courses),
            )
        )


def test_repository_factory_registers_only_csv_for_now() -> None:
    assert isinstance(create_repository("csv"), CsvRepository)
    with pytest.raises(ValueError, match="현재 사용 가능한 provider: csv"):
        create_repository("oracle")


def test_pages_and_risk_engine_do_not_read_csv_directly() -> None:
    project_root = Path(__file__).resolve().parents[1]
    protected_files = [
        project_root / "pages" / "01_dashboard.py",
        project_root / "pages" / "02_student_detail.py",
        project_root / "pages" / "03_ai_intervention.py",
        project_root / "pages" / "04_learning_path.py",
        project_root / "pages" / "07_student_view.py",
        project_root / "src" / "risk_engine.py",
    ]

    for path in protected_files:
        source = path.read_text(encoding="utf-8")
        assert "read_csv" not in source
        assert "load_demo_data" not in source
