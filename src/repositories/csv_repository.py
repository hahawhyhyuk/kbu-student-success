"""현재 synthetic CSV 데이터용 Repository 구현."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_generator import DATA_DIR, generate_and_save_data
from src.department_catalog import validate_configured_departments
from src.repositories.base import StudentSuccessRepository
from src.utils import load_app_config


class CsvRepository(StudentSuccessRepository):
    """synthetic CSV를 canonical DataFrame으로 제공한다.

    Parameters:
        data_dir: CSV 파일 디렉터리. 미지정 시 프로젝트 data/.
        auto_generate: 파일이 없을 때 고정 시드 synthetic 데이터를 만들지 여부.

    Assumptions:
        빈 문자열은 희망직무 미설정 의미이므로 pandas NaN으로 변환하지 않는다.
    """

    FILE_NAMES = {
        "students": "students.csv",
        "weekly_activity": "weekly_activity.csv",
        "checkins": "checkins.csv",
        "support_programs": "support_programs.csv",
        "courses": "courses.csv",
        "completed_courses": "completed_courses.csv",
    }

    def __init__(
        self,
        data_dir: str | Path | None = None,
        auto_generate: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.auto_generate = auto_generate

    def ensure_available(self) -> Path:
        """필수 CSV 존재를 보장하고 데이터 디렉터리를 반환한다."""

        missing = [
            self.data_dir / file_name
            for file_name in self.FILE_NAMES.values()
            if not (self.data_dir / file_name).exists()
        ]
        if missing and self.auto_generate:
            generate_and_save_data(output_dir=self.data_dir)
            missing = []
        if missing:
            missing_text = ", ".join(path.name for path in missing)
            raise FileNotFoundError(f"필수 CSV 파일이 없습니다: {missing_text}")
        return self.data_dir

    def _read(self, data_name: str) -> pd.DataFrame:
        """CSV 한 개를 공통 옵션으로 읽는다."""

        directory = self.ensure_available()
        frame = pd.read_csv(
            directory / self.FILE_NAMES[data_name], keep_default_na=False
        )
        if data_name == "students":
            validate_configured_departments(
                frame["department"],
                source_name=self.FILE_NAMES[data_name],
            )
        return frame

    def get_students(self) -> pd.DataFrame:
        """학생 정적정보 CSV를 반환한다."""

        return self._read("students")

    def get_weekly_activity(self) -> pd.DataFrame:
        """주차 활동 CSV를 반환한다."""

        return self._read("weekly_activity")

    def get_checkins(self) -> pd.DataFrame:
        """체크인 CSV를 반환한다."""

        return self._read("checkins")

    def get_support_programs(self) -> pd.DataFrame:
        """유효한 WINGS master를 우선하고 오류·누락 시 synthetic으로 복귀한다."""

        from src.master_activation_service import (
            configured_fallback_file_name,
            configured_master_file_name,
            validate_program_activation_data,
        )

        directory = self.ensure_available()
        config = load_app_config()
        master_path = directory / configured_master_file_name(
            "programs", config
        )
        fallback_path = directory / configured_fallback_file_name(
            "programs", config
        )
        if master_path.exists():
            try:
                programs = pd.read_csv(master_path, keep_default_na=False)
                validate_program_activation_data(programs)
                return programs
            except Exception:
                pass
        if not fallback_path.exists():
            raise FileNotFoundError(
                "유효한 비교과 master가 없고 synthetic fallback CSV도 없습니다: "
                f"{fallback_path.name}"
            )
        programs = pd.read_csv(fallback_path, keep_default_na=False)
        validate_program_activation_data(programs)
        return programs

    def get_courses(self) -> pd.DataFrame:
        """검토 완료 실제 교과목을 우선하고 오류 시 synthetic으로 복귀한다."""

        from src.master_activation_service import (
            configured_fallback_file_name,
            configured_master_file_name,
            validate_course_activation_data,
            validate_department_activation_data,
        )

        directory = self.ensure_available()
        config = load_app_config()
        master_path = directory / configured_master_file_name("courses", config)
        fallback_path = directory / configured_fallback_file_name(
            "courses", config
        )
        department_path = directory / configured_master_file_name(
            "departments", config
        )
        if master_path.exists() and department_path.exists():
            try:
                departments = pd.read_csv(
                    department_path, keep_default_na=False
                )
                validate_department_activation_data(departments)
                courses = pd.read_csv(master_path, keep_default_na=False)
                minimum = int(
                    config.get("course_recommendation", {}).get(
                        "minimum_courses", 3
                    )
                )
                validate_course_activation_data(
                    courses,
                    departments["department_name"],
                    minimum_courses=minimum,
                )
                return courses
            except Exception:
                pass
        if not fallback_path.exists():
            raise FileNotFoundError(
                "유효한 실제 교과목 master가 없고 synthetic fallback CSV도 없습니다: "
                f"{fallback_path.name}"
            )
        courses = pd.read_csv(fallback_path, keep_default_na=False)
        validate_configured_departments(
            courses["department"], source_name=fallback_path.name
        )
        return courses

    def get_completed_courses(self) -> pd.DataFrame:
        """현재 교과 master에 존재하는 학생별 이수과목만 반환한다."""

        completed = self._read("completed_courses")
        active_course_ids = set(self.get_courses()["course_id"].astype(str))
        return completed[
            completed["course_id"].astype(str).isin(active_course_ids)
        ].reset_index(drop=True)
