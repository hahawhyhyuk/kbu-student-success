"""발표 전 시연 준비·SQLite 백업·초기화 테스트."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ai import MockAIProvider
from src.database import (
    DEMO_WORKFLOW_TABLES,
    DemoDataStore,
    LearningPathDataStore,
    StudentCheckinDataStore,
)
from src.demo_service import DemoScenarioService
from src.intervention_recommender import ProgramRecommendation
from src.intervention_service import InterventionManagementService
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService, select_latest_student_snapshots
from src.utils import PROJECT_ROOT


CHECKIN_VALUES = {
    "week": 4,
    "major_interest": 2,
    "major_satisfaction": 2,
    "major_continuation_intent": 2,
    "learning_difficulty": 4,
    "career_clarity": 2,
    "consultation_intent": 5,
    "interest_fields": ["데이터·AI"],
    "desired_job": "서비스 기획",
    "natural_language_concern": "시연용 학생 고민입니다.",
}


def _demo_config(enabled: bool = True) -> dict[str, object]:
    """초기화 안전장치 테스트용 시연 설정을 만든다."""

    return {
        "demo": {
            "enabled": enabled,
            "representative_student_id": "S0003",
            "baseline_week": 4,
            "observation_weeks": [1, 2, 3, 4],
            "reset_confirmation_text": "시연 기록 초기화",
        }
    }


def _recommendation() -> tuple[ProgramRecommendation, ...]:
    """시연 개입 테스트용 지원프로그램 추천을 만든다."""

    return (
        ProgramRecommendation(
            program_id="P001",
            program_name="출석회복 코칭",
            program_type="학업상담",
            description="출석 계획을 함께 수립합니다.",
            department_in_charge="DEMO_UNIT01",
            operation_period="학기 중",
            score=90.0,
            reason="출결 위험을 우선 지원합니다.",
        ),
    )


def _risk_snapshot() -> dict[str, object]:
    """시연 개입에 연결할 고위험 스냅샷을 만든다."""

    return {
        "student_id": "S0003",
        "week": 4,
        "attendance_risk": 100,
        "engagement_risk": 60,
        "achievement_risk": 51.5,
        "major_adaptation_risk": 75,
        "career_risk": 70,
        "overall_risk": 71.88,
        "risk_level": "고위험",
        "primary_risk_type": "출결위험형",
        "secondary_risk_types": "진로미설정형, 전공부적응형",
    }


def _seed_demo_records(database_path: Path) -> None:
    """초기화 대상 테이블에 서로 연결된 시연 기록을 생성한다."""

    checkins = StudentCheckinDataStore(database_path)
    checkin_id = checkins.save_checkin("S0003", CHECKIN_VALUES)
    analysis = MockAIProvider().analyze_checkin(
        CHECKIN_VALUES["natural_language_concern"]
    )
    checkins.save_analysis(checkin_id, analysis, "mock", True)
    checkins.save_analysis_feedback(checkin_id, "S0003", "맞아요")

    interventions = InterventionManagementService(database_path=database_path)
    context = interventions.record_recommendation_batch(
        "S0003", _recommendation(), risk_snapshot=_risk_snapshot()
    )
    interventions.record_feedback(
        "S0003",
        context.recommendation_ids["P001"],
        "관심 있음",
    )

    LearningPathDataStore(database_path).save_analysis(
        paths=[
            {
                "student_id": "S0003",
                "path_name": "서비스 기획 기초",
                "related_job": "서비스 기획",
                "competencies": ["기획"],
                "reason": "시연용 학습경로",
                "courses": [
                    {
                        "course_id": "C001",
                        "sequence": 1,
                        "score": 80,
                        "reason": "기초 역량",
                    }
                ],
            }
        ],
        candidates=[
            {
                "candidate_code": "DEMO-01",
                "candidate_name": "서비스 기획 후보",
                "student_count": 5,
                "department_count": 2,
                "departments": ["DEPT01", "DEPT02"],
                "course_ids": ["C001"],
                "course_names": ["서비스 기획"],
                "competencies": ["기획"],
                "related_jobs": ["서비스 기획"],
                "student_ids": ["S0003", "S0004", "S0005", "S0006", "S0007"],
                "average_similarity": 0.7,
                "description": "교직원 검토용 후보",
                "status": "검토 후보",
            }
        ],
        analysis_batch_id="demo-batch",
    )


def test_wrong_confirmation_does_not_reset_records(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    _seed_demo_records(database_path)
    service = DemoScenarioService(
        config=_demo_config(),
        store=DemoDataStore(database_path),
    )
    before = service.get_record_counts()

    with pytest.raises(ValueError, match="확인 문구"):
        service.reset_demo_records("초기화")

    assert service.get_record_counts() == before
    assert not (tmp_path / "backups").exists()


def test_reset_creates_recoverable_backup_and_clears_only_workflow_records(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    _seed_demo_records(database_path)
    store = DemoDataStore(database_path)
    service = DemoScenarioService(config=_demo_config(), store=store)
    before = service.get_record_counts()

    result = service.reset_demo_records(
        "시연 기록 초기화",
        backup_dir=tmp_path / "backups",
    )

    assert result.backup_path.exists()
    assert result.deleted_total == sum(before.values())
    assert all(count == 0 for count in service.get_record_counts().values())
    with sqlite3.connect(result.backup_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM interventions"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM student_checkins"
        ).fetchone()[0] == 1
    with sqlite3.connect(database_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert set(DEMO_WORKFLOW_TABLES).issubset(table_names)


def test_reset_can_be_disabled_for_non_demo_environment(tmp_path: Path) -> None:
    service = DemoScenarioService(
        config=_demo_config(enabled=False),
        store=DemoDataStore(tmp_path / "app.db"),
    )

    with pytest.raises(ValueError, match="비활성화"):
        service.reset_demo_records("시연 기록 초기화")


def test_demo_week_filter_replays_without_changing_source(tmp_path: Path) -> None:
    """선택 주차까지만 보이되 원본 스냅샷과 미래 주차는 삭제하지 않는다."""

    repository = get_default_repository()
    snapshots = RiskAnalysisService(repository).build_snapshots()
    original = snapshots.copy(deep=True)
    service = DemoScenarioService(
        config=_demo_config(),
        store=DemoDataStore(tmp_path / "week-filter.db"),
    )

    assert service.observation_week_labels == (
        "1주차",
        "2주차",
        "3주차",
        "4주차",
    )
    assert service.resolve_observation_week("3주차") == 3
    through_week_three = service.filter_snapshots_through_week(snapshots, 3)

    assert through_week_three["week"].max() == 3
    assert len(snapshots) == len(original)
    assert snapshots.equals(original)
    assert service.representative_risk_path_is_ready(snapshots)

    week_two_latest = select_latest_student_snapshots(
        service.filter_snapshots_through_week(snapshots, 2)
    )
    week_two_candidates = service.find_alert_candidates(week_two_latest)
    assert len(week_two_candidates) == 4
    assert week_two_candidates["risk_level"].eq("고위험").all()
    assert week_two_candidates["week"].eq(2).all()

    week_four_candidates = service.find_alert_candidates(
        select_latest_student_snapshots(snapshots)
    )
    assert week_four_candidates.iloc[0]["student_id"] == "S0003"


def test_representative_student_data_becomes_high_risk_in_week_four() -> None:
    """S0003의 4주차 고위험은 고정 규칙이 아니라 synthetic 데이터 결과다."""

    repository = get_default_repository()
    snapshots = RiskAnalysisService(repository).build_snapshots()
    representative = snapshots[
        snapshots["student_id"].astype(str) == "S0003"
    ].sort_values("week")

    assert representative["risk_level"].tolist() == [
        "관심",
        "관심",
        "주의",
        "고위험",
    ]
    assert representative.iloc[-1]["primary_risk_type"] == "출결위험형"
    assert representative["overall_risk"].is_monotonic_increasing


def test_dashboard_week_control_uses_actual_risk_at_each_week(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """교직원 대시보드에서 조회 기준 주차를 바꾸면 당시 현황이 표시된다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH",
        tmp_path / "dashboard-week.db",
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "01_dashboard.py")
    ).run(timeout=30)

    assert not page.exception
    week_control = next(
        selectbox
        for selectbox in page.selectbox
        if selectbox.label == "조회 기준 주차"
    )
    assert week_control.options == ["1주차", "2주차", "3주차", "4주차"]
    assert week_control.value == "4주차"
    assert any(
        "4주차 조회 기준 조기경보 대상 14명" in str(item.value)
        for item in page.warning
    )
    assert any(
        item.label.startswith("학생 안내 보내기 · S0003")
        for item in page.expander
    )
    assert any(
        "실제 이메일 주소는 사용하지 않습니다" in str(item.value)
        for item in page.caption
    )

    week_control.set_value("2주차").run(timeout=30)
    assert not page.exception
    assert any(
        "2주차 조회 기준 조기경보 대상 4명" in str(item.value)
        for item in page.warning
    )
    assert not any(
        item.label.startswith("학생 안내 보내기 · ")
        for item in page.expander
    )
    assert any(
        "과거 주차는 조회 전용" in str(item.value)
        for item in page.caption
    )

    week_control = next(
        selectbox
        for selectbox in page.selectbox
        if selectbox.label == "조회 기준 주차"
    )
    week_control.set_value("3주차").run(timeout=30)
    assert not page.exception
    assert any(
        "3주차 조회 기준 조기경보 대상 13명" in str(item.value)
        for item in page.warning
    )
    assert any(
        "3주차" in str(item.value)
        and "주의" in str(item.value)
        for item in page.info
    )
    snapshots = RiskAnalysisService(get_default_repository()).build_snapshots()
    latest_week_three = select_latest_student_snapshots(
        snapshots[snapshots["week"] <= 3]
    )
    assert latest_week_three["week"].eq(3).all()


def test_demo_preparation_is_separate_from_normal_student_checkin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)

    preparation = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "09_demo_scenario.py")
    ).run(timeout=30)
    assert not preparation.exception
    assert any(
        "발표 전 준비 상태" in item.value
        for item in preparation.subheader
    )
    expected_statuses = {
        "synthetic 기준 데이터",
        "SQLite 시연 기록",
        "Gemini 연결 설정",
        "학생 시연 계정",
    }
    assert expected_statuses.issubset(
        {metric.label for metric in preparation.metric}
    )
    rendered = " ".join(item.value for item in preparation.markdown)
    assert "STEP 01" not in rendered
    assert "5주차 후속 체크인" not in rendered
    assert any(
        button.label == "전체 대시보드 열기"
        for button in preparation.button
    )
    assert any(
        expander.label == "시연 기록 초기화"
        for expander in preparation.expander
    )
    assert any(
        button.label == "백업 후 시연 기록 초기화"
        for button in preparation.button
    )
    assert not any(
        item.value == "실제 master 전환 사전 점검"
        for item in preparation.subheader
    )
    advanced_toggle = next(
        toggle
        for toggle in preparation.toggle
        if toggle.label == "고급 준비 도구"
    )
    advanced_toggle.set_value(True).run(timeout=30)
    assert not preparation.exception
    showcase_picker = next(
        selectbox
        for selectbox in preparation.selectbox
        if selectbox.label == "점검할 synthetic 사례"
    )
    assert len(showcase_picker.options) == 8
    assert any(option.startswith("S0003 ·") for option in showcase_picker.options)
    assert not any(
        button.label == "대표 학생 S0003 브라우저 선택 준비"
        for button in preparation.button
    )
    assert any(
        item.value == "실제 master 전환 사전 점검"
        for item in preparation.subheader
    )
    master_button = next(
        button
        for button in preparation.button
        if button.label == "세 파일 안전 점검 실행"
    )
    assert master_button.disabled
    assert any(
        item.value == "2026 전 학과 개설강좌 연결 점검"
        for item in preparation.subheader
    )
    assert any(
        button.label == "전 학과 개설강좌 연결 dry-run 실행"
        for button in preparation.button
    )
    preparation_captions = " ".join(
        str(item.value) for item in preparation.caption
    )
    assert "설명 보강 파일" in preparation_captions
    preparation_text = preparation_captions
    assert "담당자명·사번·학번·전화·이메일" in preparation_text
    assert "CSV·SQLite·설정 파일" in preparation_text

    student_page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    )
    student_page.session_state["demo_student_id"] = "S0003"
    # 기존 브라우저에 남은 예전 플래그도 자동답을 만들지 않아야 한다.
    student_page.session_state["demo_initial_preset_enabled"] = True
    student_page.session_state["demo_followup_preset_enabled"] = True
    student_page.run(timeout=30)
    assert not student_page.exception
    button_labels = {button.label for button in student_page.button}
    assert "시연 자연 대화 시작" not in button_labels
    assert "다음 학생 답변 보내기" not in button_labels
    assert "준비된 5주차 후속 체크인 확인" not in button_labels
    draft = student_page.session_state["student_checkin_S0003_draft"]
    assert draft["major_interest"] is None
    assert draft["learning_difficulty"] is None

    consent = next(
        checkbox
        for checkbox in student_page.checkbox
        if "Gemini API로 전송" in checkbox.label
    )
    consent.set_value(True).run(timeout=30)
    assert not student_page.exception
    assert {
        "요즘 수업이나 과제가 조금 버거워요",
        "전공이 나와 맞는지 고민이에요",
        "진로를 아직 정하지 못했어요",
    }.issubset({button.label for button in student_page.button})

    next(
        button
        for button in student_page.button
        if button.label == "외부 AI 전송 없이 직접 입력"
    ).click().run(timeout=30)
    assert not student_page.exception
    assert any(
        "외부 AI 전송 없는 기본 체크인" in item.value
        for item in student_page.subheader
    )
