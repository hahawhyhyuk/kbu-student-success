"""SQLite 개입 상태와 학생 추천 피드백 영속화 테스트."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.database import (
    InterventionDataStore,
    LearningPathDataStore,
    StudentCheckinDataStore,
    initialize_database,
)
from src.intervention_recommender import ProgramRecommendation
from src.intervention_service import InterventionManagementService
from src.utils import PROJECT_ROOT


def _recommendations() -> tuple[ProgramRecommendation, ...]:
    """저장 계층 테스트용 DB 프로그램 추천 세 건을 만든다."""

    return tuple(
        ProgramRecommendation(
            program_id=f"PRG{index:03d}",
            program_name=f"지원 프로그램 {index}",
            program_type="학습지원",
            description="테스트용 synthetic 지원 프로그램",
            department_in_charge="DEMO_UNIT",
            operation_period="상시",
            score=float(90 - index),
            reason=f"테스트 추천 이유 {index}.",
        )
        for index in range(1, 4)
    )


def _risk_snapshot(**overrides: object) -> dict[str, object]:
    """개입 생성 시점 저장 테스트용 위험엔진 결과를 만든다."""

    snapshot: dict[str, object] = {
        "student_id": "STU001",
        "week": 4,
        "attendance_risk": 72.0,
        "engagement_risk": 61.0,
        "achievement_risk": 48.0,
        "major_adaptation_risk": 67.0,
        "career_risk": 55.0,
        "overall_risk": 62.8,
        "risk_level": "주의",
        "primary_risk_type": "출결위험형",
        "secondary_risk_types": "전공부적응형, 학습참여저하형",
        "checkin_source": "repository",
    }
    snapshot.update(overrides)
    return snapshot


def _checkin_values() -> dict[str, object]:
    """checkin_id 연결 테스트에 필요한 최소 학생 응답을 반환한다."""

    return {
        "week": 4,
        "major_interest": 2,
        "major_satisfaction": 2,
        "major_continuation_intent": 2,
        "learning_difficulty": 4,
        "career_clarity": 2,
        "consultation_intent": 5,
        "interest_fields": ["데이터·AI"],
        "desired_job": "데이터 분석",
        "natural_language_concern": "학습과 진로 상담이 필요합니다.",
    }


def test_existing_database_is_migrated_without_removing_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE interventions (
                intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT '추천 생성'
            )
            """
        )
        connection.execute(
            "INSERT INTO interventions (student_id) VALUES ('STU001')"
        )
        connection.commit()

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(interventions)"
            ).fetchall()
        }
        row_count = connection.execute(
            "SELECT COUNT(*) FROM interventions"
        ).fetchone()[0]
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "recommendation_batch_id" in columns
    assert row_count == 1
    assert {
        "recommendations",
        "risk_snapshots",
        "interventions",
        "feedback",
        "support_plans",
        "support_plan_items",
    }.issubset(
        table_names
    )


def test_legacy_learning_path_is_migrated_to_versioned_history(
    tmp_path: Path,
) -> None:
    """기존 체크인별 경로를 v1로 보존하고 v2 저장을 허용한다."""

    database_path = tmp_path / "legacy-learning-path.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE learning_paths (
                path_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_batch_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                source_checkin_id INTEGER,
                path_name TEXT NOT NULL,
                related_job TEXT,
                competencies TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (analysis_batch_id, student_id)
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_learning_paths_checkin "
            "ON learning_paths(source_checkin_id) "
            "WHERE source_checkin_id IS NOT NULL"
        )
        connection.execute(
            """
            INSERT INTO learning_paths (
                analysis_batch_id, student_id, source_checkin_id,
                path_name, competencies, reason
            ) VALUES ('legacy-v1', 'STU001', 7, '기존 경로', '[]', '기존 근거')
            """
        )
        connection.commit()

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        legacy = connection.execute(
            "SELECT generation_version FROM learning_paths WHERE path_id = 1"
        ).fetchone()
        index_names = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(learning_paths)"
            ).fetchall()
        }
        connection.execute(
            """
            INSERT INTO learning_paths (
                analysis_batch_id, student_id, source_checkin_id,
                generation_version, path_name, competencies, reason
            ) VALUES ('new-v2', 'STU001', 7, 2, '새 경로', '[]', '새 근거')
            """
        )
        version_count = connection.execute(
            "SELECT COUNT(*) FROM learning_paths WHERE source_checkin_id = 7"
        ).fetchone()[0]

    assert legacy == (1,)
    assert "idx_learning_paths_checkin" not in index_names
    assert "idx_learning_paths_checkin_version" in index_names
    assert version_count == 2


def test_recommendation_batch_and_review_state_survive_new_service(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    service = InterventionManagementService(database_path=database_path)

    context = service.record_recommendation_batch("STU001", _recommendations())
    service.apply_review_action(context.intervention_id, "승인")

    restarted_service = InterventionManagementService(database_path=database_path)
    stored = restarted_service.get_intervention(context.intervention_id)
    saved_recommendations = restarted_service.list_recommendations(
        context.batch_id
    )

    assert stored is not None
    assert stored["student_id"] == "STU001"
    assert stored["status"] == "지원계획 작성"
    assert stored["staff_action"] == "추천 검토 승인"
    assert len(context.recommendation_ids) == 3
    assert saved_recommendations["target_id"].tolist() == [
        "PRG001",
        "PRG002",
        "PRG003",
    ]
    assert context.risk_snapshot_id is None


def test_integrated_review_updates_program_and_saved_learning_path(
    tmp_path: Path,
) -> None:
    """한 번의 통합 승인을 같은 체크인의 비교과·교과에 함께 저장한다."""

    database_path = tmp_path / "integrated-review.db"
    checkin_id = StudentCheckinDataStore(database_path).save_checkin(
        "STU001", _checkin_values()
    )
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        source_checkin_id=checkin_id,
        initial_status="교직원 검토 대기",
    )
    LearningPathDataStore(database_path).save_analysis(
        paths=[
            {
                "student_id": "STU001",
                "path_name": "synthetic 통합 검토 경로",
                "related_job": "데이터 분석",
                "competencies": ["데이터이해", "기초분석"],
                "reason": "통합 검토 영속성 테스트",
                "courses": [
                    {
                        "course_id": f"C{index:03d}",
                        "sequence": index,
                        "score": 90.0 - index,
                        "reason": f"{index}단계 과목",
                    }
                    for index in range(1, 4)
                ],
            }
        ],
        candidates=[],
        source_checkin_id=checkin_id,
    )

    path_updated = service.apply_integrated_review_action(
        context.intervention_id,
        "승인",
        "비교과 연결과 교과목 수강 조건을 확인함",
    )

    intervention = service.get_intervention(context.intervention_id)
    path = LearningPathDataStore(database_path).get_learning_path_by_checkin(
        checkin_id
    )
    assert path_updated is True
    assert intervention is not None
    assert intervention["status"] == "지원계획 작성"
    assert intervention["staff_action"] == "추천 검토 승인"
    assert path["review_status"].unique().tolist() == ["승인"]
    assert path["review_note"].unique().tolist() == [
        "비교과 연결과 교과목 수강 조건을 확인함"
    ]
    assert path["reviewed_at"].notna().all()


def test_integrated_review_reports_when_learning_path_is_not_saved(
    tmp_path: Path,
) -> None:
    """학습경로가 없으면 비교과만 갱신하고 교과 포함을 과장하지 않는다."""

    database_path = tmp_path / "program-only-review.db"
    checkin_id = StudentCheckinDataStore(database_path).save_checkin(
        "STU001", _checkin_values()
    )
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch(
        "STU001", _recommendations(), source_checkin_id=checkin_id
    )

    path_updated = service.apply_integrated_review_action(
        context.intervention_id, "보류"
    )

    assert path_updated is False
    assert service.get_intervention(context.intervention_id)["status"] == "추후 관찰"


def test_separate_review_keeps_program_and_course_decisions_independent(
    tmp_path: Path,
) -> None:
    """비교과 승인과 교과 조정 및 과목별 판정을 서로 덮어쓰지 않는다."""

    database_path = tmp_path / "separate-review.db"
    checkin_id = StudentCheckinDataStore(database_path).save_checkin(
        "STU001", _checkin_values()
    )
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        source_checkin_id=checkin_id,
        initial_status="교직원 검토 대기",
    )
    LearningPathDataStore(database_path).save_analysis(
        paths=[
            {
                "student_id": "STU001",
                "path_name": "독립 검토 경로",
                "related_job": "데이터 분석",
                "competencies": ["기초분석"],
                "reason": "독립 상태 저장 테스트",
                "courses": [
                    {
                        "course_id": f"C{index:03d}",
                        "sequence": index,
                        "score": 90.0 - index,
                        "reason": f"{index}단계 과목",
                    }
                    for index in range(1, 4)
                ],
            }
        ],
        candidates=[],
        source_checkin_id=checkin_id,
    )

    path_updated = service.apply_separate_review_actions(
        intervention_id=context.intervention_id,
        program_decision="승인",
        program_note="비교과 연결 조건 확인",
        course_decision="조정 필요",
        course_note="C003을 다른 과목으로 교체 필요",
        course_judgments={
            "C001": "적절",
            "C002": "보조적으로 적절",
            "C003": "부적절",
        },
    )

    intervention = service.get_intervention(context.intervention_id)
    path = LearningPathDataStore(database_path).get_learning_path_by_checkin(
        checkin_id
    )
    assert path_updated is True
    assert intervention is not None
    assert intervention["staff_action"] == "추천 검토 승인"
    assert intervention["staff_note"] == "비교과 연결 조건 확인"
    assert path["review_status"].unique().tolist() == ["조정 필요"]
    assert path["review_note"].unique().tolist() == [
        "C003을 다른 과목으로 교체 필요"
    ]
    assert dict(zip(path["course_id"], path["review_judgment"])) == {
        "C001": "적절",
        "C002": "보조적으로 적절",
        "C003": "부적절",
    }


def test_separate_course_approval_requires_every_course_judgment(
    tmp_path: Path,
) -> None:
    """경로 승인은 모든 과목을 명시적으로 확인한 뒤에만 허용한다."""

    database_path = tmp_path / "course-review-guard.db"
    checkin_id = StudentCheckinDataStore(database_path).save_checkin(
        "STU001", _checkin_values()
    )
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch(
        "STU001", _recommendations(), source_checkin_id=checkin_id
    )
    LearningPathDataStore(database_path).save_analysis(
        paths=[
            {
                "student_id": "STU001",
                "path_name": "승인 확인 경로",
                "related_job": "데이터 분석",
                "competencies": ["기초분석"],
                "reason": "과목별 확인 테스트",
                "courses": [
                    {
                        "course_id": f"C{index:03d}",
                        "sequence": index,
                        "score": 90.0 - index,
                        "reason": f"{index}단계 과목",
                    }
                    for index in range(1, 4)
                ],
            }
        ],
        candidates=[],
        source_checkin_id=checkin_id,
    )

    with pytest.raises(ValueError, match="모든 추천 교과목"):
        service.apply_separate_review_actions(
            intervention_id=context.intervention_id,
            program_decision="승인",
            course_decision="승인",
            course_judgments={"C001": "적절"},
        )

    with pytest.raises(ValueError, match="교과 승인 전"):
        service.apply_separate_review_actions(
            intervention_id=context.intervention_id,
            program_decision="승인",
            course_decision="승인",
            course_judgments={
                "C001": "적절",
                "C002": "보조적으로 적절",
                "C003": "부적절",
            },
        )


def test_integrated_adjustment_records_note_without_regenerating_recommendations(
    tmp_path: Path,
) -> None:
    """조정 필요는 기존 추천 버전에 상태·메모만 기록하고 새 배치를 만들지 않는다."""

    database_path = tmp_path / "adjustment-review.db"
    checkin_id = StudentCheckinDataStore(database_path).save_checkin(
        "STU001", _checkin_values()
    )
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        source_checkin_id=checkin_id,
        initial_status="교직원 검토 대기",
    )
    LearningPathDataStore(database_path).save_analysis(
        paths=[
            {
                "student_id": "STU001",
                "path_name": "조정 검토 경로",
                "related_job": "데이터 분석",
                "competencies": ["기초분석"],
                "reason": "조정 상태 테스트",
                "courses": [
                    {
                        "course_id": f"C{index:03d}",
                        "sequence": index,
                        "score": 90.0 - index,
                        "reason": f"{index}단계 과목",
                    }
                    for index in range(1, 4)
                ],
            }
        ],
        candidates=[],
        source_checkin_id=checkin_id,
    )

    original_batch_id = context.batch_id
    path_updated = service.apply_integrated_review_action(
        context.intervention_id,
        "수정",
        "두 번째 비교과와 타과 교과목을 다시 확인",
    )

    intervention = service.get_intervention(context.intervention_id)
    path = LearningPathDataStore(database_path).get_learning_path_by_checkin(
        checkin_id
    )
    assert path_updated is True
    assert intervention is not None
    assert intervention["recommendation_batch_id"] == original_batch_id
    assert intervention["staff_action"] == "추천 조정 필요"
    assert intervention["staff_note"] == "두 번째 비교과와 타과 교과목을 다시 확인"
    assert path["review_status"].unique().tolist() == ["조정 필요"]
    assert path["review_note"].unique().tolist() == [
        "두 번째 비교과와 타과 교과목을 다시 확인"
    ]


def test_risk_snapshot_is_linked_to_intervention_and_survives_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    service = InterventionManagementService(database_path=database_path)

    context = service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        risk_snapshot=_risk_snapshot(),
    )

    restarted_service = InterventionManagementService(database_path=database_path)
    stored_intervention = restarted_service.get_intervention(
        context.intervention_id
    )
    stored_snapshot = restarted_service.get_risk_snapshot(
        int(context.risk_snapshot_id or 0)
    )

    assert context.risk_snapshot_id is not None
    assert stored_intervention is not None
    assert stored_intervention["risk_snapshot_id"] == context.risk_snapshot_id
    assert stored_snapshot is not None
    assert stored_snapshot["student_id"] == "STU001"
    assert stored_snapshot["overall_risk"] == pytest.approx(62.8)
    assert stored_snapshot["primary_risk_type"] == "출결위험형"
    assert restarted_service.list_risk_snapshots("STU001")[
        "snapshot_id"
    ].tolist() == [context.risk_snapshot_id]


def test_same_student_checkin_reuses_saved_recommendation_context(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    checkin_id = StudentCheckinDataStore(database_path).save_checkin(
        "STU001", _checkin_values()
    )
    service = InterventionManagementService(database_path=database_path)
    snapshot = _risk_snapshot(
        checkin_source="student_submission",
        student_checkin_id=checkin_id,
    )

    first = service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        risk_snapshot=snapshot,
        source_checkin_id=checkin_id,
        initial_status="교직원 검토 대기",
    )
    second = service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        risk_snapshot=snapshot,
        source_checkin_id=checkin_id,
        initial_status="교직원 검토 대기",
    )

    assert second == first
    assert first.source_checkin_id == checkin_id
    assert first.generation_version == 1
    assert len(service.list_interventions()) == 1
    assert len(service.list_risk_snapshots("STU001")) == 1
    assert len(service.list_recommendations(first.batch_id)) == 3
    stored = service.get_intervention_for_checkin(checkin_id)
    assert stored is not None
    assert stored["status"] == "교직원 검토 대기"


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (_risk_snapshot(overall_risk=101), "0~100 범위"),
        (
            {
                key: value
                for key, value in _risk_snapshot().items()
                if key != "career_risk"
            },
            "필드가 누락",
        ),
    ],
)
def test_invalid_risk_snapshot_is_rejected(
    tmp_path: Path,
    snapshot: dict[str, object],
    message: str,
) -> None:
    store = InterventionDataStore(tmp_path / "app.db")

    with pytest.raises(ValueError, match=message):
        store.save_risk_snapshot("STU001", snapshot)


def test_intervention_rejects_unknown_or_other_student_risk_snapshot(
    tmp_path: Path,
) -> None:
    store = InterventionDataStore(tmp_path / "app.db")
    snapshot_id = store.save_risk_snapshot("STU001", _risk_snapshot())

    with pytest.raises(ValueError, match="존재하지 않는 위험 스냅샷"):
        store.create_intervention("STU001", "batch-1", ["PRG001"], 999)
    with pytest.raises(ValueError, match="다른 학생의 위험 스냅샷"):
        store.create_intervention(
            "STU002", "batch-2", ["PRG001"], snapshot_id
        )


def test_feedback_is_linked_to_existing_student_recommendation(
    tmp_path: Path,
) -> None:
    service = InterventionManagementService(database_path=tmp_path / "app.db")
    context = service.record_recommendation_batch("STU010", _recommendations())

    feedback_id = service.record_feedback(
        student_id="STU010",
        recommendation_id=context.recommendation_ids["PRG002"],
        feedback_type="관심 있음",
        comment="상담 일정을 알고 싶습니다.",
    )
    feedback = service.list_feedback("STU010")

    assert feedback_id > 0
    assert len(feedback) == 1
    assert feedback.iloc[0]["target_id"] == "PRG002"
    assert feedback.iloc[0]["feedback_type"] == "관심 있음"
    assert feedback.iloc[0]["comment"] == "상담 일정을 알고 싶습니다."


def test_feedback_rejects_unknown_type_and_other_student(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch("STU020", _recommendations())
    recommendation_id = context.recommendation_ids["PRG001"]

    with pytest.raises(ValueError, match="피드백 유형"):
        service.record_feedback(
            "STU020", recommendation_id, "임의 응답"
        )
    with pytest.raises(ValueError, match="다른 학생"):
        service.record_feedback(
            "STU999", recommendation_id, "관심 없음"
        )


def test_duplicate_program_recommendations_are_rejected(
    tmp_path: Path,
) -> None:
    service = InterventionManagementService(database_path=tmp_path / "app.db")
    recommendation = _recommendations()[0]

    with pytest.raises(ValueError, match="중복 프로그램"):
        service.record_recommendation_batch(
            "STU030", (recommendation, recommendation)
        )


def test_store_rejects_missing_intervention_id(tmp_path: Path) -> None:
    store = InterventionDataStore(tmp_path / "app.db")

    with pytest.raises(ValueError, match="존재하지 않는 개입 ID"):
        store.update_intervention(999, "상담 예정")


def test_support_plan_selects_replaces_and_tracks_program_execution(
    tmp_path: Path,
) -> None:
    """AI 추천 선택과 DB 프로그램 교체를 실행 계획으로 저장한다."""

    database_path = tmp_path / "support-plan.db"
    service = InterventionManagementService(database_path=database_path)
    context = service.record_recommendation_batch("STU001", _recommendations())

    plan_id = service.save_support_plan(
        intervention_id=context.intervention_id,
        selected_program_ids=["PRG001", "PRG002"],
        valid_program_ids={"PRG001", "PRG002", "PRG003", "PRG004"},
        assigned_staff="IR센터 허혁",
        planned_contact_date="2026-08-20",
        next_action="전화 안내",
        plan_note="학생이 개별 상담과 워크숍을 희망함",
        replacement_program_id="PRG004",
        replaced_program_id="PRG002",
        replacement_reason="학생 일정과 운영 기간을 반영함",
    )

    restarted = InterventionManagementService(database_path=database_path)
    plan = restarted.get_support_plan(context.intervention_id)
    items = restarted.list_support_plan_items(context.intervention_id)
    intervention = restarted.get_intervention(context.intervention_id)

    assert plan is not None
    assert plan["support_plan_id"] == plan_id
    assert plan["assigned_staff"] == "IR센터 허혁"
    assert set(items["program_id"]) == {"PRG001", "PRG004"}
    replacement = items[items["program_id"] == "PRG004"].iloc[0]
    assert replacement["selection_type"] == "교직원 교체"
    assert replacement["replaced_program_id"] == "PRG002"
    assert intervention is not None
    assert intervention["status"] == "연락 전"
    assert intervention["staff_action"] == "지원계획 확정"

    plan_item_id = int(items[items["program_id"] == "PRG001"].iloc[0]["plan_item_id"])
    restarted.update_support_plan_item(
        plan_item_id,
        "연락 완료",
        "학생과 통화함",
        "상담 일정 확인 중",
    )
    updated = restarted.list_support_plan_items(context.intervention_id)
    updated_item = updated[updated["plan_item_id"] == plan_item_id].iloc[0]
    assert updated_item["item_status"] == "연락 완료"
    assert updated_item["last_action"] == "학생과 통화함"


def test_support_plan_rejects_unknown_program_and_unexplained_replacement(
    tmp_path: Path,
) -> None:
    """지원계획은 master 외 ID와 근거 없는 교체를 허용하지 않는다."""

    service = InterventionManagementService(database_path=tmp_path / "invalid-plan.db")
    context = service.record_recommendation_batch("STU001", _recommendations())
    common = {
        "intervention_id": context.intervention_id,
        "selected_program_ids": ["PRG001"],
        "valid_program_ids": {"PRG001", "PRG002", "PRG003", "PRG004"},
        "assigned_staff": "IR센터",
        "planned_contact_date": "2026-08-20",
        "next_action": "전화 안내",
    }

    with pytest.raises(ValueError, match="master에 없는 교체"):
        service.save_support_plan(
            **common,
            replacement_program_id="PRG999",
            replaced_program_id="PRG002",
            replacement_reason="교체",
        )
    with pytest.raises(ValueError, match="교체 이유"):
        service.save_support_plan(
            **common,
            replacement_program_id="PRG004",
            replaced_program_id="PRG002",
        )


def test_management_page_displays_linked_risk_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    service = InterventionManagementService(database_path=database_path)
    service.record_recommendation_batch(
        "STU001",
        _recommendations(),
        risk_snapshot=_risk_snapshot(),
    )
    st.cache_resource.clear()

    app = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "05_intervention_management.py")
    ).run(timeout=30)

    assert not app.exception
    assert any(item.label == "조회 조건" for item in app.expander)
    assert any(
        selectbox.label == "관리할 지원" for selectbox in app.selectbox
    )
    assert not any(
        selectbox.label
        in {"위험등급", "위험유형", "상태", "학과", "관리할 지원"}
        for selectbox in app.sidebar.selectbox
    )
    assert not any(
        text_input.label == "학생 ID 검색" for text_input in app.sidebar.text_input
    )
    assert any(
        metric.label == "현재 위험등급" for metric in app.metric
    )
    assert any(item.value == "지원 현황" for item in app.subheader)
    app.pills[0].set_value("추천 근거 · 피드백").run(timeout=30)
    assert not app.exception
    assert any(
        "지원 시작 시점 스냅샷" in element.value
        for element in app.markdown
    )
    assert any(
        metric.label == "종합 위험도" and metric.value == "62.8"
        for metric in app.metric
    )

    app.pills[0].set_value("지원 계획 확정").run(timeout=30)
    assert not app.exception
    assert any(
        "AI가 제안하는 다음 조치" in str(element.value)
        for element in app.markdown
    )
    assert any(
        button.label == "다음 조치 제안" for button in app.button
    )
