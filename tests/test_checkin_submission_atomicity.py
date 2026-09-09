"""체크인 처리 실패와 재시도가 불완전한 학생 이력을 만들지 않는지 검증한다."""

from pathlib import Path
import sqlite3
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from src.database import DEMO_WORKFLOW_TABLES, InterventionDataStore
from src.utils import PROJECT_ROOT
from tests.test_student_view import (
    NARRATIVE_STUDENT_VALUES,
    STUDENT_VALUES,
    _student_service,
)


def _workflow_state(database_path: Path) -> dict[str, list[tuple[Any, ...]]]:
    """실패 전후의 기존 기록과 안내 상태를 빠짐없이 비교한다."""

    with sqlite3.connect(database_path) as connection:
        return {
            table: connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            ).fetchall()
            for table in DEMO_WORKFLOW_TABLES
        }


@pytest.mark.parametrize("values", [STUDENT_VALUES, NARRATIVE_STUDENT_VALUES])
@pytest.mark.parametrize(
    "failure_stage",
    ["analysis", "recommendation", "save_analysis", "snapshot", "intervention", "alert"],
)
def test_failed_submission_preserves_history_and_retry_saves_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Any],
    failure_stage: str,
) -> None:
    """계산·저장·안내 연결에서 실패해도 이전 기록을 보존하고 재시도는 한 건만 저장한다."""

    database_path = tmp_path / "retry.db"
    service = _student_service(database_path)
    original = service.submit_checkin("S0003", values)
    service.demo_alert_notification_service.send(
        {
            "student_id": "S0003",
            "week": 4,
            "risk_level": "고위험",
            "overall_risk": 80,
            "primary_risk_type": "복합위험형",
        },
        ["S0003"],
        created_by="STAFF001",
    )
    before = _workflow_state(database_path)
    owner, method_name = {
        "analysis": (service, "preview_checkin_analysis"),
        "recommendation": (service.program_recommender, "recommend"),
        "save_analysis": (service.checkin_store, "save_analysis"),
        "snapshot": (service.intervention_service.store, "save_risk_snapshot"),
        "intervention": (service.intervention_service.store, "create_intervention"),
        "alert": (service.demo_alert_notification_service, "mark_checkin_completed"),
    }[failure_stage]
    working_method = getattr(owner, method_name)

    def fail_after_operation(*args: Any, **kwargs: Any) -> None:
        working_method(*args, **kwargs)
        raise RuntimeError("시뮬레이션한 처리 실패")

    with monkeypatch.context() as patch:
        patch.setattr(owner, method_name, fail_after_operation)
        with pytest.raises(RuntimeError, match="시뮬레이션한 처리 실패"):
            service.submit_checkin("S0003", values)

    assert _workflow_state(database_path) == before
    context = service.get_staff_student_context("S0003")
    assert context.latest_checkin is not None
    assert context.latest_checkin["checkin_id"] == original.checkin_id

    retried = service.submit_checkin("S0003", values)
    after = _workflow_state(database_path)
    assert len(after["student_checkins"]) == len(before["student_checkins"]) + 1
    assert len(after["student_checkin_analyses"]) == (
        len(before["student_checkin_analyses"]) + 1
    )
    assert len(after["interventions"]) == len(before["interventions"]) + 1
    assert retried.intervention_context.source_checkin_id == retried.checkin_id
    context = service.get_staff_student_context("S0003")
    assert context.intervention is not None
    assert context.intervention["intervention_id"] == (
        retried.intervention_context.intervention_id
    )
    notification = service.demo_alert_notification_service.get_notification("S0003")
    assert notification is not None
    assert notification.status == "체크인 완료"
    assert notification.source_checkin_id == retried.checkin_id
    if values.get("response_mode") != "narrative":
        assert retried.profile["week"] == original.profile["week"] + 1


@pytest.mark.parametrize("failure_stage", ["path", "recommendation_link"])
def test_optional_learning_path_failure_can_be_recovered_on_same_checkin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """학습경로 저장 중 실패하면 해당 경로만 되돌리고 같은 체크인에 복구한다."""

    database_path = tmp_path / "path-recovery.db"
    service = _student_service(database_path)
    owner, name = (
        (service.learning_path_store, "save_analysis")
        if failure_stage == "path"
        else (service.intervention_service.store, "save_recommendation_batch")
    )
    working_method = getattr(owner, name)

    def fail_path_save(*args: Any, **kwargs: Any) -> Any:
        result = working_method(*args, **kwargs)
        if failure_stage == "path" or (
            kwargs["recommendations"][0]["recommendation_type"] == "learning_path"
        ):
            raise RuntimeError("시뮬레이션한 학습경로 저장 실패")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, fail_path_save)
        result = service.submit_checkin("S0003", NARRATIVE_STUDENT_VALUES)

    state = _workflow_state(database_path)
    assert len(state["student_checkins"]) == 1
    assert len(state["interventions"]) == 1
    assert len(state["recommendations"]) == 3
    assert state["learning_paths"] == []
    assert state["learning_path_courses"] == []
    assert result.learning_path_result is None
    assert result.learning_path_recommendation_id is None
    assert result.learning_path_error == "시뮬레이션한 학습경로 저장 실패"

    recovered, _ = service.build_and_store_learning_path(
        "S0003",
        result.profile,
        result.analysis_result.analysis,
        service.repository.get_all(),
    )
    after = _workflow_state(database_path)
    assert after["student_checkins"] == state["student_checkins"]
    assert after["interventions"] == state["interventions"]
    assert len(after["learning_paths"]) == 1
    assert len(after["recommendations"]) == 4
    assert len(after["learning_path_courses"]) == len(recovered.selection.courses)


def test_student_page_keeps_input_and_retries_failed_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실제 제출 버튼에서 실패 안내·입력 유지·재제출 성공을 확인한다."""

    database_path = tmp_path / "page-retry.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    service = _student_service(database_path)
    page = AppTest.from_file(str(PROJECT_ROOT / "pages" / "07_student_view.py"))
    prefix = "student_checkin_S0003_"
    page.session_state["demo_student_id"] = "S0003"
    page.session_state[prefix + "draft"] = dict(NARRATIVE_STUDENT_VALUES)
    page.session_state[prefix + "conversation_mode"] = "review"
    page.session_state[prefix + "input_mode"] = "direct"
    page.session_state[prefix + "analysis_preview"] = service.preview_checkin_analysis(
        NARRATIVE_STUDENT_VALUES["natural_language_concern"]
    )
    page.run(timeout=30)
    assert not page.exception
    before = _workflow_state(database_path)
    working_method = InterventionDataStore.create_intervention

    def fail_after_intervention(
        self: InterventionDataStore, *args: Any, **kwargs: Any,
    ) -> None:
        working_method(self, *args, **kwargs)
        raise RuntimeError("일시적인 저장 오류")

    with monkeypatch.context() as patch:
        patch.setattr(
            InterventionDataStore, "create_intervention", fail_after_intervention
        )
        next(
            button for button in page.button if button.label == "맞아요, 제출할게요"
        ).click().run(timeout=30)

    assert not page.exception
    assert any("다시 제출해 주세요" in str(error.value) for error in page.error)
    assert _workflow_state(database_path) == before
    assert page.session_state[prefix + "draft"]["natural_language_concern"] == (
        NARRATIVE_STUDENT_VALUES["natural_language_concern"]
    )
    next(
        button for button in page.button if button.label == "맞아요, 제출할게요"
    ).click().run(timeout=30)

    assert not page.exception
    assert not page.error
    after = _workflow_state(database_path)
    assert len(after["student_checkins"]) == 1
    assert len(after["interventions"]) == 1
    assert page.session_state[prefix + "conversation_mode"] == "submitted"
