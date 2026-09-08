"""조기경보 안내부터 학생 Kare 체크인까지의 시연 상태 테스트."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ai import MockAIProvider
from src.demo_alert_notification_service import (
    STATUS_CHECKIN_COMPLETED,
    STATUS_KARE_STARTED,
    STATUS_SENT,
    STATUS_VIEWED,
    DemoAlertNotificationService,
)
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService
from src.student_view_service import StudentViewService
from src.utils import PROJECT_ROOT


def _config() -> dict[str, object]:
    return {
        "demo": {
            "enabled": True,
        }
    }


def _alert_snapshot() -> dict[str, object]:
    return {
        "student_id": "S0003",
        "week": 4,
        "risk_level": "고위험",
        "overall_risk": 74.6,
        "primary_risk_type": "출결위험형",
    }


def _scaled_checkin() -> dict[str, object]:
    return {
        "week": 4,
        "major_interest": 2,
        "major_satisfaction": 2,
        "major_continuation_intent": 2,
        "learning_difficulty": 4,
        "career_clarity": 2,
        "consultation_intent": 5,
        "interest_fields": ["데이터·AI"],
        "desired_job": "데이터 분석가",
        "natural_language_concern": "최근 결석과 밀린 과제가 늘어 상담을 받고 싶어요.",
        "response_mode": "scaled",
    }


def test_preview_and_send_are_private_and_idempotent(tmp_path: Path) -> None:
    """실제 주소 없이 미리보고 같은 학생·주차 안내는 한 번만 저장한다."""

    database_path = tmp_path / "alert.db"
    service = DemoAlertNotificationService(
        database_path=database_path,
        config=_config(),
    )
    snapshot = _alert_snapshot()

    preview = service.build_preview(snapshot, ["S0003"])
    first = service.send(snapshot, ["S0003"], created_by="STAFF001")
    second = service.send(snapshot, ["S0003"], created_by="STAFF001")

    assert preview.destination_label == "교내 알림함 · S0003 (synthetic)"
    assert "@" not in preview.destination_label
    assert "S0003" not in preview.body
    assert "고위험" not in preview.body
    assert first.created is True
    assert second.created is False
    assert first.notification.notification_id == second.notification.notification_id
    assert first.notification.status == STATUS_SENT
    with sqlite3.connect(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM demo_alert_notifications"
        ).fetchone()[0]
    assert count == 1


def test_notification_status_only_advances(tmp_path: Path) -> None:
    """학생 확인·Kare 시작·체크인 완료 상태는 뒤로 되돌아가지 않는다."""

    service = DemoAlertNotificationService(
        database_path=tmp_path / "alert.db",
        config=_config(),
    )
    service.send(_alert_snapshot(), ["S0003"], created_by="STAFF001")

    viewed = service.mark_student_viewed("S0003")
    started = service.mark_kare_started("S0003")
    completed = service.mark_checkin_completed("S0003", 17)
    unchanged = service.mark_student_viewed("S0003")

    assert viewed is not None and viewed.status == STATUS_VIEWED
    assert viewed.student_viewed_at is not None
    assert started is not None and started.status == STATUS_KARE_STARTED
    assert started.kare_started_at is not None
    assert completed is not None and completed.status == STATUS_CHECKIN_COMPLETED
    assert completed.checkin_completed_at is not None
    assert completed.source_checkin_id == 17
    assert unchanged is not None and unchanged.status == STATUS_CHECKIN_COMPLETED


def test_send_rejects_non_alert_or_unknown_student(tmp_path: Path) -> None:
    """주차와 무관하게 고위험 synthetic master 학생만 안내할 수 있다."""

    service = DemoAlertNotificationService(
        database_path=tmp_path / "alert.db",
        config=_config(),
    )
    early = {**_alert_snapshot(), "week": 3, "risk_level": "주의"}

    with pytest.raises(ValueError, match="고위험"):
        service.send(early, ["S0003"], created_by="STAFF001")
    with pytest.raises(ValueError, match="master"):
        service.send(_alert_snapshot(), ["S0001"], created_by="STAFF001")

    week_two_high_risk = {**_alert_snapshot(), "week": 2}
    preview = service.build_preview(week_two_high_risk, ["S0003"])
    assert preview.alert_week == 2

    invalid_week = {**_alert_snapshot(), "week": 0}
    with pytest.raises(ValueError, match="1 이상"):
        service.build_preview(invalid_week, ["S0003"])


def test_student_checkin_completes_sent_alert(tmp_path: Path) -> None:
    """학생 체크인이 실제 저장된 뒤 시연 안내도 완료 상태로 연결된다."""

    database_path = tmp_path / "alert.db"
    notification_service = DemoAlertNotificationService(
        database_path=database_path,
        config=_config(),
    )
    notification_service.send(
        _alert_snapshot(), ["S0003"], created_by="STAFF001"
    )
    student_service = StudentViewService(
        repository=get_default_repository(),
        ai_provider=MockAIProvider(),
        database_path=database_path,
    )

    result = student_service.submit_checkin("S0003", _scaled_checkin())
    notification = notification_service.get_notification("S0003", 4)

    assert notification is not None
    assert notification.status == STATUS_CHECKIN_COMPLETED
    assert notification.source_checkin_id == result.checkin_id


def test_staff_send_and_student_kare_button_share_saved_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """교직원 발송 상태가 역할 전환 뒤 학생 화면의 Kare 버튼으로 이어진다."""

    database_path = tmp_path / "ui-alert.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    dashboard = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "01_dashboard.py")
    ).run(timeout=30)

    assert not dashboard.exception
    send_button = next(
        button
        for button in dashboard.button
        if button.label == "시연용 안내 발송"
    )
    send_button.click().run(timeout=30)
    notification_service = DemoAlertNotificationService(
        database_path=database_path,
        config=_config(),
    )
    sent = notification_service.get_notification("S0003", 4)
    assert sent is not None and sent.status == STATUS_SENT

    student_page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    )
    student_page.session_state["demo_student_id"] = "S0003"
    student_page.run(timeout=30)

    assert not student_page.exception
    assert any(
        button.label == "Kare와 이야기하기"
        for button in student_page.button
    )
    viewed = notification_service.get_notification("S0003", 4)
    assert viewed is not None and viewed.status == STATUS_VIEWED

    kare_button = next(
        button
        for button in student_page.button
        if button.label == "Kare와 이야기하기"
    )
    kare_button.click().run(timeout=30)
    started = notification_service.get_notification("S0003", 4)
    assert started is not None and started.status == STATUS_KARE_STARTED


def test_actual_week_four_snapshot_is_eligible_for_notification(
    tmp_path: Path,
) -> None:
    """현재 synthetic 대표 학생의 4주차 경보를 그대로 안내에 사용할 수 있다."""

    snapshots = RiskAnalysisService(get_default_repository()).build_snapshots()
    row = snapshots[
        (snapshots["student_id"].astype(str) == "S0003")
        & (snapshots["week"].astype(int) == 4)
    ].iloc[0]
    service = DemoAlertNotificationService(
        database_path=tmp_path / "alert.db",
        config=_config(),
    )

    preview = service.build_preview(
        row.to_dict(),
        snapshots["student_id"].astype(str).unique(),
    )

    assert preview.student_id == "S0003"
    assert preview.alert_week == 4
