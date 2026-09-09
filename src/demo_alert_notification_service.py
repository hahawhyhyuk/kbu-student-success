"""조기경보에서 학생 Kare 체크인까지 이어지는 비식별 시연 알림 흐름."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

from src.database import DemoAlertNotificationDataStore
from src.utils import load_app_config


STATUS_SENT = "안내 발송"
STATUS_VIEWED = "학생 확인"
STATUS_KARE_STARTED = "Kare 상담 시작"
STATUS_CHECKIN_COMPLETED = "체크인 완료"


@dataclass(frozen=True)
class DemoAlertMailPreview:
    """외부 주소를 포함하지 않는 교직원용 안내 미리보기."""

    student_id: str
    alert_week: int
    destination_label: str
    subject: str
    body: str


@dataclass(frozen=True)
class DemoAlertNotification:
    """SQLite에 저장된 시연용 경보 안내와 학생 행동 상태."""

    notification_id: int
    student_id: str
    alert_week: int
    risk_level: str
    risk_score: float
    primary_risk_type: str
    subject: str
    body: str
    created_by: str
    status: str
    sent_at: str
    student_viewed_at: str | None
    kare_started_at: str | None
    checkin_completed_at: str | None
    source_checkin_id: int | None


@dataclass(frozen=True)
class DemoAlertSendResult:
    """시연용 안내 발송 요청의 저장 결과."""

    notification: DemoAlertNotification
    created: bool


class DemoAlertNotificationService:
    """실제 메일 없이 조기경보 안내와 Kare 행동 상태를 연결한다."""

    SUBJECT = "[KBU 학생성공 지원] Kare 체크인 안내"
    BODY = (
        "최근 학교생활을 돌아보고 필요한 지원을 함께 찾을 수 있도록 "
        "Kare 체크인을 안내드립니다. 학생 서비스의 '나의 체크인'에서 "
        "요즘 고민을 편하게 이야기해 주세요. AI 대화 동의는 선택이며, "
        "동의하지 않아도 외부 AI 전송 없이 직접 입력할 수 있습니다."
    )

    def __init__(
        self,
        database_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        store: DemoAlertNotificationDataStore | None = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_app_config()
        demo_config = self.config["demo"]
        self.enabled = bool(demo_config["enabled"])
        self.store = store or DemoAlertNotificationDataStore(database_path)

    @staticmethod
    def _notification_from_row(
        row: Mapping[str, Any] | None,
    ) -> DemoAlertNotification | None:
        """DB mapping을 명시적인 시연 알림 모델로 변환한다."""

        if row is None:
            return None
        return DemoAlertNotification(
            notification_id=int(row["notification_id"]),
            student_id=str(row["student_id"]),
            alert_week=int(row["alert_week"]),
            risk_level=str(row["risk_level"]),
            risk_score=float(row["risk_score"]),
            primary_risk_type=str(row["primary_risk_type"]),
            subject=str(row["subject"]),
            body=str(row["body"]),
            created_by=str(row["created_by"]),
            status=str(row["status"]),
            sent_at=str(row["sent_at"]),
            student_viewed_at=(
                str(row["student_viewed_at"])
                if row.get("student_viewed_at") is not None
                else None
            ),
            kare_started_at=(
                str(row["kare_started_at"])
                if row.get("kare_started_at") is not None
                else None
            ),
            checkin_completed_at=(
                str(row["checkin_completed_at"])
                if row.get("checkin_completed_at") is not None
                else None
            ),
            source_checkin_id=(
                int(row["source_checkin_id"])
                if row.get("source_checkin_id") is not None
                else None
            ),
        )

    def _validate_alert_snapshot(
        self,
        snapshot: Mapping[str, Any],
        valid_student_ids: Iterable[str],
    ) -> tuple[str, int]:
        """현재 시연에서 안내를 만들 수 있는 synthetic 경보인지 확인한다."""

        if not self.enabled:
            raise ValueError("현재 환경에서는 시연용 경보 안내가 비활성화되어 있습니다.")
        student_id = str(snapshot.get("student_id", "")).strip()
        allowed = {str(item).strip() for item in valid_student_ids}
        if not student_id or student_id not in allowed:
            raise ValueError("학생 master에 없는 계정으로는 안내할 수 없습니다.")
        week = int(snapshot.get("week", 0) or 0)
        if week < 1:
            raise ValueError("안내할 관찰 주차는 1 이상이어야 합니다.")
        if str(snapshot.get("risk_level", "")).strip() != "고위험":
            raise ValueError("고위험 경보가 작동한 학생에게만 안내할 수 있습니다.")
        score = float(snapshot.get("overall_risk", 0) or 0)
        if not 0 <= score <= 100:
            raise ValueError("종합 위험도는 0~100 범위여야 합니다.")
        return student_id, week

    def build_preview(
        self,
        snapshot: Mapping[str, Any],
        valid_student_ids: Iterable[str],
    ) -> DemoAlertMailPreview:
        """학생 ID·위험 낙인을 본문에 넣지 않은 안내 미리보기를 만든다."""

        student_id, week = self._validate_alert_snapshot(
            snapshot, valid_student_ids
        )
        return DemoAlertMailPreview(
            student_id=student_id,
            alert_week=week,
            destination_label=f"교내 알림함 · {student_id} (synthetic)",
            subject=self.SUBJECT,
            body=self.BODY,
        )

    def send(
        self,
        snapshot: Mapping[str, Any],
        valid_student_ids: Iterable[str],
        *,
        created_by: str,
    ) -> DemoAlertSendResult:
        """외부 전송 없이 학생·주차별 안내 상태를 SQLite에 한 번 저장한다."""

        preview = self.build_preview(snapshot, valid_student_ids)
        row, created = self.store.create_notification(
            student_id=preview.student_id,
            alert_week=preview.alert_week,
            risk_level=str(snapshot["risk_level"]),
            risk_score=float(snapshot["overall_risk"]),
            primary_risk_type=str(snapshot["primary_risk_type"]),
            subject=preview.subject,
            body=preview.body,
            created_by=str(created_by),
        )
        notification = self._notification_from_row(row)
        if notification is None:
            raise RuntimeError("시연용 안내 저장 결과를 확인하지 못했습니다.")
        return DemoAlertSendResult(notification, created)

    def get_notification(
        self,
        student_id: str,
        alert_week: int | None = None,
    ) -> DemoAlertNotification | None:
        """학생의 특정 경보 주차 또는 최신 안내를 조회한다."""

        if alert_week is None:
            row = self.store.get_latest_notification(student_id)
        else:
            row = self.store.get_notification(student_id, alert_week)
        return self._notification_from_row(row)

    def mark_student_viewed(
        self,
        student_id: str,
    ) -> DemoAlertNotification | None:
        """학생이 자기 화면에서 안내를 확인한 상태로 진행한다."""

        return self._notification_from_row(
            self.store.advance_status(student_id, STATUS_VIEWED)
        )

    def mark_kare_started(
        self,
        student_id: str,
    ) -> DemoAlertNotification | None:
        """학생이 AI 동의 여부와 무관하게 Kare 체크인을 시작했음을 기록한다."""

        return self._notification_from_row(
            self.store.advance_status(student_id, STATUS_KARE_STARTED)
        )

    def mark_checkin_completed(
        self,
        student_id: str,
        checkin_id: int,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> DemoAlertNotification | None:
        """학생 체크인 저장이 성공한 뒤 안내 흐름을 완료 상태로 연결한다."""

        return self._notification_from_row(
            self.store.advance_status(
                student_id,
                STATUS_CHECKIN_COMPLETED,
                checkin_id=int(checkin_id),
                connection=connection,
            )
        )
