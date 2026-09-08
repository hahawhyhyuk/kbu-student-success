"""SQLite 연결, 스키마, 개입 이력 CRUD를 한 곳에서 관리한다."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.checkin_semantic_state import validate_semantic_state
from src.utils import PROJECT_ROOT


DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "app.db"


DEMO_WORKFLOW_TABLES: tuple[str, ...] = (
    "demo_alert_notifications",
    "feedback",
    "student_checkin_feedback",
    "student_checkin_analyses",
    "learning_path_courses",
    "support_plan_items",
    "support_plans",
    "interventions",
    "recommendations",
    "risk_snapshots",
    "student_checkins",
    "microdegree_candidates",
    "learning_paths",
)


def _connect(database_path: Path) -> sqlite3.Connection:
    """외래키 검증과 이름 기반 조회가 활성화된 SQLite 연결을 반환한다."""

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _migrate_student_checkins_for_narrative(
    connection: sqlite3.Connection,
) -> None:
    """기존 척도 체크인을 보존하며 자연어 전용 체크인을 저장할 수 있게 한다.

    SQLite는 ``NOT NULL`` 제약을 직접 제거할 수 없으므로 필요한 경우에만
    테이블을 복사해 교체한다. 기존 분석·피드백 외래키는 이름이 같은 새
    ``student_checkins`` 테이블을 계속 가리킨다.
    """

    columns = {
        str(row["name"]): row
        for row in connection.execute(
            "PRAGMA table_info(student_checkins)"
        ).fetchall()
    }
    if "response_mode" not in columns:
        connection.execute(
            "ALTER TABLE student_checkins ADD COLUMN "
            "response_mode TEXT NOT NULL DEFAULT 'scaled'"
        )
        columns = {
            str(row["name"]): row
            for row in connection.execute(
                "PRAGMA table_info(student_checkins)"
            ).fetchall()
        }
    if "semantic_states" not in columns:
        connection.execute(
            "ALTER TABLE student_checkins ADD COLUMN "
            "semantic_states TEXT NOT NULL DEFAULT '{}'"
        )
        columns = {
            str(row["name"]): row
            for row in connection.execute(
                "PRAGMA table_info(student_checkins)"
            ).fetchall()
        }
    likert_fields = (
        "major_interest",
        "major_satisfaction",
        "major_continuation_intent",
        "learning_difficulty",
        "career_clarity",
        "consultation_intent",
    )
    if not any(bool(columns[field]["notnull"]) for field in likert_fields):
        return

    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN")
        connection.execute(
            """
            CREATE TABLE student_checkins_narrative_migration (
                checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                week INTEGER,
                major_interest INTEGER,
                major_satisfaction INTEGER,
                major_continuation_intent INTEGER,
                learning_difficulty INTEGER,
                career_clarity INTEGER,
                consultation_intent INTEGER,
                interest_fields TEXT NOT NULL,
                desired_job TEXT NOT NULL,
                consultation_requested INTEGER NOT NULL DEFAULT 0,
                explore_other_fields INTEGER NOT NULL DEFAULT 0,
                natural_language_concern TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                response_mode TEXT NOT NULL DEFAULT 'scaled',
                semantic_states TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO student_checkins_narrative_migration (
                checkin_id, student_id, week, major_interest,
                major_satisfaction, major_continuation_intent,
                learning_difficulty, career_clarity, consultation_intent,
                interest_fields, desired_job, consultation_requested,
                explore_other_fields, natural_language_concern, created_at,
                response_mode, semantic_states
            )
            SELECT checkin_id, student_id, week, major_interest,
                   major_satisfaction, major_continuation_intent,
                   learning_difficulty, career_clarity, consultation_intent,
                   interest_fields, desired_job, consultation_requested,
                   explore_other_fields, natural_language_concern, created_at,
                   COALESCE(response_mode, 'scaled'),
                   COALESCE(semantic_states, '{}')
            FROM student_checkins
            """
        )
        connection.execute("DROP TABLE student_checkins")
        connection.execute(
            "ALTER TABLE student_checkins_narrative_migration "
            "RENAME TO student_checkins"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("체크인 스키마 변경 후 외래키 검증에 실패했습니다.")


def initialize_database(database_path: str | Path | None = None) -> Path:
    """개입 추천, 교직원 상태, 학생 피드백을 저장할 스키마를 준비한다.

    Parameters:
        database_path: 생성할 SQLite 파일 경로. 미지정 시 database/app.db.

    Returns:
        초기화된 SQLite 파일의 절대 경로.

    Assumptions:
        기존 SQLite DB가 있으면 데이터를 보존한 채 누락 컬럼과 테이블만 추가한다.
    """

    path = Path(database_path) if database_path else DEFAULT_DATABASE_PATH
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                score REAL NOT NULL,
                rank INTEGER NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (batch_id, target_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                week INTEGER NOT NULL,
                attendance_risk REAL NOT NULL,
                engagement_risk REAL NOT NULL,
                achievement_risk REAL NOT NULL,
                major_adaptation_risk REAL NOT NULL,
                career_risk REAL NOT NULL,
                overall_risk REAL NOT NULL,
                risk_level TEXT NOT NULL,
                primary_risk_type TEXT NOT NULL,
                secondary_risk_types TEXT NOT NULL DEFAULT '',
                checkin_source TEXT NOT NULL DEFAULT 'repository',
                student_checkin_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                risk_snapshot_id INTEGER,
                recommendation_batch_id TEXT,
                source_checkin_id INTEGER,
                generation_version INTEGER NOT NULL DEFAULT 1,
                recommended_action TEXT,
                staff_action TEXT,
                status TEXT NOT NULL DEFAULT '추천 생성',
                student_response TEXT,
                staff_note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(interventions)"
            ).fetchall()
        }
        migration_columns = {
            "risk_snapshot_id": "INTEGER",
            "recommendation_batch_id": "TEXT",
            "source_checkin_id": "INTEGER",
            "generation_version": "INTEGER DEFAULT 1",
            "recommended_action": "TEXT",
            "staff_action": "TEXT",
            "status": "TEXT DEFAULT '추천 생성'",
            "student_response": "TEXT",
            "staff_note": "TEXT",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }
        for column_name, column_type in migration_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE interventions ADD COLUMN "
                    f"{column_name} {column_type}"
                )
        connection.execute(
            """
            UPDATE interventions
            SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP),
                updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP),
                status = COALESCE(status, '추천 생성'),
                generation_version = COALESCE(generation_version, 1)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS support_plans (
                support_plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_id INTEGER NOT NULL UNIQUE,
                student_id TEXT NOT NULL,
                assigned_staff TEXT NOT NULL,
                planned_contact_date TEXT NOT NULL,
                next_action TEXT NOT NULL,
                plan_note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '계획 확정',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (intervention_id) REFERENCES interventions(intervention_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS support_plan_items (
                plan_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                support_plan_id INTEGER NOT NULL,
                program_id TEXT NOT NULL,
                source_recommendation_id INTEGER,
                selection_type TEXT NOT NULL,
                replaced_program_id TEXT,
                replacement_reason TEXT NOT NULL DEFAULT '',
                item_status TEXT NOT NULL DEFAULT '지원 예정',
                last_action TEXT NOT NULL DEFAULT '',
                staff_note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (support_plan_id, program_id),
                FOREIGN KEY (support_plan_id) REFERENCES support_plans(support_plan_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (source_recommendation_id)
                    REFERENCES recommendations(recommendation_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                recommendation_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recommendation_id)
                    REFERENCES recommendations(recommendation_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_checkins (
                checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                week INTEGER,
                major_interest INTEGER,
                major_satisfaction INTEGER,
                major_continuation_intent INTEGER,
                learning_difficulty INTEGER,
                career_clarity INTEGER,
                consultation_intent INTEGER,
                interest_fields TEXT NOT NULL,
                desired_job TEXT NOT NULL,
                consultation_requested INTEGER NOT NULL DEFAULT 0,
                explore_other_fields INTEGER NOT NULL DEFAULT 0,
                natural_language_concern TEXT NOT NULL,
                response_mode TEXT NOT NULL DEFAULT 'scaled',
                semantic_states TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        student_checkin_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(student_checkins)"
            ).fetchall()
        }
        if "week" not in student_checkin_columns:
            connection.execute(
                "ALTER TABLE student_checkins ADD COLUMN week INTEGER"
            )
        _migrate_student_checkins_for_narrative(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_checkin_analyses (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkin_id INTEGER NOT NULL UNIQUE,
                provider_name TEXT NOT NULL,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                major_concern INTEGER NOT NULL,
                learning_difficulty INTEGER NOT NULL,
                career_uncertainty INTEGER NOT NULL,
                interests TEXT NOT NULL,
                desired_jobs TEXT NOT NULL,
                support_needs TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (checkin_id) REFERENCES student_checkins(checkin_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_checkin_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkin_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (checkin_id) REFERENCES student_checkins(checkin_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_alert_notifications (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                alert_week INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                primary_risk_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                created_by TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT '안내 발송',
                sent_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                student_viewed_at TEXT,
                kare_started_at TEXT,
                checkin_completed_at TEXT,
                source_checkin_id INTEGER,
                UNIQUE (student_id, alert_week)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_paths (
                path_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_batch_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                source_checkin_id INTEGER,
                path_name TEXT NOT NULL,
                related_job TEXT,
                competencies TEXT NOT NULL,
                reason TEXT NOT NULL,
                career_connection TEXT NOT NULL DEFAULT '',
                provider_name TEXT NOT NULL DEFAULT '',
                fallback_used INTEGER NOT NULL DEFAULT 0,
                review_status TEXT NOT NULL DEFAULT '검토 대기',
                review_note TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (analysis_batch_id, student_id)
            )
            """
        )
        learning_path_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(learning_paths)"
            ).fetchall()
        }
        if "source_checkin_id" not in learning_path_columns:
            connection.execute(
                "ALTER TABLE learning_paths ADD COLUMN source_checkin_id INTEGER"
            )
        learning_path_migration_columns = {
            "career_connection": "TEXT DEFAULT ''",
            "provider_name": "TEXT DEFAULT ''",
            "fallback_used": "INTEGER DEFAULT 0",
            "review_status": "TEXT NOT NULL DEFAULT '검토 대기'",
            "review_note": "TEXT NOT NULL DEFAULT ''",
            "reviewed_at": "TEXT",
        }
        for column_name, column_type in learning_path_migration_columns.items():
            if column_name not in learning_path_columns:
                connection.execute(
                    f"ALTER TABLE learning_paths ADD COLUMN "
                    f"{column_name} {column_type}"
                )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_path_courses (
                path_id INTEGER NOT NULL,
                course_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                score REAL NOT NULL,
                reason TEXT NOT NULL,
                selection_reason TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (path_id, course_id),
                FOREIGN KEY (path_id) REFERENCES learning_paths(path_id)
                    ON DELETE CASCADE
            )
            """
        )
        learning_path_course_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(learning_path_courses)"
            ).fetchall()
        }
        if "selection_reason" not in learning_path_course_columns:
            connection.execute(
                "ALTER TABLE learning_path_courses ADD COLUMN "
                "selection_reason TEXT DEFAULT ''"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS microdegree_candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_batch_id TEXT NOT NULL,
                candidate_code TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                student_count INTEGER NOT NULL,
                department_count INTEGER NOT NULL,
                departments TEXT NOT NULL,
                course_ids TEXT NOT NULL,
                course_names TEXT NOT NULL,
                competencies TEXT NOT NULL,
                related_jobs TEXT NOT NULL,
                student_ids TEXT NOT NULL,
                average_similarity REAL NOT NULL,
                priority TEXT NOT NULL DEFAULT '재분석 필요',
                demand_share REAL NOT NULL DEFAULT 0,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (analysis_batch_id, candidate_code)
            )
            """
        )
        candidate_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(microdegree_candidates)"
            ).fetchall()
        }
        candidate_migration_columns = {
            "departments": "TEXT",
            "priority": "TEXT NOT NULL DEFAULT '재분석 필요'",
            "demand_share": "REAL NOT NULL DEFAULT 0",
        }
        for column_name, column_type in candidate_migration_columns.items():
            if column_name not in candidate_columns:
                connection.execute(
                    "ALTER TABLE microdegree_candidates ADD COLUMN "
                    f"{column_name} {column_type}"
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_interventions_student "
            "ON interventions(student_id, updated_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_snapshots_student "
            "ON risk_snapshots(student_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_recommendations_batch "
            "ON recommendations(batch_id, rank)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_student "
            "ON feedback(student_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_support_plans_student "
            "ON support_plans(student_id, updated_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_support_plan_items_plan "
            "ON support_plan_items(support_plan_id, item_status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_student_checkins_student "
            "ON student_checkins(student_id, checkin_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_demo_alert_notifications_student "
            "ON demo_alert_notifications(student_id, alert_week)"
        )
        connection.execute(
            """
            UPDATE interventions
            SET source_checkin_id = (
                    SELECT rs.student_checkin_id
                    FROM risk_snapshots AS rs
                    WHERE rs.snapshot_id = interventions.risk_snapshot_id
                ),
                generation_version = 1
            WHERE source_checkin_id IS NULL
              AND intervention_id IN (
                    SELECT MAX(i.intervention_id)
                    FROM interventions AS i
                    JOIN risk_snapshots AS rs
                      ON rs.snapshot_id = i.risk_snapshot_id
                    WHERE rs.student_checkin_id IS NOT NULL
                    GROUP BY rs.student_checkin_id
                )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "idx_interventions_checkin_version "
            "ON interventions(source_checkin_id, generation_version) "
            "WHERE source_checkin_id IS NOT NULL"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_paths_checkin "
            "ON learning_paths(source_checkin_id) "
            "WHERE source_checkin_id IS NOT NULL"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkin_feedback_student "
            "ON student_checkin_feedback(student_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_learning_paths_batch "
            "ON learning_paths(analysis_batch_id, student_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_batch "
            "ON microdegree_candidates(analysis_batch_id, candidate_code)"
        )
        connection.commit()
    return path


class DemoAlertNotificationDataStore:
    """실제 외부 발송 없이 조기경보 안내의 시연 상태를 저장한다."""

    STATUSES: tuple[str, ...] = (
        "안내 발송",
        "학생 확인",
        "Kare 상담 시작",
        "체크인 완료",
    )

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = initialize_database(database_path)

    @staticmethod
    def _normalize_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        """SQLite 행을 화면과 서비스 계층에서 사용할 mapping으로 바꾼다."""

        return dict(row) if row is not None else None

    def get_notification(
        self,
        student_id: str,
        alert_week: int,
    ) -> dict[str, Any] | None:
        """학생·경보 주차의 시연용 안내를 한 건 조회한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM demo_alert_notifications
                WHERE student_id = ? AND alert_week = ?
                LIMIT 1
                """,
                (str(student_id), int(alert_week)),
            ).fetchone()
        return self._normalize_row(row)

    def get_latest_notification(
        self,
        student_id: str,
    ) -> dict[str, Any] | None:
        """학생에게 발송된 가장 최근 시연용 안내를 조회한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM demo_alert_notifications
                WHERE student_id = ?
                ORDER BY alert_week DESC, notification_id DESC
                LIMIT 1
                """,
                (str(student_id),),
            ).fetchone()
        return self._normalize_row(row)

    def create_notification(
        self,
        *,
        student_id: str,
        alert_week: int,
        risk_level: str,
        risk_score: float,
        primary_risk_type: str,
        subject: str,
        body: str,
        created_by: str,
    ) -> tuple[dict[str, Any], bool]:
        """학생·주차별 안내를 한 번만 만들고 기존 안내가 있으면 재사용한다.

        Returns:
            ``(notification, created)``. 동일 학생·주차의 중복 요청은
            기존 행과 ``False``를 반환한다.
        """

        normalized_student_id = str(student_id).strip()
        normalized_subject = str(subject).strip()
        normalized_body = str(body).strip()
        normalized_staff = str(created_by).strip()
        if not all(
            (
                normalized_student_id,
                normalized_subject,
                normalized_body,
                normalized_staff,
            )
        ):
            raise ValueError("시연용 안내의 필수 값은 비어 있을 수 없습니다.")
        with _connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO demo_alert_notifications (
                    student_id, alert_week, risk_level, risk_score,
                    primary_risk_type, subject, body, created_by, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_student_id,
                    int(alert_week),
                    str(risk_level).strip(),
                    float(risk_score),
                    str(primary_risk_type).strip(),
                    normalized_subject,
                    normalized_body,
                    normalized_staff,
                    self.STATUSES[0],
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                """
                SELECT *
                FROM demo_alert_notifications
                WHERE student_id = ? AND alert_week = ?
                LIMIT 1
                """,
                (normalized_student_id, int(alert_week)),
            ).fetchone()
            connection.commit()
        notification = self._normalize_row(row)
        if notification is None:
            raise RuntimeError("시연용 안내를 저장하지 못했습니다.")
        return notification, created

    def advance_status(
        self,
        student_id: str,
        target_status: str,
        *,
        checkin_id: int | None = None,
    ) -> dict[str, Any] | None:
        """최근 안내 상태를 뒤로 되돌리지 않고 다음 단계로 진행한다.

        안내가 없는 학생은 ``None``을 반환한다. 학생 화면을 직접 연 경우에도
        기존 경보 안내가 없으면 새 안내를 임의로 만들지 않는다.
        """

        if target_status not in self.STATUSES:
            raise ValueError(f"지원하지 않는 안내 상태입니다: {target_status}")
        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM demo_alert_notifications
                WHERE student_id = ?
                ORDER BY alert_week DESC, notification_id DESC
                LIMIT 1
                """,
                (str(student_id),),
            ).fetchone()
            if row is None:
                return None
            current_status = str(row["status"])
            if current_status not in self.STATUSES:
                raise ValueError(f"저장된 안내 상태가 잘못되었습니다: {current_status}")
            current_index = self.STATUSES.index(current_status)
            target_index = self.STATUSES.index(target_status)
            next_status = (
                target_status if target_index > current_index else current_status
            )
            viewed_at = "CURRENT_TIMESTAMP" if target_index >= 1 else "student_viewed_at"
            started_at = "CURRENT_TIMESTAMP" if target_index >= 2 else "kare_started_at"
            completed_at = "CURRENT_TIMESTAMP" if target_index >= 3 else "checkin_completed_at"
            connection.execute(
                f"""
                UPDATE demo_alert_notifications
                SET status = ?,
                    student_viewed_at = COALESCE(student_viewed_at, {viewed_at}),
                    kare_started_at = COALESCE(kare_started_at, {started_at}),
                    checkin_completed_at = COALESCE(checkin_completed_at, {completed_at}),
                    source_checkin_id = COALESCE(source_checkin_id, ?)
                WHERE notification_id = ?
                """,
                (
                    next_status,
                    int(checkin_id) if checkin_id is not None else None,
                    int(row["notification_id"]),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM demo_alert_notifications WHERE notification_id = ?",
                (int(row["notification_id"]),),
            ).fetchone()
            connection.commit()
        return self._normalize_row(updated)


class DemoDataStore:
    """SQLite 운영기록을 백업하고 시연 초기 상태로 되돌린다."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = initialize_database(database_path)

    def get_record_counts(self) -> dict[str, int]:
        """초기화 대상 테이블별 현재 기록 수를 반환한다."""

        with _connect(self.database_path) as connection:
            return {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) AS count FROM {table}"
                    ).fetchone()["count"]
                )
                for table in DEMO_WORKFLOW_TABLES
            }

    def create_backup(self, backup_dir: str | Path | None = None) -> Path:
        """SQLite online backup API로 초기화 직전 복구본을 생성한다.

        Parameters:
            backup_dir: 백업 파일을 둘 디렉터리. 미지정 시 DB 옆 backups.

        Returns:
            생성된 SQLite 백업 파일의 절대 경로.

        Assumptions:
            이 메서드는 원본 DB를 변경하지 않는다.
        """

        destination_dir = (
            Path(backup_dir)
            if backup_dir is not None
            else self.database_path.parent / "backups"
        ).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = destination_dir / (
            f"{self.database_path.stem}_before_demo_reset_{timestamp}.db"
        )
        with _connect(self.database_path) as source:
            with sqlite3.connect(destination) as target:
                source.backup(target)
        return destination.resolve()

    def reset_workflow_records(self) -> dict[str, int]:
        """synthetic CSV는 건드리지 않고 SQLite 시연 기록만 삭제한다.

        Returns:
            삭제 전 테이블별 기록 수.

        Assumptions:
            외래키 자식 테이블을 부모보다 먼저 비우며,
            시연 재현을 위해 AUTOINCREMENT 순번도 함께 초기화한다.
        """

        counts = self.get_record_counts()
        with _connect(self.database_path) as connection:
            for table in DEMO_WORKFLOW_TABLES:
                connection.execute(f"DELETE FROM {table}")
            placeholders = ", ".join("?" for _ in DEMO_WORKFLOW_TABLES)
            connection.execute(
                f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})",
                DEMO_WORKFLOW_TABLES,
            )
            connection.commit()
        return counts

    def backup_and_reset(
        self,
        backup_dir: str | Path | None = None,
    ) -> tuple[Path, dict[str, int]]:
        """복구본 생성에 성공한 후에만 시연 기록을 초기화한다."""

        backup_path = self.create_backup(backup_dir)
        deleted_counts = self.reset_workflow_records()
        return backup_path, deleted_counts


class InterventionDataStore:
    """SQLite에 개입 추천, 검토 상태, 학생 피드백을 영속화한다."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = initialize_database(database_path)

    def save_recommendation_batch(
        self,
        student_id: str,
        recommendations: Sequence[Mapping[str, Any]],
        batch_id: str | None = None,
    ) -> tuple[str, dict[str, int]]:
        """한 번에 생성된 추천 후보를 저장하고 target별 DB ID를 반환한다.

        Parameters:
            student_id: synthetic 학생 식별자.
            recommendations: target_id, score, rank, reason을 가진 추천 목록.
            batch_id: 재현 테스트 등에 사용할 선택적 배치 ID.

        Returns:
            생성된 배치 ID와 target_id별 recommendation_id mapping.

        Assumptions:
            후보 ID 검증과 중복 제거는 상위 추천 서비스에서도 수행한다.
        """

        normalized_student_id = str(student_id).strip()
        if not normalized_student_id:
            raise ValueError("학생 ID는 비어 있을 수 없습니다.")
        if not recommendations:
            raise ValueError("저장할 추천이 없습니다.")
        active_batch_id = batch_id or uuid.uuid4().hex
        target_ids = [str(item["target_id"]).strip() for item in recommendations]
        if any(not target_id for target_id in target_ids):
            raise ValueError("추천 target_id는 비어 있을 수 없습니다.")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("한 추천 배치에 중복 target_id를 저장할 수 없습니다.")

        recommendation_ids: dict[str, int] = {}
        with _connect(self.database_path) as connection:
            for item, target_id in zip(recommendations, target_ids):
                cursor = connection.execute(
                    """
                    INSERT INTO recommendations (
                        batch_id, student_id, recommendation_type,
                        target_id, score, rank, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        active_batch_id,
                        normalized_student_id,
                        str(item.get("recommendation_type", "support_program")),
                        target_id,
                        float(item["score"]),
                        int(item["rank"]),
                        str(item["reason"]),
                    ),
                )
                recommendation_ids[target_id] = int(cursor.lastrowid)
            connection.commit()
        return active_batch_id, recommendation_ids

    def save_risk_snapshot(
        self,
        student_id: str,
        snapshot: Mapping[str, Any],
    ) -> int:
        """개입 생성 시점의 위험 결과를 변경 불가 스냅샷으로 저장한다.

        Parameters:
            student_id: 개입 대상 학생 ID.
            snapshot: 위험엔진이 산출한 주차·5개 영역·종합 점수와 분류.

        Returns:
            새로 생성된 snapshot_id.

        Assumptions:
            민감정보와 원천 활동값은 저장하지 않고 산출 결과만 고정한다.
        """

        normalized_student_id = str(student_id).strip()
        if not normalized_student_id:
            raise ValueError("학생 ID는 비어 있을 수 없습니다.")
        snapshot_student_id = str(snapshot.get("student_id", "")).strip()
        if snapshot_student_id and snapshot_student_id != normalized_student_id:
            raise ValueError("다른 학생의 위험 스냅샷을 저장할 수 없습니다.")

        required_fields = {
            "week",
            "attendance_risk",
            "engagement_risk",
            "achievement_risk",
            "major_adaptation_risk",
            "career_risk",
            "overall_risk",
            "risk_level",
            "primary_risk_type",
            "secondary_risk_types",
        }
        missing = required_fields.difference(snapshot)
        if missing:
            raise ValueError(f"위험 스냅샷 필드가 누락되었습니다: {sorted(missing)}")

        try:
            week = int(snapshot["week"])
        except (TypeError, ValueError) as error:
            raise ValueError("위험 스냅샷 주차는 양의 정수여야 합니다.") from error
        if week < 1:
            raise ValueError("위험 스냅샷 주차는 양의 정수여야 합니다.")

        score_fields = (
            "attendance_risk",
            "engagement_risk",
            "achievement_risk",
            "major_adaptation_risk",
            "career_risk",
            "overall_risk",
        )
        scores: dict[str, float] = {}
        for field in score_fields:
            try:
                score = float(snapshot[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} 점수는 0~100 숫자여야 합니다.") from error
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f"{field} 점수는 0~100 범위여야 합니다.")
            scores[field] = score

        risk_level = str(snapshot["risk_level"]).strip()
        primary_risk_type = str(snapshot["primary_risk_type"]).strip()
        if not risk_level or not primary_risk_type:
            raise ValueError("위험등급과 주요 위험유형은 비어 있을 수 없습니다.")
        secondary_value = snapshot["secondary_risk_types"]
        if isinstance(secondary_value, str):
            secondary_risk_types = secondary_value.strip()
        elif isinstance(secondary_value, Sequence):
            secondary_risk_types = ", ".join(
                str(item).strip() for item in secondary_value if str(item).strip()
            )
        else:
            secondary_risk_types = str(secondary_value).strip()

        checkin_source = str(snapshot.get("checkin_source", "repository")).strip()
        checkin_source = checkin_source or "repository"
        raw_checkin_id = snapshot.get("student_checkin_id")
        student_checkin_id = (
            None
            if raw_checkin_id is None or bool(pd.isna(raw_checkin_id))
            else int(raw_checkin_id)
        )

        with _connect(self.database_path) as connection:
            if student_checkin_id is not None:
                checkin = connection.execute(
                    "SELECT student_id FROM student_checkins WHERE checkin_id = ?",
                    (student_checkin_id,),
                ).fetchone()
                if checkin is None:
                    raise ValueError(
                        f"존재하지 않는 학생 체크인 ID입니다: {student_checkin_id}"
                    )
                if str(checkin["student_id"]) != normalized_student_id:
                    raise ValueError("다른 학생의 체크인을 스냅샷에 연결할 수 없습니다.")
            cursor = connection.execute(
                """
                INSERT INTO risk_snapshots (
                    student_id, week, attendance_risk, engagement_risk,
                    achievement_risk, major_adaptation_risk, career_risk,
                    overall_risk, risk_level, primary_risk_type,
                    secondary_risk_types, checkin_source, student_checkin_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_student_id,
                    week,
                    scores["attendance_risk"],
                    scores["engagement_risk"],
                    scores["achievement_risk"],
                    scores["major_adaptation_risk"],
                    scores["career_risk"],
                    scores["overall_risk"],
                    risk_level,
                    primary_risk_type,
                    secondary_risk_types,
                    checkin_source,
                    student_checkin_id,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_risk_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        """위험 스냅샷 ID 한 건을 mapping으로 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM risk_snapshots WHERE snapshot_id = ?",
                (int(snapshot_id),),
            ).fetchone()
        return dict(row) if row else None

    def list_risk_snapshots(self, student_id: str | None = None) -> pd.DataFrame:
        """선택 학생 또는 전체 위험 스냅샷을 최근 순으로 반환한다."""

        query = "SELECT * FROM risk_snapshots"
        parameters: tuple[str, ...] = ()
        if student_id is not None:
            query += " WHERE student_id = ?"
            parameters = (str(student_id),)
        query += " ORDER BY created_at DESC, snapshot_id DESC"
        with _connect(self.database_path) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def create_intervention(
        self,
        student_id: str,
        recommendation_batch_id: str,
        recommended_program_ids: Sequence[str],
        risk_snapshot_id: int | None = None,
        source_checkin_id: int | None = None,
        generation_version: int = 1,
        initial_status: str = "추천 생성",
    ) -> int:
        """추천 배치에 대응하는 교직원 개입 기록을 생성한다."""

        normalized_student_id = str(student_id).strip()
        if int(generation_version) < 1:
            raise ValueError("추천 생성 버전은 1 이상이어야 합니다.")
        with _connect(self.database_path) as connection:
            if risk_snapshot_id is not None:
                snapshot = connection.execute(
                    "SELECT student_id FROM risk_snapshots WHERE snapshot_id = ?",
                    (int(risk_snapshot_id),),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        f"존재하지 않는 위험 스냅샷 ID입니다: {risk_snapshot_id}"
                    )
                if str(snapshot["student_id"]) != normalized_student_id:
                    raise ValueError("다른 학생의 위험 스냅샷을 개입에 연결할 수 없습니다.")
            if source_checkin_id is not None:
                checkin = connection.execute(
                    "SELECT student_id FROM student_checkins WHERE checkin_id = ?",
                    (int(source_checkin_id),),
                ).fetchone()
                if checkin is None:
                    raise ValueError(
                        f"존재하지 않는 학생 체크인 ID입니다: {source_checkin_id}"
                    )
                if str(checkin["student_id"]) != normalized_student_id:
                    raise ValueError("다른 학생의 체크인을 개입에 연결할 수 없습니다.")
            cursor = connection.execute(
                """
                INSERT INTO interventions (
                    student_id, risk_snapshot_id, recommendation_batch_id,
                    source_checkin_id, generation_version,
                    recommended_action, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_student_id,
                    risk_snapshot_id,
                    str(recommendation_batch_id),
                    source_checkin_id,
                    int(generation_version),
                    json.dumps(list(recommended_program_ids), ensure_ascii=False),
                    str(initial_status),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def update_intervention(
        self,
        intervention_id: int,
        status: str,
        staff_action: str = "",
        staff_note: str = "",
    ) -> None:
        """기존 개입의 현재 상태와 교직원 기록을 갱신한다."""

        with _connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE interventions
                SET status = ?, staff_action = ?, staff_note = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE intervention_id = ?
                """,
                (str(status), str(staff_action), str(staff_note), intervention_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"존재하지 않는 개입 ID입니다: {intervention_id}")
            connection.commit()

    def update_integrated_recommendation_review(
        self,
        intervention_id: int,
        intervention_status: str,
        staff_action: str,
        staff_note: str,
        learning_path_status: str,
    ) -> bool:
        """비교과 추천과 교과 학습경로의 검토 상태를 함께 저장한다.

        Parameters:
            intervention_id: 비교과 추천에 연결된 지원 ID.
            intervention_status: 기존 개입 워크플로에 저장할 상태.
            staff_action: 교직원 검토 행위.
            staff_note: 비교과·교과에 공통으로 남길 검토 메모.
            learning_path_status: 학습경로에 저장할 검토 상태.

        Returns:
            같은 체크인의 학습경로까지 함께 갱신했는지 여부.

        Assumptions:
            학습경로는 개입의 ``source_checkin_id``와 같은 체크인에
            연결되어 있을 때만 통합 검토 대상이다.
        """

        with _connect(self.database_path) as connection:
            intervention = connection.execute(
                """
                SELECT source_checkin_id
                FROM interventions
                WHERE intervention_id = ?
                """,
                (int(intervention_id),),
            ).fetchone()
            if intervention is None:
                raise ValueError(
                    f"존재하지 않는 개입 ID입니다: {intervention_id}"
                )
            connection.execute(
                """
                UPDATE interventions
                SET status = ?, staff_action = ?, staff_note = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE intervention_id = ?
                """,
                (
                    str(intervention_status),
                    str(staff_action),
                    str(staff_note),
                    int(intervention_id),
                ),
            )
            source_checkin_id = intervention["source_checkin_id"]
            learning_path_updated = False
            if source_checkin_id is not None:
                cursor = connection.execute(
                    """
                    UPDATE learning_paths
                    SET review_status = ?, review_note = ?,
                        reviewed_at = CURRENT_TIMESTAMP
                    WHERE source_checkin_id = ?
                    """,
                    (
                        str(learning_path_status),
                        str(staff_note),
                        int(source_checkin_id),
                    ),
                )
                learning_path_updated = cursor.rowcount > 0
            connection.commit()
        return learning_path_updated

    def get_intervention(self, intervention_id: int) -> dict[str, Any] | None:
        """개입 ID 한 건을 mapping으로 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM interventions WHERE intervention_id = ?",
                (intervention_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_intervention(self, student_id: str) -> dict[str, Any] | None:
        """학생의 가장 최근 개입 기록을 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM interventions
                WHERE student_id = ?
                ORDER BY intervention_id DESC
                LIMIT 1
                """,
                (str(student_id),),
            ).fetchone()
        return dict(row) if row else None

    def get_intervention_by_checkin(
        self, checkin_id: int
    ) -> dict[str, Any] | None:
        """학생 체크인에 연결된 최신 추천·개입 버전을 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM interventions
                WHERE source_checkin_id = ?
                ORDER BY generation_version DESC, intervention_id DESC
                LIMIT 1
                """,
                (int(checkin_id),),
            ).fetchone()
        return dict(row) if row else None

    def list_interventions(self) -> pd.DataFrame:
        """교직원 관리 화면에 사용할 전체 개입 이력을 반환한다."""

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM interventions
                ORDER BY updated_at DESC, intervention_id DESC
                """
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def list_latest_interventions(self) -> pd.DataFrame:
        """학생별 가장 최근에 변경된 개입 상태를 한 행씩 반환한다."""

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM (
                    SELECT i.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY i.student_id
                               ORDER BY i.updated_at DESC,
                                        i.intervention_id DESC
                           ) AS row_number
                    FROM interventions AS i
                )
                WHERE row_number = 1
                ORDER BY updated_at DESC, intervention_id DESC
                """
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record.pop("row_number", None)
            records.append(record)
        return pd.DataFrame(records)

    def list_recommendations(self, batch_id: str) -> pd.DataFrame:
        """한 배치의 추천 후보를 순위 순서로 반환한다."""

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM recommendations
                WHERE batch_id = ?
                ORDER BY rank, recommendation_id
                """,
                (str(batch_id),),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def save_support_plan(
        self,
        intervention_id: int,
        student_id: str,
        assigned_staff: str,
        planned_contact_date: str,
        next_action: str,
        plan_note: str,
        items: Sequence[Mapping[str, Any]],
    ) -> int:
        """추천 검토 결과를 실행 가능한 지원계획과 프로그램 항목으로 저장한다.

        Parameters:
            intervention_id: 원본 추천 배치가 연결된 지원 ID.
            student_id: 지원 대상 학생 ID.
            assigned_staff: 후속 조치 담당자 또는 담당 부서.
            planned_contact_date: ISO 형식 연락 예정일.
            next_action: 확정 후 가장 먼저 실행할 조치.
            plan_note: 선택·교체 검토 메모.
            items: 실제 지원할 DB 프로그램 목록.

        Returns:
            생성 또는 갱신된 support_plan_id.

        Assumptions:
            프로그램 master 존재와 교체 규칙은 상위 service가 검증한다.
        """

        if not items:
            raise ValueError("지원계획에는 최소 한 개의 프로그램이 필요합니다.")
        program_ids = [str(item["program_id"]).strip() for item in items]
        if any(not program_id for program_id in program_ids):
            raise ValueError("지원계획 프로그램 ID는 비어 있을 수 없습니다.")
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("지원계획에 중복 프로그램을 저장할 수 없습니다.")

        with _connect(self.database_path) as connection:
            intervention = connection.execute(
                "SELECT student_id FROM interventions WHERE intervention_id = ?",
                (int(intervention_id),),
            ).fetchone()
            if intervention is None:
                raise ValueError(f"존재하지 않는 개입 ID입니다: {intervention_id}")
            if str(intervention["student_id"]) != str(student_id):
                raise ValueError("다른 학생의 개입에 지원계획을 연결할 수 없습니다.")
            existing = connection.execute(
                "SELECT support_plan_id FROM support_plans WHERE intervention_id = ?",
                (int(intervention_id),),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO support_plans (
                        intervention_id, student_id, assigned_staff,
                        planned_contact_date, next_action, plan_note
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(intervention_id),
                        str(student_id),
                        str(assigned_staff),
                        str(planned_contact_date),
                        str(next_action),
                        str(plan_note).strip(),
                    ),
                )
                support_plan_id = int(cursor.lastrowid)
            else:
                support_plan_id = int(existing["support_plan_id"])
                connection.execute(
                    """
                    UPDATE support_plans
                    SET assigned_staff = ?, planned_contact_date = ?,
                        next_action = ?, plan_note = ?, status = '계획 확정',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE support_plan_id = ?
                    """,
                    (
                        str(assigned_staff),
                        str(planned_contact_date),
                        str(next_action),
                        str(plan_note).strip(),
                        support_plan_id,
                    ),
                )

            placeholders = ", ".join("?" for _ in program_ids)
            connection.execute(
                f"DELETE FROM support_plan_items "
                f"WHERE support_plan_id = ? AND program_id NOT IN ({placeholders})",
                (support_plan_id, *program_ids),
            )
            for item in items:
                connection.execute(
                    """
                    INSERT INTO support_plan_items (
                        support_plan_id, program_id, source_recommendation_id,
                        selection_type, replaced_program_id, replacement_reason
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(support_plan_id, program_id) DO UPDATE SET
                        source_recommendation_id = excluded.source_recommendation_id,
                        selection_type = excluded.selection_type,
                        replaced_program_id = excluded.replaced_program_id,
                        replacement_reason = excluded.replacement_reason,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        support_plan_id,
                        str(item["program_id"]),
                        item.get("source_recommendation_id"),
                        str(item["selection_type"]),
                        item.get("replaced_program_id"),
                        str(item.get("replacement_reason", "")).strip(),
                    ),
                )
            connection.commit()
        return support_plan_id

    def get_support_plan(self, intervention_id: int) -> dict[str, Any] | None:
        """개입에 연결된 현재 지원계획을 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM support_plans WHERE intervention_id = ?",
                (int(intervention_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_support_plan_items(self, intervention_id: int) -> pd.DataFrame:
        """개입 지원계획의 프로그램별 실행 상태를 반환한다."""

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT item.*
                FROM support_plan_items AS item
                JOIN support_plans AS plan
                  ON plan.support_plan_id = item.support_plan_id
                WHERE plan.intervention_id = ?
                ORDER BY item.plan_item_id
                """,
                (int(intervention_id),),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def update_support_plan_item(
        self,
        plan_item_id: int,
        item_status: str,
        last_action: str = "",
        staff_note: str = "",
    ) -> None:
        """프로그램 하나의 실행 상태와 최신 조치를 갱신한다."""

        with _connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE support_plan_items
                SET item_status = ?, last_action = ?, staff_note = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE plan_item_id = ?
                """,
                (
                    str(item_status),
                    str(last_action).strip(),
                    str(staff_note).strip(),
                    int(plan_item_id),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"존재하지 않는 지원계획 항목 ID입니다: {plan_item_id}"
                )
            connection.commit()

    def save_feedback(
        self,
        student_id: str,
        recommendation_id: int,
        feedback_type: str,
        comment: str = "",
    ) -> int:
        """학생에게 생성된 추천에만 피드백을 연결해 저장한다."""

        with _connect(self.database_path) as connection:
            recommendation = connection.execute(
                """
                SELECT student_id FROM recommendations
                WHERE recommendation_id = ?
                """,
                (recommendation_id,),
            ).fetchone()
            if recommendation is None:
                raise ValueError(
                    f"존재하지 않는 추천 ID입니다: {recommendation_id}"
                )
            if str(recommendation["student_id"]) != str(student_id):
                raise ValueError("다른 학생의 추천에는 피드백을 저장할 수 없습니다.")
            cursor = connection.execute(
                """
                INSERT INTO feedback (
                    student_id, recommendation_id, feedback_type, comment
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(student_id),
                    recommendation_id,
                    str(feedback_type),
                    str(comment).strip(),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_feedback(self, student_id: str | None = None) -> pd.DataFrame:
        """추천 target ID와 함께 학생 피드백 이력을 반환한다."""

        query = """
            SELECT f.feedback_id, f.student_id, f.recommendation_id,
                   r.recommendation_type, r.target_id,
                   f.feedback_type, f.comment, f.created_at
            FROM feedback AS f
            JOIN recommendations AS r
              ON r.recommendation_id = f.recommendation_id
        """
        parameters: tuple[str, ...] = ()
        if student_id is not None:
            query += " WHERE f.student_id = ?"
            parameters = (str(student_id),)
        query += " ORDER BY f.created_at DESC, f.feedback_id DESC"
        with _connect(self.database_path) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return pd.DataFrame([dict(row) for row in rows])


class StudentCheckinDataStore:
    """학생이 직접 제출한 체크인과 구조화 분석을 SQLite에 저장한다."""

    LIKERT_FIELDS = (
        "major_interest",
        "major_satisfaction",
        "major_continuation_intent",
        "learning_difficulty",
        "career_clarity",
        "consultation_intent",
    )

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = initialize_database(database_path)

    def save_checkin(
        self,
        student_id: str,
        values: Mapping[str, Any],
    ) -> int:
        """검증한 학생 자기보고 체크인을 저장하고 ID를 반환한다.

        Parameters:
            student_id: 포털 로그인으로 대체 가능한 학생 식별자.
            values: 여섯 Likert 응답과 관심분야·진로·자유서술 값.

        Returns:
            생성된 checkin_id.

        Assumptions:
            학생 존재 여부는 상위 StudentViewService에서 Repository로 확인한다.
        """

        normalized_student_id = str(student_id).strip()
        if not normalized_student_id:
            raise ValueError("학생 ID는 비어 있을 수 없습니다.")
        response_mode = str(values.get("response_mode", "scaled")).strip()
        if response_mode not in {"scaled", "narrative"}:
            raise ValueError(f"지원하지 않는 체크인 방식입니다: {response_mode}")
        likert_values: dict[str, int | None] = {}
        for field in self.LIKERT_FIELDS:
            raw_value = values.get(field)
            if response_mode == "narrative" and raw_value is None:
                likert_values[field] = None
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} 응답은 1~5 정수여야 합니다.") from error
            if not 1 <= value <= 5:
                raise ValueError(f"{field} 응답은 1~5 범위여야 합니다.")
            likert_values[field] = value

        concern = str(values.get("natural_language_concern", "")).strip()
        if response_mode == "narrative" and not concern:
            raise ValueError("자연어 체크인에는 학생이 작성한 이야기가 필요합니다.")
        week = int(values["week"]) if values.get("week") is not None else None
        if response_mode == "narrative" and week is not None:
            raise ValueError("자연어 체크인은 위험분석 주차로 저장할 수 없습니다.")

        interest_fields = values.get("interest_fields", "")
        if isinstance(interest_fields, str):
            normalized_interests = [
                item.strip() for item in interest_fields.split("|") if item.strip()
            ]
        else:
            normalized_interests = [
                str(item).strip() for item in interest_fields if str(item).strip()
            ]
        semantic_states = validate_semantic_state(values.get("semantic_states"))
        with _connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO student_checkins (
                    student_id, week, major_interest, major_satisfaction,
                    major_continuation_intent, learning_difficulty,
                    career_clarity, consultation_intent, interest_fields,
                    desired_job, consultation_requested, explore_other_fields,
                    natural_language_concern, response_mode, semantic_states
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_student_id,
                    week,
                    likert_values["major_interest"],
                    likert_values["major_satisfaction"],
                    likert_values["major_continuation_intent"],
                    likert_values["learning_difficulty"],
                    likert_values["career_clarity"],
                    likert_values["consultation_intent"],
                    json.dumps(normalized_interests, ensure_ascii=False),
                    str(values.get("desired_job", "")).strip(),
                    int(bool(values.get("consultation_requested", False))),
                    int(bool(values.get("explore_other_fields", False))),
                    concern,
                    response_mode,
                    json.dumps(semantic_states, ensure_ascii=False),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def save_analysis(
        self,
        checkin_id: int,
        analysis: Mapping[str, Any],
        provider_name: str,
        fallback_used: bool,
    ) -> int:
        """JSON Schema 검증을 마친 체크인 분석을 원문 응답 없이 저장한다."""

        required = {
            "major_concern",
            "learning_difficulty",
            "career_uncertainty",
            "interests",
            "desired_jobs",
            "support_needs",
            "summary",
        }
        missing = required.difference(analysis)
        if missing:
            raise ValueError(f"체크인 분석 필드가 누락되었습니다: {sorted(missing)}")
        with _connect(self.database_path) as connection:
            if connection.execute(
                "SELECT 1 FROM student_checkins WHERE checkin_id = ?",
                (int(checkin_id),),
            ).fetchone() is None:
                raise ValueError(f"존재하지 않는 체크인 ID입니다: {checkin_id}")
            cursor = connection.execute(
                """
                INSERT INTO student_checkin_analyses (
                    checkin_id, provider_name, fallback_used, major_concern,
                    learning_difficulty, career_uncertainty, interests,
                    desired_jobs, support_needs, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(checkin_id),
                    str(provider_name),
                    int(bool(fallback_used)),
                    int(bool(analysis["major_concern"])),
                    int(bool(analysis["learning_difficulty"])),
                    int(bool(analysis["career_uncertainty"])),
                    json.dumps(list(analysis["interests"]), ensure_ascii=False),
                    json.dumps(list(analysis["desired_jobs"]), ensure_ascii=False),
                    json.dumps(list(analysis["support_needs"]), ensure_ascii=False),
                    str(analysis["summary"]),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    @staticmethod
    def _decode_checkin_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        """SQLite JSON·boolean 값을 화면용 Python 값으로 변환한다."""

        if row is None:
            return None
        result = dict(row)
        result["interest_fields"] = json.loads(result["interest_fields"] or "[]")
        result["semantic_states"] = validate_semantic_state(
            json.loads(result.get("semantic_states") or "{}")
        )
        for field in ("interests", "desired_jobs", "support_needs"):
            value = result.get(field)
            result[field] = json.loads(value) if value else []
        for field in (
            "consultation_requested",
            "explore_other_fields",
            "fallback_used",
            "major_concern",
            "analysis_learning_difficulty",
            "career_uncertainty",
        ):
            if field in result and result[field] is not None:
                result[field] = bool(result[field])
        return result

    def get_latest_checkin(self, student_id: str) -> dict[str, Any] | None:
        """선택 학생의 가장 최근 체크인과 AI 분석을 함께 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT c.*,
                       a.analysis_id, a.provider_name, a.fallback_used,
                       a.major_concern,
                       a.learning_difficulty AS analysis_learning_difficulty,
                       a.career_uncertainty, a.interests, a.desired_jobs,
                       a.support_needs, a.summary,
                       a.created_at AS analysis_created_at
                FROM student_checkins AS c
                LEFT JOIN student_checkin_analyses AS a
                  ON a.checkin_id = c.checkin_id
                WHERE c.student_id = ?
                ORDER BY c.checkin_id DESC
                LIMIT 1
                """,
                (str(student_id),),
            ).fetchone()
        return self._decode_checkin_row(row)

    def list_checkins(self) -> pd.DataFrame:
        """학생·주차별 가장 최근에 제출된 체크인을 모두 반환한다.

        Returns:
            직접 제출 이력을 위험엔진 표준 컬럼으로 변환한 DataFrame.

        Assumptions:
            같은 학생이 같은 주차를 다시 제출하면 가장 큰 checkin_id를 유효한 응답으로 본다.
        """

        columns = [
            "checkin_id",
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
            "created_at",
        ]
        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT c.*
                FROM student_checkins AS c
                JOIN (
                    SELECT student_id, week, MAX(checkin_id) AS latest_checkin_id
                    FROM student_checkins
                    WHERE week IS NOT NULL AND response_mode = 'scaled'
                    GROUP BY student_id, week
                ) AS latest
                  ON latest.latest_checkin_id = c.checkin_id
                ORDER BY c.student_id, c.week, c.checkin_id
                """
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            interests = json.loads(record.get("interest_fields") or "[]")
            record["interest_fields"] = "|".join(
                str(item) for item in interests if str(item).strip()
            )
            records.append({column: record.get(column) for column in columns})
        return pd.DataFrame(records, columns=columns)

    def list_latest_checkins(self) -> pd.DataFrame:
        """학생별 가장 최근 직접 제출 체크인을 위험엔진 표준 형식으로 반환한다.

        Returns:
            학생당 최대 한 행의 체크인 DataFrame. 관심분야는 파이프 구분 문자열이다.

        Assumptions:
            동일 학생의 최신 checkin_id가 현재 유효한 자기보고 응답이다.
        """

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT c.*
                FROM student_checkins AS c
                JOIN (
                    SELECT student_id, MAX(checkin_id) AS latest_checkin_id
                    FROM student_checkins
                    WHERE response_mode = 'scaled'
                    GROUP BY student_id
                ) AS latest
                  ON latest.latest_checkin_id = c.checkin_id
                ORDER BY c.student_id
                """
            ).fetchall()
        columns = [
            "checkin_id",
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
            "created_at",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            interests = json.loads(record.get("interest_fields") or "[]")
            record["interest_fields"] = "|".join(
                str(item) for item in interests if str(item).strip()
            )
            records.append({column: record.get(column) for column in columns})
        return pd.DataFrame(records, columns=columns)

    def get_revision(self) -> int:
        """Streamlit 위험분석 캐시 무효화에 사용할 최신 체크인 ID를 반환한다."""

        with _connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(checkin_id), 0) AS revision "
                "FROM student_checkins WHERE response_mode = 'scaled'"
            ).fetchone()
        return int(row["revision"]) if row is not None else 0

    def save_analysis_feedback(
        self,
        checkin_id: int,
        student_id: str,
        feedback_type: str,
        comment: str = "",
    ) -> int:
        """본인 체크인의 AI 이해 결과에 대한 학생 피드백을 저장한다."""

        with _connect(self.database_path) as connection:
            checkin = connection.execute(
                "SELECT student_id FROM student_checkins WHERE checkin_id = ?",
                (int(checkin_id),),
            ).fetchone()
            if checkin is None:
                raise ValueError(f"존재하지 않는 체크인 ID입니다: {checkin_id}")
            if str(checkin["student_id"]) != str(student_id):
                raise ValueError("다른 학생의 체크인에는 피드백을 저장할 수 없습니다.")
            cursor = connection.execute(
                """
                INSERT INTO student_checkin_feedback (
                    checkin_id, student_id, feedback_type, comment
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    int(checkin_id),
                    str(student_id),
                    str(feedback_type),
                    str(comment).strip(),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_analysis_feedback(
        self, student_id: str | None = None
    ) -> pd.DataFrame:
        """선택 학생 또는 전체 AI 이해 결과 피드백을 반환한다."""

        query = "SELECT * FROM student_checkin_feedback"
        parameters: tuple[str, ...] = ()
        if student_id is not None:
            query += " WHERE student_id = ?"
            parameters = (str(student_id),)
        query += " ORDER BY created_at DESC, feedback_id DESC"
        with _connect(self.database_path) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return pd.DataFrame([dict(row) for row in rows])


class LearningPathDataStore:
    """개인 학습경로와 교육과정 개발 후보 분석을 SQLite에 저장한다."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = initialize_database(database_path)

    def save_analysis(
        self,
        paths: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        analysis_batch_id: str | None = None,
        source_checkin_id: int | None = None,
    ) -> str:
        """한 번의 전체 경로·후보 분석 결과를 원자적으로 저장한다.

        Parameters:
            paths: 학생별 경로와 DB 교과목 순서를 담은 mapping 목록.
            candidates: 코드가 조건을 검증한 교육과정 후보 목록.
            analysis_batch_id: 재현 테스트용 선택적 배치 ID.

        Returns:
            저장된 분석 배치 ID.

        Assumptions:
            후보의 교과목 ID는 Repository master 검증을 이미 통과했다.
        """

        if not paths:
            raise ValueError("저장할 학습경로가 없습니다.")
        if source_checkin_id is not None:
            existing = self.get_learning_path_by_checkin(source_checkin_id)
            if not existing.empty:
                return str(existing.iloc[0]["analysis_batch_id"])
            if len(paths) != 1:
                raise ValueError(
                    "학생 체크인에는 하나의 개인 학습경로만 연결할 수 있습니다."
                )
        batch_id = analysis_batch_id or uuid.uuid4().hex
        student_ids = [str(path["student_id"]) for path in paths]
        if len(student_ids) != len(set(student_ids)):
            raise ValueError("한 분석 배치에 학생별 경로는 하나만 저장할 수 있습니다.")
        with _connect(self.database_path) as connection:
            if source_checkin_id is not None:
                checkin = connection.execute(
                    "SELECT student_id FROM student_checkins WHERE checkin_id = ?",
                    (int(source_checkin_id),),
                ).fetchone()
                if checkin is None:
                    raise ValueError(
                        f"존재하지 않는 학생 체크인 ID입니다: {source_checkin_id}"
                    )
                if str(checkin["student_id"]) != student_ids[0]:
                    raise ValueError(
                        "다른 학생의 체크인을 학습경로에 연결할 수 없습니다."
                    )
            for path in paths:
                cursor = connection.execute(
                    """
                    INSERT INTO learning_paths (
                        analysis_batch_id, student_id, source_checkin_id,
                        path_name, related_job, competencies, reason,
                        career_connection, provider_name, fallback_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        str(path["student_id"]),
                        source_checkin_id,
                        str(path["path_name"]),
                        str(path.get("related_job", "")),
                        json.dumps(
                            list(path.get("competencies", [])),
                            ensure_ascii=False,
                        ),
                        str(path["reason"]),
                        str(path.get("career_connection", "")),
                        str(path.get("provider_name", "")),
                        int(bool(path.get("fallback_used", False))),
                    ),
                )
                path_id = int(cursor.lastrowid)
                for course in path["courses"]:
                    connection.execute(
                        """
                        INSERT INTO learning_path_courses (
                            path_id, course_id, sequence, score, reason,
                            selection_reason
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            path_id,
                            str(course["course_id"]),
                            int(course["sequence"]),
                            float(course["score"]),
                            str(course["reason"]),
                            str(course.get("selection_reason", "")),
                        ),
                    )
            for candidate in candidates:
                connection.execute(
                    """
                    INSERT INTO microdegree_candidates (
                        analysis_batch_id, candidate_code, candidate_name,
                        student_count, department_count, departments, course_ids,
                        course_names, competencies, related_jobs, student_ids,
                        average_similarity, priority, demand_share,
                        description, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        str(candidate["candidate_code"]),
                        str(candidate["candidate_name"]),
                        int(candidate["student_count"]),
                        int(candidate["department_count"]),
                        json.dumps(
                            list(candidate["departments"]), ensure_ascii=False
                        ),
                        json.dumps(
                            list(candidate["course_ids"]), ensure_ascii=False
                        ),
                        json.dumps(
                            list(candidate["course_names"]), ensure_ascii=False
                        ),
                        json.dumps(
                            list(candidate["competencies"]), ensure_ascii=False
                        ),
                        json.dumps(
                            list(candidate["related_jobs"]), ensure_ascii=False
                        ),
                        json.dumps(
                            list(candidate["student_ids"]), ensure_ascii=False
                        ),
                        float(candidate["average_similarity"]),
                        str(candidate.get("priority", "재분석 필요")),
                        float(candidate.get("demand_share", 0)),
                        str(candidate["description"]),
                        str(candidate["status"]),
                    ),
                )
            connection.commit()
        return batch_id

    def get_learning_path_by_checkin(self, checkin_id: int) -> pd.DataFrame:
        """학생 체크인에 연결된 저장 학습경로와 교과목 순서를 반환한다."""

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT p.path_id, p.analysis_batch_id, p.student_id,
                       p.source_checkin_id, p.path_name, p.related_job,
                       p.competencies, p.reason AS path_reason,
                       p.career_connection, p.provider_name, p.fallback_used,
                       p.review_status, p.review_note, p.reviewed_at,
                       c.course_id, c.sequence, c.score,
                       c.reason AS course_role, c.selection_reason,
                       p.created_at
                FROM learning_paths AS p
                JOIN learning_path_courses AS c ON c.path_id = p.path_id
                WHERE p.source_checkin_id = ?
                ORDER BY c.sequence, c.course_id
                """,
                (int(checkin_id),),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["competencies"] = json.loads(
                record.get("competencies") or "[]"
            )
            record["fallback_used"] = bool(record.get("fallback_used"))
            records.append(record)
        return pd.DataFrame(records)

    def list_learning_paths(self, analysis_batch_id: str) -> pd.DataFrame:
        """한 분석 배치의 학생별 경로와 교과목을 반환한다."""

        with _connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT p.path_id, p.analysis_batch_id, p.student_id,
                       p.path_name, p.related_job, p.competencies,
                       c.course_id, c.sequence, c.score, c.reason,
                       p.created_at
                FROM learning_paths AS p
                JOIN learning_path_courses AS c ON c.path_id = p.path_id
                WHERE p.analysis_batch_id = ?
                ORDER BY p.student_id, c.sequence
                """,
                (str(analysis_batch_id),),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def list_latest_candidates(self) -> pd.DataFrame:
        """가장 최근 분석 배치의 교육과정 후보를 반환한다."""

        with _connect(self.database_path) as connection:
            latest = connection.execute(
                """
                SELECT analysis_batch_id
                FROM microdegree_candidates
                ORDER BY candidate_id DESC
                LIMIT 1
                """
            ).fetchone()
            if latest is None:
                return pd.DataFrame()
            rows = connection.execute(
                """
                SELECT * FROM microdegree_candidates
                WHERE analysis_batch_id = ?
                ORDER BY student_count DESC, department_count DESC,
                         average_similarity DESC, candidate_code
                """,
                (str(latest["analysis_batch_id"]),),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])
