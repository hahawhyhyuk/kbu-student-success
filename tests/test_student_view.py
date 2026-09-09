"""학생용 체크인·추천·학습경로와 교직원 연계 테스트."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.ai import MockAIProvider
from src.checkin_semantic_state import (
    empty_semantic_state,
    update_semantic_state_from_review,
)
from src.course_recommender import CourseRecommender
from src.database import (
    InterventionDataStore,
    LearningPathDataStore,
    StudentCheckinDataStore,
)
from src.intervention_effect_service import InterventionEffectService
from src.intervention_recommender import InterventionRecommender, ProgramRecommendation
from src.checkin_progress import build_kare_context_progress_html
from src.repositories import get_default_repository
from src.risk_service import RiskAnalysisService
from src.similarity import TokenOverlapSimilarityBackend
from src.student_view_service import (
    StudentViewQueryService,
    StudentViewService,
    build_student_support_areas,
    describe_student_support_status,
    describe_student_analysis,
)
from src.ui import (
    build_kare_chat_bubble_html,
    build_kare_typing_indicator_html,
    calculate_chat_bubble_width_rem,
)
from src.utils import PROJECT_ROOT


STUDENT_VALUES = {
    "major_interest": 2,
    "major_satisfaction": 2,
    "major_continuation_intent": 2,
    "learning_difficulty": 4,
    "career_clarity": 2,
    "consultation_intent": 5,
    "interest_fields": ["경영·마케팅", "콘텐츠·디자인"],
    "desired_job": "디지털 마케팅",
    "consultation_requested": True,
    "explore_other_fields": True,
    "natural_language_concern": (
        "현재 전공이 맞는지 고민이고 콘텐츠와 마케팅 진로를 "
        "어떻게 준비할지 모르겠습니다."
    ),
}

NARRATIVE_STUDENT_VALUES = {
    "response_mode": "narrative",
    "major_interest": None,
    "major_satisfaction": None,
    "major_continuation_intent": None,
    "learning_difficulty": None,
    "career_clarity": None,
    "consultation_intent": None,
    "interest_fields": ["데이터·AI", "보건"],
    "desired_job": "데이터 분석",
    "consultation_requested": True,
    "explore_other_fields": False,
    "natural_language_concern": (
        "수업 말이 빨라고 외울 것이 많아 과제가 밀리고 있어요. "
        "데이터 분석 진로를 어떻게 준비할지 궁금해요."
    ),
}


class CountingMockAIProvider(MockAIProvider):
    """제출 전 확인 결과가 중복 분석되지 않는지 확인하는 Mock provider."""

    def __init__(self) -> None:
        self.checkin_analysis_calls = 0

    def analyze_checkin(self, text: str) -> dict[str, object]:
        self.checkin_analysis_calls += 1
        return super().analyze_checkin(text)


class TrackingGeminiChatProvider(MockAIProvider):
    """외부 AI 동의가 여러 Streamlit rerun에서도 유지되는지 확인한다."""

    provider_name = "gemini"


def _student_service(database_path: Path) -> StudentViewService:
    """외부 모델 없이 재현 가능한 학생 여정 서비스를 만든다."""

    return StudentViewService(
        repository=get_default_repository(),
        ai_provider=MockAIProvider(),
        database_path=database_path,
        program_recommender=InterventionRecommender(
            similarity_backend=TokenOverlapSimilarityBackend()
        ),
        course_recommender=CourseRecommender(
            similarity_backend=TokenOverlapSimilarityBackend()
        ),
    )


def test_student_checkin_and_structured_analysis_are_saved_and_loaded(
    tmp_path: Path,
) -> None:
    store = StudentCheckinDataStore(tmp_path / "app.db")
    checkin_id = store.save_checkin("S0003", STUDENT_VALUES)
    analysis = MockAIProvider().analyze_checkin(
        STUDENT_VALUES["natural_language_concern"]
    )
    analysis_id = store.save_analysis(checkin_id, analysis, "mock", True)

    saved = store.get_latest_checkin("S0003")

    assert checkin_id > 0
    assert analysis_id > 0
    assert saved is not None
    assert saved["checkin_id"] == checkin_id
    assert saved["interest_fields"] == ["경영·마케팅", "콘텐츠·디자인"]
    assert saved["summary"] == analysis["summary"]
    assert saved["provider_name"] == "mock"


def test_latest_student_checkin_is_returned(tmp_path: Path) -> None:
    store = StudentCheckinDataStore(tmp_path / "app.db")
    first_id = store.save_checkin("S0003", STUDENT_VALUES)
    updated_values = dict(STUDENT_VALUES, desired_job="콘텐츠 기획")
    latest_id = store.save_checkin("S0003", updated_values)

    saved = store.get_latest_checkin("S0003")

    assert latest_id > first_id
    assert saved is not None
    assert saved["checkin_id"] == latest_id
    assert saved["desired_job"] == "콘텐츠 기획"


def test_latest_checkin_list_and_revision_follow_new_submission(
    tmp_path: Path,
) -> None:
    store = StudentCheckinDataStore(tmp_path / "app.db")
    first_id = store.save_checkin("S0003", dict(STUDENT_VALUES, week=4))
    latest_values = dict(
        STUDENT_VALUES,
        week=5,
        interest_fields=["데이터·AI", "서비스"],
        desired_job="서비스 기획",
    )
    latest_id = store.save_checkin("S0003", latest_values)

    latest = store.list_latest_checkins()

    assert store.get_revision() == latest_id
    assert latest_id > first_id
    assert len(latest) == 1
    assert latest.iloc[0]["week"] == 5
    assert latest.iloc[0]["interest_fields"] == "데이터·AI|서비스"
    assert latest.iloc[0]["desired_job"] == "서비스 기획"


def test_student_submission_updates_only_latest_risk_snapshot(
    tmp_path: Path,
) -> None:
    repository = get_default_repository()
    source = repository.get_all()
    baseline = RiskAnalysisService(repository).build_snapshots(source)
    store = StudentCheckinDataStore(tmp_path / "app.db")
    submitted_values = dict(
        STUDENT_VALUES,
        major_interest=1,
        major_satisfaction=1,
        major_continuation_intent=1,
        career_clarity=1,
        desired_job="",
        natural_language_concern="전공과 진로 방향을 다시 함께 살펴보고 싶습니다.",
        week=4,
    )
    checkin_id = store.save_checkin("S0001", submitted_values)

    integrated = RiskAnalysisService(
        repository,
        checkin_store=store,
    ).build_snapshots(source)
    baseline_student = baseline[baseline["student_id"] == "S0001"].sort_values(
        "week"
    )
    integrated_student = integrated[
        integrated["student_id"] == "S0001"
    ].sort_values("week")

    assert len(integrated) == len(baseline)
    assert integrated_student.iloc[:-1]["overall_risk"].tolist() == (
        baseline_student.iloc[:-1]["overall_risk"].tolist()
    )
    latest = integrated_student.iloc[-1]
    previous = integrated_student.iloc[-2]
    baseline_latest = baseline_student.iloc[-1]
    assert latest["checkin_source"] == "student_submission"
    assert latest["student_checkin_id"] == checkin_id
    assert latest["natural_language_concern"] == submitted_values[
        "natural_language_concern"
    ]
    assert latest["major_adaptation_risk"] > baseline_latest[
        "major_adaptation_risk"
    ]
    assert latest["career_risk"] > baseline_latest["career_risk"]
    assert latest["risk_change"] == pytest.approx(
        latest["overall_risk"] - previous["overall_risk"], abs=0.01
    )


def test_narrative_submission_keeps_existing_risk_history_unchanged(
    tmp_path: Path,
) -> None:
    """자연어 체크인은 빈 척도를 3점으로 대체하거나 위험주차를 만들지 않는다."""

    database_path = tmp_path / "narrative.db"
    repository = get_default_repository()
    source = repository.get_all()
    store = StudentCheckinDataStore(database_path)
    before = RiskAnalysisService(repository, checkin_store=store).build_snapshots(
        source
    )

    result = _student_service(database_path).submit_checkin(
        "S0003", NARRATIVE_STUDENT_VALUES
    )
    after = RiskAnalysisService(repository, checkin_store=store).build_snapshots(
        source
    )
    saved = store.get_latest_checkin("S0003")

    pd.testing.assert_frame_equal(
        before[["student_id", "week", "overall_risk"]].reset_index(drop=True),
        after[["student_id", "week", "overall_risk"]].reset_index(drop=True),
    )
    assert saved is not None
    assert saved["response_mode"] == "narrative"
    assert saved["week"] is None
    assert saved["major_interest"] is None
    assert saved["learning_difficulty"] is None
    assert result.profile["checkin_source"] == "student_narrative"
    assert len(result.recommendation_result.recommendations) == 3


def test_narrative_semantic_states_are_saved_without_creating_scores(
    tmp_path: Path,
) -> None:
    """9개 의미 상태는 JSON으로 보존되지만 1~5 컬럼을 채우지 않는다."""

    semantic_states = update_semantic_state_from_review(
        empty_semantic_state(),
        "learning_difficulty",
        "high_difficulty",
    )
    values = dict(NARRATIVE_STUDENT_VALUES, semantic_states=semantic_states)
    store = StudentCheckinDataStore(tmp_path / "semantic-state.db")

    store.save_checkin("S0003", values)
    saved = store.get_latest_checkin("S0003")

    assert saved is not None
    assert saved["learning_difficulty"] is None
    assert saved["semantic_states"]["learning_difficulty"]["state"] == (
        "high_difficulty"
    )
    assert saved["semantic_states"]["learning_difficulty"]["source"] == (
        "student_review"
    )


def test_legacy_checkin_schema_migrates_without_losing_rows(
    tmp_path: Path,
) -> None:
    """기존 NOT NULL 척도 행을 보존하고 자연어 행을 추가할 수 있게 변경한다."""

    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE student_checkins (
                checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                week INTEGER,
                major_interest INTEGER NOT NULL,
                major_satisfaction INTEGER NOT NULL,
                major_continuation_intent INTEGER NOT NULL,
                learning_difficulty INTEGER NOT NULL,
                career_clarity INTEGER NOT NULL,
                consultation_intent INTEGER NOT NULL,
                interest_fields TEXT NOT NULL,
                desired_job TEXT NOT NULL,
                consultation_requested INTEGER NOT NULL DEFAULT 0,
                explore_other_fields INTEGER NOT NULL DEFAULT 0,
                natural_language_concern TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO student_checkins (
                student_id, week, major_interest, major_satisfaction,
                major_continuation_intent, learning_difficulty,
                career_clarity, consultation_intent, interest_fields,
                desired_job, natural_language_concern
            ) VALUES ('S0003', 4, 2, 2, 2, 4, 2, 5, '[]', '', '기존 체크인');
            """
        )

    store = StudentCheckinDataStore(database_path)
    narrative_id = store.save_checkin("S0003", NARRATIVE_STUDENT_VALUES)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT checkin_id, response_mode, major_interest "
            "FROM student_checkins ORDER BY checkin_id"
        ).fetchall()
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert rows == [(1, "scaled", 2), (narrative_id, "narrative", None)]
    assert violations == []


def test_full_student_journey_uses_only_master_programs_and_courses(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.db"
    service = _student_service(database_path)
    source = get_default_repository().get_all()

    result = service.submit_checkin("S0003", STUDENT_VALUES)

    program_ids = {
        item.program_id for item in result.recommendation_result.recommendations
    }
    assert len(program_ids) == 3
    assert program_ids.issubset(set(source.support_programs["program_id"]))
    assert result.intervention_context.risk_snapshot_id is not None
    stored_snapshot = InterventionDataStore(database_path).get_risk_snapshot(
        result.intervention_context.risk_snapshot_id
    )
    assert stored_snapshot is not None
    assert stored_snapshot["student_id"] == "S0003"
    assert stored_snapshot["student_checkin_id"] == result.checkin_id
    assert stored_snapshot["checkin_source"] == "student_submission"
    assert stored_snapshot["overall_risk"] == pytest.approx(
        result.profile["overall_risk"]
    )
    stored_intervention = InterventionDataStore(
        database_path
    ).get_intervention_by_checkin(result.checkin_id)
    assert stored_intervention is not None
    assert stored_intervention["status"] == "교직원 검토 대기"
    assert stored_intervention["generation_version"] == 1
    assert result.learning_path_result is not None
    path_course_ids = {
        item.course_id for item in result.learning_path_result.selection.courses
    }
    assert 3 <= len(path_course_ids) <= 4
    assert path_course_ids.issubset(set(source.courses["course_id"]))
    completed_ids = set(
        source.completed_courses[
            (source.completed_courses["student_id"] == "S0003")
            & (source.completed_courses["completion_status"] == "completed")
        ]["course_id"]
    )
    assert path_course_ids.isdisjoint(completed_ids)
    staff_context = StudentViewQueryService(
        database_path
    ).get_staff_student_context("S0003")
    assert staff_context.intervention is not None
    assert len(staff_context.recommendations) == 3
    assert len(staff_context.learning_path) == len(path_course_ids)
    assert staff_context.learning_path.iloc[0]["source_checkin_id"] == result.checkin_id


def test_confirmed_ai_preview_is_reused_when_checkin_is_submitted(
    tmp_path: Path,
) -> None:
    """학생이 확인한 AI 이해 결과를 제출 시 다시 호출하지 않는다."""

    provider = CountingMockAIProvider()
    service = StudentViewService(
        repository=get_default_repository(),
        ai_provider=provider,
        database_path=tmp_path / "preview-reuse.db",
        program_recommender=InterventionRecommender(
            similarity_backend=TokenOverlapSimilarityBackend()
        ),
        course_recommender=CourseRecommender(
            similarity_backend=TokenOverlapSimilarityBackend()
        ),
    )

    preview = service.preview_checkin_analysis(
        STUDENT_VALUES["natural_language_concern"]
    )
    result = service.submit_checkin(
        "S0003", STUDENT_VALUES, analysis_result=preview
    )

    assert provider.checkin_analysis_calls == 1
    assert result.analysis_result == preview


def test_week_five_followup_is_appended_and_compared_with_week_four(
    tmp_path: Path,
) -> None:
    """4주차 지원 시작 후 5주차 체크인을 새 관찰값으로 비교한다."""

    database_path = tmp_path / "week-five.db"
    service = _student_service(database_path)
    baseline = service.submit_checkin(
        "S0003", dict(STUDENT_VALUES, week=4)
    )
    followup_values = dict(
        STUDENT_VALUES,
        week=5,
        major_interest=5,
        major_satisfaction=5,
        major_continuation_intent=5,
        learning_difficulty=2,
        career_clarity=5,
        consultation_intent=2,
        natural_language_concern="상담 후 전공과 진로 방향이 명확해졌습니다.",
    )
    followup = service.submit_checkin("S0003", followup_values)

    repository = get_default_repository()
    snapshots = RiskAnalysisService(
        repository,
        checkin_store=StudentCheckinDataStore(database_path),
    ).build_snapshots(repository.get_all())
    student_snapshots = snapshots[
        snapshots["student_id"] == "S0003"
    ].sort_values("week")
    direct_rows = student_snapshots[
        student_snapshots["checkin_source"] == "student_submission"
    ]

    assert baseline.profile["week"] == 4
    assert followup.profile["week"] == 5
    assert direct_rows["week"].tolist()[-2:] == [4, 5]
    assert direct_rows["student_checkin_id"].tolist()[-2:] == [
        baseline.checkin_id,
        followup.checkin_id,
    ]

    effects = InterventionEffectService(database_path=database_path).analyze(
        snapshots
    ).records
    baseline_effect = effects[
        effects["intervention_id"]
        == baseline.intervention_context.intervention_id
    ].iloc[0]
    assert baseline_effect["baseline_week"] == 4
    assert baseline_effect["current_week"] == 5
    assert bool(baseline_effect["has_followup_data"])


def test_student_feedback_is_visible_from_staff_query(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    service = _student_service(database_path)
    result = service.submit_checkin("S0003", STUDENT_VALUES)
    first_program_id = result.recommendation_result.recommendations[0].program_id

    service.record_analysis_feedback(
        "S0003", result.checkin_id, "조금 달라요", "관심분야를 더 강조해 주세요."
    )
    service.record_recommendation_feedback(
        "S0003",
        result.intervention_context.recommendation_ids[first_program_id],
        "관심 있어요",
        "신청 방법이 궁금해요.",
    )
    assert result.learning_path_recommendation_id is not None
    service.record_recommendation_feedback(
        "S0003",
        result.learning_path_recommendation_id,
        "저장할게요",
    )

    staff_context = StudentViewQueryService(
        database_path
    ).get_staff_student_context("S0003")

    assert staff_context.latest_checkin is not None
    assert staff_context.latest_checkin["consultation_requested"] is True
    assert staff_context.latest_checkin["summary"]
    assert staff_context.analysis_feedback["feedback_type"].tolist() == [
        "조금 달라요"
    ]
    assert set(staff_context.recommendation_feedback["recommendation_type"]) == {
        "support_program",
        "learning_path",
    }
    assert set(staff_context.recommendation_feedback["feedback_type"]) == {
        "관심 있음",
        "나중에 보기",
    }


def test_staff_pages_reuse_student_generated_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    service = _student_service(database_path)
    result = service.submit_checkin("S0003", STUDENT_VALUES)

    intervention_page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "03_ai_intervention.py")
    )
    intervention_page.session_state["demo_student_id"] = "S0003"
    intervention_page.run(timeout=30)

    assert not intervention_page.exception
    assert any(
        "AI를 다시 호출하거나 추천을 중복 생성하지 않았습니다" in item.value
        for item in intervention_page.success
    )
    assert not any(
        button.label == "AI 이해 및 비교과 추천 생성"
        for button in intervention_page.button
    )
    intervention_page.pills[0].set_value("비교과 추천").run(timeout=30)
    assert not intervention_page.exception
    assert len(intervention_page.get("link_button")) == 3
    assert all(
        item.label == "WINGS에서 세부 조건 확인"
        for item in intervention_page.get("link_button")
    )

    intervention_page.pills[0].set_value("교과 학습경로").run(timeout=30)
    assert not intervention_page.exception
    assert not any(
        button.label == "교과 학습경로 생성"
        for button in intervention_page.button
    )
    intervention_page.pills[0].set_value("최종 검토").run(timeout=30)
    review_action_labels = {
        button.label for button in intervention_page.button
    }
    assert {
        "검토 결과 저장",
        "전체 보류",
    }.issubset(review_action_labels)
    next(
        item for item in intervention_page.radio if item.label == "비교과 판단"
    ).set_value("승인").run(timeout=30)
    next(
        item for item in intervention_page.radio if item.label == "교과 판단"
    ).set_value("승인").run(timeout=30)
    judgment_labels = [
        item.label
        for item in intervention_page.selectbox
        if item.label.endswith(" 적절성")
    ]
    assert len(judgment_labels) == 4
    for label in judgment_labels:
        next(
            item for item in intervention_page.selectbox if item.label == label
        ).set_value("적절").run(timeout=30)
    next(
        button
        for button in intervention_page.button
        if button.label == "검토 결과 저장"
    ).click().run(timeout=30)
    assert not intervention_page.exception
    assert any(
        "비교과와 교과의 독립 검토" in item.value
        for item in intervention_page.success
    )
    interventions = InterventionDataStore(database_path).list_interventions()
    assert len(interventions) == 1
    assert interventions.iloc[0]["source_checkin_id"] == result.checkin_id
    path = LearningPathDataStore(database_path).get_learning_path_by_checkin(
        result.checkin_id
    )
    assert path["review_status"].unique().tolist() == ["승인"]


def test_adjusted_recommendations_are_regenerated_as_a_new_version(
    tmp_path: Path,
) -> None:
    """선택 제외 재추천은 이전 비교과·교과 버전을 보존한다."""

    database_path = tmp_path / "regenerated-recommendations.db"
    service = _student_service(database_path)
    original = service.submit_checkin("S0003", STUDENT_VALUES)
    assert original.learning_path_result is not None
    service.intervention_service.apply_integrated_review_action(
        original.intervention_context.intervention_id,
        "수정",
        "선택한 비교과와 교과목을 제외해 재추천",
    )

    excluded_program_id = (
        original.recommendation_result.recommendations[0].program_id
    )
    excluded_course_id = (
        original.learning_path_result.selection.courses[0].course_id
    )
    regenerated = service.regenerate_recommendations(
        "S0003",
        original.profile,
        original.analysis_result.analysis,
        service.repository.get_all(),
        excluded_program_ids=[excluded_program_id],
        excluded_course_ids=[excluded_course_id],
    )

    assert regenerated.intervention_context.generation_version == 2
    assert regenerated.learning_path_result is not None
    assert regenerated.learning_path_error is None
    assert not regenerated.program_recommendations_preserved
    assert not regenerated.learning_path_preserved
    assert excluded_program_id not in {
        item.program_id
        for item in regenerated.recommendation_result.recommendations
    }
    assert excluded_course_id not in {
        item.course_id
        for item in regenerated.learning_path_result.selection.courses
    }

    path_store = LearningPathDataStore(database_path)
    version_one = path_store.get_learning_path_version(original.checkin_id, 1)
    version_two = path_store.get_learning_path_version(original.checkin_id, 2)
    assert not version_one.empty
    assert not version_two.empty
    assert excluded_course_id in set(version_one["course_id"].astype(str))
    assert excluded_course_id not in set(version_two["course_id"].astype(str))
    assert version_one["review_status"].unique().tolist() == ["조정 필요"]
    assert version_two["review_status"].unique().tolist() == ["검토 대기"]

    service.intervention_service.apply_integrated_review_action(
        regenerated.intervention_context.intervention_id,
        "승인",
        "새 버전의 비교과·교과 확인",
    )
    version_one_after_review = path_store.get_learning_path_version(
        original.checkin_id, 1
    )
    version_two_after_review = path_store.get_learning_path_version(
        original.checkin_id, 2
    )
    assert version_one_after_review["review_status"].unique().tolist() == [
        "조정 필요"
    ]
    assert version_two_after_review["review_status"].unique().tolist() == ["승인"]

    latest_context = StudentViewQueryService(
        database_path
    ).get_staff_student_context("S0003")
    assert latest_context.intervention is not None
    assert latest_context.intervention["generation_version"] == 2
    assert latest_context.learning_path["generation_version"].unique().tolist() == [2]


def test_course_only_adjustment_can_regenerate_the_learning_path(
    tmp_path: Path,
) -> None:
    """비교과 승인 상태를 유지하면서 교과 조정 요청만으로 과목을 교체한다."""

    database_path = tmp_path / "course-only-regeneration.db"
    service = _student_service(database_path)
    original = service.submit_checkin("S0003", STUDENT_VALUES)
    assert original.learning_path_result is not None
    course_ids = [
        course.course_id
        for course in original.learning_path_result.selection.courses
    ]
    excluded_course_id = course_ids[0]
    service.intervention_service.apply_separate_review_actions(
        intervention_id=original.intervention_context.intervention_id,
        program_decision="승인",
        program_note="비교과 추천은 유지",
        course_decision="조정 필요",
        course_note="첫 번째 교과목을 다른 후보로 교체",
        course_judgments={
            course_id: (
                "부적절" if course_id == excluded_course_id else "적절"
            )
            for course_id in course_ids
        },
    )

    regenerated = service.regenerate_recommendations(
        "S0003",
        original.profile,
        original.analysis_result.analysis,
        service.repository.get_all(),
        excluded_course_ids=[excluded_course_id],
    )

    assert regenerated.intervention_context.generation_version == 2
    assert regenerated.learning_path_result is not None
    assert regenerated.program_recommendations_preserved
    assert not regenerated.learning_path_preserved
    preserved_program_review = InterventionDataStore(
        database_path
    ).get_intervention(regenerated.intervention_context.intervention_id)
    assert preserved_program_review is not None
    assert preserved_program_review["staff_action"] == "추천 검토 승인"
    assert preserved_program_review["staff_note"] == "비교과 추천은 유지"
    assert excluded_course_id not in {
        course.course_id
        for course in regenerated.learning_path_result.selection.courses
    }


def test_regeneration_requires_adjustment_and_an_explicit_exclusion(
    tmp_path: Path,
) -> None:
    """메모만으로 추천을 바꾸지 않고 명시적 조정 절차를 강제한다."""

    service = _student_service(tmp_path / "regeneration-guard.db")
    original = service.submit_checkin("S0003", STUDENT_VALUES)
    source = service.repository.get_all()

    with pytest.raises(ValueError, match="조정 필요"):
        service.regenerate_recommendations(
            "S0003",
            original.profile,
            original.analysis_result.analysis,
            source,
            excluded_program_ids=[
                original.recommendation_result.recommendations[0].program_id
            ],
        )

    service.intervention_service.apply_integrated_review_action(
        original.intervention_context.intervention_id,
        "수정",
        "재추천 테스트",
    )
    with pytest.raises(ValueError, match="하나 이상"):
        service.regenerate_recommendations(
            "S0003",
            original.profile,
            original.analysis_result.analysis,
            source,
        )


def test_integrated_review_ui_creates_a_versioned_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """통합 검토 UI에서 조정 항목을 명시해 새 버전을 만든다."""

    database_path = tmp_path / "regeneration-ui.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    original = _student_service(database_path).submit_checkin(
        "S0003", STUDENT_VALUES
    )
    excluded_program_id = (
        original.recommendation_result.recommendations[0].program_id
    )

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "03_ai_intervention.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.run(timeout=60)
    page.pills[0].set_value("최종 검토").run(timeout=60)
    next(
        item for item in page.radio if item.label == "비교과 판단"
    ).set_value("조정 필요").run(timeout=60)
    next(
        item for item in page.radio if item.label == "교과 판단"
    ).set_value("승인").run(timeout=60)
    judgment_labels = [
        item.label for item in page.selectbox if item.label.endswith(" 적절성")
    ]
    for label in judgment_labels:
        next(
            item for item in page.selectbox if item.label == label
        ).set_value("적절").run(timeout=60)
    next(
        item for item in page.text_area if item.label == "비교과 검토 메모"
    ).set_value("첫 번째 비교과를 다른 후보로 교체해 주세요.").run(
        timeout=60
    )
    next(
        button
        for button in page.button
        if button.label == "검토 결과 저장"
    ).click().run(timeout=60)

    assert not page.exception
    assert any(
        "추천 다시 만들기" in str(item.value) for item in page.markdown
    )
    replace_programs = next(
        item
        for item in page.multiselect
        if item.label == "교체할 비교과 추천"
    )
    replace_programs.set_value([excluded_program_id]).run(timeout=60)
    next(
        button
        for button in page.button
        if button.label == "선택 항목 제외하고 추천 다시 만들기"
    ).click().run(timeout=60)

    assert not page.exception
    latest = InterventionDataStore(database_path).get_intervention_by_checkin(
        original.checkin_id
    )
    assert latest is not None
    assert latest["generation_version"] == 2
    new_rows = InterventionDataStore(database_path).list_recommendations(
        str(latest["recommendation_batch_id"])
    )
    assert excluded_program_id not in set(new_rows["target_id"].astype(str))
    preserved_path = LearningPathDataStore(database_path).get_learning_path_version(
        original.checkin_id,
        2,
    )
    assert preserved_path["review_status"].unique().tolist() == ["승인"]
    assert preserved_path["review_judgment"].unique().tolist() == ["적절"]
    assert any(
        "v2 추천을 저장했습니다" in str(item.value)
        for item in page.success
    )
    assert any(
        "교과 결과는 그대로 유지했습니다" in str(item.value)
        for item in page.success
    )


def test_integrated_review_refreshes_only_stale_program_recommendations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이전 master ID는 체크인·AI 분석을 삭제하지 않고 새 추천 버전으로 교체한다."""

    database_path = tmp_path / "stale-programs.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    checkin_store = StudentCheckinDataStore(database_path)
    checkin_id = checkin_store.save_checkin("S0003", STUDENT_VALUES)
    checkin_analysis = MockAIProvider().analyze_checkin(
        STUDENT_VALUES["natural_language_concern"]
    )
    checkin_store.save_analysis(checkin_id, checkin_analysis, "mock", True)
    service = InterventionDataStore(database_path)
    batch_id, _ = service.save_recommendation_batch(
        "S0003",
        [
            {
                "recommendation_type": "support_program",
                "target_id": f"OLD{index:03d}",
                "score": 90.0 - index,
                "rank": index,
                "reason": "이전 master 추천",
            }
            for index in range(1, 4)
        ],
    )
    service.create_intervention(
        "S0003",
        batch_id,
        ["OLD001", "OLD002", "OLD003"],
        source_checkin_id=checkin_id,
        initial_status="교직원 검토 대기",
    )

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "03_ai_intervention.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.run(timeout=30)

    assert not page.exception
    refresh_button = next(
        button
        for button in page.button
        if button.label == "현재 비교과 목록으로 추천 갱신"
    )
    refresh_button.click().run(timeout=30)
    assert not page.exception

    latest_intervention = InterventionDataStore(
        database_path
    ).get_intervention_by_checkin(checkin_id)
    assert latest_intervention is not None
    assert latest_intervention["generation_version"] == 2
    current_program_ids = set(
        get_default_repository().get_support_programs()["program_id"].astype(str)
    )
    refreshed_rows = InterventionDataStore(database_path).list_recommendations(
        str(latest_intervention["recommendation_batch_id"])
    )
    assert set(refreshed_rows["target_id"]).issubset(current_program_ids)


def test_narrative_results_open_in_student_and_staff_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """점수가 비어 있는 자연어 체크인도 기존 마이페이지·교직원 흐름을 유지한다."""

    database_path = tmp_path / "narrative-pages.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    result = _student_service(database_path).submit_checkin(
        "S0003", NARRATIVE_STUDENT_VALUES
    )

    mypage = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "12_student_mypage.py")
    )
    mypage.session_state["demo_student_id"] = "S0003"
    mypage.run(timeout=30)
    assert not mypage.exception
    assert not any(
        item.label == "대화에서 직접 말한 1~5 값"
        for item in mypage.expander
    )

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "03_ai_intervention.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.run(timeout=30)
    assert not page.exception
    page.pills[0].set_value("교과 학습경로").run(timeout=30)
    assert not page.exception

    context = StudentViewQueryService(database_path).get_staff_student_context(
        "S0003"
    )
    assert context.latest_checkin is not None
    assert context.latest_checkin["checkin_id"] == result.checkin_id
    assert context.intervention is not None


def test_integrated_review_supports_students_with_different_latest_weeks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """한 학생만 5주차여도 4주차가 최신인 다른 학생을 정상 조회한다."""

    database_path = tmp_path / "mixed-weeks.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    StudentCheckinDataStore(database_path).save_checkin(
        "S0003", dict(STUDENT_VALUES, week=5)
    )
    page = AppTest.from_file(str(PROJECT_ROOT / "pages" / "03_ai_intervention.py"))
    page.session_state["demo_student_id"] = "S0001"
    page.run(timeout=30)

    assert not page.exception
    assert any(
        metric.label == "학생 ID" and metric.value == "S0001"
        for metric in page.metric
    )


def test_student_checkin_uses_free_kare_chat_and_keeps_structured_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """학생은 자유대화하고 Kare는 기존 체크인 값으로 응답을 누적한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "app.db"
    )
    monkeypatch.setattr(
        "src.ai.create_ai_provider", lambda **_: MockAIProvider()
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    ).run(timeout=30)

    assert not page.exception
    assert not page.slider
    assert len(page.chat_input) == 1
    assert any(
        button.label == "외부 AI 전송 없이 직접 입력" for button in page.button
    )
    assert page.checkbox[0].label.startswith("Kare AI 대화를 위해")
    page.checkbox[0].set_value(True).run(timeout=30)

    assert not page.exception
    assert any(
        button.label == "전공이 나와 맞는지 고민이에요" for button in page.button
    )
    chat_messages = (
        "요즘 수업 속도가 빨라서 과제가 밀리고 있어요.",
        "전공은 재미있지만 내가 잘 따라갈 수 있을지 걱정돼요.",
        "데이터와 서비스 분야에 관심 있고 서비스 기획을 하고 싶어요.",
        "상담을 받아서 밀린 과제부터 해결하고 싶어요.",
    )
    for message in chat_messages:
        assert len(page.chat_input) == 1
        page.chat_input[0].set_value(message).run(timeout=30)
        assert not page.exception

    assert any(
        button.label == "Kare가 이해한 내용 확인" for button in page.button
    )
    next(
        button
        for button in page.button
        if button.label == "Kare가 이해한 내용 확인"
    ).click().run(timeout=30)

    assert not page.exception
    assert any(
        button.label == "맞아요, 제출할게요" for button in page.button
    )
    assert any(
        button.label == "조금 수정할게요" for button in page.button
    )
    draft = page.session_state["student_checkin_S0001_draft"]
    assert draft["major_interest"] is None
    assert draft["learning_difficulty"] is None
    assert draft["interest_fields"] == ["데이터·AI", "서비스"]
    assert draft["desired_job"] == "서비스 기획"
    assert "과제가 밀리고" in draft["natural_language_concern"]
    assert draft["response_mode"] == "narrative"

    history_key = "student_checkin_S0001_conversation_history"
    mode_key = "student_checkin_S0001_conversation_mode"
    preview_key = "student_checkin_S0001_analysis_preview"
    history_before_edit = list(page.session_state[history_key])
    next(
        button
        for button in page.button
        if button.label == "조금 수정할게요"
    ).click().run(timeout=30)

    assert not page.exception
    assert page.session_state[mode_key] == "edit"
    assert page.session_state[history_key] == history_before_edit
    assert preview_key in page.session_state
    assert page.text_area[0].value == draft["natural_language_concern"]
    page.text_input[0].set_value("데이터 분석가").run(timeout=30)
    next(
        button
        for button in page.button
        if button.label == "수정 내용을 반영하고 다시 확인"
    ).click().run(timeout=30)

    assert not page.exception
    assert page.session_state[mode_key] == "review"
    assert page.session_state[history_key] == history_before_edit
    assert page.session_state["student_checkin_S0001_draft"]["desired_job"] == (
        "데이터 분석가"
    )
    assert preview_key in page.session_state

    next(
        button
        for button in page.button
        if button.label == "다시 이야기할게요"
    ).click().run(timeout=30)

    assert not page.exception
    assert page.session_state[mode_key] == "conversation"
    assert page.session_state[history_key][:-1] == history_before_edit
    assert "그대로 두고 이어서" in page.session_state[history_key][-1]["content"]
    assert page.session_state["student_checkin_S0001_draft"]["desired_job"] == (
        "데이터 분석가"
    )
    assert preview_key not in page.session_state
    assert len(page.chat_input) == 1


def test_external_ai_consent_survives_multiple_chat_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """동의 체크박스가 사라진 뒤에도 Gemini 대화 동의 상태를 유지한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "consent-rerun.db"
    )
    monkeypatch.setattr(
        "src.ai.create_ai_provider", lambda **_: TrackingGeminiChatProvider()
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    ).run(timeout=30)

    page.checkbox[0].set_value(True).run(timeout=30)
    page.chat_input[0].set_value("요즘 학교생활부터 이야기할게요").run(
        timeout=30
    )
    page.chat_input[0].set_value("전공 흥미는 3점이에요").run(timeout=30)

    assert not page.exception
    assert page.session_state[
        "student_checkin_S0001_external_ai_consent"
    ] is True
    popovers = page.get("popover")
    assert len(popovers) == 1
    assert popovers[0].proto.popover.label == "대화 설정"
    assert not any(
        item.label == "Kare가 현재까지 파악한 항목"
        for item in page.expander
    )
    assert not any(
        "체크인을 제출하면" in str(item.value) for item in page.caption
    )
    assert page.session_state["student_checkin_S0001_chat_provider"] == "gemini"
    assert "student_checkin_S0001_chat_warning" not in page.session_state


def test_existing_ai_chat_restores_persisted_consent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """수정 전 시작된 AI 대화도 기존의 명시적 동의 상태를 복구한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "consent-migration.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    )
    page.session_state[
        "student_checkin_S0001_conversation_mode"
    ] = "conversation"
    page.session_state["student_checkin_S0001_input_mode"] = "ai_chat"
    page.run(timeout=30)

    assert not page.exception
    assert page.session_state[
        "student_checkin_S0001_external_ai_consent"
    ] is True


def test_student_checkin_welcome_prioritizes_one_clean_kare_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """시작 화면은 대시보드 정보 대신 Kare 대화와 선택적 동의를 우선한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "clean-welcome.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    ).run(timeout=30)

    assert not page.exception
    rendered = "\n".join(str(item.value) for item in page.markdown)
    assert "kare-welcome-title" in rendered
    assert "kare-orb" in rendered
    assert "kare-example-prompt" in rendered
    assert '[data-testid="stCheckbox"]' in rendered
    assert "width: fit-content" in rendered
    assert "text-align: center" in rendered
    assert "margin: 0.8rem auto 1.55rem !important" in rendered
    assert "st-key-kare_messages_" in rendered
    assert ".kare-message-user" in rendered
    assert "flex-direction: row-reverse" in rendered
    assert "background: #006ab6" in rendered
    assert "width: max-content" in rendered
    assert "--kare-bubble-max" in rendered
    assert "padding-bottom: 0.8rem" in rendered
    assert "line-height: 1.62" in rendered
    assert "st-key-kare_finish_action_" in rendered
    assert "align-items: flex-start" in rendered
    assert "padding-left: 2.55rem" in rendered
    assert "st-key-kare_chat_header_controls_" in rendered
    assert "grid-template-columns: minmax(0, 1fr) auto" in rendered
    assert "오늘은 어떤 이야기를 나눠볼까요?" in rendered
    assert len(page.chat_input) == 1
    assert page.chat_input[0].disabled is True
    assert page.chat_input[0].placeholder == "AI 대화 이용 동의 후 입력할 수 있어요"
    assert [item.label for item in page.expander] == ["개인정보·AI 이용 안내"]
    assert {
        "외부 AI 전송 없이 직접 입력",
        "최근 결과 다시 보기",
    }.issubset({button.label for button in page.button})
    assert not page.info


def test_kare_chat_bubble_width_adapts_to_message_length() -> None:
    """말풍선은 짧은 문장은 작게, 긴 문장은 균형 있게 줄바꿈한다."""

    short_width = calculate_chat_bubble_width_rem("하이")
    medium_width = calculate_chat_bubble_width_rem(
        "요즘 학교생활부터 이야기할게요"
    )
    long_width = calculate_chat_bubble_width_rem(
        "전공이 나와 맞는지 고민이고 과제도 밀려서 학교생활과 진로를 "
        "어떻게 정리해야 할지 잘 모르겠어요."
    )

    assert 7.5 <= short_width < medium_width < long_width <= 33.0
    html = build_kare_chat_bubble_html("user", "<script>하이</script>")
    assert "kare-message-user" in html
    assert "--kare-bubble-max:" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize(
    ("count", "expected_stage", "expected_percent"),
    (
        (2, "이야기를 살펴보는 중", "22.2%"),
        (4, "핵심 맥락을 이해하는 중", "44.4%"),
        (7, "내용을 확인할 준비가 되었어요", "77.8%"),
    ),
)
def test_kare_context_progress_uses_non_evaluative_stages(
    count: int,
    expected_stage: str,
    expected_percent: str,
) -> None:
    """맥락 이해도는 점수 대신 안심 문구와 단계별 막대로 표시한다."""

    html = build_kare_context_progress_html(count)

    assert expected_stage in html
    assert expected_percent in html
    assert f'aria-valuenow="{count}"' in html
    assert "모든 항목을 채우지 않아도" not in html
    assert f"{count}/9" not in html


def test_kare_context_progress_clamps_invalid_counts() -> None:
    """진행 막대의 접근성 값은 항상 정상 범위를 유지한다."""

    assert 'aria-valuenow="0"' in build_kare_context_progress_html(-3)
    assert 'aria-valuenow="9"' in build_kare_context_progress_html(30)


def test_kare_typing_indicator_is_rendered_as_an_assistant_chat_bubble() -> None:
    """응답 대기 표시는 채팅 밖 spinner가 아닌 Kare 말풍선으로 보인다."""

    html = build_kare_typing_indicator_html()

    assert "kare-message-assistant" in html
    assert "kare-message-typing" in html
    assert html.count('class="kare-typing-dot"') == 3
    assert "답변을 생각하고 있어요" in html
    assert 'role="status"' in html


def test_student_can_check_in_without_external_ai_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """비동의 학생은 Gemini 호출 없이 직접 입력과 로컬 검토를 완료한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "direct-checkin.db"
    )

    def fail_if_external_provider_is_created(**_: object) -> MockAIProvider:
        raise AssertionError("비동의 경로에서 외부 AI provider를 만들면 안 됩니다.")

    monkeypatch.setattr("src.ai.create_ai_provider", fail_if_external_provider_is_created)
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    ).run(timeout=30)

    assert not page.checkbox[0].value
    next(
        button
        for button in page.button
        if button.label == "외부 AI 전송 없이 직접 입력"
    ).click().run(timeout=30)

    assert not page.exception
    assert len(page.selectbox) == 1
    page.multiselect[0].set_value(["데이터·AI"]).run(timeout=30)
    page.text_input[0].set_value("데이터 분석").run(timeout=30)
    page.text_area[0].set_value("학습 계획을 정리하고 싶어요.").run(timeout=30)
    next(
        button
        for button in page.button
        if button.label == "외부 전송 없이 입력 내용 확인"
    ).click().run(timeout=30)

    assert not page.exception
    next(
        button
        for button in page.button
        if button.label == "Kare가 이해한 내용 확인"
    ).click().run(timeout=30)

    assert not page.exception
    assert any(
        button.label == "맞아요, 제출할게요" for button in page.button
    )
    preview = page.session_state["student_checkin_S0001_analysis_preview"]
    assert preview.provider_name == "mock"
    assert "외부 AI로 전송하지 않고" in str(preview.warning)


def test_student_result_shows_ai_transformation_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """제출 결과는 학생의 말·AI 이해·DB 맞춤 연결 단계를 함께 보여준다."""

    database_path = tmp_path / "ai-transformation.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    result = _student_service(database_path).submit_checkin(
        "S0003", STUDENT_VALUES
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "07_student_view.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.session_state["student_journey_result_S0003"] = result
    page.run(timeout=30)

    assert not page.exception
    assert any(
        item.value == "내 고민이 맞춤 지원으로 연결된 과정"
        for item in page.subheader
    )
    captions = "\n".join(str(item.value) for item in page.caption)
    assert "STEP 01 · 학생의 말" in captions
    assert "STEP 02 · AI 이해" in captions
    assert "STEP 03 · 맞춤 연결" in captions
    visible_values = [*page.markdown, *page.text, *page.caption]
    assert any(
        STUDENT_VALUES["natural_language_concern"] in str(item.value)
        for item in visible_values
    )


def test_student_mypage_reuses_latest_saved_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """마이페이지는 AI를 다시 호출하지 않고 최신 저장 결과를 표시한다."""

    database_path = tmp_path / "app.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    service = _student_service(database_path)
    result = service.submit_checkin("S0003", STUDENT_VALUES)
    source = get_default_repository().get_all()
    selected_program_id = result.recommendation_result.recommendations[0].program_id
    service.intervention_service.save_support_plan(
        intervention_id=result.intervention_context.intervention_id,
        selected_program_ids=[selected_program_id],
        valid_program_ids=set(source.support_programs["program_id"].astype(str)),
        assigned_staff="IR센터 담당자",
        planned_contact_date="2026-08-20",
        next_action="전화 안내",
    )

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "12_student_mypage.py")
    )
    page.session_state["demo_student_id"] = "S0003"
    page.run(timeout=30)

    assert not page.exception
    assert any(item.value == "최근 체크인" for item in page.subheader)
    assert any(item.value == "나에게 추천된 지원" for item in page.subheader)
    assert any(
        item.value == "AI 맞춤형 역량 학습경로" for item in page.subheader
    )
    assert any(
        item.value == "담당자가 확정한 지원계획" for item in page.subheader
    )
    assert any("연락 예정" in str(item.value) for item in page.success)
    assert any("연락 준비 중" in item.value for item in page.info)
    for recommendation in result.recommendation_result.recommendations:
        assert any(
            recommendation.program_name in str(item.value)
            for item in page.markdown
        )
    assert result.learning_path_result is not None
    assert any(
        result.learning_path_result.explanation["path_name"] in str(item.value)
        for item in page.markdown
    )
    course_captions = [str(item.value) for item in page.caption]
    assert any(
        "소속 학과" in caption
        and "개설학과" in caption
        and "간호학과" in caption
        for caption in course_captions
    )
    if "catalog_source" in source.courses.columns:
        assert any(
            "공식 과목코드" in caption
            and "실제 수강 가능 여부 확인 필요" in caption
            for caption in course_captions
        )
    else:
        assert any("타과 선택 · 개설학과" in caption for caption in course_captions)
    assert len(page.get("link_button")) == 3
    assert all(
        item.label == "WINGS에서 세부 조건 확인"
        for item in page.get("link_button")
    )


def test_student_mypage_without_checkin_guides_to_checkin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """저장 결과가 없는 학생에게 빈 화면 대신 다음 행동을 안내한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "app.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "12_student_mypage.py")
    ).run(timeout=30)

    assert not page.exception
    assert any("나의 체크인" in item.value for item in page.info)


def test_internal_intervention_status_is_student_friendly() -> None:
    assert describe_student_support_status("교직원 검토 대기") == "담당자 확인 중"
    assert describe_student_support_status("연락 전") == "연락 준비 중"
    assert describe_student_support_status("알 수 없는 상태") == "진행 상황 확인 중"


def test_student_support_copy_hides_raw_risk_labels_and_scores() -> None:
    profile = {
        "major_adaptation_risk": 80,
        "career_risk": 61,
        "engagement_risk": 45,
        "attendance_risk": 10,
        "achievement_risk": 20,
    }

    areas = build_student_support_areas(profile)
    messages = describe_student_analysis(
        MockAIProvider().analyze_checkin(
            STUDENT_VALUES["natural_language_concern"]
        )
    )

    assert [area.status for area in areas] == [
        "지원 권장",
        "지원 권장",
        "점검 필요",
        "양호",
        "양호",
    ]
    assert all("위험" not in area.label + area.status for area in areas)
    assert all("위험" not in message for message in messages)


def test_student_checkin_rejects_invalid_likert_and_other_student_feedback(
    tmp_path: Path,
) -> None:
    store = StudentCheckinDataStore(tmp_path / "app.db")
    invalid_values = dict(STUDENT_VALUES, major_interest=6)
    with pytest.raises(ValueError, match="1~5 범위"):
        store.save_checkin("S0003", invalid_values)

    checkin_id = store.save_checkin("S0003", STUDENT_VALUES)
    with pytest.raises(ValueError, match="다른 학생"):
        store.save_analysis_feedback(checkin_id, "S9999", "맞아요")
