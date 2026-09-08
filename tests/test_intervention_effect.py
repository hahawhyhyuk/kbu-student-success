"""개입 시점 위험 스냅샷과 최신 위험도 비교 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.intervention_effect_service import InterventionEffectService
from src.intervention_recommender import ProgramRecommendation
from src.intervention_service import InterventionManagementService
from src.repositories import get_default_repository
from src.risk_service import create_integrated_risk_service
from src.utils import PROJECT_ROOT


def _recommendation(student_suffix: str) -> tuple[ProgramRecommendation, ...]:
    """개입 레코드 생성에 필요한 검증된 추천 한 건을 만든다."""

    return (
        ProgramRecommendation(
            program_id=f"PRG-{student_suffix}",
            program_name="학생성공 지원",
            program_type="학습지원",
            description="테스트용 지원프로그램",
            department_in_charge="DEMO_UNIT",
            operation_period="상시",
            score=88.0,
            reason="위험 원인과 지원 내용이 연결됩니다.",
        ),
    )


def _snapshot(student_id: str, score: float, week: int = 4) -> dict[str, object]:
    """동일한 영역 점수를 가진 위험 스냅샷을 만든다."""

    return {
        "student_id": student_id,
        "week": week,
        "attendance_risk": score,
        "engagement_risk": score,
        "achievement_risk": score,
        "major_adaptation_risk": score,
        "career_risk": score,
        "overall_risk": score,
        "risk_level": "고위험" if score >= 65 else "주의",
        "primary_risk_type": "복합위험형",
        "secondary_risk_types": "출결위험형, 학습참여저하형",
    }


def _current(student_id: str, score: float, week: int = 5) -> dict[str, object]:
    """비교에 필요한 최신 위험 스냅샷 한 행을 만든다."""

    return {
        "student_id": student_id,
        "week": week,
        "department": "AI융합과",
        "attendance_risk": score,
        "engagement_risk": score,
        "achievement_risk": score,
        "major_adaptation_risk": score,
        "career_risk": score,
        "overall_risk": score,
        "risk_level": "고위험" if score >= 65 else "주의",
        "primary_risk_type": "복합위험형",
    }


def _effect_service(database_path: Path) -> tuple[
    InterventionEffectService,
    InterventionManagementService,
]:
    """테스트 DB를 공유하는 개입 관리·효과 서비스를 만든다."""

    management = InterventionManagementService(database_path=database_path)
    effect = InterventionEffectService(
        config={
            "intervention_effect": {
                "meaningful_change_threshold": 5,
                "score_change_epsilon": 0.01,
            }
        },
        intervention_service=management,
    )
    return effect, management


def test_effect_classifies_improvement_stable_and_worsening(
    tmp_path: Path,
) -> None:
    effect, management = _effect_service(tmp_path / "app.db")
    cases = (
        ("S0001", 70.0, 60.0, "개선"),
        ("S0002", 50.0, 52.0, "유지"),
        ("S0003", 50.0, 57.0, "악화"),
    )
    for student_id, baseline, _, _ in cases:
        management.record_recommendation_batch(
            student_id,
            _recommendation(student_id),
            risk_snapshot=_snapshot(student_id, baseline),
        )

    analysis = effect.analyze(
        pd.DataFrame(
            [_current(student_id, current) for student_id, _, current, _ in cases]
        )
    )
    records = analysis.records.set_index("student_id")

    assert analysis.total_interventions == 3
    assert records.loc["S0001", "effect_status"] == "개선"
    assert records.loc["S0001", "overall_change"] == pytest.approx(-10)
    assert records.loc["S0002", "effect_status"] == "유지"
    assert records.loc["S0003", "effect_status"] == "악화"
    assert records.loc["S0003", "career_risk_change"] == pytest.approx(7)


def test_same_observation_is_waiting_for_followup_data(tmp_path: Path) -> None:
    effect, management = _effect_service(tmp_path / "app.db")
    management.record_recommendation_batch(
        "S0010",
        _recommendation("S0010"),
        risk_snapshot=_snapshot("S0010", 68, week=4),
    )

    analysis = effect.analyze(pd.DataFrame([_current("S0010", 68, week=4)]))
    record = analysis.records.iloc[0]

    assert record["has_followup_data"] is False or not bool(
        record["has_followup_data"]
    )
    assert record["effect_status"] == "후속 데이터 대기"


def test_legacy_and_missing_current_records_are_counted_not_compared(
    tmp_path: Path,
) -> None:
    effect, management = _effect_service(tmp_path / "app.db")
    management.record_recommendation_batch("S0020", _recommendation("legacy"))
    management.record_recommendation_batch(
        "S0021",
        _recommendation("linked"),
        risk_snapshot=_snapshot("S0021", 70),
    )

    analysis = effect.analyze(pd.DataFrame([_current("S9999", 40)]))

    assert analysis.records.empty
    assert analysis.total_interventions == 2
    assert analysis.legacy_without_snapshot == 1
    assert analysis.missing_current_students == 1


def test_effect_rejects_missing_snapshot_columns(tmp_path: Path) -> None:
    effect, _ = _effect_service(tmp_path / "app.db")

    with pytest.raises(ValueError, match="필요한 컬럼"):
        effect.analyze(pd.DataFrame([{"student_id": "S0001", "week": 4}]))


def test_effect_threshold_must_be_positive(tmp_path: Path) -> None:
    management = InterventionManagementService(database_path=tmp_path / "app.db")

    with pytest.raises(ValueError, match="0보다 커야"):
        InterventionEffectService(
            config={
                "intervention_effect": {
                    "meaningful_change_threshold": 0,
                    "score_change_epsilon": 0.01,
                }
            },
            intervention_service=management,
        )


def test_intervention_effect_page_renders_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    management = InterventionManagementService(database_path=database_path)
    management.record_recommendation_batch(
        "S0003",
        _recommendation("page"),
        risk_snapshot=_snapshot("S0003", 70, week=3),
    )
    st.cache_data.clear()
    st.cache_resource.clear()

    app = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "08_intervention_effect.py")
    ).run(timeout=30)

    assert not app.exception
    assert any(item.label == "조회 조건" for item in app.expander)
    assert any(
        selectbox.label == "학과" for selectbox in app.selectbox
    )
    assert any(
        selectbox.label == "지원 상태" for selectbox in app.selectbox
    )
    assert any(
        selectbox.label == "변화 구분" for selectbox in app.selectbox
    )
    assert any(text_input.label == "학생 ID" for text_input in app.text_input)
    assert any(date_input.label == "지원 시작일" for date_input in app.date_input)
    assert not any(
        selectbox.label in {"학과", "지원 상태", "변화 구분"}
        for selectbox in app.sidebar.selectbox
    )
    assert not any(
        text_input.label == "학생 ID" for text_input in app.sidebar.text_input
    )
    assert not any(
        date_input.label == "지원 시작일" for date_input in app.sidebar.date_input
    )
    assert any("적용 조건 ·" in str(item.value) for item in app.caption)
    assert any(metric.label == "평가 가능" for metric in app.metric)
    assert not any(
        "지원별 비교 결과" in element.value for element in app.subheader
    )
    app.pills[0].set_value("지원별 비교 결과").run(timeout=30)
    assert not app.exception
    assert any(
        "지원별 비교 결과" in element.value
        for element in app.subheader
    )


def test_completed_demo_support_uses_week_fifteen_synthetic_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """대표 학생 상담 완료 뒤 4주차와 15주차 개선 결과를 화면에 표시한다."""

    database_path = tmp_path / "completed-demo.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    management = InterventionManagementService(database_path=database_path)
    context = management.record_recommendation_batch(
        "S0003",
        _recommendation("completed-demo"),
        risk_snapshot=_snapshot("S0003", 74.63, week=4),
    )
    management.update_status(
        context.intervention_id,
        "상담 완료",
        staff_action="상담 및 지원 연결 완료",
    )
    st.cache_data.clear()
    st.cache_resource.clear()

    app = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "08_intervention_effect.py")
    ).run(timeout=30)

    assert not app.exception
    assert any(
        "15주차 synthetic 후속 관찰을 적용했습니다" in str(item.value)
        for item in app.success
    )
    assert any(
        metric.label == "개선" and metric.value == "1건"
        for metric in app.metric
    )

    app.pills[0].set_value("선택 지원 상세").run(timeout=30)
    assert not app.exception
    assert any(
        "현재값은 15주차 synthetic 후속 관찰" in str(item.value)
        for item in app.info
    )
    metric_by_label = {metric.label: metric.value for metric in app.metric}
    assert metric_by_label["현재 종합 위험도"] == "25.9"
    assert metric_by_label["변화 구분"] == "개선"


def test_waiting_detail_explains_how_followup_data_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """후속 데이터가 없을 때 다음 체크인과 주차 데이터 사용법을 안내한다."""

    database_path = tmp_path / "waiting.db"
    monkeypatch.setattr("src.database.DEFAULT_DATABASE_PATH", database_path)
    repository = get_default_repository()
    source = repository.get_all()
    snapshots = create_integrated_risk_service(repository).build_snapshots(source)
    latest = (
        snapshots[snapshots["student_id"] == "S0003"]
        .sort_values("week")
        .iloc[-1]
        .to_dict()
    )
    management = InterventionManagementService(database_path=database_path)
    management.record_recommendation_batch(
        "S0003",
        _recommendation("waiting"),
        risk_snapshot=latest,
    )
    st.cache_data.clear()
    st.cache_resource.clear()

    app = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "08_intervention_effect.py")
    ).run(timeout=30)
    app.pills[0].set_value("선택 지원 상세").run(timeout=30)

    assert not app.exception
    assert any(
        "학생이 `나의 체크인`을 다시 제출" in str(item.value)
        for item in app.info
    )
