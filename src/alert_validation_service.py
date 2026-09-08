"""초기경보 결과를 별도 참조 라벨과 비교하는 검증 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SYNTHETIC_ARCHETYPE_REFERENCE: dict[str, dict[str, Any]] = {
    "normal": {
        "reference_segment": "정상형",
        "expected_support_needed": False,
        "expected_high_risk": False,
        "expected_primary_risk_type": None,
    },
    "attendance_risk": {
        "reference_segment": "출결위험형",
        "expected_support_needed": True,
        "expected_high_risk": False,
        "expected_primary_risk_type": "출결위험형",
    },
    "engagement_risk": {
        "reference_segment": "학습참여저하형",
        "expected_support_needed": True,
        "expected_high_risk": False,
        "expected_primary_risk_type": "학습참여저하형",
    },
    "achievement_risk": {
        "reference_segment": "학업부진형",
        "expected_support_needed": True,
        "expected_high_risk": False,
        "expected_primary_risk_type": "학업부진형",
    },
    "major_mismatch": {
        "reference_segment": "전공부적응형",
        "expected_support_needed": True,
        "expected_high_risk": False,
        "expected_primary_risk_type": "전공부적응형",
    },
    "career_unclear": {
        "reference_segment": "진로미설정형",
        "expected_support_needed": True,
        "expected_high_risk": False,
        "expected_primary_risk_type": "진로미설정형",
    },
    "complex_risk": {
        "reference_segment": "복합위험형",
        "expected_support_needed": True,
        "expected_high_risk": True,
        "expected_primary_risk_type": None,
    },
}

REFERENCE_COLUMNS = frozenset(
    {
        "student_id",
        "reference_segment",
        "expected_support_needed",
        "expected_high_risk",
        "expected_primary_risk_type",
    }
)
SNAPSHOT_COLUMNS = frozenset(
    {"student_id", "week", "risk_level", "primary_risk_type"}
)


@dataclass(frozen=True)
class AlertValidationReport:
    """초기경보와 참조 라벨의 비교 결과.

    합성 archetype으로 만든 보고서는 위험엔진의 내부 동작 일치도이며 실제 학생에
    대한 예측 정확도를 뜻하지 않는다. 동일한 참조 스키마에 교직원 검토 라벨을
    넣으면 실제 데이터 검증에도 재사용할 수 있다.
    """

    student_count: int
    reference_support_count: int
    predicted_support_count: int
    agreement_rate: float
    support_signal_recall: float
    normal_pattern_specificity: float
    high_risk_recall: float | None
    primary_type_match_rate: float | None
    confusion: pd.DataFrame
    segment_breakdown: pd.DataFrame
    details: pd.DataFrame


def _percentage(numerator: int, denominator: int) -> float | None:
    """분모가 있을 때 백분율을 소수점 한 자리로 반환한다."""

    if denominator == 0:
        return None
    return round(numerator / denominator * 100, 1)


def _require_columns(
    frame: pd.DataFrame,
    required: frozenset[str],
    frame_name: str,
) -> None:
    """검증 입력에 필요한 컬럼이 모두 있는지 확인한다."""

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{frame_name} 데이터에 필수 컬럼이 없습니다: {sorted(missing)}"
        )


def build_synthetic_reference_labels(students: pd.DataFrame) -> pd.DataFrame:
    """가상학생 archetype을 초기경보 검증용 참조 라벨로 변환한다.

    Parameters:
        students: student_id와 synthetic archetype을 포함한 학생 데이터.

    Returns:
        지원 필요 여부, 고위험 여부, 기대 위험유형을 가진 참조 라벨.

    Assumptions:
        archetype은 데이터 생성 규칙이므로 결과는 독립적인 실제 정답이 아니다.
    """

    _require_columns(
        students,
        frozenset({"student_id", "archetype"}),
        "students",
    )
    if students["student_id"].astype(str).duplicated().any():
        raise ValueError("students의 student_id는 중복될 수 없습니다.")

    archetypes = students["archetype"].astype(str)
    unknown = sorted(
        set(archetypes).difference(SYNTHETIC_ARCHETYPE_REFERENCE)
    )
    if unknown:
        raise ValueError(f"알 수 없는 synthetic archetype입니다: {unknown}")

    rows: list[dict[str, Any]] = []
    for row in students[["student_id", "archetype"]].to_dict(orient="records"):
        rule = SYNTHETIC_ARCHETYPE_REFERENCE[str(row["archetype"])]
        rows.append(
            {
                "student_id": str(row["student_id"]),
                "reference_segment": rule["reference_segment"],
                "expected_support_needed": rule["expected_support_needed"],
                "expected_high_risk": rule["expected_high_risk"],
                "expected_primary_risk_type": rule[
                    "expected_primary_risk_type"
                ],
            }
        )
    return pd.DataFrame(rows)


def evaluate_initial_alerts(
    snapshots: pd.DataFrame,
    reference_labels: pd.DataFrame,
) -> AlertValidationReport:
    """최신 초기경보를 별도로 준비된 참조 라벨과 비교한다.

    Parameters:
        snapshots: 학생·주차별 위험등급과 주요 위험유형 결과.
        reference_labels: 학생별 지원 필요·고위험·위험유형 참조 라벨.

    Returns:
        이진 일치율, 포착률, 정상 보존율, 유형 일치율과 상세 비교표.

    Assumptions:
        관심·주의·고위험은 모두 지원 신호로 간주한다. 참조 라벨은 전체 학생 또는
        교직원이 검토한 일부 학생만 포함할 수 있다.
    """

    _require_columns(snapshots, SNAPSHOT_COLUMNS, "snapshots")
    _require_columns(reference_labels, REFERENCE_COLUMNS, "reference_labels")
    if reference_labels.empty:
        raise ValueError("reference_labels는 한 명 이상이어야 합니다.")

    references = reference_labels.copy()
    references["student_id"] = references["student_id"].astype(str)
    if references["student_id"].duplicated().any():
        raise ValueError("reference_labels의 student_id는 중복될 수 없습니다.")
    for column in ("expected_support_needed", "expected_high_risk"):
        if not references[column].isin([True, False]).all():
            raise ValueError(f"{column}은 bool 값이어야 합니다.")
        references[column] = references[column].astype(bool)

    prepared_snapshots = snapshots.copy()
    prepared_snapshots["student_id"] = prepared_snapshots["student_id"].astype(str)
    if prepared_snapshots[["student_id", "week"]].duplicated().any():
        raise ValueError("snapshots의 학생별 week는 중복될 수 없습니다.")
    latest = (
        prepared_snapshots.sort_values(["student_id", "week"])
        .groupby("student_id", as_index=False)
        .tail(1)
    )
    merged = references.merge(
        latest[["student_id", "week", "risk_level", "primary_risk_type"]],
        on="student_id",
        how="left",
        validate="one_to_one",
    )
    missing_predictions = merged.loc[merged["risk_level"].isna(), "student_id"]
    if not missing_predictions.empty:
        raise ValueError(
            "참조 학생의 초기경보 결과가 없습니다: "
            f"{missing_predictions.head(5).tolist()}"
        )

    merged["predicted_support_needed"] = merged["risk_level"] != "정상"
    merged["predicted_high_risk"] = merged["risk_level"] == "고위험"
    merged["support_agreement"] = (
        merged["expected_support_needed"]
        == merged["predicted_support_needed"]
    )
    type_applicable = merged["expected_primary_risk_type"].notna()
    merged["primary_type_match"] = (
        type_applicable
        & merged["predicted_support_needed"]
        & (
            merged["expected_primary_risk_type"]
            == merged["primary_risk_type"]
        )
    )

    expected_support = merged["expected_support_needed"]
    predicted_support = merged["predicted_support_needed"]
    true_positive = int((expected_support & predicted_support).sum())
    false_negative = int((expected_support & ~predicted_support).sum())
    true_negative = int((~expected_support & ~predicted_support).sum())
    false_positive = int((~expected_support & predicted_support).sum())

    expected_high = merged["expected_high_risk"]
    high_risk_matches = int(
        (expected_high & merged["predicted_high_risk"]).sum()
    )
    type_matches = int(merged["primary_type_match"].sum())
    agreement = int(merged["support_agreement"].sum())

    confusion = pd.DataFrame(
        [
            {
                "reference_group": "지원 불필요",
                "predicted_normal": true_negative,
                "predicted_support": false_positive,
            },
            {
                "reference_group": "지원 필요",
                "predicted_normal": false_negative,
                "predicted_support": true_positive,
            },
        ]
    )

    breakdown_rows: list[dict[str, Any]] = []
    for segment, group in merged.groupby("reference_segment", sort=False):
        segment_type_applicable = group["expected_primary_risk_type"].notna()
        expected_type_values = group.loc[
            segment_type_applicable, "expected_primary_risk_type"
        ].dropna().unique()
        breakdown_rows.append(
            {
                "reference_segment": segment,
                "student_count": int(len(group)),
                "expected_primary_risk_type": (
                    str(expected_type_values[0])
                    if len(expected_type_values)
                    else "해당 없음"
                ),
                "support_detection_rate": _percentage(
                    int(
                        (
                            group["expected_support_needed"]
                            & group["predicted_support_needed"]
                        ).sum()
                    ),
                    int(group["expected_support_needed"].sum()),
                ),
                "high_risk_detection_rate": _percentage(
                    int(
                        (
                            group["expected_high_risk"]
                            & group["predicted_high_risk"]
                        ).sum()
                    ),
                    int(group["expected_high_risk"].sum()),
                ),
                "primary_type_match_rate": _percentage(
                    int(group["primary_type_match"].sum()),
                    int(segment_type_applicable.sum()),
                ),
            }
        )

    segment_order = {
        str(rule["reference_segment"]): index
        for index, rule in enumerate(SYNTHETIC_ARCHETYPE_REFERENCE.values())
    }
    segment_breakdown = pd.DataFrame(breakdown_rows)
    segment_breakdown["_order"] = segment_breakdown["reference_segment"].map(
        segment_order
    ).fillna(len(segment_order))
    segment_breakdown = (
        segment_breakdown.sort_values(["_order", "reference_segment"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    detail_columns = [
        "student_id",
        "reference_segment",
        "expected_support_needed",
        "expected_high_risk",
        "expected_primary_risk_type",
        "week",
        "risk_level",
        "primary_risk_type",
        "predicted_support_needed",
        "predicted_high_risk",
        "support_agreement",
        "primary_type_match",
    ]
    return AlertValidationReport(
        student_count=int(len(merged)),
        reference_support_count=int(expected_support.sum()),
        predicted_support_count=int(predicted_support.sum()),
        agreement_rate=float(_percentage(agreement, len(merged)) or 0.0),
        support_signal_recall=float(
            _percentage(true_positive, true_positive + false_negative) or 0.0
        ),
        normal_pattern_specificity=float(
            _percentage(true_negative, true_negative + false_positive) or 0.0
        ),
        high_risk_recall=_percentage(high_risk_matches, int(expected_high.sum())),
        primary_type_match_rate=_percentage(
            type_matches, int(type_applicable.sum())
        ),
        confusion=confusion,
        segment_breakdown=segment_breakdown,
        details=merged[detail_columns].reset_index(drop=True),
    )
