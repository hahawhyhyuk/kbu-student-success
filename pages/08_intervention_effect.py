"""개입 생성 시점과 최신 위험도 변화를 보여주는 교직원 대시보드."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from src.demo_followup_service import DemoFollowupService
from src.intervention_effect_service import (
    DOMAIN_COLUMNS,
    InterventionEffectService,
)
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
)
from src.ui import render_footer, render_page_header, render_submenu


@st.cache_data
def load_current_snapshots(checkin_revision: int) -> pd.DataFrame:
    """Repository와 학생 최신 체크인을 반영한 위험 결과를 로드한다."""

    repository = get_default_repository()
    source = repository.get_all()
    return create_integrated_risk_service(repository).build_snapshots(source)


@st.cache_resource
def get_effect_service() -> InterventionEffectService:
    """SQLite 개입 기록을 사용하는 비교 서비스를 재사용한다."""

    return InterventionEffectService()


def _filter_date_range(
    records: pd.DataFrame,
    selected_range: date | tuple[date, ...],
) -> pd.DataFrame:
    """Streamlit date_input 결과를 안전한 생성일 범위로 적용한다."""

    if isinstance(selected_range, tuple):
        if len(selected_range) != 2:
            return records
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range
    return records[
        records["intervention_date"].between(start_date, end_date)
    ]


st.set_page_config(
    page_title="지원 후 변화 확인",
    page_icon="📈",
    layout="wide",
)
render_page_header(
    "지원 후 변화 확인",
    "지원 전후 위험 신호의 변화를 확인합니다.",
)

with st.expander("변화량 해석 기준", expanded=False):
    st.warning(
        "변화량은 후속 관찰 지표이며 해당 지원의 인과효과를 의미하지 않습니다. "
        "학생과 교직원의 후속 확인이 필요합니다."
    )
active_section = render_submenu(
    ("변화 요약", "변화 차트", "지원별 비교 결과", "선택 지원 상세"),
    key="intervention_effect_submenu",
)

effect_service = get_effect_service()
base_current_snapshots = load_current_snapshots(get_student_checkin_revision())
followup_service = DemoFollowupService(get_default_repository())
followup_application = followup_service.apply(
    base_current_snapshots,
    effect_service.intervention_service.list_interventions(),
)
current_snapshots = followup_application.snapshots
analysis = effect_service.analyze(current_snapshots)
records = analysis.records.copy()

if followup_application.applied_student_ids:
    applied_students = ", ".join(followup_application.applied_student_ids)
    st.success(
        f"{followup_service.followup_week}주차 synthetic 후속 관찰을 적용했습니다: "
        f"{applied_students}. 실제 학생 성과가 아니라 시연용 변화 데이터입니다."
    )

if records.empty:
    if analysis.total_interventions == 0:
        st.info(
            "저장된 학생지원 기록이 없습니다. `AI 맞춤 추천 통합 검토`에서 "
            "추천을 생성하면 개입 시점 스냅샷이 함께 저장됩니다."
        )
    else:
        st.info(
            "현재 비교할 수 있는 스냅샷 연결 개입이 없습니다. "
            "기능 도입 전 개입은 당시 위험도를 소급 추정하지 않습니다."
        )
        st.caption(
            f"전체 개입 {analysis.total_interventions}건 · "
            f"스냅샷 없는 기존 개입 {analysis.legacy_without_snapshot}건 · "
            f"현재 학생 데이터 미확인 {analysis.missing_current_students}건"
        )
else:
    records["intervention_datetime"] = pd.to_datetime(
        records["intervention_created_at"], errors="coerce"
    )
    records["intervention_date"] = records["intervention_datetime"].dt.date
    valid_dates = records["intervention_date"].dropna()

    effect_order = ("개선", "유지", "악화", "후속 데이터 대기")
    available_effects = tuple(
        item for item in effect_order if item in set(records["effect_status"])
    )
    with st.expander("조회 조건", expanded=True):
        (
            department_column,
            status_column,
            effect_column,
            student_column,
        ) = st.columns(4)
        department_filter = department_column.selectbox(
            "학과",
            ("전체",) + tuple(sorted(records["department"].dropna().unique())),
            key="effect_department_filter",
        )
        status_filter = status_column.selectbox(
            "지원 상태",
            ("전체",) + tuple(sorted(records["status"].dropna().unique())),
            key="effect_status_filter",
        )
        effect_filter = effect_column.selectbox(
            "변화 구분",
            ("전체",) + available_effects,
            key="effect_change_filter",
        )
        student_filter = student_column.text_input(
            "학생 ID",
            placeholder="예: S0003",
            key="effect_student_filter",
        ).strip()
        selected_dates: date | tuple[date, ...] | None = None
        if not valid_dates.empty:
            minimum_date = min(valid_dates)
            maximum_date = max(valid_dates)
            selected_dates = st.date_input(
                "지원 시작일",
                value=(minimum_date, maximum_date),
                min_value=minimum_date,
                max_value=maximum_date,
                key="effect_date_filter",
            )

    filtered = records.copy()
    if selected_dates is not None:
        filtered = _filter_date_range(filtered, selected_dates)
    if department_filter != "전체":
        filtered = filtered[filtered["department"] == department_filter]
    if status_filter != "전체":
        filtered = filtered[filtered["status"] == status_filter]
    if effect_filter != "전체":
        filtered = filtered[filtered["effect_status"] == effect_filter]
    if student_filter:
        filtered = filtered[
            filtered["student_id"].astype(str).str.contains(
                student_filter,
                case=False,
                regex=False,
            )
        ]

    filter_summary = [
        f"학과 {department_filter}",
        f"지원 상태 {status_filter}",
        f"변화 구분 {effect_filter}",
        f"학생 {student_filter or '전체'}",
    ]
    if selected_dates is None:
        filter_summary.append("지원 시작일 전체")
    elif isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        filter_summary.append(
            f"지원 시작일 {selected_dates[0]}~{selected_dates[1]}"
        )
    else:
        filter_summary.append(f"지원 시작일 {selected_dates}")
    st.caption("적용 조건 · " + " · ".join(filter_summary))

    status_counts = filtered["effect_status"].value_counts()
    evaluated_count = int(filtered["has_followup_data"].sum())
    if active_section == "변화 요약":
        st.subheader("지원 후 변화 요약")
        summary_columns = st.columns(5)
        summary_columns[0].metric("평가 가능", f"{evaluated_count}건")
        summary_columns[1].metric(
            "개선", f"{int(status_counts.get('개선', 0))}건"
        )
        summary_columns[2].metric(
            "유지", f"{int(status_counts.get('유지', 0))}건"
        )
        summary_columns[3].metric(
            "악화", f"{int(status_counts.get('악화', 0))}건"
        )
        summary_columns[4].metric(
            "후속 데이터 대기",
            f"{int(status_counts.get('후속 데이터 대기', 0))}건",
        )
        st.caption(
            f"필터 결과 {len(filtered)}건 · 전체 지원 {analysis.total_interventions}건 · "
            f"스냅샷 없는 기존 지원 {analysis.legacy_without_snapshot}건 · "
            f"의미 있는 변화 기준 ±{effect_service.meaningful_change_threshold:.1f}점"
        )

    if filtered.empty:
        st.warning("현재 필터에 해당하는 개입 기록이 없습니다.")
    elif active_section == "변화 차트":
        st.subheader("변화 차트")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**변화 구분별 개입 현황**")
            count_data = (
                filtered["effect_status"]
                .value_counts()
                .reindex(effect_order, fill_value=0)
                .rename_axis("변화 구분")
                .reset_index(name="개입 건수")
            )
            status_chart = px.bar(
                count_data,
                x="변화 구분",
                y="개입 건수",
                text="개입 건수",
                color="변화 구분",
                color_discrete_map={
                    "개선": "#006AB6",
                    "유지": "#7B8C9E",
                    "악화": "#EC008C",
                    "후속 데이터 대기": "#C9D4DE",
                },
                category_orders={"변화 구분": list(effect_order)},
            )
            status_chart.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=15, b=10),
                yaxis_title="개입 건수",
            )
            st.plotly_chart(status_chart, width="stretch")

        with chart_right:
            st.markdown("**평균 영역별 위험도 변화**")
            evaluated = filtered[filtered["has_followup_data"]].copy()
            if evaluated.empty:
                st.info("새 주차 또는 새 체크인이 쌓이면 평균 변화가 표시됩니다.")
            else:
                average_changes = pd.DataFrame(
                    {
                        "영역": [label for _, label in DOMAIN_COLUMNS],
                        "평균 변화": [
                            float(evaluated[f"{column}_change"].mean())
                            for column, _ in DOMAIN_COLUMNS
                        ],
                    }
                )
                domain_chart = px.bar(
                    average_changes,
                    x="영역",
                    y="평균 변화",
                    text_auto=".1f",
                    color_discrete_sequence=["#006AB6"],
                )
                domain_chart.add_hline(y=0, line_color="#60758A", line_width=1)
                domain_chart.update_layout(
                    margin=dict(l=10, r=10, t=15, b=10),
                    yaxis_title="점수 변화(음수=개선)",
                )
                st.plotly_chart(domain_chart, width="stretch")

    elif active_section == "지원별 비교 결과":
        st.subheader("지원별 비교 결과")
        display = filtered[
            [
                "intervention_id",
                "student_id",
                "department",
                "status",
                "baseline_week",
                "current_week",
                "baseline_overall_risk",
                "current_overall_risk",
                "overall_change",
                "effect_status",
                "intervention_created_at",
            ]
        ].rename(
            columns={
                "intervention_id": "지원 ID",
                "student_id": "학생 ID",
                "department": "학과",
                "status": "지원 상태",
                "baseline_week": "지원 시작 주차",
                "current_week": "현재 주차",
                "baseline_overall_risk": "지원 시작 시점",
                "current_overall_risk": "현재",
                "overall_change": "변화(음수=개선)",
                "effect_status": "변화 구분",
                "intervention_created_at": "지원 생성 시각",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)

    elif active_section == "선택 지원 상세":
        st.subheader("선택 지원 영역별 상세")
        row_lookup = {
            int(row["intervention_id"]): row
            for _, row in filtered.iterrows()
        }
        selected_intervention_id = st.selectbox(
            "확인할 지원",
            list(row_lookup),
            format_func=lambda intervention_id: (
                f"#{intervention_id} · "
                f"{row_lookup[intervention_id]['student_id']} · "
                f"{row_lookup[intervention_id]['effect_status']}"
            ),
        )
        selected = row_lookup[selected_intervention_id]
        if not bool(selected["has_followup_data"]):
            st.info(
                "**현재는 지원 전·후를 비교할 새 데이터가 없습니다.**  "
                "학생이 `나의 체크인`을 다시 제출하거나, 다음 주차 출결·LMS·성취 "
                "데이터가 들어오면 자동으로 `개선·유지·악화`를 계산합니다."
            )
            if str(selected["student_id"]) == followup_service.representative_student_id:
                eligible_labels = " · ".join(sorted(followup_service.eligible_statuses))
                st.caption(
                    "시연에서는 `학생지원 진행 관리`에서 학생지원 상태를 "
                    f"{eligible_labels} 중 하나로 기록하면 "
                    f"{followup_service.followup_week}주차 synthetic 후속 관찰이 표시됩니다."
                )
        elif str(selected["student_id"]) in set(
            followup_application.applied_student_ids
        ):
            st.info(
                f"이 비교의 현재값은 {followup_service.followup_week}주차 synthetic "
                "후속 관찰입니다. 변화는 지원 후 점검 시연용이며 인과효과를 의미하지 않습니다."
            )
        detail_columns = st.columns(4)
        detail_columns[0].metric(
            "지원 시작 시점 종합 위험도",
            f"{float(selected['baseline_overall_risk']):.1f}",
        )
        detail_columns[1].metric(
            "현재 종합 위험도",
            f"{float(selected['current_overall_risk']):.1f}",
        )
        detail_columns[2].metric(
            "종합 변화",
            f"{float(selected['overall_change']):+.1f}",
            help="음수는 위험도 감소, 양수는 증가를 의미합니다.",
        )
        detail_columns[3].metric("변화 구분", selected["effect_status"])

        threshold = effect_service.meaningful_change_threshold
        domain_detail_rows = []
        for column, label in DOMAIN_COLUMNS:
            change = float(selected[f"{column}_change"])
            if not bool(selected["has_followup_data"]):
                interpretation = "후속 데이터 대기"
            elif change <= -threshold:
                interpretation = "개선"
            elif change >= threshold:
                interpretation = "악화"
            else:
                interpretation = "유지"
            domain_detail_rows.append(
                {
                    "영역": label,
                    "개입 시점": float(selected[f"baseline_{column}"]),
                    "현재": float(selected[f"current_{column}"]),
                    "변화": change,
                    "해석": interpretation,
                }
            )
        st.dataframe(
            pd.DataFrame(domain_detail_rows).round(2),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"위험등급: {selected['baseline_risk_level']} → "
            f"{selected['current_risk_level']} · 주요 위험유형: "
            f"{selected['baseline_primary_risk_type']} → "
            f"{selected['current_primary_risk_type']}"
        )

st.divider()
st.caption(
    "위험도 비교는 지원 우선순위와 후속 확인을 위한 참고정보입니다. "
    "최종 판단은 학생의 현재 상황과 의사를 확인한 교직원이 수행합니다."
)
render_footer()
