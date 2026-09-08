"""전체 학생의 최신 초기경보 현황을 보여주는 교직원 대시보드."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.access_control import DEMO_STAFF_ACCOUNT_ID
from src.dashboard_period_service import (
    DASHBOARD_REFERENCE_WEEK_KEY,
    DashboardPeriodService,
)
from src.dashboard_priority_service import DashboardPriorityService
from src.demo_alert_notification_service import (
    STATUS_CHECKIN_COMPLETED,
    DemoAlertNotificationService,
)
from src.demo_service import DemoScenarioService
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
    select_latest_student_snapshots,
)
from src.ui import (
    render_footer,
    render_page_header,
    render_submenu,
)


RISK_LEVEL_ORDER = ["정상", "관심", "주의", "고위험"]
RISK_LEVEL_COLORS = {
    "정상": "#2E8B57",
    "관심": "#D4A72C",
    "주의": "#E67E22",
    "고위험": "#C0392B",
}


@st.cache_data
def load_dashboard_snapshots(checkin_revision: int) -> pd.DataFrame:
    """설정된 Repository의 학생·주차별 위험 스냅샷을 캐시한다."""

    service = create_integrated_risk_service(get_default_repository())
    return service.build_snapshots()


@st.cache_resource
def get_dashboard_priority_service() -> DashboardPriorityService:
    """SQLite 최신 개입 상태와 위험 스냅샷을 결합하는 서비스를 재사용한다."""

    return DashboardPriorityService()


@st.cache_resource
def get_dashboard_demo_service() -> DemoScenarioService:
    """원본 데이터를 변경하지 않는 synthetic 주차 재생 설정을 재사용한다."""

    return DemoScenarioService()


@st.cache_resource
def get_dashboard_period_service() -> DashboardPeriodService:
    """실제 데이터 기준 주차 조회 서비스를 재사용한다."""

    return DashboardPeriodService()


st.set_page_config(page_title="전체 대시보드", page_icon="📊", layout="wide")
render_page_header(
    "전체 대시보드",
    "학생 위험 신호와 지원 우선순위를 확인합니다.",
)

checkin_revision = get_student_checkin_revision()
all_snapshots = load_dashboard_snapshots(checkin_revision)
demo_service = get_dashboard_demo_service()
period_service = get_dashboard_period_service()
alert_notification_service = DemoAlertNotificationService()
demo_alert_context: tuple[object, object, pd.Series, object] | None = None
week_labels = list(period_service.week_labels(all_snapshots))
latest_available_week = period_service.latest_week(all_snapshots)
latest_week_label = f"{latest_available_week}주차"
if st.session_state.get(DASHBOARD_REFERENCE_WEEK_KEY) not in week_labels:
    st.session_state.pop(DASHBOARD_REFERENCE_WEEK_KEY, None)
with st.container(border=True, key="dashboard_reference_period"):
    week_column, department_column, level_column = st.columns(
        [1, 1.4, 2.6],
        gap="large",
        vertical_alignment="bottom",
    )
    selected_week_label = week_column.selectbox(
        "조회 기준 주차",
        week_labels,
        index=week_labels.index(latest_week_label),
        key=DASHBOARD_REFERENCE_WEEK_KEY,
        help="현재 데이터에 실제 존재하는 주차만 표시합니다.",
    )
    selected_reference_week = period_service.resolve_week(
        selected_week_label,
        all_snapshots,
    )
    snapshots = period_service.filter_snapshots_through_week(
        all_snapshots,
        selected_reference_week,
    )
    latest_unfiltered = select_latest_student_snapshots(snapshots)
    department_options = ["전체"] + sorted(
        latest_unfiltered["department"].unique().tolist()
    )
    selected_department = department_column.selectbox(
        "학과",
        department_options,
        key="dashboard_department_filter",
    )
    selected_levels = level_column.multiselect(
        "위험등급",
        RISK_LEVEL_ORDER,
        default=RISK_LEVEL_ORDER,
        key="dashboard_risk_level_filter",
    )
    coverage = period_service.coverage(
        all_snapshots,
        selected_reference_week,
    )
    coverage_parts = [f"해당 주차 입력 {coverage.current_week_count:,}명"]
    if coverage.carried_forward_count:
        coverage_parts.append(
            f"직전 자료 유지 {coverage.carried_forward_count:,}명"
        )
    if coverage.unavailable_student_count:
        coverage_parts.append(
            f"아직 자료 없음 {coverage.unavailable_student_count:,}명"
        )
    st.caption(f"데이터 상태 · {' · '.join(coverage_parts)}")
    st.caption(
        "선택 주차까지 입력된 학생별 최신 자료로 당시 위험 현황을 조회합니다. "
        "원본 데이터와 위험점수는 변경하지 않습니다."
    )
latest = latest_unfiltered.copy()
if demo_service.enabled:
    representative_rows = latest[
        latest["student_id"].astype(str)
        == demo_service.representative_student_id
    ]
    if not representative_rows.empty:
        representative = representative_rows.iloc[0]
        observation_summary = (
            f"{selected_reference_week}주차 · "
            f"대표 학생 {demo_service.representative_student_id} · "
            f"{representative['risk_level']} · 종합 위험도 "
            f"{float(representative['overall_risk']):.1f} "
            f"(전주 대비 {float(representative['risk_change']):+.1f})"
        )
        st.info(observation_summary)
submitted_count = int(
    (latest["checkin_source"] == "student_submission").sum()
)
if submitted_count:
    st.caption(
        f"학생용 화면에서 직접 제출한 최신 체크인 {submitted_count}건을 "
        "현재 위험분석에 반영했습니다."
    )

if selected_department != "전체":
    latest = latest[latest["department"] == selected_department]
if selected_levels:
    latest = latest[latest["risk_level"].isin(selected_levels)]
else:
    latest = latest.iloc[0:0]

active_section = render_submenu(
    (
        "위험등급 분포",
        "위험유형 분포",
        "학과별 위험학생 수",
        "주차별 평균 위험도 변화",
        "우선 확인 학생",
        "아직 개입되지 않은 고위험 학생",
    ),
    key="dashboard_submenu",
)

level_counts = latest["risk_level"].value_counts()
metric_columns = st.columns(5)
metric_columns[0].metric("전체학생", len(latest))
for column, level in zip(metric_columns[1:], RISK_LEVEL_ORDER):
    column.metric(level, int(level_counts.get(level, 0)))

if demo_service.enabled:
    alert_candidates = demo_service.find_alert_candidates(latest)
    if alert_candidates.empty:
        st.caption(
            f"{selected_reference_week}주차 조회 기준에는 고위험 경보 대상이 없습니다."
        )
    else:
        st.warning(
            f"{selected_reference_week}주차 조회 기준 조기경보 대상 "
            f"{len(alert_candidates)}명"
        )
        if selected_reference_week < latest_available_week:
            st.caption(
                "과거 주차는 조회 전용입니다. 학생 안내는 최신 주차에서만 "
                "발송할 수 있습니다."
            )
        else:
            current_week_candidates = alert_candidates[
                pd.to_numeric(alert_candidates["week"], errors="coerce").eq(
                    selected_reference_week
                )
            ]
            if current_week_candidates.empty:
                st.caption(
                    "최신 주차에 새로 입력된 고위험 자료가 없어 안내 발송을 "
                    "대기합니다."
                )
            else:
                alert_candidate = current_week_candidates.iloc[0]
                valid_student_ids = all_snapshots["student_id"].astype(str).unique()
                preview = alert_notification_service.build_preview(
                    alert_candidate.to_dict(),
                    valid_student_ids,
                )
                notification = alert_notification_service.get_notification(
                    preview.student_id,
                    preview.alert_week,
                )
                demo_alert_context = (
                    preview,
                    notification,
                    alert_candidate,
                    valid_student_ids,
                )

dashboard_unaddressed = (
    get_dashboard_priority_service().find_unaddressed_high_risk(latest)
)

if active_section == "위험등급 분포":
    st.subheader("위험등급 분포")
    level_frame = pd.DataFrame(
        {
            "위험등급": RISK_LEVEL_ORDER,
            "학생 수": [int(level_counts.get(level, 0)) for level in RISK_LEVEL_ORDER],
        }
    )
    level_figure = px.bar(
        level_frame,
        x="위험등급",
        y="학생 수",
        color="위험등급",
        category_orders={"위험등급": RISK_LEVEL_ORDER},
        color_discrete_map=RISK_LEVEL_COLORS,
        text_auto=True,
    )
    level_figure.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(level_figure, width="stretch")

elif active_section == "위험유형 분포":
    st.subheader("위험유형 분포")
    risk_type_frame = (
        latest[latest["risk_level"] != "정상"]
        .groupby("risk_type", as_index=False)
        .size()
        .rename(columns={"risk_type": "위험유형", "size": "학생 수"})
        .sort_values("학생 수", ascending=False)
    )
    if risk_type_frame.empty:
        st.info("선택한 조건에 위험학생이 없습니다.")
    else:
        type_figure = px.bar(
            risk_type_frame,
            x="학생 수",
            y="위험유형",
            orientation="h",
            text_auto=True,
            color="학생 수",
            color_continuous_scale="Blues",
        )
        type_figure.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(type_figure, width="stretch")

elif active_section == "학과별 위험학생 수":
    st.subheader("학과별 위험학생 수")
    at_risk = latest[latest["risk_level"] != "정상"]
    department_frame = (
        at_risk.groupby("department", as_index=False)
        .size()
        .rename(columns={"department": "학과", "size": "학생 수"})
    )
    if department_frame.empty:
        st.info("선택한 조건에 위험학생이 없습니다.")
    else:
        department_figure = px.bar(
            department_frame,
            x="학과",
            y="학생 수",
            text_auto=True,
            color="학생 수",
            color_continuous_scale="Oranges",
        )
        department_figure.update_layout(
            coloraxis_showscale=False, margin=dict(t=20, b=20)
        )
        st.plotly_chart(department_figure, width="stretch")

elif active_section == "주차별 평균 위험도 변화":
    st.subheader("주차별 평균 위험도 변화")
    visible_students = set(latest["student_id"])
    trend_source = snapshots[snapshots["student_id"].isin(visible_students)]
    weekly_frame = trend_source.groupby("week", as_index=False)[
        "overall_risk"
    ].mean()
    weekly_figure = px.line(
        weekly_frame,
        x="week",
        y="overall_risk",
        markers=True,
        labels={"week": "주차", "overall_risk": "평균 위험도"},
    )
    weekly_figure.update_xaxes(dtick=1)
    weekly_figure.update_yaxes(range=[0, 100])
    weekly_figure.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(weekly_figure, width="stretch")

elif active_section == "우선 확인 학생":
    st.subheader("우선 확인 학생")
    priority_columns = [
        "student_id",
        "department",
        "overall_risk",
        "risk_level",
        "primary_risk_type",
        "risk_change",
    ]
    priority = latest.sort_values(
        ["overall_risk", "risk_change"], ascending=[False, False]
    )[priority_columns].head(30)
    st.dataframe(
        priority,
        width="stretch",
        hide_index=True,
        column_config={
            "student_id": "학생 ID",
            "department": "학과",
            "overall_risk": st.column_config.NumberColumn(
                "전체 위험도", format="%.1f"
            ),
            "risk_level": "위험등급",
            "primary_risk_type": "주요 위험유형",
            "risk_change": st.column_config.NumberColumn(
                "전주 대비", format="%+.1f"
            ),
        },
    )

else:
    st.subheader("아직 개입되지 않은 고위험 학생")
    unaddressed = dashboard_unaddressed
    if unaddressed.empty:
        st.success("현재 선택 조건에는 지원 시작 전 상태의 고위험 학생이 없습니다.")
    else:
        unaddressed_columns = [
            "student_id",
            "department",
            "overall_risk",
            "primary_risk_type",
            "risk_change",
            "intervention_status",
            "attention_reason",
        ]
        st.dataframe(
            unaddressed[unaddressed_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "student_id": "학생 ID",
                "department": "학과",
                "overall_risk": st.column_config.NumberColumn(
                    "전체 위험도", format="%.1f"
                ),
                "primary_risk_type": "주요 위험유형",
                "risk_change": st.column_config.NumberColumn(
                    "전주 대비", format="%+.1f"
                ),
                "intervention_status": "최신 개입 상태",
                "attention_reason": "확인 사유",
            },
        )

if demo_alert_context is not None:
    preview, notification, representative, valid_student_ids = demo_alert_context
    status_label = notification.status if notification is not None else "발송 전"
    with st.expander(
        f"학생 안내 보내기 · {preview.student_id} · {status_label}",
        expanded=False,
    ):
        action_column, status_column = st.columns(
            [4, 1], vertical_alignment="center"
        )
        with action_column:
            st.markdown(f"**{preview.subject}**")
            st.write(preview.body)
        status_column.metric("연결 상태", status_label)
        st.caption(
            f"{preview.destination_label}에 저장되는 시연용 교내 알림이며, "
            "실제 이메일 주소는 사용하지 않습니다."
        )
        if notification is None:
            if st.button(
                "시연용 안내 발송",
                type="primary",
                key=f"send_demo_alert_{preview.student_id}_{preview.alert_week}",
            ):
                send_result = alert_notification_service.send(
                    representative.to_dict(),
                    valid_student_ids,
                    created_by=DEMO_STAFF_ACCOUNT_ID,
                )
                if send_result.created:
                    st.success(
                        "시연용 교내 알림을 저장했습니다. "
                        f"학생 계정 `{preview.student_id}`로 전환해 확인하세요."
                    )
                else:
                    st.info("이미 저장된 안내를 사용합니다.")
                st.rerun()
        elif notification.status == STATUS_CHECKIN_COMPLETED:
            st.success("학생이 Kare 체크인을 완료했습니다.")
        else:
            st.info(
                f"현재 `{notification.status}` 상태입니다. "
                f"학생 계정 `{preview.student_id}`에서 다음 단계를 진행하세요."
            )
render_footer()
