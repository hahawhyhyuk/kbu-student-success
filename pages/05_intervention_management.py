"""SQLite 개입 상태와 학생 추천 피드백을 관리하는 교직원 화면."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.ai_copilot_service import AICopilotService, CopilotResult
from src.intervention_service import InterventionManagementService
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
)
from src.streamlit_ai import create_streamlit_ai_provider
from src.ui import (
    queue_ai_reveal,
    render_ai_reveal,
    render_footer,
    render_page_header,
    render_submenu,
)


@st.cache_data
def load_management_master(
    checkin_revision: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repository에서 학생, 최신 위험, 지원프로그램 master를 로드한다."""

    repository = get_default_repository()
    source = repository.get_all()
    snapshots = create_integrated_risk_service(repository).build_snapshots(source)
    latest_snapshots = (
        snapshots.sort_values(["student_id", "week"])
        .groupby("student_id", as_index=False)
        .tail(1)[["student_id", "risk_level", "primary_risk_type"]]
    )
    return source.students, latest_snapshots, source.support_programs


@st.cache_resource
def get_management_service() -> InterventionManagementService:
    """SQLite 개입 관리 서비스를 rerun 간 재사용한다."""

    return InterventionManagementService()


st.set_page_config(page_title="학생지원 진행 관리", page_icon="📋", layout="wide")
render_page_header(
    "학생지원 진행 관리",
    "지원 계획과 진행 상태를 관리합니다.",
)

students, latest_snapshots, support_programs = load_management_master(
    get_student_checkin_revision()
)
service = get_management_service()
interventions = service.list_interventions()
active_section = render_submenu(
    ("지원 현황", "지원 계획 확정", "프로그램 실행 기록", "추천 근거 · 피드백"),
    key="intervention_management_submenu",
)

if interventions.empty:
    st.info(
        "저장된 학생지원 이력이 없습니다. 'AI 맞춤 추천 통합 검토'에서 "
        "추천을 생성하면 이곳에 표시됩니다."
    )
else:
    student_master = students[["student_id", "department"]].drop_duplicates()
    management_data = interventions.merge(
        student_master,
        on="student_id",
        how="left",
        validate="many_to_one",
    )
    management_data = management_data.merge(
        latest_snapshots,
        on="student_id",
        how="left",
        validate="many_to_one",
    )
    management_data["department"] = management_data["department"].fillna(
        "학과 확인 필요"
    )
    management_data["risk_level"] = management_data["risk_level"].fillna(
        "위험등급 확인 필요"
    )
    management_data["primary_risk_type"] = management_data[
        "primary_risk_type"
    ].fillna("위험유형 확인 필요")

    with st.expander("조회 조건", expanded=True):
        (
            risk_level_column,
            risk_type_column,
            status_column,
            department_column,
        ) = st.columns(4)
        risk_level_filter = risk_level_column.selectbox(
            "위험등급",
            ("전체",) + tuple(sorted(management_data["risk_level"].unique())),
            key="management_risk_level_filter",
        )
        risk_type_filter = risk_type_column.selectbox(
            "위험유형",
            ("전체",)
            + tuple(sorted(management_data["primary_risk_type"].unique())),
            key="management_risk_type_filter",
        )
        status_filter = status_column.selectbox(
            "상태",
            ("전체",) + service.statuses,
            key="management_status_filter",
        )
        departments = tuple(sorted(management_data["department"].unique()))
        department_filter = department_column.selectbox(
            "학과",
            ("전체",) + departments,
            key="management_department_filter",
        )
        student_filter = st.text_input(
            "학생 ID 검색",
            placeholder="예: STU001",
            key="management_student_filter",
        ).strip()

    filtered = management_data.copy()
    if risk_level_filter != "전체":
        filtered = filtered[filtered["risk_level"] == risk_level_filter]
    if risk_type_filter != "전체":
        filtered = filtered[
            filtered["primary_risk_type"] == risk_type_filter
        ]
    if status_filter != "전체":
        filtered = filtered[filtered["status"] == status_filter]
    if department_filter != "전체":
        filtered = filtered[filtered["department"] == department_filter]
    if student_filter:
        filtered = filtered[
            filtered["student_id"].str.contains(
                student_filter, case=False, regex=False
            )
        ]

    if filtered.empty:
        st.warning("현재 필터에 해당하는 학생지원 기록이 없습니다.")
    else:
        row_lookup = {
            int(row["intervention_id"]): row
            for _, row in filtered.iterrows()
        }
        selected_state = st.session_state.get("management_selected_intervention")
        if selected_state not in row_lookup:
            st.session_state.pop("management_selected_intervention", None)
        with st.container(
            border=True,
            key="staff_intervention_management_selector_bar",
        ):
            selector_column, risk_column = st.columns(
                [4, 1],
                gap="large",
                vertical_alignment="bottom",
            )
            selected_intervention_id = selector_column.selectbox(
                "관리할 지원",
                list(row_lookup),
                format_func=lambda intervention_id: (
                    f"#{intervention_id} · "
                    f"{row_lookup[intervention_id]['student_id']} · "
                    f"{row_lookup[intervention_id]['department']} · "
                    f"{row_lookup[intervention_id]['status']}"
                ),
                key="management_selected_intervention",
            )
            selected = row_lookup[selected_intervention_id]
            risk_column.metric(
                "현재 위험등급",
                str(selected["risk_level"]),
            )
        selected_student_id = str(selected["student_id"])
        batch_id = str(selected["recommendation_batch_id"] or "")
        program_master = support_programs.set_index("program_id").to_dict(
            orient="index"
        )
        program_names = {
            str(program_id): str(program["program_name"])
            for program_id, program in program_master.items()
        }
        recommendations = service.list_recommendations(batch_id)
        if not recommendations.empty:
            recommendations = recommendations[
                recommendations["recommendation_type"] == "support_program"
            ].copy()
            recommendations["program_name"] = recommendations["target_id"].map(
                program_names
            ).fillna("프로그램 확인 필요")
        support_plan = service.get_support_plan(selected_intervention_id)
        support_plan_items = service.list_support_plan_items(
            selected_intervention_id
        )
        next_action_result_key = (
            f"management_ai_next_action_{selected_intervention_id}"
        )
        next_action_reveal_key = (
            f"management_ai_next_action_reveal_{selected_intervention_id}"
        )
        stored_next_action: CopilotResult | None = st.session_state.get(
            next_action_result_key
        )

        if active_section in ("지원 계획 확정", "프로그램 실행 기록"):
            with st.container(border=True, key="kbu_ai_copilot_next_action"):
                suggestion_title, suggestion_button = st.columns([4, 1])
                suggestion_title.markdown("### ✨ AI가 제안하는 다음 조치")
                suggestion_title.caption(
                    "현재 지원 상태와 프로그램 진행 상태를 바탕으로 표준 조치 중 하나를 설명합니다."
                )
                generate_next_action = suggestion_button.button(
                    "다음 조치 제안",
                    type="primary",
                    key=f"management_generate_next_action_{selected_intervention_id}",
                    width="stretch",
                )
                if generate_next_action:
                    try:
                        with st.spinner("현재 진행상태를 확인하고 다음 조치를 설명하고 있습니다..."):
                            copilot_service = AICopilotService(
                                create_streamlit_ai_provider()
                            )
                            stored_next_action = copilot_service.generate_next_action(
                                intervention_status=str(selected["status"]),
                                has_support_plan=support_plan is not None,
                                plan_item_statuses=(
                                    support_plan_items["item_status"]
                                    .dropna()
                                    .astype(str)
                                    .tolist()
                                    if not support_plan_items.empty
                                    else []
                                ),
                                allowed_actions=service.next_actions,
                            )
                            st.session_state[next_action_result_key] = (
                                stored_next_action
                            )
                            queue_ai_reveal(next_action_reveal_key)
                    except Exception as error:
                        st.error(f"AI 다음 조치를 생성하지 못했습니다. ({error})")

                if stored_next_action is not None:
                    st.info(
                        f"**제안 조치:** {stored_next_action.content['action']}"
                    )
                    render_ai_reveal(
                        str(stored_next_action.content["reason"]),
                        key=next_action_reveal_key,
                        label=(
                            f"검증 완료 · {stored_next_action.provider_name} AI 제안 근거"
                        ),
                    )
                    st.markdown(
                        "**실행 전 확인:** "
                        + " · ".join(
                            stored_next_action.content["check_before_action"]
                        )
                    )
                    if stored_next_action.warning:
                        st.warning(stored_next_action.warning)
                    st.caption(
                        "AI 제안은 자동 실행되지 않습니다. 아래에서 담당자가 검토·수정한 뒤 저장해야 반영됩니다."
                    )

        if active_section == "지원 현황":
            st.subheader("지원 현황")
            summary_columns = st.columns(3)
            summary_columns[0].metric("전체 지원", len(management_data))
            summary_columns[1].metric("필터 결과", len(filtered))
            summary_columns[2].metric(
                "진행 중",
                int(
                    management_data["status"].isin(
                        ("연락 전", "상담 예정", "상담 완료", "프로그램 참여")
                    ).sum()
                ),
            )
            display_columns = [
                "intervention_id",
                "student_id",
                "department",
                "risk_level",
                "primary_risk_type",
                "status",
                "staff_action",
                "updated_at",
            ]
            st.dataframe(
                filtered[display_columns].rename(
                    columns={
                        "intervention_id": "지원 ID",
                        "student_id": "학생 ID",
                        "department": "학과",
                        "risk_level": "위험등급",
                        "primary_risk_type": "위험유형",
                        "status": "상태",
                        "staff_action": "교직원 조치",
                        "updated_at": "최종 수정",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        elif active_section == "지원 계획 확정":
            st.subheader("추천을 실제 지원계획으로 확정")
            st.info(
                "AI 추천을 그대로 선택하거나, 학생 의사·일정·운영여건을 "
                "반영해 대학 DB에 있는 다른 프로그램으로 교체할 수 있습니다."
            )
            recommended_ids = (
                recommendations["target_id"].astype(str).tolist()
                if "target_id" in recommendations.columns
                else []
            )
            if not recommendations.empty:
                st.dataframe(
                    recommendations[
                        ["rank", "program_name", "target_id", "score", "reason"]
                    ].rename(
                        columns={
                            "rank": "순위",
                            "program_name": "AI 추천 프로그램",
                            "target_id": "프로그램 ID",
                            "score": "추천점수",
                            "reason": "추천 근거",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

            if support_plan is not None:
                st.success(
                    f"지원계획 #{int(support_plan['support_plan_id'])} · "
                    f"담당 {support_plan['assigned_staff']} · "
                    f"연락 예정 {support_plan['planned_contact_date']} · "
                    f"다음 조치 {support_plan['next_action']}"
                )
                if not support_plan_items.empty:
                    saved_items = support_plan_items.copy()
                    saved_items["프로그램"] = saved_items["program_id"].map(
                        program_names
                    ).fillna("프로그램 확인 필요")
                    st.dataframe(
                        saved_items[
                            ["프로그램", "selection_type", "replacement_reason", "item_status"]
                        ].rename(
                            columns={
                                "selection_type": "선택 구분",
                                "replacement_reason": "교체 이유",
                                "item_status": "실행 상태",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            existing_ai_selected = (
                support_plan_items[
                    support_plan_items["selection_type"] == "AI 추천 선택"
                ]["program_id"].astype(str).tolist()
                if not support_plan_items.empty
                else recommended_ids
            )
            existing_replacement = (
                support_plan_items[
                    support_plan_items["selection_type"] == "교직원 교체"
                ].iloc[0]
                if not support_plan_items.empty
                and bool((support_plan_items["selection_type"] == "교직원 교체").any())
                else None
            )
            alternative_ids = sorted(set(program_names).difference(recommended_ids))
            replacement_options = ["교체 안 함", *alternative_ids]
            replacement_default = (
                str(existing_replacement["program_id"])
                if existing_replacement is not None
                else "교체 안 함"
            )
            with st.form(f"support_plan_form_{selected_intervention_id}"):
                selected_program_ids = st.multiselect(
                    "실제 지원할 AI 추천 프로그램",
                    recommended_ids,
                    default=[item for item in existing_ai_selected if item in recommended_ids],
                    format_func=lambda program_id: program_names.get(program_id, program_id),
                )
                replacement_program_id = st.selectbox(
                    "DB 프로그램으로 교체 (선택)",
                    replacement_options,
                    index=replacement_options.index(replacement_default)
                    if replacement_default in replacement_options
                    else 0,
                    format_func=lambda program_id: (
                        program_id
                        if program_id == "교체 안 함"
                        else f"{program_names.get(program_id, program_id)} · {program_id}"
                    ),
                )
                replaced_default = (
                    str(existing_replacement["replaced_program_id"] or "")
                    if existing_replacement is not None
                    else ""
                )
                replaced_program_id = st.selectbox(
                    "교체할 기존 추천",
                    ["선택 안 함", *recommended_ids],
                    index=(
                        ["선택 안 함", *recommended_ids].index(replaced_default)
                        if replaced_default in recommended_ids
                        else 0
                    ),
                    format_func=lambda program_id: (
                        program_id
                        if program_id == "선택 안 함"
                        else program_names.get(program_id, program_id)
                    ),
                )
                replacement_reason = st.text_input(
                    "교체 이유",
                    value=(
                        str(existing_replacement["replacement_reason"])
                        if existing_replacement is not None
                        else ""
                    ),
                    placeholder="예: 학생이 대면 상담보다 주간 학습계획 프로그램을 희망함",
                )
                plan_left, plan_middle, plan_right = st.columns(3)
                assigned_staff = plan_left.text_input(
                    "담당자·담당 부서",
                    value=(
                        str(support_plan["assigned_staff"])
                        if support_plan is not None
                        else "IR센터 담당자"
                    ),
                )
                contact_default = (
                    date.fromisoformat(str(support_plan["planned_contact_date"]))
                    if support_plan is not None
                    else date.today() + timedelta(days=3)
                )
                planned_contact_date = plan_middle.date_input(
                    "학생 연락 예정일", value=contact_default
                )
                next_action_default = (
                    str(support_plan["next_action"])
                    if support_plan is not None
                    and str(support_plan["next_action"]) in service.next_actions
                    else (
                        str(stored_next_action.content["action"])
                        if stored_next_action is not None
                        and str(stored_next_action.content["action"])
                        in service.next_actions
                        else service.next_actions[0]
                    )
                )
                next_action = plan_right.selectbox(
                    "다음 조치",
                    service.next_actions,
                    index=service.next_actions.index(next_action_default),
                )
                plan_note = st.text_area(
                    "지원계획 메모",
                    value=(str(support_plan["plan_note"]) if support_plan else ""),
                    placeholder="선택·제외·교체 판단 근거와 학생 의사를 기록하세요.",
                )
                plan_submitted = st.form_submit_button(
                    "지원계획 확정·저장", type="primary"
                )
            if plan_submitted:
                try:
                    service.save_support_plan(
                        intervention_id=selected_intervention_id,
                        selected_program_ids=selected_program_ids,
                        valid_program_ids=set(program_names),
                        assigned_staff=assigned_staff,
                        planned_contact_date=planned_contact_date,
                        next_action=next_action,
                        plan_note=plan_note,
                        replacement_program_id=(
                            None
                            if replacement_program_id == "교체 안 함"
                            else replacement_program_id
                        ),
                        replaced_program_id=(
                            None
                            if replaced_program_id == "선택 안 함"
                            else replaced_program_id
                        ),
                        replacement_reason=replacement_reason,
                    )
                    st.success("지원계획을 확정했습니다. 이제 프로그램별 실행을 기록하세요.")
                    st.cache_data.clear()
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

        elif active_section == "프로그램 실행 기록":
            st.subheader("프로그램별 실행 기록")
            if support_plan is None or support_plan_items.empty:
                st.info("먼저 `지원 계획 확정`에서 실제 지원할 프로그램을 선택하세요.")
            else:
                execution_items = support_plan_items.copy()
                execution_items["프로그램"] = execution_items["program_id"].map(
                    program_names
                ).fillna("프로그램 확인 필요")
                st.dataframe(
                    execution_items[
                        ["프로그램", "item_status", "last_action", "staff_note", "updated_at"]
                    ].rename(
                        columns={
                            "item_status": "현재 상태",
                            "last_action": "최신 조치",
                            "staff_note": "기록",
                            "updated_at": "최종 수정",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
                item_lookup = {
                    int(row["plan_item_id"]): row
                    for row in execution_items.to_dict(orient="records")
                }
                with st.form(f"plan_item_form_{selected_intervention_id}"):
                    plan_item_id = st.selectbox(
                        "기록할 프로그램",
                        list(item_lookup),
                        format_func=lambda item_id: item_lookup[item_id]["프로그램"],
                    )
                    current_item_status = str(item_lookup[plan_item_id]["item_status"])
                    item_status = st.selectbox(
                        "프로그램 상태",
                        service.plan_item_statuses,
                        index=(
                            service.plan_item_statuses.index(current_item_status)
                            if current_item_status in service.plan_item_statuses
                            else 0
                        ),
                    )
                    last_action = st.text_input(
                        "최신 조치",
                        value=str(item_lookup[plan_item_id]["last_action"] or ""),
                        placeholder="예: 학생과 통화하고 상담 일정을 확정함",
                    )
                    item_note = st.text_area(
                        "실행 메모",
                        value=str(item_lookup[plan_item_id]["staff_note"] or ""),
                    )
                    item_submitted = st.form_submit_button(
                        "프로그램 실행 기록 저장", type="primary"
                    )
                if item_submitted:
                    service.update_support_plan_item(
                        plan_item_id, item_status, last_action, item_note
                    )
                    st.success("프로그램 실행 상태를 저장했습니다.")
                    st.rerun()

            st.markdown("**전체 학생지원 상태**")
            current_status = str(selected["status"])
            status_index = (
                service.statuses.index(current_status)
                if current_status in service.statuses
                else 0
            )
            with st.form(f"intervention_form_{selected_intervention_id}"):
                next_status = st.selectbox(
                    "학생지원 상태",
                    service.statuses,
                    index=status_index,
                )
                staff_action = st.text_input(
                    "교직원 조치",
                    value=str(selected["staff_action"] or ""),
                    placeholder="예: 전화 상담 일정 조율",
                )
                staff_note = st.text_area(
                    "교직원 메모",
                    value=str(selected["staff_note"] or ""),
                    placeholder="학생과 확인한 내용 및 다음 조치를 기록하세요.",
                )
                status_submitted = st.form_submit_button(
                    "학생지원 상태 저장",
                    type="primary",
                )
            if status_submitted:
                service.update_status(
                    intervention_id=selected_intervention_id,
                    status=next_status,
                    staff_action=staff_action,
                    staff_note=staff_note,
                )
                st.success("학생지원 상태와 교직원 기록을 저장했습니다.")

        else:
            st.subheader("추천 근거 · 학생 피드백")
            if not recommendations.empty:
                st.markdown("**AI 추천 원본**")
                st.dataframe(
                    recommendations[
                        ["rank", "program_name", "score", "reason"]
                    ].rename(
                        columns={
                            "rank": "순위",
                            "program_name": "프로그램",
                            "score": "추천점수",
                            "reason": "추천 근거",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            snapshot_id = selected.get("risk_snapshot_id")
            risk_snapshot = (
                service.get_risk_snapshot(int(snapshot_id))
                if pd.notna(snapshot_id)
                else None
            )
            st.markdown("**지원 시작 시점 스냅샷**")
            if risk_snapshot is None:
                st.caption("이 지원은 스냅샷 연결 기능 도입 전 기록입니다.")
            else:
                snapshot_columns = st.columns(4)
                snapshot_columns[0].metric("주차", f"{int(risk_snapshot['week'])}주")
                snapshot_columns[1].metric(
                    "종합 위험도", f"{float(risk_snapshot['overall_risk']):.1f}"
                )
                snapshot_columns[2].metric("위험등급", risk_snapshot["risk_level"])
                snapshot_columns[3].metric(
                    "주요 위험유형", risk_snapshot["primary_risk_type"]
                )

            st.markdown("**학생 추천 피드백**")
            feedback = service.list_feedback(selected_student_id)
            if feedback.empty:
                st.caption("이 학생에게 저장된 추천 피드백이 없습니다.")
            else:
                feedback["program_name"] = feedback["target_id"].map(
                    support_programs.set_index("program_id")[
                        "program_name"
                    ].to_dict()
                ).fillna("프로그램 확인 필요")
                st.dataframe(
                    feedback[
                        ["created_at", "program_name", "feedback_type", "comment"]
                    ].rename(
                        columns={
                            "created_at": "기록 시각",
                            "program_name": "프로그램",
                            "feedback_type": "학생 반응",
                            "comment": "추가 의견",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

st.divider()
st.caption(
    "학생지원 상태는 지원 진행을 위한 기록이며 학생에 대한 처벌이나 "
    "행정적 불이익에 사용하지 않습니다."
)
render_footer()
