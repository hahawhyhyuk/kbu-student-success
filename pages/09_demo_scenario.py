"""발표 전 synthetic 데이터와 역할별 시연 계정을 점검하는 준비 화면."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.course_description_import_service import (
    CourseDescriptionEnrichmentReport,
    find_default_course_description_file,
    run_course_description_enrichment_dry_run,
)
from src.course_offering_import_service import (
    CourseOfferingDryRunResult,
    CourseRecommendationCatalogResult,
    build_course_recommendation_catalog,
    find_default_course_offering_files,
    run_course_offering_dry_run,
)
from src.dashboard_period_service import DASHBOARD_REFERENCE_WEEK_KEY
from src.demo_service import DEMO_OBSERVATION_WEEK_KEY, DemoScenarioService
from src.master_activation_service import (
    MasterActivationResult,
    MasterActivationService,
    prepare_activation_ready_courses,
    validate_department_activation_data,
)
from src.master_import_service import (
    MasterDryRunResult,
    MasterImportDryRunReport,
    run_master_import_dry_run,
)
from src.repositories import get_default_repository
from src.risk_service import (
    RiskAnalysisService,
    create_integrated_risk_service,
    get_student_checkin_revision,
)
from src.streamlit_ai import resolve_streamlit_gemini_api_key
from src.synthetic_showcase_service import (
    SyntheticShowcase,
    build_synthetic_showcases,
)
from src.ui import render_footer, render_page_header
from src.utils import PROJECT_ROOT


@st.cache_resource
def get_demo_service() -> DemoScenarioService:
    """시연 설정과 SQLite 초기화 서비스를 재사용한다."""

    return DemoScenarioService()


@st.cache_data
def load_demo_student_status(
    student_id: str,
    checkin_revision: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
]:
    """대표 학생의 기본 synthetic 스냅샷과 현재 최신 상태를 반환한다."""

    repository = get_default_repository()
    source = repository.get_all()
    student_rows = source.students[
        source.students["student_id"].astype(str) == str(student_id)
    ]
    if student_rows.empty:
        raise ValueError(f"시연 대표 학생이 없습니다: {student_id}")

    baseline_snapshots = RiskAnalysisService(repository).build_snapshots(source)
    baseline_rows = baseline_snapshots[
        baseline_snapshots["student_id"].astype(str) == str(student_id)
    ].sort_values("week")
    current_snapshots = create_integrated_risk_service(
        repository
    ).build_snapshots(source)
    current_rows = current_snapshots[
        current_snapshots["student_id"].astype(str) == str(student_id)
    ].sort_values("week")
    if baseline_rows.empty or current_rows.empty:
        raise ValueError(f"시연 학생 위험도가 없습니다: {student_id}")
    return (
        student_rows.iloc[0].to_dict(),
        baseline_rows.iloc[-1].to_dict(),
        current_rows.iloc[-1].to_dict(),
        tuple(baseline_rows.to_dict(orient="records")),
    )


@st.cache_data
def load_synthetic_showcases(
    representative_student_id: str,
) -> tuple[SyntheticShowcase, ...]:
    """200명의 기본 4주차 데이터에서 대표 사례 8명을 선정한다."""

    repository = get_default_repository()
    source = repository.get_all()
    snapshots = RiskAnalysisService(repository).build_snapshots(source)
    return build_synthetic_showcases(
        source.students,
        snapshots,
        representative_student_id,
    )


def clear_demo_session_results() -> None:
    """DB 초기화 후 브라우저에 남은 시연 결과와 대화도 정리한다."""

    prefixes = (
        "student_checkin_",
        "student_journey_result_",
        "ai_intervention_result_",
        "learning_path_result_",
    )
    for key in list(st.session_state):
        if str(key).startswith(prefixes):
            st.session_state.pop(key, None)
    for key in (
        "demo_initial_preset_enabled",
        "demo_followup_preset_enabled",
        "demo_microdegree_walkthrough",
        DASHBOARD_REFERENCE_WEEK_KEY,
        DEMO_OBSERVATION_WEEK_KEY,
    ):
        st.session_state.pop(key, None)


def render_demo_reset(
    demo_service: DemoScenarioService,
    record_counts: dict[str, int],
) -> None:
    """발표 시작 영역에서 안전 확인을 거쳐 SQLite 시연 기록을 초기화한다."""

    record_total = sum(record_counts.values())
    with st.expander("시연 기록 초기화", expanded=False):
        st.caption(
            f"현재 시연 기록 {record_total}행 · 학생 체크인 "
            f"{record_counts['student_checkins']} · 개입 {record_counts['interventions']} · "
            f"학교 알림 {record_counts['demo_alert_notifications']}"
        )
        st.warning(
            "초기화 직전 SQLite 백업을 `database/backups/`에 자동 생성합니다. "
            "학교 알림·체크인·AI 분석·추천·개입·피드백·학습경로·교육과정 후보만 "
            "삭제하며 synthetic CSV·설정·소스코드는 변경하지 않습니다."
        )
        with st.form("demo_reset_form"):
            reset_acknowledged = st.checkbox(
                "백업 후 SQLite 시연 기록이 삭제되는 것을 확인했습니다."
            )
            confirmation_text = st.text_input(
                f"확인을 위해 `{demo_service.reset_confirmation_text}`를 입력하세요."
            )
            reset_submitted = st.form_submit_button("백업 후 시연 기록 초기화")
        if reset_submitted:
            if not reset_acknowledged:
                st.error("초기화 내용 확인에 동의해야 합니다.")
            else:
                try:
                    result = demo_service.reset_demo_records(confirmation_text)
                    clear_demo_session_results()
                    st.cache_data.clear()
                    st.session_state["demo_reset_notice"] = {
                        "backup_path": str(result.backup_path),
                        "deleted_total": result.deleted_total,
                    }
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))


def render_master_result(result: MasterDryRunResult) -> None:
    """개인정보 없는 master dry-run 상세를 한 expander에 표시한다."""

    status = "적용 가능" if result.ready else "보완 필요"
    with st.expander(
        f"{result.display_name} · {status} · 안전 후보 {result.accepted_row_count:,}행",
        expanded=not result.ready,
    ):
        st.caption(f"원본 파일: {result.source_name}")
        metrics = st.columns(4)
        metrics[0].metric("원본 행", f"{result.source_row_count:,}")
        metrics[1].metric("안전 후보", f"{result.accepted_row_count:,}")
        metrics[2].metric("개인정보 열 차단", len(result.blocked_columns))
        metrics[3].metric("제외한 기타 열", result.ignored_column_count)
        for issue in result.issues:
            renderer = {
                "error": st.error,
                "warning": st.warning,
                "info": st.info,
            }.get(issue.severity, st.write)
            renderer(issue.message)
        if not result.preview.empty:
            st.caption("허용 열만 사용한 안전 미리보기 · 최대 5행")
            st.dataframe(result.preview, hide_index=True, use_container_width=True)


def render_master_dry_run_report(report: MasterImportDryRunReport) -> None:
    """세 master 통합 판정과 파일별 상세를 표시한다."""

    if report.ready:
        st.success(
            "세 master가 안전 점검을 통과했습니다. 아직 앱 데이터에는 적용하지 않았습니다."
        )
    else:
        st.warning(
            "일부 master는 보완이 필요합니다. 기존 synthetic master와 현재 앱 데이터는 "
            "그대로 유지됩니다."
        )
    st.dataframe(report.summary_frame(), hide_index=True, use_container_width=True)
    for result in report.results:
        render_master_result(result)


def render_course_offering_dry_run_report(
    report: CourseOfferingDryRunResult,
) -> None:
    """전 학과 실제 개설강좌의 분반 통합·master 연결 현황을 표시한다."""

    if report.ready:
        st.success(
            "두 학기 개설강좌 연결 점검을 완료했습니다. 아직 추천 데이터에는 "
            "적용하지 않았습니다."
        )
    else:
        st.warning("개설강좌 파일의 필수 열을 확인해 주세요.")
    metrics = st.columns(4)
    metrics[0].metric("유효 분반", f"{report.valid_section_count:,}")
    metrics[1].metric("통합 강좌", f"{report.course_count:,}")
    metrics[2].metric(
        "전체 교과목 연결",
        f"{report.master_code_match_count + report.master_name_match_count:,}",
    )
    metrics[3].metric(
        "5개 학과 강좌",
        f"{report.target_department_course_count:,}",
    )
    st.dataframe(
        report.summary_frame(),
        hide_index=True,
        use_container_width=True,
    )
    for issue in report.issues:
        renderer = {
            "error": st.error,
            "warning": st.warning,
            "info": st.info,
        }.get(issue.severity, st.write)
        renderer(issue.message)
    if not report.preview.empty:
        st.caption("개인정보 없는 전 학과 연결 미리보기 · 최대 8행")
        st.dataframe(report.preview, hide_index=True, use_container_width=True)


def render_course_recommendation_catalog(
    catalog: CourseRecommendationCatalogResult,
) -> None:
    """실제 개설강좌에서 만든 고유 교과목 추천 후보를 설명한다."""

    st.markdown("**추천 후보 변환 결과**")
    metrics = st.columns(4)
    metrics[0].metric("고유 교과목", f"{catalog.course_count:,}")
    metrics[1].metric("포함 학과", f"{catalog.department_count:,}")
    metrics[2].metric("원본 강의개요 연결", f"{catalog.description_linked_count:,}")
    metrics[3].metric("교과 master 미연결", f"{catalog.master_unmatched_count:,}")
    st.caption(
        "같은 교과목의 여러 학기·개설학과를 공식 교과목코드 하나로 합쳤습니다. "
        "과목명·학과·교과구분·NCS·학년·학점·수업방식만 사용하고 "
        "역량·직무·관심 태그는 만들지 않았습니다."
    )
    st.dataframe(catalog.preview, hide_index=True, use_container_width=True)


def render_course_description_enrichment_report(
    report: CourseDescriptionEnrichmentReport,
) -> None:
    """과거 강좌설명의 안전 연결 결과와 현재 범위 보존 여부를 표시한다."""

    st.success(
        "강좌설명 연결 점검을 완료했습니다. 현재 개설 교과목 범위는 유지되며 "
        "아직 추천 CSV에는 적용하지 않았습니다."
    )
    metrics = st.columns(4)
    metrics[0].metric("설명 원본 행", f"{report.source.source_row_count:,}")
    metrics[1].metric("사용 가능 설명", f"{report.usable_description_row_count:,}")
    metrics[2].metric("설명 연결 교과목", f"{report.enriched_course_count:,}")
    metrics[3].metric(
        "충돌·미연결",
        f"{report.ambiguous_course_count + report.unmatched_course_count:,}",
    )
    st.dataframe(report.term_summary, hide_index=True, use_container_width=True)
    for issue in report.issues:
        renderer = {
            "error": st.error,
            "warning": st.warning,
            "info": st.info,
        }.get(issue.severity, st.write)
        renderer(issue.message)
    if not report.preview.empty:
        st.caption("현재 개설 교과목 설명 연결 미리보기 · 최대 8행")
        st.dataframe(report.preview, hide_index=True, use_container_width=True)


def complete_master_activation(result: MasterActivationResult) -> None:
    """활성화 결과를 알리고 캐시를 비운 뒤 최신 master 상태로 다시 실행한다."""

    backup_text = (
        f" 기존 파일 백업: {result.backup_path}"
        if result.backup_path is not None
        else ""
    )
    st.session_state["master_activation_notice"] = (
        f"{result.row_count:,}행을 {result.destination_path.name}에 "
        f"활성화했습니다.{backup_text}"
    )
    st.cache_data.clear()
    st.rerun()


def render_master_activation_controls(
    report: MasterImportDryRunReport,
    service: MasterActivationService,
) -> None:
    """dry-run 안전 후보 중 교직원이 확인한 master만 활성화한다."""

    with st.expander("검토 완료 master 활성화", expanded=False):
        st.warning(
            "활성화하면 현재 추천 데이터가 바뀝니다. 기존 활성 파일은 먼저 "
            "`data/master_backups/`에 복사되며, 새 파일 오류 시 Repository는 "
            "synthetic fallback을 사용합니다."
        )
        acknowledged = st.checkbox(
            "안전 후보와 제외 사유를 확인했고 검토 완료 데이터만 활성화합니다.",
            key="master_activation_acknowledged",
        )

        department_candidates = report.departments.candidate_data
        try:
            validate_department_activation_data(department_candidates)
            department_ready = True
            department_error = ""
        except ValueError as error:
            department_ready = False
            department_error = str(error)
        department_column, program_column = st.columns(2)
        with department_column:
            st.markdown("**1. 학과 master**")
            st.caption(
                f"중복·미확정 코드를 제외한 {len(department_candidates):,}개 "
                "안전 후보만 적용합니다. 학생 모집단은 설정된 5개 학과를 유지합니다."
            )
            if department_error:
                st.error(department_error)
            if st.button(
                "학과 안전 후보 활성화",
                disabled=not acknowledged or not department_ready,
                use_container_width=True,
                key="activate_department_master",
            ):
                try:
                    complete_master_activation(
                        service.activate_departments(department_candidates)
                    )
                except (OSError, ValueError) as error:
                    st.error(f"학과 master를 활성화하지 못했습니다: {error}")

        with program_column:
            st.markdown("**2. WINGS 비교과 master**")
            st.caption(
                f"운영 상태와 무관한 재학생용 비교과 "
                f"{report.programs.accepted_row_count:,}개를 적용합니다. "
                "상태는 추적용으로만 보존하며 개인정보 가능 열 값은 저장하지 않습니다."
            )
            if st.button(
                "비교과 안전 후보 활성화",
                disabled=not acknowledged or not report.programs.ready,
                use_container_width=True,
                key="activate_program_master",
            ):
                try:
                    complete_master_activation(
                        service.activate_programs(report.programs.candidate_data)
                    )
                except (OSError, ValueError) as error:
                    st.error(f"비교과 master를 활성화하지 못했습니다: {error}")

        st.markdown("**3. 검토 완료 교과목**")
        course_candidates = prepare_activation_ready_courses(
            report.courses.candidate_data
        )
        if course_candidates.empty:
            st.info(
                "현재 파일에는 개설 여부·학년·학기·학점·설명·역량·직무·관심 분야를 "
                "모두 갖춘 교과목이 없습니다. 누락 정보를 임의로 만들지 않으므로 교과목은 "
                "synthetic fallback을 유지합니다."
            )
            reviewed_ids: list[str] = []
        else:
            option_by_label = {
                f"{row['course_id']} · {row['course_name']} · {row['department']}": str(
                    row["course_id"]
                )
                for row in course_candidates.to_dict(orient="records")
            }
            selected_labels = st.multiselect(
                "활성화할 검토 완료 교과목",
                options=list(option_by_label),
                key="reviewed_course_master_ids",
            )
            reviewed_ids = [option_by_label[label] for label in selected_labels]
            st.caption(
                "최소 3개가 필요하며, 1학년·무선수과목 3개와 선수과목 연결을 다시 검증합니다."
            )
        if st.button(
            "선택한 교과목 활성화",
            disabled=not acknowledged or len(reviewed_ids) < 3,
            use_container_width=True,
            key="activate_course_master",
        ):
            try:
                complete_master_activation(
                    service.activate_reviewed_courses(
                        course_candidates,
                        reviewed_ids,
                    )
                )
            except (OSError, ValueError) as error:
                st.error(f"교과목 master를 활성화하지 못했습니다: {error}")


st.set_page_config(page_title="시연 준비", page_icon="🧰", layout="wide")
render_page_header(
    "시연 준비",
    "발표 전 상태를 확인하고 대시보드 시연을 시작합니다.",
    audience="발표자 준비용",
)

demo_service = get_demo_service()
if not demo_service.enabled:
    st.error("현재 환경에서는 시연 준비 기능이 비활성화되어 있습니다.")
    render_footer()
    st.stop()

notice = st.session_state.pop("demo_reset_notice", None)
if notice:
    st.success(
        f"시연 기록 {notice['deleted_total']}행을 초기화했습니다. "
        f"백업: {notice['backup_path']}"
    )
master_activation_notice = st.session_state.pop(
    "master_activation_notice", None
)
if master_activation_notice:
    st.success(master_activation_notice)

student_id = demo_service.representative_student_id
student, baseline, current, baseline_history = load_demo_student_status(
    student_id,
    get_student_checkin_revision(),
)
record_counts = demo_service.get_record_counts()
record_total = sum(record_counts.values())
risk_path_ready = demo_service.representative_risk_path_is_ready(
    pd.DataFrame(baseline_history)
)
database_ready = record_total == 0
gemini_ready = resolve_streamlit_gemini_api_key() is not None
student_account_ready = str(student["student_id"]) == student_id

st.subheader("발표 전 준비 상태")
status_columns = st.columns(4)
status_columns[0].metric(
    "synthetic 기준 데이터",
    "대표 학생 위험 흐름 준비" if risk_path_ready else "확인 필요",
    "1→4주차",
)
status_columns[1].metric(
    "SQLite 시연 기록",
    "비어 있음" if database_ready else f"{record_total}행 존재",
)
status_columns[2].metric(
    "Gemini 연결 설정",
    "키 확인" if gemini_ready else "Mock 대기",
)
status_columns[3].metric(
    "학생 시연 계정",
    "사용 가능" if student_account_ready else "확인 필요",
    student_id,
)

with st.container(border=True):
    start_copy, start_action = st.columns([4, 1], vertical_alignment="center")
    with start_copy:
        st.markdown("### 대시보드에서 시연 시작")
        st.write(
            f"대표 학생 **{student_id} · {student['department']} · "
            f"{int(student['grade'])}학년**의 1~4주차 신호를 확인합니다."
        )
        st.caption("조기경보 → 학생 알림 → Kare 체크인 → AI 맞춤 추천 검토")
    if start_action.button(
        "전체 대시보드 열기",
        type="primary",
        use_container_width=True,
        key="start_demo_dashboard",
    ):
        st.switch_page("pages/01_dashboard.py")

    if int(current["week"]) > demo_service.baseline_week or not database_ready:
        st.warning(
            f"현재 {int(current['week'])}주차 기록 {record_total}건이 남아 있습니다. "
            "처음부터 시연하려면 바로 아래에서 기록을 초기화하세요."
        )

render_demo_reset(demo_service, record_counts)

if risk_path_ready and database_ready and student_account_ready:
    st.success(
        "시연 준비가 완료되었습니다. 전체 대시보드에서 바로 시작할 수 있습니다."
    )

with st.expander("시연 흐름 간단히 보기", expanded=False):
    st.markdown(
        "1. `전체 대시보드`에서 1→4주차를 이동하며 주차별 위험등급을 확인합니다.  \n"
        "2. 선택한 주차의 고위험 경보 대상에게 학생 안내를 보냅니다.  \n"
        f"3. `{student_id}` 학생 계정의 `나의 체크인`에서 Kare와 대화합니다.  \n"
        "4. 교직원 계정으로 돌아와 AI 맞춤 추천을 검토하고 지원계획을 확정합니다.  \n"
        "5. `학생지원 진행 관리`에서 학생지원 상태를 `상담 완료`로 기록합니다.  \n"
        "6. `지원 후 변화 확인`에서 4주차와 synthetic 15주차를 비교합니다."
    )

show_advanced_tools = st.toggle(
    "고급 준비 도구",
    value=False,
    key="demo_show_advanced_tools",
    help="실제 master 점검과 synthetic 사례 확인이 필요할 때만 엽니다.",
)
if not show_advanced_tools:
    render_footer()
    st.stop()

st.subheader("고급 준비 도구")

with st.expander("유형별 synthetic 대표 사례 8명 점검", expanded=False):
    st.caption(
        "발표 전 데이터 품질 확인용입니다. 여기서 선택한 사례로 발표 화면을 이동하지 않습니다."
    )
    showcases = load_synthetic_showcases(student_id)
    showcase_by_label = {item.option_label: item for item in showcases}
    selected_label = st.selectbox(
        "점검할 synthetic 사례",
        list(showcase_by_label),
        key="demo_showcase_student",
    )
    selected = showcase_by_label[selected_label]
    showcase_columns = st.columns(4)
    showcase_columns[0].metric("학과·학년", f"{selected.department} · {selected.grade}학년")
    showcase_columns[1].metric("위험등급", selected.risk_level)
    showcase_columns[2].metric("종합 위험도", f"{selected.overall_risk:.1f}")
    showcase_columns[3].metric("주 관찰 유형", selected.primary_risk_type)
    st.write(f"**1~4주 위험도 흐름**  {selected.risk_history_text}")
    st.write(f"**데이터 점검 포인트**  {selected.review_focus}")
    st.caption(f"synthetic 자유서술: {selected.natural_language_concern}")

st.subheader("실제 master 전환 사전 점검")
st.caption(
    "학과·교과·비교과 Excel을 메모리에서만 dry-run 합니다. 허용 열만 읽고 "
    "담당자명·사번·학번·전화·이메일 가능 열은 값 자체를 읽지 않습니다."
)
with st.container(border=True):
    activation_service = MasterActivationService()
    active_statuses = activation_service.inspect_statuses()
    st.markdown("**현재 master 사용 상태**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "구분": status.display_name,
                    "현재 파일": status.selected_file,
                    "행": status.row_count,
                    "사용 상태": (
                        "실제 master"
                        if status.using_actual
                        else "synthetic fallback"
                    ),
                    "설명": status.message,
                }
                for status in active_statuses
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    upload_columns = st.columns(3)
    department_file = upload_columns[0].file_uploader(
        "학과목록 Excel",
        type=("xlsx", "xls"),
        key="master_department_upload",
    )
    course_file = upload_columns[1].file_uploader(
        "전체 교과목 Excel",
        type=("xlsx", "xls"),
        key="master_course_upload",
    )
    program_file = upload_columns[2].file_uploader(
        "WINGS 비교과 Excel",
        type=("xlsx", "xls"),
        key="master_program_upload",
    )
    all_files_selected = all(
        file is not None
        for file in (department_file, course_file, program_file)
    )
    st.caption(
        "실행 결과는 확인용 보고서만 만들며 `data/` CSV·SQLite·설정 파일을 "
        "생성하거나 수정하지 않습니다."
    )
    if st.button(
        "세 파일 안전 점검 실행",
        type="primary",
        disabled=not all_files_selected,
        key="run_master_import_dry_run",
    ):
        assert department_file is not None
        assert course_file is not None
        assert program_file is not None
        with st.spinner("허용 열만 읽어 중복·결측·학과 연결을 점검하고 있습니다..."):
            st.session_state["master_import_dry_run_report"] = (
                run_master_import_dry_run(
                    department_file.getvalue(),
                    course_file.getvalue(),
                    program_file.getvalue(),
                    department_name=department_file.name,
                    course_name=course_file.name,
                    program_name=program_file.name,
                )
            )
    master_report = st.session_state.get("master_import_dry_run_report")
    if isinstance(master_report, MasterImportDryRunReport):
        render_master_dry_run_report(master_report)
        render_master_activation_controls(master_report, activation_service)

st.subheader("2026 전 학과 개설강좌 연결 점검")
st.caption(
    "학생 모집단은 5개 학과로 유지하면서 2026년 1·2학기 전 학과 개설강좌를 "
    "추천 후보로 사용할 수 있는지 점검합니다. 분반을 통합하고 전체 교과목에는 "
    "코드 우선·과목명 보조로 연결합니다."
)
with st.container(border=True):
    try:
        offering_files = find_default_course_offering_files(PROJECT_ROOT.parent)
        offering_files_ready = True
        offering_file_error = ""
        st.caption(
            "확인 파일: "
            f"{offering_files.course_master.name} · "
            f"{offering_files.semester_one.name} · "
            f"{offering_files.semester_two.name}"
        )
    except FileNotFoundError as error:
        offering_files = None
        offering_files_ready = False
        offering_file_error = str(error)
        st.warning(offering_file_error)
    try:
        description_file = find_default_course_description_file(
            PROJECT_ROOT.parent
        )
        description_file_error = ""
        st.caption(f"설명 보강 파일: {description_file.name}")
    except FileNotFoundError as error:
        description_file = None
        description_file_error = str(error)
    st.caption(
        "허용된 교과·강좌 열만 읽으며 결과는 메모리에만 보관합니다. "
        "이 단계에서는 현재 추천용 CSV를 변경하지 않습니다."
    )
    if st.button(
        "전 학과 개설강좌 연결 dry-run 실행",
        type="primary",
        disabled=not offering_files_ready,
        key="run_course_offering_dry_run",
    ):
        assert offering_files is not None
        st.session_state.pop("course_description_enrichment_report", None)
        with st.spinner("두 학기 분반을 통합하고 전체 교과목과 연결하고 있습니다..."):
            try:
                st.session_state["course_offering_dry_run_report"] = (
                    run_course_offering_dry_run(
                        offering_files.course_master,
                        offering_files.semester_one,
                        offering_files.semester_two,
                    )
                )
            except (OSError, TypeError, ValueError) as error:
                st.error(f"개설강좌를 점검하지 못했습니다: {error}")
    offering_report = st.session_state.get("course_offering_dry_run_report")
    if isinstance(offering_report, CourseOfferingDryRunResult):
        render_course_offering_dry_run_report(offering_report)
        try:
            course_catalog = build_course_recommendation_catalog(offering_report)
            render_course_recommendation_catalog(course_catalog)
        except ValueError as error:
            course_catalog = None
            st.error(f"추천 후보로 변환하지 못했습니다: {error}")
        if course_catalog is not None:
            activation_catalog_data = course_catalog.candidate_data
            with st.expander("강좌기본정보 설명 보강", expanded=False):
                st.caption(
                    "과거 정규학기의 강좌설명은 과목 의미 분석에만 사용합니다. "
                    "추천 대상과 실제 개설 여부는 현재 2026년 개설강좌를 그대로 "
                    "유지합니다."
                )
                if description_file is None:
                    st.warning(description_file_error)
                if st.button(
                    "강좌설명 보강 dry-run 실행",
                    type="primary",
                    disabled=description_file is None,
                    key="run_course_description_enrichment",
                ):
                    assert description_file is not None
                    with st.spinner(
                        "현재 개설 교과목과 과거 강좌설명을 안전하게 연결하고 있습니다..."
                    ):
                        try:
                            st.session_state[
                                "course_description_enrichment_report"
                            ] = run_course_description_enrichment_dry_run(
                                description_file,
                                course_catalog.candidate_data,
                            )
                        except (OSError, TypeError, ValueError) as error:
                            st.error(
                                f"강좌설명을 점검하지 못했습니다: {error}"
                            )
                description_report = st.session_state.get(
                    "course_description_enrichment_report"
                )
                if isinstance(
                    description_report,
                    CourseDescriptionEnrichmentReport,
                ):
                    if description_report.applies_to(
                        course_catalog.candidate_data
                    ):
                        render_course_description_enrichment_report(
                            description_report
                        )
                        activation_catalog_data = (
                            description_report.candidate_data
                        )
                    else:
                        st.warning(
                            "개설강좌 점검 결과가 바뀌었습니다. 강좌설명 dry-run을 "
                            "다시 실행해 주세요."
                        )
            with st.expander("실제 개설강좌 추천 master 활성화", expanded=False):
                st.warning(
                    "활성화하면 개인 교과 추천과 마이크로디그리 후보가 "
                    "2026년 전 학과 개설강좌를 사용합니다. 실제 이수내역·"
                    "선수과목·강의계획서 역량은 연결되지 않아 화면에 확인 필요로 "
                    "표시됩니다."
                )
                offering_activation_acknowledged = st.checkbox(
                    "의미 태그를 만들지 않고 실제 개설강좌와 검증된 강좌설명을 추천 후보로 사용합니다.",
                    key="course_offering_activation_acknowledged",
                )
                if st.button(
                    "전 학과 개설강좌 추천 master 활성화",
                    type="primary",
                    disabled=not offering_activation_acknowledged,
                    use_container_width=True,
                    key="activate_course_offering_catalog",
                ):
                    try:
                        complete_master_activation(
                            activation_service.activate_course_offering_catalog(
                                activation_catalog_data
                            )
                        )
                    except (OSError, ValueError) as error:
                        st.error(f"개설강좌 추천 master를 활성화하지 못했습니다: {error}")

st.divider()
st.caption(
    "대시보드의 시연 주차 버튼은 원본 데이터를 수정하지 않는 조회 제어입니다. "
    "실제 체크인과 지원 기록은 각 서비스 화면에서 사용자가 직접 생성하며, "
    "15주차 후속값은 지원 완료 상태에서만 적용되는 synthetic 시연 데이터입니다."
)
render_footer()
