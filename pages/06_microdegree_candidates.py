"""반복 학습경로에서 신규 마이크로디그리 개발 후보를 찾는 교직원 화면."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from src.ai import create_ai_provider
from src.course_recommender import CourseRecommender
from src.database import LearningPathDataStore
from src.microdegree_service import (
    MicrodegreeAnalysisResult,
    MicrodegreeCandidateService,
)
from src.repositories import get_default_repository
from src.risk_service import (
    create_integrated_risk_service,
    get_student_checkin_revision,
)
from src.ui import (
    queue_ai_reveal,
    render_ai_reveal,
    render_footer,
    render_page_header,
)
from src.utils import load_app_config


@st.cache_data
def load_candidate_data(
    checkin_revision: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repository의 최신 위험스냅샷·교과목·이수내역을 로드한다."""

    repository = get_default_repository()
    source = repository.get_all()
    snapshots = create_integrated_risk_service(repository).build_snapshots(source)
    return snapshots, source.courses, source.completed_courses


@st.cache_resource
def get_course_recommender() -> CourseRecommender:
    """대량 분석에서 교과목 임베딩과 모델을 재사용한다."""

    return CourseRecommender()


def get_learning_path_store() -> LearningPathDataStore:
    """현재 설정된 SQLite에 연결된 학습경로 저장소를 만든다."""

    return LearningPathDataStore()


def resolve_gemini_api_key() -> str | None:
    """환경변수를 우선하고 Streamlit secrets를 보조로 확인한다."""

    environment_key = os.getenv("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        return secret_key or None
    except Exception:
        return None


def _as_list(value: Any) -> list[str]:
    """SQLite JSON 문자열 또는 sequence를 화면용 문자열 목록으로 바꾼다."""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(item) for item in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def render_candidate(
    candidate: Mapping[str, Any],
    *,
    demand_rank: int,
    reveal_key: str,
    provider_name: str = "AI",
) -> None:
    """실시간 또는 SQLite 후보를 같은 교직원 카드 형식으로 표시한다."""

    candidate_code = str(candidate["candidate_code"])
    course_names = _as_list(candidate["course_names"])
    competencies = _as_list(candidate["competencies"])
    related_jobs = _as_list(candidate["related_jobs"])
    departments = _as_list(candidate["departments"])
    student_ids = _as_list(candidate["student_ids"])
    demand_share = float(candidate.get("demand_share") or 0)
    with st.container(border=True):
        title_column, status_column = st.columns([4, 1])
        title_column.markdown(f"### {candidate['candidate_name']}")
        status_column.info(f"수요 순위 {demand_rank}")
        st.caption(f"개발 상태: {candidate['status']}")
        render_ai_reveal(
            str(candidate["description"]),
            key=reveal_key,
            label=f"검증 완료 · {provider_name} 교육과정 후보 설명",
        )

        metrics = st.columns(4)
        metrics[0].metric("수요 학생", f"{int(candidate['student_count'])}명")
        metrics[1].metric(
            "전체 경로 중 비율",
            f"{demand_share * 100:.1f}%",
        )
        metrics[2].metric(
            "관련 학과", f"{int(candidate['department_count'])}개"
        )
        metrics[3].metric(
            "경로 구성 참고 유사도",
            f"{float(candidate['average_similarity']) * 100:.1f}%",
        )
        candidate_scope = "다학과 공통 후보" if len(departments) >= 2 else "학과 특화 후보"
        st.markdown(f"**후보 유형:** {candidate_scope}")

        st.markdown("**후보 교과목**")
        for sequence, course_name in enumerate(course_names, start=1):
            st.markdown(f"{sequence}. {course_name}")
        if competencies:
            st.markdown("**공통역량:** " + " · ".join(competencies))
        else:
            st.info(
                "원본 교과목에 공통역량 태그가 없어 반복된 실제 과목 조합을 "
                "후보 근거로 사용했습니다."
            )
        st.markdown(
            "**주요 희망직무:** "
            + (" · ".join(related_jobs) or "공통값 없음")
        )
        st.caption(
            f"관련 학과: {', '.join(departments)} · 후보 ID: {candidate_code}"
        )
        with st.expander("수요 근거 확인"):
            st.write(
                "동일·유사한 개인 맞춤 학습경로를 추천받은 synthetic 학생: "
                + ", ".join(student_ids)
            )


def render_candidate_groups(
    candidates: Sequence[Mapping[str, Any]],
    *,
    batch_id: str,
    provider_name: str = "AI",
) -> None:
    """등급을 붙이지 않고 수요 인원·학과 수 순으로 모든 후보를 표시한다."""

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -int(candidate["student_count"]),
            -int(candidate["department_count"]),
            -float(candidate["average_similarity"]),
            str(candidate["candidate_code"]),
        ),
    )
    st.subheader("수요 기반 개발 후보")
    st.caption(
        "등급을 부여하지 않고 수요 학생 수, 관련 학과 수, 반복 과목 근거가 "
        "큰 순서로 표시합니다."
    )
    for demand_rank, candidate in enumerate(ranked, start=1):
        render_candidate(
            candidate,
            demand_rank=demand_rank,
            reveal_key=f"microdegree_{batch_id}_{candidate['candidate_code']}",
            provider_name=provider_name,
        )


st.set_page_config(
    page_title="신규 마이크로디그리 개발 후보",
    page_icon="🧩",
    layout="wide",
)
render_page_header(
    "신규 마이크로디그리 개발 후보",
    "반복되는 교과 수요를 교육과정 개발 후보로 확인합니다.",
    audience="교직원·교육과정 검토용",
)
with st.expander("후보 해석·생성 기준", expanded=False):
    st.warning(
        "표시되는 조합은 공식 마이크로디그리가 아닙니다. "
        "교육과정위원회 검토가 필요한 개발 후보입니다."
    )
    st.write(
        "실제 2026년 개설 교과목과 개인정보 없는 synthetic 학생 200명을 결합한 "
        "수요 시뮬레이션이며 실제 재학생 수요 통계는 아닙니다."
    )
    candidate_config = load_app_config()["microdegree_candidate"]
    st.write(
        "① 전공적응·진로 지원 대상의 개인 경로 생성  →  "
        f"② {int(candidate_config['repeated_course_count'])}개 이상 반복된 교과목 조합 집계  →  "
        "③ 최소 수요 조건을 충족한 모든 후보를 수요순으로 표시"
    )
    st.caption(
        f"다학과 후보: {int(candidate_config['minimum_students'])}명 이상·"
        f"{int(candidate_config['minimum_departments'])}개 학과 이상 · "
        f"학과 특화 후보: 한 학과 {int(candidate_config['single_department_minimum_students'])}명 이상"
    )

snapshots, courses, completed_courses = load_candidate_data(
    get_student_checkin_revision()
)
store = get_learning_path_store()
result_key = "microdegree_candidate_analysis"

if st.button("전체 학습경로 수요 분석", type="primary"):
    with st.spinner(
        "대상 학생별 교과 경로를 만들고 유사한 수요를 군집화하고 있습니다..."
    ):
        try:
            provider = create_ai_provider(api_key=resolve_gemini_api_key())
            service = MicrodegreeCandidateService(
                course_recommender=get_course_recommender(),
                provider=provider,
            )
            result = service.analyze(
                snapshots=snapshots,
                courses=courses,
                completed_courses=completed_courses,
            )
            batch_id = service.persist_analysis(result, store)
            st.session_state[result_key] = (result, batch_id)
            queue_ai_reveal(
                *(
                    f"microdegree_{batch_id}_{candidate.candidate_id}"
                    for candidate in result.candidates
                )
            )
        except Exception as error:
            st.error(f"교육과정 개발 후보를 분석하지 못했습니다: {error}")

stored_result = st.session_state.get(result_key)
if stored_result:
    result: MicrodegreeAnalysisResult
    batch_id: str
    result, batch_id = stored_result
    if result.warning:
        st.warning(result.warning)
    candidate_student_count = len(
        {
            student_id
            for candidate in result.candidates
            for student_id in candidate.student_ids
        }
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("분석 대상", f"{result.target_student_count}명")
    summary_columns[1].metric("생성 경로", f"{len(result.demands)}개")
    summary_columns[2].metric("개발 후보", f"{len(result.candidates)}개")
    summary_columns[3].metric("후보 포함 학생", f"{candidate_student_count}명")
    st.caption(
        f"후보 설명 provider: {result.provider_name} · 분석 배치: {batch_id} · "
        f"경로 생성 제외: {result.skipped_student_count}명 · "
        f"반복 교과목 기준: {result.minimum_repeated_courses}개 이상"
    )

    if not result.candidates:
        st.info("현재 설정 기준을 모두 충족하는 개발 후보가 없습니다.")
    live_candidates = [
        {
            "candidate_code": candidate.candidate_id,
            "candidate_name": candidate.candidate_name,
            "description": candidate.description,
            "student_count": candidate.student_count,
            "department_count": candidate.department_count,
            "departments": candidate.departments,
            "course_names": candidate.course_names,
            "competencies": candidate.competencies,
            "related_jobs": candidate.related_jobs,
            "student_ids": candidate.student_ids,
            "average_similarity": candidate.average_similarity,
            "priority": candidate.priority,
            "demand_share": candidate.demand_share,
            "status": candidate.status,
        }
        for candidate in result.candidates
    ]
    render_candidate_groups(
        live_candidates,
        batch_id=batch_id,
        provider_name=result.provider_name,
    )
else:
    latest_candidates = store.list_latest_candidates()
    if latest_candidates.empty:
        st.info(
            "분석 버튼을 누르면 전공적응·진로설계 지원 대상의 학습경로를 생성하고 반복 수요를 찾습니다."
        )
    else:
        latest_batch = str(latest_candidates.iloc[0]["analysis_batch_id"])
        st.info(f"SQLite에 저장된 최근 분석 결과입니다. 분석 배치: {latest_batch}")
        render_candidate_groups(
            latest_candidates.to_dict(orient="records"),
            batch_id=latest_batch,
        )

st.divider()
st.caption(
    "모든 결과는 교육과정 개발 검토 후보이며 공식 과정 승인을 의미하지 않습니다. "
    "최소 수요 기준은 config/app_config.yaml에서 조정할 수 있습니다."
)
render_footer()
