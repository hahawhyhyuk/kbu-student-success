"""검증된 학생 프로필에서 DB 지원프로그램 Top 3를 선택한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.similarity import (
    ResilientSimilarityBackend,
    SimilarityBackend,
    create_similarity_backend,
)
from src.utils import load_app_config


DOMAIN_LABELS = {
    "attendance": "출결",
    "engagement": "학습참여",
    "achievement": "학업성취",
    "major_adaptation": "전공적응",
    "career": "진로설계",
    "complex": "복합위험",
}
SUPPORT_NEED_TERMS = {
    "전공 탐색": ("전공", "관심", "적응"),
    "학습 지원": ("학습", "과제", "튜터링", "성취"),
    "진로 탐색": ("진로", "직무", "목표"),
    "정기 모니터링": ("상담", "코칭", "점검"),
}


@dataclass(frozen=True)
class ProgramRecommendation:
    """DB master의 한 지원프로그램 추천 결과."""

    program_id: str
    program_name: str
    program_type: str
    description: str
    department_in_charge: str
    operation_period: str
    score: float
    reason: str


@dataclass(frozen=True)
class RecommendationResult:
    """Top 3 추천과 실제 similarity backend 상태."""

    recommendations: tuple[ProgramRecommendation, ...]
    similarity_backend: str
    warning: str | None = None


class InterventionRecommender:
    """AI가 아닌 deterministic 코드로 DB 프로그램 후보와 순위를 결정한다."""

    def __init__(
        self,
        similarity_backend: SimilarityBackend | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or load_app_config()
        self.similarity_backend = similarity_backend or create_similarity_backend(
            self.config
        )

    @staticmethod
    def _active_risk_domains(profile: Mapping[str, Any]) -> set[str]:
        scores = {
            domain: float(profile.get(f"{domain}_risk", 0) or 0)
            for domain in DOMAIN_LABELS
            if domain != "complex"
        }
        active = {domain for domain, score in scores.items() if score >= 60}
        if scores:
            active.add(max(scores, key=scores.get))
        if bool(profile.get("is_complex", False)):
            active.add("complex")
        return active

    @staticmethod
    def _profile_text(
        profile: Mapping[str, Any], analysis: Mapping[str, Any]
    ) -> str:
        values = [
            str(profile.get("natural_language_concern", "")),
            str(profile.get("interest_fields", "")).replace("|", " "),
            str(profile.get("desired_job", "")),
            str(analysis.get("summary", "")),
            " ".join(str(item) for item in analysis.get("support_needs", [])),
            " ".join(str(item) for item in analysis.get("interests", [])),
            " ".join(str(item) for item in analysis.get("desired_jobs", [])),
        ]
        return " ".join(value for value in values if value.strip())

    @staticmethod
    def _program_text(program: Mapping[str, Any]) -> str:
        return " ".join(
            str(program.get(column, "")).replace("|", " ")
            for column in (
                "program_name",
                "program_type",
                "description",
                "provided_competencies",
                "target_students",
            )
        )

    @staticmethod
    def _support_need_score(
        support_needs: Sequence[str], program_text: str
    ) -> float:
        if not support_needs:
            return 0.0
        normalized_program = program_text.replace(" ", "")
        matched = 0
        for need in support_needs:
            terms = SUPPORT_NEED_TERMS.get(str(need), (str(need),))
            if any(term.replace(" ", "") in normalized_program for term in terms):
                matched += 1
        return matched / len(support_needs)

    @staticmethod
    def _format_match_reason(
        active_domains: set[str],
        matched_domains: set[str],
        semantic_score: float,
        support_needs: Sequence[str],
        support_match: float,
    ) -> str:
        """세 추천 요소의 실제 일치 정도를 같은 형식으로 설명한다."""

        risk_match = (
            len(matched_domains) / len(active_domains) if active_domains else 0.0
        )
        matched_labels = [
            DOMAIN_LABELS[domain] for domain in sorted(matched_domains)
        ]
        risk_detail = (
            f"{'·'.join(matched_labels)} ({risk_match:.0%})"
            if matched_labels
            else "일치 영역 없음 (0%)"
        )
        support_detail = (
            f"{round(support_match * len(support_needs))}/{len(support_needs)}개 "
            f"({support_match:.0%})"
            if support_needs
            else "학생이 요청한 지원수요 없음"
        )
        return " · ".join(
            (
                f"위험영역 일치: {risk_detail}",
                f"학생 서술과 프로그램 설명 유사도: {semantic_score:.0%}",
                f"체크인 지원수요 일치: {support_detail}",
            )
        )

    def explain_program_matches(
        self,
        profile: Mapping[str, Any],
        analysis: Mapping[str, Any],
        programs: pd.DataFrame,
    ) -> dict[str, str]:
        """저장 버전과 무관하게 현재 master 기준 프로그램별 근거를 반환한다."""

        if programs.empty:
            return {}
        active_domains = self._active_risk_domains(profile)
        query = self._profile_text(profile, analysis)
        program_records = programs.to_dict(orient="records")
        documents = [self._program_text(program) for program in program_records]
        semantic_scores = self.similarity_backend.similarities(query, documents)
        support_needs = [str(item) for item in analysis.get("support_needs", [])]
        reasons: dict[str, str] = {}
        for program, semantic_score, program_text in zip(
            program_records, semantic_scores, documents
        ):
            targets = {
                item.strip()
                for item in str(program["target_risk_types"]).split("|")
                if item.strip()
            }
            matched_domains = active_domains & targets
            support_match = self._support_need_score(support_needs, program_text)
            reasons[str(program["program_id"])] = self._format_match_reason(
                active_domains,
                matched_domains,
                float(semantic_score),
                support_needs,
                support_match,
            )
        return reasons

    def recommend(
        self,
        profile: Mapping[str, Any],
        analysis: Mapping[str, Any],
        programs: pd.DataFrame,
        top_k: int | None = None,
    ) -> RecommendationResult:
        """Repository에 존재하는 프로그램만 점수화해 중복 없는 Top K를 반환한다.

        Parameters:
            profile: 최신 위험 스냅샷과 체크인 지표.
            analysis: JSON Schema 검증을 마친 자유서술 구조화 결과.
            programs: Repository가 반환한 지원프로그램 master.
            top_k: 반환 개수. 미지정 시 app_config의 값.

        Returns:
            추천 프로그램과 similarity backend 상태.

        Assumptions:
            생성형 AI는 후보 ID 선정에 관여하지 않으며 program_id 중복은 허용하지 않는다.
        """

        if programs.empty:
            raise ValueError("추천 가능한 지원프로그램이 없습니다.")
        if programs["program_id"].duplicated().any():
            raise ValueError("지원프로그램 ID는 중복될 수 없습니다.")

        recommendation_config = self.config["recommendation"]
        weights = recommendation_config["weights"]
        limit = int(top_k or recommendation_config["top_k"])
        limit = max(1, min(limit, len(programs)))
        active_domains = self._active_risk_domains(profile)
        query = self._profile_text(profile, analysis)
        program_records = programs.to_dict(orient="records")
        documents = [self._program_text(program) for program in program_records]
        semantic_scores = self.similarity_backend.similarities(query, documents)
        support_needs = [str(item) for item in analysis.get("support_needs", [])]

        ranked: list[tuple[float, str, ProgramRecommendation]] = []
        for program, semantic_score, program_text in zip(
            program_records, semantic_scores, documents
        ):
            targets = {
                item.strip()
                for item in str(program["target_risk_types"]).split("|")
                if item.strip()
            }
            matched_domains = active_domains & targets
            risk_match = (
                len(matched_domains) / len(active_domains) if active_domains else 0.0
            )
            support_match = self._support_need_score(
                support_needs, program_text
            )
            total_score = 100 * (
                float(weights["risk_type_match"]) * risk_match
                + float(weights["semantic_similarity"]) * float(semantic_score)
                + float(weights["support_need_match"]) * support_match
            )
            recommendation = ProgramRecommendation(
                program_id=str(program["program_id"]),
                program_name=str(program["program_name"]),
                program_type=str(program["program_type"]),
                description=str(program["description"]),
                department_in_charge=str(program["department_in_charge"]),
                operation_period=str(program["operation_period"]),
                score=round(min(100.0, max(0.0, total_score)), 2),
                reason=self._format_match_reason(
                    active_domains,
                    matched_domains,
                    float(semantic_score),
                    support_needs,
                    support_match,
                ),
            )
            ranked.append((recommendation.score, recommendation.program_id, recommendation))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        backend_name = self.similarity_backend.backend_name
        warning = None
        if isinstance(self.similarity_backend, ResilientSimilarityBackend):
            backend_name = self.similarity_backend.last_backend_name
            warning = self.similarity_backend.last_warning
        return RecommendationResult(
            recommendations=tuple(item[2] for item in ranked[:limit]),
            similarity_backend=backend_name,
            warning=warning,
        )
