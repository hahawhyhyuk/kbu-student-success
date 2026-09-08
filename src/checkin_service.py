"""AI provider 결과를 UI가 안전하게 소비할 수 있도록 정리하는 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ai import AIProvider, ResilientAIProvider, validate_checkin_analysis


@dataclass(frozen=True)
class CheckinAnalysisResult:
    """검증된 체크인 분석과 실제 사용 provider 메타데이터."""

    analysis: dict[str, Any]
    provider_name: str
    fallback_used: bool
    warning: str | None = None


class CheckinAnalysisService:
    """원문 AI 응답을 UI에 전달하지 않고 검증된 결과만 반환한다."""

    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def analyze(self, text: str) -> CheckinAnalysisResult:
        """자유서술을 분석하고 provider 상태와 함께 반환한다."""

        analysis = validate_checkin_analysis(
            self.provider.analyze_checkin(str(text or ""))
        )
        if isinstance(self.provider, ResilientAIProvider):
            provider_name = self.provider.last_provider_name
            warning = self.provider.last_error
        else:
            provider_name = self.provider.provider_name
            warning = (
                "Gemini API key가 없어 Mock 분석을 사용했습니다."
                if provider_name == "mock"
                else None
            )
        return CheckinAnalysisResult(
            analysis=analysis,
            provider_name=provider_name,
            fallback_used=provider_name == "mock",
            warning=warning,
        )
