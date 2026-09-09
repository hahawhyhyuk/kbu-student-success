"""환경에 따라 Gemini+fallback 또는 Mock provider를 생성한다."""

from __future__ import annotations

import os
from typing import Any, Mapping

from src.ai.base import AIProvider
from src.ai.gemini_provider import GeminiProvider
from src.ai.mock_provider import MockAIProvider
from src.ai.resilient_provider import ResilientAIProvider
from src.utils import load_app_config


def create_ai_provider(
    api_key: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> AIProvider:
    """API key가 있으면 Gemini를 우선하고 항상 Mock fallback을 제공한다."""

    active_config = config or load_app_config()
    ai_config = active_config["ai"]
    key = api_key or os.getenv(str(ai_config["api_key_env"]), "")
    fallback = MockAIProvider()
    if not str(key).strip() or str(ai_config.get("provider", "auto")) == "mock":
        return fallback
    primary = GeminiProvider(
        api_key=str(key),
        model=str(ai_config["gemini_model"]),
        timeout_seconds=int(ai_config["timeout_seconds"]),
        kare_timeout_seconds=int(
            ai_config.get(
                "kare_timeout_seconds",
                ai_config["timeout_seconds"],
            )
        ),
    )
    return ResilientAIProvider(primary=primary, fallback=fallback)
