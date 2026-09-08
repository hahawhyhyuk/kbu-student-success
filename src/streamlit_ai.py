"""Streamlit 환경변수·secrets를 공통 AI provider 생성으로 연결한다."""

from __future__ import annotations

import os

import streamlit as st

from src.ai import AIProvider, create_ai_provider


def resolve_streamlit_gemini_api_key() -> str | None:
    """환경변수를 우선하고 Streamlit secrets를 보조로 API key를 찾는다."""

    environment_key = os.getenv("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        return secret_key or None
    except Exception:
        return None


def create_streamlit_ai_provider() -> AIProvider:
    """현재 Streamlit 실행환경에서 Gemini+Mock fallback provider를 만든다."""

    return create_ai_provider(api_key=resolve_streamlit_gemini_api_key())
