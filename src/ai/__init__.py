"""학생 체크인 분석 AI provider 공개 API."""

from src.ai.base import AIProvider, AIProviderError
from src.ai.factory import create_ai_provider
from src.ai.gemini_provider import GeminiProvider
from src.ai.mock_provider import MockAIProvider
from src.ai.resilient_provider import ResilientAIProvider
from src.ai.schema import (
    CHECKIN_ANALYSIS_SCHEMA,
    CHECKIN_CHAT_TURN_SCHEMA,
    KARE_CHAT_REPLY_SCHEMA,
    LEARNING_PATH_EXPLANATION_SCHEMA,
    MICRODEGREE_CANDIDATE_DESCRIPTION_SCHEMA,
    NEXT_ACTION_SUGGESTION_SCHEMA,
    STAFF_BRIEFING_SCHEMA,
    validate_checkin_analysis,
    validate_checkin_chat_turn,
    validate_kare_chat_reply,
    validate_learning_path_explanation,
    validate_microdegree_candidate_description,
    validate_next_action_suggestion,
    validate_staff_briefing,
)

__all__ = [
    "AIProvider",
    "AIProviderError",
    "CHECKIN_ANALYSIS_SCHEMA",
    "CHECKIN_CHAT_TURN_SCHEMA",
    "KARE_CHAT_REPLY_SCHEMA",
    "GeminiProvider",
    "LEARNING_PATH_EXPLANATION_SCHEMA",
    "MICRODEGREE_CANDIDATE_DESCRIPTION_SCHEMA",
    "NEXT_ACTION_SUGGESTION_SCHEMA",
    "MockAIProvider",
    "ResilientAIProvider",
    "STAFF_BRIEFING_SCHEMA",
    "create_ai_provider",
    "validate_checkin_analysis",
    "validate_checkin_chat_turn",
    "validate_kare_chat_reply",
    "validate_learning_path_explanation",
    "validate_microdegree_candidate_description",
    "validate_next_action_suggestion",
    "validate_staff_briefing",
]
