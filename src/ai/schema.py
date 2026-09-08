"""학생 자유서술 구조화 결과의 JSON Schema와 검증 함수."""

from __future__ import annotations

import re
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from src.ai.base import AIProviderError
from src.checkin_semantic_state import (
    SEMANTIC_ALLOWED_STATES,
    SEMANTIC_DIMENSIONS,
)


CHECKIN_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "major_concern": {"type": "boolean"},
        "learning_difficulty": {"type": "boolean"},
        "career_uncertainty": {"type": "boolean"},
        "interests": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "desired_jobs": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "support_needs": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 300},
    },
    "required": [
        "major_concern",
        "learning_difficulty",
        "career_uncertainty",
        "interests",
        "desired_jobs",
        "support_needs",
        "summary",
    ],
}

CHECKIN_CHAT_FIELDS: tuple[str, ...] = (
    "major_interest",
    "major_satisfaction",
    "major_continuation_intent",
    "learning_difficulty",
    "career_clarity",
    "consultation_intent",
    "interest_fields",
    "desired_job",
    "natural_language_concern",
)

CHECKIN_CHAT_TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "assistant_message": {
            "type": "string",
            "minLength": 1,
            "maxLength": 700,
        },
        "field_updates": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "major_interest": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 5,
                },
                "major_satisfaction": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 5,
                },
                "major_continuation_intent": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 5,
                },
                "learning_difficulty": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 5,
                },
                "career_clarity": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 5,
                },
                "consultation_intent": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 5,
                },
                "interest_fields": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "maxItems": 8,
                    "uniqueItems": True,
                },
                "desired_job": {
                    "type": ["string", "null"],
                    "maxLength": 120,
                },
                "consultation_requested": {"type": ["boolean", "null"]},
                "explore_other_fields": {"type": ["boolean", "null"]},
                "natural_language_concern": {
                    "type": ["string", "null"],
                    "maxLength": 1200,
                },
            },
            "required": [
                "major_interest",
                "major_satisfaction",
                "major_continuation_intent",
                "learning_difficulty",
                "career_clarity",
                "consultation_intent",
                "interest_fields",
                "desired_job",
                "consultation_requested",
                "explore_other_fields",
                "natural_language_concern",
            ],
        },
        "addressed_fields": {
            "type": "array",
            "items": {"type": "string", "enum": list(CHECKIN_CHAT_FIELDS)},
            "maxItems": len(CHECKIN_CHAT_FIELDS),
            "uniqueItems": True,
        },
        "next_focus": {
            "type": "string",
            "enum": [*CHECKIN_CHAT_FIELDS, "review"],
        },
        "ready_for_review": {"type": "boolean"},
    },
    "required": [
        "assistant_message",
        "field_updates",
        "addressed_fields",
        "next_focus",
        "ready_for_review",
    ],
}

KARE_CHAT_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "assistant_message": {
            "type": "string",
            "minLength": 1,
            "maxLength": 700,
        },
        "dimension_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": list(SEMANTIC_DIMENSIONS),
                    },
                    "state": {
                        "type": "string",
                        "enum": sorted(
                            {
                                state
                                for states in SEMANTIC_ALLOWED_STATES.values()
                                for state in states
                            }
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "detail": {
                        "type": ["string", "null"],
                        "maxLength": 300,
                    },
                },
                "required": [
                    "dimension",
                    "state",
                    "evidence",
                    "confidence",
                    "detail",
                ],
            },
            "maxItems": len(SEMANTIC_DIMENSIONS),
        },
        "next_focus": {
            "type": "string",
            "enum": [*SEMANTIC_DIMENSIONS, "review"],
        },
        "suggest_review": {"type": "boolean"},
    },
    "required": [
        "assistant_message",
        "dimension_updates",
        "next_focus",
        "suggest_review",
    ],
}

LEARNING_PATH_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "path_name": {"type": "string", "minLength": 1, "maxLength": 80},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "competencies": {"type": "array", "items": {"type": "string"}},
        "course_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "course_id": {"type": "string"},
                    "role": {"type": "string", "minLength": 1},
                    "sequence": {"type": "integer", "minimum": 1},
                },
                "required": ["course_id", "role", "sequence"],
            },
            "minItems": 3,
            "maxItems": 4,
        },
        "career_connection": {"type": "string", "minLength": 1, "maxLength": 300},
    },
    "required": ["path_name", "reason", "competencies", "course_roles", "career_connection"],
}

MICRODEGREE_CANDIDATE_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
        },
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "status": {"const": "교육과정 검토 후보"},
    },
    "required": ["candidate_name", "description", "status"],
}

STAFF_BRIEFING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string", "minLength": 1, "maxLength": 80},
        "summary": {"type": "string", "minLength": 1, "maxLength": 500},
        "focus_risk_types": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
        },
        "recommended_action": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
        },
    },
    "required": [
        "headline",
        "summary",
        "focus_risk_types",
        "recommended_action",
    ],
}

NEXT_ACTION_SUGGESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "minLength": 1, "maxLength": 80},
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
        "check_before_action": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
        },
    },
    "required": ["action", "reason", "check_before_action"],
}


def validate_checkin_analysis(value: Mapping[str, Any]) -> dict[str, Any]:
    """AI 구조화 결과를 JSON Schema로 검증하고 안전한 사본을 반환한다.

    Parameters:
        value: JSON parsing을 마친 mapping.

    Returns:
        UI와 추천 서비스에 전달 가능한 검증된 dict.

    Assumptions:
        검증 실패 시 원문 응답을 반환하지 않고 일반화된 AIProviderError를 발생시킨다.
    """

    if not isinstance(value, Mapping):
        raise AIProviderError("AI 구조화 결과가 JSON object가 아닙니다.")
    errors = sorted(
        Draft202012Validator(CHECKIN_ANALYSIS_SCHEMA).iter_errors(dict(value)),
        key=lambda error: list(error.path),
    )
    if errors:
        raise AIProviderError("AI 구조화 결과가 필수 JSON Schema를 만족하지 않습니다.")
    return dict(value)


def validate_checkin_chat_turn(
    value: Mapping[str, Any],
    allowed_interest_fields: set[str],
) -> dict[str, Any]:
    """Kare의 대화 응답과 슬롯 추출 결과를 검증한다.

    Parameters:
        value: Gemini 또는 Mock provider가 반환한 대화 턴 결과.
        allowed_interest_fields: 현재 서비스에서 선택 가능한 관심 분야 master.

    Returns:
        UI와 대화 상태 서비스에서 사용할 수 있는 검증된 사본.

    Assumptions:
        자유 텍스트 직무·고민 외의 분류값은 사전에 정의된 필드와 관심 분야만 허용한다.
    """

    if not isinstance(value, Mapping):
        raise AIProviderError("Kare 대화 결과가 JSON object가 아닙니다.")
    errors = sorted(
        Draft202012Validator(CHECKIN_CHAT_TURN_SCHEMA).iter_errors(dict(value)),
        key=lambda error: list(error.path),
    )
    if errors:
        raise AIProviderError("Kare 대화 결과가 필수 JSON Schema를 만족하지 않습니다.")
    interests = value["field_updates"].get("interest_fields")
    if interests is not None and not set(map(str, interests)).issubset(
        allowed_interest_fields
    ):
        raise AIProviderError("Kare 대화 결과에 허용되지 않은 관심 분야가 포함되었습니다.")
    return {
        **dict(value),
        "field_updates": dict(value["field_updates"]),
        "addressed_fields": list(value["addressed_fields"]),
    }


def validate_kare_chat_reply(value: Mapping[str, Any]) -> dict[str, Any]:
    """Kare의 근거 기반 의미 추출과 자유대화 한 턴을 검증한다."""

    if not isinstance(value, Mapping):
        raise AIProviderError("Kare 대화 결과가 JSON object가 아닙니다.")
    errors = sorted(
        Draft202012Validator(KARE_CHAT_REPLY_SCHEMA).iter_errors(dict(value)),
        key=lambda error: list(error.path),
    )
    if errors:
        raise AIProviderError("Kare 대화 결과가 필수 JSON Schema를 만족하지 않습니다.")
    updates = []
    for item in value["dimension_updates"]:
        dimension = str(item["dimension"])
        state = str(item["state"])
        if state not in SEMANTIC_ALLOWED_STATES[dimension]:
            raise AIProviderError(
                f"Kare 대화가 {dimension}에 지원하지 않는 상태를 반환했습니다."
            )
        updates.append(dict(item))
    return {
        "assistant_message": str(value["assistant_message"]).strip(),
        "dimension_updates": updates,
        "next_focus": str(value["next_focus"]),
        "suggest_review": bool(value["suggest_review"]),
    }


def validate_learning_path_explanation(
    value: Mapping[str, Any], allowed_course_ids: set[str]
) -> dict[str, Any]:
    """학습경로 설명 Schema와 DB 교과목 ID·순서를 검증한다."""

    if not isinstance(value, Mapping):
        raise AIProviderError("학습경로 설명이 JSON object가 아닙니다.")
    errors = list(Draft202012Validator(LEARNING_PATH_EXPLANATION_SCHEMA).iter_errors(dict(value)))
    if errors:
        raise AIProviderError("학습경로 설명이 필수 JSON Schema를 만족하지 않습니다.")
    roles = list(value["course_roles"])
    returned_ids = {str(item["course_id"]) for item in roles}
    if returned_ids != allowed_course_ids or len(returned_ids) != len(roles):
        raise AIProviderError("학습경로 설명에 DB에 없는 교과목 ID가 포함되었습니다.")
    if sorted(int(item["sequence"]) for item in roles) != list(range(1, len(roles) + 1)):
        raise AIProviderError("학습경로 설명의 권장 순서가 유효하지 않습니다.")
    return dict(value)


def validate_microdegree_candidate_description(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """후보가 공식 과정으로 오인되지 않도록 AI 설명 Schema를 검증한다."""

    if not isinstance(value, Mapping):
        raise AIProviderError("교육과정 후보 설명이 JSON object가 아닙니다.")
    errors = list(
        Draft202012Validator(
            MICRODEGREE_CANDIDATE_DESCRIPTION_SCHEMA
        ).iter_errors(dict(value))
    )
    if errors:
        raise AIProviderError(
            "교육과정 후보 설명이 필수 JSON Schema를 만족하지 않습니다."
        )
    return dict(value)


def validate_staff_briefing(
    value: Mapping[str, Any],
    allowed_focuses: set[str],
    allowed_actions: set[str],
) -> dict[str, Any]:
    """주간 브리핑이 코드 집계 범위 밖의 숫자·위험유형·행동을 만들지 않게 검증한다."""

    if not isinstance(value, Mapping):
        raise AIProviderError("AI 브리핑 결과가 JSON object가 아닙니다.")
    errors = list(
        Draft202012Validator(STAFF_BRIEFING_SCHEMA).iter_errors(dict(value))
    )
    if errors:
        raise AIProviderError("AI 브리핑 결과가 필수 JSON Schema를 만족하지 않습니다.")
    returned_focuses = {str(item) for item in value["focus_risk_types"]}
    if not returned_focuses.issubset(allowed_focuses):
        raise AIProviderError("AI 브리핑에 집계되지 않은 위험유형이 포함되었습니다.")
    if str(value["recommended_action"]) not in allowed_actions:
        raise AIProviderError("AI 브리핑에 허용되지 않은 화면 행동이 포함되었습니다.")
    prose = f"{value['headline']} {value['summary']}"
    if re.search(r"\d", prose):
        raise AIProviderError("AI 브리핑 설명은 검증되지 않은 숫자를 포함할 수 없습니다.")
    return dict(value)


def validate_next_action_suggestion(
    value: Mapping[str, Any],
    allowed_actions: set[str],
    allowed_checks: set[str],
) -> dict[str, Any]:
    """다음 조치가 설정에 있는 행동과 교직원 확인 항목만 사용하도록 검증한다."""

    if not isinstance(value, Mapping):
        raise AIProviderError("AI 다음 조치 결과가 JSON object가 아닙니다.")
    errors = list(
        Draft202012Validator(NEXT_ACTION_SUGGESTION_SCHEMA).iter_errors(
            dict(value)
        )
    )
    if errors:
        raise AIProviderError(
            "AI 다음 조치 결과가 필수 JSON Schema를 만족하지 않습니다."
        )
    if str(value["action"]) not in allowed_actions:
        raise AIProviderError("AI가 설정에 없는 다음 조치를 제안했습니다.")
    returned_checks = {str(item) for item in value["check_before_action"]}
    if not returned_checks.issubset(allowed_checks):
        raise AIProviderError("AI가 허용되지 않은 확인 항목을 제안했습니다.")
    if re.search(r"\b(?:S|STU)\d+\b", str(value["reason"]), re.IGNORECASE):
        raise AIProviderError("AI 다음 조치 설명에 학생 식별자를 포함할 수 없습니다.")
    return dict(value)
