"""검증 완료된 AI 문장의 1회 스트리밍 표시 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import build_ai_reveal_chunks


def test_ai_reveal_chunks_preserve_validated_text() -> None:
    """문장·마크다운 공백은 애니메이션 조각을 합쳐도 바뀌지 않는다."""

    text = "학생의 고민을 이해했어요.\n\n**희망직무 연결:** 서비스 기획"

    chunks = build_ai_reveal_chunks(text, words_per_chunk=2)

    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_ai_reveal_chunk_size_must_be_positive() -> None:
    """잘못된 UI 설정은 빈 반복 대신 명확한 오류를 낸다."""

    with pytest.raises(ValueError, match="1어절 이상"):
        build_ai_reveal_chunks("검증된 설명", words_per_chunk=0)


def test_ai_reveal_animates_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """새 생성 직후만 스트리밍하고 다음 rerun에서는 즉시 표시한다."""

    monkeypatch.setattr(
        "src.ui.load_app_config",
        lambda: {
            "ui": {
                "ai_reveal": {
                    "enabled": True,
                    "words_per_chunk": 2,
                    "delay_seconds": 0,
                }
            }
        },
    )
    app_file = tmp_path / "ai_reveal_app.py"
    app_file.write_text(
        """
import streamlit as st
from src.ui import queue_ai_reveal, render_ai_reveal

if st.button("새 AI 결과"):
    queue_ai_reveal("demo-result")
animated = render_ai_reveal(
    "검증된 AI 설명이 자연스럽게 표시됩니다.",
    key="demo-result",
)
st.caption(f"animated={animated}")
""",
        encoding="utf-8",
    )

    page = AppTest.from_file(str(app_file)).run(timeout=30)
    assert not page.exception
    assert any(item.value == "animated=False" for item in page.caption)

    page.button[0].click().run(timeout=30)
    assert not page.exception
    assert any(item.value == "animated=True" for item in page.caption)

    page.run(timeout=30)
    assert not page.exception
    assert any(item.value == "animated=False" for item in page.caption)

