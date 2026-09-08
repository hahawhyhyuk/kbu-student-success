"""관리자 소개의 AI 역할과 운영 대시보드의 간결성을 확인하는 UI 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.access_control import AUTH_IDENTITY_KEY, ROLE_STAFF, ROLE_STUDENT
from src.utils import PROJECT_ROOT


def test_home_exposes_ai_copilot_without_calling_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """첫 화면에서 사용자가 직접 브리핑을 요청하기 전에는 AI를 호출하지 않는다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "home.db"
    )
    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "00_service_intro.py")
    ).run(timeout=30)

    assert not page.exception
    assert any(
        item.value == "AI 학생성공 코파일럿" for item in page.subheader
    )
    assert any(
        button.label == "AI 브리핑 생성" for button in page.button
    )
    assert any(
        item.value == "바로 시작하기" for item in page.subheader
    )
    rendered_markdown = "\n".join(str(item.value) for item in page.markdown)
    assert all(
        title in rendered_markdown
        for title in ("전체 초기경보", "AI 맞춤 추천 통합 검토", "시연 준비")
    )
    assert any(
        item.label == "서비스 구성·운영 원칙 자세히 보기"
        for item in page.expander
    )
    assert page.pills[0].options == [
        "오늘 우선 확인할 신호",
        "지원 전 학생 확인",
        "위험유형 흐름 요약",
    ]


def test_app_opens_role_entry_before_any_service_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """미인증 루트 주소는 학생 데이터를 열지 않고 시연용 계정을 확인한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "default-start.db"
    )
    page = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)

    assert not page.exception
    rendered_markdown = "\n".join(str(item.value) for item in page.markdown)
    assert "이용할 계정을 확인해 주세요" in rendered_markdown
    assert len(page.chat_input) == 0
    assert any(button.label == "계정으로 시작" for button in page.button)


@pytest.mark.parametrize(
    ("identity", "expected_text"),
    (
        (
            {
                "role": ROLE_STUDENT,
                "account_id": "S0003",
                "student_id": "S0003",
            },
            "오늘은 어떤 이야기를 나눠볼까요?",
        ),
        (
            {
                "role": ROLE_STAFF,
                "account_id": "STAFF001",
                "student_id": None,
            },
            "전체 대시보드",
        ),
    ),
)
def test_authenticated_role_opens_its_own_default_page(
    identity: dict[str, str | None],
    expected_text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """학생과 교직원은 로그인 직후 서로 다른 기본 화면으로 이동한다."""

    monkeypatch.setattr(
        "src.database.DEFAULT_DATABASE_PATH", tmp_path / "role-default.db"
    )
    page = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    page.session_state[AUTH_IDENTITY_KEY] = identity
    page.run(timeout=30)

    assert not page.exception
    rendered = "\n".join(str(item.value) for item in page.markdown)
    assert expected_text in rendered
    if identity["role"] == ROLE_STUDENT:
        assert page.session_state["demo_student_id"] == "S0003"
        assert not any(
            selectbox.label == "학생 모드 데모 ID"
            for selectbox in page.selectbox
        )
    else:
        assert "오늘은 어떤 이야기를 나눠볼까요?" not in rendered
        navigation_groups = {
            item.label: item.proto.expanded
            for item in page.expander
            if item.label in {"교직원 핵심 흐름", "교육과정", "관리자"}
        }
        assert navigation_groups == {
            "교직원 핵심 흐름": True,
            "교육과정": True,
            "관리자": False,
        }


def test_dashboard_does_not_render_ai_briefing() -> None:
    """전체 대시보드는 위험 현황에 집중하고 별도 AI 브리핑을 표시하지 않는다."""

    page = AppTest.from_file(
        str(PROJECT_ROOT / "pages" / "01_dashboard.py")
    ).run(timeout=30)

    assert not page.exception
    rendered = "\n".join(str(item.value) for item in page.markdown)
    assert "AI 오늘의 브리핑" not in rendered
    assert not any(
        button.label == "현재 조건 분석" for button in page.button
    )
    assert any("전체 대시보드" in str(item.value) for item in page.markdown)
