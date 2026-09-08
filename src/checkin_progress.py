"""Kare 대화의 맥락 이해 진행 표시를 구성한다."""

from __future__ import annotations


def build_kare_context_progress_html(
    understood_count: int,
    total_count: int = 9,
) -> str:
    """Kare가 파악한 대화 맥락을 비평가적 진행 막대로 표시한다."""

    normalized_total = max(1, int(total_count))
    normalized_count = min(
        normalized_total,
        max(0, int(understood_count)),
    )
    progress_percent = normalized_count / normalized_total * 100
    if normalized_count <= 2:
        stage_label = "이야기를 살펴보는 중"
    elif normalized_count <= 5:
        stage_label = "핵심 맥락을 이해하는 중"
    else:
        stage_label = "내용을 확인할 준비가 되었어요"
    return (
        '<div class="kare-context-progress">'
        '<div class="kare-context-progress-head">'
        '<span class="kare-context-progress-title">'
        'Kare가 대화의 맥락을 이해하고 있어요'
        "</span>"
        f'<span class="kare-context-progress-stage">{stage_label}</span>'
        "</div>"
        '<div class="kare-context-progress-track" role="progressbar" '
        'aria-label="Kare 대화 맥락 이해 진행" aria-valuemin="0" '
        f'aria-valuemax="{normalized_total}" aria-valuenow="{normalized_count}">'
        '<span class="kare-context-progress-fill" '
        f'style="--kare-context-progress: {progress_percent:.1f}%"></span>'
        "</div></div>"
    )
