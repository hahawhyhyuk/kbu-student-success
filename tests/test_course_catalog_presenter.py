"""실제 개설강좌의 출처 표시가 의미를 만들지 않는지 검증한다."""

from src.course_catalog_presenter import format_course_caption


def test_source_backed_caption_uses_only_actual_fields_and_review_warning() -> None:
    caption = format_course_caption(
        {
            "course_id": "OFF-001",
            "source_course_id": "OFF-001",
            "catalog_source": "actual_course_offerings_2026",
            "department": "간호학과",
            "offering_departments": "간호학과|소프트웨어융합과",
            "grade_levels": "1|2",
            "offered_semesters": "1|2",
            "course_area": "전공",
            "course_type": "NCS",
            "class_method": "이론",
            "registration_check_required": True,
        },
        "소프트웨어융합과",
    )

    assert "소속 학과" in caption
    assert "공식 과목코드: OFF-001" in caption
    assert "간호학과, 소프트웨어융합과" in caption
    assert "전공 · NCS · 이론" in caption
    assert "실제 수강 가능 여부 확인 필요" in caption
    assert "역량" not in caption


def test_synthetic_caption_preserves_existing_prerequisite_display() -> None:
    caption = format_course_caption(
        {
            "course_id": "C002",
            "department": "간호학과",
            "grade_level": 2,
            "semester": 1,
            "difficulty": "중급",
            "credit": 3,
            "prerequisites": "C001",
        },
        "소프트웨어융합과",
    )

    assert "타과 선택" in caption
    assert "ID: C002" in caption
    assert "선수과목: C001" in caption
