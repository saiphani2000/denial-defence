"""Tests for IMR retrieval."""

from agents.imr_retrieval import format_precedent_context, retrieve_similar_cases


def test_retrieve_returns_list():
    cases = retrieve_similar_cases(
        "Medical necessity denial for pharmacy medication diabetes",
        "Type 2 diabetes patient failed metformin",
        top_k=2,
    )
    assert isinstance(cases, list)


def test_format_empty():
    text = format_precedent_context([])
    assert "No indexed IMR" in text


def test_format_with_cases():
    cases = [{"reference_id": "MN24-123", "report_year": 2024, "diagnosis": "Endocrine",
              "treatment": "Pharmacy", "determination": "Overturned", "reviewer_excerpt": "x"}]
    text = format_precedent_context(cases)
    assert "MN24-123" in text
