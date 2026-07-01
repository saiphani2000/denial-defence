"""Tests for agent output schemas."""

from agents.schemas import AgentClaimOutput, CritiqueOutput, validate_agent_output


def test_agent_claim_output():
    data = validate_agent_output(
        {"claim": "Test", "evidence": ["a", "b"]},
        AgentClaimOutput,
    )
    assert data["claim"] == "Test"
    assert len(data["evidence"]) == 2


def test_critique_revision_flag():
    data = validate_agent_output(
        {
            "weakest_claim_attacked": "policy_citation",
            "critique": "Missing criteria",
            "what_would_strengthen_appeal": "Add dates",
            "revision_needed": False,
        },
        CritiqueOutput,
    )
    assert data["revision_needed"] is False
