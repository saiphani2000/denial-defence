"""Pydantic schemas for agent JSON outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentClaimOutput(BaseModel):
    claim: str = ""
    evidence: list[Any] = Field(default_factory=list)
    exceptions_to_invoke: list[str] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]


class PrecedentOutput(BaseModel):
    claim: str = ""
    evidence: list[Any] = Field(default_factory=list)
    imr_references: list[str] = Field(default_factory=list)


class CritiqueOutput(BaseModel):
    weakest_claim_attacked: str = ""
    critique: str = ""
    what_would_strengthen_appeal: str = ""
    revision_needed: bool = True
    round: int = 0


def validate_agent_output(data: dict, schema: type[BaseModel]) -> dict:
    """Validate and normalize agent JSON; raises ValueError on hard failure."""
    return schema.model_validate(data).model_dump()
