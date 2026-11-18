"""Closed outputs; only bounded, allowlisted Lean proofs reach the compiler."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SimulatorRequest(StructuredModel):
    operation: Literal["first_passage", "compare_potentials"]
    graph: Literal["path3", "paper8", "paper9"]
    q: float | list[float]
    q_other: float | list[float] | None
    T: int = Field(ge=1, le=64)
    h: float = Field(gt=0, le=1)
    observed: list[int] | None


class Certificate(StructuredModel):
    kind: Literal["lean4"]
    target_id: str = Field(min_length=1, max_length=100)
    proof: str = Field(min_length=1, max_length=16000)


class Repair(StructuredModel):
    gap_id: str = Field(min_length=1, max_length=100)
    change: str = Field(min_length=1, max_length=4000)


class MilestoneAttempt(StructuredModel):
    benchmark_id: str = Field(min_length=1, max_length=100)
    proof: str = Field(min_length=1, max_length=16000)


class Proposal(StructuredModel):
    claim: str = Field(min_length=1, max_length=12000)
    assumptions: list[str] = Field(max_length=40)
    proof: str = Field(max_length=200000)
    milestone_attempts: list[MilestoneAttempt] = Field(default_factory=list, max_length=8)
    repairs: list[Repair] = Field(max_length=20)
    remaining_gaps: list[str] = Field(max_length=20)
    certificate: Certificate | None
    simulator_requests: list[SimulatorRequest] = Field(max_length=2)


class Gap(StructuredModel):
    gap_id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=4000)
    severity: Literal["fatal", "major", "minor"]
    scope: Literal["general_proof", "finite_certificate", "numerical_probe"]
    suggested_repair: str = Field(max_length=4000)


class Critique(StructuredModel):
    assessment: str = Field(min_length=1, max_length=8000)
    target_preserved: bool
    assumptions_preserved: bool
    gaps: list[Gap] = Field(max_length=20)
    simulator_requests: list[SimulatorRequest] = Field(max_length=2)


class VerificationAdvice(StructuredModel):
    assessment: str = Field(max_length=24000)
    predicted_status: Literal["verified", "not_verified", "uncertain"]
    diagnostic: str = Field(max_length=24000)
    suggested_next_step: str = Field(max_length=24000)
