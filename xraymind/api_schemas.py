"""Pydantic schemas for the XRayMind hosted API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CaseStatus = Literal["pending", "reviewed", "deferred", "flagged", "archived"]
CasePriority = Literal["routine", "elevated", "urgent"]
ReviewDecision = Literal["agree", "disagree", "uncertain", "defer", "flag"]


class ServiceHealth(BaseModel):
    status: str
    service: str
    version: str
    db_path: str
    auth_mode: str
    disclaimer: str


class CaseRecord(BaseModel):
    id: int
    image_path: str
    image_id: str | None = None
    source_filename: str | None = None
    model_name: str | None = None
    status: CaseStatus
    priority: CasePriority
    patient_context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    assigned_to: str | None = None
    due_at: str | None = None
    needs_second_reader: bool = False
    created_at: str
    updated_at: str


class PredictionRecord(BaseModel):
    id: int
    case_id: int
    model_name: str
    threshold: float
    top_k: int
    prediction_json: dict[str, Any]
    max_probability: float | None = None
    low_confidence: bool = False
    created_at: str


class ReviewRecord(BaseModel):
    id: int
    case_id: int
    reviewer: str | None = None
    decision: ReviewDecision
    notes: str | None = None
    final_labels: dict[str, Any] = Field(default_factory=dict)
    review_round: int = 1
    created_at: str


class CaseDetailResponse(BaseModel):
    case: CaseRecord
    latest_prediction: PredictionRecord | None = None
    reviews: list[ReviewRecord] = Field(default_factory=list)


class CaseListResponse(BaseModel):
    cases: list[CaseRecord]
    count: int


class CaseEnvelope(BaseModel):
    case: CaseRecord


class ReviewEnvelope(BaseModel):
    review: ReviewRecord
    case_detail: CaseDetailResponse


class ReviewerQueueResponse(BaseModel):
    cases: list[dict[str, Any]]
    count: int


class PrincipalResponse(BaseModel):
    key_id: str
    role: str
