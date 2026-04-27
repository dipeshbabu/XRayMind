"""Pydantic schemas for the XRayMind hosted API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CaseStatus = Literal["pending", "reviewed", "deferred", "flagged", "archived"]
CasePriority = Literal["routine", "elevated", "urgent"]
ReviewDecision = Literal["agree", "disagree", "uncertain", "defer", "flag"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


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


class HostedJobRecord(BaseModel):
    id: int
    tenant_id: str
    job_type: str
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    attempts: int = 0
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class CaseDetailResponse(BaseModel):
    case: CaseRecord
    latest_prediction: PredictionRecord | None = None
    reviews: list[ReviewRecord] = Field(default_factory=list)


class CaseListResponse(BaseModel):
    cases: list[CaseRecord]
    count: int


class JobListResponse(BaseModel):
    jobs: list[HostedJobRecord]
    count: int


class CaseEnvelope(BaseModel):
    case: CaseRecord


class HostedJobEnvelope(BaseModel):
    job: HostedJobRecord | None = None


class ReviewEnvelope(BaseModel):
    review: ReviewRecord
    case_detail: CaseDetailResponse


class ReviewerQueueResponse(BaseModel):
    cases: list[dict[str, Any]]
    count: int


class PrincipalResponse(BaseModel):
    key_id: str
    role: str
