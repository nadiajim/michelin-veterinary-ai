"""Haiiro - caregiver-facing triage endpoints."""

from fastapi import APIRouter, Depends

from michelin_core.api.dependencies import get_haiiro_agent
from michelin_core.schemas.patient import HaiiroTriageRequest, HaiiroTriageResponse
from michelin_core.services.agents import HaiiroTriageAgent

router = APIRouter(prefix="/api/haiiro", tags=["haiiro"])


@router.post("/triage", response_model=HaiiroTriageResponse, summary="Triage a caregiver home observation")
def triage(request: HaiiroTriageRequest, agent: HaiiroTriageAgent = Depends(get_haiiro_agent)) -> HaiiroTriageResponse:
    """Classify a home observation as RED / YELLOW / GREEN, persist it to the SSOT and
    escalate to Dr. Andy when thresholds are breached. Deterministic red-flag rules can
    escalate but never be downgraded by the model."""
    return agent.triage(request)
