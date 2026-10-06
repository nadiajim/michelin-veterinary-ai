"""Michelin - veterinary co-pilot endpoints for Dr. Andy."""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from michelin_core.api.dependencies import get_michelin_agent, get_store
from michelin_core.schemas.patient import Escalation, MichelinCrossExamRequest, MichelinCrossExamResponse
from michelin_core.services.agents import MichelinCoPilotAgent
from michelin_core.services.patient_store import PatientStore

router = APIRouter(prefix="/api/michelin", tags=["michelin"])


@router.post("/cross-examine", response_model=MichelinCrossExamResponse, summary="Clinical cross-examination")
def cross_examine(
    request: MichelinCrossExamRequest, agent: MichelinCoPilotAgent = Depends(get_michelin_agent)
) -> MichelinCrossExamResponse:
    """Challenge the clinician's working hypothesis using SSOT data: IRIS staging, triaditis
    pattern, longitudinal trends, ambiguous data and open Haiiro escalations. All model
    output is verified against the SSOT; unverifiable numbers are redacted."""
    return agent.cross_examine(request)


@router.get("/escalations", response_model=list[Escalation], summary="Haiiro escalations awaiting Dr. Andy")
def escalations(
    status: Literal["OPEN", "ACKNOWLEDGED", "RESOLVED"] | None = Query(default="OPEN"),
    store: PatientStore = Depends(get_store),
) -> list[Escalation]:
    out: list[Escalation] = []
    for pid in store.list_ids():
        out += [e for e in store.get(pid).escalations if status is None or e.status == status]
    return sorted(out, key=lambda e: e.created_at, reverse=True)
