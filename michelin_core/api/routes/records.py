"""Central Patient Store (SSOT) endpoints."""

from fastapi import APIRouter, Depends

from michelin_core.api.dependencies import get_extraction_agent, get_store
from michelin_core.schemas.patient import (
    LabExtractionRequest,
    LabExtractionResponse,
    LabResult,
    PatientRecord,
    TrendSummary,
)
from michelin_core.services import verification as v
from michelin_core.services.agents import LabExtractionAgent
from michelin_core.services.patient_store import PatientStore

router = APIRouter(prefix="/api/records", tags=["records"])


@router.get("", response_model=list[str], summary="List patient IDs")
def list_patients(store: PatientStore = Depends(get_store)) -> list[str]:
    return store.list_ids()


@router.get("/{patient_id}", response_model=PatientRecord, summary="Full SSOT patient record")
def get_record(patient_id: str, store: PatientStore = Depends(get_store)) -> PatientRecord:
    return store.get(patient_id)


@router.get("/{patient_id}/labs", response_model=list[LabResult], summary="Lab results (optionally by analyte)")
def get_labs(patient_id: str, analyte: str | None = None, store: PatientStore = Depends(get_store)) -> list[LabResult]:
    record = store.get(patient_id)
    if analyte is None:
        return sorted(record.labs, key=lambda lab: (lab.collected_at, lab.analyte))
    return v.labs_for(record, v.canonical_analyte(analyte))


@router.get("/{patient_id}/trends/{analyte}", response_model=TrendSummary, summary="Longitudinal trend")
def get_trend(patient_id: str, analyte: str, store: PatientStore = Depends(get_store)) -> TrendSummary:
    record = store.get(patient_id)
    if analyte in ("weight", "body_weight"):
        return v.compute_weight_trend(record)
    return v.compute_lab_trend(record, v.canonical_analyte(analyte))


@router.post(
    "/{patient_id}/labs/extract", response_model=LabExtractionResponse,
    summary="Strict-protocol OCR of a lab report (optionally commit to SSOT)",
)
def extract_labs(
    patient_id: str, request: LabExtractionRequest, agent: LabExtractionAgent = Depends(get_extraction_agent)
) -> LabExtractionResponse:
    return agent.extract(patient_id, request)
