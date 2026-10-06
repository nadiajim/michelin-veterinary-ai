"""FastAPI application entry point.

Run with:  uvicorn michelin_core.api.main:app --reload
"""

import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from michelin_core.api.dependencies import get_llm
from michelin_core.api.routes import haiiro, michelin, records
from michelin_core.services.gemini_client import (
    CROSS_EXAM_MODEL,
    EXTRACTION_MODEL,
    TRIAGE_MODEL,
    GeminiClient,
    LLMResponseError,
    LLMUnavailableError,
)
from michelin_core.services.patient_store import PatientNotFoundError, VersionConflictError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Michelin Veterinary Clinical Ecosystem",
        version="0.1.0",
        description=(
            "Michelin (veterinary co-pilot) + Haiiro (caregiver triage) over a shared Central Patient Store. "
            "Zero-hallucination guardrails: ambiguous data is always null + flagged; model output is verified "
            "against the SSOT before it reaches a clinician or caregiver."
        ),
    )
    app.include_router(records.router)
    app.include_router(haiiro.router)
    app.include_router(michelin.router)

    @app.exception_handler(PatientNotFoundError)
    async def _not_found(_: Request, exc: PatientNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Patient '{exc.patient_id}' not found"})

    @app.exception_handler(VersionConflictError)
    async def _conflict(_: Request, exc: VersionConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "current_version": exc.actual})

    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMResponseError)
    async def _llm_error(_: Request, exc: LLMResponseError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", tags=["system"], summary="Root endpoint - Redirects to Swagger UI docs")
    def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["system"])
    def health(llm: GeminiClient = Depends(get_llm)) -> dict:
        return {
            "status": "ok",
            "llm_available": llm.available,
            "models": {"triage": TRIAGE_MODEL, "cross_examination": CROSS_EXAM_MODEL, "extraction": EXTRACTION_MODEL},
        }

    return app


app = create_app()
