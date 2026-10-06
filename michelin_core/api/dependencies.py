"""Dependency providers. Tests override ``get_store`` and ``get_llm`` via ``app.dependency_overrides``."""

import os
from functools import lru_cache

from fastapi import Depends

from michelin_core.services.agents import HaiiroTriageAgent, LabExtractionAgent, MichelinCoPilotAgent
from michelin_core.services.gemini_client import GeminiClient
from michelin_core.services.patient_store import DEFAULT_SEED_PATH, PatientStore


@lru_cache(maxsize=1)
def get_store() -> PatientStore:
    return PatientStore.from_json(os.getenv("MICHELIN_SEED_PATH", str(DEFAULT_SEED_PATH)))


@lru_cache(maxsize=1)
def get_llm() -> GeminiClient:
    return GeminiClient()


def get_haiiro_agent(
    store: PatientStore = Depends(get_store), llm: GeminiClient = Depends(get_llm)
) -> HaiiroTriageAgent:
    return HaiiroTriageAgent(llm=llm, store=store)


def get_michelin_agent(
    store: PatientStore = Depends(get_store), llm: GeminiClient = Depends(get_llm)
) -> MichelinCoPilotAgent:
    return MichelinCoPilotAgent(llm=llm, store=store)


def get_extraction_agent(
    store: PatientStore = Depends(get_store), llm: GeminiClient = Depends(get_llm)
) -> LabExtractionAgent:
    return LabExtractionAgent(llm=llm, store=store)
