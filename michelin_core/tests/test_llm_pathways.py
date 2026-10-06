"""Mock LLM tests verifying that live-style LLM structured responses are correctly ingested
and guarded against hallucinations or protocol violations.
"""

from unittest.mock import MagicMock
import pytest

from michelin_core.schemas.patient import (
    AMBIGUOUS_FLAG,
    CitedLabValue,
    Differential,
    HaiiroTriageRequest,
    HomeObservationInput,
    LLMCrossExamination,
    LLMTriageAssessment,
    MediaAssessment,
    MediaAttachment,
    MichelinCrossExamRequest,
    TriageLevel,
)
from michelin_core.services.agents import HaiiroTriageAgent, MichelinCoPilotAgent
from michelin_core.services.gemini_client import GeminiClient
from michelin_core.services.patient_store import PatientStore


class TestLLMPathwaysWithMockClient:
    def test_haiiro_with_mock_gemini_client(self):
        store = PatientStore.from_json()

        # Mock the SDK client
        mock_sdk = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = LLMTriageAssessment(
            triage_level=TriageLevel.YELLOW,
            rationale="Cat is exhibiting reduced appetite and vomiting in the context of chronic kidney disease.",
            red_flags=["YELLOW: vomiting", "YELLOW: reduced appetite"],
            recommend_vet_escalation=True,
            caregiver_message="Thank you for checking in on Michelin. Because of his kidney history, we want Dr. Andy's team to evaluate this today.",
            recommended_actions=["Call clinic today", "Do not give NSAIDs"],
            media_assessments=[
                MediaAssessment(
                    media_id="MED-TEST-1",
                    status="AMBIGUOUS",
                    finding="Possible vomit",  # Model attempted to write a finding for ambiguous media!
                    flag=AMBIGUOUS_FLAG,
                )
            ],
        )
        mock_sdk.models.generate_content.return_value = mock_response

        client = GeminiClient(sdk_client=mock_sdk, enabled=True)
        agent = HaiiroTriageAgent(llm=client, store=store)

        req = HaiiroTriageRequest(
            patient_id="MICH-001",
            observation=HomeObservationInput(
                symptoms=["vomiting", "reduced appetite"],
                free_text="Michelin threw up and only ate half his food",
                vomiting_episodes_24h=2,
                appetite="reduced",
                media=[
                    MediaAttachment(
                        media_id="MED-TEST-1",
                        media_type="photo",
                        mime_type="image/jpeg",
                        uri="file://tmp/vomit.jpg",
                    )
                ],
            ),
        )

        resp = agent.triage(req)

        assert resp.llm_used is True
        assert resp.model == "gemini-2.5-flash"
        assert resp.triage_level == TriageLevel.YELLOW
        assert resp.escalate_to_vet is True

        # Check media guardrail: finding was scrubbed because status is AMBIGUOUS
        assert resp.media_assessments[0].status == "AMBIGUOUS"
        assert resp.media_assessments[0].finding is None
        assert resp.media_assessments[0].flag == AMBIGUOUS_FLAG
        assert any(v.code == "MEDIA_AMBIGUITY_ENFORCED" for v in resp.guardrail_violations)

    def test_michelin_with_mock_gemini_client_and_hallucination_neutralization(self):
        store = PatientStore.from_json()

        mock_sdk = MagicMock()
        mock_response = MagicMock()
        # LLM attempts to fabricate a lab value (Creatinine 7.2 mg/dL) and invent a platelet number
        mock_response.parsed = LLMCrossExamination(
            clinical_summary="Michelin has progressive azotemia with Creatinine at 2.3 mg/dL and recent surge to 7.2 mg/dL.",
            challenges_to_hypothesis=[
                "Working hypothesis of simple gastritis does not account for IRIS Stage 2 CKD or fPLI of 6.8 µg/L.",
                "Platelet count of 450 K/µL indicates reactive thrombocytosis." # Hallucinated 450!
            ],
            differentials=[
                Differential(
                    condition="Feline Chronic Kidney Disease (IRIS Stage 2)",
                    likelihood="high",
                    supporting_evidence=["Creatinine 2.3 mg/dL with inadequate USG 1.022"],
                    contradicting_evidence=[],
                    next_steps=["Recheck renal panel in 2 weeks"],
                )
            ],
            cited_lab_values=[
                CitedLabValue(analyte="creatinine", collected_at="2026-09-22", value=2.3, unit="mg/dL"),
                # Fabricated citation:
                CitedLabValue(analyte="creatinine", collected_at="2026-10-06", value=7.2, unit="mg/dL"),
                # Ambiguous citation:
                CitedLabValue(analyte="platelets", collected_at="2026-09-22", value=450.0, unit="K/µL"),
            ],
            recommended_diagnostics=["Abdominal ultrasound", "Total T4"],
            data_gaps=["Cobalamin", "Folate"],
        )
        mock_sdk.models.generate_content.return_value = mock_response

        client = GeminiClient(sdk_client=mock_sdk, enabled=True)
        agent = MichelinCoPilotAgent(llm=client, store=store)

        req = MichelinCrossExamRequest(
            patient_id="MICH-001",
            clinician="Dr. Andy",
            working_hypothesis="Simple gastritis",
        )

        resp = agent.cross_examine(req)

        assert resp.llm_used is True
        assert resp.model == "gemini-2.5-pro"

        # 1. Hallucinated 7.2 in summary was redacted!
        assert "7.2" not in resp.clinical_summary
        assert "[UNVERIFIED-VALUE-REDACTED]" in resp.clinical_summary

        # 2. Fabricated citation 7.2 was removed from verified_citations
        assert not any(c.collected_at == "2026-10-06" for c in resp.verified_citations)

        # 3. Ambiguous platelet citation was forced to null
        plt_cite = next(c for c in resp.verified_citations if c.analyte == "platelets")
        assert plt_cite.value is None

        # 4. Guardrail violations recorded
        violation_codes = [v.code for v in resp.guardrail_violations]
        assert "UNVERIFIED_NUMBER_IN_TEXT" in violation_codes
        assert "FABRICATED_VALUE" in violation_codes
        assert "AMBIGUOUS_VALUE_FILLED" in violation_codes

    def test_lab_extraction_endpoint_with_mock_gemini(self):
        from fastapi.testclient import TestClient
        from michelin_core.api.main import create_app
        from michelin_core.api.dependencies import get_store, get_llm
        from michelin_core.schemas.patient import LLMLabExtraction, LLMExtractedLabItem

        store = PatientStore.from_json()
        mock_sdk = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = LLMLabExtraction(
            items=[
                LLMExtractedLabItem(
                    analyte="creatinine",
                    display_name="Creatinine",
                    raw_text="2.5",
                    value=2.5,
                    unit="mg/dL",
                    reference_low=0.6,
                    reference_high=2.4,
                    legible=True,
                    confidence=0.98,
                ),
                LLMExtractedLabItem(
                    analyte="platelets",
                    display_name="Platelet Count",
                    raw_text="blurred clump",
                    value=None,
                    unit="K/µL",
                    reference_low=151.0,
                    reference_high=600.0,
                    legible=False,
                    confidence=0.40,
                ),
            ],
            document_notes="Faint carbon copy",
        )
        mock_sdk.models.generate_content.return_value = mock_response
        client = GeminiClient(sdk_client=mock_sdk, enabled=True)

        app = create_app()
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_llm] = lambda: client
        test_client = TestClient(app)

        payload = {
            "document_text": "Chemistry: Creatinine 2.5 mg/dL (0.6-2.4). CBC: Platelets blurred clump.",
            "collected_at": "2026-10-06",
            "source_document": "Followup-Report.pdf",
            "laboratory": "Reference Lab",
            "commit": True,
        }

        res = test_client.post("/api/records/MICH-001/labs/extract", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["committed"] is True
        assert data["ambiguous_count"] == 1
        assert len(data["labs"]) == 2

        # Check creatinine
        crea = next(lab for lab in data["labs"] if lab["analyte"] == "creatinine")
        assert crea["value"] == 2.5
        assert crea["status"] == "HIGH"

        # Check platelets
        plt = next(lab for lab in data["labs"] if lab["analyte"] == "platelets")
        assert plt["value"] is None
        assert plt["status"] == "AMBIGUOUS"
        assert plt["flag"] == AMBIGUOUS_FLAG

        # Check persistence in store
        persisted = store.get("MICH-001")
        assert any(l.value == 2.5 and l.analyte == "creatinine" for l in persisted.labs)

