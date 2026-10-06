"""Integration and endpoint tests for the multi-agent ecosystem:
- Haiiro: Caregiver triage companion (RED/YELLOW/GREEN risk, media ambiguity, vet escalation)
- Michelin: Veterinary co-pilot for Dr. Andy (hypothesis cross-examination, differential audit, lab trends)
- Central Patient Store (SSOT): Unified multimodal record preventing state drift
- FastAPI endpoints for all agent operations
"""

import copy
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from michelin_core.api.main import create_app
from michelin_core.api.dependencies import get_store, get_llm
from michelin_core.schemas.patient import (
    AMBIGUOUS_FLAG,
    HaiiroTriageRequest,
    HomeObservationInput,
    MediaAttachment,
    MichelinCrossExamRequest,
    PatientRecord,
    TriageLevel,
)
from michelin_core.services.agents import HaiiroTriageAgent, MichelinCoPilotAgent
from michelin_core.services.gemini_client import GeminiClient
from michelin_core.services.patient_store import PatientStore


@pytest.fixture
def fresh_store() -> PatientStore:
    """Fixture providing an isolated in-memory PatientStore seeded from mock_records.json."""
    return PatientStore.from_json()


@pytest.fixture
def dummy_llm() -> GeminiClient:
    """Fixture providing an LLM client in deterministic fallback mode (available=False)."""
    return GeminiClient(sdk_client=None, enabled=False)


@pytest.fixture
def app_client(fresh_store: PatientStore, dummy_llm: GeminiClient) -> TestClient:
    """FastAPI TestClient with injected test store and dummy LLM."""
    app = create_app()
    app.dependency_overrides[get_store] = lambda: fresh_store
    app.dependency_overrides[get_llm] = lambda: dummy_llm
    return TestClient(app)


# =========================================================================== #
# Haiiro Triage Agent Tests
# =========================================================================== #


class TestHaiiroTriageAgent:
    def test_emergency_red_triage_breathing_difficulty(self, fresh_store: PatientStore, dummy_llm: GeminiClient):
        agent = HaiiroTriageAgent(llm=dummy_llm, store=fresh_store)
        req = HaiiroTriageRequest(
            patient_id="MICH-001",
            observation=HomeObservationInput(
                symptoms=["open-mouth breathing", "lethargy"],
                free_text="Michelin is panting with his mouth open and laying on the floor",
                breathing_difficulty=True,
            ),
        )

        initial_version = fresh_store.get("MICH-001").record_version
        resp = agent.triage(req)

        assert resp.triage_level == TriageLevel.RED
        assert resp.escalate_to_vet is True
        assert resp.escalation is not None
        assert resp.escalation.recipient == "Dr. Andy"
        assert resp.escalation.level == TriageLevel.RED
        assert any("breathing difficulty" in hit.lower() for hit in resp.red_flags)
        assert resp.record_version == initial_version + 1

        # Check SSOT persistence - no state drift
        updated = fresh_store.get("MICH-001")
        assert len(updated.home_observations) > 1
        assert updated.escalations[-1].escalation_id == resp.escalation.escalation_id

    def test_emergency_red_triage_male_cat_straining_to_urinate(self, fresh_store: PatientStore, dummy_llm: GeminiClient):
        agent = HaiiroTriageAgent(llm=dummy_llm, store=fresh_store)
        req = HaiiroTriageRequest(
            patient_id="MICH-001",
            observation=HomeObservationInput(
                symptoms=["straining in litter box", "crying"],
                free_text="He goes into the box every 5 minutes and cries, no urine coming out",
                straining_to_urinate=True,
                hours_since_last_urination=24,
            ),
        )

        resp = agent.triage(req)
        assert resp.triage_level == TriageLevel.RED
        assert resp.escalate_to_vet is True
        assert any("straining to urinate" in rf.lower() for rf in resp.red_flags)

    def test_yellow_triage_escalation_for_comorbid_feline(self, fresh_store: PatientStore, dummy_llm: GeminiClient):
        # Vomiting 2x in 24h + reduced appetite in a cat with active CKD / pancreatitis -> YELLOW + escalate
        agent = HaiiroTriageAgent(llm=dummy_llm, store=fresh_store)
        req = HaiiroTriageRequest(
            patient_id="MICH-001",
            observation=HomeObservationInput(
                symptoms=["vomiting", "reduced appetite"],
                free_text="He threw up bile twice today and did not finish his food.",
                vomiting_episodes_24h=2,
                appetite="reduced",
            ),
        )

        resp = agent.triage(req)
        assert resp.triage_level == TriageLevel.YELLOW
        assert resp.escalate_to_vet is True  # Escalates because of active problems
        assert resp.escalation is not None
        assert "vomiting 2x" in resp.escalation.reason

    def test_green_triage_routine_observation(self, fresh_store: PatientStore, dummy_llm: GeminiClient):
        agent = HaiiroTriageAgent(llm=dummy_llm, store=fresh_store)
        req = HaiiroTriageRequest(
            patient_id="MICH-001",
            observation=HomeObservationInput(
                symptoms=[],
                free_text="Michelin ate all his wet food and drank normally today. Good energy.",
                appetite="normal",
                water_intake="normal",
                vomiting_episodes_24h=0,
            ),
        )

        resp = agent.triage(req)
        assert resp.triage_level == TriageLevel.GREEN
        assert resp.escalate_to_vet is False
        assert resp.escalation is None

    def test_media_ambiguity_enforced_in_triage(self, fresh_store: PatientStore, dummy_llm: GeminiClient):
        agent = HaiiroTriageAgent(llm=dummy_llm, store=fresh_store)
        req = HaiiroTriageRequest(
            patient_id="MICH-001",
            observation=HomeObservationInput(
                symptoms=["vomiting"],
                free_text="Here is a photo of the vomit",
                media=[
                    MediaAttachment(
                        media_type="photo",
                        mime_type="image/jpeg",
                        uri="file://tmp/blurry_vomit.jpg",
                        caregiver_caption="Blurry photo taken in the dark",
                        assessment_status="AMBIGUOUS",
                        flag=AMBIGUOUS_FLAG,
                    )
                ],
            ),
        )

        resp = agent.triage(req)
        assert len(resp.media_assessments) == 1
        m_assess = resp.media_assessments[0]
        assert m_assess.status == "AMBIGUOUS"
        assert m_assess.finding is None
        assert m_assess.flag == AMBIGUOUS_FLAG


# =========================================================================== #
# Michelin Co-Pilot Agent Tests
# =========================================================================== #


class TestMichelinCoPilotAgent:
    def test_cross_examination_challenges_dr_andy_hypothesis(self, fresh_store: PatientStore, dummy_llm: GeminiClient):
        agent = MichelinCoPilotAgent(llm=dummy_llm, store=fresh_store)
        req = MichelinCrossExamRequest(
            patient_id="MICH-001",
            clinician="Dr. Andy",
            working_hypothesis="Simple gastritis / dietary indiscretion. Labs nominally within reference range.",
            clinical_question="Can we manage conservatively with antiemetics?",
        )

        resp = agent.cross_examine(req)

        # 1. IRIS CKD Staging: Stage 2 despite nominal lab normal creatinine
        assert resp.iris_staging.stage == 2
        assert "IRIS Stage 2 CKD" in resp.iris_staging.stage_label
        assert resp.iris_staging.latest_creatinine == 2.3
        assert resp.iris_staging.usg_adequate is False

        # 2. Triaditis assessment: Pancreatic elevation + GI + Liver
        assert resp.triaditis.classification == "triaditis_suspected"
        assert resp.triaditis.latest_fpli == 6.8

        # 3. Longitudinal trends present
        crea_trend = next(t for t in resp.lab_trends if t.analyte == "creatinine")
        assert crea_trend.direction == "rising"
        assert crea_trend.first_value == 1.4
        assert crea_trend.last_value == 2.3

        # 4. Ambiguous data highlighted
        assert len(resp.ambiguous_data) >= 1
        amb_plt = next(a for a in resp.ambiguous_data if a.analyte == "platelets")
        assert amb_plt.value is None
        assert amb_plt.status == "AMBIGUOUS"
        assert amb_plt.flag == AMBIGUOUS_FLAG

        # 5. Differentials include IRIS CKD and Triaditis
        diff_conditions = [d.condition for d in resp.differentials]
        assert any("CKD" in cond for cond in diff_conditions)
        assert any("triaditis" in cond.lower() or "pancreatitis" in cond.lower() for cond in diff_conditions)

        # 6. Challenges to hypothesis challenge the "simple gastritis" premise
        assert len(resp.challenges_to_hypothesis) > 0


# =========================================================================== #
# FastAPI Endpoint Integration Tests
# =========================================================================== #


class TestFastAPIEndpoints:
    def test_root_redirects_to_docs(self, app_client: TestClient):
        res = app_client.get("/", follow_redirects=False)
        assert res.status_code in (302, 307)
        assert res.headers["location"] == "/docs"

    def test_health_endpoint(self, app_client: TestClient):
        res = app_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "models" in data
        assert data["models"]["triage"] == "gemini-2.5-flash"
        assert data["models"]["cross_examination"] == "gemini-2.5-pro"

    def test_get_patient_record(self, app_client: TestClient):
        res = app_client.get("/api/records/MICH-001")
        assert res.status_code == 200
        data = res.json()
        assert data["patient_id"] == "MICH-001"
        assert data["name"] == "Michelin"
        assert data["species"] == "feline"
        assert data["age_years"] == 10.5
        assert len(data["labs"]) > 10

    def test_get_patient_not_found(self, app_client: TestClient):
        res = app_client.get("/api/records/NONEXISTENT-999")
        assert res.status_code == 404

    def test_get_patient_labs_and_trends(self, app_client: TestClient):
        res_labs = app_client.get("/api/records/MICH-001/labs?analyte=creatinine")
        assert res_labs.status_code == 200
        labs = res_labs.json()
        assert len(labs) == 3
        assert [lab["value"] for lab in labs] == [1.4, 1.9, 2.3]

        res_trend = app_client.get("/api/records/MICH-001/trends/creatinine")
        assert res_trend.status_code == 200
        trend = res_trend.json()
        assert trend["direction"] == "rising"
        assert trend["first_value"] == 1.4
        assert trend["last_value"] == 2.3
        assert trend["percent_change"] == 64.3

    def test_haiiro_triage_endpoint(self, app_client: TestClient):
        payload = {
            "patient_id": "MICH-001",
            "observation": {
                "symptoms": ["breathing difficulty", "open mouth panting"],
                "free_text": "He is breathing heavily with mouth open",
                "breathing_difficulty": True,
            },
        }
        res = app_client.post("/api/haiiro/triage", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["triage_level"] == "RED"
        assert data["escalate_to_vet"] is True
        assert data["escalation"]["recipient"] == "Dr. Andy"

    def test_michelin_cross_examine_endpoint(self, app_client: TestClient):
        payload = {
            "patient_id": "MICH-001",
            "clinician": "Dr. Andy",
            "working_hypothesis": "Acute indiscretion",
            "clinical_question": "Does this look like acute or chronic illness?",
        }
        res = app_client.post("/api/michelin/cross-examine", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["iris_staging"]["stage"] == 2
        assert data["triaditis"]["classification"] == "triaditis_suspected"
        assert len(data["challenges_to_hypothesis"]) > 0

    def test_michelin_escalations_inbox(self, app_client: TestClient):
        # Trigger an escalation via Haiiro first
        triage_payload = {
            "patient_id": "MICH-001",
            "observation": {
                "symptoms": ["vomiting"],
                "free_text": "Vomiting repeatedly today",
                "vomiting_episodes_24h": 3,
            },
        }
        triage_res = app_client.post("/api/haiiro/triage", json=triage_payload)
        assert triage_res.status_code == 200
        esc_id = triage_res.json()["escalation"]["escalation_id"]

        # Dr. Andy fetches escalations inbox
        esc_res = app_client.get("/api/michelin/escalations?status=OPEN")
        assert esc_res.status_code == 200
        escalations = esc_res.json()
        assert any(e["escalation_id"] == esc_id for e in escalations)
