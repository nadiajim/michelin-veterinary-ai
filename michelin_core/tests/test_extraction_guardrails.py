"""Tests for non-negotiable medical guardrails:
1. Zero Hallucination: Never interpolate, guess, or extrapolate numeric values.
2. Strict OCR / Extraction Protocol: Any ambiguous, smudged, or blurred metric
   must yield status="AMBIGUOUS", value=null, and the exact required flag:
   "Data unreadable / ambiguous — manual clinical verification required".
3. Feline Domain Nuance: Longitudinal CKD staging (Creatinine 1.4 -> 2.3 mg/dL
   with USG 1.022 = IRIS Stage 2 CKD despite being within nominal reference limits)
   and Feline Triaditis pattern recognition.
"""

from datetime import date
import pytest
from pydantic import ValidationError

from michelin_core.schemas.patient import (
    AMBIGUOUS_FLAG,
    CitedLabValue,
    LabResult,
    LabStatus,
    LLMExtractedLabItem,
    LLMLabExtraction,
    PatientRecord,
)
from michelin_core.services.patient_store import PatientStore
from michelin_core.services import verification as v


class TestStrictPydanticLabResultGuardrails:
    """Type-level validation prevents invalid medical states from existing."""

    def test_ambiguous_result_must_have_null_value(self):
        with pytest.raises(ValidationError, match="AMBIGUOUS results must have value=null"):
            LabResult(
                lab_id="LAB-TEST-001",
                analyte="platelets",
                display_name="Platelet Count",
                panel="cbc",
                value=250.0,  # Invalid: cannot have value if status is AMBIGUOUS
                unit="K/µL",
                reference_low=151.0,
                reference_high=600.0,
                status=LabStatus.AMBIGUOUS,
                flag=AMBIGUOUS_FLAG,
                collected_at=date(2026, 9, 22),
                source_document="test.pdf",
            )

    def test_ambiguous_result_requires_exact_flag(self):
        with pytest.raises(ValidationError, match="must carry flag"):
            LabResult(
                lab_id="LAB-TEST-002",
                analyte="platelets",
                display_name="Platelet Count",
                panel="cbc",
                value=None,
                unit="K/µL",
                reference_low=151.0,
                reference_high=600.0,
                status=LabStatus.AMBIGUOUS,
                flag="Unclear value",  # Invalid: must match exact AMBIGUOUS_FLAG string
                collected_at=date(2026, 9, 22),
                source_document="test.pdf",
            )

    def test_null_value_must_be_flagged_ambiguous(self):
        with pytest.raises(ValidationError, match="missing value must be recorded as status=AMBIGUOUS"):
            LabResult(
                lab_id="LAB-TEST-003",
                analyte="creatinine",
                display_name="Creatinine",
                panel="chemistry",
                value=None,  # Cannot be silently null with NORMAL status
                unit="mg/dL",
                reference_low=0.6,
                reference_high=2.4,
                status=LabStatus.NORMAL,
                collected_at=date(2026, 9, 22),
                source_document="test.pdf",
            )

    def test_status_must_not_contradict_reference_intervals(self):
        with pytest.raises(ValidationError, match="contradicts reference interval"):
            LabResult(
                lab_id="LAB-TEST-004",
                analyte="alt",
                display_name="ALT",
                panel="chemistry",
                value=150.0,  # 150 > 130, expected HIGH
                unit="U/L",
                reference_low=12.0,
                reference_high=130.0,
                status=LabStatus.NORMAL,  # Contradiction!
                collected_at=date(2026, 9, 22),
                source_document="test.pdf",
            )

    def test_valid_ambiguous_result_succeeds(self):
        res = LabResult(
            lab_id="LAB-TEST-005",
            analyte="platelets",
            display_name="Platelet Count",
            panel="cbc",
            value=None,
            unit="K/µL",
            reference_low=151.0,
            reference_high=600.0,
            status=LabStatus.AMBIGUOUS,
            flag=AMBIGUOUS_FLAG,
            raw_text="2?8 (smudged)",
            collected_at=date(2026, 9, 22),
            source_document="RefLab.pdf",
        )
        assert res.value is None
        assert res.status == LabStatus.AMBIGUOUS
        assert res.flag == AMBIGUOUS_FLAG


class TestStrictExtractionProtocol:
    """Verifies OCR ambiguity parsing and sanitization."""

    @pytest.mark.parametrize(
        "raw_text",
        [
            "2?8",
            "~140",
            "unreadable",
            "smudged print",
            "blurred",
            "<0.5",  # censored interval is ambiguous for exact quantitative staging
            "1,234.5",  # ambiguous separator
            "12.4 15.6",  # multiple numbers
            "",
            None,
        ],
    )
    def test_parse_raw_numeric_rejects_ambiguity(self, raw_text: str | None):
        assert v.parse_raw_numeric(raw_text) is None

    @pytest.mark.parametrize(
        "raw_text,expected",
        [
            ("1.4", 1.4),
            ("2.3 mg/dL", 2.3),
            ("148 H", 148.0),
            ("1.022", 1.022),
            ("310", 310.0),
        ],
    )
    def test_parse_raw_numeric_parses_clean_numbers(self, raw_text: str, expected: float):
        parsed = v.parse_raw_numeric(raw_text)
        assert parsed is not None
        assert abs(parsed - expected) < 1e-6

    def test_sanitize_extracted_labs_nulls_ambiguous_platelet_count(self):
        extraction = LLMLabExtraction(
            items=[
                LLMExtractedLabItem(
                    analyte="creatinine",
                    display_name="Creatinine",
                    raw_text="2.3",
                    value=2.3,
                    unit="mg/dL",
                    reference_low=0.6,
                    reference_high=2.4,
                    legible=True,
                    confidence=0.99,
                ),
                LLMExtractedLabItem(
                    analyte="platelets",
                    display_name="Platelet Count",
                    raw_text="2?8",  # smudged
                    value=248.0,  # Model attempted to guess/extrapolate the digit!
                    unit="K/µL",
                    reference_low=151.0,
                    reference_high=600.0,
                    legible=False,
                    confidence=0.60,
                ),
            ],
            document_notes="Platelet field was smudged on paper",
        )

        labs, violations = v.sanitize_extracted_labs(
            extraction,
            collected_at=date(2026, 9, 22),
            source_document="Report.pdf",
        )

        assert len(labs) == 2
        crea, plt = labs[0], labs[1]

        # Creatinine clean
        assert crea.analyte == "creatinine"
        assert crea.value == 2.3
        assert crea.status == LabStatus.NORMAL

        # Platelet count: guardrails must null the guessed value and set exact flag
        assert plt.analyte == "platelets"
        assert plt.value is None
        assert plt.status == LabStatus.AMBIGUOUS
        assert plt.flag == AMBIGUOUS_FLAG

        # Violations tracked
        assert any(vio.code in ("AMBIGUOUS_VALUE_FILLED", "LOW_CONFIDENCE_EXTRACTION") for vio in violations)


class TestZeroHallucinationVerification:
    """Verifies that model narrative and cited values cannot fabricate data."""

    def test_redact_unverified_numbers(self, tmp_path):
        store = PatientStore.from_json()
        record = store.get("MICH-001")
        allowed = v.allowed_numbers(record)

        # 2.3 mg/dL is in SSOT. 4.5 mg/dL is fabricated.
        text = "Creatinine was 2.3 mg/dL previously, but now jumped to 4.5 mg/dL."
        redacted, violations = v.redact_unverified_numbers(text, allowed, "test_field")

        assert "2.3 mg/dL" in redacted
        assert "4.5" not in redacted
        assert v.REDACTION_TOKEN in redacted
        assert len(violations) >= 1
        assert violations[0].code == "UNVERIFIED_NUMBER_IN_TEXT"

    def test_verify_cited_values_enforces_ssot(self):
        store = PatientStore.from_json()
        record = store.get("MICH-001")

        cited = [
            # Legitimate match in SSOT
            CitedLabValue(analyte="creatinine", collected_at="2026-09-22", value=2.3, unit="mg/dL"),
            # Fabricated measurement date
            CitedLabValue(analyte="creatinine", collected_at="2026-10-05", value=3.1, unit="mg/dL"),
            # Model tried to state a number for the ambiguous platelet lab
            CitedLabValue(analyte="platelets", collected_at="2026-09-22", value=248.0, unit="K/µL"),
        ]

        verified, violations = v.verify_cited_values(cited, record)

        # 1. Creatinine 2026-09-22 verified
        v_crea = next(c for c in verified if c.analyte == "creatinine" and c.collected_at == "2026-09-22")
        assert v_crea.value == 2.3

        # 2. Fabricated creatinine discarded from verified citations
        assert not any(c.collected_at == "2026-10-05" for c in verified)

        # 3. Ambiguous platelet metric forced to null
        v_plt = next(c for c in verified if c.analyte == "platelets" and c.collected_at == "2026-09-22")
        assert v_plt.value is None

        assert any(v.code == "FABRICATED_VALUE" for v in violations)
        assert any(v.code == "AMBIGUOUS_VALUE_FILLED" for v in violations)


class TestFelineDomainNuance:
    """Verifies longitudinal trend calculation, IRIS CKD staging, and Triaditis detection."""

    def test_michelin_synthetic_case_longitudinal_ckd_iris_stage_2(self):
        store = PatientStore.from_json()
        record = store.get("MICH-001")

        iris = v.assess_iris_ckd(record)

        # Creatinine 2.3 is <= 2.4 (nominal reference upper limit), BUT:
        # rising from 1.4 -> 1.9 -> 2.3 mg/dL and inadequate USG 1.022 (< 1.035)
        # proves IRIS Stage 2 CKD!
        assert iris.stage == 2
        assert "IRIS Stage 2 CKD" in iris.stage_label
        assert iris.latest_creatinine == 2.3
        assert iris.usg_adequate is False
        assert iris.latest_usg == 1.022

        # Verify longitudinal rise is captured in evidence
        assert any("1.4 -> 1.9 -> 2.3" in ev or "Creatinine rising" in ev for ev in iris.evidence)
        assert any("reference-interval 'normal' does NOT exclude feline CKD" in ev for ev in iris.evidence)

    def test_michelin_synthetic_case_triaditis_suspected(self):
        store = PatientStore.from_json()
        record = store.get("MICH-001")

        triaditis = v.assess_triaditis(record)

        assert triaditis.classification == "triaditis_suspected"
        assert triaditis.latest_fpli == 6.8
        assert len(triaditis.pancreatic_evidence) > 0
        assert len(triaditis.gastrointestinal_evidence) > 0
        assert len(triaditis.hepatobiliary_evidence) > 0

        # Diagnostics include abdominal US, cobalamin/folate
        assert any("ultrasound" in d.lower() for d in triaditis.recommended_diagnostics)
        assert any("cobalamin" in d.lower() for d in triaditis.recommended_diagnostics)

    def test_longitudinal_trends_exclude_ambiguous_and_no_extrapolation(self):
        store = PatientStore.from_json()
        record = store.get("MICH-001")

        # Platelets has an ambiguous point on 2026-09-22
        plt_trend = v.compute_lab_trend(record, "platelets")
        assert plt_trend.ambiguous_points == 1
        assert plt_trend.points[-1].value is None
        assert plt_trend.points[-1].status == LabStatus.AMBIGUOUS
        # Trend was computed from the remaining 2 verified points only
        assert plt_trend.first_value == 310.0
        assert plt_trend.last_value == 285.0
        assert "AMBIGUOUS point(s) excluded" in plt_trend.note
