"""Strict Pydantic v2 schemas for the Michelin / Haiiro clinical ecosystem.

Three families of models live here:

1. **SSOT models** (``PatientRecord`` and children) - the Central Patient Store.
   They are strict (``extra="forbid"``) and self-validating: a lab result can
   never carry a numeric value while flagged AMBIGUOUS, can never be missing a
   value without being flagged AMBIGUOUS, and its status must agree with its
   own reference interval.
2. **API contracts** - request/response bodies for the FastAPI layer.
3. **LLM structured-output contracts** (``LLM*`` models) - the JSON schemas we
   hand to Gemini via ``response_schema``. They are intentionally simple
   (no dicts, no ``additionalProperties``) so the Gemini schema translator
   accepts them, and they are *never* trusted directly: every LLM output is
   passed through ``services.verification`` before reaching the SSOT or a user.
"""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Guardrail constants
# --------------------------------------------------------------------------- #

AMBIGUOUS_FLAG = "Data unreadable / ambiguous — manual clinical verification required"
"""Exact flag string mandated by the OCR / extraction protocol."""

PATIENT_ID_PATTERN = r"^[A-Z]{3,5}-\d{3,6}$"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base for SSOT and API models: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class LabStatus(str, Enum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    LOW = "LOW"
    NO_REFERENCE = "NO_REFERENCE"  # numeric value present but no reference interval supplied
    AMBIGUOUS = "AMBIGUOUS"


class TriageLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    @property
    def severity(self) -> int:
        return {"GREEN": 0, "YELLOW": 1, "RED": 2}[self.value]

    @classmethod
    def max(cls, *levels: "TriageLevel | None") -> "TriageLevel":
        present = [lvl for lvl in levels if lvl is not None]
        if not present:
            return cls.GREEN
        return max(present, key=lambda lvl: lvl.severity)


class Sex(str, Enum):
    MALE_NEUTERED = "MN"
    MALE_INTACT = "MI"
    FEMALE_SPAYED = "FS"
    FEMALE_INTACT = "FI"


Actor = Literal["michelin", "haiiro", "clinic", "system"]
MediaStatus = Literal["READABLE", "AMBIGUOUS"]


def derive_status(value: float | None, low: float | None, high: float | None) -> LabStatus:
    """Deterministically derive a lab status from a value and its reference interval."""
    if value is None:
        return LabStatus.AMBIGUOUS
    if low is None and high is None:
        return LabStatus.NO_REFERENCE
    if low is not None and value < low:
        return LabStatus.LOW
    if high is not None and value > high:
        return LabStatus.HIGH
    return LabStatus.NORMAL


# --------------------------------------------------------------------------- #
# SSOT building blocks
# --------------------------------------------------------------------------- #


class LabResult(StrictModel):
    """A single measured analyte. Immutable once created."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    lab_id: str = Field(min_length=3)
    analyte: str = Field(min_length=2, description="Canonical snake_case key, e.g. 'creatinine'")
    display_name: str
    panel: Literal["chemistry", "cbc", "urinalysis", "endocrine", "special", "vitals"]
    value: float | None
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    status: LabStatus
    flag: str | None = None
    raw_text: str | None = Field(
        default=None, description="Verbatim transcription of the value as printed on the source document"
    )
    collected_at: date
    source_document: str
    laboratory: str | None = None

    @model_validator(mode="after")
    def _enforce_guardrails(self) -> "LabResult":
        if (
            self.reference_low is not None
            and self.reference_high is not None
            and self.reference_low > self.reference_high
        ):
            raise ValueError(f"{self.analyte}: reference_low > reference_high")

        if self.status is LabStatus.AMBIGUOUS:
            if self.value is not None:
                raise ValueError(
                    f"{self.analyte}: AMBIGUOUS results must have value=null (zero-hallucination guardrail)"
                )
            if self.flag != AMBIGUOUS_FLAG:
                raise ValueError(f"{self.analyte}: AMBIGUOUS results must carry flag '{AMBIGUOUS_FLAG}'")
            return self

        if self.value is None:
            raise ValueError(
                f"{self.analyte}: missing value must be recorded as status=AMBIGUOUS, never silently null"
            )
        if self.flag == AMBIGUOUS_FLAG:
            raise ValueError(f"{self.analyte}: AMBIGUOUS flag used on a non-AMBIGUOUS result")
        expected = derive_status(self.value, self.reference_low, self.reference_high)
        if expected is not self.status:
            raise ValueError(
                f"{self.analyte}: status {self.status.value} contradicts reference interval "
                f"(expected {expected.value} for value {self.value})"
            )
        return self


class WeightRecord(StrictModel):
    recorded_at: date
    weight_kg: float = Field(gt=0, lt=20)
    body_condition_score: int | None = Field(default=None, ge=1, le=9)
    source: Actor = "clinic"


class ClinicalEncounter(StrictModel):
    encounter_id: str
    encounter_date: date
    clinician: str
    reason: str
    findings: str
    assessment: str
    plan: str


class Medication(StrictModel):
    name: str
    dose: str
    frequency: str
    started: date
    ended: date | None = None


class Caregiver(StrictModel):
    name: str
    contact_channel: str


class MediaAttachment(StrictModel):
    """Photo/video attached to a home observation.

    ``data_base64`` is accepted on input but stripped before persistence; the
    SSOT keeps only the URI plus the verified assessment.
    """

    media_id: str | None = None
    media_type: Literal["photo", "video"]
    mime_type: str = Field(pattern=r"^(image|video)/[\w.+-]+$")
    uri: str | None = None
    data_base64: str | None = None
    caregiver_caption: str | None = None
    assessment_status: MediaStatus | None = None
    assessment_finding: str | None = None
    flag: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "MediaAttachment":
        if not self.uri and not self.data_base64:
            raise ValueError("media requires either 'uri' or 'data_base64'")
        if self.assessment_status == "AMBIGUOUS":
            if self.assessment_finding is not None:
                raise ValueError("AMBIGUOUS media must have assessment_finding=null")
            if self.flag != AMBIGUOUS_FLAG:
                raise ValueError(f"AMBIGUOUS media must carry flag '{AMBIGUOUS_FLAG}'")
        return self


class HomeObservationInput(StrictModel):
    """Caregiver-reported observation as received by Haiiro."""

    observed_at: datetime = Field(default_factory=utcnow)
    reported_by: str = "caregiver"
    symptoms: list[str] = Field(default_factory=list, max_length=30)
    free_text: str = Field(default="", max_length=4000)
    appetite: Literal["normal", "reduced", "none", "unknown"] = "unknown"
    water_intake: Literal["normal", "increased", "decreased", "unknown"] = "unknown"
    vomiting_episodes_24h: int | None = Field(default=None, ge=0, le=50)
    hours_since_last_meal: float | None = Field(default=None, ge=0, le=720)
    hours_since_last_urination: float | None = Field(default=None, ge=0, le=720)
    straining_to_urinate: bool = False
    breathing_difficulty: bool = False
    lethargy: bool = False
    media: list[MediaAttachment] = Field(default_factory=list, max_length=10)

    @field_validator("symptoms")
    @classmethod
    def _normalise_symptoms(cls, value: list[str]) -> list[str]:
        return [s.strip().lower() for s in value if s and s.strip()]


class HomeObservation(HomeObservationInput):
    observation_id: str
    channel: Literal["haiiro"] = "haiiro"


class TriageEvent(StrictModel):
    event_id: str
    created_at: datetime
    observation_id: str
    level: TriageLevel
    escalated_to_vet: bool
    escalation_reason: str | None = None
    rule_hits: list[str] = Field(default_factory=list)
    llm_level: TriageLevel | None = None
    llm_used: bool
    model: str | None = None


class Escalation(StrictModel):
    escalation_id: str
    created_at: datetime
    patient_id: str
    level: TriageLevel
    recipient: str
    reason: str
    observation_id: str
    status: Literal["OPEN", "ACKNOWLEDGED", "RESOLVED"] = "OPEN"


class PatientRecord(StrictModel):
    """Central Patient Store record - the single source of truth (SSOT)."""

    patient_id: str = Field(pattern=PATIENT_ID_PATTERN)
    name: str
    species: Literal["feline"]
    breed: str
    sex: Sex
    date_of_birth: date | None = None
    age_years: float = Field(ge=0, le=35)
    caregiver: Caregiver
    attending_veterinarian: str
    active_problems: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    weights: list[WeightRecord] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    encounters: list[ClinicalEncounter] = Field(default_factory=list)
    home_observations: list[HomeObservation] = Field(default_factory=list)
    triage_events: list[TriageEvent] = Field(default_factory=list)
    escalations: list[Escalation] = Field(default_factory=list)
    record_version: int = Field(ge=1)
    updated_at: datetime
    last_updated_by: Actor

    @model_validator(mode="after")
    def _unique_ids(self) -> "PatientRecord":
        for label, ids in (
            ("lab_id", [lab.lab_id for lab in self.labs]),
            ("observation_id", [o.observation_id for o in self.home_observations]),
            ("encounter_id", [e.encounter_id for e in self.encounters]),
            ("event_id", [t.event_id for t in self.triage_events]),
            ("escalation_id", [e.escalation_id for e in self.escalations]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} in patient record")
        return self


# --------------------------------------------------------------------------- #
# Deterministic analysis outputs
# --------------------------------------------------------------------------- #


class TrendPoint(StrictModel):
    collected_at: date
    value: float | None
    status: LabStatus


class TrendSummary(StrictModel):
    analyte: str
    unit: str | None
    points: list[TrendPoint]
    first_value: float | None
    last_value: float | None
    absolute_change: float | None
    percent_change: float | None
    direction: Literal["rising", "falling", "stable", "insufficient_data"]
    ambiguous_points: int
    note: str = "Computed only from verified measured values; no interpolation or extrapolation."


class IrisStaging(StrictModel):
    stage: int | None = Field(default=None, ge=1, le=4)
    stage_label: str
    creatinine_stage: int | None = None
    sdma_stage: int | None = None
    latest_creatinine: float | None = None
    latest_sdma: float | None = None
    latest_usg: float | None = None
    usg_adequate: bool | None = None
    creatinine_trend: TrendSummary | None = None
    proteinuria_substage: str | None = None
    hypertension_substage: str | None = None
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


class TriaditisAssessment(StrictModel):
    classification: Literal[
        "triaditis_suspected", "pancreatitis_with_gi_signs", "pancreatitis_not_excluded", "not_supported"
    ]
    latest_fpli: float | None = None
    pancreatic_evidence: list[str] = Field(default_factory=list)
    gastrointestinal_evidence: list[str] = Field(default_factory=list)
    hepatobiliary_evidence: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    recommended_diagnostics: list[str] = Field(default_factory=list)


class GuardrailViolation(StrictModel):
    code: Literal[
        "FABRICATED_VALUE",
        "VALUE_MISMATCH",
        "AMBIGUOUS_VALUE_FILLED",
        "UNVERIFIED_NUMBER_IN_TEXT",
        "TRIAGE_DOWNGRADE_BLOCKED",
        "MEDIA_AMBIGUITY_ENFORCED",
        "LOW_CONFIDENCE_EXTRACTION",
    ]
    field: str
    detail: str


class CitedLabValue(BaseModel):
    """A lab value cited by the LLM. Also used (after verification) in responses."""

    analyte: str
    collected_at: str = Field(description="ISO date (YYYY-MM-DD) of the cited measurement")
    value: float | None
    unit: str


class Differential(BaseModel):
    condition: str
    likelihood: Literal["high", "moderate", "low"]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    next_steps: list[str]


class MediaAssessment(BaseModel):
    media_id: str
    status: MediaStatus
    finding: str | None
    flag: str | None


# --------------------------------------------------------------------------- #
# LLM structured-output contracts (Gemini response_schema)
# --------------------------------------------------------------------------- #


class LLMTriageAssessment(BaseModel):
    triage_level: TriageLevel
    rationale: str
    red_flags: list[str]
    recommend_vet_escalation: bool
    caregiver_message: str
    recommended_actions: list[str]
    media_assessments: list[MediaAssessment]


class LLMCrossExamination(BaseModel):
    clinical_summary: str
    challenges_to_hypothesis: list[str]
    differentials: list[Differential]
    cited_lab_values: list[CitedLabValue]
    recommended_diagnostics: list[str]
    data_gaps: list[str]


class LLMExtractedLabItem(BaseModel):
    analyte: str
    display_name: str
    raw_text: str | None = Field(description="Exact characters printed for the value, verbatim; null if unreadable")
    value: float | None
    unit: str | None
    reference_low: float | None
    reference_high: float | None
    legible: bool
    confidence: float = Field(ge=0.0, le=1.0)


class LLMLabExtraction(BaseModel):
    items: list[LLMExtractedLabItem]
    document_notes: str | None


# --------------------------------------------------------------------------- #
# API contracts
# --------------------------------------------------------------------------- #


class HaiiroTriageRequest(StrictModel):
    patient_id: str = Field(pattern=PATIENT_ID_PATTERN)
    observation: HomeObservationInput


class HaiiroTriageResponse(StrictModel):
    patient_id: str
    observation_id: str
    triage_level: TriageLevel
    escalate_to_vet: bool
    escalation: Escalation | None
    caregiver_message: str
    recommended_actions: list[str]
    red_flags: list[str]
    media_assessments: list[MediaAssessment]
    llm_used: bool
    model: str | None
    llm_triage_level: TriageLevel | None
    guardrail_violations: list[GuardrailViolation]
    guardrail_notes: list[str]
    record_version: int


class MichelinCrossExamRequest(StrictModel):
    patient_id: str = Field(pattern=PATIENT_ID_PATTERN)
    clinician: str = "Dr. Andy"
    working_hypothesis: str = Field(min_length=3, max_length=2000)
    clinical_question: str | None = Field(default=None, max_length=2000)


class MichelinCrossExamResponse(StrictModel):
    patient_id: str
    clinician: str
    working_hypothesis: str
    generated_at: datetime
    record_version: int
    iris_staging: IrisStaging
    triaditis: TriaditisAssessment
    lab_trends: list[TrendSummary]
    weight_trend: TrendSummary
    ambiguous_data: list[LabResult]
    open_escalations: list[Escalation]
    recent_home_observations: list[HomeObservation]
    clinical_summary: str | None
    challenges_to_hypothesis: list[str]
    differentials: list[Differential]
    verified_citations: list[CitedLabValue]
    recommended_diagnostics: list[str]
    data_gaps: list[str]
    guardrail_violations: list[GuardrailViolation]
    llm_used: bool
    model: str | None
    guardrail_notes: list[str]


class LabExtractionRequest(StrictModel):
    document_text: str | None = Field(default=None, max_length=50_000)
    image_base64: str | None = None
    mime_type: str | None = Field(default=None, pattern=r"^(image/[\w.+-]+|application/pdf)$")
    collected_at: date
    source_document: str
    laboratory: str | None = None
    commit: bool = False
    expected_record_version: int | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "LabExtractionRequest":
        if not self.document_text and not self.image_base64:
            raise ValueError("provide 'document_text' or 'image_base64'")
        if self.image_base64 and not self.mime_type:
            raise ValueError("'mime_type' is required with 'image_base64'")
        return self


class LabExtractionResponse(StrictModel):
    patient_id: str
    labs: list[LabResult]
    ambiguous_count: int
    guardrail_violations: list[GuardrailViolation]
    llm_used: bool
    model: str | None
    committed: bool
    record_version: int
