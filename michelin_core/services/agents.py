"""Agent orchestration for the Michelin ecosystem.

Every agent follows the same safety pipeline::

    SSOT read -> deterministic analysis -> Gemini (optional) -> verification -> SSOT write / response

Gemini adds clinical reasoning and empathetic language; deterministic code owns
every number, every red flag and every AMBIGUOUS decision. If Gemini is not
configured or fails, agents return their deterministic result with
``llm_used=False`` rather than failing or guessing.
"""

import base64
import binascii
import json
import logging
import uuid
from typing import Any

from michelin_core.schemas.patient import (
    AMBIGUOUS_FLAG,
    Differential,
    Escalation,
    GuardrailViolation,
    HaiiroTriageRequest,
    HaiiroTriageResponse,
    HomeObservation,
    IrisStaging,
    LabExtractionRequest,
    LabExtractionResponse,
    LabStatus,
    LLMCrossExamination,
    LLMLabExtraction,
    LLMTriageAssessment,
    MediaAssessment,
    MichelinCrossExamRequest,
    MichelinCrossExamResponse,
    PatientRecord,
    Sex,
    TriageEvent,
    TriageLevel,
    TriaditisAssessment,
    utcnow,
)
from michelin_core.services import verification as v
from michelin_core.services.gemini_client import (
    CROSS_EXAM_MODEL,
    EXTRACTION_MODEL,
    TRIAGE_MODEL,
    GeminiClient,
    LLMResponseError,
    LLMUnavailableError,
    bytes_part,
    text_part,
)
from michelin_core.services.patient_store import PatientStore

logger = logging.getLogger("michelin.agents")

ESCALATION_RECIPIENT = "Dr. Andy"


def _short_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def _record_for_prompt(record: PatientRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json", exclude={"triage_events"})
    for obs in data["home_observations"]:
        for m in obs["media"]:
            m.pop("data_base64", None)
    return data


# =========================================================================== #
# Haiiro - caregiver triage companion (gemini-2.5-flash)
# =========================================================================== #

HAIIRO_SYSTEM_PROMPT = f"""You are Haiiro, an empathetic triage companion for people caring for cats at home.
You work under the supervision of the attending veterinarian, {ESCALATION_RECIPIENT}.

Classify the caregiver's observation:
- RED: emergency - needs veterinary care immediately.
- YELLOW: needs same-day contact with the veterinary team.
- GREEN: safe to monitor at home.

Feline red flags (always RED): breathing difficulty or open-mouth breathing; a male cat straining or unable
to urinate; no urine for 24 h or more; no food for 48 h or more (hepatic lipidosis risk); 3 or more vomiting
episodes in 24 h; seizures; collapse; lily, antifreeze or other toxin exposure; sudden hind-limb weakness.
When torn between two levels, choose the more urgent one.

Non-negotiable rules:
1. Never diagnose, never recommend or dose medication, never state or guess numeric lab values.
2. Media: return exactly one media_assessment per attached photo/video using its exact media_id. If an image
   or video is blurred, dark, cropped, or otherwise unclear, set status "AMBIGUOUS", finding null and
   flag "{AMBIGUOUS_FLAG}". Never guess what an unclear image shows. For clear media, describe only what is
   visibly present, with flag null.
3. caregiver_message: warm, calm, plain language (2-4 sentences), no medical jargon, never minimise risk.
4. recommend_vet_escalation must be true for RED, and for YELLOW when the patient has relevant chronic problems.
"""

CAREGIVER_TEMPLATES = {
    TriageLevel.RED: (
        "Thank you for telling me about {name} - what you're describing needs urgent veterinary attention. "
        "I've alerted {vet}'s team right now. Please contact the clinic or the nearest emergency veterinary "
        "hospital immediately; please don't wait to see if it improves."
    ),
    TriageLevel.YELLOW: (
        "Thank you for keeping such a close eye on {name}. These changes are worth a same-day check-in, "
        "especially with {pronoun} history, {escalation_clause} Keep fresh water available, note any further "
        "vomiting, eating or litter-box changes, and call the clinic straight away if anything gets worse."
    ),
    TriageLevel.GREEN: (
        "Thanks for the update on {name}. Nothing you've described needs urgent attention right now. "
        "Keep an eye on {pronoun} appetite, water intake, litter-box habits and energy, and tell me about any change."
    ),
}

RECOMMENDED_ACTIONS = {
    TriageLevel.RED: [
        "Contact the clinic or an emergency veterinary hospital now",
        "Keep your cat calm, warm and confined in a carrier",
        "Do not give food, water by force, or any human medication",
    ],
    TriageLevel.YELLOW: [
        "Call the clinic today to arrange a check",
        "Record each vomiting episode, food eaten and litter-box use",
        "Keep fresh water available; do not give any medication unless prescribed",
        "Escalate immediately if your cat stops eating entirely, vomits 3+ times, or strains to urinate",
    ],
    TriageLevel.GREEN: [
        "Continue normal routine and monitoring",
        "Log appetite, water intake and litter-box habits daily in Haiiro",
    ],
}


class HaiiroTriageAgent:
    def __init__(self, llm: GeminiClient, store: PatientStore, model: str = TRIAGE_MODEL):
        self.llm, self.store, self.model = llm, store, model

    def _build_contents(self, record: PatientRecord, obs: HomeObservation, comorbidities: list[str]) -> list[Any]:
        context = {
            "patient": {
                "name": record.name, "species": record.species, "breed": record.breed,
                "sex": record.sex.value, "age_years": record.age_years,
                "active_problems": record.active_problems,
                "current_medications": [m.name for m in record.medications if m.ended is None],
                "clinical_context_flags": comorbidities,
            },
            "recent_observations": [
                o.model_dump(mode="json", exclude={"media"}) for o in record.home_observations[-3:]
            ],
            "new_observation": obs.model_dump(mode="json", exclude={"media"}),
        }
        parts: list[Any] = [text_part("CONTEXT:\n" + json.dumps(context, indent=2, default=str))]
        for m in obs.media:
            header = f"ATTACHMENT media_id={m.media_id} type={m.media_type} caption={m.caregiver_caption!r}"
            if m.data_base64:
                parts.append(text_part(header))
                parts.append(bytes_part(base64.b64decode(m.data_base64), m.mime_type))
            else:
                parts.append(text_part(header + " - NOT available for visual inspection (reference only)."))
        return parts

    def triage(self, request: HaiiroTriageRequest) -> HaiiroTriageResponse:
        record = self.store.get(request.patient_id)

        media = []
        for m in request.observation.media:
            if m.data_base64:
                try:
                    base64.b64decode(m.data_base64, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"media {m.media_id or m.media_type}: invalid base64 payload") from exc
            media.append(m.model_copy(update={"media_id": m.media_id or _short_id("MED")}))
        obs = HomeObservation(
            **request.observation.model_dump(exclude={"media"}),
            media=media,
            observation_id=_short_id("OBS"),
        )

        rules = v.evaluate_triage_rules(obs, record)
        notes: list[str] = []
        violations: list[GuardrailViolation] = []
        llm_result: LLMTriageAssessment | None = None
        llm_used = False

        if self.llm.available:
            try:
                llm_result = self.llm.generate_structured(
                    model=self.model, system_instruction=HAIIRO_SYSTEM_PROMPT,
                    contents=self._build_contents(record, obs, rules.comorbidities), schema=LLMTriageAssessment,
                )
                llm_used = True
            except LLMResponseError as exc:
                notes.append(f"LLM unavailable ({exc}); deterministic triage rules applied.")
        else:
            notes.append("Gemini not configured; deterministic triage rules applied.")

        llm_level = llm_result.triage_level if llm_result else None
        final_level, merge_violations = v.merge_triage(rules, llm_level)
        violations.extend(merge_violations)

        media_assessments, media_violations = v.enforce_media_guardrails(
            llm_result.media_assessments if llm_result else [], obs.media
        )
        violations.extend(media_violations)
        by_id: dict[str, MediaAssessment] = {a.media_id: a for a in media_assessments}
        obs = obs.model_copy(update={"media": [
            m.model_copy(update={
                "assessment_status": by_id[m.media_id].status,
                "assessment_finding": by_id[m.media_id].finding,
                "flag": by_id[m.media_id].flag,
            }) for m in obs.media
        ]})
        if any(a.status == "AMBIGUOUS" for a in media_assessments):
            notes.append("One or more attachments are AMBIGUOUS; flagged for manual clinical review.")

        llm_escalate = bool(llm_result and llm_result.recommend_vet_escalation)
        escalate = (
            final_level is TriageLevel.RED
            or (final_level is TriageLevel.YELLOW and bool(rules.comorbidities))
            or llm_escalate
        )
        reason_parts = [h for h in rules.hits] or ([llm_result.rationale] if llm_result else [])
        if final_level is TriageLevel.YELLOW and rules.comorbidities:
            reason_parts.append("patient has active chronic problems: " + "; ".join(rules.comorbidities[:4]))
        reason = "; ".join(reason_parts) or "Escalation recommended by triage model"

        # Caregiver language: use the LLM message only if it was produced for the final level.
        pronoun = "his" if record.sex in (Sex.MALE_NEUTERED, Sex.MALE_INTACT) else "her"
        if llm_result and llm_level is final_level and llm_result.caregiver_message.strip():
            message = llm_result.caregiver_message.strip()
            actions = llm_result.recommended_actions or RECOMMENDED_ACTIONS[final_level]
        else:
            if llm_result:
                notes.append("LLM caregiver message replaced: it was written for a lower triage level.")
            message = CAREGIVER_TEMPLATES[final_level].format(
                name=record.name, vet=ESCALATION_RECIPIENT, pronoun=pronoun,
                escalation_clause=(
                    f"so I've shared your update with {ESCALATION_RECIPIENT}." if escalate
                    else "so please call the clinic today."
                ),
            )
            actions = RECOMMENDED_ACTIONS[final_level]

        now = utcnow()
        escalation = (
            Escalation(
                escalation_id=_short_id("ESC"), created_at=now, patient_id=record.patient_id,
                level=final_level, recipient=ESCALATION_RECIPIENT, reason=reason,
                observation_id=obs.observation_id,
            ) if escalate else None
        )
        event = TriageEvent(
            event_id=_short_id("TRI"), created_at=now, observation_id=obs.observation_id, level=final_level,
            escalated_to_vet=escalate, escalation_reason=reason if escalate else None, rule_hits=rules.hits,
            llm_level=llm_level, llm_used=llm_used, model=self.model if llm_used else None,
        )
        updated = self.store.commit_haiiro_observation(record.patient_id, obs, event, escalation)

        red_flags = list(rules.hits)
        if llm_result:
            red_flags += [f"MODEL: {f}" for f in llm_result.red_flags if f]
        return HaiiroTriageResponse(
            patient_id=record.patient_id, observation_id=obs.observation_id, triage_level=final_level,
            escalate_to_vet=escalate, escalation=escalation, caregiver_message=message,
            recommended_actions=actions, red_flags=red_flags, media_assessments=media_assessments,
            llm_used=llm_used, model=self.model if llm_used else None, llm_triage_level=llm_level,
            guardrail_violations=violations, guardrail_notes=notes, record_version=updated.record_version,
        )


# =========================================================================== #
# Michelin - veterinary co-pilot (gemini-2.5-pro)
# =========================================================================== #

MICHELIN_SYSTEM_PROMPT = f"""You are Michelin, a veterinary clinical co-pilot for {ESCALATION_RECIPIENT}, specialised in
feline internal medicine. Your role is proactive clinical cross-examination: challenge the working hypothesis,
audit the differential list, and analyse longitudinal laboratory trends.

Non-negotiable rules:
1. ZERO HALLUCINATION. Use only data present in PATIENT_RECORD and DETERMINISTIC_ANALYSIS. Never interpolate,
   estimate, round, extrapolate or invent any numeric value, date or finding. Missing data must be named in
   data_gaps, never filled in.
2. AMBIGUOUS DATA. A lab with status "AMBIGUOUS" has value null ("{AMBIGUOUS_FLAG}"). Never propose a value for
   it. Cite it with value null and recommend repeat measurement / manual verification.
3. CITATIONS. Every lab value you mention in any text field must also appear in cited_lab_values using the
   record's analyte key, collected_at (YYYY-MM-DD) and the exact value from the record.
4. FELINE NUANCE. Prioritise longitudinal trends over single reference-interval flags. A creatinine inside the
   laboratory reference interval can still lie in IRIS Stage 2 (1.6-2.8 mg/dL); with inadequate USG (< 1.035)
   and/or rising creatinine or SDMA this supports CKD. Consider hyperthyroidism masking azotemia, pre-renal
   factors, systemic hypertension and proteinuria. Consider feline triaditis (pancreatitis + cholangitis +
   inflammatory enteropathy) when fPLI is elevated alongside GI signs and hepatobiliary changes.
5. The DETERMINISTIC_ANALYSIS was produced by validated code. Do not contradict its numbers or IRIS stage.
6. Be direct and evidence-based. Explicitly state which findings contradict or are unexplained by the working
   hypothesis.
"""


def _deterministic_differentials(iris: IrisStaging, tri: TriaditisAssessment) -> list[Differential]:
    diffs: list[Differential] = []
    if iris.stage is not None:
        diffs.append(Differential(
            condition=iris.stage_label, likelihood="high", supporting_evidence=iris.evidence,
            contradicting_evidence=[], next_steps=[
                "Repeat creatinine, SDMA and USG in 2-4 weeks once hydrated",
                "Confirm blood pressure with repeated measurements", "Measure total T4",
            ],
        ))
    if tri.classification != "not_supported":
        diffs.append(Differential(
            condition={
                "triaditis_suspected": "Feline triaditis (pancreatitis + cholangitis + enteropathy)",
                "pancreatitis_with_gi_signs": "Chronic pancreatitis with GI signs",
                "pancreatitis_not_excluded": "Pancreatitis (not excluded)",
            }[tri.classification],
            likelihood="high" if tri.classification == "triaditis_suspected" else "moderate",
            supporting_evidence=tri.pancreatic_evidence + tri.hepatobiliary_evidence + tri.gastrointestinal_evidence,
            contradicting_evidence=["Definitive diagnosis requires imaging and histopathology"],
            next_steps=tri.recommended_diagnostics,
        ))
    if any("Total T4" in m for m in iris.missing_data):
        diffs.append(Differential(
            condition="Hyperthyroidism (not excluded)", likelihood="low",
            supporting_evidence=["Senior cat with weight loss and vomiting"],
            contradicting_evidence=[], next_steps=["Measure total T4 before finalising renal staging"],
        ))
    return diffs


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x.lower() in seen or seen.add(x.lower()))]


class MichelinCoPilotAgent:
    def __init__(self, llm: GeminiClient, store: PatientStore, model: str = CROSS_EXAM_MODEL):
        self.llm, self.store, self.model = llm, store, model

    def cross_examine(self, request: MichelinCrossExamRequest) -> MichelinCrossExamResponse:
        record = self.store.get(request.patient_id)
        iris = v.assess_iris_ckd(record)
        tri = v.assess_triaditis(record)
        trends = [v.compute_lab_trend(record, a) for a in v.KEY_TREND_ANALYTES if v.labs_for(record, a)]
        weight_trend = v.compute_weight_trend(record)
        ambiguous = v.ambiguous_labs(record)
        open_escalations = [e for e in record.escalations if e.status == "OPEN"]

        det_diagnostics = list(tri.recommended_diagnostics) + [
            f"Repeat {lab.display_name} (value from {lab.collected_at} is AMBIGUOUS - manual verification required)"
            for lab in ambiguous
        ]
        det_gaps = iris.missing_data + tri.missing_data + [
            f"{lab.display_name} ({lab.collected_at}): {AMBIGUOUS_FLAG}" for lab in ambiguous
        ]
        det_challenges = []
        if iris.stage is not None and iris.evidence:
            det_challenges.append(f"{iris.stage_label}: {iris.evidence[0]}")
        if tri.classification != "not_supported":
            det_challenges.append(
                f"Pancreatic/hepatobiliary pattern ({tri.classification}) is not explained by the working "
                f"hypothesis '{request.working_hypothesis}'."
            )
        det_challenges += [
            f"{lab.display_name} on {lab.collected_at} is AMBIGUOUS and cannot support or refute any hypothesis."
            for lab in ambiguous
        ]

        notes: list[str] = []
        violations: list[GuardrailViolation] = []
        llm_used = False
        summary: str | None = None
        challenges, differentials, citations = det_challenges, _deterministic_differentials(iris, tri), []
        diagnostics, gaps = det_diagnostics, det_gaps

        if self.llm.available:
            payload = {
                "WORKING_HYPOTHESIS": request.working_hypothesis,
                "CLINICAL_QUESTION": request.clinical_question,
                "CLINICIAN": request.clinician,
                "DETERMINISTIC_ANALYSIS": {
                    "iris_staging": iris.model_dump(mode="json"),
                    "triaditis": tri.model_dump(mode="json"),
                    "lab_trends": [t.model_dump(mode="json") for t in trends],
                    "weight_trend": weight_trend.model_dump(mode="json"),
                },
                "PATIENT_RECORD": _record_for_prompt(record),
            }
            try:
                result = self.llm.generate_structured(
                    model=self.model, system_instruction=MICHELIN_SYSTEM_PROMPT,
                    contents=[text_part(json.dumps(payload, indent=2, default=str))], schema=LLMCrossExamination,
                )
                llm_used = True
                allowed = v.allowed_numbers(record, v.trend_numbers([*trends, weight_trend]))
                citations, cite_v = v.verify_cited_values(result.cited_lab_values, record)
                summary, s_v = v.redact_unverified_numbers(result.clinical_summary, allowed, "clinical_summary")
                llm_challenges, c_v = v.redact_list(result.challenges_to_hypothesis, allowed, "challenges")
                llm_diag, d_v = v.redact_list(result.recommended_diagnostics, allowed, "recommended_diagnostics")
                llm_gaps, g_v = v.redact_list(result.data_gaps, allowed, "data_gaps")
                violations += cite_v + s_v + c_v + d_v + g_v
                differentials = []
                for i, d in enumerate(result.differentials):
                    cond, v1 = v.redact_unverified_numbers(d.condition, allowed, f"differentials[{i}].condition")
                    sup, v2 = v.redact_list(d.supporting_evidence, allowed, f"differentials[{i}].supporting")
                    con, v3 = v.redact_list(d.contradicting_evidence, allowed, f"differentials[{i}].contradicting")
                    nxt, v4 = v.redact_list(d.next_steps, allowed, f"differentials[{i}].next_steps")
                    violations += v1 + v2 + v3 + v4
                    differentials.append(Differential(
                        condition=cond, likelihood=d.likelihood, supporting_evidence=sup,
                        contradicting_evidence=con, next_steps=nxt,
                    ))
                challenges = _dedupe(llm_challenges + det_challenges)
                diagnostics = _dedupe(llm_diag + det_diagnostics)
                gaps = _dedupe(llm_gaps + det_gaps)
            except LLMResponseError as exc:
                notes.append(f"LLM unavailable ({exc}); deterministic analysis only.")
        else:
            notes.append("Gemini not configured; deterministic analysis only.")

        if summary is None:
            summary = (
                f"Deterministic analysis: {iris.stage_label}. Triaditis assessment: {tri.classification}. "
                f"{len(ambiguous)} AMBIGUOUS data point(s) require manual verification."
            )
        if violations:
            notes.append(f"{len(violations)} guardrail violation(s) detected in model output and neutralised.")

        return MichelinCrossExamResponse(
            patient_id=record.patient_id, clinician=request.clinician,
            working_hypothesis=request.working_hypothesis, generated_at=utcnow(),
            record_version=record.record_version, iris_staging=iris, triaditis=tri, lab_trends=trends,
            weight_trend=weight_trend, ambiguous_data=ambiguous, open_escalations=open_escalations,
            recent_home_observations=record.home_observations[-5:], clinical_summary=summary,
            challenges_to_hypothesis=challenges, differentials=differentials, verified_citations=citations,
            recommended_diagnostics=diagnostics, data_gaps=gaps, guardrail_violations=violations,
            llm_used=llm_used, model=self.model if llm_used else None, guardrail_notes=notes,
        )


# =========================================================================== #
# Strict OCR / lab extraction (gemini-2.5-pro)
# =========================================================================== #

EXTRACTION_SYSTEM_PROMPT = f"""You are a veterinary laboratory report transcription engine operating under a strict
OCR protocol. Transcribe; never interpret.

For every analyte on the report return:
- raw_text: the exact characters printed for the result (including H/L flags). If any character is unreadable,
  write '?' in its place. Null if the result cannot be seen at all.
- value: the number in raw_text ONLY if every digit and the decimal point are fully legible; otherwise null.
- unit, reference_low, reference_high: exactly as printed; null if absent or unreadable.
- legible: false if the value is blurred, smudged, cut off, overwritten, faint, or uncertain in any way.
- confidence: your probability (0-1) that every character of the value is correct.

Never infer digits from reference intervals, previous results, other analytes or clinical plausibility.
Ambiguous values are recorded downstream as status AMBIGUOUS with the flag "{AMBIGUOUS_FLAG}".
"""


class LabExtractionAgent:
    def __init__(self, llm: GeminiClient, store: PatientStore, model: str = EXTRACTION_MODEL):
        self.llm, self.store, self.model = llm, store, model

    def extract(self, patient_id: str, request: LabExtractionRequest) -> LabExtractionResponse:
        record = self.store.get(patient_id)
        if not self.llm.available:
            raise LLMUnavailableError("Lab extraction requires a configured Gemini model; nothing was guessed.")

        contents: list[Any] = [text_part(
            f"Transcribe the laboratory report for patient {record.patient_id} ({record.name}), "
            f"collected {request.collected_at}, source '{request.source_document}'."
        )]
        if request.image_base64:
            try:
                data = base64.b64decode(request.image_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image_base64 is not valid base64") from exc
            contents.append(bytes_part(data, request.mime_type or "image/png"))
        if request.document_text:
            contents.append(text_part("REPORT TEXT:\n" + request.document_text))

        extraction = self.llm.generate_structured(
            model=self.model, system_instruction=EXTRACTION_SYSTEM_PROMPT, contents=contents,
            schema=LLMLabExtraction,
        )
        labs, violations = v.sanitize_extracted_labs(
            extraction, collected_at=request.collected_at, source_document=request.source_document,
            laboratory=request.laboratory,
        )
        version = record.record_version
        if request.commit and labs:
            version = self.store.add_labs(
                patient_id, labs, actor="michelin", expected_version=request.expected_record_version
            ).record_version
        return LabExtractionResponse(
            patient_id=patient_id, labs=labs,
            ambiguous_count=sum(lab.status is LabStatus.AMBIGUOUS for lab in labs),
            guardrail_violations=violations, llm_used=True, model=self.model,
            committed=bool(request.commit and labs), record_version=version,
        )
