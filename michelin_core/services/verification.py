"""Deterministic clinical guardrails and feline domain logic.

Everything in this module is pure, deterministic Python. It is the safety net
that sits *between* Gemini and the Central Patient Store / the end user:

* Strict OCR / extraction protocol - any unreadable, low-confidence or
  self-inconsistent extraction becomes ``status=AMBIGUOUS, value=None`` with
  the mandated flag. Values are taken from the verbatim ``raw_text`` the model
  transcribed, never from a "cleaned up" number.
* Zero-hallucination verification - every lab value an LLM cites is checked
  against the SSOT; numbers embedded in narrative text next to a lab unit or
  analyte name must exist in the SSOT (or be a published clinical threshold),
  otherwise they are redacted.
* Feline domain logic - longitudinal trends, IRIS CKD staging (2023 feline
  thresholds), triaditis pattern recognition and caregiver triage red flags.
  None of these functions ever interpolate or extrapolate a value.
"""

import math
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from michelin_core.schemas.patient import (
    AMBIGUOUS_FLAG,
    CitedLabValue,
    GuardrailViolation,
    HomeObservationInput,
    IrisStaging,
    LabResult,
    LabStatus,
    LLMLabExtraction,
    MediaAssessment,
    MediaAttachment,
    PatientRecord,
    Sex,
    TrendPoint,
    TrendSummary,
    TriageLevel,
    TriaditisAssessment,
    derive_status,
)

FLOAT_TOL = 1e-6
EXTRACTION_CONFIDENCE_THRESHOLD = 0.90

# --------------------------------------------------------------------------- #
# Published feline thresholds (IRIS 2023, Spec fPL)
# --------------------------------------------------------------------------- #

USG_ADEQUATE_CAT = 1.035
SDMA_PERSISTENT_THRESHOLD = 14.0
CREATININE_SIGNIFICANT_RISE = 0.3  # mg/dL - IRIS: a rise >= 0.3 mg/dL is clinically relevant
FPLI_NORMAL_MAX = 3.5
FPLI_CONSISTENT_MIN = 5.4

CLINICAL_THRESHOLDS: frozenset[float] = frozenset(
    {
        1.6, 2.8, 2.9, 5.0,  # IRIS creatinine (mg/dL)
        18, 25, 26, 38, 14,  # IRIS SDMA (ug/dL)
        1.035,  # feline USG adequacy
        0.2, 0.4,  # IRIS UPC substaging
        140, 159, 160, 179, 180,  # IRIS blood-pressure substaging (mmHg)
        3.5, 3.6, 5.3, 5.4,  # Spec fPL bands (ug/L)
        0.3,  # IRIS creatinine rise
        1, 2, 3, 4,  # IRIS stage numbers
    }
)

STABILITY_THRESHOLDS: dict[str, float] = {  # absolute change below which a trend is "stable"
    "creatinine": CREATININE_SIGNIFICANT_RISE,
    "usg": 0.005,
    "sdma": 2.0,
}
DEFAULT_RELATIVE_STABILITY = 0.05

ANALYTE_ALIASES: dict[str, tuple[str, ...]] = {
    "creatinine": ("creatinine", "crea", "creat", "cre"),
    "sdma": ("sdma", "symmetric dimethylarginine"),
    "bun": ("bun", "urea", "urea nitrogen", "blood urea nitrogen"),
    "phosphorus": ("phosphorus", "phos", "phosphate"),
    "potassium": ("potassium", "k", "k+"),
    "alt": ("alt", "alanine aminotransferase", "sgpt"),
    "alp": ("alp", "alkp", "alkaline phosphatase"),
    "ggt": ("ggt", "gamma glutamyl transferase"),
    "total_bilirubin": ("total bilirubin", "tbil", "bilirubin"),
    "fpli": ("fpli", "spec fpl", "spec fpl (fpli)", "fpl", "feline pancreatic lipase"),
    "usg": ("usg", "urine specific gravity", "specific gravity", "sg"),
    "upc": ("upc", "upcr", "urine protein:creatinine", "urine protein creatinine ratio"),
    "platelets": ("platelets", "platelet count", "plt"),
    "hct": ("hct", "hematocrit", "haematocrit", "pcv"),
    "total_t4": ("total t4", "tt4", "t4"),
    "cobalamin": ("cobalamin", "b12", "vitamin b12"),
    "folate": ("folate",),
    "systolic_bp": ("systolic bp", "sbp", "systolic blood pressure"),
}
_ALIAS_LOOKUP = {alias: key for key, aliases in ANALYTE_ALIASES.items() for alias in aliases}

PANEL_BY_ANALYTE = {
    "platelets": "cbc",
    "hct": "cbc",
    "usg": "urinalysis",
    "upc": "urinalysis",
    "total_t4": "endocrine",
    "fpli": "special",
    "cobalamin": "special",
    "folate": "special",
    "systolic_bp": "vitals",
}


def canonical_analyte(name: str) -> str:
    norm = re.sub(r"\s+", " ", name.strip().lower())
    if norm in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[norm]
    return re.sub(r"[^a-z0-9]+", "_", norm).strip("_") or "unknown"


def fmt_value(analyte: str, value: float | None) -> str:
    if value is None:
        return "null"
    if analyte == "usg":
        return f"{value:.3f}"
    return f"{value:g}"


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=FLOAT_TOL)


# --------------------------------------------------------------------------- #
# Lab access helpers
# --------------------------------------------------------------------------- #


def labs_for(record: PatientRecord, analyte: str) -> list[LabResult]:
    return sorted((lab for lab in record.labs if lab.analyte == analyte), key=lambda lab: lab.collected_at)


def latest_any(record: PatientRecord, analyte: str) -> LabResult | None:
    series = labs_for(record, analyte)
    return series[-1] if series else None


def latest_verified(record: PatientRecord, analyte: str) -> LabResult | None:
    series = [lab for lab in labs_for(record, analyte) if lab.status is not LabStatus.AMBIGUOUS]
    return series[-1] if series else None


def ambiguous_labs(record: PatientRecord) -> list[LabResult]:
    return [lab for lab in record.labs if lab.status is LabStatus.AMBIGUOUS]


# --------------------------------------------------------------------------- #
# Longitudinal trends (no interpolation, no extrapolation)
# --------------------------------------------------------------------------- #


def _trend(analyte: str, unit: str | None, points: list[TrendPoint]) -> TrendSummary:
    numeric = [p for p in points if p.value is not None]
    ambiguous = len(points) - len(numeric)
    if len(numeric) < 2:
        only = numeric[0].value if numeric else None
        return TrendSummary(
            analyte=analyte, unit=unit, points=points, first_value=only, last_value=only,
            absolute_change=None, percent_change=None, direction="insufficient_data",
            ambiguous_points=ambiguous,
            note="Fewer than two verified measurements - trend NOT estimated (no extrapolation).",
        )
    first, last = numeric[0].value, numeric[-1].value
    assert first is not None and last is not None
    change = round(last - first, 4)
    percent = round(change / first * 100, 1) if first != 0 else None
    if analyte in STABILITY_THRESHOLDS:
        stable = abs(change) < STABILITY_THRESHOLDS[analyte] - FLOAT_TOL
    else:
        stable = abs(change) < abs(first) * DEFAULT_RELATIVE_STABILITY
    direction = "stable" if stable else ("rising" if change > 0 else "falling")
    note = "Computed only from verified measured values; no interpolation or extrapolation."
    if ambiguous:
        note += (
            f" {ambiguous} AMBIGUOUS point(s) excluded (value null); trend reflects verified values "
            f"{numeric[0].collected_at} to {numeric[-1].collected_at} only."
        )
    return TrendSummary(
        analyte=analyte, unit=unit, points=points, first_value=first, last_value=last,
        absolute_change=change, percent_change=percent, direction=direction, ambiguous_points=ambiguous,
        note=note,
    )


def compute_lab_trend(record: PatientRecord, analyte: str) -> TrendSummary:
    series = labs_for(record, analyte)
    points = [TrendPoint(collected_at=lab.collected_at, value=lab.value, status=lab.status) for lab in series]
    units = {lab.unit for lab in series}
    if len(units) > 1:
        return TrendSummary(
            analyte=analyte, unit=None, points=points, first_value=None, last_value=None,
            absolute_change=None, percent_change=None, direction="insufficient_data",
            ambiguous_points=sum(p.value is None for p in points),
            note=f"Unit mismatch across measurements {sorted(units)} - trend NOT computed.",
        )
    return _trend(analyte, units.pop() if units else None, points)


def compute_weight_trend(record: PatientRecord) -> TrendSummary:
    weights = sorted(record.weights, key=lambda w: w.recorded_at)
    points = [TrendPoint(collected_at=w.recorded_at, value=w.weight_kg, status=LabStatus.NO_REFERENCE) for w in weights]
    return _trend("body_weight", "kg", points)


KEY_TREND_ANALYTES = ("creatinine", "sdma", "bun", "usg", "phosphorus", "fpli", "alt", "platelets", "hct")


# --------------------------------------------------------------------------- #
# IRIS CKD staging (feline, IRIS 2023)
# --------------------------------------------------------------------------- #


def iris_stage_from_creatinine(value: float) -> int:
    if value < 1.6:
        return 1
    if value <= 2.8:
        return 2
    if value <= 5.0:
        return 3
    return 4


def iris_stage_from_sdma(value: float) -> int:
    if value < 18:
        return 1
    if value <= 25:
        return 2
    if value <= 38:
        return 3
    return 4


def _proteinuria_substage(upc: float) -> str:
    if upc < 0.2:
        return "Non-proteinuric (UPC < 0.2)"
    if upc <= 0.4:
        return "Borderline proteinuric (UPC 0.2-0.4)"
    return "Proteinuric (UPC > 0.4)"


def _bp_substage(sbp: float) -> str:
    if sbp < 140:
        return "Normotensive (< 140 mmHg)"
    if sbp < 160:
        return "Prehypertensive (140-159 mmHg)"
    if sbp < 180:
        return "Hypertensive (160-179 mmHg)"
    return "Severely hypertensive (>= 180 mmHg)"


CKD_EXPECTED_DATA = {
    "total_t4": "Total T4 (hyperthyroidism can mask azotemia in senior cats)",
    "upc": "Urine protein:creatinine ratio (IRIS proteinuria substage)",
    "systolic_bp": "Systolic blood pressure (IRIS hypertension substage)",
    "potassium": "Serum potassium",
    "urine_culture": "Urine culture",
}


def assess_iris_ckd(record: PatientRecord) -> IrisStaging:
    crea_trend = compute_lab_trend(record, "creatinine")
    crea = latest_verified(record, "creatinine")
    sdma = latest_verified(record, "sdma")
    usg = latest_verified(record, "usg")
    upc = latest_verified(record, "upc")
    sbp = latest_verified(record, "systolic_bp")

    evidence: list[str] = []
    caveats: list[str] = [
        "IRIS staging is provisional: it requires a stable, well-hydrated patient with persistent findings. "
        "Repeat creatinine, SDMA and USG within 2-4 weeks (after correcting any dehydration) to confirm."
    ]
    missing: list[str] = []

    newest_crea = latest_any(record, "creatinine")
    if newest_crea is not None and newest_crea.status is LabStatus.AMBIGUOUS:
        caveats.append(
            f"Most recent creatinine ({newest_crea.collected_at}) is AMBIGUOUS and was excluded; "
            "staging uses the most recent verified value only."
        )

    for analyte, label in CKD_EXPECTED_DATA.items():
        if latest_verified(record, analyte) is None:
            missing.append(f"{label} - not present in SSOT; value NOT inferred")

    if crea is None:
        return IrisStaging(
            stage=None, stage_label="Cannot stage: no verified creatinine in the SSOT",
            creatinine_trend=crea_trend, caveats=caveats, missing_data=missing,
        )

    crea_stage = iris_stage_from_creatinine(crea.value)  # type: ignore[arg-type]
    sdma_stage = iris_stage_from_sdma(sdma.value) if sdma and sdma.value is not None else None
    usg_adequate = (usg.value >= USG_ADEQUATE_CAT) if usg and usg.value is not None else None

    supporting: list[str] = []
    if usg_adequate is False:
        supporting.append(
            f"USG {fmt_value('usg', usg.value)} ({usg.collected_at}) is below {USG_ADEQUATE_CAT:.3f}: "  # type: ignore[union-attr]
            "inadequate urine concentrating ability for a cat"
        )
    if crea_trend.direction == "rising":
        verified_pts = [p for p in crea_trend.points if p.value is not None]
        path = " -> ".join(fmt_value("creatinine", p.value) for p in verified_pts)
        supporting.append(
            f"Creatinine rising {path} mg/dL across {len(verified_pts)} verified measurements "
            f"({verified_pts[0].collected_at} to {verified_pts[-1].collected_at}; "
            f"+{crea_trend.absolute_change:g} mg/dL, +{crea_trend.percent_change:g}%)"
        )
    sdma_high = [lab for lab in labs_for(record, "sdma") if lab.value is not None and lab.value > SDMA_PERSISTENT_THRESHOLD]
    if len(sdma_high) >= 2:
        supporting.append(
            "SDMA persistently > 14 ug/dL ("
            + ", ".join(f"{fmt_value('sdma', lab.value)} on {lab.collected_at}" for lab in sdma_high)
            + ")"
        )
    evidence.extend(supporting)

    if crea.status is LabStatus.NORMAL and crea_stage >= 2:
        evidence.insert(
            0,
            f"Latest creatinine {fmt_value('creatinine', crea.value)} mg/dL ({crea.collected_at}) is inside the "
            f"laboratory reference interval ({crea.reference_low:g}-{crea.reference_high:g}) but within the IRIS "
            f"Stage {crea_stage} creatinine band - a reference-interval 'normal' does NOT exclude feline CKD.",
        )

    if crea_stage >= 2 and supporting:
        stage: int | None = crea_stage
        label = f"IRIS Stage {crea_stage} CKD (provisional)"
    elif crea_stage == 1 and supporting:
        stage = 1
        label = "IRIS Stage 1 CKD (provisional; non-azotemic with renal markers)"
    elif crea_stage >= 2:
        stage = None
        label = f"Indeterminate: creatinine in IRIS Stage {crea_stage} band without corroborating renal markers"
    else:
        stage = None
        label = "Not consistent with CKD on available verified data"

    if sdma_stage is not None and sdma_stage > crea_stage:
        caveats.append(
            f"SDMA suggests IRIS Stage {sdma_stage} (higher than creatinine Stage {crea_stage}); per IRIS guidance "
            "consider the higher stage, particularly in cats with low muscle mass."
        )

    proteinuria = _proteinuria_substage(upc.value) if upc and upc.value is not None else None  # type: ignore[arg-type]
    hypertension = None
    if sbp and sbp.value is not None:
        hypertension = _bp_substage(sbp.value)
        caveats.append(
            "Blood-pressure substage is based on a single verified reading; confirm with repeated measurements "
            "before treating."
        )

    return IrisStaging(
        stage=stage, stage_label=label, creatinine_stage=crea_stage, sdma_stage=sdma_stage,
        latest_creatinine=crea.value, latest_sdma=sdma.value if sdma else None,
        latest_usg=usg.value if usg else None, usg_adequate=usg_adequate, creatinine_trend=crea_trend,
        proteinuria_substage=proteinuria, hypertension_substage=hypertension,
        evidence=evidence, caveats=caveats, missing_data=missing,
    )


# --------------------------------------------------------------------------- #
# Feline triaditis pattern recognition
# --------------------------------------------------------------------------- #

GI_KEYWORDS: dict[str, str] = {
    "vomit": "vomiting",
    "threw up": "vomiting",
    "diarrh": "diarrhoea",
    "anorex": "anorexia",
    "inappet": "inappetence",
    "reduced appetite": "reduced appetite",
    "hyporex": "hyporexia",
    "ate about half": "reduced appetite",
    "weight loss": "weight loss",
    "abdominal discomfort": "abdominal discomfort",
    "abdominal pain": "abdominal discomfort",
}

TRIADITIS_DIAGNOSTICS = [
    "Abdominal ultrasound (pancreas, liver/biliary tree, intestinal wall layering)",
    "Serum cobalamin and folate (small-intestinal involvement)",
    "ALP, GGT and bile acids (hepatobiliary/cholangitis work-up)",
    "Repeat Spec fPL in 2-3 weeks to assess trajectory",
    "Hepatic FNA/biopsy and full-thickness intestinal biopsies if imaging is abnormal "
    "(definitive triaditis diagnosis requires histopathology)",
]


_NEGATION_RE = re.compile(r"\b(no|not|without|denies|denied|absent|negative for|nor)\b[^.;]{0,25}$", re.IGNORECASE)


def _mentions_affirmatively(text: str, keyword: str) -> bool:
    """True if ``keyword`` occurs at least once without a preceding negation in the same clause."""
    low = text.lower()
    start = low.find(keyword)
    while start != -1:
        if not _NEGATION_RE.search(low[max(0, start - 40):start]):
            return True
        start = low.find(keyword, start + 1)
    return False


def _gi_signs(record: PatientRecord) -> dict[str, list[str]]:
    signs: dict[str, list[str]] = {}

    def scan(text: str, source: str) -> None:
        for kw, sign in GI_KEYWORDS.items():
            if _mentions_affirmatively(text, kw) and source not in signs.setdefault(sign, []):
                signs[sign].append(source)

    for enc in record.encounters:
        scan(f"{enc.reason} {enc.findings}", f"{enc.encounter_id} ({enc.encounter_date})")
    for obs in record.home_observations:
        src = f"{obs.observation_id} ({obs.observed_at.date()})"
        scan(" ".join(obs.symptoms) + " " + obs.free_text, src)
        if obs.vomiting_episodes_24h:
            signs.setdefault("vomiting", [])
            if src not in signs["vomiting"]:
                signs["vomiting"].append(src)
        if obs.appetite in ("reduced", "none"):
            signs.setdefault("reduced appetite", [])
            if src not in signs["reduced appetite"]:
                signs["reduced appetite"].append(src)
    return {k: v for k, v in signs.items() if v}


def assess_triaditis(record: PatientRecord) -> TriaditisAssessment:
    fpli = latest_verified(record, "fpli")
    fpli_trend = compute_lab_trend(record, "fpli")
    pancreatic: list[str] = []
    gi: list[str] = []
    hepato: list[str] = []

    if fpli is not None and fpli.value is not None:
        if fpli.value >= FPLI_CONSISTENT_MIN:
            pancreatic.append(f"Spec fPL {fpli.value:g} ug/L ({fpli.collected_at}) >= 5.4: consistent with pancreatitis")
        elif fpli.value > FPLI_NORMAL_MAX:
            pancreatic.append(f"Spec fPL {fpli.value:g} ug/L ({fpli.collected_at}) in equivocal band (3.6-5.3)")
        if fpli_trend.direction == "rising":
            path = " -> ".join(f"{p.value:g}" for p in fpli_trend.points if p.value is not None)
            pancreatic.append(f"Spec fPL rising {path} ug/L")

    for sign, sources in _gi_signs(record).items():
        gi.append(f"{sign}: {', '.join(sources)}")
    wt = compute_weight_trend(record)
    if wt.direction == "falling":
        gi.append(
            f"Body weight falling {wt.first_value:g} -> {wt.last_value:g} kg ({wt.percent_change:g}%)"
        )

    for analyte in ("alt", "alp", "ggt", "total_bilirubin"):
        lab = latest_verified(record, analyte)
        if lab is not None and lab.status is LabStatus.HIGH:
            hepato.append(
                f"{lab.display_name} {lab.value:g} {lab.unit} ({lab.collected_at}) above reference "
                f"(upper limit {lab.reference_high:g})"
            )

    missing = [
        f"{label} - not present in SSOT; value NOT inferred"
        for analyte, label in (
            ("cobalamin", "Serum cobalamin"), ("folate", "Serum folate"), ("alp", "ALP"), ("ggt", "GGT"),
        )
        if latest_verified(record, analyte) is None
    ]

    fpli_value = fpli.value if fpli else None
    consistent = fpli_value is not None and fpli_value >= FPLI_CONSISTENT_MIN
    if consistent and gi and hepato:
        classification = "triaditis_suspected"
    elif consistent and gi:
        classification = "pancreatitis_with_gi_signs"
    elif fpli_value is not None and fpli_value > FPLI_NORMAL_MAX:
        classification = "pancreatitis_not_excluded"
    else:
        classification = "not_supported"

    return TriaditisAssessment(
        classification=classification, latest_fpli=fpli_value, pancreatic_evidence=pancreatic,
        gastrointestinal_evidence=gi, hepatobiliary_evidence=hepato, missing_data=missing,
        recommended_diagnostics=TRIADITIS_DIAGNOSTICS if classification != "not_supported" else [],
    )


# --------------------------------------------------------------------------- #
# Strict OCR / extraction protocol
# --------------------------------------------------------------------------- #

_AMBIGUITY_MARKERS = ("?", "#", "~", "illegible", "unreadable", "smudge", "blur", "obscured", "faded")
_LEADING_NUMBER = re.compile(r"^(\d+(?:[.,]\d+)?)")
_SCI_NOTATION = re.compile(r"x?\s*10\s*\^?\s*\d+", re.IGNORECASE)
_FLAG_TOKENS = re.compile(r"\b(h|l|high|low|hi|lo)\b", re.IGNORECASE)


def parse_raw_numeric(raw: str | None) -> float | None:
    """Parse a verbatim transcription into a number, or ``None`` if not unambiguous.

    Rejects: ambiguity markers, censored values (``<0.5``), comma decimals
    (thousands vs decimal separator is ambiguous) and any extra digits beyond
    a single number plus unit / H-L flag.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    low = text.lower()
    if any(marker in low for marker in _AMBIGUITY_MARKERS):
        return None
    if text[0] in "<>≤≥":
        return None
    match = _LEADING_NUMBER.match(text)
    if not match or "," in match.group(1):
        return None
    rest = _FLAG_TOKENS.sub("", _SCI_NOTATION.sub("", text[match.end():]))
    if re.search(r"\d", rest):
        return None
    return float(match.group(1))


def _new_lab_id(collected_at: date, analyte: str) -> str:
    return f"LAB-{collected_at:%Y%m%d}-{analyte.upper()[:8]}-{uuid.uuid4().hex[:6]}"


def ambiguous_lab(
    *, analyte: str, display_name: str, unit: str | None, collected_at: date, source_document: str,
    raw_text: str | None, laboratory: str | None, reference_low: float | None = None,
    reference_high: float | None = None, lab_id: str | None = None,
) -> LabResult:
    return LabResult(
        lab_id=lab_id or _new_lab_id(collected_at, analyte), analyte=analyte, display_name=display_name,
        panel=PANEL_BY_ANALYTE.get(analyte, "chemistry"), value=None, unit=unit or "",
        reference_low=reference_low, reference_high=reference_high, status=LabStatus.AMBIGUOUS,
        flag=AMBIGUOUS_FLAG, raw_text=raw_text, collected_at=collected_at, source_document=source_document,
        laboratory=laboratory,
    )


def sanitize_extracted_labs(
    extraction: LLMLabExtraction, *, collected_at: date, source_document: str, laboratory: str | None = None,
    confidence_threshold: float = EXTRACTION_CONFIDENCE_THRESHOLD,
) -> tuple[list[LabResult], list[GuardrailViolation]]:
    """Convert raw LLM OCR output into SSOT-grade ``LabResult`` objects."""
    labs: list[LabResult] = []
    violations: list[GuardrailViolation] = []

    for item in extraction.items:
        analyte = canonical_analyte(item.analyte)
        parsed = parse_raw_numeric(item.raw_text)
        reasons: list[str] = []
        field_name = f"labs.{analyte}"

        if not item.legible:
            reasons.append("model marked value illegible")
            if item.value is not None:
                violations.append(GuardrailViolation(
                    code="AMBIGUOUS_VALUE_FILLED", field=field_name,
                    detail=f"Model supplied {item.value:g} for a value it marked illegible; nulled.",
                ))
        if item.confidence < confidence_threshold:
            reasons.append(f"confidence {item.confidence:.2f} < {confidence_threshold:.2f}")
            violations.append(GuardrailViolation(
                code="LOW_CONFIDENCE_EXTRACTION", field=field_name,
                detail=f"Extraction confidence {item.confidence:.2f} below threshold; value nulled.",
            ))
        if item.raw_text is None or parsed is None:
            reasons.append("verbatim raw text missing or not an unambiguous number")
            if item.value is not None and item.legible:
                violations.append(GuardrailViolation(
                    code="AMBIGUOUS_VALUE_FILLED", field=field_name,
                    detail=f"Model supplied {item.value:g} but raw text {item.raw_text!r} is ambiguous; nulled.",
                ))
        if item.value is None:
            reasons.append("no value returned")
        elif parsed is not None and not _isclose(parsed, item.value):
            reasons.append("value disagrees with verbatim raw text")
            violations.append(GuardrailViolation(
                code="VALUE_MISMATCH", field=field_name,
                detail=f"Model value {item.value:g} != printed raw text {item.raw_text!r}; nulled.",
            ))
        if not item.unit and analyte != "usg":
            reasons.append("unit missing")
        ref_low, ref_high = item.reference_low, item.reference_high
        if ref_low is not None and ref_high is not None and ref_low > ref_high:
            reasons.append("invalid reference interval")
            ref_low = ref_high = None

        if reasons:
            labs.append(ambiguous_lab(
                analyte=analyte, display_name=item.display_name, unit=item.unit, collected_at=collected_at,
                source_document=source_document, raw_text=item.raw_text, laboratory=laboratory,
                reference_low=ref_low, reference_high=ref_high,
            ))
            continue

        labs.append(LabResult(
            lab_id=_new_lab_id(collected_at, analyte), analyte=analyte, display_name=item.display_name,
            panel=PANEL_BY_ANALYTE.get(analyte, "chemistry"), value=parsed, unit=item.unit or "",
            reference_low=ref_low, reference_high=ref_high, status=derive_status(parsed, ref_low, ref_high),
            raw_text=item.raw_text, collected_at=collected_at, source_document=source_document,
            laboratory=laboratory,
        ))
    return labs, violations


# --------------------------------------------------------------------------- #
# Zero-hallucination verification of LLM narrative / citations
# --------------------------------------------------------------------------- #


def allowed_numbers(record: PatientRecord, extra: Iterable[float | None] = ()) -> set[float]:
    values: set[float] = set(CLINICAL_THRESHOLDS)
    values.add(record.age_years)
    for lab in record.labs:
        for v in (lab.value, lab.reference_low, lab.reference_high):
            if v is not None:
                values.add(v)
    for w in record.weights:
        values.add(w.weight_kg)
        if w.body_condition_score is not None:
            values.add(float(w.body_condition_score))
    for v in extra:
        if v is not None:
            values.add(v)
            values.add(abs(v))
    return values


def trend_numbers(trends: Iterable[TrendSummary]) -> list[float | None]:
    out: list[float | None] = []
    for t in trends:
        out.extend([t.absolute_change, t.percent_change])
    return out


def verify_cited_values(
    cited: Iterable[CitedLabValue], record: PatientRecord
) -> tuple[list[CitedLabValue], list[GuardrailViolation]]:
    """Every cited value must match the SSOT exactly; the SSOT value always wins."""
    index = {(lab.analyte, lab.collected_at.isoformat()): lab for lab in record.labs}
    verified: list[CitedLabValue] = []
    violations: list[GuardrailViolation] = []
    for c in cited:
        analyte = canonical_analyte(c.analyte)
        key = (analyte, c.collected_at.strip()[:10])
        lab = index.get(key)
        fld = f"cited_lab_values.{analyte}@{key[1]}"
        if lab is None:
            if c.value is not None:
                violations.append(GuardrailViolation(
                    code="FABRICATED_VALUE", field=fld,
                    detail=f"No SSOT measurement for {analyte} on {key[1]}; cited value {c.value:g} discarded.",
                ))
            continue
        if lab.status is LabStatus.AMBIGUOUS:
            if c.value is not None:
                violations.append(GuardrailViolation(
                    code="AMBIGUOUS_VALUE_FILLED", field=fld,
                    detail=f"SSOT value is AMBIGUOUS; model-cited {c.value:g} discarded and set to null.",
                ))
            verified.append(CitedLabValue(analyte=lab.analyte, collected_at=key[1], value=None, unit=lab.unit))
            continue
        if c.value is not None and not _isclose(c.value, lab.value):  # type: ignore[arg-type]
            violations.append(GuardrailViolation(
                code="VALUE_MISMATCH", field=fld,
                detail=f"Model cited {c.value:g}; SSOT value is {fmt_value(analyte, lab.value)}. SSOT value used.",
            ))
        verified.append(CitedLabValue(analyte=lab.analyte, collected_at=key[1], value=lab.value, unit=lab.unit))
    return verified, violations


_NUM = r"\d+(?:\.\d+)?"
_UNITS = (
    r"mg/dl|µg/dl|μg/dl|ug/dl|mcg/dl|µg/l|μg/l|ug/l|mcg/l|ng/ml|u/l|iu/l|k/µl|k/μl|k/ul|"
    r"x\s?10\^?[39]/[uµμ]?l|mmol/l|mmhg|kg"
)
_NUM_UNIT_RE = re.compile(
    rf"(?<![\w.])(?P<nums>{_NUM}(?:\s*(?:–|—|-|→|->|to|and|,)\s*{_NUM})*)\s*(?P<unit>{_UNITS})(?![a-z])",
    re.IGNORECASE,
)
_ANALYTE_WORDS = (
    r"platelets?|plt|creatinine|crea|sdma|bun|urea|phosphorus|phosphate|alt|alp|ggt|fpli|spec fpl|fpl|"
    r"usg|specific gravity|hct|hematocrit|haematocrit|pcv|potassium|upc|bilirubin|t4|cobalamin|folate"
)
_ANALYTE_NUM_RE = re.compile(
    rf"\b(?P<analyte>{_ANALYTE_WORDS})\b(?P<gap>[^.\d\n]{{0,25}}?)(?<![\d-])(?P<num>{_NUM})(?![\d-])",
    re.IGNORECASE,
)
_USG_RE = re.compile(r"(?<![\w.])(1\.0\d{1,3})(?![\d])")
REDACTION_TOKEN = "[UNVERIFIED-VALUE-REDACTED]"


def _is_allowed(num_text: str, allowed: set[float]) -> bool:
    try:
        n = float(num_text)
    except ValueError:
        return False
    return any(_isclose(n, a) for a in allowed)


def redact_unverified_numbers(
    text: str, allowed: set[float], field_name: str
) -> tuple[str, list[GuardrailViolation]]:
    """Redact numbers that look like clinical measurements but are absent from the SSOT."""
    violations: list[GuardrailViolation] = []

    def flag(num_text: str, context: str) -> str:
        violations.append(GuardrailViolation(
            code="UNVERIFIED_NUMBER_IN_TEXT", field=field_name,
            detail=f"'{context.strip()}' contains {num_text}, which is not present in the SSOT; redacted.",
        ))
        return REDACTION_TOKEN

    def sub_num_unit(m: re.Match[str]) -> str:
        nums = re.sub(
            _NUM, lambda n: n.group(0) if _is_allowed(n.group(0), allowed) else flag(n.group(0), m.group(0)),
            m.group("nums"),
        )
        return m.string[m.start():m.start("nums")] + nums + m.string[m.end("nums"):m.end()]

    def sub_analyte(m: re.Match[str]) -> str:
        num = m.group("num")
        if _is_allowed(num, allowed):
            return m.group(0)
        return m.group("analyte") + m.group("gap") + flag(num, m.group(0))

    def sub_usg(m: re.Match[str]) -> str:
        return m.group(0) if _is_allowed(m.group(1), allowed) else flag(m.group(1), m.group(0))

    text = _NUM_UNIT_RE.sub(sub_num_unit, text)
    text = _ANALYTE_NUM_RE.sub(sub_analyte, text)
    text = _USG_RE.sub(sub_usg, text)
    return text, violations


def redact_list(items: list[str], allowed: set[float], field_name: str) -> tuple[list[str], list[GuardrailViolation]]:
    out: list[str] = []
    violations: list[GuardrailViolation] = []
    for i, item in enumerate(items):
        clean, v = redact_unverified_numbers(item, allowed, f"{field_name}[{i}]")
        out.append(clean)
        violations.extend(v)
    return out, violations


# --------------------------------------------------------------------------- #
# Media guardrails
# --------------------------------------------------------------------------- #


def enforce_media_guardrails(
    assessments: Iterable[MediaAssessment], media: list[MediaAttachment]
) -> tuple[list[MediaAssessment], list[GuardrailViolation]]:
    """Every attachment ends up with exactly one assessment; anything unclear is AMBIGUOUS."""
    known = {m.media_id for m in media if m.media_id}
    by_id: dict[str, MediaAssessment] = {}
    violations: list[GuardrailViolation] = []
    for a in assessments:
        if a.media_id not in known:
            violations.append(GuardrailViolation(
                code="FABRICATED_VALUE", field=f"media_assessments.{a.media_id}",
                detail="Assessment references a media item that was not submitted; discarded.",
            ))
            continue
        ambiguous = a.status == "AMBIGUOUS" or a.flag == AMBIGUOUS_FLAG or not (a.finding or "").strip()
        if ambiguous:
            if a.finding:
                violations.append(GuardrailViolation(
                    code="MEDIA_AMBIGUITY_ENFORCED", field=f"media_assessments.{a.media_id}",
                    detail="Media flagged ambiguous but a finding was supplied; finding removed.",
                ))
            by_id[a.media_id] = MediaAssessment(media_id=a.media_id, status="AMBIGUOUS", finding=None, flag=AMBIGUOUS_FLAG)
        else:
            by_id[a.media_id] = MediaAssessment(media_id=a.media_id, status="READABLE", finding=a.finding, flag=None)
    for mid in known:
        by_id.setdefault(mid, MediaAssessment(media_id=mid, status="AMBIGUOUS", finding=None, flag=AMBIGUOUS_FLAG))
    return [by_id[m.media_id] for m in media if m.media_id], violations


# --------------------------------------------------------------------------- #
# Haiiro deterministic triage rules (rules can escalate, never downgrade)
# --------------------------------------------------------------------------- #

RED_KEYWORDS: dict[str, str] = {
    "open mouth": "open-mouth breathing / respiratory distress",
    "open-mouth": "open-mouth breathing / respiratory distress",
    "gasping": "respiratory distress",
    "can't breathe": "respiratory distress",
    "seizure": "seizure activity",
    "convuls": "seizure activity",
    "collapse": "collapse",
    "unresponsive": "unresponsive",
    "lily": "possible lily ingestion (acute kidney injury risk)",
    "lilies": "possible lily ingestion (acute kidney injury risk)",
    "antifreeze": "possible ethylene glycol ingestion",
    "can't pee": "inability to urinate",
    "cannot pee": "inability to urinate",
    "can't urinate": "inability to urinate",
    "no urine": "inability to urinate",
    "dragging": "hind-limb paresis (possible aortic thromboembolism)",
    "back legs": "hind-limb weakness (possible aortic thromboembolism)",
    "blue gums": "cyanosis",
    "pale gums": "pale mucous membranes",
    "blood in vomit": "haematemesis",
    "vomiting blood": "haematemesis",
}
YELLOW_KEYWORDS: dict[str, str] = {
    "diarrh": "diarrhoea",
    "hiding": "behavioural change (hiding)",
    "weight loss": "weight loss",
    "drinking more": "increased water intake (polydipsia)",
    "increased thirst": "increased water intake (polydipsia)",
    "peeing more": "polyuria",
    "letharg": "lethargy",
    "constipat": "constipation",
}


@dataclass
class RuleEvaluation:
    level: TriageLevel
    hits: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)


def patient_comorbidities(record: PatientRecord) -> list[str]:
    reasons = list(record.active_problems)
    for analyte in ("creatinine", "sdma", "usg", "fpli"):
        lab = latest_verified(record, analyte)
        if lab is not None and lab.status in (LabStatus.HIGH, LabStatus.LOW):
            reasons.append(f"abnormal {lab.display_name} on {lab.collected_at}")
    return reasons


def evaluate_triage_rules(obs: HomeObservationInput, record: PatientRecord) -> RuleEvaluation:
    red: list[str] = []
    yellow: list[str] = []
    male = record.sex in (Sex.MALE_NEUTERED, Sex.MALE_INTACT)

    if obs.breathing_difficulty:
        red.append("breathing difficulty reported")
    if obs.straining_to_urinate:
        (red if male else yellow).append(
            "straining to urinate" + (" in a male cat (possible urethral obstruction)" if male else "")
        )
    if obs.hours_since_last_urination is not None and obs.hours_since_last_urination >= 24:
        red.append(f"no urination for {obs.hours_since_last_urination:g} h")
    if obs.vomiting_episodes_24h is not None:
        if obs.vomiting_episodes_24h >= 3:
            red.append(f"vomiting {obs.vomiting_episodes_24h}x in 24h")
        elif obs.vomiting_episodes_24h >= 1:
            yellow.append(f"vomiting {obs.vomiting_episodes_24h}x in 24h")
    if obs.hours_since_last_meal is not None:
        if obs.hours_since_last_meal >= 48:
            red.append(f"no food for {obs.hours_since_last_meal:g} h (hepatic lipidosis risk)")
        elif obs.hours_since_last_meal >= 24:
            yellow.append(f"no food for {obs.hours_since_last_meal:g} h")
    if obs.appetite == "none":
        yellow.append("not eating")
    elif obs.appetite == "reduced":
        yellow.append("reduced appetite")
    if obs.water_intake == "increased":
        yellow.append("increased water intake (polydipsia)")
    elif obs.water_intake == "decreased":
        yellow.append("decreased water intake")
    if obs.lethargy:
        yellow.append("lethargy")

    text = (" ".join(obs.symptoms) + " " + obs.free_text).lower()
    for kw, label in RED_KEYWORDS.items():
        if kw in text and label not in red:
            red.append(label)
    for kw, label in YELLOW_KEYWORDS.items():
        if kw in text and label not in yellow:
            yellow.append(label)

    hits = [f"RED: {h}" for h in red] + [f"YELLOW: {h}" for h in yellow]
    level = TriageLevel.RED if red else TriageLevel.YELLOW if yellow else TriageLevel.GREEN
    return RuleEvaluation(level=level, hits=hits, comorbidities=patient_comorbidities(record))


def merge_triage(
    rules: RuleEvaluation, llm_level: TriageLevel | None
) -> tuple[TriageLevel, list[GuardrailViolation]]:
    final = TriageLevel.max(rules.level, llm_level)
    violations: list[GuardrailViolation] = []
    if llm_level is not None and llm_level.severity < rules.level.severity:
        violations.append(GuardrailViolation(
            code="TRIAGE_DOWNGRADE_BLOCKED", field="triage_level",
            detail=f"LLM proposed {llm_level.value} but deterministic red-flag rules require "
                   f"{rules.level.value}; downgrade blocked.",
        ))
    return final, violations
