"""Central Patient Store - the single source of truth (SSOT) shared by Michelin and Haiiro.

Design goals:
* **No state drift** - both agents read and write through this one store. Every
  mutation is atomic (under a lock), re-validated against the strict
  ``PatientRecord`` schema, and bumps ``record_version``.
* **Optimistic concurrency** - writers may pass ``expected_version``; a stale
  writer gets ``VersionConflictError`` instead of silently overwriting.
* **Defensive copies** - callers always receive deep copies so they cannot
  mutate shared state outside a transaction.

The prototype keeps records in memory, seeded from ``data/mock_records.json``.
Swap ``PatientStore`` for a database-backed implementation with the same
interface for production persistence.
"""

import json
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from michelin_core.schemas.patient import (
    Actor,
    Escalation,
    HomeObservation,
    LabResult,
    PatientRecord,
    TriageEvent,
    utcnow,
)

DEFAULT_SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_records.json"


class PatientNotFoundError(KeyError):
    def __init__(self, patient_id: str):
        super().__init__(patient_id)
        self.patient_id = patient_id


class VersionConflictError(RuntimeError):
    def __init__(self, patient_id: str, expected: int, actual: int):
        super().__init__(f"{patient_id}: expected record_version {expected}, current is {actual}")
        self.patient_id, self.expected, self.actual = patient_id, expected, actual


class PatientStore:
    def __init__(self, records: Iterable[PatientRecord] = ()):
        self._lock = threading.RLock()
        self._records: dict[str, PatientRecord] = {r.patient_id: r for r in records}

    @classmethod
    def from_json(cls, path: Path | str = DEFAULT_SEED_PATH) -> "PatientStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(PatientRecord.model_validate(p) for p in payload["patients"])

    # ----------------------------------------------------------------- reads
    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._records)

    def get(self, patient_id: str) -> PatientRecord:
        with self._lock:
            record = self._records.get(patient_id)
            if record is None:
                raise PatientNotFoundError(patient_id)
            return record.model_copy(deep=True)

    # ---------------------------------------------------------------- writes
    def _mutate(
        self, patient_id: str, actor: Actor, fn: Callable[[dict], None], expected_version: int | None = None
    ) -> PatientRecord:
        with self._lock:
            current = self._records.get(patient_id)
            if current is None:
                raise PatientNotFoundError(patient_id)
            if expected_version is not None and expected_version != current.record_version:
                raise VersionConflictError(patient_id, expected_version, current.record_version)
            data = current.model_dump(mode="python")
            fn(data)
            data["record_version"] = current.record_version + 1
            data["updated_at"] = utcnow()
            data["last_updated_by"] = actor
            updated = PatientRecord.model_validate(data)  # full re-validation: SSOT can never hold invalid state
            self._records[patient_id] = updated
            return updated.model_copy(deep=True)

    def commit_haiiro_observation(
        self, patient_id: str, observation: HomeObservation, triage_event: TriageEvent,
        escalation: Escalation | None,
    ) -> PatientRecord:
        def apply(data: dict) -> None:
            # Inline media bytes are never persisted in the SSOT.
            obs = observation.model_dump(mode="python")
            for m in obs["media"]:
                m["data_base64"] = None
            data["home_observations"].append(obs)
            data["triage_events"].append(triage_event.model_dump(mode="python"))
            if escalation is not None:
                data["escalations"].append(escalation.model_dump(mode="python"))

        return self._mutate(patient_id, "haiiro", apply)

    def add_labs(
        self, patient_id: str, labs: list[LabResult], actor: Actor = "michelin", expected_version: int | None = None
    ) -> PatientRecord:
        def apply(data: dict) -> None:
            data["labs"].extend(lab.model_dump(mode="python") for lab in labs)

        return self._mutate(patient_id, actor, apply, expected_version)
