"""Thin, testable wrapper around the Google GenAI SDK (``google-genai``).

* ``gemini-2.5-flash`` - Haiiro caregiver triage (low latency).
* ``gemini-2.5-pro``   - Michelin clinical cross-examination and lab OCR (deep reasoning).

All calls use ``temperature=0`` and JSON structured output bound to a Pydantic
schema. The parsed object is returned to the caller *untrusted*: callers must
pass it through ``services.verification`` before use.

Credentials are read by the SDK from the environment (``GEMINI_API_KEY`` /
``GOOGLE_API_KEY``, or Vertex AI via ``GOOGLE_GENAI_USE_VERTEXAI`` +
``GOOGLE_CLOUD_PROJECT`` + ``GOOGLE_CLOUD_LOCATION``). When no credentials are
configured the client reports ``available == False`` and the agents fall back
to their deterministic engines - they never invent output.
"""

import logging
import os
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("michelin.gemini")

TRIAGE_MODEL = os.getenv("MICHELIN_TRIAGE_MODEL", "gemini-2.5-flash")
CROSS_EXAM_MODEL = os.getenv("MICHELIN_CROSS_EXAM_MODEL", "gemini-2.5-pro")
EXTRACTION_MODEL = os.getenv("MICHELIN_EXTRACTION_MODEL", "gemini-2.5-pro")

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """No Gemini credentials configured (or client disabled)."""


class LLMResponseError(RuntimeError):
    """Gemini call failed or returned output that does not match the schema."""


def _credentials_configured() -> bool:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return True
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


class GeminiClient:
    """Structured-output gateway to Gemini.

    ``sdk_client`` may be any object exposing ``models.generate_content(...)``
    (a real ``google.genai.Client`` or a test double).
    """

    def __init__(self, sdk_client: Any | None = None, *, enabled: bool | None = None):
        if sdk_client is None and (enabled if enabled is not None else _credentials_configured()):
            from google import genai  # imported lazily so the API boots without credentials

            sdk_client = genai.Client()
        self._client = sdk_client

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate_structured(
        self, *, model: str, system_instruction: str, contents: list[Any], schema: type[T]
    ) -> T:
        if self._client is None:
            raise LLMUnavailableError("Gemini client is not configured")

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        try:
            response = self._client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:  # network, quota, safety block, auth ...
            logger.exception("Gemini call to %s failed", model)
            raise LLMResponseError(f"Gemini call to {model} failed: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise LLMResponseError(f"Gemini model {model} returned an empty response")
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMResponseError(f"Gemini model {model} returned schema-invalid JSON: {exc}") from exc


def text_part(text: str) -> Any:
    from google.genai import types

    return types.Part.from_text(text=text)


def bytes_part(data: bytes, mime_type: str) -> Any:
    from google.genai import types

    return types.Part.from_bytes(data=data, mime_type=mime_type)
