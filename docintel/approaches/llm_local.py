"""The offline general-purpose LLM baseline: Qwen2.5-3B-Instruct via Ollama.

This tier plays two roles:

1. the **baseline** the other approaches are measured against, and
2. the **teacher** that produces silver labels for the small model.

Structured output is enforced with a JSON schema rather than by asking politely and
parsing whatever comes back. Ollama constrains decoding to the schema, so the response
is always shaped correctly and the only remaining failure mode is a wrong *value* —
which is the failure mode we actually want to measure.

An honest caveat that belongs in the report rather than buried here: on 7.4GB of RAM
the largest workable local model is ~3B at 4-bit. A 3B instruct model is not a stand-in
for a frontier LLM, so :mod:`docintel.approaches.llm_frontier` supplies a separate
reference tier. Reading this tier's numbers as "what an LLM can do" would be wrong.
"""

from __future__ import annotations

import json
import time
from typing import Any

import urllib.error
import urllib.request

from ..schema import (
    DOC_TYPES,
    FIELD_NAMES,
    DocType,
    ExtractedField,
    ExtractionResult,
    OcrDocument,
    ServiceLine,
)

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b-instruct"

#: Published pricing is not applicable to a local model; electricity is the only
#: marginal cost and it rounds to zero per document. Recorded explicitly so the cost
#: column in the report has a defensible number rather than a blank.
COST_PER_DOC_USD = 0.0

FIELD_DESCRIPTIONS: dict[str, str] = {
    "patient_name": "Full name of the patient the document is about",
    "patient_dob": "Patient's date of birth, exactly as printed",
    "referring_provider_name": "Name of the referring/ordering clinician, as printed",
    "referring_provider_npi": "The 10-digit National Provider Identifier",
    "servicing_facility": "Organisation that performs or performed the service",
    "payer_name": "Insurance company or health plan",
    "member_id": "Patient's insurance member/policy/subscriber identifier",
    "document_reference": "This document's own reference/claim/authorisation/order number",
    "document_date": "Date this document was issued, exactly as printed",
    "date_of_service": "Date the service was or will be performed, exactly as printed",
    "diagnosis_code": "ICD-10 diagnosis code",
    "procedure_code": "CPT or HCPCS procedure code",
    "total_charge": "Total amount billed, exactly as printed",
    "amount_paid": "Amount paid by the plan, exactly as printed",
    "patient_responsibility": "Amount the patient owes, exactly as printed",
}


def build_schema() -> dict:
    """JSON schema Ollama constrains generation to."""
    properties: dict[str, Any] = {
        "document_type": {"type": "string", "enum": [d.value for d in DOC_TYPES]},
    }
    for name in FIELD_NAMES:
        properties[name] = {
            "type": ["string", "null"],
            "description": FIELD_DESCRIPTIONS.get(name, name),
        }
    line_properties = {
        "procedure_code": {"type": ["string", "null"]},
        "date_of_service": {"type": ["string", "null"]},
        "units": {"type": ["string", "null"]},
        "charge": {"type": ["string", "null"]},
        "paid": {"type": ["string", "null"]},
    }
    properties["service_lines"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": line_properties,
            "required": list(line_properties),
        },
    }
    # Every key must be listed in `required`, not merely in `properties`.
    #
    # JSON Schema treats `properties` as permissive: a key that is not required may
    # simply be omitted. Ollama compiles the schema into a decoding grammar faithfully,
    # so with only `document_type` required the model was free to emit five keys and
    # stop -- which it did, discarding two thirds of the fields and making recall look
    # like a model-quality problem when it was a schema bug. Requiring every key forces
    # an explicit decision (a value or an explicit null) for each field.
    return {
        "type": "object",
        "properties": properties,
        "required": ["document_type", *FIELD_NAMES, "service_lines"],
    }


SYSTEM_PROMPT = (
    "You extract structured data from healthcare documents. "
    "Copy values EXACTLY as they appear in the document - do not reformat dates, "
    "amounts, or identifiers. If a field is not present, return null. "
    "Never guess or invent a value that is not written in the document."
)


def build_prompt(text: str, max_chars: int = 6000) -> str:
    if len(text) > max_chars:
        # Keep the head and tail: headers carry identity, footers carry totals.
        text = text[: max_chars // 2] + "\n...\n" + text[-max_chars // 2 :]
    return (
        "Extract the fields below from this healthcare document.\n\n"
        "Rules:\n"
        "- Copy each value character-for-character as printed.\n"
        "- Use null for anything not present.\n"
        "- document_type must be one of: "
        + ", ".join(d.value for d in DOC_TYPES)
        + ".\n\n=== DOCUMENT ===\n"
        + text
        + "\n=== END ===\n"
    )


def ollama_available(url: str = OLLAMA_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def list_models(url: str = OLLAMA_URL) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as response:
            data = json.loads(response.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


class LocalLlmExtractor:
    """Schema-constrained extraction with a local instruct model."""

    name = "llm_local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        timeout: int = 180,
    ):
        self.model = model
        self.url = url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.schema = build_schema()

    def _call(self, prompt: str) -> tuple[dict, dict]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "format": self.schema,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "seed": 42,  # requested determinism; see the report's determinism column
            },
        }
        request = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        note = None
        try:
            parsed = json.loads(body.get("response", "{}"))
        except json.JSONDecodeError:
            # Schema-constrained decoding makes this rare, but under memory
            # pressure Ollama can return a truncated body. Silently becoming an
            # empty result would be indistinguishable from the model reading the
            # page and finding nothing.
            parsed = {}
            note = "the model returned unparseable JSON; this is a failure, not an empty document"
        meta = {
            "note": note,
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
            "total_duration_ms": (body.get("total_duration") or 0) / 1e6,
        }
        return parsed, meta

    def extract(self, doc: OcrDocument) -> ExtractionResult:
        started = time.perf_counter()
        try:
            parsed, meta = self._call(build_prompt(doc.text))
            error = None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            parsed, meta, error = {}, {}, str(exc)

        raw_type = (parsed.get("document_type") or "").strip()
        try:
            doc_type = DocType(raw_type)
        except ValueError:
            doc_type = DocType.unknown

        fields: dict[str, ExtractedField | None] = {}
        for name in FIELD_NAMES:
            value = parsed.get(name)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip()
                position = doc.text.find(cleaned)
                fields[name] = ExtractedField(
                    value=cleaned,
                    raw=cleaned,
                    start=position if position >= 0 else None,
                    end=position + len(cleaned) if position >= 0 else None,
                )
            else:
                fields[name] = None

        service_lines = []
        for row in parsed.get("service_lines") or []:
            if isinstance(row, dict):
                service_lines.append(
                    ServiceLine(
                        procedure_code=row.get("procedure_code"),
                        date_of_service=row.get("date_of_service"),
                        units=row.get("units"),
                        charge=row.get("charge"),
                        paid=row.get("paid"),
                    )
                )

        return ExtractionResult(
            doc_id=doc.doc_id,
            approach=self.name,
            doc_type=doc_type,
            doc_type_confidence=None,
            fields=fields,
            service_lines=service_lines,
            org_roles={
                "referring_org": None,
                "servicing_org": (fields.get("servicing_facility") or ExtractedField(value="")).value or None,
                "payer_org": (fields.get("payer_name") or ExtractedField(value="")).value or None,
            },
            latency_ms=(time.perf_counter() - started) * 1000,
            meta={"model": self.model, "error": error, "condition": doc.condition, **meta},
        )
