"""Core data contracts shared by every approach, the evaluator, the CLI and the API.

Every extraction approach implements ``extract(doc: OcrDocument) -> ExtractionResult``
so that downstream code never needs to know which system produced a result. Keeping
this contract narrow is what lets the evaluator score four very different systems
(rules, a fine-tuned encoder, a local LLM, a frontier LLM) with one code path.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DocType(str, Enum):
    """The six healthcare intake/referral document classes."""

    referral_letter = "referral_letter"
    prior_auth_request = "prior_auth_request"
    lab_order = "lab_order"
    insurance_claim = "insurance_claim"
    remittance_advice = "remittance_advice"
    patient_intake_form = "patient_intake_form"
    unknown = "unknown"


DOC_TYPES: list[DocType] = [d for d in DocType if d is not DocType.unknown]


class FieldKind(str, Enum):
    """What a field *is*, which decides how it is normalised, validated and scored."""

    person = "person"
    org = "org"
    date = "date"
    amount = "amount"
    reference = "reference"
    code = "code"


#: The 15 extraction fields, mapped to the normaliser/validator family they belong to.
FIELD_KINDS: dict[str, FieldKind] = {
    "patient_name": FieldKind.person,
    "referring_provider_name": FieldKind.person,
    "patient_dob": FieldKind.date,
    "date_of_service": FieldKind.date,
    "document_date": FieldKind.date,
    "total_charge": FieldKind.amount,
    "amount_paid": FieldKind.amount,
    "patient_responsibility": FieldKind.amount,
    "member_id": FieldKind.reference,
    "referring_provider_npi": FieldKind.reference,
    "document_reference": FieldKind.reference,
    "servicing_facility": FieldKind.org,
    "payer_name": FieldKind.org,
    "diagnosis_code": FieldKind.code,
    "procedure_code": FieldKind.code,
}

FIELD_NAMES: list[str] = list(FIELD_KINDS)

#: BIO label space for the token classifier: O + B-/I- per field.
BIO_LABELS: list[str] = ["O"] + [
    f"{prefix}-{name}" for name in FIELD_NAMES for prefix in ("B", "I")
]
LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}


#: Which fields a given document type is actually expected to carry.
#:
#: This matters for fair scoring: a lab order has no ``amount_paid``, so an approach
#: must not be punished for correctly leaving it empty, and must not be rewarded for
#: inventing one. The evaluator scores each document only over its type's field set.
FIELDS_BY_DOCTYPE: dict[DocType, set[str]] = {
    DocType.referral_letter: {
        "patient_name", "patient_dob", "referring_provider_name",
        "referring_provider_npi", "servicing_facility", "payer_name",
        "member_id", "document_reference", "document_date", "diagnosis_code",
    },
    DocType.prior_auth_request: {
        "patient_name", "patient_dob", "referring_provider_name",
        "referring_provider_npi", "servicing_facility", "payer_name",
        "member_id", "document_reference", "document_date", "date_of_service",
        "diagnosis_code", "procedure_code", "total_charge",
    },
    DocType.lab_order: {
        "patient_name", "patient_dob", "referring_provider_name",
        "referring_provider_npi", "servicing_facility", "document_reference",
        "document_date", "date_of_service", "diagnosis_code", "procedure_code",
    },
    DocType.insurance_claim: {
        "patient_name", "patient_dob", "referring_provider_name",
        "referring_provider_npi", "servicing_facility", "payer_name",
        "member_id", "document_reference", "document_date", "date_of_service",
        "diagnosis_code", "procedure_code", "total_charge",
    },
    DocType.remittance_advice: {
        "patient_name", "member_id", "payer_name", "servicing_facility",
        "document_reference", "document_date", "date_of_service",
        "procedure_code", "total_charge", "amount_paid", "patient_responsibility",
    },
    DocType.patient_intake_form: {
        "patient_name", "patient_dob", "referring_provider_name",
        "payer_name", "member_id", "document_date",
    },
}

#: Document types that carry a service-line table (the relationship task).
DOCTYPES_WITH_SERVICE_LINES: set[DocType] = {
    DocType.insurance_claim,
    DocType.remittance_advice,
}

#: The organisation-role relationship task: which org on the page plays which role.
ORG_ROLES: list[str] = ["referring_org", "servicing_org", "payer_org"]


class Word(BaseModel):
    """One OCR token, carrying both its char offset into the page text and its box.

    The char offsets let the token classifier and the span aligner work on plain text;
    the box lets the NLP approach do spatial label-anchored lookup ("value to the right
    of the 'Patient Name:' anchor"). Both views are needed, so both are stored.
    """

    text: str
    start: int
    end: int
    page: int = 0
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


class OcrDocument(BaseModel):
    """A document after OCR/text extraction — the single input to every approach."""

    doc_id: str
    text: str
    words: list[Word] = Field(default_factory=list)
    page_count: int = 1
    condition: Literal["clean", "scanned"] = "clean"
    lang: str = "en"
    source_path: str | None = None

    def line_starts(self) -> list[int]:
        """Char offsets of each line start — used by the label-anchored extractor."""
        offsets, pos = [0], 0
        for line in self.text.split("\n")[:-1]:
            pos += len(line) + 1
            offsets.append(pos)
        return offsets


class ExtractedField(BaseModel):
    """One extracted field value, with provenance back into the source text.

    ``start``/``end`` are None for generative approaches that return a value without
    telling us where it came from. That absence is meaningful: the evaluator uses it,
    together with a substring check, to flag hallucinated values.
    """

    value: str
    raw: str | None = None
    start: int | None = None
    end: int | None = None
    confidence: float | None = None


class ServiceLine(BaseModel):
    """One row of a claim/remittance service table."""

    procedure_code: str | None = None
    date_of_service: str | None = None
    units: str | None = None
    charge: str | None = None
    paid: str | None = None


class ExtractionResult(BaseModel):
    """The uniform output of every approach."""

    doc_id: str
    approach: str
    doc_type: DocType = DocType.unknown
    doc_type_confidence: float | None = None
    fields: dict[str, ExtractedField | None] = Field(default_factory=dict)
    service_lines: list[ServiceLine] = Field(default_factory=list)
    org_roles: dict[str, str | None] = Field(default_factory=dict)
    latency_ms: float | None = None
    meta: dict = Field(default_factory=dict)

    def value(self, field: str) -> str | None:
        """Convenience accessor returning the plain value or None."""
        f = self.fields.get(field)
        return f.value if f else None


class GoldAnnotation(BaseModel):
    """A human-verified test-set record.

    ``verified`` and ``corrections`` are what let the report state, with evidence,
    that manual verification actually happened and how much the pre-annotation was
    wrong — rather than merely asserting the set is gold.
    """

    doc_id: str
    doc_type: DocType
    fields: dict[str, str | None] = Field(default_factory=dict)
    service_lines: list[ServiceLine] = Field(default_factory=list)
    org_roles: dict[str, str | None] = Field(default_factory=dict)
    provenance: Literal["synthetic_unseen_template", "real_document"] = (
        "synthetic_unseen_template"
    )
    template_id: str | None = None
    condition: Literal["clean", "scanned"] = "clean"
    lang: str = "en"
    verified: bool = False
    corrections: list[str] = Field(default_factory=list)


def expected_fields(doc_type: DocType) -> set[str]:
    """Fields a document of this type should carry; empty set for unknown types."""
    return FIELDS_BY_DOCTYPE.get(doc_type, set())
