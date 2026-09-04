"""The traditional NLP approach: rules, dictionaries, spatial anchors and validators.

The workhorse here is **not** regex — it is *label-anchored spatial extraction*. For a
semi-structured document, the reliable signal is "the value sits to the right of, or
directly below, a label that names it". So the extractor finds label anchors, then reads
the value out of the geometry around them.

That geometry is why the layouts matter. Three of the four are handled well:

* ``inline`` -- value to the right of the anchor, on the same visual line
* ``below``  -- value on the next line, in the same horizontal column as the anchor
* ``grid``   -- identical to ``below`` once you use x-overlap rather than reading order

The fourth, ``prose``, has no anchors at all, and no amount of rule engineering fixes
that. This approach is expected to degrade sharply there, and it should: that gap is
precisely the evidence for where learned models earn their cost.

Type validators act as a second, independent source of truth. A 10-digit run that fails
the NPI checksum is rejected outright, which buys precision no amount of positional
guessing can. Where anchors fail entirely, the extractor falls back to typed scans with
document-level heuristics (the oldest date is the DOB, the largest amount is the total).
"""

from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..normalize import (
    ICD10_RE,
    is_valid_npi,
    normalize_amount,
    normalize_date,
    normalize_field,
    squash,
    strip_accents,
)
from ..schema import (
    DOC_TYPES,
    FIELD_KINDS,
    DocType,
    ExtractedField,
    ExtractionResult,
    FieldKind,
    OcrDocument,
    ServiceLine,
    Word,
    expected_fields,
)
from .nlp_labels import TRAIN_LABELS

# --------------------------------------------------------------------------------------
# Label dictionary. The extractor does not know which template it is looking at, so it
# carries every label variant it has ever seen, for every field, in both languages.
#
# Crucially, that vocabulary comes from :mod:`nlp_labels`, which was built by observing
# the *training* documents only. It is deliberately NOT imported from the generator:
# doing so would hand the rules engine foreknowledge of wording that appears solely on
# held-out templates, and inflate its score on precisely the documents the evaluation
# exists to measure. See that module's docstring for the full argument.
# --------------------------------------------------------------------------------------

def _build_label_dictionary() -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {
        field: list(variants) for field, variants in TRAIN_LABELS.items()
    }
    # Extra real-world variants from general domain knowledge of clinical forms. These
    # are the synonyms an engineer would add from experience rather than from the data,
    # and none of them are wording this corpus emits.
    merged["patient_name"] += ["Patient Name(Last, First)", "Insured's Name", "Pt Name"]
    merged["patient_dob"] += ["Birth", "DOB/Sex", "Date of Birth (MM/DD/YYYY)"]
    merged["referring_provider_npi"] += ["Referring NPI", "Rendering NPI"]
    merged["member_id"] += ["ID Number", "Insured's ID", "Policy Number"]
    merged["total_charge"] += ["Billed Amount", "Total Billed", "Charges"]
    merged["amount_paid"] += ["Insurance Paid", "Amount Allowed"]
    merged["document_reference"] += ["Claim Number", "Authorization #", "Order #"]
    return merged


LABEL_DICT = _build_label_dictionary()

#: Every label string in the dictionary, used to detect where one value stops and the
#: next label begins on a shared line.
ALL_LABEL_STRINGS = sorted(
    {v for variants in LABEL_DICT.values() for v in variants},
    key=len,
    reverse=True,
)

_CLEAN_LABEL = re.compile(r"[^a-z0-9]+")


def _label_key(text: str) -> str:
    return _CLEAN_LABEL.sub("", strip_accents(text).lower())


LABEL_KEYS: dict[str, list[str]] = {
    field: [_label_key(v) for v in variants] for field, variants in LABEL_DICT.items()
}
ALL_LABEL_KEYS = {_label_key(v) for v in ALL_LABEL_STRINGS}

# Must handle both comma-grouped and plain numbers. An earlier version required the
# thousands separator and so truncated "$1087.02" to "$108" -- a silent, systematic
# corruption of every un-grouped amount on the page.
AMOUNT_RE = re.compile(
    r"(?:USD\s*)?\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{2,4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+de\s+[a-zé]+\s+de\s+\d{4})\b",
    re.IGNORECASE,
)
CPT_STRICT_RE = re.compile(r"\b(\d{5}|[A-CEG-VX]\d{4})\b")
TEN_DIGIT_RE = re.compile(r"\b\d{10}\b")
MEMBER_ID_RE = re.compile(r"\b(?:[A-Z]{1,3}\d{6,12}[A-Z]?|\d{3}-\d{2}-\d{4}|\d{9,12})\b")
REF_RE = re.compile(r"\b(?:REF|RF|AUTH|PA|ORD|LAB|CLM|CL|EOB|RA)[-\s]?\d{5,9}\b", re.I)


@dataclass
class Line:
    """One visual line of the document, with its words and geometry."""

    text: str
    start: int
    end: int
    words: list[Word]
    y: float
    x0: float
    x1: float


def lines_of(doc: OcrDocument) -> list[Line]:
    """Reconstruct visual lines from the document text and word offsets."""
    lines: list[Line] = []
    cursor = 0
    for raw in doc.text.split("\n"):
        start, end = cursor, cursor + len(raw)
        words = [w for w in doc.words if w.start >= start and w.end <= end]
        if raw.strip():
            ys = [(w.y0 + w.y1) / 2 for w in words] or [0.0]
            lines.append(
                Line(
                    text=raw, start=start, end=end, words=words,
                    y=sum(ys) / len(ys),
                    x0=min((w.x0 for w in words), default=0.0),
                    x1=max((w.x1 for w in words), default=0.0),
                )
            )
        cursor = end + 1
    return lines


@dataclass
class Anchor:
    field: str
    line_index: int
    char_start: int
    char_end: int
    x0: float
    x1: float


def find_anchors(doc: OcrDocument, lines: list[Line]) -> list[Anchor]:
    """Locate every label occurrence in the document.

    Matching happens on a punctuation-stripped key so that ``D.O.B.``, ``DOB:`` and
    ``DOB`` are one anchor, and on word sequences so multi-word labels work.
    """
    anchors: list[Anchor] = []
    for li, line in enumerate(lines):
        words = line.words
        i = 0
        while i < len(words):
            # Longest match wins, and consumes the words it matched.
            #
            # Matching shortest-first was a real bug: "Patient" is a valid variant of
            # patient_name, so on a page reading "Patient Name: Linda Davis" the anchor
            # ended after "Patient" and the extracted value became "Name: Linda Davis".
            # The same defect turned "Insurance Carrier:" into a value of "Carrier: ...".
            matched = 0
            for span in range(min(4, len(words) - i), 0, -1):
                chunk = words[i:i + span]
                key = _label_key(" ".join(w.text for w in chunk))
                if not key or key not in ALL_LABEL_KEYS:
                    continue
                for field, keys in LABEL_KEYS.items():
                    if key in keys:
                        anchors.append(
                            Anchor(
                                field=field, line_index=li,
                                char_start=chunk[0].start, char_end=chunk[-1].end,
                                x0=chunk[0].x0, x1=chunk[-1].x1,
                            )
                        )
                matched = span
                break
            i += matched or 1
    return anchors


def _cut_at_embedded_label(value: str) -> str:
    """Truncate a candidate at the first label that starts a *different* field.

    Protects against a value swallowing its neighbour on a shared line, which is how
    ``payer_name`` ended up as ``"Member ID: 80628861504"``.
    """
    tokens = value.split()
    for start in range(len(tokens)):
        for span in range(min(4, len(tokens) - start), 0, -1):
            if _label_key(" ".join(tokens[start:start + span])) in ALL_LABEL_KEYS:
                return " ".join(tokens[:start]).strip(" :;,-")
    return value


def _strip_leading_punct(text: str) -> str:
    return text.lstrip(" :.-–\t")


def value_right_of(line: Line, anchor: Anchor, anchors_on_line: list[Anchor]) -> str | None:
    """Read the value printed to the right of an anchor on the same line.

    Stops at the next anchor on that line, which is what makes two-column ``inline``
    layouts ("Patient Name: X    DOB: Y") resolve correctly instead of swallowing the
    neighbouring field.
    """
    following = [
        a.char_start for a in anchors_on_line
        if a.char_start > anchor.char_end
    ]
    stop = min(following) if following else line.end
    raw = line.text[anchor.char_end - line.start:stop - line.start]
    value = squash(_strip_leading_punct(raw))
    return value or None


def value_below(
    lines: list[Line], anchor: Anchor, max_lookahead: int = 2
) -> str | None:
    """Read the value printed below an anchor, in the same horizontal column.

    Column alignment is the whole trick for ``below`` and ``grid`` layouts: the next
    line holds several fields' values side by side, and only x-overlap with the anchor
    tells you which one belongs to this label.
    """
    for offset in range(1, max_lookahead + 1):
        idx = anchor.line_index + offset
        if idx >= len(lines):
            return None
        line = lines[idx]
        # Words whose horizontal extent overlaps the anchor's column.
        picked = [
            w for w in line.words
            if w.x1 > anchor.x0 - 4 and w.x0 < anchor.x1 + 90
        ]
        if not picked:
            continue
        # Do not read another label as a value.
        text = squash(" ".join(w.text for w in picked))
        if _label_key(text) in ALL_LABEL_KEYS:
            continue
        value = _strip_leading_punct(text)
        if value:
            return value
    return None


def _type_score(field: str, value: str) -> float:
    """How well a candidate string matches the expected type of the field.

    This is what lets the extractor prefer a plausible value over a merely nearby one,
    and it is where the validators pay for themselves.
    """
    if not value:
        return 0.0
    kind = FIELD_KINDS.get(field)
    normalized = normalize_field(field, value)
    if kind is FieldKind.date:
        return 1.0 if normalize_date(value) else 0.0
    if kind is FieldKind.amount:
        return 1.0 if normalize_amount(value) else 0.0
    if field == "referring_provider_npi":
        return 1.0 if is_valid_npi(value) else 0.05
    if field == "diagnosis_code":
        return 1.0 if ICD10_RE.fullmatch(value.strip().upper()) else 0.1
    if field == "procedure_code":
        return 1.0 if CPT_STRICT_RE.fullmatch(value.strip().upper()) else 0.1
    if kind is FieldKind.reference:
        return 0.9 if re.search(r"\d", value) and len(value) <= 24 else 0.2
    if kind is FieldKind.person:
        words = value.split()
        if not 1 < len(words) <= 5:
            return 0.2
        return 0.9 if all(w[:1].isalpha() for w in words) else 0.3
    if kind is FieldKind.org:
        # A single-token organisation is not implausible — many payers are exactly one
        # word (WellCare, Aetna, Cigna, Humana, Medicare). Requiring two words scored
        # those at 0.3 and lost the field outright. A lone token still has to look like
        # a name rather than a group code ("GRP26112"), so it is accepted at a lower
        # confidence than a multi-word name.
        words = [w.strip(" :;,.-") for w in value.split()]
        words = [w for w in words if w]
        if not words or not any(c.isalpha() for c in value) or len(words) > 8:
            return 0.3 if words else 0.0
        if len(words) == 1:
            token = words[0]
            return 0.75 if token[:1].isupper() and token.isalpha() else 0.25
        return 0.9
    return 0.5 if normalized else 0.0


def _fallback_scan(doc: OcrDocument, field: str, taken: set[str]) -> str | None:
    """Typed document-wide scan, used when no anchor produced a usable value.

    The heuristics encode genuine domain knowledge rather than guesses: the oldest date
    on a clinical document is the date of birth, the largest money figure is the total
    charge, and the only checksum-valid 10-digit number is the NPI.
    """
    text = doc.text
    kind = FIELD_KINDS.get(field)

    if field == "referring_provider_npi":
        for candidate in TEN_DIGIT_RE.findall(text):
            if is_valid_npi(candidate):
                return candidate
        return None

    if field == "diagnosis_code":
        for match in ICD10_RE.findall(text):
            if not re.fullmatch(r"[A-Z]{2,}", match):
                return match
        return None

    if field == "procedure_code":
        candidates = [c for c in CPT_STRICT_RE.findall(text) if c not in taken]
        return candidates[0] if candidates else None

    if field == "document_reference":
        match = REF_RE.search(text)
        return match.group(0) if match else None

    if kind is FieldKind.date:
        found = []
        for match in DATE_RE.finditer(text):
            iso = normalize_date(match.group(0))
            if iso:
                found.append((iso, match.group(0)))
        if not found:
            return None
        found.sort()
        if field == "patient_dob":
            return found[0][1]                      # oldest date on the page
        if field == "document_date":
            return found[-1][1]                     # most recent
        return found[len(found) // 2][1]            # date of service: in between

    if kind is FieldKind.amount:
        amounts = []
        for match in AMOUNT_RE.finditer(text):
            value = normalize_amount(match.group(0))
            if value and float(value) > 0:
                amounts.append((float(value), match.group(0)))
        if not amounts:
            return None
        amounts.sort()
        if field == "total_charge":
            return amounts[-1][1]
        if field == "patient_responsibility":
            return amounts[0][1]
        return amounts[len(amounts) // 2][1]

    if field == "member_id":
        match = MEMBER_ID_RE.search(text)
        return match.group(0) if match else None

    return None


class NlpExtractor:
    """Rules + dictionaries + validators, with an optional TF-IDF doc-type classifier."""

    name = "nlp"

    def __init__(self, model_dir: str | Path | None = "models/nlp"):
        self.model_dir = Path(model_dir) if model_dir else None
        self.doctype_clf = None
        if self.model_dir and (self.model_dir / "doctype.pkl").exists():
            with (self.model_dir / "doctype.pkl").open("rb") as fh:
                self.doctype_clf = pickle.load(fh)

    # -- document type -----------------------------------------------------------
    def fit_doctype(self, texts: list[str], labels: list[str]) -> None:
        """Train the TF-IDF + LinearSVC document-type classifier."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC

        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                    max_features=60000,
                )),
                # Char n-grams make the classifier robust to OCR character errors,
                # which is the whole point on the scanned condition.
                ("char", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                    sublinear_tf=True, max_features=80000,
                )),
            ])),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
        ])
        pipeline.fit(texts, labels)
        self.doctype_clf = pipeline
        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            with (self.model_dir / "doctype.pkl").open("wb") as fh:
                pickle.dump(pipeline, fh)

    def predict_doctype(self, text: str) -> tuple[DocType, float]:
        if self.doctype_clf is not None:
            label = self.doctype_clf.predict([text])[0]
            try:
                confidence = float(self.doctype_clf.predict_proba([text]).max())
            except Exception:
                confidence = 0.0
            return DocType(label), confidence
        return self._doctype_by_keywords(text)

    @staticmethod
    def _doctype_by_keywords(text: str) -> tuple[DocType, float]:
        """Keyword fallback so the approach works before anything is trained."""
        lowered = strip_accents(text).lower()
        rules = [
            (DocType.remittance_advice,
             ["explanation of benefits", "remittance", "payment advice", "plan paid",
              "explicacion de beneficios"]),
            (DocType.insurance_claim,
             ["claim form", "cms-1500", "health insurance claim", "reclamacion"]),
            (DocType.prior_auth_request,
             ["prior authorization", "pre-authorization", "autorizacion previa"]),
            (DocType.lab_order,
             ["laboratory order", "lab requisition", "diagnostic order",
              "orden de laboratorio"]),
            (DocType.patient_intake_form,
             ["intake form", "registration", "patient information", "admision"]),
            (DocType.referral_letter,
             ["referral", "referring", "remision", "dear colleague"]),
        ]
        for doc_type, keywords in rules:
            if any(k in lowered for k in keywords):
                return doc_type, 0.6
        return DocType.referral_letter, 0.2

    # -- fields ------------------------------------------------------------------
    def extract(self, doc: OcrDocument) -> ExtractionResult:
        started = time.perf_counter()
        lines = lines_of(doc)
        anchors = find_anchors(doc, lines)
        by_line: dict[int, list[Anchor]] = {}
        for anchor in anchors:
            by_line.setdefault(anchor.line_index, []).append(anchor)

        doc_type, confidence = self.predict_doctype(doc.text)
        wanted = expected_fields(doc_type) or set(FIELD_KINDS)

        fields: dict[str, ExtractedField | None] = {}
        taken: set[str] = set()

        for field in wanted:
            best: tuple[float, str, int | None] | None = None
            for anchor in [a for a in anchors if a.field == field]:
                line = lines[anchor.line_index]
                for candidate, where in (
                    (value_right_of(line, anchor, by_line[anchor.line_index]), "right"),
                    (value_below(lines, anchor), "below"),
                ):
                    if not candidate:
                        continue
                    candidate = self._trim_candidate(field, candidate)
                    if not candidate:
                        continue
                    score = _type_score(field, candidate)
                    # Prefer same-line values slightly: an inline template is the more
                    # common case, and a below-match on an inline page is usually the
                    # next row's value.
                    score += 0.05 if where == "right" else 0.0
                    if best is None or score > best[0]:
                        best = (score, candidate, anchor.char_end)

            value = best[1] if best and best[0] >= 0.5 else None
            if value is None:
                value = self._trim_candidate(field, _fallback_scan(doc, field, taken) or "")
            if value:
                start = doc.text.find(value)
                fields[field] = ExtractedField(
                    value=value,
                    raw=value,
                    start=start if start >= 0 else None,
                    end=start + len(value) if start >= 0 else None,
                    confidence=round(best[0], 3) if best else 0.35,
                )
                taken.add(value)
            else:
                fields[field] = None

        result = ExtractionResult(
            doc_id=doc.doc_id,
            approach=self.name,
            doc_type=doc_type,
            doc_type_confidence=confidence,
            fields=fields,
            service_lines=self._service_lines(doc, lines),
            org_roles=self._org_roles(doc, fields),
            latency_ms=(time.perf_counter() - started) * 1000,
            meta={"anchors": len(anchors), "condition": doc.condition},
        )
        return result

    @staticmethod
    def _trim_candidate(field: str, value: str) -> str | None:
        """Cut a raw candidate down to the part that plausibly *is* the field."""
        if not value:
            return None
        value = squash(value).strip(" :;,-")
        if not value:
            return None
        kind = FIELD_KINDS.get(field)
        if kind is FieldKind.date:
            match = DATE_RE.search(value)
            return match.group(0) if match else (value if normalize_date(value) else None)
        if kind is FieldKind.amount:
            match = AMOUNT_RE.search(value)
            return match.group(0).strip() if match else None
        if field == "referring_provider_npi":
            for candidate in TEN_DIGIT_RE.findall(value):
                if is_valid_npi(candidate):
                    return candidate
            match = TEN_DIGIT_RE.search(value)
            return match.group(0) if match else None
        if field == "diagnosis_code":
            match = ICD10_RE.search(value.upper())
            return match.group(0) if match else None
        if field == "procedure_code":
            match = CPT_STRICT_RE.search(value.upper())
            return match.group(0) if match else None
        # Names and orgs: stop before a neighbouring field's label bleeds in.
        return _cut_at_embedded_label(value)[:80] or None

    def _service_lines(self, doc: OcrDocument, lines: list[Line]) -> list[ServiceLine]:
        """Recover the service table by finding rows that start with a CPT code."""
        out: list[ServiceLine] = []
        for line in lines:
            match = CPT_STRICT_RE.match(line.text.strip())
            if not match:
                continue
            amounts = [m.group(0) for m in AMOUNT_RE.finditer(line.text)]
            amounts = [a for a in amounts if normalize_amount(a)]
            date_match = DATE_RE.search(line.text)
            units = None
            unit_match = re.search(r"\b([1-9])\b(?=\s*(?:\$|USD|\d+[.,]))", line.text)
            if unit_match:
                units = unit_match.group(1)
            out.append(
                ServiceLine(
                    procedure_code=match.group(0),
                    date_of_service=date_match.group(0) if date_match else None,
                    units=units,
                    charge=amounts[0] if amounts else None,
                    paid=amounts[1] if len(amounts) > 1 else None,
                )
            )
        return out

    @staticmethod
    def _org_roles(doc: OcrDocument, fields: dict) -> dict[str, str | None]:
        """Assign organisations to roles.

        Deliberately simple: the servicing facility and payer come from their own
        anchored fields, and the referring organisation is guessed as the letterhead —
        the first substantial line of the page. That last heuristic is the weak point,
        and the error analysis is expected to show it failing whenever the letterhead
        belongs to the payer or the facility instead.
        """
        first_line = next(
            (ln for ln in doc.text.split("\n") if len(ln.strip()) > 8), ""
        ).strip()
        servicing = fields.get("servicing_facility")
        payer = fields.get("payer_name")
        return {
            "referring_org": first_line or None,
            "servicing_org": servicing.value if servicing else None,
            "payer_org": payer.value if payer else None,
        }
