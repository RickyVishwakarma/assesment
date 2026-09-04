"""Aligns field *values* to character *spans* in the document text.

This is the load-bearing, unglamorous step between the LLM teacher and the small-model
student. The teacher returns ``{"patient_name": "Jane Doe"}``; a token classifier needs
to know that "Jane Doe" occupies characters 142-150. Values that cannot be located
cannot become training labels at all, so the alignment rate directly caps how much
supervision the student receives — which is why this module reports its own hit rate
rather than silently dropping what it cannot place.

The cascade runs cheapest-and-strictest first:

1. **exact** substring match
2. **whitespace-flexible** match, so a value broken across a line break still aligns
3. **compact** match ignoring all punctuation and case, which survives OCR mangling
   separators (``003-32-2453`` vs ``003 32 2453``)
4. **fuzzy** match above a similarity threshold, for genuine OCR character errors

Anything that survives all four is recorded as unalignable, with its reason.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as _field

from .schema import FIELD_KINDS, FieldKind, OcrDocument

try:  # rapidfuzz gives us alignment offsets directly and is much faster
    from rapidfuzz import fuzz as _rf_fuzz

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover - exercised only when rapidfuzz is absent
    import difflib

    _HAS_RAPIDFUZZ = False


#: Similarity below which a fuzzy match is rejected as "not really this value".
FUZZY_THRESHOLD = 88.0


@dataclass
class Alignment:
    """Where a value was found, and how hard we had to work to find it."""

    field: str
    value: str
    start: int | None = None
    end: int | None = None
    method: str = "none"          # exact | whitespace | compact | fuzzy | none
    score: float = 0.0
    occurrences: int = 0          # >1 means the value is ambiguous on the page

    @property
    def ok(self) -> bool:
        return self.start is not None


@dataclass
class AlignmentReport:
    """Aggregate alignment statistics, reported rather than hidden."""

    total: int = 0
    aligned: int = 0
    by_method: dict[str, int] = _field(default_factory=dict)
    by_field_total: dict[str, int] = _field(default_factory=dict)
    by_field_aligned: dict[str, int] = _field(default_factory=dict)
    ambiguous: int = 0

    def add(self, a: Alignment) -> None:
        self.total += 1
        self.by_field_total[a.field] = self.by_field_total.get(a.field, 0) + 1
        self.by_method[a.method] = self.by_method.get(a.method, 0) + 1
        if a.ok:
            self.aligned += 1
            self.by_field_aligned[a.field] = self.by_field_aligned.get(a.field, 0) + 1
        if a.occurrences > 1:
            self.ambiguous += 1

    @property
    def rate(self) -> float:
        return self.aligned / self.total if self.total else 0.0

    def field_rates(self) -> dict[str, float]:
        return {
            f: self.by_field_aligned.get(f, 0) / n
            for f, n in sorted(self.by_field_total.items())
            if n
        }


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _compact(text: str) -> tuple[str, list[int]]:
    """Lowercased alphanumeric-only view of ``text`` plus a map back to real offsets."""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(_strip_accents(text)):
        if ch.isalnum():
            chars.append(ch.lower())
            index_map.append(i)
    return "".join(chars), index_map


def _exact(text: str, value: str) -> tuple[int, int, int] | None:
    """Exact substring, returning span and total occurrence count."""
    idx = text.find(value)
    if idx < 0:
        return None
    return idx, idx + len(value), text.count(value)


def _whitespace_flexible(text: str, value: str) -> tuple[int, int, int] | None:
    """Match where any whitespace run in the value may be any whitespace run in text.

    Necessary because the page text carries newlines wherever the layout wrapped, so a
    value like ``"Cedar Park Imaging Center"`` may straddle a line break.
    """
    tokens = [re.escape(t) for t in value.split() if t]
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(tokens), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[0].start(), matches[0].end(), len(matches)


def _compact_match(text: str, value: str) -> tuple[int, int, int] | None:
    """Match ignoring case, accents and every non-alphanumeric character."""
    hay, index_map = _compact(text)
    needle, _ = _compact(value)
    if not needle:
        return None
    idx = hay.find(needle)
    if idx < 0:
        return None
    start = index_map[idx]
    end = index_map[idx + len(needle) - 1] + 1
    return start, end, hay.count(needle)


def _fuzzy(text: str, value: str) -> tuple[int, int, float] | None:
    """Best approximate match, for values corrupted by OCR character errors."""
    hay, index_map = _compact(text)
    needle, _ = _compact(value)
    if len(needle) < 4 or not hay:
        return None

    if _HAS_RAPIDFUZZ:
        alignment = _rf_fuzz.partial_ratio_alignment(needle, hay, score_cutoff=FUZZY_THRESHOLD)
        if alignment is None:
            return None
        lo, hi = alignment.dest_start, alignment.dest_end
        score = _rf_fuzz.ratio(needle, hay[lo:hi])
        if score < FUZZY_THRESHOLD or hi <= lo:
            return None
    else:  # pragma: no cover - fallback path
        best, lo, hi = 0.0, 0, 0
        window = len(needle)
        for start in range(0, max(1, len(hay) - window + 1)):
            for width in (window, int(window * 1.15) + 1):
                chunk = hay[start:start + width]
                if not chunk:
                    continue
                score = difflib.SequenceMatcher(None, needle, chunk).ratio() * 100
                if score > best:
                    best, lo, hi = score, start, start + len(chunk)
        if best < FUZZY_THRESHOLD:
            return None
        score = best

    if hi > len(index_map):
        hi = len(index_map)
    if lo >= hi:
        return None
    return index_map[lo], index_map[hi - 1] + 1, float(score)


def find_span(text: str, value: str, field: str | None = None) -> Alignment:
    """Locate ``value`` in ``text`` using the escalating cascade."""
    fname = field or "?"
    alignment = Alignment(field=fname, value=value)
    if not value or not text:
        return alignment

    for method, fn in (
        ("exact", _exact),
        ("whitespace", _whitespace_flexible),
        ("compact", _compact_match),
    ):
        hit = fn(text, value)
        if hit:
            start, end, count = hit
            return Alignment(fname, value, start, end, method, 100.0, count)

    hit = _fuzzy(text, value)
    if hit:
        start, end, score = hit
        return Alignment(fname, value, start, end, "fuzzy", score, 1)

    return alignment


def _resolve_overlaps(alignments: list[Alignment]) -> list[Alignment]:
    """Drop lower-confidence spans that collide with higher-confidence ones.

    Two fields must not claim the same characters — a token can only carry one BIO
    label. Exact matches win over fuzzy ones; ties break toward the longer span, since
    a longer match is the more specific claim.
    """
    method_rank = {"exact": 4, "whitespace": 3, "compact": 2, "fuzzy": 1, "none": 0}
    ordered = sorted(
        [a for a in alignments if a.ok],
        key=lambda a: (method_rank[a.method], a.score, (a.end or 0) - (a.start or 0)),
        reverse=True,
    )
    kept: list[Alignment] = []
    for candidate in ordered:
        if any(
            candidate.start < k.end and k.start < candidate.end for k in kept
        ):
            continue
        kept.append(candidate)
    kept_fields = {id(k) for k in kept}
    return [
        a if (id(a) in kept_fields or not a.ok)
        else Alignment(a.field, a.value, None, None, "overlap", a.score, a.occurrences)
        for a in alignments
    ]


def align_fields(
    doc: OcrDocument,
    values: dict[str, str | None],
    report: AlignmentReport | None = None,
) -> dict[str, Alignment]:
    """Align every field value for one document, resolving span collisions."""
    alignments = [
        find_span(doc.text, value, field)
        for field, value in values.items()
        if value
    ]
    alignments = _resolve_overlaps(alignments)
    if report is not None:
        for a in alignments:
            report.add(a)
    return {a.field: a for a in alignments}


def spans_to_bio(
    text: str,
    alignments: dict[str, Alignment],
    offsets: list[tuple[int, int]],
) -> list[str]:
    """Convert aligned character spans into BIO tags over tokenizer offsets.

    ``offsets`` comes from a HuggingFace tokenizer called with
    ``return_offsets_mapping=True``. Special tokens carry ``(0, 0)`` and are tagged
    ``O``; the trainer masks them out separately.
    """
    labels = ["O"] * len(offsets)
    spans = sorted(
        [(a.start, a.end, a.field) for a in alignments.values() if a.ok],
        key=lambda s: s[0],
    )
    for start, end, field in spans:
        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:  # special token
                continue
            if tok_start >= end or tok_end <= start:
                continue
            labels[i] = f"{'B' if first else 'I'}-{field}"
            first = False
    return labels


def bio_to_values(
    text: str,
    labels: list[str],
    offsets: list[tuple[int, int]],
) -> dict[str, list[tuple[str, int, int]]]:
    """Decode BIO tags back into field values with their spans.

    A stray ``I-`` tag with no preceding ``B-`` starts a new entity rather than being
    discarded: the model's span boundaries are often right even when its prefix is
    wrong, and throwing the span away would cost recall for no benefit.
    """
    out: dict[str, list[tuple[str, int, int]]] = {}
    current_field: str | None = None
    start = end = 0

    def flush() -> None:
        nonlocal current_field
        if current_field is not None and end > start:
            out.setdefault(current_field, []).append((text[start:end], start, end))
        current_field = None

    for label, (tok_start, tok_end) in zip(labels, offsets):
        if tok_start == tok_end == 0:
            continue
        if label == "O":
            flush()
            continue
        prefix, _, fname = label.partition("-")
        if prefix == "B" or fname != current_field:
            flush()
            current_field, start, end = fname, tok_start, tok_end
        else:
            end = tok_end
    flush()
    return out


def pick_best_value(
    candidates: list[tuple[str, int, int]], field: str
) -> str | None:
    """Choose one value when the model tagged several spans for the same field.

    Preference goes to the longest span: partial extractions ("Mercy General" for
    "Mercy General Hospital") are the dominant failure mode, and the longer candidate
    is nearly always the more complete one.
    """
    if not candidates:
        return None
    if FIELD_KINDS.get(field) in (FieldKind.amount, FieldKind.date):
        return candidates[0][0]  # first mention is the headline figure/date
    return max(candidates, key=lambda c: len(c[0]))[0]
