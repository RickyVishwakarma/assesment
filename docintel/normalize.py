"""Value normalisation, shared by every approach *and* by the scorer.

This module is deliberately the single source of truth for "are these two values the
same thing?". If each approach normalised its own output, an approach that happened to
emit ``2025-03-14`` would beat one that emitted ``03/14/2025`` for reasons that have
nothing to do with document understanding. Everything funnels through here instead.

Design decisions worth defending:

* **Dates are US-order.** ``03/04/2025`` is read as 4 March, not 3 April. These are US
  healthcare documents (NPI, CPT, CMS-1500), so month-first is the right prior. Genuinely
  ambiguous slash dates are recorded in :data:`AMBIGUOUS_DATE_HITS` so the error analysis
  can quantify how often this assumption is load-bearing rather than hand-waving it.
* **Amounts become ``Decimal`` with 2dp**, never floats — money in binary floating point
  is how you get ``0.1 + 0.2``-class bugs in a financial extraction system.
* **Normalisation is lossy on purpose.** Stripping ``Inc``/``LLC`` from organisations and
  dots from ICD-10 codes means the metric measures whether the system found the right
  entity, not whether it reproduced incidental punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from .schema import FIELD_KINDS, FieldKind

# Slash/dot/dash dates whose day component is <= 12 are order-ambiguous. We resolve them
# US-first, but count them so the report can state the exposure.
AMBIGUOUS_DATE_HITS: list[str] = []

_WS = re.compile(r"\s+")

# Honorifics and post-nominals that carry no identity information.
_NAME_TITLES = re.compile(
    r"\b(?:dr|doctor|mr|mrs|ms|miss|prof|sr|sra|srta)\b\.?",
    re.IGNORECASE,
)
_NAME_SUFFIXES = re.compile(
    r"\b(?:m\.?d|d\.?o|r\.?n|n\.?p|p\.?a[- ]?c|d\.?d\.?s|ph\.?d|f\.?a\.?c\.?p"
    r"|jr|sr|ii|iii|iv|esq)\b\.?",
    re.IGNORECASE,
)

# Corporate/legal suffixes that differ between how a document prints an org and how a
# system extracts it, without changing which organisation is meant.
_ORG_SUFFIXES = re.compile(
    r"\b(?:inc|llc|l\.l\.c|llp|ltd|limited|corp|corporation|co|company|pc|p\.c"
    r"|pa|p\.a|plc|group|grp|assoc|associates|partners|holdings|sa|s\.a)\b\.?",
    re.IGNORECASE,
)

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "septiembre": 9, "setiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}

_NUMERIC_DATE = re.compile(r"^(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})$")
_TEXT_DATE_MDY = re.compile(r"^([a-zé]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})$")
_TEXT_DATE_DMY = re.compile(
    r"^(\d{1,2})(?:st|nd|rd|th)?[\s\-]+(?:de\s+)?([a-zé]+)\.?[\s\-,]+(?:de\s+)?(\d{2,4})$"
)


def strip_accents(text: str) -> str:
    """Fold accents so Spanish documents compare equal to their ASCII extractions."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def squash(text: str) -> str:
    """Collapse all whitespace runs (including OCR's stray newlines) to single spaces."""
    return _WS.sub(" ", text).strip()


def _expand_year(year: int) -> int:
    """Two-digit years: <=30 is 20xx, otherwise 19xx (patient DOBs are mostly 19xx)."""
    if year >= 100:
        return year
    return 2000 + year if year <= 30 else 1900 + year


def _safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def normalize_date(value: str) -> str | None:
    """Parse a date in any format this corpus emits and return ISO-8601, else None."""
    if not value:
        return None
    text = squash(strip_accents(value)).lower().strip(" ,;:")
    # Extractors often capture the anchor along with the value ("DOS: 03/14/2025").
    # Strip a leading label rather than relying on fuzzy parsing, which would happily
    # turn "patient 5 of 12" into a date and silently manufacture a scoring match.
    text = re.sub(r"^[a-z .]{1,30}:\s*", "", text)
    text = re.sub(r"^(?:on|el|dated?|fecha|de)\s+", "", text).strip()

    m = _NUMERIC_DATE.match(text)
    if m:
        a, b, c = (int(g) for g in m.groups())
        # ISO-ish: 2025-03-14
        if len(m.group(1)) == 4:
            return _safe_date(a, b, c)
        year = _expand_year(c)
        if a > 12:  # unambiguous day-first
            return _safe_date(year, b, a)
        if b > 12:  # unambiguous month-first
            return _safe_date(year, a, b)
        AMBIGUOUS_DATE_HITS.append(value)  # both readings valid -> US order wins
        return _safe_date(year, a, b)

    m = _TEXT_DATE_MDY.match(text)
    if m and m.group(1) in _MONTHS:
        return _safe_date(_expand_year(int(m.group(3))), _MONTHS[m.group(1)], int(m.group(2)))

    m = _TEXT_DATE_DMY.match(text)
    if m and m.group(2) in _MONTHS:
        return _safe_date(_expand_year(int(m.group(3))), _MONTHS[m.group(2)], int(m.group(1)))

    try:  # last resort, only for shapes we did not anticipate
        from dateutil import parser as _dateutil

        # fuzzy=False on purpose: a normaliser that invents dates from arbitrary prose
        # corrupts every metric that depends on it.
        return _dateutil.parse(text, dayfirst=False, fuzzy=False).date().isoformat()
    except Exception:
        return None


def normalize_amount(value: str) -> str | None:
    """Return a 2dp decimal string, or None if no monetary quantity is present."""
    if not value:
        return None
    text = squash(strip_accents(value))
    negative = "(" in text and ")" in text or text.lstrip().startswith("-")
    text = re.sub(r"(?i)\b(?:usd|us\$|dollars?|total|amount|monto|importe)\b", " ", text)
    text = re.sub(r"[^\d,.\-]", "", text)
    if not re.search(r"\d", text):
        return None

    # Distinguish 1.234,56 (European) from 1,234.56 (US) by which separator is last.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # A single comma with exactly 2 trailing digits is a decimal comma.
        text = text.replace(",", "." if re.search(r",\d{2}$", text) else "")

    text = re.sub(r"[^\d.\-]", "", text).lstrip("-")
    if text.count(".") > 1:  # e.g. thousands dots that survived: 1.234.567
        head, _, tail = text.rpartition(".")
        text = head.replace(".", "") + "." + tail
    try:
        amount = Decimal(text).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None
    return str(-amount if negative else amount)


def normalize_person(value: str) -> str | None:
    """Casefolded ``first last``, with titles and post-nominals removed."""
    if not value:
        return None
    # Strip titles/post-nominals *before* touching punctuation: the patterns are written
    # to match the dotted forms ("M.D."), which vanish if dots are removed first.
    text = squash(strip_accents(value))
    text = _NAME_TITLES.sub(" ", text)
    text = _NAME_SUFFIXES.sub(" ", text)
    text = text.replace(".", " ")
    text = _NAME_SUFFIXES.sub(" ", text)  # again for undotted forms exposed above
    text = re.sub(r"[^\w\s,'\-]", " ", text)
    # "Doe, Jane" and "Jane Doe" are the same person.
    if "," in text:
        last, _, first = text.partition(",")
        if first.strip():
            text = f"{first} {last}"
    text = squash(text).lower().strip(" ,-'")
    return text or None


def normalize_org(value: str) -> str | None:
    """Casefolded organisation name with legal suffixes and punctuation removed."""
    if not value:
        return None
    # Same ordering constraint as names: "P.C." only matches while its dots survive.
    text = squash(strip_accents(value))
    text = _ORG_SUFFIXES.sub(" ", text)
    text = re.sub(r"[^\w\s&\-]", " ", text)
    text = _ORG_SUFFIXES.sub(" ", text)  # again for undotted forms exposed above
    return squash(text).lower().strip(" -&") or None


def normalize_reference(value: str) -> str | None:
    """Uppercase alphanumerics only — separators in IDs carry no information."""
    if not value:
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", strip_accents(value)).upper()
    return text or None


def normalize_code(value: str) -> str | None:
    """Uppercase alphanumerics — folds ``A01.1`` and ``A011`` together."""
    if not value:
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", strip_accents(value)).upper()
    return text or None


_NORMALIZERS = {
    FieldKind.date: normalize_date,
    FieldKind.amount: normalize_amount,
    FieldKind.person: normalize_person,
    FieldKind.org: normalize_org,
    FieldKind.reference: normalize_reference,
    FieldKind.code: normalize_code,
}


def normalize_field(field: str, value: str | None) -> str | None:
    """Normalise ``value`` according to the kind of ``field``.

    Unknown field names fall back to whitespace-squashed casefolding rather than
    raising, so an approach inventing an out-of-schema field still scores as wrong
    instead of crashing the evaluator.
    """
    if value is None:
        return None
    kind = FIELD_KINDS.get(field)
    if kind is None:
        return squash(strip_accents(str(value))).lower() or None
    return _NORMALIZERS[kind](str(value))


def values_match(field: str, a: str | None, b: str | None) -> bool:
    """True when two raw values mean the same thing for this field."""
    if a is None or b is None:
        return a is None and b is None
    na, nb = normalize_field(field, a), normalize_field(field, b)
    return na is not None and na == nb


# --------------------------------------------------------------------------------------
# Validators — used by the NLP approach as extractors, and by error analysis as evidence.
# --------------------------------------------------------------------------------------

def npi_check_digit(first_nine: str) -> int:
    """Luhn check digit for an NPI, per the CMS spec (prefix ``80840``)."""
    payload = "80840" + first_nine
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:  # the check digit will occupy an odd position once appended
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def is_valid_npi(value: str | None) -> bool:
    """True for a syntactically valid 10-digit NPI with a correct check digit.

    Real NPIs carry this checksum, which makes it a *free* precision win: a 10-digit
    run that fails the check is almost certainly a fax number or an account number,
    so the rules approach can reject it without any learned model.
    """
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 10:
        return False
    return npi_check_digit(digits[:9]) == int(digits[9])


ICD10_RE = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.?[0-9A-Z]{1,4})?)\b")
CPT_RE = re.compile(r"\b(\d{4}[\dF]|[A-CEG-VX][\d]{4})\b")


def looks_like_icd10(value: str | None) -> bool:
    return bool(value and ICD10_RE.fullmatch(value.strip().upper()))


def looks_like_cpt(value: str | None) -> bool:
    return bool(value and CPT_RE.fullmatch(value.strip().upper()))


def appears_in_text(value: str | None, text: str, field: str | None = None) -> bool:
    """Whether ``value`` is actually present in the source document.

    This is the hallucination detector. A generative model that returns a value whose
    normalised form appears nowhere in the text did not extract it — it invented it.
    Comparison is done on an accent-folded, non-alphanumeric-stripped view so that
    reformatting (``$1,234.56`` -> ``1234.56``) is not mistaken for invention.
    """
    if not value:
        return True
    haystack = re.sub(r"[^a-z0-9]", "", strip_accents(text).lower())
    needle = re.sub(r"[^a-z0-9]", "", strip_accents(str(value)).lower())
    if not needle:
        return True
    if needle in haystack:
        return True
    # Dates legitimately get reformatted, so also accept any surface form that
    # normalises to the same ISO date somewhere in the text.
    if field and FIELD_KINDS.get(field) is FieldKind.date:
        target = normalize_date(str(value))
        if target:
            for cand in re.findall(r"[0-9][0-9/\-.]{4,12}[0-9]", text):
                if normalize_date(cand) == target:
                    return True
    return False
