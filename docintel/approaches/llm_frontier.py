"""The frontier-LLM reference tier, served from a cache.

Why this exists: the machine this project runs on has 7.4GB of RAM, which caps a local
model at roughly 3B parameters at 4-bit. A 3B instruct model is itself arguably a small
model, so using it alone as "the general-purpose LLM" would make the central engineering
question unanswerable — you cannot say where a general LLM earns its cost if you never
measured one.

So the gold set is additionally annotated by a frontier model (Claude), working through
the Claude Code session that builds this project. No API key and no marginal spend are
involved, which is why this tier exists at all given the constraints. The annotations
are written to a JSON cache and this module replays them, so the CLI, the API and the
evaluator treat this tier exactly like the other three.

Two honesty requirements, enforced here rather than left to the write-up:

* Cost per document is **estimated from published per-token pricing** applied to measured
  token counts. It was never metered by a billing system. :data:`COST_BASIS` records
  that, and the evaluator propagates it into the report so no table can silently imply
  a measured cost.
* This tier is **not reproducible by a third party** the way the other three are. It is
  a reference point, not a runnable baseline, and it is labelled as such.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..schema import (
    FIELD_NAMES,
    DocType,
    ExtractedField,
    ExtractionResult,
    OcrDocument,
    ServiceLine,
)

DEFAULT_CACHE = Path("data/frontier/annotations.json")

#: Model identity and pricing used for the estimated cost column.
MODEL_NAME = "claude-opus-5"
USD_PER_MTOK_INPUT = 5.0
USD_PER_MTOK_OUTPUT = 25.0
COST_BASIS = "estimated from published per-token pricing; not metered"


class FrontierLlmExtractor:
    """Replays frontier-model annotations recorded for the gold set."""

    name = "llm_frontier"

    def __init__(self, cache_path: str | Path = DEFAULT_CACHE):
        self.cache_path = Path(cache_path)
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    @property
    def available(self) -> bool:
        return bool(self._cache)

    def covers(self, doc_id: str) -> bool:
        return doc_id in self._cache

    def extract(self, doc: OcrDocument) -> ExtractionResult:
        started = time.perf_counter()
        record = self._cache.get(doc.doc_id)

        if record is None:
            # Absent is *not* the same as "predicted nothing". Scoring a missing
            # annotation as an empty prediction would silently credit this tier with
            # a perfect precision score on documents it never saw.
            return ExtractionResult(
                doc_id=doc.doc_id,
                approach=self.name,
                doc_type=DocType.unknown,
                fields={name: None for name in FIELD_NAMES},
                latency_ms=None,
                meta={
                    "missing_annotation": True,
                    "cost_basis": COST_BASIS,
                    "note": (
                        "no cached annotation for this document -- this tier is "
                        "annotated on the gold split only, so an empty result means "
                        "'not run', not 'found nothing'"
                    ),
                },
            )

        raw_type = (record.get("document_type") or "").strip()
        try:
            doc_type = DocType(raw_type)
        except ValueError:
            doc_type = DocType.unknown

        fields: dict[str, ExtractedField | None] = {}
        for name in FIELD_NAMES:
            value = record.get(name)
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

        service_lines = [
            ServiceLine(
                procedure_code=row.get("procedure_code"),
                date_of_service=row.get("date_of_service"),
                units=row.get("units"),
                charge=row.get("charge"),
                paid=row.get("paid"),
            )
            for row in (record.get("service_lines") or [])
            if isinstance(row, dict)
        ]

        tokens_in = record.get("prompt_tokens") or 0
        tokens_out = record.get("completion_tokens") or 0
        estimated_cost = (
            tokens_in / 1e6 * USD_PER_MTOK_INPUT
            + tokens_out / 1e6 * USD_PER_MTOK_OUTPUT
        )

        return ExtractionResult(
            doc_id=doc.doc_id,
            approach=self.name,
            doc_type=doc_type,
            doc_type_confidence=None,
            fields=fields,
            service_lines=service_lines,
            org_roles=record.get("org_roles") or {
                "referring_org": None,
                "servicing_org": (fields.get("servicing_facility") or ExtractedField(value="")).value or None,
                "payer_org": (fields.get("payer_name") or ExtractedField(value="")).value or None,
            },
            # Replay latency is meaningless; the recorded wall-clock is what counts.
            latency_ms=record.get("latency_ms"),
            meta={
                "model": MODEL_NAME,
                "replayed_from_cache": True,
                "cost_basis": COST_BASIS,
                "estimated_cost_usd": round(estimated_cost, 6),
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "condition": doc.condition,
            },
        )


def mean_cost_per_doc(cache_path: str | Path = DEFAULT_CACHE) -> float:
    """Mean estimated USD per document across the cached annotations.

    This has to be computed from the cache rather than hard-coded, because cost scales
    with document length and the corpus mixes short intake forms with long claims.

    It also has to be reported *at all*: the profile previously left this at its 0.0
    default, which made the one tier that actually costs money appear free, and inverted
    the central cost comparison. The figure is an estimate from published per-token
    pricing applied to estimated token counts -- see COST_BASIS -- never a metered bill.
    """
    path = Path(cache_path)
    if not path.exists():
        return 0.0
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return 0.0
    total = sum(
        (r.get("prompt_tokens") or 0) / 1e6 * USD_PER_MTOK_INPUT
        + (r.get("completion_tokens") or 0) / 1e6 * USD_PER_MTOK_OUTPUT
        for r in records.values()
    )
    return total / len(records)


def write_annotations(records: dict[str, dict], path: str | Path = DEFAULT_CACHE) -> None:
    """Persist frontier annotations, merging with anything already recorded."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
    existing.update(records)
    target.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
