"""Scoring: per-field P/R/F1, document-type accuracy, service lines, and bootstrap CIs.

Every approach is scored through this one module so that no approach gets an advantage
from formatting. Values are compared *after* the shared normalisers in
:mod:`docintel.normalize`, which is what makes ``03/14/2025`` and ``2025-03-14`` count
as the same answer regardless of which system emitted which.

Three strictness levels are reported, because they answer different questions:

``exact``
    Byte-identical string. Harsh, but it is what a downstream system that does no
    parsing of its own would actually get.
``normalised``
    Equal after type-aware canonicalisation. **This is the headline metric** — it
    measures whether the system found the right entity.
``partial``
    Character-similarity score. Separates "found the wrong thing" from "found the
    right thing but clipped a word off it", which matters a lot for error analysis.

Scoring is restricted to the fields a document's type actually carries, so no approach
is penalised for correctly declining to invent an ``amount_paid`` on a lab order.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field as _field
from difflib import SequenceMatcher

from ..normalize import appears_in_text, normalize_field
from ..schema import (
    DocType,
    ExtractionResult,
    ServiceLine,
    expected_fields,
)


@dataclass
class FieldOutcome:
    """The result of scoring one field on one document."""

    doc_id: str
    field: str
    gold: str | None
    pred: str | None
    exact: bool
    correct: bool          # normalised match -- the headline notion of "right"
    partial: float
    gold_present: bool
    pred_present: bool
    hallucinated: bool     # predicted a value that occurs nowhere in the source text
    slice_keys: dict[str, str] = _field(default_factory=dict)

    @property
    def tp(self) -> bool:
        return self.gold_present and self.pred_present and self.correct

    @property
    def fp(self) -> bool:
        return self.pred_present and not self.tp

    @property
    def fn(self) -> bool:
        return self.gold_present and not self.tp

    @property
    def tn(self) -> bool:
        return not self.gold_present and not self.pred_present


@dataclass
class PRF:
    """Precision/recall/F1/accuracy with the counts they came from."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "support": self.support,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def _partial_score(field: str, gold: str | None, pred: str | None) -> float:
    if not gold or not pred:
        return 0.0
    a = normalize_field(field, gold) or ""
    b = normalize_field(field, pred) or ""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_document(
    doc_id: str,
    gold_fields: dict[str, str | None],
    doc_type_gold: DocType,
    prediction: ExtractionResult,
    source_text: str,
    slice_keys: dict[str, str] | None = None,
) -> list[FieldOutcome]:
    """Score every field this document type carries."""
    outcomes: list[FieldOutcome] = []
    fields = expected_fields(doc_type_gold) or set(gold_fields)

    for name in sorted(fields):
        gold = gold_fields.get(name)
        pred = prediction.value(name)
        gold_present = bool(gold)
        pred_present = bool(pred)

        n_gold = normalize_field(name, gold) if gold_present else None
        n_pred = normalize_field(name, pred) if pred_present else None
        correct = bool(n_gold) and n_gold == n_pred
        exact = gold_present and pred_present and gold.strip() == pred.strip()

        outcomes.append(
            FieldOutcome(
                doc_id=doc_id,
                field=name,
                gold=gold,
                pred=pred,
                exact=exact,
                correct=correct,
                partial=_partial_score(name, gold, pred),
                gold_present=gold_present,
                pred_present=pred_present,
                hallucinated=(
                    pred_present and not appears_in_text(pred, source_text, name)
                ),
                slice_keys=dict(slice_keys or {}),
            )
        )
    return outcomes


def aggregate_by_field(outcomes: list[FieldOutcome]) -> dict[str, PRF]:
    """Per-field precision/recall/F1."""
    table: dict[str, PRF] = {}
    for outcome in outcomes:
        prf = table.setdefault(outcome.field, PRF())
        if outcome.tp:
            prf.tp += 1
        elif outcome.fp and outcome.fn:
            # A wrong value on a field that had a gold answer is both a false positive
            # and a false negative -- it hurt precision and recall alike.
            prf.fp += 1
            prf.fn += 1
        elif outcome.fp:
            prf.fp += 1
        elif outcome.fn:
            prf.fn += 1
        else:
            prf.tn += 1
    return table


def micro_average(outcomes: list[FieldOutcome]) -> PRF:
    """Pool every field decision into one PRF — dominated by common fields."""
    total = PRF()
    for prf in aggregate_by_field(outcomes).values():
        total.tp += prf.tp
        total.fp += prf.fp
        total.fn += prf.fn
        total.tn += prf.tn
    return total


def macro_f1(outcomes: list[FieldOutcome]) -> float:
    """Mean of per-field F1 — treats a rare field as equal to a common one."""
    scores = [prf.f1 for prf in aggregate_by_field(outcomes).values()]
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------------------------------
# Document type
# --------------------------------------------------------------------------------------

def doctype_metrics(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy, macro-F1 and confusion matrix from (gold, predicted) label pairs."""
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    confusion = {g: {p: 0 for p in labels} for g in labels}
    correct = 0
    for gold, pred in pairs:
        confusion[gold][pred] += 1
        correct += gold == pred

    per_class = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": tp + fn,
        }
        if tp + fn:
            f1s.append(f1)

    return {
        "accuracy": round(correct / len(pairs), 4) if pairs else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "n": len(pairs),
    }


# --------------------------------------------------------------------------------------
# Service lines (the relationship task)
# --------------------------------------------------------------------------------------

def _line_key(line: ServiceLine | dict) -> tuple:
    get = line.get if isinstance(line, dict) else lambda k, d=None: getattr(line, k, d)
    return (
        normalize_field("procedure_code", get("procedure_code")),
        normalize_field("date_of_service", get("date_of_service")),
        normalize_field("total_charge", get("charge")),
    )


def service_line_prf(
    gold_lines: list, pred_lines: list
) -> PRF:
    """Set-F1 over normalised (code, date, charge) tuples.

    Compared as multisets: a claim legitimately repeats a procedure code across two
    dates, and collapsing those into a set would quietly forgive a missed row.
    """
    prf = PRF()
    gold_keys = [_line_key(line) for line in gold_lines]
    pred_keys = [_line_key(line) for line in pred_lines]
    remaining = list(gold_keys)
    for key in pred_keys:
        if key in remaining:
            remaining.remove(key)
            prf.tp += 1
        else:
            prf.fp += 1
    prf.fn += len(remaining)
    return prf


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------

def bootstrap_ci(
    outcomes: list[FieldOutcome],
    statistic=lambda o: micro_average(o).f1,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for a statistic, resampling **documents**, not fields.

    Resampling fields would treat the 15 fields of one document as 15 independent
    observations. They are not: a document that OCR mangled tends to fail on all of
    them at once, so field-level resampling would produce misleadingly tight intervals
    and manufacture significance that is not there.
    """
    by_doc: dict[str, list[FieldOutcome]] = {}
    for outcome in outcomes:
        by_doc.setdefault(outcome.doc_id, []).append(outcome)
    doc_ids = list(by_doc)
    if not doc_ids:
        return 0.0, 0.0, 0.0

    point = statistic(outcomes)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        drawn: list[FieldOutcome] = []
        for _ in range(len(doc_ids)):
            drawn.extend(by_doc[doc_ids[rng.randrange(len(doc_ids))]])
        samples.append(statistic(drawn))
    samples.sort()
    lo_idx = int((1 - confidence) / 2 * len(samples))
    hi_idx = min(len(samples) - 1, int((1 + confidence) / 2 * len(samples)))
    return point, samples[lo_idx], samples[hi_idx]


def paired_bootstrap_pvalue(
    outcomes_a: list[FieldOutcome],
    outcomes_b: list[FieldOutcome],
    statistic=lambda o: micro_average(o).f1,
    resamples: int = 1000,
    seed: int = 0,
) -> float:
    """Two-sided paired bootstrap p-value that A and B differ on the same documents.

    Paired on document id, because both systems saw exactly the same gold set — an
    unpaired test would throw away that pairing and lose power.
    """
    a_by_doc: dict[str, list[FieldOutcome]] = {}
    b_by_doc: dict[str, list[FieldOutcome]] = {}
    for outcome in outcomes_a:
        a_by_doc.setdefault(outcome.doc_id, []).append(outcome)
    for outcome in outcomes_b:
        b_by_doc.setdefault(outcome.doc_id, []).append(outcome)
    doc_ids = sorted(set(a_by_doc) & set(b_by_doc))
    if not doc_ids:
        return 1.0

    observed = abs(
        statistic([o for d in doc_ids for o in a_by_doc[d]])
        - statistic([o for d in doc_ids for o in b_by_doc[d]])
    )
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(resamples):
        drawn_a: list[FieldOutcome] = []
        drawn_b: list[FieldOutcome] = []
        for _ in range(len(doc_ids)):
            doc = doc_ids[rng.randrange(len(doc_ids))]
            # Randomly swap the two systems' results for this document under H0.
            if rng.random() < 0.5:
                drawn_a.extend(a_by_doc[doc])
                drawn_b.extend(b_by_doc[doc])
            else:
                drawn_a.extend(b_by_doc[doc])
                drawn_b.extend(a_by_doc[doc])
        if abs(statistic(drawn_a) - statistic(drawn_b)) >= observed:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (resamples + 1)


def slice_outcomes(
    outcomes: list[FieldOutcome], key: str
) -> dict[str, list[FieldOutcome]]:
    """Split outcomes by a slice key (condition, provenance, layout, language)."""
    buckets: dict[str, list[FieldOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.slice_keys.get(key, "?"), []).append(outcome)
    return buckets


def hallucination_rate(outcomes: list[FieldOutcome]) -> float:
    """Share of emitted values that appear nowhere in the source document."""
    emitted = [o for o in outcomes if o.pred_present]
    if not emitted:
        return 0.0
    return sum(o.hallucinated for o in emitted) / len(emitted)
