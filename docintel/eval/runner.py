"""Runs approaches over a split and produces the comparison tables.

One code path scores all four systems on exactly the same documents, which is the only
way the comparison means anything. The runner also collects the operational columns the
assignment asks for — latency, cost, model size, offline capability, determinism — since
those are properties of a *run*, not of a scoring function.

Determinism is measured rather than assumed: each system is optionally run three times
over a subset and the byte-identical rate of its output is reported. A local LLM at
temperature 0 with a fixed seed is usually but not always reproducible, and that is a
result worth showing next to the rules approach's trivial 100%.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field as _field
from pathlib import Path

from ..schema import DocType, ExtractionResult, OcrDocument
from .metrics import (
    FieldOutcome,
    aggregate_by_field,
    bootstrap_ci,
    doctype_metrics,
    hallucination_rate,
    macro_f1,
    micro_average,
    paired_bootstrap_pvalue,
    score_document,
    service_line_prf,
    slice_outcomes,
)


@dataclass
class ApproachProfile:
    """The operational characteristics of one approach."""

    name: str
    model_size_mb: float | None = None
    offline: bool = True
    deterministic: bool | None = None
    determinism_rate: float | None = None
    cost_per_doc_usd: float = 0.0
    cost_basis: str = "measured"
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    notes: str = ""


@dataclass
class RunResult:
    approach: str
    outcomes: list[FieldOutcome] = _field(default_factory=list)
    doctype_pairs: list[tuple[str, str]] = _field(default_factory=list)
    service_line_tp: int = 0
    service_line_fp: int = 0
    service_line_fn: int = 0
    latencies: list[float] = _field(default_factory=list)
    profile: ApproachProfile | None = None
    errors: list[str] = _field(default_factory=list)


def load_split(corpus: Path, split: str) -> list[dict]:
    """Manifest records for one split."""
    manifest = corpus / "manifest.jsonl"
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == split
    ]


def load_gold(corpus: Path, split: str, gold_dir: Path | None) -> list[dict]:
    """Gold records, preferring human-verified annotations when they exist.

    The verified file is authoritative: it is the artefact a human actually checked.
    Falling back to the generator's truth is correct for a smoke run but must never be
    mistaken for the gold set in the report, so the source is recorded per record.
    """
    records = load_split(corpus, split)
    verified_path = (gold_dir / f"{split}.jsonl") if gold_dir else None
    if verified_path and verified_path.exists():
        verified = {
            json.loads(line)["doc_id"]: json.loads(line)
            for line in verified_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for record in records:
            match = verified.get(record["doc_id"])
            if match:
                record["truth"] = match.get("fields", record["truth"])
                record["service_lines"] = match.get(
                    "service_lines", record["service_lines"]
                )
                # Respect the record's own verdict rather than assuming that presence
                # in the gold file implies a human looked at it. The review tool writes
                # records in a non-interactive ``--auto-only`` pass too, where tier-2
                # fields are merely *flagged* and left unresolved. Labelling those
                # "human_verified" would let the report claim a manual verification that
                # never happened — the one claim this evaluation cannot afford to fake.
                if match.get("verified"):
                    record["gold_source"] = "human_verified"
                elif match.get("adjudicated_by"):
                    # Tier-2 fields were resolved, but by a model rather than a person.
                    # This must stay distinct from "human_verified": the assignment asks
                    # for a manually verified gold set, and reporting a machine pass as a
                    # human one would be exactly the false claim this branch exists to
                    # prevent.
                    record["gold_source"] = f"{match['adjudicated_by']}_adjudicated"
                else:
                    record["gold_source"] = "auto_verified_only"
                record["corrections"] = match.get("corrections", [])
            else:
                record["gold_source"] = "generator_truth"
    else:
        for record in records:
            record["gold_source"] = "generator_truth"
    return records


def run_approach(
    extractor,
    records: list[dict],
    corpus: Path,
    profile: ApproachProfile | None = None,
) -> RunResult:
    """Run one approach over the records and score every document."""
    result = RunResult(approach=getattr(extractor, "name", "unknown"))

    for record in records:
        doc = OcrDocument.model_validate_json(
            (corpus / record["text"]).read_text(encoding="utf-8")
        )
        try:
            prediction: ExtractionResult = extractor.extract(doc)
        except Exception as exc:  # a crash is a result, not a reason to lose the run
            result.errors.append(f"{record['doc_id']}: {type(exc).__name__}: {exc}")
            continue

        if prediction.latency_ms is not None:
            result.latencies.append(prediction.latency_ms)

        gold_type = DocType(record["doc_type"])
        result.doctype_pairs.append((record["doc_type"], prediction.doc_type.value))
        result.outcomes.extend(
            score_document(
                doc_id=record["doc_id"],
                gold_fields=record["truth"],
                doc_type_gold=gold_type,
                prediction=prediction,
                source_text=doc.text,
                slice_keys={
                    "condition": record.get("condition", "?"),
                    "lang": record.get("lang", "?"),
                    "template_id": str(record.get("template_id", "?")),
                    "doc_type": record["doc_type"],
                    "gold_source": record.get("gold_source", "?"),
                },
            )
        )

        prf = service_line_prf(record.get("service_lines") or [], prediction.service_lines)
        result.service_line_tp += prf.tp
        result.service_line_fp += prf.fp
        result.service_line_fn += prf.fn

    if profile:
        if result.latencies:
            ordered = sorted(result.latencies)
            profile.latency_p50_ms = round(statistics.median(ordered), 2)
            profile.latency_p95_ms = round(
                ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2
            )
        result.profile = profile
    return result


def measure_determinism(
    extractor, records: list[dict], corpus: Path, repeats: int = 3, sample: int = 10
) -> tuple[bool, float]:
    """Run the same documents repeatedly and report the byte-identical rate."""
    subset = records[:sample]
    signatures: list[list[str]] = []
    for _ in range(repeats):
        run: list[str] = []
        for record in subset:
            doc = OcrDocument.model_validate_json(
                (corpus / record["text"]).read_text(encoding="utf-8")
            )
            try:
                prediction = extractor.extract(doc)
                run.append(
                    json.dumps(
                        {
                            "t": prediction.doc_type.value,
                            "f": {
                                k: (v.value if v else None)
                                for k, v in sorted(prediction.fields.items())
                            },
                        },
                        sort_keys=True,
                    )
                )
            except Exception as exc:
                run.append(f"ERROR:{type(exc).__name__}")
        signatures.append(run)

    if not signatures or not signatures[0]:
        return True, 1.0
    identical = sum(
        all(signatures[r][i] == signatures[0][i] for r in range(1, repeats))
        for i in range(len(signatures[0]))
    )
    rate = identical / len(signatures[0])
    return rate == 1.0, round(rate, 4)


def summarise(result: RunResult, bootstrap: int = 1000) -> dict:
    """Build the full result dictionary for one approach."""
    outcomes = result.outcomes
    micro = micro_average(outcomes)
    point, low, high = bootstrap_ci(outcomes, resamples=bootstrap)

    sl_precision = (
        result.service_line_tp / (result.service_line_tp + result.service_line_fp)
        if (result.service_line_tp + result.service_line_fp) else 0.0
    )
    sl_recall = (
        result.service_line_tp / (result.service_line_tp + result.service_line_fn)
        if (result.service_line_tp + result.service_line_fn) else 0.0
    )
    sl_f1 = (
        2 * sl_precision * sl_recall / (sl_precision + sl_recall)
        if (sl_precision + sl_recall) else 0.0
    )

    summary = {
        "approach": result.approach,
        "n_documents": len({o.doc_id for o in outcomes}),
        "n_field_decisions": len(outcomes),
        "overall": {
            **micro.as_dict(),
            "macro_f1": round(macro_f1(outcomes), 4),
            "f1_ci95": [round(low, 4), round(high, 4)],
            "exact_match_rate": round(
                sum(o.exact for o in outcomes) / len(outcomes), 4
            ) if outcomes else 0.0,
            "partial_mean": round(
                sum(o.partial for o in outcomes) / len(outcomes), 4
            ) if outcomes else 0.0,
            "hallucination_rate": round(hallucination_rate(outcomes), 4),
        },
        "by_field": {
            name: prf.as_dict() for name, prf in sorted(aggregate_by_field(outcomes).items())
        },
        "doc_type": doctype_metrics(result.doctype_pairs),
        "service_lines": {
            "precision": round(sl_precision, 4),
            "recall": round(sl_recall, 4),
            "f1": round(sl_f1, 4),
            "tp": result.service_line_tp,
            "fp": result.service_line_fp,
            "fn": result.service_line_fn,
        },
        "slices": {},
        "errors": result.errors[:20],
        "n_errors": len(result.errors),
    }

    for key in ("condition", "lang", "doc_type", "gold_source"):
        summary["slices"][key] = {
            value: {
                **micro_average(bucket).as_dict(),
                "hallucination_rate": round(hallucination_rate(bucket), 4),
            }
            for value, bucket in sorted(slice_outcomes(outcomes, key).items())
        }

    if result.profile:
        summary["profile"] = asdict(result.profile)
    return summary


def compare(results: dict[str, RunResult], bootstrap: int = 1000) -> dict:
    """Pairwise significance tests between approaches on the same documents."""
    names = sorted(results)
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = paired_bootstrap_pvalue(
                results[a].outcomes, results[b].outcomes, resamples=bootstrap
            )
            comparisons[f"{a}_vs_{b}"] = {
                "f1_a": round(micro_average(results[a].outcomes).f1, 4),
                "f1_b": round(micro_average(results[b].outcomes).f1, 4),
                "p_value": round(p, 4),
                "significant_at_05": p < 0.05,
            }
    return comparisons
