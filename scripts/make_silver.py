"""Generates silver training labels with the local LLM, then aligns them to spans.

Run:
    python scripts/make_silver.py --splits train val --limit 50   # smoke
    python scripts/make_silver.py --splits train val              # full

Pipeline per document:

    OCR text -> local LLM (schema-constrained) -> field values
             -> span alignment -> character spans -> BIO-ready training example

Two things this script measures that most silver-data pipelines quietly skip:

**Teacher accuracy.** Because the generator knows the true values, we can score the
LLM's silver labels against them. That gives a per-field number for how good the
supervision actually is — and it is the baseline the student must be compared against.
Reporting "the student got 0.87 F1" means little; reporting "the student got 0.87 from
a teacher that was only 0.81" is the interesting claim.

**Alignment rate.** A value the aligner cannot locate in the text cannot become a
training label, so it is silently lost supervision. Rather than dropping it quietly,
the rate is reported per field.

Results are cached per document, so an interrupted run resumes instead of re-paying
for inference already done.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.align import AlignmentReport, align_fields  # noqa: E402
from docintel.approaches.llm_local import (  # noqa: E402
    DEFAULT_MODEL,
    LocalLlmExtractor,
    list_models,
    ollama_available,
)
from docintel.normalize import values_match  # noqa: E402
from docintel.schema import OcrDocument  # noqa: E402


def load_manifest(corpus: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--out", default="data/silver", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache", default=".tmp/silver_cache", type=Path)
    args = parser.parse_args()

    if not ollama_available():
        print(
            "ERROR: Ollama is not reachable at 127.0.0.1:11434.\n"
            "Start it with 'ollama serve' and pull the model:\n"
            f"    ollama pull {args.model}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    available = list_models()
    if args.model not in available and available:
        print(f"WARNING: {args.model} not in {available}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    extractor = LocalLlmExtractor(model=args.model)
    manifest = [r for r in load_manifest(args.corpus) if r["split"] in args.splits]
    if args.limit:
        manifest = manifest[: args.limit]

    report = AlignmentReport()
    teacher_hits: dict[str, int] = defaultdict(int)
    teacher_total: dict[str, int] = defaultdict(int)
    doctype_hits = doctype_total = 0
    examples: list[dict] = []
    latencies: list[float] = []

    for index, record in enumerate(manifest, 1):
        doc_id = record["doc_id"]
        cache_file = args.cache / f"{doc_id}.json"
        doc = OcrDocument.model_validate_json(
            (args.corpus / record["text"]).read_text(encoding="utf-8")
        )

        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            started = time.perf_counter()
            result = extractor.extract(doc)
            payload = {
                "doc_type": result.doc_type.value,
                "fields": {k: (v.value if v else None) for k, v in result.fields.items()},
                "latency_ms": (time.perf_counter() - started) * 1000,
                "meta": result.meta,
            }
            cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        latencies.append(payload.get("latency_ms") or 0.0)

        # --- teacher quality, measured against the generator's ground truth ---------
        for field, gold in record["truth"].items():
            if gold is None:
                continue
            teacher_total[field] += 1
            if values_match(field, gold, payload["fields"].get(field)):
                teacher_hits[field] += 1
        doctype_total += 1
        doctype_hits += payload["doc_type"] == record["doc_type"]

        # --- alignment: values -> character spans -----------------------------------
        alignments = align_fields(doc, payload["fields"], report)
        labels = {
            field: (a.start, a.end)
            for field, a in alignments.items()
            if a.ok
        }
        examples.append({
            "doc_id": doc_id,
            "split": record["split"],
            "text": doc.text,
            "labels": labels,
            "doc_type": payload["doc_type"],
            "doc_type_gold": record["doc_type"],
            "condition": record["condition"],
            "template_id": record["template_id"],
            "lang": record["lang"],
        })

        if index % 25 == 0:
            print(
                f"  {index}/{len(manifest)}  align={report.rate:.1%}  "
                f"mean_latency={sum(latencies)/len(latencies):.0f}ms",
                flush=True,
            )

    for split in args.splits:
        rows = [e for e in examples if e["split"] == split]
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")

    # --- the two numbers that make this pipeline auditable -------------------------
    teacher = {
        field: {
            "accuracy": round(teacher_hits[field] / n, 4),
            "n": n,
        }
        for field, n in sorted(teacher_total.items())
    }
    overall_teacher = (
        sum(teacher_hits.values()) / sum(teacher_total.values())
        if teacher_total else 0.0
    )
    summary = {
        "model": args.model,
        "n_documents": len(examples),
        "teacher_field_accuracy": teacher,
        "teacher_overall_accuracy": round(overall_teacher, 4),
        "teacher_doctype_accuracy": round(doctype_hits / doctype_total, 4)
        if doctype_total else 0.0,
        "alignment_rate": round(report.rate, 4),
        "alignment_by_method": report.by_method,
        "alignment_by_field": {k: round(v, 4) for k, v in report.field_rates().items()},
        "ambiguous_values": report.ambiguous,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
    }
    (args.out / "silver_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nTeacher overall accuracy : {overall_teacher:.1%}")
    print(f"Teacher doc-type accuracy: {summary['teacher_doctype_accuracy']:.1%}")
    print(f"Span alignment rate      : {report.rate:.1%}")
    print(f"Alignment methods        : {report.by_method}")
    print("\nWeakest teacher fields:")
    for field, stats in sorted(teacher.items(), key=lambda kv: kv[1]["accuracy"])[:6]:
        print(f"  {field:26s} {stats['accuracy']:6.1%}  (n={stats['n']})")


if __name__ == "__main__":
    main()
