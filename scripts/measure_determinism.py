"""Measure output determinism for one approach and record it in the eval report.

    python scripts/measure_determinism.py --approach small_model --sample 5

The assignment lists determinism as a comparison dimension, and it is one of the few
where the approaches genuinely differ in kind rather than degree: a rules engine is
deterministic by construction, a sampling LLM is not, and a temperature-0 LLM is
*usually* but not always reproducible. Asserting that from first principles is cheap;
measuring it is the point.

Method: run the same documents ``repeats`` times, serialise each result, and report the
fraction of documents whose output was byte-identical across every run.

This exists as a standalone script rather than only as ``evaluate --determinism`` because
on a memory-constrained machine re-running the full 90-document evaluation just to add
one column segfaults torch. This loads the model once and touches a handful of documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.cli import build_extractor  # noqa: E402
from docintel.eval.runner import load_split  # noqa: E402
from docintel.schema import OcrDocument  # noqa: E402


def signature(result) -> str:
    """A canonical string for one extraction, ignoring incidental ordering and timing."""
    payload = {
        "doc_type": result.doc_type.value,
        "fields": {
            name: (field.value if field else None)
            for name, field in sorted(result.fields.items())
        },
        "service_lines": [
            [line.procedure_code, line.date_of_service, line.units, line.charge, line.paid]
            for line in result.service_lines
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", required=True)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--split", default="gold_synth")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", default="reports/eval_gold_synth.json", type=Path)
    args = parser.parse_args()

    records = load_split(args.corpus, args.split)[: args.sample]
    docs = [
        OcrDocument.model_validate_json(
            (args.corpus / r["text"]).read_text(encoding="utf-8")
        )
        for r in records
    ]

    extractor = build_extractor(args.approach)
    runs: list[list[str]] = []
    for attempt in range(args.repeats):
        runs.append([signature(extractor.extract(doc)) for doc in docs])
        print(f"  run {attempt + 1}/{args.repeats} done", file=sys.stderr, flush=True)

    identical = sum(
        1 for i in range(len(docs)) if len({run[i] for run in runs}) == 1
    )
    rate = identical / len(docs) if docs else 0.0
    print(f"\n{args.approach}: {identical}/{len(docs)} documents byte-identical "
          f"across {args.repeats} runs -> determinism rate {rate:.2f}")

    if not runs or len({tuple(r) for r in runs}) > 1:
        # Show one divergence so a non-deterministic result is explainable rather than
        # just a number.
        for i in range(len(docs)):
            variants = {run[i] for run in runs}
            if len(variants) > 1:
                a, b = list(variants)[:2]
                print(f"\n  first divergence on {records[i]['doc_id']}:")
                print(f"    run A: {a[:220]}")
                print(f"    run B: {b[:220]}")
                break

    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        entry = report.get("approaches", {}).get(args.approach)
        if entry is not None:
            profile = entry.setdefault("profile", {})
            profile["deterministic"] = rate == 1.0
            profile["determinism_rate"] = round(rate, 4)
            profile["determinism_basis"] = (
                f"{identical}/{len(docs)} documents byte-identical across "
                f"{args.repeats} runs"
            )
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nrecorded in {args.report}")
        else:
            print(f"\n{args.approach} not present in {args.report}; nothing recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
